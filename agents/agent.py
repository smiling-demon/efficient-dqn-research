from __future__ import annotations

import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Tuple, Union

from networks import DQN, RNDModel, NoisyLinear
from replay import ReplayBuffer, PrioritizedReplayBuffer
from losses import LossFactory


class Agent:
    """Deep Q-Learning agent supporting modern DQN extensions such as
    C51, QR-DQN, RND, CURL, NoisyNet, and PER.
    """

    def __init__(self, env: Any, eval_env: Any, params: Dict[str, Dict[str, Any]]) -> None:
        """Initialize the agent and all its components.

        Args:
            env: Training environment.
            eval_env: Evaluation environment.
            params: Configuration dictionary with keys "agent_params", "dqn_params", and "loss_params".
        """
        self.env = env
        self.eval_env = eval_env

        agent_params = params["agent_params"]
        dqn_params = params["dqn_params"]
        loss_params = params["loss_params"]

        self.device: torch.device = agent_params["device"]

        # Networks
        self.net: DQN = DQN(dqn_params).to(self.device)
        self.target: DQN = DQN(dqn_params).to(self.device)
        self.target.load_state_dict(self.net.state_dict())

        # DQN extensions
        self.atoms: int = agent_params["atoms"]
        self.use_noisy: bool = agent_params["use_noisy"]
        self.use_c51: bool = agent_params["use_c51"]
        self.use_qr: bool = agent_params["use_qr"]
        self.use_curl: bool = agent_params["use_curl"]

        # Optimizer and scheduler
        self.optimizer: optim.Optimizer = optim.Adam(self.net.parameters(), lr=agent_params["lr"])
        warmup_steps: int = agent_params["warmup_steps"]
        decay_steps: int = agent_params["decay_steps"]

        def lr_lambda(step: int) -> float:
            """Linear warm-up followed by decay."""
            if step < warmup_steps:
                return step / warmup_steps
            elif step < decay_steps:
                progress = (step - warmup_steps) / (decay_steps - warmup_steps)
                return 1.0 - 0.9 * progress
            return 0.1

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # Replay buffer
        self.per: bool = agent_params["use_per"]
        replay_cls = PrioritizedReplayBuffer if self.per else ReplayBuffer
        self.replay: Union[ReplayBuffer, PrioritizedReplayBuffer] = replay_cls(**params["replay_params"])

        # Random Network Distillation (RND)
        self.use_rnd: bool = agent_params["use_rnd"]
        if self.use_rnd:
            obs_dim = int(np.prod(env.observation_space.shape))
            self.rnd: RNDModel = RNDModel(
                input_dim=obs_dim,
                lr=agent_params["lr"],
                device=self.device
            )
            self.rnd_coef: float = agent_params["rnd_coef"]
        else:
            self.rnd = None
            self.rnd_coef = 0.0

        # Hyperparameters
        self.batch_size: int = agent_params["batch_size"]
        self.epsilon: float = agent_params["epsilon"]
        self.beta_start: float = agent_params["beta_start"]
        self.beta_end: float = agent_params["beta_end"]
        self.train_freq: int = agent_params["train_freq"]
        self.learning_start: int = agent_params["learning_start"]
        self.eval_freq: int = agent_params["eval_freq"]
        self.eval_count: int = agent_params["eval_count"]
        self.target_update_freq: int = agent_params["target_update_freq"]
        self.env_lives: int = agent_params["env_lives"]
        self.eval_help: bool = agent_params["eval_help"]
        self.save_model: bool = agent_params["save_model"]
        self.load_model: bool = agent_params["load_model"]
        self.load_path: str = agent_params["load_path"]

        # Logging
        self.model_name: str = f"{agent_params['log_name']}:{time.strftime('%d%H%M%S')}"
        self.logging: bool = agent_params["logging"]
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir=os.path.join("runs", self.model_name)) if self.logging else None
        )

        # Loss function
        self.loss_obj = LossFactory.create(loss_params["loss_name"], loss_params)

        # C51 value grid
        self.grid: torch.Tensor = torch.linspace(
            loss_params["v_min"],
            loss_params["v_max"],
            self.atoms,
            device=self.device
        )

        # Misc parameters
        self.max_norm: float = agent_params["max_grad_norm"]
        self.eps = agent_params["eps"]

    def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select an action using epsilon-greedy or noisy policy."""
        if not self.use_noisy and np.random.rand() < epsilon:
            return self.env.action_space.sample()

        state_t = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.net(state_t)
            if self.use_qr:
                q = out.mean(dim=2)
            elif self.use_c51:
                probs = torch.softmax(out, dim=2)
                q = (probs * self.grid).sum(dim=2)
            else:
                q = out.squeeze(-1) if out.shape[-1] == 1 else out
            return int(q.argmax(dim=1).item())

    def push(self, *args: Any) -> None:
        """Store a transition in the replay buffer."""
        self.replay.push(*args)

    def _normalize_sample(self, sample: Tuple) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Convert a raw replay sample into tensors."""
        if len(sample) == 7:
            states, actions, rewards, next_states, dones, indices, weights = sample
        elif len(sample) == 5:
            states, actions, rewards, next_states, dones = sample
            indices, weights = None, np.ones(len(states), np.float32)
        else:
            raise RuntimeError(f"Unexpected sample length: {len(sample)}")

        return {
            "states": torch.tensor(states, dtype=torch.float32, device=self.device),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "next_states": torch.tensor(next_states, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=self.device),
            "indices": np.asarray(indices, np.int64) if indices is not None else None,
        }

    def update(self, beta: float = 0.4) -> Optional[Tuple[float, float, float, float]]:
        """Perform one gradient update step."""
        if len(self.replay) < self.batch_size:
            return None

        raw = (
            self.replay.sample(self.batch_size, beta=beta)
            if isinstance(self.replay, PrioritizedReplayBuffer)
            else self.replay.sample(self.batch_size)
        )
        batch = self._normalize_sample(raw)
        indices = batch["indices"]

        # Random Network Distillation
        rnd_loss = 0.0
        if self.use_rnd:
            next_states_flat = batch["next_states"].reshape(self.batch_size, -1)
            rnd_loss = self.rnd.update(next_states_flat)
            with torch.no_grad():
                r_int = self.rnd.compute_intrinsic_reward(next_states_flat)
                r_int_norm = self.rnd.normalize_intrinsic(r_int)
            batch["rewards"] += self.rnd_coef * r_int_norm

        # Compute loss
        loss, td_errors = self.loss_obj.compute_loss(batch, self.net, self.target, self.device)

        # CURL loss
        curl_loss = 0.0
        if self.use_curl:
            curl_loss_tensor = self.net(batch["states"], return_curl_loss=True)[1]
            loss += curl_loss_tensor
            curl_loss = curl_loss_tensor.item()

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
        self.optimizer.step()
        self.scheduler.step()

        # PER priority update
        if isinstance(self.replay, PrioritizedReplayBuffer):
            td_np = td_errors.abs().detach().cpu().numpy()
            self.replay.update_priorities(np.asarray(indices, np.int64), td_np + self.eps)

        return loss.item(), grad_norm.item(), rnd_loss, curl_loss

    def train(self, num_frames: int = 1_000_000, max_train_time: float = 36_000) -> None:
        """Run the main training loop."""
        state, _ = self.env.reset()
        episode_reward, learning_step = 0.0, 0
        start_time: Optional[float] = None

        if self.load_model:
            self.net.load_state_dict(torch.load(self.load_path))
            self.target.load_state_dict(self.net.state_dict())

        for step in range(1, num_frames + 1):
            # Start counting learning time after warm-up
            if step > self.learning_start:
                if start_time is None:
                    start_time = time.time()
                learning_step += 1
                if time.time() - start_time >= max_train_time:
                    break

            if self.use_noisy:
                for m in self.net.modules():
                    if isinstance(m, NoisyLinear):
                        m.reset_noise()

            action = self.act(state, self.epsilon)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            if done:
                state, _ = self.env.reset()
                print(step, episode_reward)
                if step > self.learning_start and self.writer:
                    self.writer.add_scalar("Reward/episode", episode_reward, learning_step)
                episode_reward = 0.0

            if step > self.learning_start and step % self.train_freq == 0:
                beta = min(self.beta_start + (self.beta_end - self.beta_start) * (step / num_frames), self.beta_end)
                res = self.update(beta)
                if res and self.writer:
                    loss, grad_norm, rnd_loss, curl_loss = res
                    self.writer.add_scalar("Train/loss", loss, learning_step)
                    self.writer.add_scalar("Train/grad_norm", grad_norm, learning_step)
                    self.writer.add_scalar("Train/rnd_loss", rnd_loss, learning_step)
                    self.writer.add_scalar("Train/curl_loss", curl_loss, learning_step)

            if step > self.learning_start and step % self.target_update_freq == 0:
                self.target.load_state_dict(self.net.state_dict())
                if self.save_model:
                    os.makedirs("models", exist_ok=True)
                    torch.save(self.net.state_dict(), f"models/{self.model_name}.pt")

            if step > self.learning_start and step % self.eval_freq == 0:
                eval_reward = self.evaluate_episode(num_episodes=self.eval_count)
                if self.writer:
                    self.writer.add_scalar("Eval/avg_reward", eval_reward, learning_step)

        if self.writer:
            self.writer.close()

    @torch.no_grad()
    def evaluate_episode(self, num_episodes: int = 5) -> float:
        """Evaluate average reward over several episodes."""
        rewards = []
        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            total_reward, done = 0.0, False
            prev_lives = self.env_lives

            while not done:
                if self.use_noisy:
                    for m in self.net.modules():
                        if isinstance(m, NoisyLinear):
                            m.reset_noise()
                action = self.act(state, self.epsilon)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state

                # Handle life loss (useful for Atari environments)
                if self.eval_help and "lives" in info:
                    lives = info["lives"]
                    if prev_lives > lives > 0:
                        prev_lives = lives
                        next_state, reward, terminated, truncated, info = self.eval_env.step(1)
                        done = terminated or truncated
                        total_reward += reward
                        state = next_state

            rewards.append(total_reward)

        return float(np.mean(rewards))
