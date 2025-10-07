from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, Tuple

from .vanilla import VanillaLoss
from .c51 import C51Loss
from .qr import QRLoss


class MunchausenMixin:
    """
    MunchausenMixin — provides Munchausen reward augmentation utilities.

    Core idea:
        augment immediate reward with scaled clipped log-policy:
            r' = normalized_reward + alpha * clamp(log_pi(a|s), clip_min, 0)

    Reference:
        Vieillard et al., "Munchausen Reinforcement Learning", NeurIPS 2020 / arXiv:2007.14430.
        https://arxiv.org/abs/2007.14430
    """

    def __init__(
        self,
        alpha: float = 0.3,
        tau: float = 0.1,
        clip_min: float = -1.0,
        eps: float = 1e-8,
        reward_momentum: float = 0.999
    ) -> None:
        self.alpha = alpha
        self.tau = tau
        self.clip_min = clip_min
        self.eps = eps

        self.reward_momentum = reward_momentum
        
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

    def _update_reward_stats(self, rewards: torch.Tensor) -> None:
        with torch.no_grad():
            cpu_vals = rewards.detach().cpu().flatten().numpy()
            if cpu_vals.size == 0:
                return
            batch_mean = float(cpu_vals.mean())
            batch_var = float(cpu_vals.var()) if cpu_vals.size > 1 else 0.0

            if self.reward_count == 0:
                self.reward_mean = batch_mean
                self.reward_var = max(batch_var, 1e-6)
            else:
                m = self.reward_momentum
                self.reward_mean = m * self.reward_mean + (1.0 - m) * batch_mean
                self.reward_var = m * self.reward_var + (1.0 - m) * batch_var

            self.reward_count += 1

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(self.reward_var) + self.eps
        mean = self.reward_mean
        return (rewards - mean) / std

    def compute_munchausen_reward(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Munchausen-augmented reward.

        Args
        ----
        q_values: (B, A) or (B, A, 1) or (B,) — expected Q values (if distributional, pass expectation).
        actions: (B,) long tensor of actions.
        rewards: (B,) external rewards.

        Returns
        -------
        Tensor (B,) on same device as `rewards`.
        """
        # update running stats (CPU)
        self._update_reward_stats(rewards)
        rewards_norm = self._normalize_rewards(rewards)

        # harmonize q_values -> (B, A)
        if q_values.ndim == 3 and q_values.shape[-1] == 1:
            q_det = q_values.squeeze(-1)
        else:
            q_det = q_values

        if q_det.ndim == 1:
            q_det = q_det.unsqueeze(1)

        q_det = q_det.detach()

        # compute log-policy with temperature and clamp
        log_policy = torch.log_softmax(q_det / (self.tau + self.eps), dim=1)
        log_policy = torch.clamp(log_policy, min=self.clip_min)

        idx = torch.arange(q_det.size(0), device=q_det.device)
        actions_clamped = actions.clamp(min=0, max=q_det.size(1) - 1)

        log_pi_a = log_policy[idx, actions_clamped]
        munchausen_add = self.alpha * torch.clamp(log_pi_a, min=self.clip_min, max=0.0)

        return rewards_norm + munchausen_add


class MunchausenLoss(MunchausenMixin, VanillaLoss):
    """
    MunchausenLoss — Vanilla DQN loss augmented with Munchausen rewards.
    """

    def __init__(
        self,
        # VanillaLoss params
        gamma: float = 0.99,
        n_step: int = 1,
        use_double: bool = True,
        # Munchausen params
        alpha: float = 0.3,
        tau: float = 0.1,
        clip_min: float = -1.0,
        eps: float = 1e-8,
        reward_momentum: float = 0.999,
        **kwargs: Any,
    ) -> None:
        VanillaLoss.__init__(self, gamma=gamma, n_step=n_step, use_double=use_double, **kwargs)
        MunchausenMixin.__init__(
            self,
            alpha=alpha,
            tau=tau,
            clip_min=clip_min,
            eps=eps,
            reward_momentum=reward_momentum,
        )

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        net: nn.Module,
        target: nn.Module,
        device: str,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Munchausen-augmented Vanilla loss.

        Input:
            batch: dict with keys 'states','actions','rewards','next_states','dones' and optional 'weights'
            net, target, device as usual.

        Returns:
            (loss, td_errors) where td_errors = (target - current) (B,)
        """
        states, actions, rewards, next_states, dones, weights = self._unpack_batch(batch, device)

        # get online Q-values (handle tuple outputs)
        q_out = net(states)
        if isinstance(q_out, tuple):
            q_out = q_out[0]
        q_values = self._maybe_squeeze_atom_dim(q_out)
        if q_values.ndim == 1:
            q_values = q_values.unsqueeze(1)

        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # munchausen reward
        munchausen_rewards = self.compute_munchausen_reward(q_values, actions, rewards)

        # compute next soft-value using target network
        next_net_out, next_tgt_out = self._compute_next_net_outputs(net, target, next_states)
        if isinstance(next_tgt_out, tuple):
            next_tgt_out = next_tgt_out[0]
        next_q = self._maybe_squeeze_atom_dim(next_tgt_out)

        with torch.no_grad():
            next_log_policy = torch.log_softmax(next_q / (self.tau + self.eps), dim=1)
            next_log_policy = torch.clamp(next_log_policy, min=self.clip_min)
            next_policy = torch.exp(next_log_policy)

            v_next = (next_policy * (next_q - self.tau * next_log_policy)).sum(dim=1)
            target_q = munchausen_rewards + (self.gamma ** self.n_step) * v_next * (1.0 - dones)

        td_errors = (target_q - q_action).detach()
        loss_per_sample = F.smooth_l1_loss(q_action, target_q, reduction="none")
        loss = self._aggregate_loss(loss_per_sample, weights)
        return loss, td_errors


class MunchausenC51Loss(MunchausenMixin, C51Loss):
    """
    MunchausenC51Loss — C51 loss using Munchausen-augmented rewards.
    """

    def __init__(
        self,
        # C51 params
        gamma: float = 0.99,
        n_step: int = 1,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_double: bool = True,
        # Munchausen params
        alpha: float = 0.3,
        tau: float = 0.1,
        clip_min: float = -1.0,
        eps: float = 1e-8,
        reward_momentum: float = 0.999,
        **kwargs: Any,
    ) -> None:
        C51Loss.__init__(self, gamma=gamma, n_step=n_step, atoms=atoms, v_min=v_min, v_max=v_max, use_double=use_double, **kwargs)
        MunchausenMixin.__init__(
            self,
            alpha=alpha,
            tau=tau,
            clip_min=clip_min,
            eps=eps,
            reward_momentum=reward_momentum
        )

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        net: nn.Module,
        target: nn.Module,
        device: str,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Munchausen-augmented C51 loss.

        Input:
            batch: dict with 'states','actions','rewards','next_states','dones' and optional 'weights'
        Returns:
            (loss, td_errors) where td_errors = (q_target - q_expected) (B,)
        """
        states, actions, rewards, next_states, dones, weights = self._unpack_batch(batch, device)

        logits = net(states)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=2).clamp(min=self.eps)  # (B, A, atoms)
        grid = self.grid.to(probs.device)
        q_values = (probs * grid).sum(dim=2)  # (B, A)

        munchausen_rewards = self.compute_munchausen_reward(q_values, actions, rewards)

        log_p = torch.log(probs)

        # next outputs
        next_net_out, next_tgt_out = self._compute_next_net_outputs(net, target, next_states)
        if isinstance(next_net_out, tuple):
            next_net_out = next_net_out[0]
        if isinstance(next_tgt_out, tuple):
            next_tgt_out = next_tgt_out[0]

        next_probs = torch.softmax(next_tgt_out, dim=2).clamp(min=self.eps)
        next_q = (next_probs * grid).sum(dim=2)  # (B, A)

        # choose next action (Double DQN style)
        if self.use_double:
            net_next = next_net_out
            if isinstance(net_next, tuple):
                net_next = net_next[0]
            net_next_probs = torch.softmax(net_next, dim=2).clamp(min=self.eps)
            next_q_online = (net_next_probs * grid).sum(dim=2)
            next_actions = next_q_online.argmax(dim=1)
        else:
            next_actions = next_q.argmax(dim=1)

        arange = torch.arange(states.size(0), device=probs.device)
        next_dist_a = next_probs[arange, next_actions]  # (B, atoms)

        proj_dist = self._project_distribution(next_dist_a, munchausen_rewards, dones, device)

        chosen_log_p = log_p[arange, actions]  # (B, atoms)
        loss_per_sample = -(proj_dist * chosen_log_p).sum(dim=1)
        loss = self._aggregate_loss(loss_per_sample, weights)

        with torch.no_grad():
            q_expected_a = q_values[arange, actions]
            q_target = (proj_dist * grid).sum(dim=1)
            td_errors = (q_target - q_expected_a).detach()

        return loss, td_errors


class MunchausenQRLoss(MunchausenMixin, QRLoss):
    """
    MunchausenQRLoss — Quantile Regression DQN loss augmented with Munchausen rewards.
    """

    def __init__(
        self,
        # QR params
        gamma: float = 0.99,
        n_step: int = 1,
        quantiles: int = 200,
        kappa: float = 1.0,
        use_double: bool = True,
        target_clip: bool = True,
        target_clip_value: float = 10.0,
        # Munchausen params
        alpha: float = 0.3,
        tau: float = 0.1,
        clip_min: float = -1.0,
        eps: float = 1e-8,
        reward_momentum: float = 0.999,
        **kwargs: Any,
    ) -> None:
        QRLoss.__init__(self, gamma=gamma, n_step=n_step, quantiles=quantiles, kappa=kappa, use_double=use_double, target_clip=target_clip, target_clip_value=target_clip_value, **kwargs)
        MunchausenMixin.__init__(
            self,
            alpha=alpha,
            tau=tau,
            clip_min=clip_min,
            eps=eps,
            reward_momentum=reward_momentum,
        )
        self.target_clip = target_clip
        self.target_clip_value = float(target_clip_value)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        net: nn.Module,
        target: nn.Module,
        device: str,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Munchausen-augmented QR loss.

        Input:
            batch: dict with 'states','actions','rewards','next_states','dones' and optional 'weights'
        Returns:
            (loss, td_err) where td_err = (q_target_mean - q_pred_mean) (B,)
        """
        states, actions, rewards, next_states, dones, weights = self._unpack_batch(batch, device)
        device_t = states.device
        taus = self.registered_taus.to(device_t)

        dist = net(states)
        if isinstance(dist, tuple):
            dist = dist[0]
        batch_size = states.size(0)
        arange = torch.arange(batch_size, device=device_t)

        quantiles_pred = dist[arange, actions]  # (B, N_pred)
        q_values = quantiles_pred.mean(dim=1)
        munchausen_rewards = self.compute_munchausen_reward(q_values.detach(), actions, rewards)

        # build targets
        next_net_out, next_tgt_out = self._compute_next_net_outputs(net, target, next_states)
        if isinstance(next_net_out, tuple):
            next_net_out = next_net_out[0]
        if isinstance(next_tgt_out, tuple):
            next_tgt_out = next_tgt_out[0]

        next_q = next_tgt_out.mean(dim=2)  # (B, A)
        if self.use_double:
            net_next = next_net_out
            if isinstance(net_next, tuple):
                net_next = net_next[0]
            next_actions = net_next.mean(dim=2).argmax(dim=1)
        else:
            next_actions = next_q.argmax(dim=1)

        next_quantiles = next_tgt_out[arange, next_actions]  # (B, N_tgt)
        target_quantiles = munchausen_rewards.unsqueeze(1) + (self.gamma ** self.n_step) * (1.0 - dones.unsqueeze(1)) * next_quantiles

        if self.target_clip:
            target_quantiles = target_quantiles.clamp(-self.target_clip_value, self.target_clip_value)

        td_errors = target_quantiles.unsqueeze(1) - quantiles_pred.unsqueeze(2)

        loss_per_sample = self._quantile_huber_loss(td_errors, taus, self.kappa)  # (B,)
        loss = self._aggregate_loss(loss_per_sample, weights, normalize_by_sum=True)

        with torch.no_grad():
            q_expected = quantiles_pred.mean(dim=1)
            q_target = target_quantiles.mean(dim=1)
            td_err = (q_target - q_expected).detach()

        return loss, td_err
