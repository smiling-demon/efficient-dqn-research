from __future__ import annotations
import random
from collections import deque, namedtuple
from typing import Deque, Tuple, Any
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    ReplayBuffer — a uniform experience replay buffer for off-policy RL algorithms.

    Stores environment transitions and supports n-step returns for improved credit
    assignment. Transitions are sampled uniformly at random.

    This implementation follows the replay mechanism used in:
        Mnih et al., *"Human-level control through deep reinforcement learning"*
        (Nature, 2015) — https://www.nature.com/articles/nature14236

    Parameters
    ----------
    capacity : int, optional
        Maximum number of transitions to store. When full, the oldest samples are discarded.
        Default is 100_000.
    n_step : int, optional
        Number of steps for n-step return computation. Default is 5.
    gamma : float, optional
        Discount factor used for n-step returns. Default is 0.99.

    Attributes
    ----------
    buffer : Deque[Transition]
        Circular buffer storing transitions up to `capacity`.
    n_step_buffer : Deque[tuple]
        Temporary buffer for computing n-step transitions.
    """

    def __init__(self, capacity: int = 100_000, n_step: int = 5, gamma: float = 0.99) -> None:
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.n_step: int = n_step
        self.n_step_buffer: Deque[Tuple[Any, ...]] = deque(maxlen=n_step)
        self.gamma: float = gamma

    def _get_n_step_info(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Compute n-step cumulative reward and final next state."""
        reward, next_state, done = 0.0, None, False
        for idx, (_, _, r, n_s, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            next_state, done = n_s, d
            if done:
                break
        state, action = self.n_step_buffer[0][:2]
        return state, action, reward, next_state, done

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add one transition; stores the aggregated n-step transition once ready."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step and not done:
            return
        state, action, reward, next_state, done = self._get_n_step_info()
        self.buffer.append(Transition(state, action, reward, next_state, done))
        if done:
            self.n_step_buffer.clear()

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)
