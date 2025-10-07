from __future__ import annotations
import numpy as np
from collections import deque, namedtuple
from typing import Deque, Tuple, List
from .utils import SumTree


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    """
    PrioritizedReplayBuffer — a replay buffer that samples transitions based on
    their temporal-difference (TD) error magnitude.

    Uses a **SumTree** for efficient O(log N) sampling and priority updates.
    Supports n-step returns similarly to the standard replay buffer.

    Based on:
        Schaul et al., *"Prioritized Experience Replay"*
        (arXiv:1511.05952, 2015) — https://arxiv.org/abs/1511.05952

    Parameters
    ----------
    capacity : int, optional
        Maximum number of transitions to store. Default is 100_000.
    n_step : int, optional
        Number of steps for n-step return computation. Default is 5.
    gamma : float, optional
        Discount factor for n-step rewards. Default is 0.99.
    alpha : float, optional
        Controls how much prioritization is used (0 → uniform, 1 → full prioritization).
        Default is 0.6.
    eps : float, optional
        Small constant added to priorities to prevent zero probability. Default is 1e-6.

    Attributes
    ----------
    tree : SumTree
        Binary tree data structure storing priorities and samples.
    n_step_buffer : Deque[tuple]
        Temporary buffer for constructing n-step transitions.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        n_step: int = 5,
        gamma: float = 0.99,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n_step)

    def _get_priority(self, td_error: float) -> float:
        """Compute priority from TD error."""
        return (abs(td_error) + self.eps) ** self.alpha

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
        done: bool,
    ) -> None:
        """Add a new n-step transition with maximum priority."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        state, action, reward, next_state, done = self._get_n_step_info()
        transition = Transition(state, action, reward, next_state, done)

        max_prio = float(np.max(self.tree.tree[-self.tree.capacity:]))
        if max_prio == 0:
            max_prio = 1.0

        self.tree.add(max_prio, transition)
        if done:
            self.n_step_buffer.clear()

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
        """
        Sample a batch of transitions according to their priorities.

        Returns
        -------
        tuple
            (states, actions, rewards, next_states, dones, idxs, weights)
        """
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs + 1e-6) ** (-beta)
        weights /= (weights.max() + 1e-6)

        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after TD error computation."""
        for idx, td_error in zip(indices, td_errors):
            self.tree.update(idx, self._get_priority(float(td_error)))

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.tree.n_entries
