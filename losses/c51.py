from __future__ import annotations
from typing import Any, Tuple
import torch
from torch import nn

from .base import BaseLoss


class C51Loss(BaseLoss):
    """
    C51Loss — categorical distributional DQN loss (C51).

    Implements the projection of the target categorical distribution onto the fixed
    support as described in:
        Bellemare et al., "A Distributional Perspective on Reinforcement Learning"
        (ICML 2017) — https://arxiv.org/abs/1707.06887

    Parameters
    ----------
    gamma : float, optional
        Discount factor for future rewards. Default is 0.99.
    n_step : int, optional
        N-step return horizon. Default is 1.
    atoms : int, optional
        Number of discrete support atoms. Default is 51.
    v_min : float, optional
        Minimum value of the support. Default is -10.0.
    v_max : float, optional
        Maximum value of the support. Default is 10.0.
    use_double : bool, optional
        Whether to use Double DQN for target selection. Default is True.
    eps : float, optional
        Small epsilon for numerical stability when taking logs. Default 1e-8.
    **kwargs :
        Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        n_step: int = 1,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_double: bool = True,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(gamma=gamma, n_step=n_step, use_double=use_double, **kwargs)
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.eps = eps
        self.register_buffer = getattr(self, "register_buffer", None)
        # support grid kept as tensor (generated on-the-fly in compute_loss to device)
        self.grid = torch.linspace(self.v_min, self.v_max, self.atoms)

    def _project_distribution(
        self,
        next_dist_a: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """
        Project the next-state categorical distribution onto the fixed support.

        next_dist_a: (B, atoms) probability vectors (already clipped)
        rewards: (B,)
        dones: (B,)
        returns: (B, atoms) projected probability vectors
        """
        device_t = torch.device(device)
        batch_size = rewards.size(0)
        delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        grid = self.grid.to(device_t)

        # Tz: (B, atoms)
        Tz = rewards.unsqueeze(1) + (1.0 - dones).unsqueeze(1) * (self.gamma ** self.n_step) * grid.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / delta_z  # (B, atoms)
        l = b.floor().long().clamp(0, self.atoms - 1)
        u = b.ceil().long().clamp(0, self.atoms - 1)
        offset = (b - l.float())  # fractional part

        next_p = next_dist_a  # (B, atoms)
        same_mask = (u == l)

        # m_l and m_u have shape (B, atoms)
        m_l = torch.where(same_mask, next_p, next_p * (1.0 - offset))
        m_u = torch.where(same_mask, torch.zeros_like(next_p), next_p * offset)

        proj_dist = torch.zeros((batch_size, self.atoms), device=device_t)
        proj_dist.scatter_add_(1, l, m_l)
        proj_dist.scatter_add_(1, u, m_u)

        # normalize to avoid numerical drift
        proj_dist = proj_dist / proj_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return proj_dist

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        net: nn.Module,
        target: nn.Module,
        device: str,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute C51 cross-entropy loss between projected target distribution and
        predicted log-probabilities.

        Returns (loss, td_errors) where td_errors is a per-sample scalar (for PER).
        """
        states, actions, rewards, next_states, dones, weights = self._unpack_batch(batch, device)

        # network outputs: dist (B, A, atoms)
        dist = net(states)  # (B, A, atoms)
        # numerical stable probabilities & log-probs
        probs = torch.clamp(torch.softmax(dist, dim=2), min=self.eps, max=1.0)
        log_p = torch.log(probs)

        # compute next distributions (no grad)
        next_net, next_tgt = self._compute_next_net_outputs(net, target, next_states)
        # probabilities for target network
        next_probs = torch.clamp(torch.softmax(next_tgt, dim=2), min=self.eps, max=1.0)

        device_t = torch.device(device)
        batch_size = states.size(0)
        arange = torch.arange(batch_size, device=device_t)

        # choose next actions (Double DQN style) using expectation over the support
        if self.use_double:
            # expectation under online net to choose actions
            next_net_probs = torch.clamp(torch.softmax(next_net, dim=2), min=self.eps, max=1.0)
            grid = self.grid.to(device_t)
            next_q_vals = (next_net_probs * grid).sum(dim=2)  # (B, A)
            next_actions = next_q_vals.argmax(dim=1)
        else:
            # choose action by target expectation
            grid = self.grid.to(device_t)
            next_q_vals = (next_probs * grid).sum(dim=2)  # (B, A)
            next_actions = next_q_vals.argmax(dim=1)

        # select the probability vectors for chosen next actions: (B, atoms)
        next_dist_a = next_probs[arange, next_actions]

        # project distribution to support
        proj_dist = self._project_distribution(next_dist_a, rewards, dones, device)

        # chosen action log-probs for current dist
        chosen_log_p = log_p[arange, actions]  # (B, atoms)

        # cross entropy per sample: - sum(proj * log p)
        loss_per_sample = -(proj_dist * chosen_log_p).sum(dim=1)
        loss = self._aggregate_loss(loss_per_sample, weights)

        # td_errors for PER: difference between expected Qs (pred - target)
        with torch.no_grad():
            q_expected = (probs * grid).sum(dim=2)  # (B, A)
            q_expected_a = q_expected[arange, actions]  # (B,)
            q_target = (proj_dist * grid).sum(dim=1)    # (B,)
            td_errors = (q_expected_a - q_target).detach()

        return loss, td_errors
