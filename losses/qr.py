from __future__ import annotations
from typing import Any, Tuple
import torch
from torch import nn

from .base import BaseLoss


class QRLoss(BaseLoss):
    """
    QRLoss — Quantile Regression DQN loss (QR-DQN / IQN style quantile loss).

    Implements the quantile Huber loss between predicted quantiles and target
    quantiles as in:
        Dabney et al., "Distributional Reinforcement Learning with Quantile Regression"
        (ICLR 2018) — https://arxiv.org/abs/1710.10044

    Parameters
    ----------
    gamma : float, optional
        Discount factor for future rewards. Default 0.99.
    n_step : int, optional
        N-step horizon. Default 1.
    quantiles : int, optional
        Number of quantile fractions (atoms). Default 200.
    kappa : float, optional
        Huber threshold for quantile loss. Default 1.0.
    use_double : bool, optional
        Whether to use Double DQN. Default True.
    target_clip : bool, optional
        Whether to clip target quantiles for stability. Default True.
    target_clip_value : float, optional
        Clip value for targets (symmetric). Default 10.0.
    **kwargs :
        Additional unused parameters.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        n_step: int = 1,
        atoms: int = 200,
        kappa: float = 1.0,
        use_double: bool = True,
        target_clip: bool = True,
        target_clip_value: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(gamma=gamma, n_step=n_step, use_double=use_double, **kwargs)
        self.quantiles = atoms
        self.kappa = kappa
        self.target_clip = target_clip
        self.target_clip_value = target_clip_value

        taus = torch.linspace(1.0 / (2 * self.quantiles), 1.0 - 1.0 / (2 * self.quantiles), self.quantiles)
        self.registered_taus = taus.unsqueeze(0)  # shape (1, N)

    def _quantile_huber_loss(self, td_errors: torch.Tensor, taus: torch.Tensor, kappa: float) -> torch.Tensor:
        """
        Compute quantile Huber loss.

        td_errors: (B, N_pred, N_tgt)
        taus: (1, N_pred)
        returns: per-sample loss tensor (B,)
        """
        huber = torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa),
        )  # (B, N_pred, N_tgt)

        weight = torch.abs(taus.unsqueeze(2) - (td_errors.detach() < 0).float())  # (1, N_pred, 1) -> broadcast
        quantile_loss = weight * huber  # (B, N_pred, N_tgt)
        return quantile_loss.mean(dim=(1, 2))  # (B,)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        net: nn.Module,
        target: nn.Module,
        device: str,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantile regression loss.

        Returns (loss, td_errors) where td_errors is per-sample scalar for PER.
        """
        states, actions, rewards, next_states, dones, weights = self._unpack_batch(batch, device)
        taus = self.registered_taus.to(torch.device(device))  # (1, N)

        # predicted quantiles: dist (B, A, N)
        dist = net(states)
        batch_size = states.size(0)
        arange = torch.arange(batch_size, device=dist.device)

        quant_pred = dist[arange, actions]  # (B, N_pred)

        # compute next distributions (no grad)
        next_net, next_tgt = self._compute_next_net_outputs(net, target, next_states)

        # select next actions (by expectation) using online or target depending on use_double
        next_q_online = next_net.mean(dim=2)  # (B, A)
        next_q_target = next_tgt.mean(dim=2)  # (B, A)
        if self.use_double:
            next_actions = next_q_online.argmax(dim=1)
        else:
            next_actions = next_q_target.argmax(dim=1)

        next_quantiles = next_tgt[arange, next_actions]  # (B, N_tgt)

        # build target quantiles: (B, N_tgt)
        target_quant = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * (1.0 - dones.unsqueeze(1)) * next_quantiles

        if self.target_clip:
            tv = float(self.target_clip_value)
            target_quant = target_quant.clamp(-tv, tv)

        # td errors: (B, N_pred, N_tgt)
        td_errors = target_quant.unsqueeze(1) - quant_pred.unsqueeze(2)

        # per-sample loss via quantile huber
        loss_per_sample = self._quantile_huber_loss(td_errors, taus, self.kappa)  # (B,)

        # weighted aggregate (normalize by sum of weights to be stable)
        loss = self._aggregate_loss(loss_per_sample, weights, normalize_by_sum=True)

        with torch.no_grad():
            td_err = target_quant.mean(dim=1) - quant_pred.mean(dim=1)  # (B,)

        return loss, td_err
