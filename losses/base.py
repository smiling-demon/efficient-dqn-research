from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
from torch import nn

class BaseLoss:
    """
    BaseLoss — abstract helper class providing common utilities for DQN losses.

    Provides:
      - batch unpacking
      - safe weight handling
      - helper для double-dqn выбора next actions
      - утилиты для squeeze-совместимости atoms==1

    Subclasses should implement `_compute_target(...)` which returns:
      - target (scalar per-sample)  OR
      - target distribution / target quantiles (depending on loss)
    """

    def __init__(self, gamma: float = 0.99, n_step: int = 1, use_double: bool = True, **kwargs: Any) -> None:
        self.gamma = gamma
        self.n_step = n_step
        self.use_double = use_double

    @staticmethod
    def _unpack_batch(batch: Dict[str, torch.Tensor], device: str):
        device = torch.device(device)
        states = batch["states"].to(device)
        actions = batch["actions"].long().to(device)
        rewards = batch["rewards"].to(device)
        next_states = batch["next_states"].to(device)
        dones = batch["dones"].to(device)
        weights = batch.get("weights", torch.ones(states.size(0), device=device)).to(device)
        return states, actions, rewards, next_states, dones, weights

    @staticmethod
    def _maybe_squeeze_atom_dim(x: torch.Tensor) -> torch.Tensor:
        # if network outputs (B, A, 1) for atoms==1, squeeze last dim for scalar Q
        if x is not None and x.ndim == 3 and x.size(-1) == 1:
            return x.squeeze(-1)
        return x

    @staticmethod
    def _compute_next_net_outputs(net: nn.Module, target: nn.Module, next_states: torch.Tensor):
        # return next outputs from both networks (without grad)
        with torch.no_grad():
            next_net = net(next_states)
            next_tgt = target(next_states)
        return next_net, next_tgt

    @staticmethod
    def _aggregate_loss(loss_per_sample: torch.Tensor, weights: torch.Tensor,
                        normalize_by_sum: bool = False) -> torch.Tensor:
        if normalize_by_sum:
            denom = weights.sum().clamp(min=1e-6)
            return (loss_per_sample * weights).sum() / denom
        return (loss_per_sample * weights).mean()

    def _select_next_action_and_value(self, next_net: torch.Tensor, next_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generic selection helper:
            - next_net, next_tgt may be either (B, A) scalar Q-values
              or (B, A, atoms) for distributional (caller should adapt)
        Returns:
            - next_actions: LongTensor(B,)  (selected actions by argmax on next_net)
            - next_values: Tensor(B,)       (value of next_states under selected action from next_tgt)
        """
        # ensure scalar-case
        next_net_s = self._maybe_squeeze_atom_dim(next_net)
        next_tgt_s = self._maybe_squeeze_atom_dim(next_tgt)

        next_actions = next_net_s.argmax(dim=1)
        next_values = next_tgt_s.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        return next_actions, next_values

    # Subclasses must implement:
    def compute_loss(self, batch: Dict[str, torch.Tensor], net: nn.Module, target: nn.Module, device: str, **kwargs):
        raise NotImplementedError
