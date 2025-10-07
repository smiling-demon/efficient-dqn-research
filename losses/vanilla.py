import torch
from torch.nn import functional as F
from .base import BaseLoss
from torch import nn
from typing import Dict, Any

class VanillaLoss(BaseLoss):
    def __init__(self, gamma: float = 0.99, n_step: int = 1, use_double: bool = True, **kwargs: Any):
        super().__init__(gamma=gamma, n_step=n_step, use_double=use_double, **kwargs)

    def compute_loss(self, batch: Dict[str, torch.Tensor], net: nn.Module, target: nn.Module, device: str, **kwargs):
        states, actions, rewards, next_states, dones, weights = self._unpack_batch(batch, device)

        q_values = net(states)
        q_values = self._maybe_squeeze_atom_dim(q_values)
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_net, next_tgt = self._compute_next_net_outputs(net, target, next_states)
        next_net = self._maybe_squeeze_atom_dim(next_net)
        next_tgt = self._maybe_squeeze_atom_dim(next_tgt)

        with torch.no_grad():
            if self.use_double:
                next_actions = next_net.argmax(dim=1)
                next_q = next_tgt.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = next_tgt.max(dim=1)[0]
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1.0 - dones)

        td_errors = (q_action - target_q).detach()
        loss_per_sample = F.smooth_l1_loss(q_action, target_q, reduction="none")
        loss = self._aggregate_loss(loss_per_sample, weights)
        return loss, td_errors
