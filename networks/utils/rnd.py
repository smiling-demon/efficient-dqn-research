import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class RNDModel(nn.Module):
    """
    Random Network Distillation (RND) for intrinsic motivation.

    Implements the mechanism introduced in
    *Burda et al., “Exploration by Random Network Distillation”* (2018, https://arxiv.org/abs/1810.12894)
    In RND, a fixed random target network defines features, while a predictor network
    is trained to match those features. The prediction error is used as an intrinsic reward,
    encouraging exploration of novel states.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (e.g. flattened observation embedding).
    feature_dim : int, optional
        Dimensionality of the feature space output by target / predictor. Default is 128.
    hidden_dim : int, optional
        Width of the hidden layer in both target and predictor networks. Default is 256.
    lr : float, optional
        Learning rate for the predictor’s optimizer. Default is 1e-4.
    device : str, optional
        Device to use ("cuda", "cpu"). If None, chooses automatically. Default is None.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        device: str | None = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Target network (fixed, random)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(self.device)

        # Optimizer for predictor
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running statistics for normalizing intrinsic rewards
        self.int_running_mean: float = 0.0
        self.int_running_var: float = 1.0
        self.int_count: float = 1e-4
        self.eps: float = 1e-8

    def _update_running_stats(self, vals: Tensor) -> None:
        """
        Online update of mean and variance of intrinsic error values.

        Uses Welford’s algorithm style update to maintain a running mean/variance.
        """
        v = vals.detach().cpu().numpy()
        batch_mean = float(np.mean(v))
        batch_var = float(np.var(v))
        batch_count = v.size

        delta = batch_mean - self.int_running_mean
        tot_count = self.int_count + batch_count

        # Update mean
        new_mean = self.int_running_mean + delta * (batch_count / tot_count)

        # Combine variances
        m_a = self.int_running_var * self.int_count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.int_count * batch_count / tot_count) / tot_count

        self.int_running_mean = new_mean
        self.int_running_var = new_var
        self.int_count = tot_count

    def normalize_intrinsic(self, int_tensor: Tensor) -> Tensor:
        """
        Normalize intrinsic rewards using running mean and std.

        Parameters
        ----------
        int_tensor : Tensor
            Raw intrinsic reward tensor (shape (B,)).

        Returns
        -------
        Tensor
            Normalized intrinsic rewards (shape (B,)).
        """
        std = math.sqrt(self.int_running_var) + self.eps
        mean = self.int_running_mean
        return (int_tensor - mean) / std

    @torch.no_grad()
    def compute_intrinsic_reward(self, next_states: Tensor) -> Tensor:
        """
        Compute intrinsic reward for a batch of next states (without updating predictor).

        Intrinsic reward is the MSE between target and predictor features:
            r_int(s) = ‖f_target(s) – f_predictor(s)‖²

        Parameters
        ----------
        next_states : Tensor
            Batch of state representations (shape (B, input_dim)).

        Returns
        -------
        Tensor
            Intrinsic rewards (shape (B,)).
        """
        t_feats = self.target(next_states)
        p_feats = self.predictor(next_states)
        # MSE per sample
        intrinsic = (t_feats - p_feats).pow(2).mean(dim=1)
        return intrinsic

    def update(self, next_states: Tensor) -> float:
        """
        Train predictor to minimize the MSE to the fixed target, and update stats.

        Parameters
        ----------
        next_states : Tensor
            Batch of states (shape (B, input_dim)).

        Returns
        -------
        float
            Scalar MSE loss value (predictor vs target).
        """
        t_feats = self.target(next_states)
        p_feats = self.predictor(next_states)
        loss = F.mse_loss(p_feats, t_feats.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            errors = (t_feats - p_feats).pow(2).mean(dim=1)
        self._update_running_stats(errors)

        return loss.item()
