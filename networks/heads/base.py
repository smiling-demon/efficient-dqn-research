import torch
import torch.nn as nn
import warnings

from ..utils.noisy_linear import NoisyLinear
from ..utils.q_dropout import QDropout


class BaseQHead(nn.Module):
    """
    Base Q-value head module for reinforcement learning agents.

    Supports:
    - **Noisy Networks for Exploration** (Fortunato et al., 2018)
        https://arxiv.org/abs/1706.10295
    - **Dropout Q-Functions** for efficient uncertainty estimation
      (Hiraoka et al., 2021) https://arxiv.org/abs/2110.02034

    Optionally outputs a multi-atom representation for distributional RL
    (e.g., C51 or QR-DQN).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    num_actions : int
        Number of discrete actions.
    atoms : int, optional
        Number of value atoms per action (1 for standard DQN, >1 for distributional). Default is 1.
    use_noisy : bool, optional
        If True, replaces the linear layer with a NoisyLinear layer (for exploration). Default is False.
    use_dropout : bool, optional
        If True, applies QDropout regularization before the output layer. Default is False.
    **kwargs :
        Additional keyword arguments (ignored, kept for compatibility).
    """

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
        atoms: int = 1,
        use_noisy: bool = False,
        use_dropout: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Warn if both NoisyNet and Dropout are enabled
        if use_noisy and use_dropout:
            warnings.warn(
                "Both NoisyNet and Dropout are enabled in BaseQHead. "
                "This combination may lead to unstable learning dynamics."
            )

        self.num_actions = num_actions
        self.atoms = atoms

        # Choose linear layer type
        linear_cls = NoisyLinear if use_noisy else nn.Linear

        # Optional dropout preprocessing
        self.preprocess = QDropout(in_dim) if use_dropout else nn.Identity()

        # Final linear projection
        self.fc = linear_cls(in_dim, num_actions * atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            Q-values of shape:
            - (B, num_actions) if atoms == 1
            - (B, num_actions, atoms) if atoms > 1
        """
        x = self.preprocess(x)
        q = self.fc(x)
        return q if self.atoms == 1 else q.view(-1, self.num_actions, self.atoms)
