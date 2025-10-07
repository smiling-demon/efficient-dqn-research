import torch
from .base import BaseQHead


class DuelingQHead(BaseQHead):
    """
    Dueling Q-value head for reinforcement learning agents.

    Implements the **Dueling Network Architecture**
    (Wang et al., 2016, https://arxiv.org/abs/1511.06581),
    which decomposes the Q-function into a state-value stream `V(s)`
    and an advantage stream `A(s, a)`:

        Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    num_actions : int
        Number of discrete actions.
    atoms : int, optional
        Number of atoms per action (1 for standard DQN, >1 for distributional). Default is 1.
    use_noisy : bool, optional
        If True, uses NoisyLinear layers. Default is False.
    use_dropout : bool, optional
        If True, applies QDropout regularization. Default is False.
    **kwargs :
        Additional arguments passed to BaseQHead.
    """

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
        atoms: int = 1,
        **kwargs,
    ):
        super().__init__(in_dim, num_actions, atoms, **kwargs)

        # Determine layer type from BaseQHead (Linear or NoisyLinear)
        layer_cls = self.fc.__class__

        # Separate value (V) and advantage (A) streams
        self.v_stream = layer_cls(in_dim, atoms)
        self.a_stream = layer_cls(in_dim, num_actions * atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            - Shape (B, num_actions) if atoms == 1
            - Shape (B, num_actions, atoms) if atoms > 1
        """
        x = self.preprocess(x)
        v = self.v_stream(x).view(-1, 1, self.atoms)
        a = self.a_stream(x).view(-1, self.num_actions, self.atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q if self.atoms > 1 else q.squeeze(-1)
