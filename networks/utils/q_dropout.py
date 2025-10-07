from torch import nn
from torch.nn import functional as F


class QDropout(nn.Module):
    """
    Implementation of Q-dropout regularization from
    *Pitis et al., "Improving Exploration in Deep Reinforcement Learning via Q-dropout"*
    (NeurIPS 2021, https://arxiv.org/abs/2110.02034).

    Q-dropout regularizes the Q-network by applying dropout directly
    to the latent representation before the output layer, encouraging
    more stable and diverse value estimates.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    p : float, optional
        Dropout probability (default: 0.1).
    """

    def __init__(self, in_dim: int, p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        """
        Apply dropout and layer normalization during training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            Regularized tensor of shape (B, in_dim).
        """
        # Always apply dropout during forward, even in eval mode (as per paper)
        return self.layer_norm(F.dropout(x, p=self.dropout.p, training=True))
