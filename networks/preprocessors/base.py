from torch import nn
import torch


class BasePreprocessor(nn.Module):
    """
    Base class for observation preprocessing modules.

    Acts as an identity transformation by default. Designed to be subclassed
    for data augmentation or normalization in reinforcement learning pipelines.

    Parameters
    ----------
    **kwargs :
        Additional keyword arguments for compatibility with subclasses.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (identity mapping).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor
            The same tensor `x`, unchanged.
        """
        return x
