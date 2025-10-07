import torch
import torch.nn.functional as F
from .base import BasePreprocessor


class DrQPreprocessor(BasePreprocessor):
    """
    Data augmentation module based on **DrQ**
    (Yarats et al., 2020, https://arxiv.org/abs/2004.13649).

    Applies random shift augmentation to image-like observations during training
    (replication padding + random crop).

    Parameters
    ----------
    padding : int, optional
        Number of pixels used for replication padding. Default is 4.
    **kwargs :
        Additional keyword arguments passed to BasePreprocessor.
    """

    def __init__(self, padding: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying random shift augmentation (if in training mode).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Augmented tensor of shape (B, C, H, W). If not training, returns `x` unchanged.
        """
        if not self.training:
            return x

        n, c, h, w = x.shape
        x = F.pad(x, (self.padding,) * 4, mode="replicate")

        # Random crop offsets for each image
        h_shift = torch.randint(0, 2 * self.padding + 1, (n,), device=x.device)
        w_shift = torch.randint(0, 2 * self.padding + 1, (n,), device=x.device)

        # Apply crops using per-sample shifts
        crops = [
            img[:, dh:dh + h, dw:dw + w]
            for img, dh, dw in zip(x, h_shift, w_shift)
        ]

        return torch.stack(crops, dim=0)
