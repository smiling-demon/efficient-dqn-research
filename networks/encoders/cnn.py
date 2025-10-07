import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class CNNEncoder(nn.Module):
    """
    CNNEncoder — a convolutional neural network encoder that transforms image-like
    observations (e.g., Atari frames) into compact feature vectors.

    The convolutional architecture follows the one used in the **Rainbow DQN** paper:

        Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning"
        (AAAI 2018) — https://arxiv.org/abs/1710.02298

    Parameters
    ----------
    in_shape : tuple[int, int, int], optional
        Input tensor shape in the format (C, H, W). Default is (4, 84, 84),
        where C=4 corresponds to stacked grayscale frames.
    out_dim : int, optional
        Dimension of the output feature vector. Default is 512.
    **kwargs :
        Additional keyword arguments (not used, kept for compatibility).

    Attributes
    ----------
    conv : nn.Sequential
        Convolutional feature extractor using the standard Atari CNN architecture.
    fc : nn.Sequential
        Fully connected projection layer mapping features to a fixed-size embedding.
    """

    def __init__(self, in_shape: Tuple[int, int, int] = (4, 84, 84), out_dim: int = 512, **kwargs) -> None:
        super().__init__()
        c, h, w = in_shape

        # Convolutional feature extractor (Rainbow / DQN-style)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Automatically compute flattened output size
        with torch.no_grad():
            conv_out_dim = self.conv(torch.zeros(1, *in_shape)).shape[1]

        # Fully connected projection layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the CNN encoder.

        Parameters
        ----------
        x : Tensor
            Input batch of images with shape (B, C, H, W) and pixel values in [0, 255].

        Returns
        -------
        Tensor
            Encoded feature tensor of shape (B, out_dim).
        """
        # Normalize pixel values to [0, 1]
        x = x / 255.0
        features = self.conv(x)
        embeddings = self.fc(features)
        return embeddings
