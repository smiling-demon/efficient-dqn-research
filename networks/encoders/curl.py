import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CURLEncoder(nn.Module):
    """
    CURL — Contrastive Unsupervised Representation Learning module.

    This module implements the **CURL** objective proposed in:
        Srinivas et al., "CURL: Contrastive Unsupervised Representations
        for Reinforcement Learning" (ICML 2020)
        https://arxiv.org/abs/2004.04136

    The module maintains two encoders:
        - A **query encoder** (`encoder_q`), trained with gradients.
        - A **momentum (key) encoder** (`encoder_k`), updated via exponential moving average (EMA).

    During training, two random augmentations of each observation are encoded
    into latent vectors (`z_q` and `z_k`), and a contrastive InfoNCE loss is computed
    to align positive pairs (same sample, different views) while contrasting negatives.

    Parameters
    ----------
    encoder : nn.Module
        The base CNN encoder network (e.g., `CNNEncoder`) that maps images to latent features.
    preprocessor : callable
        A data augmentation function applied independently to each view.
        Must accept and return a batch tensor (B, C, H, W).
    latent_dim : int, optional
        Dimensionality of the latent feature space. Default is 512.
    momentum : float, optional
        Momentum for the exponential moving average (EMA) update of the key encoder. Default is 0.99.
    temperature : float, optional
        Temperature parameter τ for the InfoNCE loss scaling. Default is 0.05.
    **kwargs :
        Extra arguments (kept for compatibility).

    Attributes
    ----------
    encoder_q : nn.Module
        The main (query) encoder, trained with gradients.
    encoder_k : nn.Module
        The momentum (key) encoder, updated via EMA.
    W : nn.Parameter
        Trainable bilinear weight matrix used in the contrastive similarity function.
    """

    def __init__(
        self,
        encoder: nn.Module,
        preprocessor,
        latent_dim: int = 512,
        momentum: float = 0.99,
        temperature: float = 0.05,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder_q = encoder                                # trainable encoder
        self.encoder_k = self._make_momentum_encoder(encoder)    # EMA (frozen) encoder
        self.preprocessor = preprocessor

        self.momentum = float(momentum)
        self.temperature = float(temperature)

        self.latent_dim = latent_dim

        # Trainable bilinear projection matrix W
        self.W = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim) * 0.1)

    def _make_momentum_encoder(self, encoder: nn.Module) -> nn.Module:
        """
        Create a frozen (momentum) copy of the encoder.

        All parameters are cloned and `requires_grad` is set to False.
        """
        encoder_k = copy.deepcopy(encoder)
        for p in encoder_k.parameters():
            p.requires_grad = False
        return encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Update the momentum encoder using exponential moving average (EMA).

        For each parameter:
            p_k = momentum * p_k + (1 - momentum) * p_q
        """
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(self.momentum).add_(p_q.data, alpha=1.0 - self.momentum)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the CURL module.

        Parameters
        ----------
        x : Tensor
            Batch of input observations with shape (B, C, H, W).

        Returns
        -------
        z_q : Tensor
            Normalized latent embeddings from the query encoder, shape (B, latent_dim).
        curl_loss : Tensor
            Scalar InfoNCE loss computed across the batch.
        """
        # Generate two independent augmented views
        x_q = self.preprocessor(x)
        x_k = self.preprocessor(x)

        # Compute query embeddings
        z_q = self.encoder_q(x_q)  # (B, D)

        # Momentum update and compute key embeddings (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k = self.encoder_k(x_k).detach()  # (B, D)

        # L2 normalize both embeddings
        z_q = F.normalize(z_q, dim=1)
        z_k = F.normalize(z_k, dim=1)

        # Compute logits: z_qᵀ W z_k / τ
        Wz_k = torch.matmul(self.W, z_k.T)               # (D, B)
        logits = torch.matmul(z_q, Wz_k) / self.temperature  # (B, B)

        # Numerical stability (subtract row-wise max)
        logits = logits - logits.max(dim=1, keepdim=True).values

        # InfoNCE target: positive pairs are diagonal elements (i == j)
        labels = torch.arange(logits.size(0), device=logits.device)
        curl_loss = F.cross_entropy(logits, labels)

        return z_q, curl_loss
