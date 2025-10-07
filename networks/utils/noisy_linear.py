import math
import torch
from torch import nn


class NoisyLinear(nn.Module):
    """
    Linear layer with learnable, factorized Gaussian noise as proposed in
    *Fortunato et al., "Noisy Networks for Exploration"*
    (ICLR 2018, https://arxiv.org/abs/1706.10295).

    This layer replaces deterministic weights and biases with noisy counterparts,
    enabling stochastic exploration directly in the function approximator.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    sigma_init : float, optional
        Initial value for noise scale (σ). Default is 0.5.
    """

    def __init__(self, in_dim: int, out_dim: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_init = sigma_init

        # Learnable parameters for mean (μ) and standard deviation (σ)
        self.weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias_mu = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))

        # Noise buffers (not trainable, updated with reset_noise())
        self.register_buffer("weight_epsilon", torch.empty(out_dim, in_dim))
        self.register_buffer("bias_epsilon", torch.empty(out_dim))

        self.reset_parameters()
        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Helper transformation: f(x) = sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_parameters(self) -> None:
        """Initialize mean and sigma parameters."""
        mu_range = 1 / math.sqrt(self.in_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_dim))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_dim))

    def reset_noise(self) -> None:
        """Sample new noise for weights and biases."""
        epsilon_in = self._f(torch.randn(self.in_dim, device=self.weight_mu.device))
        epsilon_out = self._f(torch.randn(self.out_dim, device=self.weight_mu.device))
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with noisy weights.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_dim).
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu

        return torch.nn.functional.linear(x, weight, bias)
