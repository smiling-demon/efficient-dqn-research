import torch
import torch.nn as nn

from .encoders import CURLEncoder
from .utils import get_class


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture integrating encoder, preprocessor, and Q-head.

    Optionally supports CURL (Contrastive Unsupervised Representation Learning)
    for improved representation learning.

    Parameters
    ----------
    params : dict
        Configuration dictionary defining model components and hyperparameters. Expected keys:
            - "preprocessor": str — name of the preprocessor ("base", "drq", ...).
            - "encoder": str — name of the encoder ("cnn", "pretrained", ...).
            - "head": str — name of the Q-head ("base", "dueling", ...).
            - "preprocessor_params": dict — arguments for preprocessor constructor.
            - "encoder_params": dict — arguments for encoder constructor.
            - "head_params": dict — arguments for Q-head constructor.
            - "use_curl": bool — whether to enable CURL representation learning.
            - "curl_params": dict — optional params for CURL module (momentum, temperature, etc.).
    """

    def __init__(self, params: dict):
        super().__init__()

        # Dynamically resolve component classes via registry
        preproc_cls = get_class("preprocessor", params["preprocessor"], "base")
        encoder_cls = get_class("encoder", params["encoder"], "cnn")
        head_cls = get_class("head", params["head"], "base")

        # Instantiate network components
        self.preprocessor = preproc_cls(**params["preprocessor_params"])
        self.encoder = encoder_cls(**params["encoder_params"])
        self.head = head_cls(**params["head_params"])

        # Optional CURL encoder (contrastive loss)
        self.use_curl = params.get("use_curl", False)
        if self.use_curl:
            if params["preprocessor"].lower() == "base":
                raise ValueError("CURL requires an augmentation preprocessor (e.g. DrQ).")

            self.curl = CURLEncoder(
                self.encoder,
                self.preprocessor,
                **params.get("curl_params", {}),
            )
        else:
            self.curl = None

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_curl_loss: bool = False):
        """
        Forward pass through the DQN network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch of frames or states).
        return_curl_loss : bool, optional
            If True and CURL is enabled, returns both Q-values and CURL loss.

        Returns
        -------
        torch.Tensor or tuple(torch.Tensor, torch.Tensor)
            - Q-values tensor of shape (B, num_actions)
            - Optionally, CURL loss scalar if return_curl_loss=True
        """
        if self.use_curl and return_curl_loss:
            x, curl_loss = self.curl(x)
            q_values = self.head(x)
            return q_values, curl_loss

        x = self.preprocessor(x)
        x = self.encoder(x)
        q_values = self.head(x)
        return q_values
