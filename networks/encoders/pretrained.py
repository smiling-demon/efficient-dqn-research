import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


class PretrainedEncoder(nn.Module):
    """
    Universal visual encoder using a pretrained vision backbone (e.g., ViT, DINO, CLIP)
    from Hugging Face Transformers.

    Designed for inputs with 4 channels (e.g., stacked grayscale frames) that are
    adapted to 3-channel RGB by averaging adjacent channels before feeding into
    the pretrained model.

    Parameters
    ----------
    model_name : str, optional
        Name of the pretrained model from Hugging Face. Default is "facebook/dinov2-small".
    out_dim : int, optional
        Dimension of the final output embedding. Default is 512.
    image_size : int, optional
        Input resolution for the pretrained backbone. Default is 224.
    freeze_backbone : bool, optional
        If True, freezes all backbone parameters (no fine-tuning). Default is True.
    device : str, optional
        Device to use ("cuda", "cpu"). If None, chooses automatically. Default is None.
    **kwargs :
        Additional keyword arguments (ignored, kept for compatibility).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        out_dim: int = 512,
        image_size: int = 224,
        freeze_backbone: bool = True,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Load pretrained backbone
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)

        # Get output dimension from the backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            sample_out = self.model(pixel_values=dummy).last_hidden_state[:, 0].shape[-1]

        # Projection layer to desired feature size
        self.proj = nn.Linear(sample_out, out_dim)

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        self.to(self.device)

    @staticmethod
    def channel_adapter(x: torch.Tensor) -> torch.Tensor:
        """
        Adapt a 4-channel input to 3 channels by averaging adjacent channels.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, 4, H, W).

        Returns
        -------
        Tensor
            Adapted tensor of shape (B, 3, H, W).

        Raises
        ------
        ValueError
            If the number of input channels is not equal to 4.
        """
        if x.size(1) != 4:
            raise ValueError(f"channel_adapter expects 4-channel input, got {x.size(1)} channels.")
        return (x[:, :-1] + x[:, 1:]) / 2  # (B, 3, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pretrained encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, 4, H, W) with pixel values in [0, 255].

        Returns
        -------
        Tensor
            Encoded feature tensor of shape (B, out_dim).
        """
        x = self.channel_adapter(x)
        x = x / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        with torch.no_grad():
            feats = self.model(pixel_values=x).last_hidden_state[:, 0]

        return self.proj(feats)
