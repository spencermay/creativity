"""Appraiser model: Convolutional Autoencoder for evaluating memorability.

The Appraiser is a class-conditional autoencoder that measures how "learnable"
a generated image is. The key idea:

- The encoder is fully frozen after warmup (extracts features).
- The decoder's first layer (phi0) is the only trainable part during joint training.
- Given a creation from the Creator, the Appraiser:
  1. Resets the first decoder layer to its initial state phi0.
  2. Measures reconstruction loss l0 (before adaptation).
  3. Trains the first decoder layer for T steps to minimize reconstruction loss.
  4. Measures reconstruction loss lT (after adaptation).
  5. The "learnability gap" (l0 - lT) indicates memorability.

Architecture:
- Encoder: Conv layers → latent representation (fully frozen after warmup)
- Decoder: Latent → reconstructed image (only first layer trainable)
- Class conditioning via embedding added to the latent space
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder for the Appraiser autoencoder.

    Extracts a compact latent representation from input images.
    Fully frozen after warmup.

    Args:
        img_channels: Number of input image channels.
        hidden_dim: Base channel count.
        latent_dim: Dimensionality of the latent vector.
        img_size: Input image resolution (assumes square).
    """

    def __init__(
        self,
        img_channels: int = 1,
        hidden_dim: int = 32,
        latent_dim: int = 64,
        img_size: int = 28,
    ):
        super().__init__()
        # 28 -> 14 -> 7 -> 3 (with padding adjustments)
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, 3, stride=2, padding=1),      # 28->14
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),    # 14->7
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),  # 7->4
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size: hidden_dim*4 * 4 * 4 for img_size=28
        self._flat_size = hidden_dim * 4 * (img_size // 8) * (img_size // 8)
        # Handle img_size=28: 28//8=3, so 3*3 spatial
        # Actually: 28 -> ceil(28/2)=14 -> ceil(14/2)=7 -> ceil(7/2)=4
        self._spatial = img_size
        # Compute actual spatial size after 3 stride-2 convs
        s = img_size
        for _ in range(3):
            s = (s + 1) // 2  # ceil division with padding=1, stride=2
        self._spatial_out = s
        self._flat_size = hidden_dim * 4 * s * s

        self.fc = nn.Linear(self._flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent vectors.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Latent vectors (B, latent_dim).
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class Decoder(nn.Module):
    """Convolutional decoder for the Appraiser autoencoder.

    Reconstructs images from latent vectors + class conditioning.
    After warmup, only the first layer (self.first_layer) remains trainable.

    Args:
        img_channels: Number of output image channels.
        hidden_dim: Base channel count.
        latent_dim: Dimensionality of the latent input.
        n_classes: Number of conditioning classes.
        img_size: Output image resolution.
    """

    def __init__(
        self,
        img_channels: int = 1,
        hidden_dim: int = 32,
        latent_dim: int = 64,
        n_classes: int = 10,
        img_size: int = 28,
    ):
        super().__init__()
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        # Class conditioning
        self.class_emb = nn.Embedding(n_classes, latent_dim)

        # Compute spatial size to project to (mirror encoder)
        s = img_size
        for _ in range(3):
            s = (s + 1) // 2
        self._spatial_out = s
        self._proj_size = hidden_dim * 4 * s * s

        # First layer: the only part that stays trainable during joint training
        # Takes latent + class_emb and projects to spatial feature map
        self.first_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, self._proj_size),
            nn.ReLU(inplace=True),
        )

        # Remaining decoder layers (frozen after warmup)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),  # s->2s
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),      # 2s->4s
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, img_channels, 4, stride=2, padding=1),        # 4s->8s
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images.

        Args:
            z: Latent vectors (B, latent_dim).
            class_labels: Class labels (B,).

        Returns:
            Reconstructed images (B, C, H, W).
        """
        # Combine latent with class embedding
        c = self.class_emb(class_labels)
        zc = torch.cat([z, c], dim=1)

        # First layer (trainable during joint training)
        h = self.first_layer(zc)
        h = h.view(h.size(0), self.hidden_dim * 4, self._spatial_out, self._spatial_out)

        # Remaining layers (frozen after warmup)
        h = self.deconv(h)

        # Crop/interpolate to exact output size
        if h.shape[2] != self.img_size or h.shape[3] != self.img_size:
            h = F.interpolate(h, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return h

    def forward_with_first_layer(
        self, z: torch.Tensor, class_labels: torch.Tensor, functional_first_layer
    ) -> torch.Tensor:
        """Decode using an external functional first_layer (from `higher`).

        This allows differentiating through the inner loop adaptation
        of first_layer while keeping the rest of the decoder frozen.

        Args:
            z: Latent vectors (B, latent_dim).
            class_labels: Class labels (B,).
            functional_first_layer: A `higher` functional module replacing self.first_layer.

        Returns:
            Reconstructed images (B, C, H, W).
        """
        c = self.class_emb(class_labels)
        zc = torch.cat([z, c], dim=1)

        # Use the functional first_layer (tracked by `higher`)
        h = functional_first_layer(zc)
        h = h.view(h.size(0), self.hidden_dim * 4, self._spatial_out, self._spatial_out)

        # Remaining layers (frozen)
        h = self.deconv(h)

        if h.shape[2] != self.img_size or h.shape[3] != self.img_size:
            h = F.interpolate(h, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return h


class Appraiser(nn.Module):
    """Autoencoder-based Appraiser for the Creator-Appraiser framework.

    Evaluates creativity by measuring how quickly the decoder's first layer
    can learn to reconstruct a given creation. The full encoder is frozen;
    only the first decoder layer adapts.

    Training phases:
    1. Warmup: Train full autoencoder on MNIST reconstruction.
    2. Freeze: Freeze encoder + all decoder layers except first_layer.
    3. Joint: For each creation, reset first_layer to phi0, measure l0,
       adapt T steps, measure lT.

    Args:
        img_channels: Image channels.
        hidden_dim: Base channel count.
        latent_dim: Latent space dimensionality.
        n_classes: Number of classes for conditioning.
        img_size: Image resolution.
    """

    def __init__(
        self,
        img_channels: int = 1,
        hidden_dim: int = 32,
        latent_dim: int = 64,
        n_classes: int = 10,
        img_size: int = 28,
    ):
        super().__init__()
        self.encoder = Encoder(img_channels, hidden_dim, latent_dim, img_size)
        self.decoder = Decoder(img_channels, hidden_dim, latent_dim, n_classes, img_size)

        # phi0: saved initial state of the first decoder layer (set after warmup)
        self._phi0: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass: encode then decode.

        Args:
            x: Input images (B, C, H, W).
            class_labels: Class labels (B,).

        Returns:
            Reconstructed images (B, C, H, W).
        """
        z = self.encoder(x)
        return self.decoder(z, class_labels)

    def reconstruction_loss(
        self, x: torch.Tensor, class_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss (MSE).

        Args:
            x: Input images (B, C, H, W).
            class_labels: Class labels (B,).

        Returns:
            Scalar MSE reconstruction loss.
        """
        x_recon = self.forward(x, class_labels)
        return F.mse_loss(x_recon, x)

    def save_phi0(self) -> None:
        """Save the current first decoder layer state as phi0.

        Called after warmup, before joint training begins.
        """
        self._phi0 = {
            name: param.clone().detach()
            for name, param in self.decoder.first_layer.named_parameters()
        }

    def reset_to_phi0(self) -> None:
        """Reset the first decoder layer back to its saved initial state phi0.

        Called at the start of each creativity evaluation episode.
        """
        if self._phi0 is None:
            raise RuntimeError("phi0 not saved. Call save_phi0() after warmup.")

        with torch.no_grad():
            for name, param in self.decoder.first_layer.named_parameters():
                param.copy_(self._phi0[name])

    def freeze_except_first_decoder_layer(self) -> None:
        """Freeze entire model except decoder.first_layer.

        After warmup:
        - Encoder: fully frozen
        - Decoder.first_layer: trainable (reset to phi0 each episode)
        - Decoder.deconv + class_emb: frozen
        """
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only decoder.first_layer
        for param in self.decoder.first_layer.parameters():
            param.requires_grad = True

        n_frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"  Appraiser frozen: {n_frozen} param tensors frozen, "
              f"{n_trainable} trainable (decoder.first_layer)")
