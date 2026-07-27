"""Creator model: class-conditional DDPM diffusion model.

Implements a Denoising Diffusion Probabilistic Model (Ho et al., 2020) with
class-conditional generation for producing novel digit/concept images.

The Creator generates images via iterative denoising:
1. Start from pure Gaussian noise x_T
2. Iteratively denoise: x_{t-1} = denoise(x_t, t, class_label)
3. Final x_0 is the generated image

Architecture:
- UNet with time embeddings and class conditioning
- Linear beta schedule
- Standard DDPM forward/reverse process

For the Creator-Appraiser framework, the Creator is trained with:
- Standard diffusion loss (reconstruct clean images from noisy versions)
- Creativity objective (from Appraiser feedback via meta-learning)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Diffusion schedule utilities
# ============================================================


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear schedule for beta (noise variance) from DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Cosine schedule from Improved DDPM (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputed diffusion schedule constants.

    Stores all the alpha, beta, and derived values needed for the
    forward (noising) and reverse (denoising) processes.
    """

    def __init__(self, timesteps: int = 1000, schedule: str = "linear"):
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Calculations for forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Calculations for reverse process posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionSchedule":
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


# ============================================================
# UNet building blocks
# ============================================================


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding (Vaswani et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


def _num_groups(channels: int) -> int:
    """Pick a reasonable number of groups for GroupNorm given channel count."""
    for g in [32, 16, 8, 4]:
        if channels % g == 0 and channels // g >= 1:
            return g
    return 1


class ResBlock(nn.Module):
    """Residual block with time and class conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, class_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(_num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.class_mlp(c_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ============================================================
# UNet denoiser
# ============================================================


class UNet(nn.Module):
    """Compact UNet for DDPM denoising with time + class conditioning.

    Designed for 28x28 or 32x32 images (MNIST-scale).

    Args:
        img_channels: Input/output image channels (1 for MNIST).
        base_channels: Base channel multiplier.
        channel_mults: Channel multiplier at each resolution level.
        n_classes: Number of conditioning classes.
        time_emb_dim: Dimensionality of time embedding.
    """

    def __init__(
        self,
        img_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        n_classes: int = 10,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.n_classes = n_classes

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding
        class_emb_dim = time_emb_dim
        self.class_emb = nn.Embedding(n_classes, class_emb_dim)

        # Initial conv
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels

        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(ResBlock(in_ch, out_ch, time_emb_dim, class_emb_dim))
            channels.append(out_ch)
            self.downsamples.append(Downsample(out_ch))
            in_ch = out_ch

        # Bottleneck
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim, class_emb_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim, class_emb_dim)

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            # Input includes skip connection
            self.upsamples.append(Upsample(in_ch))
            self.up_blocks.append(ResBlock(in_ch + out_ch, out_ch, time_emb_dim, class_emb_dim))
            in_ch = out_ch

        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(_num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, img_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise given noisy image, timestep, and class label.

        Args:
            x: Noisy image (B, C, H, W).
            t: Timestep indices (B,) in [0, T-1].
            class_labels: Class conditioning labels (B,).

        Returns:
            Predicted noise (B, C, H, W).
        """
        # Embeddings
        t_emb = self.time_emb(t)
        c_emb = self.class_emb(class_labels)

        # Initial conv
        h = self.init_conv(x)

        # Encoder
        skips = [h]
        for down_block, downsample in zip(self.down_blocks, self.downsamples):
            h = down_block(h, t_emb, c_emb)
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb, c_emb)
        h = self.mid_block2(h, t_emb, c_emb)

        # Decoder
        for up_block, upsample in zip(self.up_blocks, self.upsamples):
            h = upsample(h)
            skip = skips.pop()
            # Handle size mismatch from downsampling odd dimensions
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb, c_emb)

        return self.final(h)


# ============================================================
# DDPM Creator
# ============================================================


class DDPMCreator(nn.Module):
    """DDPM-based Creator for the Creator-Appraiser framework.

    Generates class-conditional images via iterative denoising. Can be
    trained with both:
    - Standard diffusion loss (denoise real images)
    - Creativity loss (from Appraiser feedback)

    Args:
        unet: The UNet denoiser model.
        timesteps: Number of diffusion timesteps.
        schedule: Beta schedule type ("linear" or "cosine").
        n_way: Number of classes per episode.
        k_shot: Images per class per episode.
    """

    def __init__(
        self,
        unet: UNet,
        timesteps: int = 1000,
        schedule: str = "linear",
        n_way: int = 5,
        k_shot: int = 1,
    ):
        super().__init__()
        self.unet = unet
        self.n_way = n_way
        self.k_shot = k_shot
        self.timesteps = timesteps

        # Precompute schedule
        self.schedule = DiffusionSchedule(timesteps, schedule)

    def to(self, device: torch.device) -> "DDPMCreator":
        """Override to also move schedule tensors."""
        super().to(device)
        self.schedule.to(device)
        return self

    # ----------------------------------------------------------
    # Forward process (noising)
    # ----------------------------------------------------------

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean images.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0: Clean images (B, C, H, W).
            t: Timestep indices (B,).
            noise: Optional pre-sampled noise.

        Returns:
            Noisy images x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    # ----------------------------------------------------------
    # Training loss
    # ----------------------------------------------------------

    def diffusion_loss(
        self, x_0: torch.Tensor, class_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard DDPM training loss (simplified, predicting noise).

        L = E[||epsilon - epsilon_theta(x_t, t, c)||^2]

        Args:
            x_0: Clean images (B, C, H, W).
            class_labels: Class labels (B,).

        Returns:
            Scalar MSE loss.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward process: get noisy images
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.unet(x_t, t, class_labels)

        # MSE loss between true and predicted noise
        return F.mse_loss(noise_pred, noise)

    # ----------------------------------------------------------
    # Reverse process (sampling / generation)
    # ----------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self, x_t: torch.Tensor, t: int, class_labels: torch.Tensor
    ) -> torch.Tensor:
        """Single reverse diffusion step: x_t -> x_{t-1}.

        Args:
            x_t: Current noisy images (B, C, H, W).
            t: Current timestep (scalar).
            class_labels: Class labels (B,).

        Returns:
            Denoised images x_{t-1}.
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = self.unet(x_t, t_tensor, class_labels)

        # Compute x_{t-1}
        alpha = self.schedule.alphas[t]
        alpha_bar = self.schedule.alphas_cumprod[t]
        beta = self.schedule.betas[t]

        # Mean of p(x_{t-1} | x_t)
        mean = (1.0 / alpha.sqrt()) * (
            x_t - (beta / (1.0 - alpha_bar).sqrt()) * noise_pred
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.schedule.posterior_variance[t].sqrt()
            return mean + sigma * noise
        else:
            return mean

    @torch.no_grad()
    def sample(
        self,
        class_labels: torch.Tensor,
        img_size: int = 28,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate images via full reverse diffusion process.

        Args:
            class_labels: Class labels to condition on (B,).
            img_size: Spatial resolution of generated images.
            device: Device for computation.

        Returns:
            Generated images (B, C, H, W) in [-1, 1] range.
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = class_labels.shape[0]
        img_channels = self.unet.img_channels

        # Start from pure noise
        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)

        # Iterative denoising
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, class_labels)

        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_fast(
        self,
        class_labels: torch.Tensor,
        img_size: int = 28,
        device: torch.device | None = None,
        skip_steps: int = 10,
    ) -> torch.Tensor:
        """Fast sampling with stride (non-differentiable version).

        Args:
            class_labels: Class labels (B,).
            img_size: Image resolution.
            device: Device.
            skip_steps: Sample every N-th timestep.

        Returns:
            Generated images (B, C, H, W).
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = class_labels.shape[0]
        img_channels = self.unet.img_channels

        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)

        # Subsample timesteps
        timesteps = list(range(0, self.timesteps, skip_steps))[::-1]

        for t in timesteps:
            x = self.p_sample(x, t, class_labels)

        return x.clamp(-1, 1)

    def sample_from_noise(
        self,
        x_T: torch.Tensor,
        class_labels: torch.Tensor,
        skip_steps: int = 10,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable sampling from a given initial noise x_T.

        Unlike sample_fast, this keeps gradients flowing through the
        denoising process so we can optimize x_T via backprop.

        Supports classifier-free guidance: at each step, combines
        unconditional and conditional noise predictions to amplify
        class-conditional signal.

        Args:
            x_T: Initial noise tensor (B, C, H, W) — the variable we optimize.
            class_labels: Class labels (B,).
            skip_steps: Sample every N-th timestep.
            guidance_scale: Classifier-free guidance scale.
                            1.0 = no guidance, >1.0 = stronger class conditioning.

        Returns:
            Generated images (B, C, H, W) with grad connected to x_T.
        """
        batch_size = x_T.shape[0]
        device = x_T.device
        x = x_T

        timesteps = list(range(0, self.timesteps, skip_steps))[::-1]

        # Null label for unconditional prediction (use label 0 as proxy)
        null_labels = torch.zeros_like(class_labels)

        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if guidance_scale != 1.0:
                # Classifier-free guidance: combine conditional + unconditional
                noise_cond = self.unet(x, t_tensor, class_labels)
                noise_uncond = self.unet(x, t_tensor, null_labels)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.unet(x, t_tensor, class_labels)

            # Deterministic DDIM-like step
            alpha = self.schedule.alphas[t]
            alpha_bar = self.schedule.alphas_cumprod[t]
            beta = self.schedule.betas[t]

            mean = (1.0 / alpha.sqrt()) * (
                x - (beta / (1.0 - alpha_bar).sqrt()) * noise_pred
            )

            x = mean

        return x.clamp(-1, 1)

    # ----------------------------------------------------------
    # Episode generation for meta-learning
    # ----------------------------------------------------------

    @torch.no_grad()
    def generate_episode(
        self,
        class_ids: torch.Tensor | None = None,
        device: torch.device | None = None,
        fast: bool = True,
        skip_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate a full meta-learning episode (support + query sets).

        Args:
            class_ids: Optional tensor of class IDs to use.
                       If None, samples n_way random classes from [0, n_classes).
            device: Device for tensor allocation.
            fast: Use fast (strided) sampling for speed during training.
            skip_steps: Override skip_steps for fast sampling.

        Returns:
            Dictionary with keys:
                - 'support_images': (n_way * k_shot, C, H, W)
                - 'support_labels': (n_way * k_shot,)
                - 'query_images': (n_way * q_query, C, H, W)
                - 'query_labels': (n_way * q_query,)
        """
        if device is None:
            device = next(self.parameters()).device

        n_classes = self.unet.n_classes

        if class_ids is None:
            class_ids = torch.randperm(n_classes, device=device)[: self.n_way]

        images_list = []
        labels_list = []

        sample_fn = self.sample_fast if fast else self.sample
        sample_kwargs = {"img_size": 28, "device": device}
        if fast and skip_steps is not None:
            sample_kwargs["skip_steps"] = skip_steps

        for local_label, class_id in enumerate(class_ids):
            labels_c = class_id.expand(self.k_shot).to(device)
            imgs_c = sample_fn(labels_c, **sample_kwargs)
            images_list.append(imgs_c)
            labels_list.append(
                torch.full((self.k_shot,), class_id.item(), device=device, dtype=torch.long)
            )

        return {
            "images": torch.cat(images_list, dim=0),
            "class_labels": torch.cat(labels_list, dim=0),
        }
