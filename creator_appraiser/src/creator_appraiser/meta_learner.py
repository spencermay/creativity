"""Meta-learning training loop: orchestrates Creator-Appraiser joint training.

Training proceeds in three phases:

Phase 1 - Creator pre-training:
    Train the DDPM Creator on MNIST with standard diffusion loss.

Phase 2 - Appraiser warmup:
    Train the autoencoder Appraiser on MNIST reconstruction. Then freeze
    encoder + most of decoder, save phi0 (first decoder layer weights).

Phase 3 - Joint training:
    For each episode:
    1. Creator generates an image c (conditioned on a class).
    2. Appraiser resets first decoder layer to phi0.
    3. Record l0 = reconstruction loss on c.
    4. Inner loop: train first decoder layer for T steps on c.
    5. Record lT = reconstruction loss after adaptation.
    6. Compute reward R = (l0 - lT) * exp(-(l0-mu)^2/tau^2) * exp(-lT/tau)
    7. Creator loss = -R (weighted diffusion loss via REINFORCE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .appraiser import Appraiser
from .creator import DDPMCreator


@dataclass
class MetaTrainConfig:
    """Configuration for the meta-learning training loop."""

    # Outer-loop learning rates
    outer_lr_creator: float = 0.0002
    outer_lr_appraiser: float = 0.001

    # Inner-loop (Appraiser decoder adaptation)
    inner_lr: float = 0.01
    inner_steps: int = 10  # T steps of decoder first-layer adaptation

    # Episode structure
    n_way: int = 3
    k_shot: int = 1       # Images per class per episode (for the reward computation)

    # Training schedule
    max_episodes: int = 5000
    creator_pretrain_epochs: int = 10
    warmup_episodes: int = 500  # Autoencoder warmup epochs (not episodes)

    # Reward hyperparameters
    mu: float = 1.0       # Target initial loss (calibrate to your l0 values)
    tau: float = 0.5      # Temperature (controls width of Gaussian novelty bonus)

    # Data
    img_channels: int = 1
    img_size: int = 28
    n_classes: int = 10
    data_root: str = "data"
    batch_size_diffusion: int = 256
    max_pretrain_samples: int | None = 5000

    # Diffusion sampling
    sample_skip_steps: int = 10
    timesteps: int = 50
    guidance_scale: float = 1.0   # Classifier-free guidance (>1.0 = stronger class signal)

    # Evolution strategies (noise optimization)
    n_perturbations: int = 8      # Number of random perturbations per ES step
    perturbation_std: float = 0.05  # Std of perturbations to x_T

    # Infrastructure
    device: str = "cpu"
    log_interval: int = 50
    save_interval: int = 1000
    output_dir: str = "outputs"


class MetaLearner:
    """Orchestrates the Creator-Appraiser meta-learning training.

    Three-phase training:
    1. DDPM Creator pre-training on MNIST
    2. Autoencoder Appraiser warmup on MNIST reconstruction
    3. Joint training with creativity reward

    Args:
        creator: DDPMCreator model.
        appraiser: Appraiser autoencoder model.
        config: Training configuration.
    """

    def __init__(
        self,
        creator: DDPMCreator,
        appraiser: Appraiser,
        config: MetaTrainConfig,
    ):
        self.creator = creator.to(config.device)
        self.appraiser = appraiser.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        # Optimizers
        self.creator_optimizer = optim.Adam(
            self.creator.parameters(), lr=config.outer_lr_creator
        )
        self.appraiser_optimizer = optim.Adam(
            self.appraiser.parameters(), lr=config.outer_lr_appraiser
        )

        # Training state
        self.episode_count = 0
        self.history: list[dict[str, float]] = []

    # ==============================================================
    # Phase 1: Creator (DDPM) pre-training
    # ==============================================================

    def pretrain_creator(self, n_epochs: int | None = None, refresh_interval: int = 30) -> None:
        """Pre-train the DDPM Creator on MNIST with standard diffusion loss.

        Loads a fresh random subset of MNIST every `refresh_interval` epochs
        to avoid overfitting. Resets the optimizer on each refresh to clear
        stale momentum and uses cosine annealing within each refresh window.

        Args:
            n_epochs: Total number of epochs to train.
            refresh_interval: Load a new random subset every this many epochs.
        """
        from .data import create_mnist_diffusion_dataloader

        if n_epochs is None:
            n_epochs = self.config.creator_pretrain_epochs

        # Use a standard diffusion training LR (not the noise optimization LR)
        pretrain_lr = 2e-4

        print(f"Phase 1: Pre-training Creator (DDPM) for {n_epochs} epochs "
              f"(lr={pretrain_lr}, refreshing data every {refresh_interval} epochs)...")

        self.creator.train()
        dataloader = None

        # Single persistent optimizer — do NOT reset on each refresh
        optimizer = optim.Adam(
            self.creator.parameters(), lr=pretrain_lr, betas=(0.9, 0.999)
        )
        # Cosine annealing with warm restarts every refresh_interval epochs
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=refresh_interval, T_mult=1, eta_min=pretrain_lr * 0.1
        )

        for epoch in range(n_epochs):
            # Refresh dataset every refresh_interval epochs (but keep optimizer state)
            if epoch % refresh_interval == 0:
                # Generate and save a sample image to check progress
                self._save_pretrain_sample(epoch)

                current_seed = torch.randint(0, 2**31, (1,)).item()
                dataloader = create_mnist_diffusion_dataloader(
                    root=self.config.data_root,
                    batch_size=self.config.batch_size_diffusion,
                    train=True,
                    img_size=self.config.img_size,
                    max_samples=self.config.max_pretrain_samples,
                    subset_seed=current_seed,
                )
                print(f"  [Loaded fresh subset, seed={current_seed}]", flush=True)

            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                loss = self.creator.diffusion_loss(images, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.creator.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % 30 == 0:
                    print(f"    batch {batch_idx + 1}, loss: {loss.item():.4f}", flush=True)

            scheduler.step(epoch)
            avg_loss = epoch_loss / n_batches
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [Epoch {epoch + 1}/{n_epochs}] Diffusion Loss: {avg_loss:.4f} "
                  f"(lr={current_lr:.6f})", flush=True)

        # Update the main optimizer reference
        self.creator_optimizer = optimizer
        print("Creator pre-training complete.")

    def _save_pretrain_sample(self, epoch: int) -> None:
        """Generate and save a sample grid during pre-training to check progress."""
        import os
        import numpy as np

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        self.creator.eval()

        # Generate one image per class (0-9)
        labels = torch.arange(self.config.n_classes, device=self.device)
        x_T = torch.randn(
            self.config.n_classes, self.config.img_channels,
            self.config.img_size, self.config.img_size, device=self.device
        )

        with torch.no_grad():
            images = self.creator.sample_from_noise(
                x_T, labels,
                skip_steps=self.config.sample_skip_steps,
                guidance_scale=self.config.guidance_scale,
            )

        # Save as a single row of digits
        fig, axes = plt.subplots(1, self.config.n_classes, figsize=(2 * self.config.n_classes, 2))
        for i in range(self.config.n_classes):
            img = images[i].squeeze().cpu().numpy()
            img = (img + 1) / 2
            axes[i].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(str(i), fontsize=8)
            axes[i].axis("off")

        plt.suptitle(f"DDPM samples @ epoch {epoch}", fontsize=10)
        plt.tight_layout()

        samples_dir = os.path.join(self.config.output_dir, "pretrain_samples")
        os.makedirs(samples_dir, exist_ok=True)
        path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  [Saved sample: {path}]", flush=True)

        self.creator.train()

    # ==============================================================
    # Phase 2: Appraiser warmup (autoencoder reconstruction)
    # ==============================================================

    def warmup_appraiser(self, n_epochs: int | None = None) -> None:
        """Pre-train Appraiser autoencoder on MNIST reconstruction.

        Trains the full autoencoder (encoder + decoder) to reconstruct MNIST
        digits. After training, freezes encoder + decoder except first_layer,
        and saves phi0.

        Args:
            n_epochs: Number of training epochs.
        """
        from .data import create_mnist_diffusion_dataloader

        if n_epochs is None:
            n_epochs = self.config.warmup_episodes  # Reused as epochs here

        print(f"Phase 2: Warming up Appraiser (autoencoder) for {n_epochs} epochs...")

        dataloader = create_mnist_diffusion_dataloader(
            root=self.config.data_root,
            batch_size=self.config.batch_size_diffusion,
            train=True,
            img_size=self.config.img_size,
            max_samples=self.config.max_pretrain_samples,
        )

        self.appraiser.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.appraiser_optimizer.zero_grad()
                loss = self.appraiser.reconstruction_loss(images, labels)
                loss.backward()
                self.appraiser_optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            print(f"  [Epoch {epoch + 1}/{n_epochs}] Recon Loss: {avg_loss:.4f}")

        print("Appraiser warmup complete.")

        # Save phi0 and freeze
        self.appraiser.save_phi0()
        self.appraiser.freeze_except_first_decoder_layer()

        # Rebuild appraiser optimizer with only trainable params (first_layer)
        trainable_params = [p for p in self.appraiser.parameters() if p.requires_grad]
        self.appraiser_optimizer = optim.Adam(
            trainable_params, lr=self.config.inner_lr
        )

        # Freeze Creator UNet — during joint training we optimize noise, not weights
        for param in self.creator.parameters():
            param.requires_grad = False
        print("  Creator UNet frozen (noise optimization mode).")

    # ==============================================================
    # Phase 3: Creative image generation via noise optimization
    # ==============================================================

    def generate_creative_image(
        self,
        class_label: int | None = None,
        K: int = 50,
        lr: float | None = None,
        verbose: bool = False,
        n_perturbations: int | None = None,
        perturbation_std: float | None = None,
    ) -> dict[str, Any]:
        """Generate a single creative image via K steps of noise optimization.

        Uses evolution strategies (ES) to optimize x_T: at each step, perturb
        x_T in multiple random directions, evaluate the reward for each, and
        step in the reward-weighted average direction.

        Args:
            class_label: Digit class to condition on. Random if None.
            K: Number of noise optimization steps.
            lr: Learning rate for noise updates. Uses config default if None.
            verbose: Print per-step metrics.
            n_perturbations: ES population size. Uses config default if None.
            perturbation_std: Perturbation magnitude. Uses config default if None.

        Returns:
            Dictionary with image, initial_image, class_label, history.
        """
        if lr is None:
            lr = self.config.outer_lr_creator
        if n_perturbations is None:
            n_perturbations = self.config.n_perturbations
        if perturbation_std is None:
            perturbation_std = self.config.perturbation_std

        if class_label is None:
            class_label = torch.randint(0, self.config.n_classes, (1,)).item()
        label_tensor = torch.tensor([class_label], device=self.device)

        # Initialize noise — this is what we optimize
        x_T = torch.randn(
            1, self.config.img_channels, self.config.img_size, self.config.img_size,
            device=self.device, requires_grad=True,
        )

        noise_optimizer = optim.Adam([x_T], lr=lr)
        step_history = []
        initial_image = None

        for k in range(K):
            noise_optimizer.zero_grad()

            # (a) Denoise x_T → image c (differentiable through frozen DDPM)
            generated = self.creator.sample_from_noise(
                x_T, label_tensor,
                skip_steps=self.config.sample_skip_steps,
                guidance_scale=self.config.guidance_scale,
            )

            if k == 0:
                initial_image = generated.detach().clone()

            # (b) Reset Appraiser first_layer to phi0
            self.appraiser.reset_to_phi0()

            # Compute l0 as a DIFFERENTIABLE tensor.
            # l0 = reconstruction loss at phi0 (before inner loop).
            # Encoder is frozen, so grad flows: generated → encoder(frozen) → z → first_layer(phi0) → recon → loss
            z_for_l0 = self.appraiser.encoder(generated)
            recon_0 = self.appraiser.decoder(z_for_l0, label_tensor)
            l0_tensor = F.mse_loss(recon_0, generated)
            l0 = l0_tensor.item()

            # (c) DIFFERENTIABLE inner loop using `higher`
            import higher

            inner_opt = optim.SGD(
                self.appraiser.decoder.first_layer.parameters(),
                lr=self.config.inner_lr,
            )

            with higher.innerloop_ctx(
                self.appraiser.decoder.first_layer,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=True,
            ) as (fmodel, diffopt):
                for _ in range(self.config.inner_steps):
                    z = self.appraiser.encoder(generated)
                    recon = self.appraiser.decoder.forward_with_first_layer(
                        z, label_tensor, fmodel
                    )
                    inner_loss = F.mse_loss(recon, generated)
                    diffopt.step(inner_loss)

                # (d) Compute lT — DIFFERENTIABLE w.r.t. generated → x_T
                z = self.appraiser.encoder(generated)
                recon_T = self.appraiser.decoder.forward_with_first_layer(
                    z, label_tensor, fmodel
                )
                lT_tensor = F.mse_loss(recon_T, generated)

            lT = lT_tensor.item()

            # Compute R as a DIFFERENTIABLE tensor
            mu = self.config.mu
            tau = self.config.tau
            gap_tensor = l0_tensor - lT_tensor
            novelty_term = torch.exp(-((l0_tensor - mu) ** 2) / (tau ** 2))
            memorability_term = torch.exp(-lT_tensor / tau)
            R_tensor = gap_tensor * novelty_term * memorability_term

            R = R_tensor.item()

            step_history.append({
                "step": k, "reward": R, "l0": l0, "lT": lT, "gap": l0 - lT,
            })

            if verbose and (k + 1) % 10 == 0:
                print(f"    [Step {k+1}/{K}] R={R:.6f} "
                      f"l0={l0:.4f} lT={lT:.4f} gap={l0 - lT:.4f}")

            # (e) Backprop: MAXIMIZE R → minimize -R
            creator_loss = -R_tensor
            creator_loss.backward()
            noise_optimizer.step()

        # Final image
        with torch.no_grad():
            final_image = self.creator.sample_from_noise(
                x_T, label_tensor,
                skip_steps=self.config.sample_skip_steps,
                guidance_scale=self.config.guidance_scale,
            )

        return {
            "image": final_image.detach().cpu(),
            "initial_image": initial_image.cpu(),
            "class_label": class_label,
            "history": step_history,
        }

    def _evaluate_reward(
        self, image: torch.Tensor, label_tensor: torch.Tensor
    ) -> dict[str, float]:
        """Evaluate the creativity reward for a single image.

        Resets Appraiser to phi0, computes l0, runs inner loop, computes lT and R.

        Args:
            image: Generated image (1, C, H, W).
            label_tensor: Class label tensor (1,).

        Returns:
            Dict with keys: R, l0, lT, gap.
        """
        self.appraiser.reset_to_phi0()

        with torch.no_grad():
            l0 = self.appraiser.reconstruction_loss(image, label_tensor).item()

        # Inner loop: adapt first_layer
        inner_opt = optim.SGD(
            self.appraiser.decoder.first_layer.parameters(),
            lr=self.config.inner_lr,
        )

        for _ in range(self.config.inner_steps):
            inner_opt.zero_grad()
            loss = self.appraiser.reconstruction_loss(image, label_tensor)
            loss.backward()
            inner_opt.step()

        with torch.no_grad():
            lT = self.appraiser.reconstruction_loss(image, label_tensor).item()

        mu = self.config.mu
        tau = self.config.tau
        gap = l0 - lT
        novelty_exp = min((l0 - mu) ** 2 / (tau ** 2), 20.0)
        memorability_exp = min(lT / tau, 20.0)
        novelty_bonus = torch.exp(torch.tensor(-novelty_exp)).item()
        memorability_bonus = torch.exp(torch.tensor(-memorability_exp)).item()
        R = gap * novelty_bonus * memorability_bonus

        return {"R": R, "l0": l0, "lT": lT, "gap": gap}

    def generate_creative_batch(
        self,
        n_images: int = 8,
        K: int = 50,
        class_labels: list[int] | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate multiple creative images, resetting after each.

        Args:
            n_images: Number of images to generate.
            K: Noise optimization steps per image.
            class_labels: Optional list of class labels. Random if None.
            verbose: Print progress.

        Returns:
            List of result dicts from generate_creative_image.
        """
        results = []

        if class_labels is None:
            class_labels = [None] * n_images

        for i in range(n_images):
            if verbose:
                label_str = f"class={class_labels[i]}" if class_labels[i] is not None else "random"
                print(f"  Generating image {i+1}/{n_images} ({label_str})...")

            result = self.generate_creative_image(
                class_label=class_labels[i], K=K, verbose=verbose,
            )
            results.append(result)

            if verbose:
                final = result["history"][-1]
                print(f"    Final: R={final['reward']:.6f} gap={final['gap']:.4f}")

        return results

    # ==============================================================
    # Legacy: single-step episode (for backwards compat with train loop)
    # ==============================================================

    def train_episode(self) -> dict[str, float]:
        """Run one creative image generation as a training episode.

        Generates a single image via K steps of noise optimization.
        Returns aggregated metrics.
        """
        result = self.generate_creative_image(K=50, verbose=False)
        history = result["history"]

        # Aggregate metrics from the K steps
        final = history[-1]
        mean_reward = sum(h["reward"] for h in history) / len(history)

        metrics = {
            "creator_loss": 0.0,
            "mean_reward": mean_reward,
            "final_reward": final["reward"],
            "mean_l0": sum(h["l0"] for h in history) / len(history),
            "mean_lT": sum(h["lT"] for h in history) / len(history),
            "mean_gap": sum(h["gap"] for h in history) / len(history),
            "final_gap": final["gap"],
        }

        self.episode_count += 1
        self.history.append(metrics)

        return metrics

    # ==============================================================
    # Training orchestration
    # ==============================================================

    def train(self, callback: Any | None = None) -> list[dict[str, float]]:
        """Run the full meta-training loop (Phase 3).

        Args:
            callback: Optional callable(episode_idx, metrics).

        Returns:
            List of per-episode metric dictionaries.
        """
        for episode_idx in range(self.config.max_episodes):
            metrics = self.train_episode()

            if callback is not None:
                callback(episode_idx, metrics)

            if (episode_idx + 1) % self.config.log_interval == 0:
                self._log_progress(episode_idx, metrics)

            if (episode_idx + 1) % self.config.save_interval == 0:
                self.save_checkpoint(episode_idx)

        return self.history

    def _log_progress(self, episode_idx: int, metrics: dict[str, float]) -> None:
        """Log training progress."""
        print(
            f"[Image {episode_idx + 1}/{self.config.max_episodes}] "
            f"Final R: {metrics['final_reward']:.6f} | "
            f"Avg R: {metrics['mean_reward']:.6f} | "
            f"l0: {metrics['mean_l0']:.4f} | "
            f"lT: {metrics['mean_lT']:.4f} | "
            f"Gap: {metrics['final_gap']:.4f}"
        )

    def save_checkpoint(self, episode_idx: int) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"checkpoint_ep{episode_idx + 1}.pt")

        torch.save({
            "episode": episode_idx + 1,
            "creator_state_dict": self.creator.state_dict(),
            "appraiser_state_dict": self.appraiser.state_dict(),
            "appraiser_phi0": self.appraiser._phi0,
            "creator_optimizer_state_dict": self.creator_optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.creator.load_state_dict(checkpoint["creator_state_dict"])
        self.appraiser.load_state_dict(checkpoint["appraiser_state_dict"])

        # Restore phi0 if saved
        if "appraiser_phi0" in checkpoint and checkpoint["appraiser_phi0"] is not None:
            self.appraiser._phi0 = {
                k: v.to(self.device) for k, v in checkpoint["appraiser_phi0"].items()
            }

        self.episode_count = checkpoint.get("episode", 0)
        self.history = checkpoint.get("history", [])

        # Rebuild creator optimizer
        self.creator_optimizer = optim.Adam(
            self.creator.parameters(), lr=self.config.outer_lr_creator
        )

        print(f"Loaded checkpoint from episode {self.episode_count}")
