"""Visualize creative image generation from the Creator-Appraiser framework.

Loads warmed-up Creator + Appraiser, runs the meta-learning loop to produce
creative images, and saves a visualization grid showing:
- Initial images (from random noise, before optimization)
- Final images (after K steps of noise optimization)
- Per-image reward curves

Usage:
    python scripts/visualize.py --config configs/fast.yaml --n_images 8 --K 50
    python scripts/visualize.py --config configs/fast.yaml --n_images 4 --K 100 --classes 0,1,2,3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from creator_appraiser.appraiser import Appraiser
from creator_appraiser.creator import DDPMCreator, UNet
from creator_appraiser.meta_learner import MetaLearner, MetaTrainConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_models(config: dict) -> tuple[DDPMCreator, Appraiser]:
    """Build models from config."""
    creator_cfg = config["creator"]
    appraiser_cfg = config["appraiser"]

    unet = UNet(
        img_channels=creator_cfg["img_channels"],
        base_channels=creator_cfg["base_channels"],
        channel_mults=tuple(creator_cfg["channel_mults"]),
        n_classes=creator_cfg["n_classes"],
        time_emb_dim=creator_cfg["time_emb_dim"],
    )

    creator = DDPMCreator(
        unet=unet,
        timesteps=creator_cfg["timesteps"],
        schedule=creator_cfg["schedule"],
        n_way=config["episode"]["n_way"],
        k_shot=config["episode"].get("k_shot", 1),
    )

    appraiser = Appraiser(
        img_channels=creator_cfg["img_channels"],
        hidden_dim=appraiser_cfg["hidden_dim"],
        latent_dim=appraiser_cfg["latent_dim"],
        n_classes=creator_cfg["n_classes"],
    )

    return creator, appraiser


def build_meta_config(config: dict) -> MetaTrainConfig:
    """Build MetaTrainConfig from config dict."""
    meta_cfg = config["meta"]
    training_cfg = config["training"]
    creator_cfg = config["creator"]
    appraiser_cfg = config["appraiser"]
    reward_cfg = config["reward"]
    data_cfg = config["data"]

    return MetaTrainConfig(
        outer_lr_creator=meta_cfg["outer_lr_creator"],
        outer_lr_appraiser=meta_cfg.get("outer_lr_appraiser", 0.001),
        inner_lr=appraiser_cfg["inner_lr"],
        inner_steps=appraiser_cfg["inner_steps"],
        n_way=config["episode"]["n_way"],
        k_shot=config["episode"].get("k_shot", 1),
        max_episodes=meta_cfg["max_episodes"],
        creator_pretrain_epochs=training_cfg["creator_pretrain_epochs"],
        warmup_episodes=training_cfg["appraiser_warmup_epochs"],
        mu=reward_cfg["mu"],
        tau=reward_cfg["tau"],
        img_channels=creator_cfg["img_channels"],
        img_size=creator_cfg["img_size"],
        n_classes=creator_cfg["n_classes"],
        data_root=data_cfg["root"],
        batch_size_diffusion=data_cfg["batch_size_diffusion"],
        max_pretrain_samples=data_cfg.get("max_pretrain_samples"),
        sample_skip_steps=creator_cfg["sample_skip_steps"],
        timesteps=creator_cfg["timesteps"],
        device=training_cfg["device"],
        log_interval=training_cfg["log_interval"],
        save_interval=training_cfg["save_interval"],
        output_dir=training_cfg["output_dir"],
    )


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Convert (1, 1, H, W) tensor in [-1,1] to (H, W) numpy in [0,1]."""
    img = t.squeeze().cpu().numpy()
    img = (img + 1) / 2  # [-1,1] → [0,1]
    return np.clip(img, 0, 1)


def visualize_results(
    results: list[dict],
    output_path: str,
    show: bool = False,
) -> None:
    """Create a visualization grid of creative images.

    Shows initial (random) vs final (optimized) images, plus reward curves.
    """
    n_images = len(results)
    fig, axes = plt.subplots(3, n_images, figsize=(3 * n_images, 9))

    if n_images == 1:
        axes = axes[:, np.newaxis]

    for i, result in enumerate(results):
        # Row 1: Initial image (before optimization)
        ax = axes[0, i]
        ax.imshow(tensor_to_image(result["initial_image"]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Initial (class {result['class_label']})", fontsize=9)
        ax.axis("off")

        # Row 2: Final image (after K steps)
        ax = axes[1, i]
        ax.imshow(tensor_to_image(result["image"]), cmap="gray", vmin=0, vmax=1)
        final_r = result["history"][-1]["reward"]
        ax.set_title(f"Final (R={final_r:.4f})", fontsize=9)
        ax.axis("off")

        # Row 3: Reward curve over K steps
        ax = axes[2, i]
        steps = [h["step"] for h in result["history"]]
        rewards = [h["reward"] for h in result["history"]]
        ax.plot(steps, rewards, "b-", linewidth=1)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Reward", fontsize=8)
        ax.set_title("Reward curve", fontsize=9)
        ax.tick_params(labelsize=7)

    plt.suptitle("Creator-Appraiser: Creative Image Generation", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

    if show:
        plt.show()
    plt.close()


def visualize_reward_components(
    results: list[dict],
    output_path: str,
    mu: float,
    tau: float,
) -> None:
    """Plot l0, lT, gap, and reward components over steps for each image."""
    n_images = len(results)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 3 * n_images))

    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i, result in enumerate(results):
        history = result["history"]
        steps = [h["step"] for h in history]
        l0s = [h["l0"] for h in history]
        lTs = [h["lT"] for h in history]
        gaps = [h["gap"] for h in history]
        rewards = [h["reward"] for h in history]

        # Panel 1: l0 and lT
        ax = axes[i, 0]
        ax.plot(steps, l0s, "r-", label="l0 (initial)")
        ax.plot(steps, lTs, "b-", label="lT (adapted)")
        ax.axhline(mu, color="g", linestyle="--", alpha=0.5, label=f"mu={mu}")
        ax.legend(fontsize=7)
        ax.set_title(f"Image {i+1} (class {result['class_label']}): Losses", fontsize=9)
        ax.set_xlabel("Step")

        # Panel 2: Gap
        ax = axes[i, 1]
        ax.plot(steps, gaps, "purple", linewidth=1)
        ax.set_title("Learnability Gap (l0 - lT)", fontsize=9)
        ax.set_xlabel("Step")

        # Panel 3: Reward
        ax = axes[i, 2]
        ax.plot(steps, rewards, "green", linewidth=1)
        ax.set_title("Reward R", fontsize=9)
        ax.set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward analysis to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize creative image generation")
    parser.add_argument("--config", type=str, default="configs/fast.yaml")
    parser.add_argument("--n_images", type=int, default=6, help="Number of images to generate")
    parser.add_argument("--K", type=int, default=50, help="Noise optimization steps per image")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class labels (e.g. '0,1,2,3')")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Display plot interactively")

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or os.path.join(config["training"]["output_dir"], "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Parse class labels
    class_labels = None
    if args.classes:
        class_labels = [int(x.strip()) for x in args.classes.split(",")]
        if len(class_labels) < args.n_images:
            # Repeat to fill
            class_labels = (class_labels * (args.n_images // len(class_labels) + 1))[:args.n_images]

    # Build models and meta-learner
    creator, appraiser = build_models(config)
    meta_config = build_meta_config(config)
    meta_learner = MetaLearner(creator=creator, appraiser=appraiser, config=meta_config)

    # Load warmup checkpoint
    warmup_path = os.path.join(config["training"]["output_dir"], "warmup_checkpoint.pt")
    if os.path.exists(warmup_path):
        meta_learner.load_checkpoint(warmup_path)
        # Freeze Creator + Appraiser setup
        for param in meta_learner.creator.parameters():
            param.requires_grad = False
        meta_learner.appraiser.freeze_except_first_decoder_layer()
        print()
    else:
        print(f"No warmup checkpoint found at {warmup_path}")
        print("Run training first: python scripts/train.py --config configs/fast.yaml")
        sys.exit(1)

    # Generate creative images
    print(f"Generating {args.n_images} creative images (K={args.K} steps each)...")
    print("=" * 60)

    results = meta_learner.generate_creative_batch(
        n_images=args.n_images,
        K=args.K,
        class_labels=class_labels,
        verbose=True,
    )

    # Visualize
    print("\n" + "=" * 60)
    print("Creating visualizations...")

    visualize_results(
        results,
        output_path=os.path.join(output_dir, "creative_images.png"),
        show=args.show,
    )

    visualize_reward_components(
        results,
        output_path=os.path.join(output_dir, "reward_analysis.png"),
        mu=meta_config.mu,
        tau=meta_config.tau,
    )

    # Print summary
    print("\nSummary:")
    for i, result in enumerate(results):
        final = result["history"][-1]
        init_r = result["history"][0]["reward"]
        print(f"  Image {i+1} (class {result['class_label']}): "
              f"R: {init_r:.6f} → {final['reward']:.6f}, "
              f"gap: {final['gap']:.4f}")


if __name__ == "__main__":
    main()
