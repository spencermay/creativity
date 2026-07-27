"""Training script for Creator-Appraiser (DDPM Creator + Autoencoder Appraiser).

Three-phase training:
1. Pre-train DDPM Creator on MNIST
2. Warmup Appraiser autoencoder on MNIST reconstruction, then freeze + save phi0
3. Joint training with creativity reward

Usage:
    python scripts/train.py --config configs/fast.yaml
    python scripts/train.py --config configs/default.yaml --training.device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from creator_appraiser.appraiser import Appraiser
from creator_appraiser.creator import DDPMCreator, UNet
from creator_appraiser.meta_learner import MetaLearner, MetaTrainConfig


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def override_config(config: dict, overrides: list[str]) -> dict:
    """Apply --key value or --key=value overrides with dot-notation."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        if "=" in key:
            key, value = key.split("=", 1)
        else:
            i += 1
            if i >= len(overrides):
                break
            value = overrides[i]

        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = _cast_value(value)
        i += 1

    return config


def _cast_value(value: str):
    """Cast string to appropriate type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() == "null":
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return [int(x.strip()) for x in value[1:-1].split(",")]
        except ValueError:
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def build_models(config: dict) -> tuple[DDPMCreator, Appraiser]:
    """Instantiate DDPMCreator and Appraiser from config."""
    creator_cfg = config["creator"]
    appraiser_cfg = config["appraiser"]
    episode_cfg = config["episode"]

    # Build UNet denoiser
    unet = UNet(
        img_channels=creator_cfg["img_channels"],
        base_channels=creator_cfg["base_channels"],
        channel_mults=tuple(creator_cfg["channel_mults"]),
        n_classes=creator_cfg["n_classes"],
        time_emb_dim=creator_cfg["time_emb_dim"],
    )

    # Wrap in DDPMCreator
    creator = DDPMCreator(
        unet=unet,
        timesteps=creator_cfg["timesteps"],
        schedule=creator_cfg["schedule"],
        n_way=episode_cfg["n_way"],
        k_shot=episode_cfg["k_shot"],
    )

    # Build Appraiser autoencoder
    appraiser = Appraiser(
        img_channels=creator_cfg["img_channels"],
        hidden_dim=appraiser_cfg["hidden_dim"],
        latent_dim=appraiser_cfg["latent_dim"],
        n_classes=creator_cfg["n_classes"],
        img_size=creator_cfg["img_size"],
    )

    return creator, appraiser


def build_meta_config(config: dict) -> MetaTrainConfig:
    """Build MetaTrainConfig from config dictionary."""
    meta_cfg = config["meta"]
    training_cfg = config["training"]
    creator_cfg = config["creator"]
    appraiser_cfg = config["appraiser"]
    data_cfg = config["data"]
    episode_cfg = config["episode"]
    reward_cfg = config["reward"]

    return MetaTrainConfig(
        outer_lr_creator=meta_cfg["outer_lr_creator"],
        outer_lr_appraiser=meta_cfg["outer_lr_appraiser"],
        inner_lr=appraiser_cfg["inner_lr"],
        inner_steps=appraiser_cfg["inner_steps"],
        n_way=episode_cfg["n_way"],
        k_shot=episode_cfg["k_shot"],
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
        guidance_scale=creator_cfg.get("guidance_scale", 1.0),
        timesteps=creator_cfg["timesteps"],
        n_perturbations=meta_cfg.get("n_perturbations", 8),
        perturbation_std=meta_cfg.get("perturbation_std", 0.05),
        device=training_cfg["device"],
        log_interval=training_cfg["log_interval"],
        save_interval=training_cfg["save_interval"],
        output_dir=training_cfg["output_dir"],
    )


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Creator-Appraiser")
    parser.add_argument("--config", type=str, default="configs/fast.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--extend-pretrain", type=int, default=0,
                        help="Additional DDPM pretrain epochs on a fresh MNIST subset")

    args, unknown = parser.parse_known_args()

    # Load config
    config = load_config(args.config)
    if unknown:
        config = override_config(config, unknown)

    # Seed
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Print summary
    print("=" * 60)
    print("Creator-Appraiser Training (DDPM + Autoencoder)")
    print("=" * 60)
    print(f"Device: {config['training']['device']}")
    print(f"Creator: DDPM UNet (ch={config['creator']['base_channels']}, "
          f"mults={config['creator']['channel_mults']}, T={config['creator']['timesteps']})")
    print(f"Appraiser: Autoencoder (hidden={config['appraiser']['hidden_dim']}, "
          f"latent={config['appraiser']['latent_dim']})")
    print(f"Inner loop: {config['appraiser']['inner_steps']} steps @ lr={config['appraiser']['inner_lr']}")
    print(f"Reward: mu={config['reward']['mu']}, tau={config['reward']['tau']}")
    print(f"Training: pretrain={config['training']['creator_pretrain_epochs']}ep, "
          f"warmup={config['training']['appraiser_warmup_epochs']}ep, "
          f"joint={config['meta']['max_episodes']}ep")
    print("=" * 60)

    # Build models
    creator, appraiser = build_models(config)

    creator_params = sum(p.numel() for p in creator.parameters())
    appraiser_params = sum(p.numel() for p in appraiser.parameters())
    print(f"Creator parameters: {creator_params:,}")
    print(f"Appraiser parameters: {appraiser_params:,}")
    print(f"Total parameters: {creator_params + appraiser_params:,}")
    print()

    # Build meta config
    meta_config = build_meta_config(config)

    # Create output directory
    os.makedirs(meta_config.output_dir, exist_ok=True)
    with open(os.path.join(meta_config.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Build meta-learner
    meta_learner = MetaLearner(creator=creator, appraiser=appraiser, config=meta_config)

    # Resume or warmup
    if args.resume:
        meta_learner.load_checkpoint(args.resume)
        # Re-freeze appraiser after loading
        meta_learner.appraiser.freeze_except_first_decoder_layer()
    else:
        # Check for cached warmup
        creator_cache = os.path.join(meta_config.output_dir, "creator_warmup_checkpoint.pt")
        appraiser_cache = os.path.join(meta_config.output_dir, "appraiser_warmup_checkpoint.pt")

        # Phase 1: Creator pre-training
        if os.path.exists(creator_cache):
            print(f"Loading cached Creator from {creator_cache}")
            ckpt = torch.load(creator_cache, map_location=meta_config.device, weights_only=False)
            creator.load_state_dict(ckpt["creator_state_dict"])
            meta_learner.creator_optimizer = torch.optim.Adam(
                creator.parameters(), lr=meta_config.outer_lr_creator
            )
        else:
            meta_learner.pretrain_creator()
            torch.save(
                {"creator_state_dict": creator.state_dict()},
                creator_cache,
            )
            print(f"Cached Creator to {creator_cache}")

        # Extend pre-training on a fresh subset if requested
        if args.extend_pretrain > 0:
            print(f"\nExtending Creator pre-training for {args.extend_pretrain} epochs "
                  f"on a fresh MNIST subset...")
            meta_learner.pretrain_creator(n_epochs=args.extend_pretrain)
            # Update the cache with improved weights
            torch.save(
                {"creator_state_dict": creator.state_dict()},
                creator_cache,
            )
            print(f"Updated cached Creator at {creator_cache}")

        print()

        # Phase 2: Appraiser warmup
        if os.path.exists(appraiser_cache):
            print(f"Loading cached Appraiser from {appraiser_cache}")
            ckpt = torch.load(appraiser_cache, map_location=meta_config.device, weights_only=False)
            appraiser.load_state_dict(ckpt["appraiser_state_dict"])
            appraiser._phi0 = {k: v.to(meta_config.device) for k, v in ckpt["phi0"].items()}
            appraiser.freeze_except_first_decoder_layer()
            # Rebuild appraiser optimizer for trainable params
            trainable = [p for p in appraiser.parameters() if p.requires_grad]
            meta_learner.appraiser_optimizer = torch.optim.Adam(
                trainable, lr=meta_config.inner_lr
            )
        else:
            meta_learner.warmup_appraiser()
            torch.save(
                {
                    "appraiser_state_dict": appraiser.state_dict(),
                    "phi0": appraiser._phi0,
                },
                appraiser_cache,
            )
            print(f"Cached Appraiser to {appraiser_cache}")
        print()

    # Phase 3: Generate creative images (save each one immediately)
    n_images = meta_config.max_episodes
    K = config["meta"].get("K", 50)

    print(f"Phase 3: Generating {n_images} creative images (K={K} steps each)...")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    vis_dir = os.path.join(meta_config.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    results = []
    for i in range(n_images):
        class_label = None  # random
        print(f"\n  Generating image {i+1}/{n_images}...")

        result = meta_learner.generate_creative_image(
            class_label=class_label, K=K, verbose=True,
        )
        results.append(result)

        final = result["history"][-1]
        init = result["history"][0]
        print(f"    Done: class={result['class_label']}, "
              f"R: {init['reward']:.6f} → {final['reward']:.6f}, "
              f"gap: {init['gap']:.4f} → {final['gap']:.4f}")

        # Save individual image immediately
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # Initial
        img = result["initial_image"].squeeze().numpy()
        img = (img + 1) / 2
        axes[0].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Initial (cls {result['class_label']})", fontsize=9)
        axes[0].axis("off")

        # Final
        img = result["image"].squeeze().numpy()
        img = (img + 1) / 2
        axes[1].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Final (R={final['reward']:.4f})", fontsize=9)
        axes[1].axis("off")

        # Reward curve
        rewards = [h["reward"] for h in result["history"]]
        axes[2].plot(rewards, "b-", linewidth=1)
        axes[2].set_xlabel("Step", fontsize=8)
        axes[2].set_ylabel("Reward", fontsize=8)
        axes[2].set_title("Reward curve", fontsize=9)

        plt.tight_layout()
        img_path = os.path.join(vis_dir, f"creative_{i+1:02d}_cls{result['class_label']}.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {img_path}")

    # Save combined grid
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for i, result in enumerate(results):
        final = result["history"][-1]
        init = result["history"][0]
        print(f"  Image {i+1} (class {result['class_label']}): "
              f"R: {init['reward']:.6f} → {final['reward']:.6f}, "
              f"gap: {init['gap']:.4f} → {final['gap']:.4f}")

    if len(results) > 1:
        n = len(results)
        fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
        if n == 1:
            axes = axes[:, np.newaxis]

        for i, result in enumerate(results):
            img = result["initial_image"].squeeze().numpy()
            img = (img + 1) / 2
            axes[0, i].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title(f"Init (cls {result['class_label']})", fontsize=8)
            axes[0, i].axis("off")

            img = result["image"].squeeze().numpy()
            img = (img + 1) / 2
            axes[1, i].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title(f"Final (R={result['history'][-1]['reward']:.4f})", fontsize=8)
            axes[1, i].axis("off")

            rewards = [h["reward"] for h in result["history"]]
            axes[2, i].plot(rewards, "b-", linewidth=1)
            axes[2, i].set_xlabel("Step", fontsize=7)
            axes[2, i].set_ylabel("R", fontsize=7)
            axes[2, i].tick_params(labelsize=6)

        plt.suptitle("Creator-Appraiser: All Creative Images", fontsize=11)
        plt.tight_layout()
        grid_path = os.path.join(vis_dir, "creative_all.png")
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nGrid saved to {grid_path}")

    meta_learner.save_checkpoint(n_images)
    print("\nDone.")


if __name__ == "__main__":
    main()
