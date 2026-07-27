"""Sample baseline images from the pre-trained DDPM (no creative guidance).

Generates a grid of images directly from the DDPM after warmup to visualize
what the Creator produces without any noise optimization. This serves as the
baseline to compare creative images against.

Usage:
    python scripts/sample_baseline.py --config configs/fast.yaml
    python scripts/sample_baseline.py --config configs/fast.yaml --n_per_class 5 --guidance_scale 3.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from creator_appraiser.creator import DDPMCreator, UNet
from creator_appraiser.appraiser import Appraiser


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_creator(config: dict) -> DDPMCreator:
    """Build Creator from config."""
    creator_cfg = config["creator"]

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

    return creator


def main():
    parser = argparse.ArgumentParser(description="Sample baseline DDPM images (no creativity)")
    parser.add_argument("--config", type=str, default="configs/fast.yaml")
    parser.add_argument("--n_per_class", type=int, default=4, help="Images per digit class")
    parser.add_argument("--classes", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated digit classes to sample")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Override guidance scale (default: from config)")
    parser.add_argument("--skip_steps", type=int, default=None,
                        help="Override sampling skip steps")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)
    creator_cfg = config["creator"]

    # Load creator
    creator = build_creator(config)
    device = torch.device(config["training"]["device"])

    # Load warmup checkpoint
    output_dir = config["training"]["output_dir"]
    creator_cache = os.path.join(output_dir, "creator_warmup_checkpoint.pt")

    if os.path.exists(creator_cache):
        checkpoint = torch.load(creator_cache, map_location=device, weights_only=False)
        creator.load_state_dict(checkpoint["creator_state_dict"])
        print(f"Loaded Creator from {creator_cache}")
    else:
        print(f"WARNING: No cached Creator at {creator_cache}")
        print("Run training first. Using random weights (images will be noise).")

    creator = creator.to(device)
    creator.eval()

    # Parameters
    classes = [int(x.strip()) for x in args.classes.split(",")]
    n_per_class = args.n_per_class
    guidance_scale = args.guidance_scale or creator_cfg.get("guidance_scale", 1.0)
    skip_steps = args.skip_steps or creator_cfg["sample_skip_steps"]

    print(f"Sampling {n_per_class} images per class, classes={classes}")
    print(f"Guidance scale: {guidance_scale}, skip_steps: {skip_steps}")

    # Generate images
    all_images = []
    for cls in classes:
        labels = torch.full((n_per_class,), cls, device=device, dtype=torch.long)
        x_T = torch.randn(n_per_class, creator_cfg["img_channels"],
                          creator_cfg["img_size"], creator_cfg["img_size"], device=device)

        with torch.no_grad():
            images = creator.sample_from_noise(x_T, labels,
                                               skip_steps=skip_steps,
                                               guidance_scale=guidance_scale)
        all_images.append(images.cpu())

    # Visualize grid: rows=classes, cols=samples
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(classes)
    n_cols = n_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (cls, images) in enumerate(zip(classes, all_images)):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            img = images[col_idx].squeeze().numpy()
            img = (img + 1) / 2  # [-1,1] → [0,1]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(f"Digit {cls}", fontsize=10)

    plt.suptitle(f"Baseline DDPM Samples (guidance={guidance_scale})", fontsize=12)
    plt.tight_layout()

    # Save
    out_path = args.output or os.path.join(output_dir, "visualizations", "baseline_samples.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
