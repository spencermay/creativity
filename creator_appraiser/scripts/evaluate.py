"""Evaluation script for trained Creator-Appraiser models.

Loads a trained checkpoint and runs comprehensive creativity evaluation,
including metric computation, visualization of generated concepts, and
comparison-ready output.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoint_ep60000.pt \
        --config configs/default.yaml --n_episodes 500

    python scripts/evaluate.py --checkpoint outputs/checkpoint_ep60000.pt \
        --config configs/default.yaml --visualize --save_images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from creator_appraiser.appraiser import Appraiser
from creator_appraiser.creator import ConditionalGenerator, Creator
from creator_appraiser.metrics import (
    CreativityMetrics,
    compute_creativity_score,
    compute_diversity_score,
    compute_inter_class_diversity,
    compute_intra_class_diversity,
    compute_memorability_score,
    compute_novelty_score,
    evaluate_creator,
)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_models(config: dict) -> tuple[Creator, Appraiser]:
    """Instantiate Creator and Appraiser from config."""
    creator_cfg = config["creator"]
    appraiser_cfg = config["appraiser"]
    episode_cfg = config["episode"]

    generator = ConditionalGenerator(
        latent_dim=creator_cfg["latent_dim"],
        n_classes=creator_cfg["n_classes"],
        img_channels=creator_cfg["img_channels"],
        img_size=creator_cfg["img_size"],
        hidden_dim=creator_cfg["hidden_dim"],
    )

    creator = Creator(
        generator=generator,
        n_way=episode_cfg["n_way"],
        k_shot=episode_cfg["k_shot"],
        q_query=episode_cfg["q_query"],
    )

    appraiser = Appraiser(
        img_channels=creator_cfg["img_channels"],
        hidden_dim=appraiser_cfg["hidden_dim"],
        img_size=creator_cfg["img_size"],
        n_way=episode_cfg["n_way"],
        inner_lr=appraiser_cfg["inner_lr"],
        inner_steps=appraiser_cfg["inner_steps"],
    )

    return creator, appraiser


def load_checkpoint(
    creator: Creator, appraiser: Appraiser, checkpoint_path: str, device: torch.device
) -> int:
    """Load model weights from checkpoint.

    Args:
        creator: Creator model to load weights into.
        appraiser: Appraiser model to load weights into.
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to map tensors to.

    Returns:
        Episode number from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    creator.load_state_dict(checkpoint["creator_state_dict"])
    appraiser.load_state_dict(checkpoint["appraiser_state_dict"])
    episode = checkpoint.get("episode", 0)
    print(f"Loaded checkpoint from episode {episode}")
    return episode


@torch.no_grad()
def detailed_evaluation(
    creator: Creator,
    appraiser: Appraiser,
    n_episodes: int,
    device: torch.device,
) -> dict:
    """Run detailed evaluation with per-episode breakdowns.

    Args:
        creator: Trained Creator model.
        appraiser: Trained Appraiser model.
        n_episodes: Number of evaluation episodes.
        device: Torch device.

    Returns:
        Dictionary of detailed results including per-episode metrics.
    """
    creator.eval()
    appraiser.eval()

    per_episode = []

    for i in range(n_episodes):
        episode = creator.generate_episode(device=device)
        support_images = episode["support_images"]
        support_labels = episode["support_labels"]
        query_images = episode["query_images"]
        query_labels = episode["query_labels"]

        results = appraiser.evaluate_episode(
            support_images, support_labels, query_images, query_labels
        )

        pre_acc = results["pre_adaptation_acc"].item()
        post_acc = results["post_adaptation_acc"].item()

        episode_metrics = {
            "episode_idx": i,
            "pre_adaptation_acc": pre_acc,
            "post_adaptation_acc": post_acc,
            "creativity_score": compute_creativity_score(pre_acc, post_acc),
            "novelty_score": compute_novelty_score(results["pre_adaptation_logits"]),
            "memorability_score": compute_memorability_score(
                results["post_adaptation_logits"], query_labels
            ),
            "diversity": compute_diversity_score(query_images),
            "intra_class_diversity": compute_intra_class_diversity(
                query_images, query_labels
            ),
            "inter_class_diversity": compute_inter_class_diversity(
                query_images, query_labels
            ),
        }
        per_episode.append(episode_metrics)

    # Aggregate
    keys = [k for k in per_episode[0] if k != "episode_idx"]
    aggregated = {}
    for key in keys:
        values = [ep[key] for ep in per_episode]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_min"] = float(np.min(values))
        aggregated[f"{key}_max"] = float(np.max(values))

    return {
        "aggregated": aggregated,
        "per_episode": per_episode,
        "n_episodes": n_episodes,
    }


@torch.no_grad()
def generate_samples(
    creator: Creator,
    n_episodes: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Generate sample episodes for visualization.

    Args:
        creator: Trained Creator model.
        n_episodes: Number of episodes to generate.
        device: Torch device.

    Returns:
        List of episode dictionaries with generated images.
    """
    creator.eval()
    episodes = []
    for _ in range(n_episodes):
        episode = creator.generate_episode(device=device)
        # Move to CPU for saving
        episodes.append({k: v.cpu() for k, v in episode.items()})
    return episodes


def save_visualizations(
    episodes: list[dict[str, torch.Tensor]],
    output_dir: str,
    n_episodes: int = 5,
) -> None:
    """Save visualization grids of generated concepts.

    Args:
        episodes: List of generated episodes.
        output_dir: Directory to save images.
        n_episodes: Number of episodes to visualize.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping visualizations.")
        return

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for ep_idx in range(min(n_episodes, len(episodes))):
        episode = episodes[ep_idx]
        query_images = episode["query_images"]
        query_labels = episode["query_labels"]

        n_way = query_labels.unique().size(0)
        n_per_class = (query_labels == 0).sum().item()

        # Create grid: rows=classes, cols=examples
        fig, axes = plt.subplots(
            n_way, min(n_per_class, 8), figsize=(min(n_per_class, 8) * 2, n_way * 2)
        )

        if n_way == 1:
            axes = axes.unsqueeze(0) if hasattr(axes, "unsqueeze") else [axes]

        for class_idx in range(n_way):
            class_mask = query_labels == class_idx
            class_images = query_images[class_mask]

            for img_idx in range(min(n_per_class, 8)):
                ax = axes[class_idx][img_idx] if n_way > 1 else axes[img_idx]
                img = class_images[img_idx]

                # Denormalize for display (approximate)
                img = img * 0.5 + 0.5
                img = img.clamp(0, 1)

                if img.shape[0] == 1:
                    ax.imshow(img.squeeze(0).numpy(), cmap="gray")
                else:
                    ax.imshow(img.permute(1, 2, 0).numpy())

                ax.axis("off")
                if img_idx == 0:
                    ax.set_ylabel(f"Class {class_idx}", fontsize=10)

        plt.suptitle(f"Generated Episode {ep_idx + 1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"episode_{ep_idx + 1}.png"), dpi=150)
        plt.close()

    print(f"Saved {min(n_episodes, len(episodes))} visualizations to {vis_dir}/")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Creator-Appraiser model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as checkpoint dir)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save visualization grids",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save raw generated images as tensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: from config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine device
    device_str = args.device or config["training"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    print("=" * 60)
    print("Creator-Appraiser Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output: {eval_dir}")
    print("=" * 60)

    # Build and load models
    creator, appraiser = build_models(config)
    creator = creator.to(device)
    appraiser = appraiser.to(device)
    episode_num = load_checkpoint(creator, appraiser, args.checkpoint, device)

    # Run detailed evaluation
    print("\nRunning detailed evaluation...")
    results = detailed_evaluation(creator, appraiser, args.n_episodes, device)

    # Print summary
    agg = results["aggregated"]
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Creativity Score:     {agg['creativity_score_mean']:.4f} +/- {agg['creativity_score_std']:.4f}")
    print(f"  Novelty Score:        {agg['novelty_score_mean']:.4f} +/- {agg['novelty_score_std']:.4f}")
    print(f"  Memorability Score:   {agg['memorability_score_mean']:.4f} +/- {agg['memorability_score_std']:.4f}")
    print(f"  Pre-Adapt Accuracy:   {agg['pre_adaptation_acc_mean']:.4f} +/- {agg['pre_adaptation_acc_std']:.4f}")
    print(f"  Post-Adapt Accuracy:  {agg['post_adaptation_acc_mean']:.4f} +/- {agg['post_adaptation_acc_std']:.4f}")
    print(f"  Diversity:            {agg['diversity_mean']:.4f} +/- {agg['diversity_std']:.4f}")
    print(f"  Intra-Class Div:      {agg['intra_class_diversity_mean']:.4f} +/- {agg['intra_class_diversity_std']:.4f}")
    print(f"  Inter-Class Div:      {agg['inter_class_diversity_mean']:.4f} +/- {agg['inter_class_diversity_std']:.4f}")
    print("=" * 60)

    # Save results as JSON
    results_path = os.path.join(eval_dir, "results.json")
    # Convert numpy types for JSON serialization
    json_results = {
        "checkpoint": args.checkpoint,
        "episode": episode_num,
        "n_episodes": args.n_episodes,
        "aggregated": agg,
    }
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        episodes = generate_samples(creator, n_episodes=10, device=device)
        save_visualizations(episodes, eval_dir, n_episodes=5)

    # Save raw generated images
    if args.save_images:
        print("\nSaving generated image tensors...")
        episodes = generate_samples(creator, n_episodes=20, device=device)
        images_path = os.path.join(eval_dir, "generated_episodes.pt")
        torch.save(episodes, images_path)
        print(f"Saved {len(episodes)} episodes to {images_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
