"""
Embedding-based image novelty evaluation.

Compares generated images against WikiArt reference images using deep
feature embeddings (ResNet152 or CLIP). The paper uses ResNet152 features
from a style classifier trained on WikiArt.

Metrics:
- Average maximum cosine similarity to WikiArt (lower = more novel)
- Pairwise win counts: for each pair (alien, baseline), count how often
  the alien image is further from WikiArt than the baseline image

Usage:
    # First, build the reference embedding index from WikiArt:
    python evaluation/image_novelty_embed.py build-index \
        --wikiart-dir ./data/raw/wikiart_images/ \
        --output ./data/processed/wikiart_embeddings.npy \
        --model resnet152

    # Then evaluate generated images against the index:
    python evaluation/image_novelty_embed.py evaluate \
        --generated-dir ./outputs/images/romanticism_mountain/ \
        --reference-embeddings ./data/processed/wikiart_embeddings.npy \
        --output ./outputs/reports/image_novelty_embed.json

    # Or compare alien vs baseline images:
    python evaluation/image_novelty_embed.py compare \
        --alien-dir ./outputs/images/alien/ \
        --baseline-dir ./outputs/images/baseline/ \
        --reference-embeddings ./data/processed/wikiart_embeddings.npy \
        --output ./outputs/reports/image_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_image_paths(directory: Path) -> list[Path]:
    """Recursively find all image files in a directory."""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(directory.rglob(f"*{ext}"))
    paths.sort()
    return paths


def build_resnet_extractor(device: str) -> tuple[nn.Module, transforms.Compose]:
    """
    Build a ResNet152 feature extractor (penultimate layer).
    Returns the model and preprocessing transform.
    """
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    # Remove the final classification layer to get 2048-dim features
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, preprocess


@torch.no_grad()
def extract_embeddings(
    image_paths: list[Path],
    model: nn.Module,
    preprocess: transforms.Compose,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract normalized embeddings for a list of images."""
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
                # Use a zero tensor as placeholder
                batch_tensors.append(torch.zeros(3, 224, 224))

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        features = model(batch)
        features = features.squeeze(-1).squeeze(-1)  # (B, 2048)

        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        all_embeddings.append(features.cpu().numpy())

    if not all_embeddings:
        return np.array([])

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def compute_max_cosine_similarities(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    batch_size: int = 100,
) -> np.ndarray:
    """
    For each query embedding, compute the maximum cosine similarity
    to any reference embedding.

    Both inputs should be L2-normalized, so cosine sim = dot product.
    """
    n_queries = len(query_embeddings)
    max_sims = np.zeros(n_queries, dtype=np.float32)

    for i in tqdm(range(0, n_queries, batch_size), desc="Computing similarities"):
        batch = query_embeddings[i:i + batch_size]
        # (batch, dim) @ (dim, n_ref) -> (batch, n_ref)
        sims = batch @ reference_embeddings.T
        max_sims[i:i + batch_size] = sims.max(axis=1)

    return max_sims


def build_index(args):
    """Build reference embedding index from WikiArt images."""
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[INFO] Building embedding index from {args.wikiart_dir}")
    print(f"[INFO] Device: {device}")

    image_paths = get_image_paths(args.wikiart_dir)
    print(f"[INFO] Found {len(image_paths)} images")

    if not image_paths:
        print("[ERROR] No images found")
        return

    model, preprocess = build_resnet_extractor(device)
    batch_size = args.batch_size or config["evaluation"]["embedding_batch_size"]

    embeddings = extract_embeddings(image_paths, model, preprocess, device, batch_size)
    print(f"[INFO] Extracted embeddings: shape={embeddings.shape}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)

    # Also save the path list for reference
    paths_file = args.output.with_suffix(".paths.json")
    with open(paths_file, "w", encoding="utf-8") as f:
        json.dump([str(p) for p in image_paths], f, indent=2)

    print(f"[INFO] Saved embeddings to {args.output}")
    print(f"[INFO] Saved path index to {paths_file}")


def evaluate(args):
    """Evaluate generated images against reference embeddings."""
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[INFO] Evaluating images in {args.generated_dir}")

    # Load reference embeddings
    reference_embeddings = np.load(args.reference_embeddings)
    print(f"[INFO] Reference embeddings: shape={reference_embeddings.shape}")

    # Get generated image paths
    image_paths = get_image_paths(args.generated_dir)
    print(f"[INFO] Found {len(image_paths)} generated images")

    if not image_paths:
        print("[ERROR] No generated images found")
        return

    # Extract embeddings for generated images
    model, preprocess = build_resnet_extractor(device)
    batch_size = args.batch_size or config["evaluation"]["embedding_batch_size"]
    generated_embeddings = extract_embeddings(image_paths, model, preprocess, device, batch_size)

    # Compute max cosine similarities
    max_sims = compute_max_cosine_similarities(generated_embeddings, reference_embeddings)

    # Results
    per_image = []
    for path, sim in zip(image_paths, max_sims):
        per_image.append({
            "image": str(path.name),
            "max_cosine_similarity": float(sim),
        })

    summary = {
        "num_generated": len(image_paths),
        "num_reference": len(reference_embeddings),
        "avg_max_cosine_sim": float(np.mean(max_sims)),
        "std_max_cosine_sim": float(np.std(max_sims)),
        "min_max_cosine_sim": float(np.min(max_sims)),
        "max_max_cosine_sim": float(np.max(max_sims)),
        "median_max_cosine_sim": float(np.median(max_sims)),
    }

    output_data = {
        "generated_dir": str(args.generated_dir),
        "reference_file": str(args.reference_embeddings),
        "summary": summary,
        "per_image": per_image,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"IMAGE EMBEDDING NOVELTY EVALUATION")
    print(f"{'='*60}")
    print(f"Generated images: {summary['num_generated']}")
    print(f"Reference images: {summary['num_reference']}")
    print(f"\nMax cosine similarity to WikiArt (lower = more novel):")
    print(f"  Mean:   {summary['avg_max_cosine_sim']:.4f} ± {summary['std_max_cosine_sim']:.4f}")
    print(f"  Median: {summary['median_max_cosine_sim']:.4f}")
    print(f"  Range:  [{summary['min_max_cosine_sim']:.4f}, {summary['max_max_cosine_sim']:.4f}]")
    print(f"{'='*60}")
    print(f"\n[INFO] Report saved to {args.output}")


def compare(args):
    """Compare alien vs baseline images in terms of novelty."""
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load reference
    reference_embeddings = np.load(args.reference_embeddings)
    print(f"[INFO] Reference embeddings: shape={reference_embeddings.shape}")

    model, preprocess = build_resnet_extractor(device)
    batch_size = args.batch_size or config["evaluation"]["embedding_batch_size"]

    # Extract alien embeddings
    alien_paths = get_image_paths(args.alien_dir)
    print(f"[INFO] Alien images: {len(alien_paths)}")
    alien_embeddings = extract_embeddings(alien_paths, model, preprocess, device, batch_size)
    alien_sims = compute_max_cosine_similarities(alien_embeddings, reference_embeddings)

    # Extract baseline embeddings
    baseline_paths = get_image_paths(args.baseline_dir)
    print(f"[INFO] Baseline images: {len(baseline_paths)}")
    baseline_embeddings = extract_embeddings(baseline_paths, model, preprocess, device, batch_size)
    baseline_sims = compute_max_cosine_similarities(baseline_embeddings, reference_embeddings)

    # Pairwise comparison: for each pair, does alien have LOWER similarity (= more novel)?
    n_pairs = min(len(alien_sims), len(baseline_sims))
    alien_wins = 0
    baseline_wins = 0
    ties = 0

    for i in range(n_pairs):
        if alien_sims[i] < baseline_sims[i]:
            alien_wins += 1  # Lower similarity = more novel
        elif alien_sims[i] > baseline_sims[i]:
            baseline_wins += 1
        else:
            ties += 1

    result = {
        "alien_dir": str(args.alien_dir),
        "baseline_dir": str(args.baseline_dir),
        "num_alien": len(alien_paths),
        "num_baseline": len(baseline_paths),
        "num_pairs_compared": n_pairs,
        "alien_avg_max_sim": float(np.mean(alien_sims)),
        "baseline_avg_max_sim": float(np.mean(baseline_sims)),
        "alien_wins": alien_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "alien_win_rate": alien_wins / max(n_pairs, 1),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"ALIEN vs BASELINE IMAGE NOVELTY COMPARISON")
    print(f"{'='*60}")
    print(f"Alien avg max sim:    {result['alien_avg_max_sim']:.4f}")
    print(f"Baseline avg max sim: {result['baseline_avg_max_sim']:.4f}")
    print(f"\nPairwise comparison ({n_pairs} pairs):")
    print(f"  Alien wins (more novel):    {alien_wins} ({alien_wins/max(n_pairs,1)*100:.1f}%)")
    print(f"  Baseline wins (more novel): {baseline_wins} ({baseline_wins/max(n_pairs,1)*100:.1f}%)")
    print(f"  Ties:                       {ties}")
    print(f"{'='*60}")
    print(f"\n[INFO] Report saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Embedding-based image novelty evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # build-index
    p_build = subparsers.add_parser("build-index", help="Build reference embedding index")
    p_build.add_argument("--wikiart-dir", type=Path, required=True)
    p_build.add_argument("--output", type=Path, required=True)
    p_build.add_argument("--batch-size", type=int, default=None)
    p_build.add_argument("--config", type=Path, default=None)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate generated images")
    p_eval.add_argument("--generated-dir", type=Path, required=True)
    p_eval.add_argument("--reference-embeddings", type=Path, required=True)
    p_eval.add_argument("--output", type=Path, required=True)
    p_eval.add_argument("--batch-size", type=int, default=None)
    p_eval.add_argument("--config", type=Path, default=None)

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare alien vs baseline images")
    p_comp.add_argument("--alien-dir", type=Path, required=True)
    p_comp.add_argument("--baseline-dir", type=Path, required=True)
    p_comp.add_argument("--reference-embeddings", type=Path, required=True)
    p_comp.add_argument("--output", type=Path, required=True)
    p_comp.add_argument("--batch-size", type=int, default=None)
    p_comp.add_argument("--config", type=Path, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "build-index":
        build_index(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare":
        compare(args)


if __name__ == "__main__":
    main()
