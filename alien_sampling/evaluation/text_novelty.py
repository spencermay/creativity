"""
Text-based novelty evaluation for generated concept sequences.

Computes two novelty metrics from the paper:

1. N_art (Artwork-level novelty):
   Given generated concept set S and all artwork concept sets A_i:
       N_art(S) = min_i |S \\ A_i|
   i.e., the minimum number of concepts in S that are NOT in any single artwork.
   Higher = more novel relative to individual artworks.

2. N_cog (Cognitive-availability novelty):
   Given generated concept set S and all artist concept pools B_j:
       N_cog(S) = min_j |S \\ B_j|
   i.e., the minimum number of concepts in S that are NOT in any single artist's
   known concept pool.
   Higher = more novel relative to what any artist has done.

Key insight from the paper:
- Generating artwork-level novelty (N_art > 0) is relatively easy
- Generating artist-level cognitive novelty (N_cog > 0) is much harder

Usage:
    python evaluation/text_novelty.py \
        --sequence-file ./outputs/sequences/romanticism_mountain.json \
        --artworks ./data/processed/artworks.parquet \
        --artists ./data/processed/artist_concepts.parquet \
        --output ./outputs/reports/romanticism_mountain_novelty.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


def load_artwork_concepts(artworks_path: Path) -> list[set[str]]:
    """
    Load artwork concept sets from parquet or JSONL.
    Returns a list of sets, one per artwork.
    """
    if artworks_path.suffix == ".parquet":
        df = pd.read_parquet(artworks_path)
    elif artworks_path.suffix == ".jsonl":
        df = pd.read_json(artworks_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {artworks_path.suffix}")

    concept_sets = []
    col = "all10_concepts" if "all10_concepts" in df.columns else "top9_concepts"

    for concepts in df[col]:
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        concept_sets.append(set(concepts))

    print(f"[INFO] Loaded {len(concept_sets)} artwork concept sets from {artworks_path}")
    return concept_sets


def load_artist_concepts(artists_path: Path) -> list[set[str]]:
    """
    Load artist concept pools from parquet or JSONL.
    Returns a list of sets, one per artist (the union of all their concepts).
    """
    if artists_path.suffix == ".parquet":
        df = pd.read_parquet(artists_path)
    elif artists_path.suffix == ".jsonl":
        df = pd.read_json(artists_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {artists_path.suffix}")

    concept_pools = []
    for concepts in df["concepts_union"]:
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        concept_pools.append(set(concepts))

    print(f"[INFO] Loaded {len(concept_pools)} artist concept pools from {artists_path}")
    return concept_pools


def compute_n_art(sequence: list[str], artwork_concepts: list[set[str]]) -> int:
    """
    Compute artwork-level novelty: N_art(S) = min_i |S \\ A_i|

    The minimum number of concepts in S not found in any single artwork.
    """
    s = set(sequence)
    if not artwork_concepts:
        return len(s)

    min_novel = len(s)
    for artwork_set in artwork_concepts:
        novel_count = len(s - artwork_set)
        min_novel = min(min_novel, novel_count)
        if min_novel == 0:
            break  # Can't do better than 0

    return min_novel


def compute_n_cog(sequence: list[str], artist_concepts: list[set[str]]) -> int:
    """
    Compute cognitive-availability novelty: N_cog(S) = min_j |S \\ B_j|

    The minimum number of concepts in S not found in any single artist's pool.
    """
    s = set(sequence)
    if not artist_concepts:
        return len(s)

    min_novel = len(s)
    for artist_pool in artist_concepts:
        novel_count = len(s - artist_pool)
        min_novel = min(min_novel, novel_count)
        if min_novel == 0:
            break

    return min_novel


def evaluate_sequences(
    sequences: list[list[str]],
    artwork_concepts: list[set[str]],
    artist_concepts: list[set[str]],
) -> list[dict]:
    """Evaluate all sequences and return per-sequence metrics."""
    results = []

    for seq in tqdm(sequences, desc="Computing text novelty"):
        n_art = compute_n_art(seq, artwork_concepts)
        n_cog = compute_n_cog(seq, artist_concepts)

        results.append({
            "sequence": seq,
            "sequence_str": " ".join(seq),
            "n_art": n_art,
            "n_cog": n_cog,
            "seq_len": len(seq),
        })

    return results


def compute_summary_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics over all evaluated sequences."""
    n_arts = [r["n_art"] for r in results]
    n_cogs = [r["n_cog"] for r in results]

    summary = {
        "count": len(results),
        "n_art": {
            "mean": float(np.mean(n_arts)),
            "std": float(np.std(n_arts)),
            "min": int(np.min(n_arts)),
            "max": int(np.max(n_arts)),
            "median": float(np.median(n_arts)),
            "pct_gt_zero": float(np.mean([x > 0 for x in n_arts])),
        },
        "n_cog": {
            "mean": float(np.mean(n_cogs)),
            "std": float(np.std(n_cogs)),
            "min": int(np.min(n_cogs)),
            "max": int(np.max(n_cogs)),
            "median": float(np.median(n_cogs)),
            "pct_gt_zero": float(np.mean([x > 0 for x in n_cogs])),
        },
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute text-based novelty metrics")
    parser.add_argument(
        "--sequence-file", type=Path, required=True,
        help="JSON file with sequences (from alien_sampling.py or generate_sequences.py)"
    )
    parser.add_argument("--artworks", type=Path, required=True, help="Artworks parquet/JSONL")
    parser.add_argument("--artists", type=Path, required=True, help="Artist concepts parquet/JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output report JSON")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    # Load reference data
    artwork_concepts = load_artwork_concepts(args.artworks)
    artist_concepts = load_artist_concepts(args.artists)

    # Load sequences to evaluate
    with open(args.sequence_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract sequences from various possible formats
    sequences = []
    if "selected" in data:
        for entry in data["selected"]:
            if isinstance(entry, dict):
                sequences.append(entry.get("sequence", []))
            elif isinstance(entry, list):
                sequences.append(entry)
    elif "all_ranked" in data:
        for entry in data["all_ranked"]:
            if isinstance(entry, dict):
                sequences.append(entry.get("sequence", []))
    elif "candidates" in data:
        sequences = data["candidates"]

    sequences = [s for s in sequences if s]  # Filter empties
    print(f"[INFO] Evaluating {len(sequences)} sequences")

    if not sequences:
        print("[ERROR] No sequences found to evaluate")
        return

    # Evaluate
    results = evaluate_sequences(sequences, artwork_concepts, artist_concepts)
    summary = compute_summary_stats(results)

    # Output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "source_file": str(args.sequence_file),
        "num_artworks": len(artwork_concepts),
        "num_artists": len(artist_concepts),
        "summary": summary,
        "per_sequence": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEXT NOVELTY EVALUATION")
    print(f"{'='*60}")
    print(f"Sequences evaluated: {summary['count']}")
    print(f"\nN_art (artwork-level novelty):")
    print(f"  Mean:   {summary['n_art']['mean']:.2f} ± {summary['n_art']['std']:.2f}")
    print(f"  Median: {summary['n_art']['median']:.1f}")
    print(f"  Range:  [{summary['n_art']['min']}, {summary['n_art']['max']}]")
    print(f"  % > 0:  {summary['n_art']['pct_gt_zero']*100:.1f}%")
    print(f"\nN_cog (cognitive-availability novelty):")
    print(f"  Mean:   {summary['n_cog']['mean']:.2f} ± {summary['n_cog']['std']:.2f}")
    print(f"  Median: {summary['n_cog']['median']:.1f}")
    print(f"  Range:  [{summary['n_cog']['min']}, {summary['n_cog']['max']}]")
    print(f"  % > 0:  {summary['n_cog']['pct_gt_zero']*100:.1f}%")
    print(f"{'='*60}")
    print(f"\n[INFO] Full report saved to {args.output}")


if __name__ == "__main__":
    main()
