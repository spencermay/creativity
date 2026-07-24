"""
Alien Sampling: rank candidate sequences by combining Art-model plausibility
with Cognitive Availability model unavailability.

The paper's core insight:
- Low Art perplexity = plausible as artwork (good)
- High Cog perplexity = cognitively unavailable to artists (novel)

The fused ranking uses inverse-rank aggregation:
    score(s) = (1 - beta) * rank_art(s) + beta * rank_cog_inv(s)

Where:
- rank_art(s) ranks by Art perplexity ascending (lower PPL = better rank)
- rank_cog_inv(s) ranks by Cog perplexity descending (higher PPL = better rank)
- beta controls the "alienness" trade-off

Higher beta emphasizes cognitive unavailability more strongly.

Usage:
    python inference/alien_sampling.py \
        --scored ./outputs/sequences/romanticism_mountain_scored.json \
        --beta 0.85 \
        --top-k 5 \
        --output ./outputs/sequences/romanticism_mountain_alien.json

    # Can also run end-to-end: generate + score + rank
    python inference/alien_sampling.py \
        --seed-sequence "romanticism mountain" \
        --vocab ./data/processed/vocab/vocab_words.json \
        --num-candidates 150 \
        --beta 0.85 \
        --top-k 5 \
        --output ./outputs/sequences/romanticism_mountain.json \
        --config ./configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


def compute_ranks(values: list[float], ascending: bool = True) -> list[int]:
    """
    Compute 1-based ranks for a list of values.
    Ties are broken by position (first occurrence gets better rank).

    ascending=True: smallest value gets rank 1
    ascending=False: largest value gets rank 1
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=(not ascending))
    ranks = [0] * len(values)
    for rank, (orig_idx, _) in enumerate(indexed, start=1):
        ranks[orig_idx] = rank
    return ranks


def alien_rank(
    scored_candidates: list[dict],
    beta: float,
) -> list[dict]:
    """
    Apply the Alien sampling ranking to scored candidates.

    Parameters
    ----------
    scored_candidates : list of dicts with 'art_perplexity' and 'cog_perplexity'
    beta : float in [0, 1], controls alienness weight

    Returns
    -------
    list of dicts sorted by fused score (lower = better), with rank info added
    """
    # Filter out candidates with missing scores
    valid = [
        s for s in scored_candidates
        if s.get("art_perplexity") is not None and s.get("cog_perplexity") is not None
    ]

    if not valid:
        print("[WARN] No valid scored candidates to rank")
        return []

    n = len(valid)
    art_ppls = [s["art_perplexity"] for s in valid]
    cog_ppls = [s["cog_perplexity"] for s in valid]

    # Art rank: ascending (lower PPL = rank 1 = most plausible)
    art_ranks = compute_ranks(art_ppls, ascending=True)

    # Cog rank: descending (higher PPL = rank 1 = most cognitively unavailable)
    cog_ranks = compute_ranks(cog_ppls, ascending=False)

    # Fused score: weighted combination of normalized ranks
    # Lower fused score = better candidate
    fused_scores = []
    for i in range(n):
        art_norm = art_ranks[i] / n
        cog_norm = cog_ranks[i] / n
        fused = (1 - beta) * art_norm + beta * cog_norm
        fused_scores.append(fused)

    # Attach scores to candidates
    for i, s in enumerate(valid):
        s["art_rank"] = art_ranks[i]
        s["cog_rank"] = cog_ranks[i]
        s["fused_score"] = fused_scores[i]
        s["fused_rank"] = None  # Will be assigned after sorting

    # Sort by fused score (lower = better)
    valid.sort(key=lambda s: s["fused_score"])

    # Assign final fused ranks
    for rank, s in enumerate(valid, start=1):
        s["fused_rank"] = rank

    return valid


def select_top_k(ranked: list[dict], top_k: int) -> list[dict]:
    """Select the top-k candidates from ranked list."""
    return ranked[:top_k]


def run_end_to_end(args, config):
    """Run the full pipeline: generate (NB sampling) -> score (NB) -> rank."""
    from inference.generate_sequences import generate_candidates, load_vocabulary
    from inference.naive_bayes_model import (
        NaiveBayesConceptModel,
        load_artwork_concept_sets,
        load_artist_concept_sets,
        score_all_candidates as nb_score_all,
    )

    gen_cfg = config["generation"]

    # Load vocabulary
    vocabulary_set = load_vocabulary(args.vocab)
    vocabulary_list = sorted(vocabulary_set)
    seed_concepts = args.seed_sequence.strip().lower().split()

    print(f"[INFO] End-to-end Alien sampling (Naive Bayes generation + scoring)")
    print(f"[INFO] Seed: {seed_concepts}")
    print(f"[INFO] Beta: {args.beta}")

    # Build co-occurrence models from data
    artworks_path = Path(args.artworks or config["paths"]["artworks_parquet"])
    artists_path = Path(args.artists or config["paths"]["artists_parquet"])

    artwork_sets = load_artwork_concept_sets(artworks_path)
    artist_sets = load_artist_concept_sets(artists_path)

    art_model = NaiveBayesConceptModel(smoothing=1.0)
    art_model.fit(artwork_sets)

    cog_model = NaiveBayesConceptModel(smoothing=1.0)
    cog_model.fit(artist_sets)

    # Generate candidates by sampling from Art model distribution
    temperature = args.temperature or gen_cfg["temperature"]
    num_candidates = args.num_candidates or gen_cfg["num_candidates"]

    candidates = generate_candidates(
        model=art_model,
        seed_concepts=seed_concepts,
        vocabulary=vocabulary_list,
        num_candidates=num_candidates,
        temperature=temperature,
        min_seq_len=gen_cfg["min_sequence_len"],
        max_seq_len=gen_cfg["max_sequence_len"],
    )

    if not candidates:
        print("[ERROR] No candidates generated")
        return

    # Score candidates with both Naive Bayes models
    scored = nb_score_all(candidates, art_model, cog_model)

    # Rank
    ranked = alien_rank(scored, beta=args.beta)
    top_k = args.top_k or config["alien_sampling"]["top_k_output"]
    selected = select_top_k(ranked, top_k)

    return {
        "seed_sequence": seed_concepts,
        "scoring_method": "naive_bayes",
        "generation_method": "naive_bayes_sampling",
        "temperature": temperature,
        "beta": args.beta,
        "num_candidates_generated": len(candidates),
        "num_valid_scored": len(ranked),
        "top_k": top_k,
        "selected": selected,
        "all_ranked": ranked,
    }


def run_from_scored(args, config):
    """Run ranking on pre-scored candidates."""
    with open(args.scored, "r", encoding="utf-8") as f:
        data = json.load(f)

    scored_candidates = data["scored_candidates"]
    print(f"[INFO] Loaded {len(scored_candidates)} scored candidates")

    beta = args.beta or config["alien_sampling"]["beta"]
    top_k = args.top_k or config["alien_sampling"]["top_k_output"]

    # Rank
    ranked = alien_rank(scored_candidates, beta=beta)
    selected = select_top_k(ranked, top_k)

    return {
        "seed_sequence": data.get("seed_sequence", []),
        "art_model": data.get("art_model", "unknown"),
        "cog_model": data.get("cog_model", "unknown"),
        "beta": beta,
        "num_valid_scored": len(ranked),
        "top_k": top_k,
        "selected": selected,
        "all_ranked": ranked,
    }


def main():
    parser = argparse.ArgumentParser(description="Alien Sampling: rank by plausibility + unavailability")

    # Pre-scored input mode
    parser.add_argument("--scored", type=Path, default=None, help="Pre-scored candidates JSON")

    # End-to-end mode
    parser.add_argument("--seed-sequence", type=str, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--art-model", type=str, default=None)
    parser.add_argument("--cog-model", type=str, default=None)
    parser.add_argument("--artworks", type=Path, default=None, help="Artworks parquet (for Naive Bayes)")
    parser.add_argument("--artists", type=Path, default=None, help="Artists parquet (for Naive Bayes)")

    # Ranking parameters
    parser.add_argument("--beta", type=float, default=None, help="Alienness weight [0,1]")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top sequences to select")

    # Output
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)

    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    # Default beta from config
    if args.beta is None:
        args.beta = config["alien_sampling"]["beta"]

    if args.scored is not None:
        # Rank pre-scored candidates
        result = run_from_scored(args, config)
    elif args.seed_sequence is not None:
        # End-to-end: generate + score + rank
        result = run_end_to_end(args, config)
    else:
        parser.error("Provide either --scored (pre-scored JSON) or --seed-sequence (end-to-end)")
        return

    if result is None:
        print("[ERROR] No results produced")
        return

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved Alien sampling results to {args.output}")
    print(f"[INFO] Beta = {result['beta']}")
    print(f"[INFO] Selected top-{result['top_k']} from {result['num_valid_scored']} candidates")

    # Print selected sequences
    print("\n[INFO] Top selected sequences:")
    for i, s in enumerate(result["selected"], start=1):
        seq_str = s.get("sequence_str", " ".join(s.get("sequence", [])))
        print(
            f"  #{i}: {seq_str}\n"
            f"       art_ppl={s['art_perplexity']:.2f} "
            f"cog_ppl={s['cog_perplexity']:.2f} "
            f"fused={s['fused_score']:.4f}"
        )

    # Also produce a baseline comparison: top-k by Art model alone
    if result.get("all_ranked"):
        baseline = sorted(
            result["all_ranked"],
            key=lambda s: s.get("art_perplexity", float("inf"))
        )[:result["top_k"]]
        print("\n[INFO] Baseline (Art-only) top sequences for comparison:")
        for i, s in enumerate(baseline, start=1):
            seq_str = s.get("sequence_str", " ".join(s.get("sequence", [])))
            print(
                f"  #{i}: {seq_str}\n"
                f"       art_ppl={s['art_perplexity']:.2f} "
                f"cog_ppl={s['cog_perplexity']:.2f}"
            )


if __name__ == "__main__":
    main()
