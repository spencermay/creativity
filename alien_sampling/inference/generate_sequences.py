"""
Constrained candidate sequence generation by sampling from Naive Bayes
co-occurrence probabilities.

Given a seed sequence (e.g. "romanticism mountain"), this script generates
candidate continuations by repeatedly sampling the next concept from:

    P(v | S) ∝ ∏_{i ∈ S} P(v | i)

where P(v|i) comes from the Art co-occurrence model. Temperature controls
how peaked vs uniform the sampling distribution is.

No LLM or Ollama required — generation is purely from the co-occurrence statistics.

Usage:
    python inference/generate_sequences.py \
        --seed-sequence "romanticism mountain" \
        --vocab ./data/processed/vocab/vocab_words.json \
        --artworks ./data/processed/artworks.parquet \
        --num-candidates 150 \
        --temperature 2.5 \
        --output ./outputs/sequences/romanticism_mountain_candidates.json \
        --config ./configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed
from inference.naive_bayes_model import (
    NaiveBayesConceptModel,
    load_artwork_concept_sets,
)


def load_vocabulary(vocab_path: Path) -> set[str]:
    """Load the constrained vocabulary from JSON."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        words = json.load(f)
    vocab = set(words)
    print(f"[INFO] Loaded vocabulary with {len(vocab)} concepts")
    return vocab


def sample_next_concept(
    model: NaiveBayesConceptModel,
    context: list[str],
    vocabulary: list[str],
    exclude: set[str],
    temperature: float = 1.0,
) -> Optional[str]:
    """
    Sample the next concept from the Naive Bayes distribution.

    Computes log P(v | context) for all v in vocabulary, applies temperature
    scaling, converts to a proper distribution via softmax, then samples.

    Parameters
    ----------
    model : NaiveBayesConceptModel
    context : current concepts in the sequence so far
    vocabulary : list of all valid candidate concepts
    exclude : concepts to exclude (already in sequence)
    temperature : >1 = more uniform/random, <1 = more peaked/greedy

    Returns
    -------
    sampled concept string, or None if no valid candidates
    """
    # Compute log-scores for all candidates
    candidates = []
    log_scores = []

    for v in vocabulary:
        if v in exclude:
            continue
        if v not in model.concept_to_idx:
            continue

        score = model.score_candidate(context, v)
        if score > float("-inf"):
            candidates.append(v)
            log_scores.append(score)

    if not candidates:
        return None

    # Apply temperature scaling
    log_scores = np.array(log_scores, dtype=np.float64)
    if temperature != 1.0:
        log_scores = log_scores / temperature

    # Softmax to get probabilities
    log_scores -= log_scores.max()  # numerical stability
    probs = np.exp(log_scores)
    probs /= probs.sum()

    # Sample
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]


def generate_single_candidate(
    model: NaiveBayesConceptModel,
    seed_concepts: list[str],
    vocabulary: list[str],
    temperature: float,
    min_seq_len: int,
    max_seq_len: int,
) -> Optional[list[str]]:
    """
    Generate a single candidate sequence by iteratively sampling from
    the Naive Bayes distribution.

    Starting from the seed, sample concepts one at a time until
    max_seq_len is reached.
    """
    sequence = list(seed_concepts)
    used = set(seed_concepts)

    # How many concepts to generate
    target_len = random.randint(min_seq_len, max_seq_len)
    concepts_to_add = target_len - len(sequence)

    for _ in range(concepts_to_add):
        next_concept = sample_next_concept(
            model=model,
            context=sequence,
            vocabulary=vocabulary,
            exclude=used,
            temperature=temperature,
        )

        if next_concept is None:
            break

        sequence.append(next_concept)
        used.add(next_concept)

    # Check minimum length
    if len(sequence) < min_seq_len:
        return None

    return sequence


def generate_candidates(
    model: NaiveBayesConceptModel,
    seed_concepts: list[str],
    vocabulary: list[str],
    num_candidates: int,
    temperature: float,
    min_seq_len: int,
    max_seq_len: int,
) -> list[list[str]]:
    """
    Generate num_candidates unique sequences by sampling from the
    Naive Bayes Art model.
    """
    candidates = []
    seen_sequences = set()

    pbar = tqdm(total=num_candidates, desc="Generating candidates (Naive Bayes)")

    # We almost never fail to produce a valid sequence, but cap attempts anyway
    max_attempts = num_candidates * 3
    attempts = 0

    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1

        seq = generate_single_candidate(
            model=model,
            seed_concepts=seed_concepts,
            vocabulary=vocabulary,
            temperature=temperature,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
        )

        if seq is None:
            continue

        # Deduplicate
        seq_key = tuple(seq)
        if seq_key in seen_sequences:
            continue

        seen_sequences.add(seq_key)
        candidates.append(seq)
        pbar.update(1)

    pbar.close()

    if len(candidates) < num_candidates:
        print(
            f"[WARN] Only generated {len(candidates)}/{num_candidates} valid candidates "
            f"after {attempts} attempts"
        )

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Generate constrained concept sequences via Naive Bayes sampling"
    )
    parser.add_argument(
        "--seed-sequence", type=str, required=True,
        help="Space-separated seed concepts (e.g. 'romanticism mountain')"
    )
    parser.add_argument("--vocab", type=Path, required=True, help="Path to vocab_words.json")
    parser.add_argument("--artworks", type=Path, required=True, help="Artworks parquet/JSONL")
    parser.add_argument("--art-model-dir", type=Path, default=None,
                        help="Pre-built Art NB model dir (skip rebuilding)")
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min-seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    gen_cfg = config["generation"]
    num_candidates = args.num_candidates or gen_cfg["num_candidates"]
    temperature = args.temperature or gen_cfg["temperature"]
    min_seq_len = args.min_seq_len or gen_cfg["min_sequence_len"]
    max_seq_len = args.max_seq_len or gen_cfg["max_sequence_len"]

    # Parse seed
    seed_concepts = args.seed_sequence.strip().lower().split()
    print(f"[INFO] Seed sequence: {seed_concepts}")
    print(f"[INFO] Temperature: {temperature}, Candidates: {num_candidates}")
    print(f"[INFO] Sequence length: [{min_seq_len}, {max_seq_len}]")

    # Load vocabulary
    vocabulary_set = load_vocabulary(args.vocab)
    vocabulary_list = sorted(vocabulary_set)

    # Validate seed concepts
    for c in seed_concepts:
        if c not in vocabulary_set:
            print(f"[WARN] Seed concept '{c}' not in vocabulary - proceeding anyway")

    # Build or load Art model
    if args.art_model_dir and args.art_model_dir.exists():
        art_model = NaiveBayesConceptModel.load(args.art_model_dir)
    else:
        print("[INFO] Building Art co-occurrence model...")
        artwork_sets = load_artwork_concept_sets(args.artworks)
        art_model = NaiveBayesConceptModel(smoothing=args.smoothing)
        art_model.fit(artwork_sets)

    # Generate
    candidates = generate_candidates(
        model=art_model,
        seed_concepts=seed_concepts,
        vocabulary=vocabulary_list,
        num_candidates=num_candidates,
        temperature=temperature,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "seed_sequence": seed_concepts,
        "generation_method": "naive_bayes_sampling",
        "temperature": temperature,
        "num_candidates_requested": num_candidates,
        "num_candidates_generated": len(candidates),
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "candidates": candidates,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {len(candidates)} candidates to {args.output}")

    # Print a few examples
    print("\n[INFO] Example candidates:")
    for seq in candidates[:5]:
        print(f"  {' '.join(seq)}")


if __name__ == "__main__":
    main()
