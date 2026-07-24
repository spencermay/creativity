"""
Naive Bayes co-occurrence model for scoring concept sequences.

Replaces expensive LoRA fine-tuning of gemma3:4b with a fast, closed-form
estimate of concept plausibility based on co-occurrence statistics.

Given a set S = {c_0, ..., c_{k-1}}, score a candidate concept v by:

    P(v | S) ∝ P(S ∪ {v}) ≈ ∏_{i ∈ S} P(v | i)

where:
    P(v | i) = #{artworks containing both i and v} / #{artworks containing i}

This is the Naive Bayes independence assumption applied to concept co-occurrence.

Two separate models are built:
- Art model: co-occurrence over ALL artworks (measures artwork-level plausibility)
- Cog model: co-occurrence over per-ARTIST concept pools (measures cognitive availability)

Scoring:
- Art score: high P(v|S) means v is plausible alongside S (low "art perplexity")
- Cog score: high P(v|S) means v is cognitively available (low "cog perplexity")
- For alien sampling we want: high Art score + low Cog score

Usage:
    # Build models and score candidates:
    python inference/naive_bayes_model.py \
        --artworks ./data/processed/artworks.parquet \
        --artists ./data/processed/artist_concepts.parquet \
        --candidates ./outputs/sequences/romanticism_mountain_candidates.json \
        --output ./outputs/sequences/romanticism_mountain_scored.json

    # Or just build and save the co-occurrence matrices:
    python inference/naive_bayes_model.py build \
        --artworks ./data/processed/artworks.parquet \
        --artists ./data/processed/artist_concepts.parquet \
        --output-dir ./data/processed/cooccurrence/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


class NaiveBayesConceptModel:
    """
    Naive Bayes concept co-occurrence model.

    Builds a co-occurrence matrix from a collection of concept sets,
    then scores new concept combinations by the product of pairwise
    conditional probabilities.
    """

    def __init__(self, smoothing: float = 1.0):
        """
        Parameters
        ----------
        smoothing : float
            Laplace smoothing parameter. Prevents zero probabilities
            for unseen co-occurrences.
        """
        self.smoothing = smoothing
        self.concept_to_idx: dict[str, int] = {}
        self.idx_to_concept: dict[int, str] = {}
        self.vocab_size: int = 0

        # Counts
        self.concept_count: np.ndarray | None = None  # (V,) how many sets contain concept i
        self.cooccurrence: np.ndarray | None = None  # (V, V) how many sets contain both i and j
        self.total_sets: int = 0

    def fit(self, concept_sets: list[set[str]]) -> None:
        """
        Fit the model on a collection of concept sets.

        Parameters
        ----------
        concept_sets : list of sets
            Each set is a collection of concepts that co-occur
            (e.g., one artwork's 10 concepts, or one artist's full pool).
        """
        # Build vocabulary from all observed concepts
        all_concepts = set()
        for s in concept_sets:
            all_concepts.update(s)

        self.concept_to_idx = {c: i for i, c in enumerate(sorted(all_concepts))}
        self.idx_to_concept = {i: c for c, i in self.concept_to_idx.items()}
        self.vocab_size = len(self.concept_to_idx)
        self.total_sets = len(concept_sets)

        # Count occurrences and co-occurrences
        self.concept_count = np.zeros(self.vocab_size, dtype=np.float64)
        self.cooccurrence = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)

        for concept_set in tqdm(concept_sets, desc="Building co-occurrence matrix"):
            indices = [self.concept_to_idx[c] for c in concept_set if c in self.concept_to_idx]
            for idx in indices:
                self.concept_count[idx] += 1
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i + 1:]:
                    self.cooccurrence[idx_i, idx_j] += 1
                    self.cooccurrence[idx_j, idx_i] += 1

        print(f"[INFO] Built model: {self.vocab_size} concepts, {self.total_sets} sets")

    def log_prob_v_given_i(self, v_idx: int, i_idx: int) -> float:
        """
        Compute log P(v | i) with Laplace smoothing.

        P(v | i) = (count(i, v) + α) / (count(i) + α * V)
        """
        numerator = self.cooccurrence[i_idx, v_idx] + self.smoothing
        denominator = self.concept_count[i_idx] + self.smoothing * self.vocab_size
        return math.log(numerator / denominator)

    def score_candidate(self, context: list[str], candidate: str) -> float:
        """
        Score a candidate concept given a context set.

        Returns log P(candidate | context) under the Naive Bayes model:
            log P(v | S) = Σ_{i ∈ S} log P(v | i) + const

        Higher score = more probable/plausible.
        """
        if candidate not in self.concept_to_idx:
            return float("-inf")

        v_idx = self.concept_to_idx[candidate]
        log_score = 0.0

        for c in context:
            if c in self.concept_to_idx:
                i_idx = self.concept_to_idx[c]
                log_score += self.log_prob_v_given_i(v_idx, i_idx)

        return log_score

    def score_sequence(self, sequence: list[str]) -> float:
        """
        Score a full concept sequence by accumulating Naive Bayes scores.

        For sequence [c_0, c_1, ..., c_n], compute:
            score = Σ_{k=1}^{n} log P(c_k | c_0, ..., c_{k-1})

        This gives a measure analogous to negative log-likelihood.
        Lower (more negative) = less probable = higher "perplexity".
        """
        if len(sequence) < 2:
            return 0.0

        total_log_prob = 0.0
        for k in range(1, len(sequence)):
            context = sequence[:k]
            candidate = sequence[k]
            total_log_prob += self.score_candidate(context, candidate)

        return total_log_prob

    def perplexity(self, sequence: list[str]) -> float:
        """
        Compute perplexity of a sequence under this model.

        perplexity = exp(-1/N * total_log_prob)

        Lower perplexity = more plausible under the model.
        """
        if len(sequence) < 2:
            return 1.0

        total_log_prob = self.score_sequence(sequence)
        n = len(sequence) - 1  # number of predictions made
        avg_neg_log_prob = -total_log_prob / n
        return math.exp(avg_neg_log_prob)

    def rank_candidates(
        self,
        context: list[str],
        vocabulary: list[str] | None = None,
        top_k: int = 20,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank all candidate concepts given a context.

        Returns top-k (concept, score) pairs sorted by descending score.
        """
        exclude = exclude or set()
        candidates = vocabulary if vocabulary else list(self.concept_to_idx.keys())

        scored = []
        for c in candidates:
            if c in exclude or c in context:
                continue
            score = self.score_candidate(context, c)
            if score > float("-inf"):
                scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "concept_count.npy", self.concept_count)
        np.save(path / "cooccurrence.npy", self.cooccurrence)
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.concept_to_idx, f, ensure_ascii=False, indent=2)
        meta = {"total_sets": self.total_sets, "smoothing": self.smoothing}
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "NaiveBayesConceptModel":
        """Load model from disk."""
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            concept_to_idx = json.load(f)

        model = cls(smoothing=meta["smoothing"])
        model.concept_to_idx = concept_to_idx
        model.idx_to_concept = {i: c for c, i in concept_to_idx.items()}
        model.vocab_size = len(concept_to_idx)
        model.total_sets = meta["total_sets"]
        model.concept_count = np.load(path / "concept_count.npy")
        model.cooccurrence = np.load(path / "cooccurrence.npy")
        print(f"[INFO] Loaded model from {path}: {model.vocab_size} concepts, {model.total_sets} sets")
        return model


def load_artwork_concept_sets(artworks_path: Path) -> list[set[str]]:
    """Load per-artwork concept sets for the Art model."""
    if artworks_path.suffix == ".parquet":
        df = pd.read_parquet(artworks_path)
    else:
        df = pd.read_json(artworks_path, orient="records", lines=True)

    col = "all10_concepts" if "all10_concepts" in df.columns else "top9_concepts"
    concept_sets = []
    for concepts in df[col]:
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        concept_sets.append(set(concepts))

    print(f"[INFO] Loaded {len(concept_sets)} artwork concept sets")
    return concept_sets


def load_artist_concept_sets(artists_path: Path) -> list[set[str]]:
    """Load per-artist concept pools for the Cog model."""
    if artists_path.suffix == ".parquet":
        df = pd.read_parquet(artists_path)
    else:
        df = pd.read_json(artists_path, orient="records", lines=True)

    concept_sets = []
    for concepts in df["concepts_union"]:
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        concept_sets.append(set(concepts))

    print(f"[INFO] Loaded {len(concept_sets)} artist concept pools")
    return concept_sets


def score_all_candidates(
    candidates: list[list[str]],
    art_model: NaiveBayesConceptModel,
    cog_model: NaiveBayesConceptModel,
) -> list[dict]:
    """Score all candidate sequences under both models."""
    scored = []

    for seq in tqdm(candidates, desc="Scoring candidates (Naive Bayes)"):
        art_ppl = art_model.perplexity(seq)
        cog_ppl = cog_model.perplexity(seq)

        scored.append({
            "sequence": seq,
            "sequence_str": " ".join(seq),
            "art_perplexity": art_ppl,
            "cog_perplexity": cog_ppl,
            "art_log_prob": art_model.score_sequence(seq),
            "cog_log_prob": cog_model.score_sequence(seq),
        })

    # Stats
    art_ppls = [s["art_perplexity"] for s in scored]
    cog_ppls = [s["cog_perplexity"] for s in scored]
    print(f"[INFO] Art PPL: min={min(art_ppls):.2f}, max={max(art_ppls):.2f}, "
          f"mean={sum(art_ppls)/len(art_ppls):.2f}")
    print(f"[INFO] Cog PPL: min={min(cog_ppls):.2f}, max={max(cog_ppls):.2f}, "
          f"mean={sum(cog_ppls)/len(cog_ppls):.2f}")

    return scored


def cmd_build(args):
    """Build and save co-occurrence models."""
    artwork_sets = load_artwork_concept_sets(args.artworks)
    artist_sets = load_artist_concept_sets(args.artists)

    smoothing = args.smoothing if args.smoothing else 1.0

    # Art model
    print("\n[INFO] Building Art model...")
    art_model = NaiveBayesConceptModel(smoothing=smoothing)
    art_model.fit(artwork_sets)
    art_model.save(args.output_dir / "art_model")

    # Cog model
    print("\n[INFO] Building Cog model...")
    cog_model = NaiveBayesConceptModel(smoothing=smoothing)
    cog_model.fit(artist_sets)
    cog_model.save(args.output_dir / "cog_model")

    print(f"\n[INFO] Both models saved to {args.output_dir}")


def cmd_score(args):
    """Score candidate sequences using pre-built or freshly-built models."""
    config = load_config(args.config)

    # Build or load models
    if args.art_model_dir and args.art_model_dir.exists():
        art_model = NaiveBayesConceptModel.load(args.art_model_dir)
    else:
        artwork_sets = load_artwork_concept_sets(args.artworks)
        art_model = NaiveBayesConceptModel(smoothing=args.smoothing or 1.0)
        art_model.fit(artwork_sets)

    if args.cog_model_dir and args.cog_model_dir.exists():
        cog_model = NaiveBayesConceptModel.load(args.cog_model_dir)
    else:
        artist_sets = load_artist_concept_sets(args.artists)
        cog_model = NaiveBayesConceptModel(smoothing=args.smoothing or 1.0)
        cog_model.fit(artist_sets)

    # Load candidates
    with open(args.candidates, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]
    print(f"[INFO] Scoring {len(candidates)} candidates")

    # Score
    scored = score_all_candidates(candidates, art_model, cog_model)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "seed_sequence": data.get("seed_sequence", []),
        "scoring_method": "naive_bayes",
        "art_model": "co-occurrence (artwork-level)",
        "cog_model": "co-occurrence (artist-level)",
        "num_scored": len(scored),
        "scored_candidates": scored,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved scored candidates to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Naive Bayes concept co-occurrence model")
    subparsers = parser.add_subparsers(dest="command")

    # Build sub-command
    p_build = subparsers.add_parser("build", help="Build and save co-occurrence models")
    p_build.add_argument("--artworks", type=Path, required=True)
    p_build.add_argument("--artists", type=Path, required=True)
    p_build.add_argument("--output-dir", type=Path, required=True)
    p_build.add_argument("--smoothing", type=float, default=1.0)

    # Score sub-command
    p_score = subparsers.add_parser("score", help="Score candidate sequences")
    p_score.add_argument("--candidates", type=Path, required=True)
    p_score.add_argument("--artworks", type=Path, default=None)
    p_score.add_argument("--artists", type=Path, default=None)
    p_score.add_argument("--art-model-dir", type=Path, default=None)
    p_score.add_argument("--cog-model-dir", type=Path, default=None)
    p_score.add_argument("--output", type=Path, required=True)
    p_score.add_argument("--smoothing", type=float, default=1.0)
    p_score.add_argument("--config", type=Path, default=None)

    # Default mode (no subcommand): score directly with inline model building
    parser.add_argument("--artworks", type=Path, default=None)
    parser.add_argument("--artists", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--config", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.candidates and args.output:
        # Default mode: build models inline and score
        args.art_model_dir = None
        args.cog_model_dir = None
        cmd_score(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
