"""Build Naive Bayes models and test generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.naive_bayes_model import (
    NaiveBayesConceptModel,
    load_artwork_concept_sets,
    load_artist_concept_sets,
)

# Build Art model
print("Building Art model...")
art_sets = load_artwork_concept_sets(Path("data/processed/artworks.jsonl"))
art_model = NaiveBayesConceptModel(smoothing=1.0)
art_model.fit(art_sets)
art_model.save(Path("data/processed/cooccurrence/art_model"))

# Build Cog model
print("\nBuilding Cog model...")
cog_sets = load_artist_concept_sets(Path("data/processed/artist_concepts.jsonl"))
cog_model = NaiveBayesConceptModel(smoothing=1.0)
cog_model.fit(cog_sets)
cog_model.save(Path("data/processed/cooccurrence/cog_model"))

# Quick test: rank candidates given a seed
print("\n=== Art model: top concepts given context=[mountain] ===")
top = art_model.rank_candidates(["mountain"], top_k=15)
for concept, score in top:
    print(f"  {concept:20s}  art_score={score:.3f}")

print("\n=== Cog model: top concepts given context=[mountain] ===")
top_cog = cog_model.rank_candidates(["mountain"], top_k=15)
for concept, score in top_cog:
    print(f"  {concept:20s}  cog_score={score:.3f}")

# Test full sequence generation
print("\n=== Generating 5 sample sequences from seed='mountain' ===")
from inference.generate_sequences import generate_candidates
import json

vocab_path = Path("data/processed/vocab/vocab_words.json")
with open(vocab_path) as f:
    vocab_list = sorted(json.load(f))

candidates = generate_candidates(
    model=art_model,
    seed_concepts=["mountain"],
    vocabulary=vocab_list,
    num_candidates=5,
    temperature=2.0,
    min_seq_len=8,
    max_seq_len=10,
)

for i, seq in enumerate(candidates, 1):
    print(f"  #{i}: {' '.join(seq)}")

# Score them
print("\n=== Scoring with Art + Cog perplexity ===")
from inference.naive_bayes_model import score_all_candidates

scored = score_all_candidates(candidates, art_model, cog_model)
for s in scored:
    print(f"  {s['sequence_str']}")
    print(f"    art_ppl={s['art_perplexity']:.2f}  cog_ppl={s['cog_perplexity']:.2f}")
