"""Demo: full Alien sampling pipeline end-to-end."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.naive_bayes_model import (
    NaiveBayesConceptModel,
    load_artwork_concept_sets,
    load_artist_concept_sets,
    score_all_candidates,
)
from inference.generate_sequences import generate_candidates, load_vocabulary
from inference.alien_sampling import alien_rank, select_top_k

# Load models
print("Loading models...")
art_model = NaiveBayesConceptModel.load(Path("data/processed/cooccurrence/art_model"))
cog_model = NaiveBayesConceptModel.load(Path("data/processed/cooccurrence/cog_model"))

# Load vocabulary
vocab_set = load_vocabulary(Path("data/processed/vocab/vocab_words.json"))
vocab_list = sorted(vocab_set)

# Generate candidates from seed
seed = ["impressionism"]
temperature = 2.5
num_candidates = 100
beta = 0.85

print(f"\nSeed: {seed}")
print(f"Temperature: {temperature}")
print(f"Beta: {beta}")
print(f"Generating {num_candidates} candidates...\n")

candidates = generate_candidates(
    model=art_model,
    seed_concepts=seed,
    vocabulary=vocab_list,
    num_candidates=num_candidates,
    temperature=temperature,
    min_seq_len=8,
    max_seq_len=10,
)

# Score all candidates under both models
print(f"\nScoring {len(candidates)} candidates...")
scored = score_all_candidates(candidates, art_model, cog_model)

# Apply Alien sampling
ranked = alien_rank(scored, beta=beta)
alien_top = select_top_k(ranked, top_k=5)

# Baseline: top by Art model alone
baseline_top = sorted(scored, key=lambda s: s["art_perplexity"])[:5]

print(f"\n{'='*70}")
print(f"ALIEN SAMPLING RESULTS (beta={beta})")
print(f"{'='*70}")
print(f"\nTop-5 ALIEN sequences (plausible + cognitively unavailable):")
for i, s in enumerate(alien_top, 1):
    print(f"  #{i}: {s['sequence_str']}")
    print(f"      art_ppl={s['art_perplexity']:.0f}  cog_ppl={s['cog_perplexity']:.0f}  "
          f"fused={s['fused_score']:.4f}")

print(f"\nTop-5 BASELINE sequences (Art-only, most plausible):")
for i, s in enumerate(baseline_top, 1):
    print(f"  #{i}: {s['sequence_str']}")
    print(f"      art_ppl={s['art_perplexity']:.0f}  cog_ppl={s['cog_perplexity']:.0f}")

print(f"\n{'='*70}")
print("Key insight: Alien sequences should have HIGH cog_ppl (novel to artists)")
print("while maintaining reasonable art_ppl (still plausible as artworks).")
print(f"{'='*70}")
