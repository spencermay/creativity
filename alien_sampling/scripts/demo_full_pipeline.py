"""
Full end-to-end pipeline demo:
1. Load NB models (already built)
2. Generate candidates from seed
3. Score under Art + Cog models
4. Apply Alien sampling
5. Build image prompts
6. (Optionally) generate images with FLUX
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.naive_bayes_model import (
    NaiveBayesConceptModel,
    score_all_candidates,
)
from inference.generate_sequences import generate_candidates, load_vocabulary
from inference.alien_sampling import alien_rank, select_top_k
from inference.make_images import build_prompt, generate_image, save_image

# --- Config ---
SEED_CONCEPTS = ["impressionism"]
TEMPERATURE = 2.5
NUM_CANDIDATES = 100
BETA = 0.85
TOP_K = 5
MODEL = "x/flux2-klein:4b"
OUTPUT_DIR = Path("outputs/demo")

# --- Load ---
print("=" * 60)
print("ALIEN RECOMBINATION - FULL PIPELINE DEMO")
print("=" * 60)

print("\n[1/5] Loading models...")
art_model = NaiveBayesConceptModel.load(Path("data/processed/cooccurrence/art_model"))
cog_model = NaiveBayesConceptModel.load(Path("data/processed/cooccurrence/cog_model"))
vocab_set = load_vocabulary(Path("data/processed/vocab/vocab_words.json"))
vocab_list = sorted(vocab_set)

# --- Generate ---
print(f"\n[2/5] Generating {NUM_CANDIDATES} candidates from seed={SEED_CONCEPTS}, T={TEMPERATURE}...")
candidates = generate_candidates(
    model=art_model,
    seed_concepts=SEED_CONCEPTS,
    vocabulary=vocab_list,
    num_candidates=NUM_CANDIDATES,
    temperature=TEMPERATURE,
    min_seq_len=8,
    max_seq_len=10,
)

# --- Score ---
print(f"\n[3/5] Scoring {len(candidates)} candidates under Art + Cog models...")
scored = score_all_candidates(candidates, art_model, cog_model)

# --- Rank ---
print(f"\n[4/5] Alien sampling (beta={BETA})...")
ranked = alien_rank(scored, beta=BETA)
alien_top = select_top_k(ranked, top_k=TOP_K)
baseline_top = sorted(scored, key=lambda s: s["art_perplexity"])[:TOP_K]

# --- Results ---
print(f"\n[5/5] Results")
print(f"\n{'='*60}")
print(f"ALIEN TOP-{TOP_K} (plausible + cognitively unavailable):")
print(f"{'='*60}")
for i, s in enumerate(alien_top, 1):
    prompt = build_prompt(s["sequence"])
    print(f"\n  #{i}: {s['sequence_str']}")
    print(f"      art_ppl={s['art_perplexity']:.0f}  cog_ppl={s['cog_perplexity']:.0f}")
    print(f"      FLUX prompt: \"{prompt}\"")

print(f"\n{'='*60}")
print(f"BASELINE TOP-{TOP_K} (art-only, for comparison):")
print(f"{'='*60}")
for i, s in enumerate(baseline_top, 1):
    prompt = build_prompt(s["sequence"])
    print(f"\n  #{i}: {s['sequence_str']}")
    print(f"      art_ppl={s['art_perplexity']:.0f}  cog_ppl={s['cog_perplexity']:.0f}")
    print(f"      FLUX prompt: \"{prompt}\"")

# --- Attempt image generation ---
print(f"\n{'='*60}")
print(f"ATTEMPTING IMAGE GENERATION...")
print(f"{'='*60}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try generating one image
top_seq = alien_top[0]["sequence"]
prompt = build_prompt(top_seq)
print(f"\nPrompt: {prompt}")
print(f"Model: {MODEL}")

img_bytes = generate_image(MODEL, prompt, width=512, height=512, steps=4, seed=42)
if img_bytes:
    path = OUTPUT_DIR / "alien_top1.png"
    save_image(img_bytes, path)
    print(f"SUCCESS! Saved to {path} ({len(img_bytes)} bytes)")
else:
    print("Image generation failed (likely OOM on 8GB machine).")
    print("To generate images, close Kiro and run from terminal:")
    print(f"  python inference/make_images.py \\")
    print(f"    --prompt \"{prompt}\" \\")
    print(f"    --output-dir ./outputs/demo")

# Save the selected sequences for later use
output_file = OUTPUT_DIR / "alien_results.json"
with open(output_file, "w") as f:
    json.dump({
        "seed_sequence": SEED_CONCEPTS,
        "temperature": TEMPERATURE,
        "beta": BETA,
        "selected": alien_top,
        "baseline": baseline_top,
    }, f, indent=2)
print(f"\nResults saved to {output_file}")
