import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.naive_bayes_model import NaiveBayesConceptModel

# Load art model
art_model = NaiveBayesConceptModel.load(Path("data/processed/cooccurrence/art_model"))

# Find most common concepts
with open("data/processed/artworks.jsonl") as f:
    artworks = [json.loads(l) for l in f]

concept_counts = Counter()
for a in artworks:
    for c in a["all10_concepts"]:
        concept_counts[c] += 1

print("Most common concepts in dataset:")
for c, n in concept_counts.most_common(20):
    print(f"  {c}: {n}")

# Test ranking with common concepts
print("\n=== Top next concepts given [abstract expressionism, insight] ===")
top = art_model.rank_candidates(["abstract expressionism", "insight"], top_k=10)
for concept, score in top:
    print(f"  {concept:25s}  score={score:.4f}")

print("\n=== Top next concepts given [impressionism, landscape] ===")
# 'landscape' might not be in vocab, try with what's common
top = art_model.rank_candidates(["impressionism"], top_k=10)
for concept, score in top:
    print(f"  {concept:25s}  score={score:.4f}")
