"""Quick test that vocab parsing works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess.preprocess import read_wordnet_core

words = read_wordnet_core(
    Path("./data/processed/cache"),
    min_len=3,
    max_len=24,
    allow_phrases=False,
    remove_generic=True,
)
print(f"Got {len(words)} concepts")
print("First 20:", words[:20])
print("Last 20:", words[-20:])
