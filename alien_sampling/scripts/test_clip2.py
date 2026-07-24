import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.preprocess import ClipEmbedder

embedder = ClipEmbedder("openai/clip-vit-base-patch32", "cpu")
emb = embedder.encode_texts(["a painting of castle", "a painting of mountain"], batch_size=2)
print(f"Text embeddings shape: {emb.shape}")
print(f"Norm of first: {(emb[0]**2).sum()**0.5:.4f}")
