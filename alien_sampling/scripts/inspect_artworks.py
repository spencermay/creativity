import json
from collections import Counter

with open("data/processed/artworks.jsonl") as f:
    lines = [json.loads(l) for l in f]

print(f"Total artworks: {len(lines)}")
print("\nSample entry:")
print(json.dumps(lines[0], indent=2))

artists = Counter(r["artist"] for r in lines)
print(f"\nUnique artists: {len(artists)}")
print("Top 10 artists:")
for a, c in artists.most_common(10):
    print(f"  {a}: {c}")

styles = Counter(r["style"] for r in lines)
print(f"\nUnique styles: {len(styles)}")
print("Top 10 styles:")
for s, c in styles.most_common(10):
    print(f"  {s}: {c}")

# Check image_member format
print("\nSample image_members:")
for r in lines[:5]:
    print(f"  {r['image_member']}")
    print(f"    artist={r['artist']}, style={r['style']}")
