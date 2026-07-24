"""
Fix artist names in artworks.jsonl by parsing them from the image filename.

The WikiArt zip structure is: wikiart/<Style>/<artist-name>_<title>.jpg
The artist name is everything before the FIRST underscore in the filename,
with hyphens as word separators.

Example:
  henri-matisse_the-king-s-sadness-1952.jpg -> artist = "henri matisse"
  william-congdon_the-black-city-i-new-york-1949.jpg -> artist = "william congdon"

This also rebuilds artist_concepts.parquet/jsonl with correct artist grouping.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def extract_artist_from_filename(image_member: str) -> str:
    """Extract artist name from wikiart filename."""
    # Get just the filename: "henri-matisse_the-king-s-sadness-1952.jpg"
    filename = Path(image_member).stem
    # Split on first underscore: artist is the left part
    parts = filename.split("_", 1)
    artist_raw = parts[0]  # "henri-matisse"
    # Convert hyphens to spaces
    artist = artist_raw.replace("-", " ")
    return artist


# Load artworks
artworks_path = Path("data/processed/artworks.jsonl")
with open(artworks_path) as f:
    artworks = [json.loads(line) for line in f]

print(f"Loaded {len(artworks)} artworks")

# Fix artist names
for row in artworks:
    row["artist"] = extract_artist_from_filename(row["image_member"])

# Check results
artists = Counter(r["artist"] for r in artworks)
print(f"Unique artists: {len(artists)}")
print("Top 15 artists:")
for a, c in artists.most_common(15):
    print(f"  {a}: {c}")

# Save fixed artworks.jsonl
with open(artworks_path, "w", encoding="utf-8") as f:
    for row in artworks:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"\nSaved fixed {artworks_path}")

# Rebuild artworks.parquet
artworks_df = pd.DataFrame(artworks)
artworks_df.to_parquet("data/processed/artworks.parquet", index=False)
print("Saved data/processed/artworks.parquet")

# Rebuild artist_concepts table
artist_groups = defaultdict(lambda: {"concepts": set(), "count": 0})
for row in artworks:
    artist = row["artist"]
    artist_groups[artist]["count"] += 1
    for c in row["all10_concepts"]:
        artist_groups[artist]["concepts"].add(c)

artist_rows = []
for artist, data in sorted(artist_groups.items()):
    artist_rows.append({
        "artist": artist,
        "n_artworks": data["count"],
        "concepts_union": sorted(data["concepts"]),
    })

artists_df = pd.DataFrame(artist_rows)
artists_df.to_parquet("data/processed/artist_concepts.parquet", index=False)
artists_df.to_json("data/processed/artist_concepts.jsonl", orient="records", lines=True, force_ascii=False)
print(f"Saved artist_concepts: {len(artist_rows)} artists")

# Also rebuild cog_corpus.txt (artist-level concept samples)
import random
random.seed(42)

cog_lines = []
for row in artist_rows:
    pool = row["concepts_union"]
    if not pool:
        continue
    for _ in range(20):  # 20 samples per artist
        if len(pool) <= 10:
            sample = pool[:]
        else:
            sample = random.sample(pool, 10)
        random.shuffle(sample)
        cog_lines.append(" ".join(sample))

with open("data/processed/cog_corpus.txt", "w", encoding="utf-8") as f:
    for line in cog_lines:
        f.write(line + "\n")

print(f"Saved cog_corpus.txt: {len(cog_lines)} lines")
print("\nDone!")
