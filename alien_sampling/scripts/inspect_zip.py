"""Quick inspection of the wikiart zip structure."""
import zipfile
import sys
from collections import Counter
from pathlib import Path

zip_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/spencermay/Downloads/wikiart.zip"

zf = zipfile.ZipFile(zip_path)
members = zf.namelist()
print(f"Total members: {len(members)}")

print("\nFirst 30 members:")
for m in members[:30]:
    print(f"  {m}")

# Extensions
exts = Counter()
for m in members:
    name = Path(m).name
    if "." in name:
        exts[Path(m).suffix.lower()] += 1
    else:
        exts["(directory)"] += 1

print("\nExtensions:")
for ext, count in exts.most_common(10):
    print(f"  {ext}: {count}")

# Path depth
depths = Counter()
for m in members:
    depths[len(Path(m).parts)] += 1

print("\nPath depth distribution:")
for d, c in sorted(depths.items()):
    print(f"  depth {d}: {c}")

# Sample some image paths
img_exts = {".jpg", ".jpeg", ".png"}
images = [m for m in members if Path(m).suffix.lower() in img_exts]
print(f"\nTotal images: {len(images)}")
print("Sample image paths:")
for m in images[:10]:
    print(f"  {m}")

# Unique first-level dirs (likely styles or artists)
top_dirs = Counter()
for m in images:
    parts = Path(m).parts
    if len(parts) >= 2:
        top_dirs[parts[0]] += 1

print(f"\nTop-level directories ({len(top_dirs)} unique):")
for d, c in top_dirs.most_common(15):
    print(f"  {d}: {c} images")
