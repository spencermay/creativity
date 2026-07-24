"""
End-to-end preprocessing script for reproducing the data pipeline in:

  Hernandez et al. (2024)
  "Alien Recombination: Exploring Concept Blends Beyond Human Cognitive Availability in Visual Art"

What this script does
---------------------
1. Downloads WordNet Core from Princeton
2. Normalizes and optionally prunes semantically redundant concepts using CLIP text embeddings
3. Downloads a WikiArt zip file to a temporary location
4. Iterates through images inside the zip without permanently extracting the whole dataset
5. Infers metadata from path structure and/or optional metadata CSV
6. Uses CLIP to extract top-k semantic concepts per image
7. Adds style metadata as the final concept
8. Saves a compact artwork table
9. Builds:
   - Art corpus: per-artwork concept permutations
   - Cognitive Availability corpus: per-artist aggregated concept-set samples

Recommended usage
-----------------
python preprocess_alien_recombination.py \
  --wikiart-zip-url "https://your/wikiart.zip" \
  --output-dir "./alien_data" \
  --metadata-csv "./wikiart_metadata.csv"

If you already have the zip locally:
python preprocess_alien_recombination.py \
  --wikiart-zip-path "./wikiart.zip" \
  --output-dir "./alien_data" \
  --metadata-csv "./wikiart_metadata.csv"

Notes
-----
- The script uses Hugging Face transformers CLIP.
- It keeps the WikiArt zip ephemeral if you pass --wikiart-zip-url.
- It saves only compact outputs permanently.
- Metadata parsing for artist/style depends on your WikiArt archive layout.
  If available, pass a metadata CSV mapping image files to artist/style.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


WORDNET_CORE_URL = "https://wordnetcode.princeton.edu/standoff-files/core-wordnet.txt"


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


GENERIC_STOPWORDS = {
    # overly generic terms that are often unhelpful for visual concept extraction
    "entity", "object", "thing", "item", "whole", "unit", "part", "piece",
    "person", "someone", "somebody", "man", "woman", "people", "individual",
    "group", "kind", "sort", "type", "form", "shape", "way", "area", "place",
    "location", "position", "space", "time", "day", "year", "system", "set",
    "matter", "property", "fact", "process", "activity", "action", "change",
    "state", "event", "condition", "measure", "medium", "surface", "work",
    "art", "artwork", "painting", "image", "picture",
}


@dataclass
class ArtworkRecord:
    artwork_id: str
    image_member: str
    artist: str
    style: str
    top9_concepts: List[str]
    top9_scores: List[float]
    all10_concepts: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_ ]+", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_concept(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_style(text: str) -> str:
    return slugify(text).replace("_", " ")


def is_image_member(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def download_file(url: str, dst_path: Path, chunk_size: int = 1 << 20) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dst_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def read_wordnet_core(
    cache_dir: Path,
    min_len: int = 3,
    max_len: int = 24,
    allow_phrases: bool = False,
    remove_generic: bool = True,
) -> List[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    core_path = cache_dir / "core-wordnet.txt"
    if not core_path.exists():
        print(f"[INFO] Downloading WordNet Core from {WORDNET_CORE_URL}")
        download_file(WORDNET_CORE_URL, core_path)

    words = []
    with open(core_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: POS [synset_key] [lemma] optional_gloss
            # Extract the lemma from the second bracketed field
            bracket_matches = re.findall(r"\[([^\]]+)\]", line)
            if len(bracket_matches) < 2:
                continue
            raw_lemma = bracket_matches[1]  # second bracket is the lemma
            w = normalize_concept(raw_lemma)
            if not w:
                continue
            if len(w) < min_len or len(w) > max_len:
                continue
            if not allow_phrases and " " in w:
                continue
            if not re.fullmatch(r"[a-z ]+", w):
                continue
            if remove_generic and w in GENERIC_STOPWORDS:
                continue
            words.append(w)

    words = sorted(set(words))
    print(f"[INFO] Loaded {len(words)} candidate WordNet Core concepts")
    return words


class ClipEmbedder:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
        normalize: bool = True,
    ) -> np.ndarray:
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = list(texts[i:i + batch_size])
            inputs = self.processor(
                text=batch,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
            out = self.model.get_text_features(**inputs)
            # Handle both tensor and ModelOutput returns (transformers version compat)
            if hasattr(out, "pooler_output"):
                emb = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state[:, 0, :]  # CLS token
            else:
                emb = out
            if normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_emb.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(all_emb, axis=0)

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        all_emb = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding image batch"):
            batch = list(images[i:i + batch_size])
            inputs = self.processor(
                text=None,
                images=batch,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "input_ids"}
            out = self.model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                emb = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state[:, 0, :]
            else:
                emb = out
            if normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_emb.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(all_emb, axis=0)

    @torch.no_grad()
    def encode_single_image(self, image: Image.Image, normalize: bool = True) -> np.ndarray:
        inputs = self.processor(text=None, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "input_ids"}
        out = self.model.get_image_features(**inputs)
        if hasattr(out, "pooler_output"):
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0, :]
        else:
            emb = out
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb[0].cpu().numpy().astype(np.float32)


def build_text_prompts(words: Sequence[str], template: str) -> List[str]:
    return [template.format(w) for w in words]


def greedy_semantic_prune(
    words: Sequence[str],
    embeddings: np.ndarray,
    cosine_threshold: float = 0.88,
    priority: Optional[Sequence[float]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Greedy pruning:
    keep a word only if its maximum cosine similarity to already-kept words is < threshold.

    Assumes embeddings are unit-normalized.
    """
    idxs = list(range(len(words)))
    if priority is not None:
        idxs.sort(key=lambda i: priority[i], reverse=True)

    kept_words = []
    kept_embs = []

    for i in tqdm(idxs, desc="Pruning semantic near-duplicates"):
        e = embeddings[i]
        if not kept_embs:
            kept_words.append(words[i])
            kept_embs.append(e)
            continue

        sims = np.dot(np.stack(kept_embs, axis=0), e)
        if float(np.max(sims)) < cosine_threshold:
            kept_words.append(words[i])
            kept_embs.append(e)

    kept_embs = np.stack(kept_embs, axis=0).astype(np.float32)
    print(f"[INFO] Pruned vocab from {len(words)} -> {len(kept_words)} concepts")
    return kept_words, kept_embs


def save_vocab_artifacts(
    out_dir: Path,
    vocab_words: Sequence[str],
    vocab_embeddings: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vocab_words.json", "w", encoding="utf-8") as f:
        json.dump(list(vocab_words), f, ensure_ascii=False, indent=2)
    np.save(out_dir / "vocab_embeddings.npy", vocab_embeddings)
    print(f"[INFO] Saved vocabulary artifacts to {out_dir}")


def load_metadata_csv(metadata_csv: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Expected CSV columns:
      image_member OR filename OR path
      artist
      style

    Matching is attempted by exact member path, filename, and filename stem.
    """
    if metadata_csv is None:
        return {}

    df = pd.read_csv(metadata_csv)
    cols = {c.lower(): c for c in df.columns}

    path_col = None
    for candidate in ["image_member", "filename", "path", "image_path", "file"]:
        if candidate in cols:
            path_col = cols[candidate]
            break

    if path_col is None or "artist" not in cols or "style" not in cols:
        raise ValueError(
            "metadata CSV must contain a path column (image_member/filename/path/image_path/file), plus artist and style"
        )

    mapping = {}
    for _, row in df.iterrows():
        raw_path = str(row[path_col])
        artist = str(row[cols["artist"]])
        style = str(row[cols["style"]])

        keys = {
            raw_path,
            Path(raw_path).name,
            Path(raw_path).stem,
        }
        for k in keys:
            mapping[k] = {"artist": artist, "style": style}

    print(f"[INFO] Loaded metadata for {len(mapping)} path keys from {metadata_csv}")
    return mapping


def infer_metadata_from_member(
    member_name: str,
    metadata_map: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    """
    Fallback heuristics if no explicit metadata is found.

    Tries:
    1. metadata map by full path / filename / stem
    2. path patterns like:
         style/artist/file.jpg
         artist/style/file.jpg
    3. defaults to unknown
    """
    for key in [member_name, Path(member_name).name, Path(member_name).stem]:
        if key in metadata_map:
            artist = metadata_map[key]["artist"]
            style = metadata_map[key]["style"]
            return normalize_style(artist), normalize_style(style)

    parts = [p for p in Path(member_name).parts if p not in (".", "")]
    if len(parts) >= 3:
        a, b = parts[-3], parts[-2]

        # heuristic: first is style, second is artist
        style1, artist1 = normalize_style(a), normalize_style(b)
        # alternate: first is artist, second is style
        artist2, style2 = normalize_style(a), normalize_style(b)

        art_keywords = {
            "renaissance", "baroque", "romanticism", "impressionism", "realism",
            "surrealism", "cubism", "minimalism", "expressionism", "ukiyo e",
            "high renaissance", "post impressionism", "northern renaissance",
            "abstract expressionism", "rococo", "symbolism",
        }

        if style1 in art_keywords:
            return artist1, style1
        if style2 in art_keywords:
            return artist2, style2

        return artist2, style2

    return "unknown_artist", "unknown_style"


def open_image_from_zip(zf: zipfile.ZipFile, member_name: str) -> Optional[Image.Image]:
    try:
        with zf.open(member_name) as f:
            data = f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, zipfile.BadZipFile):
        return None


def top_k_concepts(
    image_emb: np.ndarray,
    vocab_words: Sequence[str],
    vocab_emb: np.ndarray,
    k: int = 9,
    banned: Optional[set] = None,
) -> Tuple[List[str], List[float]]:
    sims = vocab_emb @ image_emb
    idxs = np.argsort(-sims)

    concepts = []
    scores = []
    seen = set()
    banned = banned or set()

    for idx in idxs:
        w = vocab_words[idx]
        if w in seen or w in banned:
            continue
        concepts.append(w)
        scores.append(float(sims[idx]))
        seen.add(w)
        if len(concepts) == k:
            break

    return concepts, scores


def random_permutations_of_concepts(
    concepts: Sequence[str],
    n_permutations: int,
    rng: random.Random,
) -> List[str]:
    concepts = list(dict.fromkeys(concepts))
    lines = []
    for _ in range(n_permutations):
        arr = concepts[:]
        rng.shuffle(arr)
        lines.append(" ".join(arr))
    return lines


def sample_artist_sequences(
    concept_pool: Sequence[str],
    seq_len: int,
    n_samples: int,
    rng: random.Random,
) -> List[str]:
    concept_pool = list(dict.fromkeys(concept_pool))
    if not concept_pool:
        return []

    lines = []
    for _ in range(n_samples):
        if len(concept_pool) <= seq_len:
            sample = concept_pool[:]
        else:
            sample = rng.sample(concept_pool, seq_len)
        rng.shuffle(sample)
        lines.append(" ".join(sample))
    return lines


def maybe_download_wikiart_zip(
    zip_url: Optional[str],
    zip_path: Optional[Path],
    tmp_dir: Path,
) -> Tuple[Path, bool]:
    """
    Returns (path, should_delete_after)
    """
    if zip_path is not None:
        return zip_path, False

    if zip_url is None:
        raise ValueError("You must provide either --wikiart-zip-url or --wikiart-zip-path")

    tmp_zip = tmp_dir / "wikiart_tmp.zip"
    print(f"[INFO] Downloading WikiArt zip to temporary file: {tmp_zip}")
    download_file(zip_url, tmp_zip)
    return tmp_zip, True


def build_artwork_table(
    wikiart_zip_path: Path,
    metadata_map: Dict[str, Dict[str, str]],
    clip_embedder: ClipEmbedder,
    vocab_words: Sequence[str],
    vocab_emb: np.ndarray,
    top_k: int,
    max_images: Optional[int] = None,
    skip_unknown_style: bool = False,
) -> List[ArtworkRecord]:
    records: List[ArtworkRecord] = []

    with zipfile.ZipFile(wikiart_zip_path, "r") as zf:
        members = [m for m in zf.namelist() if is_image_member(m)]
        print(f"[INFO] Found {len(members)} image members in zip")

        if max_images is not None:
            members = members[:max_images]
            print(f"[INFO] Limiting to first {len(members)} images due to --max-images")

        for member in tqdm(members, desc="Processing WikiArt images"):
            artist, style = infer_metadata_from_member(member, metadata_map)
            if skip_unknown_style and style == "unknown_style":
                continue

            img = open_image_from_zip(zf, member)
            if img is None:
                continue

            image_emb = clip_embedder.encode_single_image(img)
            banned = {style} if style != "unknown_style" else set()
            concepts, scores = top_k_concepts(
                image_emb=image_emb,
                vocab_words=vocab_words,
                vocab_emb=vocab_emb,
                k=top_k,
                banned=banned,
            )

            if len(concepts) < top_k:
                continue

            all10 = concepts[:] + ([style] if style != "unknown_style" else [])
            if style == "unknown_style":
                # If style unknown, still keep only the top-k concepts
                all10 = concepts[:]

            artwork_id = slugify(Path(member).stem)
            records.append(
                ArtworkRecord(
                    artwork_id=artwork_id,
                    image_member=member,
                    artist=artist,
                    style=style,
                    top9_concepts=concepts,
                    top9_scores=scores,
                    all10_concepts=all10,
                )
            )

    print(f"[INFO] Created artwork table with {len(records)} rows")
    return records


def records_to_dataframe(records: Sequence[ArtworkRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "artwork_id": r.artwork_id,
                "image_member": r.image_member,
                "artist": r.artist,
                "style": r.style,
                "top9_concepts": r.top9_concepts,
                "top9_scores": r.top9_scores,
                "all10_concepts": r.all10_concepts,
            }
        )
    return pd.DataFrame(rows)


def build_artist_table(artworks_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = artworks_df.groupby("artist")
    for artist, group in grouped:
        concept_counter = Counter()
        for concepts in group["all10_concepts"].tolist():
            for c in concepts:
                concept_counter[c] += 1

        concept_union = sorted(concept_counter.keys())
        rows.append(
            {
                "artist": artist,
                "n_artworks": int(len(group)),
                "concepts_union": concept_union,
                "concept_freqs": dict(concept_counter),
            }
        )
    return pd.DataFrame(rows)


def write_corpora(
    artworks_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    out_dir: Path,
    art_permutations_per_artwork: int,
    cog_samples_per_artist: int,
    cog_seq_len: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    art_lines = []
    for _, row in artworks_df.iterrows():
        concepts = row["all10_concepts"]
        art_lines.extend(
            random_permutations_of_concepts(
                concepts=concepts,
                n_permutations=art_permutations_per_artwork,
                rng=rng,
            )
        )

    cog_lines = []
    for _, row in artists_df.iterrows():
        pool = row["concepts_union"]
        cog_lines.extend(
            sample_artist_sequences(
                concept_pool=pool,
                seq_len=cog_seq_len,
                n_samples=cog_samples_per_artist,
                rng=rng,
            )
        )

    with open(out_dir / "art_corpus.txt", "w", encoding="utf-8") as f:
        for line in art_lines:
            f.write(line + "\n")

    with open(out_dir / "cog_corpus.txt", "w", encoding="utf-8") as f:
        for line in cog_lines:
            f.write(line + "\n")

    print(f"[INFO] Wrote {len(art_lines)} lines to {out_dir / 'art_corpus.txt'}")
    print(f"[INFO] Wrote {len(cog_lines)} lines to {out_dir / 'cog_corpus.txt'}")


def save_dataframes(
    artworks_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    artworks_parquet = out_dir / "artworks.parquet"
    artists_parquet = out_dir / "artist_concepts.parquet"
    artworks_jsonl = out_dir / "artworks.jsonl"
    artists_jsonl = out_dir / "artist_concepts.jsonl"

    try:
        artworks_df.to_parquet(artworks_parquet, index=False)
        artists_df.to_parquet(artists_parquet, index=False)
        print(f"[INFO] Saved {artworks_parquet}")
        print(f"[INFO] Saved {artists_parquet}")
    except Exception as e:
        print(f"[WARN] Parquet save failed ({e}); falling back to JSONL only")

    artworks_df.to_json(artworks_jsonl, orient="records", lines=True, force_ascii=False)
    artists_df.to_json(artists_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved {artworks_jsonl}")
    print(f"[INFO] Saved {artists_jsonl}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wikiart-zip-url", type=str, default=None)
    parser.add_argument("--wikiart-zip-path", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--top-k-concepts", type=int, default=9)
    parser.add_argument("--text-template", type=str, default="a painting of {}")

    parser.add_argument("--semantic-prune-threshold", type=float, default=0.88)
    parser.add_argument("--disable-semantic-prune", action="store_true")

    parser.add_argument("--min-word-len", type=int, default=3)
    parser.add_argument("--max-word-len", type=int, default=24)
    parser.add_argument("--allow-phrases", action="store_true")

    parser.add_argument("--art-permutations-per-artwork", type=int, default=5)
    parser.add_argument("--cog-samples-per-artist", type=int, default=20)
    parser.add_argument("--cog-seq-len", type=int, default=10)

    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-unknown-style", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using device: {args.device}")

    # 1) Load WordNet Core
    vocab_words = read_wordnet_core(
        cache_dir=cache_dir,
        min_len=args.min_word_len,
        max_len=args.max_word_len,
        allow_phrases=args.allow_phrases,
        remove_generic=True,
    )

    # 2) CLIP encode text vocabulary
    clip_embedder = ClipEmbedder(model_name=args.clip_model, device=args.device)
    prompts = build_text_prompts(vocab_words, args.text_template)
    vocab_text_emb = clip_embedder.encode_texts(prompts, batch_size=128, normalize=True)

    # 3) Optional semantic pruning of vocabulary
    if args.disable_semantic_prune:
        shortlist_words = vocab_words
        shortlist_emb = vocab_text_emb
        print(f"[INFO] Semantic pruning disabled; using all {len(shortlist_words)} words")
    else:
        # Heuristic priority: prefer shorter words and non-generic words
        priority = []
        for w in vocab_words:
            p = 0.0
            p += 1.0 / max(len(w), 1)
            if w not in GENERIC_STOPWORDS:
                p += 1.0
            priority.append(p)

        shortlist_words, shortlist_emb = greedy_semantic_prune(
            words=vocab_words,
            embeddings=vocab_text_emb,
            cosine_threshold=args.semantic_prune_threshold,
            priority=priority,
        )

    vocab_out_dir = out_dir / "vocab"
    save_vocab_artifacts(vocab_out_dir, shortlist_words, shortlist_emb)

    # 4) Load metadata
    metadata_map = load_metadata_csv(args.metadata_csv)

    # 5) Temp download or use local zip
    with tempfile.TemporaryDirectory(prefix="wikiart_tmp_") as tmp:
        tmp_dir = Path(tmp)
        wikiart_zip_path, should_delete = maybe_download_wikiart_zip(
            zip_url=args.wikiart_zip_url,
            zip_path=args.wikiart_zip_path,
            tmp_dir=tmp_dir,
        )

        # 6) Build artwork-level table
        records = build_artwork_table(
            wikiart_zip_path=wikiart_zip_path,
            metadata_map=metadata_map,
            clip_embedder=clip_embedder,
            vocab_words=shortlist_words,
            vocab_emb=shortlist_emb,
            top_k=args.top_k_concepts,
            max_images=args.max_images,
            skip_unknown_style=args.skip_unknown_style,
        )

        artworks_df = records_to_dataframe(records)
        if len(artworks_df) == 0:
            raise RuntimeError("No artwork rows were produced. Check zip path/layout/metadata.")

        # 7) Build artist-level table
        artists_df = build_artist_table(artworks_df)

        # 8) Save compact structured outputs
        save_dataframes(artworks_df, artists_df, out_dir)

        # 9) Build corpora for Art and Cognitive Availability models
        write_corpora(
            artworks_df=artworks_df,
            artists_df=artists_df,
            out_dir=out_dir,
            art_permutations_per_artwork=args.art_permutations_per_artwork,
            cog_samples_per_artist=args.cog_samples_per_artist,
            cog_seq_len=args.cog_seq_len,
            seed=args.seed,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()