# Preprocessing instructions

## Install

```bash
pip install torch torchvision transformers pillow pandas pyarrow requests tqdm
```

## Example usage

### If you have a local WikiArt zip
```bash
python preprocess_alien_recombination.py \
  --wikiart-zip-path ./wikiart.zip \
  --metadata-csv ./wikiart_metadata.csv \
  --output-dir ./alien_data
```

### If you want temporary download
```bash
python preprocess_alien_recombination.py \
  --wikiart-zip-url "https://example.com/wikiart.zip" \
  --metadata-csv ./wikiart_metadata.csv \
  --output-dir ./alien_data
```

### Faster smoke test
```bash
python preprocess_alien_recombination.py \
  --wikiart-zip-path ./wikiart.zip \
  --metadata-csv ./wikiart_metadata.csv \
  --output-dir ./alien_data_test \
  --max-images 500
```

## Expected outputs

Inside `./alien_data/`:

- `vocab/vocab_words.json`
- `vocab/vocab_embeddings.npy`
- `artworks.parquet`
- `artist_concepts.parquet`
- `artworks.jsonl`
- `artist_concepts.jsonl`
- `art_corpus.txt`
- `cog_corpus.txt`

## Expected metadata CSV format

At minimum:

| image_member or filename or path | artist | style |
|---|---|---|
| `Impressionism/Claude_Monet/foo.jpg` | `Claude Monet` | `Impressionism` |

For example:

```csv
path,artist,style
Impressionism/Claude_Monet/water_lilies.jpg,Claude Monet,Impressionism
Romanticism/Caspar_David_Friedrich/wanderer.jpg,Caspar David Friedrich,Romanticism
```

## Notes on fidelity to the paper

This script matches the paper’s preprocessing idea closely:

- top-9 CLIP concepts per artwork
- style appended as concept 10
- concepts constrained to WordNet Core
- artwork-level concept sequences for the Art dataset
- artist-level aggregated concept sequences for the Cognitive Availability dataset

The one deliberate addition is:

- semantic de-duplication of WordNet Core using CLIP text embedding similarity

which you requested.

## Recommended defaults

Good starting settings:

- `--clip-model openai/clip-vit-base-patch32`
- `--semantic-prune-threshold 0.88`

If the concept shortlist feels too small:
- increase threshold to `0.90`

If it feels too redundant:
- decrease to `0.85`

## Important caveat

The archive structure of WikiArt zips differs across mirrors. If your zip layout is unusual, the fallback path-based artist/style inference may be wrong. In practice, you should pass a metadata CSV whenever possible.