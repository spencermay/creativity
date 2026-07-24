# Alien Recombination with Ollama

A full reproduction-oriented implementation of **Alien Recombination: Exploring Concept Blends Beyond Human Cognitive Availability in Visual Art** using local, open-weights models served through **Ollama**.

This project replaces the paper’s proprietary components with:

- `gemma3:4b` for:
  - concept-sequence generation
  - artwork plausibility scoring
  - cognitive-availability scoring
  - image-pair novelty judging
- `x/flux2-klein:4b` for:
  - text-to-image generation

Total unique base weights: **8B**

---

## Paper

This repository implements the core pipeline from:

**Hernandez et al. (NeurIPS 2024)**  
*Alien Recombination: Exploring Concept Blends Beyond Human Cognitive Availability in Visual Art*

Core idea:
- represent artworks as concept sets
- model:
  - what concepts plausibly co-occur in artworks
  - what concepts are cognitively available to artists
- generate candidate concept combinations
- rank them by:
  - **artistic fit**
  - **cognitive unavailability**
- visualize the top-ranked combinations as images

The paper’s original implementation uses:
- CLIP for concept extraction
- GPT-2 for the Art model
- GPT-2 for the Cognitive Availability model
- DALL-E for image generation
- GPT-4 for image-based novelty evaluation

This project replaces those with a practical Ollama-based stack.

---

## Model mapping

| Paper component | Original | This project |
|---|---|---|
| Concept extraction | CLIP | Hugging Face CLIP |
| Art model | GPT-2 fine-tuned on artwork concept sequences | Naive Bayes co-occurrence (or optional `gemma3:4b` LoRA) |
| Cognitive Availability model | GPT-2 fine-tuned on artist concept aggregates | Naive Bayes co-occurrence (or optional `gemma3:4b` LoRA) |
| Text-to-image | DALL-E | `x/flux2-klein:4b` |
| Image novelty evaluator | GPT-4 | `gemma3:4b` |

---

## Why this project exists

The paper’s method is reproducible in spirit, but the published setup depends on APIs or models many people do not have access to.

This repository aims to provide a **fully local, open-weights implementation** of the full workflow:

1. preprocess WikiArt into concept sequences
2. train or adapt compact local models
3. generate candidate concept blends
4. perform Alien sampling
5. render images locally
6. evaluate novelty with local models and embedding methods

---

## Repository structure

```text
.
├── README.md
├── preprocess/
│   └── preprocess.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── training/
│   ├── art_model/
│   └── cog_model/
├── inference/
│   ├── generate_sequences.py
│   ├── score_sequences.py
│   ├── alien_sampling.py
│   └── make_images.py
├── evaluation/
│   ├── text_novelty.py
│   ├── image_novelty_embed.py
│   └── image_novelty_vlm.py
├── configs/
├── outputs/
│   ├── sequences/
│   ├── images/
│   └── reports/
└── scripts/
```

Current implemented preprocessing entrypoint:

- `preprocess/preprocess.py`

---

## Full pipeline overview

The project follows the paper’s workflow with local substitutions.

### 1. Concept extraction from WikiArt
For each image:
- extract **9 semantic-level concepts** using CLIP
- append **style** metadata as concept 10

Concepts are constrained to **WordNet Core**, matching the paper.

### 2. Dataset construction
Build two text datasets:

#### Art dataset
- one concept sequence per artwork
- represented as random permutations of the artwork’s concept list

#### Cognitive Availability dataset
- aggregate concepts at the artist level
- sample random subsets/permutations from each artist’s concept pool

### 3. Small-model training
Train or adapt local models for two roles:

#### Art model
Estimates artwork-level plausibility:
$$
P_{\text{art}}(c_i \mid c_0, \dots, c_{i-1})
$$

#### Cognitive Availability model
Estimates artist-level concept availability:
$$
P_{\text{cog}}(c_i \mid c_0, \dots, c_{i-1})
$$

In this implementation, both are based on `gemma3:4b`, typically via:
- separate fine-tunes
- or separate LoRA adapters

### 4. Candidate sequence generation
Given an input seed like:
- `insect`
- `romanticism mountain`
- `impressionism landscape`

the Art model generates many candidate continuations.

Only valid concepts from the constrained vocabulary are allowed.

### 5. Alien sampling
Each candidate sequence is ranked by:
- Art-model plausibility
- Cognitive-Availability-model unavailability

The fused score follows the paper’s weighted-rank idea:
- low Art-model perplexity is preferred
- high Cognitive Availability perplexity is preferred

Parameter $\beta$ controls “alienness”:
- higher $\beta$ emphasizes cognitive unavailability more strongly

### 6. Image generation
Top-ranked concept sequences are turned into prompts and rendered with:
- `x/flux2-klein:4b`

### 7. Evaluation
The paper uses:
- text-based novelty
- image-based novelty by GPT-4
- embedding-based similarity via ResNet

This repository mirrors that with:
- exact text novelty metrics
- local multimodal judging via `gemma3:4b`
- embedding-based image novelty

---

## Core research objective

The goal is not just to generate weird concept combinations.

The goal is to generate combinations that are:

1. **plausible as artworks**
2. **novel relative to artworks in the dataset**
3. **novel relative to any individual artist’s concept space**

This distinction is central to the paper.

The paper defines two novelty views:

### Artwork-level novelty
Given generated concept set $S$ and artwork concept sets $A_i$:
$$
N_{\text{art}} = \min_i |S \setminus A_i|
$$

### Cognitive-availability novelty
Given artist concept sets $B_j$:
$$
N_{\text{cog}} = \min_j |S \setminus B_j|
$$

The important empirical claim is:
- generating unseen artwork-level combinations is relatively easy
- generating **artist-level cognitively unavailable** combinations is much harder

This repository is built to reproduce exactly that distinction.

---

## Models used

## `gemma3:4b`
Used for:
- candidate concept generation
- sequence scoring
- pairwise image novelty evaluation

Suggested usage modes:
- base + LoRA for Art model
- base + separate LoRA for Cognitive Availability model
- zero-shot or instruction-tuned mode for image comparison

## `x/flux2-klein:4b`
Used for:
- text-to-image generation from selected concept prompts

This replaces the paper’s DALL-E step while staying local and within the 8B budget.

---

## Installation

## 1. Python dependencies

Example:

```bash
pip install torch torchvision transformers pillow pandas pyarrow requests tqdm numpy scikit-learn nltk matplotlib seaborn
```

Optional:
```bash
pip install accelerate peft datasets sentencepiece safetensors
```

## 2. Ollama

Install Ollama, then pull the required models:

```bash
ollama pull gemma3:4b
ollama pull x/flux2-klein:4b
```

## 3. WordNet / NLTK
If you use the preprocessing pipeline with NLTK helpers:

```python
import nltk
nltk.download("wordnet")
```

The project also uses **WordNet Core** directly.

---

## Data requirements

You will need:

### WikiArt dataset
A zip or equivalent archive of WikiArt images.

### Metadata CSV
Recommended columns:
- `path` or `filename`
- `artist`
- `style`

### WordNet Core
Used to constrain the concept vocabulary.

---

## Preprocessing

The preprocessing pipeline lives at:

- `preprocess/preprocess.py`

It does the following:

1. downloads WordNet Core
2. filters and normalizes concept vocabulary
3. optionally prunes semantically redundant concepts using CLIP text embeddings
4. iterates over WikiArt from zip
5. extracts top-9 CLIP concepts per image
6. appends style as concept 10
7. writes:
   - artwork-level structured data
   - artist-level concept aggregates
   - `art_corpus.txt`
   - `cog_corpus.txt`

### Example

```bash
python preprocess/preprocess.py \
  --wikiart-zip-path ./wikiart.zip \
  --metadata-csv ./wikiart_metadata.csv \
  --output-dir ./data/processed
```

---

## Training design

The paper fine-tunes two GPT-2 models.  
This project replaces expensive neural fine-tuning with a **Naive Bayes co-occurrence model** that is:
- instant to build (seconds, not hours)
- requires no GPU
- directly estimates concept plausibility from the preprocessed data

### Naive Bayes scoring

Given a concept set $S = \{c_0, \ldots, c_{k-1}\}$, the probability of a candidate concept $v$ is:

$$P(v \mid S) \propto \prod_{i \in S} P(v \mid i)$$

where:

$$P(v \mid i) = \frac{\#\{\text{sets containing both } i \text{ and } v\}}{\#\{\text{sets containing } i\}}$$

Two co-occurrence models are built from the preprocessed data:
- **Art model**: co-occurrence over all artwork concept sets (measures artwork-level plausibility)
- **Cog model**: co-occurrence over per-artist concept pools (measures cognitive availability)

This gives the same functional interface as the paper's GPT-2 models — a perplexity score under each model — but computed in closed form.

### Alternative: LoRA fine-tuning (optional)

For those who want neural scoring, the repository also includes:
- `training/train_art_model.py` — LoRA fine-tune `gemma3:4b` on `art_corpus.txt`
- `training/train_cog_model.py` — LoRA fine-tune `gemma3:4b` on `cog_corpus.txt`

Expect ~2-4 hours per model on a 24GB GPU, or 1-2 days on Apple Silicon via MPS.

---

## Inference workflow

Given a seed sequence, for example:

- `insect`
- `romanticism mountain`
- `impressionism landscape`

the inference pipeline should:

1. sample $N$ candidate sequences from the Art co-occurrence model (Naive Bayes)
2. all generated concepts are guaranteed valid (constrained to vocabulary)
3. score each candidate under:
   - Art model (Naive Bayes co-occurrence, artwork-level)
   - Cognitive Availability model (Naive Bayes co-occurrence, artist-level)
4. rank by weighted aggregation
5. return top-$k$
6. generate images using FLUX

---

## Prompting strategy

The paper uses short concept prompts for image generation:

> A painting that contains the concepts: `<input sequence + generated sequence>`

This repository should preserve that baseline for fidelity.

### Example FLUX prompt
```text
A painting that contains the concepts: romanticism mountain jungle dragon classical wilderness climb
```

If style is present, the prompt can be adjusted:

```text
A romanticism painting that contains the concepts: mountain jungle dragon classical wilderness climb
```

---

## Evaluation plan

This project aims to reproduce all three evaluation layers.

## 1. Text-based novelty
Compute:
- $N_{\text{art}}$
- $N_{\text{cog}}$

using the processed concept tables.

## 2. Image-based embedding novelty
The paper uses ResNet152 embeddings against WikiArt.

This repository can reproduce that approximately by:
- training or fine-tuning a style classifier on WikiArt
- extracting image embeddings
- comparing generated images to nearest WikiArt neighbors

Metrics:
- average max cosine similarity to WikiArt
- pairwise win counts between methods

## 3. Vision-language image comparison
The paper uses GPT-4 with the instruction:

> As an art expert, please write a sentence indicating which image is more novel, focusing on concept combination novelty.

This repository replaces that with `gemma3:4b` in multimodal mode, using a comparable evaluation prompt.

---

## Baselines

The paper compares Alien sampling to a baseline:

### Baseline
- generate sequences with the Art model
- select sequences by random sampling

### Alien Recombination
- generate sequences with the Art model
- rank by art plausibility + cognitive unavailability

This project should preserve both conditions for comparison.

---

## Reproduction targets

A successful reproduction should be able to show:

1. increasing temperature generates more artwork-level novelty
2. temperature alone does not reliably generate artist-level cognitive novelty
3. Alien sampling increases $N_{\text{cog}}$
4. Alien-generated images are judged more novel on average than baseline images

---

## Example end-to-end workflow

### Step 1: preprocess
```bash
python preprocess/preprocess.py \
  --wikiart-zip-path ./wikiart.zip \
  --metadata-csv ./wikiart_metadata.csv \
  --output-dir ./data/processed
```

### Step 2: build Naive Bayes co-occurrence models
```bash
python inference/naive_bayes_model.py build \
  --artworks ./data/processed/artworks.parquet \
  --artists ./data/processed/artist_concepts.parquet \
  --output-dir ./data/processed/cooccurrence
```

### Step 3: generate candidates and rank with Alien sampling
```bash
python inference/alien_sampling.py \
  --seed-sequence "romanticism mountain" \
  --vocab ./data/processed/vocab/vocab_words.json \
  --artworks ./data/processed/artworks.parquet \
  --artists ./data/processed/artist_concepts.parquet \
  --beta 0.85 \
  --temperature 2.5 \
  --num-candidates 150 \
  --top-k 5 \
  --output ./outputs/sequences/romanticism_mountain.json
```

No LLM or GPU needed — generation samples directly from the co-occurrence distribution,
and scoring is instant matrix lookups.

### Step 4: generate images
```bash
python inference/make_images.py \
  --sequence-file ./outputs/sequences/romanticism_mountain.json \
  --output-dir ./outputs/images/romanticism_mountain
```

### Step 6: evaluate
```bash
python evaluation/text_novelty.py \
  --sequence-file ./outputs/sequences/romanticism_mountain.json \
  --artworks ./data/processed/artworks.parquet \
  --artists ./data/processed/artist_concepts.parquet
```

---

## Current status

### Implemented
- preprocessing pipeline in `preprocess/preprocess.py`
- **Naive Bayes co-occurrence scoring** in `inference/naive_bayes_model.py` (replaces LoRA fine-tuning)
- Art model LoRA training in `training/train_art_model.py` (optional, for neural comparison)
- Cognitive Availability model LoRA training in `training/train_cog_model.py` (optional)
- constrained candidate generation in `inference/generate_sequences.py`
- sequence scoring in `inference/score_sequences.py` (Ollama-based, optional)
- weighted-rank Alien sampling in `inference/alien_sampling.py`
- FLUX image generation wrapper in `inference/make_images.py`
- text novelty evaluation (N_art, N_cog) in `evaluation/text_novelty.py`
- embedding-based image novelty in `evaluation/image_novelty_embed.py`
- VLM pairwise image judging in `evaluation/image_novelty_vlm.py`
- experiment orchestration in `scripts/run_experiment.py`
- project config in `configs/default.yaml` and `configs/experiment.yaml`

### Remaining
- run preprocessing on actual WikiArt archive
- train Art and Cog adapters
- run full experiment sweep
- generate and compare results

---

## Design choices vs the paper

This repository aims for fidelity, but not literal architectural identity.

### Same
- CLIP-based concept extraction
- WordNet Core-constrained concepts
- artwork-level and artist-level datasets
- candidate generation from art-distribution model
- dual ranking with $\beta$ control
- text-to-image visualization
- novelty evaluation at text and image level

### Different
- Naive Bayes co-occurrence instead of GPT-2 for scoring (fast, no training needed)
- `gemma3:4b` instead of GPT-2 for candidate generation
- `x/flux2-klein:4b` instead of DALL-E
- `gemma3:4b` instead of GPT-4 for VLM evaluation
- optional LoRA/adapters available for those who want neural scoring

These substitutions are made for local reproducibility and open-weights access.

---

## Limitations

- WikiArt metadata and archive layout vary
- CLIP concept extraction is noisy
- semantic pruning of WordNet Core is an approximation
- FLUX outputs may differ substantially from DALL-E outputs
- Gemma-based perplexity proxies may differ from GPT-2 scoring behavior
- image novelty remains difficult to evaluate robustly

---

## Ethics and bias

This project inherits the same concerns discussed in the paper:

- WikiArt is historically and culturally biased
- CLIP has representational biases
- image generators can revert to stereotyped outputs
- “cognitive unavailability” is defined relative to the dataset, not reality

A concept may be marked “alien” because it is missing from the dataset, not because it is implausible or genuinely absent from human culture.

---

## Citation

If you use this repository, please cite the original paper:

```bibtex
@inproceedings{hernandez2024alien,
  title={Alien Recombination: Exploring Concept Blends Beyond Human Cognitive Availability in Visual Art},
  author={Hernandez, Alejandro and Brinkmann, Levin and Serna, Ignacio and Rahaman, Nasim and Abu Alhaija, Hassan and Yakura, Hiromu and Canet Sola, Mar and Sch{\"o}lkopf, Bernhard and Rahwan, Iyad},
  booktitle={NeurIPS},
  year={2024}
}
```

---

# TODO

## Project setup
- [ ] create full repository skeleton
- [ ] add `requirements.txt`
- [ ] add `pyproject.toml`
- [ ] add `.gitignore`
- [ ] add config files for local paths and model names
- [ ] add reproducible random seed handling across scripts

## Data and preprocessing
- [x] implement `preprocess/preprocess.py`
- [ ] verify preprocessing on the exact WikiArt archive used in experiments
- [ ] add metadata normalization script for WikiArt filenames, styles, and artists
- [ ] add preprocessing smoke-test mode with sample outputs
- [ ] cache CLIP text embeddings for faster reruns
- [ ] add duplicate-image / corrupt-image handling report
- [ ] add dataset statistics report:
  - [ ] number of artworks
  - [ ] number of artists
  - [ ] style counts
  - [ ] concept frequency histogram
- [ ] add nearest-neighbor inspection tool for extracted concepts
- [ ] validate top-9 concept quality on a hand-checked sample

## Vocabulary and concept extraction
- [ ] add explicit downloader for WordNet Core
- [ ] add option to disable semantic pruning
- [ ] benchmark different CLIP prompt templates:
  - [ ] `a painting of {}`
  - [ ] `an artwork containing {}`
  - [ ] bare token
- [ ] benchmark different semantic-pruning thresholds
- [ ] add stoplist expansion for overly generic concepts
- [ ] add optional phrase support for multiword concepts
- [ ] compare filtered WordNet Core vs raw WordNet Core

## Art and Cognitive Availability training
- [ ] implement `training/train_art_model.py`
- [ ] implement `training/train_cog_model.py`
- [ ] add LoRA-based Gemma fine-tuning configs
- [ ] support separate Art and Cog adapters on the same `gemma3:4b` base
- [ ] add validation split creation
- [ ] add training loss logging
- [ ] add checkpoint saving and resume logic
- [ ] add model card / training summary output
- [ ] test whether exact token-level logprobs are accessible in the chosen inference path
- [ ] if not, implement a stable ranking proxy compatible with Gemma

## Candidate generation
- [ ] implement constrained sequence generation from the Art model
- [ ] enforce vocabulary membership in generated sequences
- [ ] reject non-WordNet-Core tokens
- [ ] reject duplicates if required
- [ ] continue sampling until enough valid candidates are obtained
- [ ] expose generation hyperparameters:
  - [ ] temperature
  - [ ] top-k
  - [ ] top-p
  - [ ] max tokens
  - [ ] number of candidates

## Alien sampling
- [ ] implement Art-model perplexity/ranking
- [ ] implement Cognitive Availability perplexity/ranking
- [ ] implement inverse-rank handling for cognitive unavailability
- [ ] implement weighted rank aggregation with $\beta$
- [ ] support top-$k$ output sequences
- [ ] write `inference/alien_sampling.py`
- [ ] add baseline random-sampling selector for direct comparison

## Image generation
- [ ] implement `inference/make_images.py`
- [ ] add Ollama wrapper for `x/flux2-klein:4b`
- [ ] standardize prompt templates for FLUX
- [ ] support style-aware prompt rewriting
- [ ] support multiple images per selected sequence
- [ ] store prompt, seed, temperature, and model metadata with each output
- [ ] add retry logic for failed image generations

## Text evaluation
- [ ] implement `evaluation/text_novelty.py`
- [ ] compute $N_{\text{art}}$
- [ ] compute $N_{\text{cog}}$
- [ ] aggregate metrics by:
  - [ ] temperature
  - [ ] $\beta$
  - [ ] input seed
- [ ] reproduce the paper’s “art-level novelty easy, artist-level novelty hard” plots

## Image evaluation
- [ ] implement embedding-based image novelty evaluation
- [ ] choose and train reproduction-friendly ResNet style classifier
- [ ] embed all WikiArt images
- [ ] compute nearest WikiArt cosine similarity for each generated image
- [ ] implement pairwise comparison counts between baseline and Alien images
- [ ] generate comparison plots across temperatures

## Vision-language evaluation
- [ ] implement pairwise image judging with `gemma3:4b`
- [ ] create prompt template matching the paper’s GPT-4 instruction
- [ ] support A/B randomized image ordering
- [ ] parse model responses into win/loss labels
- [ ] measure judge consistency across repeated prompts
- [ ] compare local VLM judgments with embedding-based judgments

## Experiment reproduction
- [ ] reproduce 50-input experiment from the paper
- [ ] support 1-concept and 2-concept input seeds
- [ ] generate 150 sequences per temperature
- [ ] scan temperatures from 0.1 to 3.1 in steps of 0.3
- [ ] scan multiple $\beta$ values
- [ ] reproduce baseline-vs-alien tables
- [ ] reproduce figure-style plots for:
  - [ ] $N_{\text{art}}$ vs temperature
  - [ ] $N_{\text{cog}}$ vs temperature
  - [ ] image novelty wins vs temperature
  - [ ] average nearest-image cosine similarity

## Reporting and analysis
- [ ] save all experiment runs as structured JSON/CSV
- [ ] add notebook or script for report generation
- [ ] add HTML or Markdown experiment summaries
- [ ] store selected top sequences and generated images together
- [ ] add side-by-side galleries for baseline vs Alien outputs
- [ ] add qualitative examples matching the paper’s figures

## Reliability and usability
- [ ] add command-line interfaces for all stages
- [ ] add logging to file
- [ ] add progress bars everywhere appropriate
- [ ] add dry-run mode
- [ ] add unit tests for:
  - [ ] concept extraction
  - [ ] ranking
  - [ ] novelty metrics
  - [ ] prompt construction
- [ ] add integration test for one tiny end-to-end pipeline run

## Stretch goals
- [ ] compare `gemma3:4b` against another sub-8B open model for text roles
- [ ] compare FLUX prompt variants for style adherence
- [ ] add exact constrained decoding over concept vocabulary
- [ ] build a small web UI for entering seed concepts and browsing outputs
- [ ] support interactive $\beta$ and temperature sweeps
- [ ] add human evaluation protocol for concept novelty
- [ ] test whether a single multitask adapter can replace separate Art and Cog adapters

---

## Development priorities

Suggested implementation order:

1. preprocessing validation
2. Art and Cog model training
3. constrained candidate generation
4. Alien sampling
5. FLUX rendering
6. text novelty evaluation
7. image novelty evaluation
8. experiment-scale automation

---

## Short project summary

This repository is a local, open-weights implementation of Alien Recombination using:

- `gemma3:4b` for text and multimodal reasoning
- `x/flux2-klein:4b` for image generation

It aims to reproduce the paper’s central claim:
that **explicitly searching for cognitively unavailable but artistically plausible concept combinations** produces more novel visual art prompts than temperature scaling or random sampling alone.