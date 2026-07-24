"""
Generate images from selected concept sequences using Hugging Face diffusers.

Uses Stable Diffusion XL with sequential CPU offloading to fit within 8GB RAM
on Apple Silicon. Falls back to SD 1.5 if SDXL can't fit.

Usage:
    python inference/make_images.py \
        --sequence-file ./outputs/sequences/romanticism_mountain.json \
        --output-dir ./outputs/images/romanticism_mountain

    # Single prompt:
    python inference/make_images.py \
        --prompt "A painting that contains the concepts: mountain castle dragon wilderness" \
        --output-dir ./outputs/images/test

    # Use a specific model:
    python inference/make_images.py \
        --prompt "impressionist painting of a lawn with benches" \
        --output-dir ./outputs/images/test \
        --model stabilityai/stable-diffusion-xl-base-1.0
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


KNOWN_STYLES = {
    "renaissance", "baroque", "romanticism", "impressionism", "realism",
    "surrealism", "cubism", "minimalism", "expressionism", "rococo",
    "symbolism", "abstract", "modernism", "post impressionism",
    "northern renaissance", "high renaissance", "ukiyo e",
    "abstract expressionism", "art nouveau", "mannerism",
}

# Models ordered by memory requirement (smallest first)
DEFAULT_MODELS = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",  # SD 1.5: ~1.7GB UNet, fast on 8GB
    "stabilityai/sdxl-turbo",                         # SDXL Turbo: better quality, more memory
    "stabilityai/stable-diffusion-xl-base-1.0",       # SDXL: best quality, most memory
]


def build_prompt(concepts: list[str], style: Optional[str] = None) -> str:
    """
    Build an image generation prompt that encourages all concepts to appear.

    Uses an explicit enumerative structure rather than a flat list, which
    helps diffusion models attend to each concept individually.
    """
    style_concept = None
    content_concepts = concepts[:]

    if concepts and concepts[0].lower() in KNOWN_STYLES:
        style_concept = concepts[0]
        content_concepts = concepts[1:]
    elif concepts and concepts[-1].lower() in KNOWN_STYLES:
        style_concept = concepts[-1]
        content_concepts = concepts[:-1]

    if style:
        style_concept = style

    # Build a structured prompt that names each concept explicitly
    style_prefix = f"A {style_concept} painting" if style_concept else "A painting"
    concept_phrases = ", ".join(content_concepts[:-1])
    if len(content_concepts) > 1:
        concept_phrases += f", and {content_concepts[-1]}"
    else:
        concept_phrases = content_concepts[0] if content_concepts else ""

    return f"{style_prefix} depicting: {concept_phrases}."


def load_pipeline(model_id: str, device: str):
    """
    Load a diffusion pipeline with memory optimizations for 8GB systems.

    Uses sequential CPU offloading: only one layer is on GPU at a time,
    so total GPU memory usage is minimal.
    """
    from diffusers import (
        AutoPipelineForText2Image,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    print(f"[INFO] Loading pipeline: {model_id}")
    print(f"[INFO] Device: {device}")

    # Use float16 — necessary to fit in 8GB
    dtype = torch.float16

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

    # Memory optimization for 8GB systems:
    # - attention_slicing("auto") lets diffusers pick optimal slice size
    # - On MPS: use float32 to avoid NaN issues with fp16 guidance
    # - On CUDA: use model-level CPU offload
    pipe.enable_attention_slicing("auto")
    # # --------------------
    # if hasattr(pipe, "vae"):
    #     # Force VAE to float32 to prevent half/float precision mismatch errors
    #     pipe.vae.to(dtype=torch.float32)
        
    #     # Patch the VAE forward pass to auto-convert inputs if needed
    #     original_decode = pipe.vae.decode
    #     def patched_decode(latents, *args, **kwargs):
    #         return original_decode(latents.to(dtype=torch.float32), *args, **kwargs)
    #     pipe.vae.decode = patched_decode
    
    # if hasattr(pipe, "unet"):
    #     pipe.unet.to(dtype=torch.float16)  # Keep the main UNet fast in fp16
        
    #     # Specifically upcast attention mechanics if your script doesn't use xformers
    #     pipe.enable_attention_slicing() 
    # # --------------------

    if device == "mps":
        pipe = pipe.to(device)
    elif device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cpu")

    print(f"[INFO] Pipeline loaded successfully")
    return pipe


def generate_image(
    pipe,
    concepts: list[str],
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
) -> Optional[tuple["Image", str]]:
    """
    Generate a single image, retrying with permuted concept order on failure.

    Returns (image, prompt_used) on success, None on failure.
    Tries up to 3 permutations of the concept list — each permutation produces
    a different text encoding which avoids prompt-specific MPS NaN issues.
    """
    from PIL import Image
    import numpy as np
    import random

    max_permutations = 3
    for attempt in range(max_permutations):
        # Permute concepts on retry (first attempt uses original order)
        if attempt > 0:
            concepts = concepts.copy()
            random.shuffle(concepts)

        prompt = build_prompt(concepts)
        current_seed = (seed * 31 + attempt * 1337 + 7919) if seed is not None else None
        generator = None
        if current_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(current_seed)

        try:
            with torch.no_grad():
                # # --- PLACE INSIDE THE LOOP, RIGHT BEFORE RUNNING THE PIPELINE ---
                # if hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "set_timesteps"):
                #     pipe.scheduler.set_timesteps(steps) # Match your script's step variable
                # # ----------------------------------------------------------------

                result = pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    feature_extractor=None,
                )
            img = result.images[0]

            # Check for NaN-corrupted output (common MPS issue)
            img_array = np.array(img)
            if img_array.max() == 0 or img_array.std() < 1.0:
                print(f"[WARN] Corrupt image (attempt {attempt + 1}), permuting concepts...")
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                gc.collect()
                continue

            return img, prompt
        except Exception as e:
            print(f"[ERROR] Generation failed (attempt {attempt + 1}): {e}")
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            gc.collect()
            continue

    return None


def process_sequence_file(
    sequence_file: Path,
    output_dir: Path,
    pipe,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    num_images_per_sequence: int = 1,
    num_output: int = 5,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Process sequences from an alien sampling output file.

    If a sequence fails after permutation retries, skips to the next ranked
    candidate rather than wasting time. Continues until num_output images
    are successfully generated.
    """
    with open(sequence_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use all_ranked if available (more candidates to fall through)
    candidates = data.get("all_ranked", data.get("selected", data.get("candidates", [])))
    if not candidates:
        print("[ERROR] No sequences found in input file")
        return []

    print(f"[INFO] Target: {num_output} images (from {len(candidates)} candidates)")

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    output_idx = 0

    for i, entry in enumerate(candidates):
        if output_idx >= num_output:
            break

        if isinstance(entry, list):
            concepts = entry
        elif isinstance(entry, dict):
            concepts = entry.get("sequence", entry.get("concepts", []))
        else:
            continue

        if not concepts:
            continue

        img_seed = (seed + i * 100) if seed is not None else None
        result = generate_image(
            pipe, concepts,
            width=width, height=height, steps=steps,
            seed=img_seed, guidance_scale=guidance_scale,
        )

        if result:
            image, prompt_used = result
            filename = f"alien_{output_idx:03d}.png"
            output_path = output_dir / filename
            image.save(output_path)

            results.append({
                "output_idx": output_idx,
                "candidate_rank": i,
                "concepts": concepts,
                "prompt": prompt_used,
                "seed": img_seed,
                "output_path": str(output_path),
                "success": True,
            })
            output_idx += 1
            print(f"[INFO] [{output_idx}/{num_output}] Saved {filename}")
        else:
            print(f"[INFO] Skipping candidate #{i} (all permutations failed), trying next...")

        # Free memory between sequences
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"[INFO] Generated {len(results)}/{num_output} images successfully")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate images from concept sequences")
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (direct mode)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="HuggingFace model ID")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30,
                        help="Inference steps (30 for SD 1.5 no-guidance; 20 with guidance on CUDA)")
    parser.add_argument("--guidance-scale", type=float, default=0.0,
                        help="CFG scale (0.0 for MPS stability; 7.5 for CUDA with SD 1.5)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    img_seed = args.seed if args.seed is not None else config["seed"]

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Size: {args.width}x{args.height}, Steps: {args.steps}")

    if args.dry_run:
        if args.prompt:
            print(f"[DRY RUN] Prompt: {args.prompt}")
        elif args.sequence_file:
            with open(args.sequence_file) as f:
                data = json.load(f)
            selected = data.get("selected", data.get("candidates", []))
            print(f"[DRY RUN] Would render {len(selected)} sequences")
            for i, entry in enumerate(selected[:5]):
                concepts = entry.get("sequence", entry) if isinstance(entry, dict) else entry
                print(f"  #{i+1}: {build_prompt(concepts)}")
        return

    # Load pipeline
    pipe = load_pipeline(args.model, device)

    if args.prompt:
        # Direct single-prompt mode — parse concepts from the prompt
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Treat the prompt as a concept list for generate_image
        concepts = args.prompt.split()
        result = generate_image(
            pipe, concepts,
            width=args.width, height=args.height, steps=args.steps,
            seed=img_seed, guidance_scale=args.guidance_scale,
        )
        if result:
            image, prompt_used = result
            output_path = args.output_dir / "direct_00.png"
            image.save(output_path)
            print(f"[INFO] Saved: {output_path}")
            print(f"[INFO] Prompt: {prompt_used}")
            results = [{"success": True, "path": str(output_path), "prompt": prompt_used}]
        else:
            print("[ERROR] Generation failed")
            results = [{"success": False, "path": None}]

    elif args.sequence_file:
        results = process_sequence_file(
            sequence_file=args.sequence_file,
            output_dir=args.output_dir,
            pipe=pipe,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_output=args.num_images,
            seed=img_seed,
        )
    else:
        parser.error("Provide either --sequence-file or --prompt")
        return

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "model": args.model,
        "device": device,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": img_seed,
        "results": results,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Manifest saved to {manifest_path}")

    # Cleanup
    del pipe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
