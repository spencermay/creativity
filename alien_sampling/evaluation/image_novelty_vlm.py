"""
Vision-language model (VLM) pairwise image novelty evaluation.

Replaces the paper's GPT-4 image comparison with gemma3:4b in multimodal mode.

The paper uses this instruction:
    "As an art expert, please write a sentence indicating which image is more
     novel, focusing on concept combination novelty."

This script:
1. Takes pairs of images (alien vs baseline)
2. Randomizes A/B ordering to avoid position bias
3. Prompts gemma3:4b with both images + the evaluation prompt
4. Parses the response for a winner (A or B)
5. Aggregates win/loss statistics

Usage:
    python evaluation/image_novelty_vlm.py \
        --alien-dir ./outputs/images/alien/ \
        --baseline-dir ./outputs/images/baseline/ \
        --output ./outputs/reports/vlm_comparison.json \
        --config ./configs/default.yaml

    # Or compare specific image pairs from a manifest:
    python evaluation/image_novelty_vlm.py \
        --pairs-file ./outputs/reports/pairs.json \
        --output ./outputs/reports/vlm_comparison.json
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import ollama
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_image_paths(directory: Path) -> list[Path]:
    """Find all image files in a directory, sorted."""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(directory.rglob(f"*{ext}"))
    paths.sort()
    return paths


def image_to_base64(image_path: Path) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_comparison_prompt(config: dict) -> str:
    """Get the VLM comparison prompt from config."""
    return config["evaluation"]["vlm_comparison_prompt"]


def judge_pair(
    model: str,
    image_a_path: Path,
    image_b_path: Path,
    prompt: str,
) -> Optional[str]:
    """
    Ask the VLM to judge which image (A or B) is more novel.

    Returns "A", "B", or None if parsing fails.
    """
    img_a_b64 = image_to_base64(image_a_path)
    img_b_b64 = image_to_base64(image_b_path)

    full_prompt = (
        f"You are comparing two paintings for concept combination novelty.\n\n"
        f"Image A is the first image. Image B is the second image.\n\n"
        f"{prompt}\n\n"
        f"Reply with exactly 'A' or 'B' as the first character of your response, "
        f"followed by a brief explanation."
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                    "images": [img_a_b64, img_b_b64],
                }
            ],
            options={"temperature": 0.1},
        )

        response_text = response["message"]["content"].strip()
        return parse_winner(response_text)

    except Exception as e:
        print(f"[WARN] VLM judging failed: {e}")
        return None


def parse_winner(response: str) -> Optional[str]:
    """
    Parse the VLM response to extract a winner (A or B).

    Tries multiple patterns:
    - First character is A or B
    - "Image A" or "Image B" appears
    - "A is more" or "B is more" appears
    """
    if not response:
        return None

    text = response.strip()

    # Check first character
    if text[0].upper() in ("A", "B"):
        return text[0].upper()

    # Check for "Image A" or "Image B" patterns
    text_upper = text.upper()
    if "IMAGE A" in text_upper and "IMAGE B" not in text_upper:
        return "A"
    if "IMAGE B" in text_upper and "IMAGE A" not in text_upper:
        return "B"

    # Check which appears first
    pos_a = text_upper.find("IMAGE A")
    pos_b = text_upper.find("IMAGE B")

    if pos_a >= 0 and pos_b >= 0:
        # Look for "more novel" near each mention
        # Simple heuristic: first mention of A/B followed by positive language
        pass

    # Look for patterns like "A is more novel" or "B is more novel"
    match = re.search(r"\b([AB])\b\s+is\s+more", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Last resort: count mentions
    count_a = text_upper.count(" A ") + text_upper.count("(A)")
    count_b = text_upper.count(" B ") + text_upper.count("(B)")
    if count_a > count_b:
        return "A"
    if count_b > count_a:
        return "B"

    return None


def run_pairwise_evaluation(
    alien_paths: list[Path],
    baseline_paths: list[Path],
    model: str,
    prompt: str,
    randomize_order: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Run pairwise VLM comparisons between alien and baseline images.

    For each pair, randomizes A/B position to avoid order bias,
    then maps the judgment back to alien/baseline.
    """
    rng = random.Random(seed)
    n_pairs = min(len(alien_paths), len(baseline_paths))

    results = []

    for i in tqdm(range(n_pairs), desc="VLM pairwise judging"):
        alien_path = alien_paths[i]
        baseline_path = baseline_paths[i]

        # Randomize order
        if randomize_order and rng.random() < 0.5:
            image_a = baseline_path
            image_b = alien_path
            alien_position = "B"
        else:
            image_a = alien_path
            image_b = baseline_path
            alien_position = "A"

        # Judge
        winner = judge_pair(model, image_a, image_b, prompt)

        # Map back to alien/baseline
        if winner is None:
            alien_won = None
        elif winner == alien_position:
            alien_won = True
        else:
            alien_won = False

        results.append({
            "pair_idx": i,
            "alien_image": str(alien_path.name),
            "baseline_image": str(baseline_path.name),
            "alien_position": alien_position,
            "raw_winner": winner,
            "alien_won": alien_won,
        })

    return results


def compute_vlm_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics from pairwise results."""
    valid = [r for r in results if r["alien_won"] is not None]
    alien_wins = sum(1 for r in valid if r["alien_won"])
    baseline_wins = sum(1 for r in valid if not r["alien_won"])
    invalid = len(results) - len(valid)

    return {
        "total_pairs": len(results),
        "valid_judgments": len(valid),
        "invalid_judgments": invalid,
        "alien_wins": alien_wins,
        "baseline_wins": baseline_wins,
        "alien_win_rate": alien_wins / max(len(valid), 1),
        "baseline_win_rate": baseline_wins / max(len(valid), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="VLM pairwise image novelty judging")
    parser.add_argument("--alien-dir", type=Path, default=None, help="Directory of alien images")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Directory of baseline images")
    parser.add_argument("--pairs-file", type=Path, default=None, help="JSON file specifying pairs")
    parser.add_argument("--output", type=Path, required=True, help="Output report JSON")
    parser.add_argument("--model", type=str, default=None, help="Override Ollama VLM model")
    parser.add_argument("--no-randomize", action="store_true", help="Disable A/B order randomization")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    model = args.model or config["models"]["ollama_text_model"]
    prompt = build_comparison_prompt(config)

    print(f"[INFO] VLM model: {model}")
    print(f"[INFO] Randomize order: {not args.no_randomize}")

    if args.pairs_file:
        # Load pre-specified pairs
        with open(args.pairs_file, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)

        alien_paths = [Path(p["alien"]) for p in pairs_data["pairs"]]
        baseline_paths = [Path(p["baseline"]) for p in pairs_data["pairs"]]

    elif args.alien_dir and args.baseline_dir:
        alien_paths = get_image_paths(args.alien_dir)
        baseline_paths = get_image_paths(args.baseline_dir)
        print(f"[INFO] Alien images: {len(alien_paths)}")
        print(f"[INFO] Baseline images: {len(baseline_paths)}")

    else:
        parser.error("Provide either --alien-dir + --baseline-dir, or --pairs-file")
        return

    if not alien_paths or not baseline_paths:
        print("[ERROR] No images found")
        return

    # Run evaluation
    results = run_pairwise_evaluation(
        alien_paths=alien_paths,
        baseline_paths=baseline_paths,
        model=model,
        prompt=prompt,
        randomize_order=not args.no_randomize,
        seed=config["seed"],
    )

    summary = compute_vlm_summary(results)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": model,
        "prompt": prompt,
        "randomized_order": not args.no_randomize,
        "summary": summary,
        "pairwise_results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"VLM PAIRWISE NOVELTY EVALUATION")
    print(f"{'='*60}")
    print(f"Total pairs:        {summary['total_pairs']}")
    print(f"Valid judgments:     {summary['valid_judgments']}")
    print(f"Invalid judgments:   {summary['invalid_judgments']}")
    print(f"\nResults:")
    print(f"  Alien wins:       {summary['alien_wins']} ({summary['alien_win_rate']*100:.1f}%)")
    print(f"  Baseline wins:    {summary['baseline_wins']} ({summary['baseline_win_rate']*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\n[INFO] Report saved to {args.output}")


if __name__ == "__main__":
    main()
