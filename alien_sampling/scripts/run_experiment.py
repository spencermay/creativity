"""
Run a full experiment sweep reproducing the paper's evaluation protocol.

This script orchestrates:
1. Generate candidates at multiple temperatures
2. Score all candidates
3. Apply Alien sampling at multiple beta values
4. Generate images for selected sequences
5. Evaluate text novelty
6. Evaluate image novelty (embedding + VLM)

Usage:
    python scripts/run_experiment.py \
        --config ./configs/experiment.yaml \
        --output-dir ./outputs/experiment_run_01

    # Dry run (print plan without executing):
    python scripts/run_experiment.py \
        --config ./configs/experiment.yaml \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


def run_command(cmd: list[str], dry_run: bool = False) -> int:
    """Run a subprocess command, or print it in dry-run mode."""
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return 0
    else:
        print(f"  [RUN] {cmd_str}")
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full Alien Recombination experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    seeds = exp_cfg.get("seeds", ["insect"])
    temperatures = exp_cfg.get("temperatures", [2.5])
    betas = exp_cfg.get("betas", [0.85])
    num_candidates = exp_cfg.get("num_candidates_per_temperature", 150)
    top_k = exp_cfg.get("top_k_per_condition", 5)

    output_dir = args.output_dir or Path(config["paths"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    vocab_path = Path(config["paths"]["vocab_words"])
    artworks_path = Path(config["paths"]["artworks_parquet"])
    artists_path = Path(config["paths"]["artists_parquet"])

    # Save experiment plan
    plan = {
        "name": exp_name,
        "timestamp": timestamp,
        "config_file": str(args.config),
        "seeds": seeds,
        "temperatures": temperatures,
        "betas": betas,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "total_conditions": len(seeds) * len(temperatures) * len(betas),
    }

    plan_path = output_dir / "experiment_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"{'='*60}")
    print(f"ALIEN RECOMBINATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Name: {exp_name}")
    print(f"Seeds: {len(seeds)}")
    print(f"Temperatures: {temperatures}")
    print(f"Betas: {betas}")
    print(f"Total conditions: {plan['total_conditions']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Phase 1: Generate and score candidates for each seed x temperature
    if not args.skip_generation:
        print("\n[PHASE 1] Generating and scoring candidates...\n")
        for seed_seq in seeds:
            seed_slug = seed_seq.replace(" ", "_")
            for temp in temperatures:
                cand_dir = output_dir / "candidates" / seed_slug
                cand_dir.mkdir(parents=True, exist_ok=True)
                cand_file = cand_dir / f"t{temp:.1f}_candidates.json"

                # Generate
                cmd = [
                    sys.executable, "inference/generate_sequences.py",
                    "--seed-sequence", seed_seq,
                    "--vocab", str(vocab_path),
                    "--num-candidates", str(num_candidates),
                    "--temperature", str(temp),
                    "--output", str(cand_file),
                    "--config", str(args.config),
                ]
                rc = run_command(cmd, args.dry_run)
                if rc != 0:
                    print(f"  [WARN] Generation failed for {seed_seq} at t={temp}")
                    continue

                # Score
                scored_file = cand_dir / f"t{temp:.1f}_scored.json"
                cmd = [
                    sys.executable, "inference/score_sequences.py",
                    "--candidates", str(cand_file),
                    "--output", str(scored_file),
                    "--config", str(args.config),
                ]
                run_command(cmd, args.dry_run)

    # Phase 2: Apply Alien sampling at each beta
    print("\n[PHASE 2] Alien sampling...\n")
    for seed_seq in seeds:
        seed_slug = seed_seq.replace(" ", "_")
        for temp in temperatures:
            scored_file = output_dir / "candidates" / seed_slug / f"t{temp:.1f}_scored.json"
            if not args.dry_run and not scored_file.exists():
                continue

            for beta in betas:
                alien_dir = output_dir / "alien" / seed_slug
                alien_dir.mkdir(parents=True, exist_ok=True)
                alien_file = alien_dir / f"t{temp:.1f}_b{beta:.2f}.json"

                cmd = [
                    sys.executable, "inference/alien_sampling.py",
                    "--scored", str(scored_file),
                    "--beta", str(beta),
                    "--top-k", str(top_k),
                    "--output", str(alien_file),
                    "--config", str(args.config),
                ]
                run_command(cmd, args.dry_run)

    # Phase 3: Generate images
    if not args.skip_images:
        print("\n[PHASE 3] Generating images...\n")
        for seed_seq in seeds:
            seed_slug = seed_seq.replace(" ", "_")
            alien_dir = output_dir / "alien" / seed_slug
            if not alien_dir.exists():
                continue

            for alien_file in sorted(alien_dir.glob("*.json")):
                img_dir = output_dir / "images" / seed_slug / alien_file.stem
                cmd = [
                    sys.executable, "inference/make_images.py",
                    "--sequence-file", str(alien_file),
                    "--output-dir", str(img_dir),
                    "--config", str(args.config),
                ]
                run_command(cmd, args.dry_run)

    # Phase 4: Evaluation
    if not args.skip_eval:
        print("\n[PHASE 4] Evaluating...\n")

        # Text novelty for all alien selections
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        for seed_seq in seeds:
            seed_slug = seed_seq.replace(" ", "_")
            alien_dir = output_dir / "alien" / seed_slug
            if not alien_dir.exists():
                continue

            for alien_file in sorted(alien_dir.glob("*.json")):
                report_file = reports_dir / f"text_novelty_{seed_slug}_{alien_file.stem}.json"
                cmd = [
                    sys.executable, "evaluation/text_novelty.py",
                    "--sequence-file", str(alien_file),
                    "--artworks", str(artworks_path),
                    "--artists", str(artists_path),
                    "--output", str(report_file),
                    "--config", str(args.config),
                ]
                run_command(cmd, args.dry_run)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
