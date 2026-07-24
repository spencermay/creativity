"""
Train the Art model via LoRA fine-tuning of Gemma on artwork concept sequences.

The Art model estimates:
    P_art(c_i | c_0, ..., c_{i-1})

It is trained on art_corpus.txt, where each line is a random permutation
of the 10 concepts (9 CLIP + 1 style) associated with a single artwork.

Usage:
    python training/train_art_model.py \
        --train-file ./data/processed/art_corpus.txt \
        --output-dir ./training/art_model/checkpoints \
        --config ./configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import load_config, set_seed


def load_corpus(path: Path, max_lines: int | None = None) -> list[str]:
    """Load concept sequences from a text file (one per line)."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    print(f"[INFO] Loaded {len(lines)} training sequences from {path}")
    return lines


def create_dataset(lines: list[str], tokenizer, max_seq_len: int) -> Dataset:
    """Tokenize concept sequences for causal language modeling."""

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        # For causal LM, labels = input_ids (shifted internally by the model)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_dict({"text": lines})
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return dataset


def split_dataset(dataset: Dataset, val_fraction: float = 0.05):
    """Split into train and validation."""
    split = dataset.train_test_split(test_size=val_fraction, seed=42)
    return split["train"], split["test"]


def main():
    parser = argparse.ArgumentParser(description="Train Art model (LoRA on Gemma)")
    parser.add_argument("--train-file", type=Path, required=True, help="Path to art_corpus.txt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Checkpoint output dir")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("--max-lines", type=int, default=None, help="Limit training lines (debug)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (QLoRA)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    train_cfg = config["training"]
    model_name = config["models"]["base_model"]
    output_dir = args.output_dir or Path(config["paths"]["art_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Base model: {model_name}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] LoRA r={train_cfg['lora_r']}, alpha={train_cfg['lora_alpha']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (optionally quantized)
    model_kwargs = {"trust_remote_code": True}
    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model_kwargs["device_map"] = device
        model_kwargs["torch_dtype"] = torch.bfloat16 if device != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if args.quantize:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    lines = load_corpus(args.train_file, max_lines=args.max_lines)
    dataset = create_dataset(lines, tokenizer, max_seq_len=train_cfg["max_seq_len"])
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=args.val_fraction)
    print(f"[INFO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=train_cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("[INFO] Starting training...")
    train_result = trainer.train()

    # Save final adapter
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training summary
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["val_samples"] = len(val_dataset)
    val_metrics = trainer.evaluate()
    metrics.update(val_metrics)

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"[INFO] Training complete. Final adapter saved to {final_dir}")
    print(f"[INFO] Training summary: {summary_path}")
    print(f"[INFO] Final eval loss: {val_metrics.get('eval_loss', 'N/A'):.4f}")
    perplexity = math.exp(val_metrics["eval_loss"]) if "eval_loss" in val_metrics else None
    if perplexity:
        print(f"[INFO] Final eval perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
