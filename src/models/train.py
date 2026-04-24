"""
train.py
========
Fine-tune mT5 for culturally-aware Hinglish + Manglish → English translation.

Training strategy:
  - Instruction-based seq2seq fine-tuning (T5 format)
  - Optional auxiliary emotion classification head (multi-task)
  - Supports mt5-small (local) and mt5-base (Colab GPU)
  - Mixed precision (fp16) when GPU available
  - Gradient checkpointing for memory efficiency

Run locally:
  python src/models/train.py --config config/config.yaml

Run on Colab:
  See notebooks/02_model_training.ipynb
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# ── Transformers ───────────────────────────────────────────────────────────────
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
import sacrebleu


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for instruction-formatted translation pairs.

    Each item in the JSON file must have:
      - instruction_input  : "Translate the following... '{code-mixed}'"
      - instruction_target : "Natural English"
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.tokenizer        = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        source = sample["instruction_input"]
        target = sample["instruction_target"]

        # Encode source
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Encode target — transformers v5: use text_target kwarg (as_target_tokenizer removed)
        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored in loss
        label_ids = labels["input_ids"].squeeze().clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class HinglishTrainer:
    """
    Wraps HuggingFace Seq2SeqTrainer for Hinglish/Manglish translation.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.model_cfg    = self.cfg["model"]
        self.train_cfg    = self.cfg["training"]
        self.paths        = self.cfg["paths"]
        self.project_root = Path(config_path).parent.parent

        set_seed(self.train_cfg.get("seed", 42))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.tokenizer: Optional[AutoTokenizer]              = None
        self.model:     Optional[AutoModelForSeq2SeqLM]     = None

    # ── Setup ──────────────────────────────────────────────────────────────────

    def load_model(self, checkpoint: Optional[str] = None) -> None:
        """Load tokenizer and model (from HuggingFace or local checkpoint)."""
        model_name = checkpoint or self.model_cfg["base"]
        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False  # mT5 requires SentencePiece tokenizer
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if self.model_cfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded — {total_params:,} parameters")

    def load_datasets(self) -> Tuple[TranslationDataset, TranslationDataset]:
        """Load train and validation datasets."""
        max_in  = self.model_cfg["max_input_length"]
        max_out = self.model_cfg["max_target_length"]

        train_path = self.project_root / self.paths["train_file"]
        val_path   = self.project_root / self.paths["val_file"]

        train_ds = TranslationDataset(str(train_path), self.tokenizer, max_in, max_out)
        val_ds   = TranslationDataset(str(val_path),   self.tokenizer, max_in, max_out)
        return train_ds, val_ds

    # ── Compute Metrics ────────────────────────────────────────────────────────

    def _compute_metrics(self, eval_preds):
        """BLEU score computation during training for early stopping signal."""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in preds then decode
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Replace -100 in labels then decode
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]  # sacrebleu expects list of refs

        result = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
        return {"bleu": round(result.score, 4)}

    # ── Train ──────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Full training loop using HuggingFace Seq2SeqTrainer."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        train_ds, val_ds = self.load_datasets()

        output_dir = self.project_root / self.train_cfg["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine FP16 support
        fp16 = self.train_cfg.get("fp16", False) and torch.cuda.is_available()

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.train_cfg["epochs"],
            per_device_train_batch_size=self.train_cfg["batch_size"],
            per_device_eval_batch_size=self.train_cfg.get("batch_size", 8),
            learning_rate=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
            warmup_steps=self.train_cfg["warmup_steps"],
            fp16=fp16,
            predict_with_generate=True,  # needed for BLEU during eval
            eval_strategy="steps",        # v5: renamed from evaluation_strategy
            eval_steps=self.train_cfg["eval_steps"],
            save_strategy="steps",
            save_steps=self.train_cfg["save_steps"],
            logging_steps=self.train_cfg["logging_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            report_to="none",             # set "wandb" if using W&B
            seed=self.train_cfg.get("seed", 42),
            generation_max_length=self.model_cfg["max_target_length"],
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if fp16 else None,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,  # v5: renamed from tokenizer=
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final model + tokenizer
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        # Save training metrics
        metrics = train_result.metrics
        metrics_path = output_dir / "train_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Model saved to {output_dir}")
        logger.info(f"Metrics: {metrics}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def translate(self, text: str, instruction_prefix: Optional[str] = None) -> str:
        """Translate a single code-mixed sentence."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prefix = instruction_prefix or self.cfg.get(
            "instruction_prefix",
            "Translate the following Hinglish/Manglish sentence into natural English preserving cultural meaning:\n"
        )
        prompt = f'{prefix}"{text}"'

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.model_cfg["max_input_length"],
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.model_cfg["max_target_length"],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    def load_for_inference(self, checkpoint: Optional[str] = None) -> None:
        """Load model from saved checkpoint for inference."""
        path = checkpoint or str(self.project_root / self.train_cfg["output_dir"])
        self.load_model(checkpoint=path)
        self.model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train mT5 on Hinglish+Manglish dataset")
    parser.add_argument("--config",    default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--translate",  default=None, help="Quick inference mode: pass a sentence")
    args = parser.parse_args()

    trainer = HinglishTrainer(config_path=args.config)

    if args.translate:
        trainer.load_for_inference()
        result = trainer.translate(args.translate)
        print(f"\nInput : {args.translate}")
        print(f"Output: {result}")
    else:
        trainer.load_model(checkpoint=args.checkpoint)
        trainer.train()


if __name__ == "__main__":
    main()
