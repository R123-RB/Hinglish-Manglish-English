"""
evaluate.py
===========
Evaluation pipeline for the Hinglish + Manglish → English translation model.

Metrics:
  1. BLEU score      (sacrebleu) — lexical baseline
  2. BERTScore       (bert_score) — semantic similarity PRIMARY metric
  3. ROUGE-L         (rouge_score) — recall-oriented measure

Also provides:
  - Side-by-side output comparison
  - Human evaluation template (printed to console/CSV)
  - Per-category breakdown

Usage:
  python src/models/evaluate.py \\
      --model_path outputs/model \\
      --test_data  data/processed/test.json \\
      --output_csv results/evaluation.csv
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
import numpy as np
from loguru import logger

import sacrebleu
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TranslationEvaluator:
    """
    Evaluates translation quality using BLEU, BERTScore, and ROUGE-L.
    Produces human-readable comparison tables and CSV exports.
    """

    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.model_cfg    = self.cfg["model"]
        self.project_root = Path(config_path).parent.parent

        logger.info(f"Loading model from: {model_path}")
        self._load_model(model_path)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _load_model(self, model_path: str) -> None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _generate(self, prompts: List[str]) -> List[str]:
        """Batch generation with beam search."""
        results = []
        batch_size = 16
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.model_cfg["max_input_length"],
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_length=self.model_cfg["max_target_length"],
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            results.extend([d.strip() for d in decoded])
        return results

    # ── Metrics ────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_bleu(predictions: List[str], references: List[str]) -> float:
        """Corpus BLEU score."""
        refs = [[r] for r in references]
        result = sacrebleu.corpus_bleu(predictions, refs)
        return round(result.score, 4)

    @staticmethod
    def compute_bertscore(
        predictions: List[str], references: List[str], lang: str = "en"
    ) -> Dict[str, float]:
        """BERTScore F1 (semantic similarity) — PRIMARY metric."""
        P, R, F1 = bert_score_fn(
            predictions, references,
            lang=lang,
            verbose=False,
            model_type="distilbert-base-uncased",  # lightweight; upgrade to roberta-large on Colab
        )
        return {
            "precision": round(P.mean().item(), 4),
            "recall":    round(R.mean().item(), 4),
            "f1":        round(F1.mean().item(), 4),
        }

    def compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Average ROUGE-L F1."""
        scores = []
        for pred, ref in zip(predictions, references):
            s = self.rouge.score(ref, pred)
            scores.append(s["rougeL"].fmeasure)
        return round(np.mean(scores), 4)

    # ── Per-Category Breakdown ─────────────────────────────────────────────────

    @staticmethod
    def category_breakdown(
        samples: List[Dict],
        predictions: List[str],
    ) -> Dict[str, Dict]:
        """Group BLEU by category."""
        from collections import defaultdict
        groups = defaultdict(list)  # category → [(pred, ref), ...]
        for sample, pred in zip(samples, predictions):
            cat = sample.get("category", "unknown")
            groups[cat].append((pred, sample["instruction_target"]))

        breakdown = {}
        for cat, pairs in groups.items():
            preds = [p for p, _ in pairs]
            refs  = [[r] for _, r in pairs]
            score = sacrebleu.corpus_bleu(preds, refs).score
            breakdown[cat] = {"bleu": round(score, 2), "count": len(pairs)}
        return breakdown

    # ── Main Evaluate ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_path: str,
        output_csv: Optional[str] = None,
        n_display: int = 20,
    ) -> Dict:
        """
        Run full evaluation on a test JSON file.

        Returns:
            {
              "bleu":       float,
              "bertscore":  {"precision", "recall", "f1"},
              "rouge_l":    float,
              "categories": {...},
            }
        """
        with open(test_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        logger.info(f"Evaluating {len(samples)} test samples...")

        prompts     = [s["instruction_input"]  for s in samples]
        references  = [s["instruction_target"] for s in samples]
        predictions = self._generate(prompts)

        # ── Compute metrics ────────────────────────────────────────────────────
        bleu_score  = self.compute_bleu(predictions, references)
        bert_scores = self.compute_bertscore(predictions, references)
        rouge_l     = self.compute_rouge(predictions, references)
        categories  = self.category_breakdown(samples, predictions)

        results = {
            "bleu":       bleu_score,
            "bertscore":  bert_scores,
            "rouge_l":    rouge_l,
            "categories": categories,
            "n_samples":  len(samples),
        }

        # ── Print summary ──────────────────────────────────────────────────────
        self._print_summary(results)

        # ── Side-by-side display ───────────────────────────────────────────────
        self._print_side_by_side(samples, predictions, references, n=n_display)

        # ── Human eval template ────────────────────────────────────────────────
        self._print_human_eval_template(samples[:5], predictions[:5])

        # ── Optional CSV export ────────────────────────────────────────────────
        if output_csv:
            self._save_csv(samples, predictions, references, output_csv, results)

        return results

    # ── Display Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(results: Dict) -> None:
        bs = results["bertscore"]
        print(f"\n{'═'*58}")
        print(f"  EVALUATION RESULTS  —  {results['n_samples']} samples")
        print(f"{'═'*58}")
        print(f"  BLEU score          : {results['bleu']:.4f}")
        print(f"  BERTScore (F1)      : {bs['f1']:.4f}  ← PRIMARY metric")
        print(f"  BERTScore Precision : {bs['precision']:.4f}")
        print(f"  BERTScore Recall    : {bs['recall']:.4f}")
        print(f"  ROUGE-L             : {results['rouge_l']:.4f}")
        print(f"{'─'*58}")
        print(f"  Per-Category BLEU:")
        for cat, stats in sorted(results["categories"].items()):
            print(f"    {cat:<30} {stats['bleu']:>6.2f}  (n={stats['count']})")
        print(f"{'═'*58}\n")

    @staticmethod
    def _print_side_by_side(
        samples: List[Dict],
        predictions: List[str],
        references: List[str],
        n: int = 10,
    ) -> None:
        print(f"\n{'─'*65}")
        print("  SIDE-BY-SIDE COMPARISON (first {} samples)".format(min(n, len(samples))))
        print(f"{'─'*65}")
        for i, (s, pred, ref) in enumerate(zip(samples[:n], predictions[:n], references[:n])):
            input_text = s.get("input", "")
            print(f"\n  [{i+1}] Input    : {input_text}")
            print(f"       Reference: {ref}")
            print(f"       Predicted: {pred}")
            match_ok = "✅" if pred.lower().strip() == ref.lower().strip() else "🔸"
            print(f"       Match    : {match_ok}")

    @staticmethod
    def _print_human_eval_template(
        samples: List[Dict], predictions: List[str]
    ) -> None:
        print(f"\n{'═'*65}")
        print("  HUMAN EVALUATION TEMPLATE")
        print("  Rate each output on: Meaning ✅/❌ | Tone ✅/❌ | Natural ✅/❌")
        print(f"{'─'*65}")
        for s, pred in zip(samples, predictions):
            print(f"\n  Input    : {s.get('input', '')}")
            print(f"  Expected : {s.get('instruction_target', '')}")
            print(f"  Got      : {pred}")
            print(f"  Emotion  : {s.get('emotion', 'N/A')}")
            print(f"  Meaning: __ | Tone: __ | Natural: __")
        print(f"{'═'*65}\n")

    @staticmethod
    def _save_csv(
        samples: List[Dict],
        predictions: List[str],
        references: List[str],
        output_csv: str,
        summary: Dict,
    ) -> None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "input", "literal", "reference", "predicted",
                    "emotion", "category",
                    "meaning_ok", "tone_ok", "natural_ok", "notes"
                ]
            )
            writer.writeheader()
            for s, pred, ref in zip(samples, predictions, references):
                writer.writerow({
                    "input":      s.get("input", ""),
                    "literal":    s.get("literal", ""),
                    "reference":  ref,
                    "predicted":  pred,
                    "emotion":    s.get("emotion", ""),
                    "category":   s.get("category", ""),
                    "meaning_ok": "",
                    "tone_ok":    "",
                    "natural_ok": "",
                    "notes":      "",
                })
        logger.info(f"Results saved to {output_csv}")
        # Append summary row
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            f.write(f"\n# Summary: BLEU={summary['bleu']}, BERTScore F1={summary['bertscore']['f1']}, ROUGE-L={summary['rouge_l']}\n")



# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hinglish+Manglish translation model")
    parser.add_argument("--model_path", required=True,             help="Path to trained model dir")
    parser.add_argument("--test_data",  required=True,             help="Path to test.json")
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--output_csv", default=None,              help="Optional CSV export path")
    parser.add_argument("--n_display",  type=int, default=20,      help="Samples to show side-by-side")
    args = parser.parse_args()

    evaluator = TranslationEvaluator(
        model_path=args.model_path,
        config_path=args.config,
    )
    evaluator.evaluate(
        test_path=args.test_data,
        output_csv=args.output_csv,
        n_display=args.n_display,
    )
