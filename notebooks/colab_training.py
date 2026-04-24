# ============================================================
#  GOOGLE COLAB TRAINING NOTEBOOK
#  Hinglish + Manglish → Culturally Accurate English
#  Run each CELL block sequentially in Colab
# ============================================================
# Instructions:
#   1. Upload your project zip to Colab (or clone from GitHub)
#   2. Go to Runtime → Change runtime type → GPU (T4)
#   3. Copy each CELL block into a Colab cell and run in order
# ============================================================


# ══════════════════════════════════════════════════════════════
# CELL 1 — Install Dependencies
# ══════════════════════════════════════════════════════════════
"""
!pip install transformers sentencepiece datasets accelerate \
             sacrebleu bert-score rouge-score \
             nltk language-tool-python \
             fastapi uvicorn streamlit plotly \
             pyyaml loguru tqdm -q
"""


# ══════════════════════════════════════════════════════════════
# CELL 2 — Mount Drive & Upload Project
# ══════════════════════════════════════════════════════════════
"""
from google.colab import drive, files
drive.mount('/content/drive')

# Option A: upload zip
# uploaded = files.upload()
# !unzip hinglish_manglish.zip -d /content/project

# Option B: if already on Drive
# !cp -r /content/drive/MyDrive/hinglish_manglish /content/project

import os
os.chdir('/content/project')
print("Working dir:", os.getcwd())
"""


# ══════════════════════════════════════════════════════════════
# CELL 3 — Verify GPU
# ══════════════════════════════════════════════════════════════
"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("VRAM:", torch.cuda.get_device_properties(0).total_memory // (1024**3), "GB")
"""


# ══════════════════════════════════════════════════════════════
# CELL 4 — Switch to mt5-base + Enable fp16 in config
# ══════════════════════════════════════════════════════════════
"""
import yaml

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["base"]        = "google/mt5-base"     # 580M params
cfg["training"]["fp16"]     = True                  # Colab T4 supports this
cfg["training"]["batch_size"]  = 32
cfg["training"]["epochs"]      = 10
cfg["training"]["eval_steps"]  = 100
cfg["training"]["save_steps"]  = 200

with open("config/config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print("Config updated for Colab GPU:")
print(f"  Model      : {cfg['model']['base']}")
print(f"  FP16       : {cfg['training']['fp16']}")
print(f"  Batch size : {cfg['training']['batch_size']}")
print(f"  Epochs     : {cfg['training']['epochs']}")
"""


# ══════════════════════════════════════════════════════════════
# CELL 5 — Generate Dataset
# ══════════════════════════════════════════════════════════════
"""
import sys
sys.path.insert(0, '/content/project')

from src.data.dataset_builder import DatasetBuilder

builder = DatasetBuilder(config_path="config/config.yaml")
splits  = builder.build_and_save(augment=True)
builder.print_stats(splits)
"""


# ══════════════════════════════════════════════════════════════
# CELL 6 — Download mT5-base (shows download progress)
# ══════════════════════════════════════════════════════════════
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Downloading mt5-base tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)

print("Downloading mt5-base model (~2.3 GB)...")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""


# ══════════════════════════════════════════════════════════════
# CELL 7 — Train
# ══════════════════════════════════════════════════════════════
"""
import subprocess, sys

result = subprocess.run(
    [sys.executable, "src/models/train.py", "--config", "config/config.yaml"],
    capture_output=False,
)
print("Training exit code:", result.returncode)
"""

# OR inline training:
"""
from src.models.train import HinglishTrainer

trainer = HinglishTrainer(config_path="config/config.yaml")
trainer.load_model()
trainer.train()
"""


# ══════════════════════════════════════════════════════════════
# CELL 8 — Evaluate
# ══════════════════════════════════════════════════════════════
"""
from src.models.evaluate import TranslationEvaluator

evaluator = TranslationEvaluator(
    model_path="outputs/model",
    config_path="config/config.yaml",
)
results = evaluator.evaluate(
    test_path="data/processed/test.json",
    output_csv="results/evaluation.csv",
    n_display=20,
)

print(f"\nBLEU        : {results['bleu']}")
print(f"BERTScore F1: {results['bertscore']['f1']}  ← PRIMARY")
print(f"ROUGE-L     : {results['rouge_l']}")
"""


# ══════════════════════════════════════════════════════════════
# CELL 9 — Quick Inference Demo
# ══════════════════════════════════════════════════════════════
"""
from src.models.train import HinglishTrainer

trainer = HinglishTrainer(config_path="config/config.yaml")
trainer.load_for_inference(checkpoint="outputs/model")

test_sentences = [
    "Enikku vayya da, full tired aanu",
    "Scene kya hai bhai?",
    "Bro njan innu varilla, mood illa",
    "Kal njan office varilla bro",
    "Set aayit poyi machane!",
    "Aiyyo njan poyi da",
    "Araam se bhai, tension mat le",
    "I kal come cheyyum, pakka",
]

print("\\n" + "="*60)
print("  INFERENCE DEMO")
print("="*60)
for s in test_sentences:
    output = trainer.translate(s)
    print(f"\\n  Input : {s}")
    print(f"  Output: {output}")
"""


# ══════════════════════════════════════════════════════════════
# CELL 10 — Save Model to Drive
# ══════════════════════════════════════════════════════════════
"""
import shutil

drive_path = "/content/drive/MyDrive/hinglish_mt5_model"
shutil.copytree("outputs/model", drive_path, dirs_exist_ok=True)
print(f"Model saved to Drive: {drive_path}")

# Also save evaluation results
shutil.copy("results/evaluation.csv", "/content/drive/MyDrive/evaluation.csv")
print("Evaluation CSV saved to Drive.")
"""


# ══════════════════════════════════════════════════════════════
# CELL 11 — Optional: Fluency Refiner Demo
# ══════════════════════════════════════════════════════════════
"""
from src.models.fluency_refiner import FluencyRefiner, DEMO_CASES

refiner = FluencyRefiner(use_grammar_tool=True)

print("\\nFluency Refiner Demo:")
print("="*60)
for raw, expected in DEMO_CASES:
    refined = refiner.refine(raw)
    print(f"\\n  Raw      : {raw}")
    print(f"  Refined  : {refined}")
    print(f"  Expected : {expected}")
"""


# ══════════════════════════════════════════════════════════════
# CELL 12 — BERTScore vs BLEU: Why BLEU Fails Here
# ══════════════════════════════════════════════════════════════
"""
import sacrebleu
from bert_score import score as bert_score

reference = "I'm exhausted, man."
candidates = [
    "I am not able brother, full tired is",     # literal (BAD)
    "I'm totally worn out, dude.",               # cultural (GOOD)
    "I'm exhausted, man.",                       # perfect match
]

print("\\nWhy BLEU fails for cultural translation:")
print("="*60)
for cand in candidates:
    bleu  = sacrebleu.corpus_bleu([cand], [[reference]]).score
    _, _, f1 = bert_score([cand], [reference], lang="en", verbose=False)
    print(f"\\n  Candidate  : {cand}")
    print(f"  BLEU       : {bleu:.2f}  (lower = bad, but doesn't tell full story)")
    print(f"  BERTScore  : {f1.mean().item():.4f}  ← BETTER for cultural meaning")
"""
