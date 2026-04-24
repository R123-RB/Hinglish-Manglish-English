"""
main.py  —  FastAPI Backend
============================
REST API for the Hinglish + Manglish → English translation system.

Endpoints:
  GET  /health                → model status
  POST /translate             → single sentence
  POST /translate/batch       → up to 50 sentences
  POST /feedback              → collect human corrections

Run:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    TranslationRequest, TranslationResponse,
    BatchTranslationRequest, BatchTranslationResponse,
    HealthResponse, FeedbackRequest,
)

# ── Lazy-loaded model components ───────────────────────────────────────────────
_tokenizer   = None
_model       = None
_preprocessor = None
_refiner     = None
_config      = None
_device      = None

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")
FEEDBACK_LOG = Path("data/feedback_log.jsonl")


def _load_config():
    global _config
    if _config is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def _load_model():
    """Load model, tokenizer, preprocessor, and refiner on first request."""
    global _tokenizer, _model, _preprocessor, _refiner, _device

    if _model is not None:
        return  # Already loaded

    cfg = _load_config()
    model_dir = cfg["training"]["output_dir"]

    model_path = Path(model_dir)
    # Check that the directory exists AND contains at least a config or tokenizer file
    required_files = ["config.json", "tokenizer_config.json", "tokenizer.model", "spiece.model"]
    has_model_files = model_path.exists() and any(
        (model_path / f).exists() for f in required_files
    )
    if not has_model_files:
        raise RuntimeError(
            f"No trained model found at '{model_dir}'. "
            "Please train the model first: python src/models/train.py"
        )

    logger.info(f"Loading model from {model_dir}...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    _model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
    _model.eval()

    from src.data.preprocessing import TextPreprocessor
    _preprocessor = TextPreprocessor(config_path=CONFIG_PATH)

    from src.models.fluency_refiner import FluencyRefiner
    _refiner = FluencyRefiner(use_grammar_tool=True)

    logger.info("Model ready for inference.")


def _translate_single(text: str, refine: bool = True) -> dict:
    """Core translation logic."""
    cfg    = _load_config()
    prefix = cfg.get("instruction_prefix",
        "Translate the following Hinglish/Manglish sentence into natural English preserving cultural meaning:\n"
    )
    mc     = cfg["model"]

    # Preprocess
    proc_result = _preprocessor.process(text)
    normalized  = proc_result["normalized"]

    # Build instruction prompt
    prompt  = f'{prefix}"{normalized}"'
    inputs  = _tokenizer(
        prompt, return_tensors="pt",
        max_length=mc["max_input_length"], truncation=True,
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=mc["max_target_length"],
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    translation = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if refine and _refiner:
        translation = _refiner.refine(translation)

    return {
        "input":       text,
        "translation": translation,
        "lang_script": proc_result["lang_script"],
        "lang_tags":   [[tok, tag] for tok, tag in proc_result["lang_tags"]],
    }


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model at startup."""
    try:
        _load_model()
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Model not loaded at startup (server will start anyway): {e}")
    yield  # app runs here
    # (optional shutdown logic goes here)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hinglish + Manglish → English Translator",
    description=(
        "Culturally-aware translation of code-mixed Hinglish and Manglish "
        "into natural, fluent English."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Check API and model status."""
    cfg    = _load_config()
    loaded = _model is not None
    return HealthResponse(
        status="ready" if loaded else "model_not_loaded",
        model=cfg["model"]["base"],
        device=str(_device) if _device else "unknown",
    )


@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate(req: TranslationRequest):
    """
    Translate a single Hinglish / Manglish / Mixed code-mixed sentence.

    Returns culturally accurate, fluent English — NOT a literal translation.
    """
    try:
        _load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        result = _translate_single(req.text, refine=req.refine_fluency)
        return TranslationResponse(**result, literal=None, emotion=None)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")


@app.post("/translate/batch", response_model=BatchTranslationResponse, tags=["Translation"])
async def translate_batch(req: BatchTranslationRequest):
    """
    Translate up to 50 code-mixed sentences in one request.
    """
    try:
        _load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if len(req.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 sentences per batch.")

    results = []
    for text in req.texts:
        try:
            r = _translate_single(text, refine=req.refine_fluency)
            results.append(TranslationResponse(**r, literal=None, emotion=None))
        except Exception as e:
            logger.warning(f"Batch item failed: {text!r} → {e}")
            results.append(TranslationResponse(
                input=text,
                translation="[Translation failed]",
                lang_script="unknown",
                lang_tags=[],
            ))

    return BatchTranslationResponse(results=results, total=len(results))


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(req: FeedbackRequest):
    """
    Submit human evaluation feedback for continuous improvement.
    Saved to data/feedback_log.jsonl for future training data.
    """
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = req.model_dump()
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Feedback logged: {entry}")
    return {"status": "ok", "message": "Thank you for your feedback!"}
