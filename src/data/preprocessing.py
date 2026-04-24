"""
preprocessing.py
================
Normalization and language-detection pipeline for Hinglish + Manglish text.

Steps:
  1. Unicode cleaning & whitespace normalization
  2. Abbreviation / internet-slang expansion
  3. Spelling-variant normalization (normalization_map.json)
  4. Script detection  (Latin / Devanagari / Malayalam / Mixed)
  5. Token-level language tagging (EN / HI / ML / UNK)

Usage:
    from src.data.preprocessing import TextPreprocessor
    pp = TextPreprocessor()
    result = pp.process("Enikku vayya da, full tired aanu")
    print(result["normalized"])   # → cleaned text
    print(result["lang_tags"])    # → [('Enikku','ML'), ('vayya','ML'), ...]
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from loguru import logger


# ── Unicode code-point ranges ─────────────────────────────────────────────────
_ML_LOW, _ML_HIGH = 0x0D00, 0x0D7F      # Malayalam script
_HI_LOW, _HI_HIGH = 0x0900, 0x097F      # Devanagari (Hindi) script

# ── Known English function words (seed list for EN detection) ─────────────────
_EN_TOKENS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "yes", "and",
    "or", "but", "so", "because", "if", "when", "where", "how", "what",
    "who", "why", "which", "come", "go", "coming", "going", "today",
    "tomorrow", "yesterday", "now", "here", "there", "man", "bro", "dude",
    "full", "very", "too", "really", "just", "also", "already", "never",
    "always", "sometimes", "office", "home", "work", "done", "need",
}

# ── Known Hinglish tokens (transliterated Hindi) ──────────────────────────────
_HI_TOKENS = {
    "yaar", "bhai", "kal", "aaj", "abhi", "kya", "kyu", "kyun", "kaise",
    "bahut", "thoda", "ekdum", "pakka", "sahi", "chalo", "araam", "mast",
    "suno", "dekho", "matlab", "ofc", "bindaas", "dhamaal", "jhakaas",
    "paisa", "thodi", "ho", "gaya", "nahi", "kar", "sakta", "aana",
    "jaana", "karna", "wala", "wali", "hua", "hui", "raha", "rahi",
}

# ── Known Manglish tokens (transliterated Malayalam) ──────────────────────────
_ML_TOKENS = {
    "njan", "nee", "avan", "aval", "nammal", "innu", "nale", "varilla",
    "varum", "varumo", "cheyyum", "poyi", "aanu", "alle", "aano",
    "vayya", "maduthu", "kashtam", "aiyyo", "machane", "machi", "da",
    "di", "mon", "mol", "nokku", "parayum", "ingane", "angane",
    "evidaanu", "enthaa", "mood", "illa", "poli", "mass", "scene",
    "set", "avande", "athu", "ithu", "ethu", "oru", "chetta", "chechi",
}


class TextPreprocessor:
    """
    End-to-end text normalization pipeline for Hinglish + Manglish input.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = self._load_config(config_path)
        paths = cfg.get("paths", {})
        base = Path(config_path).parent.parent  # project root

        norm_path = base / paths.get("normalization_map", "data/dictionaries/normalization_map.json")
        slang_path = base / paths.get("slang_dict", "data/dictionaries/slang_dict.json")

        self.norm_map: Dict[str, str] = {}
        self.slang: Dict = {}

        if norm_path.exists():
            raw = json.loads(norm_path.read_text(encoding="utf-8"))
            self.norm_map = raw.get("spelling_variants", {})
            self.abbrevs = raw.get("number_words", {})
            logger.debug(f"Loaded {len(self.norm_map)} normalization entries.")
        else:
            logger.warning(f"Normalization map not found at {norm_path}.")
            self.abbrevs = {}

        if slang_path.exists():
            self.slang = json.loads(slang_path.read_text(encoding="utf-8"))
            logger.debug("Slang dictionary loaded.")
        else:
            logger.warning(f"Slang dict not found at {slang_path}.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, text: str) -> Dict:
        """
        Full pipeline. Returns a dict with:
          - original     : input as-is
          - cleaned      : after unicode + whitespace clean
          - normalized   : after spelling normalization
          - lang_script  : detected dominant script
          - lang_tags    : [(token, lang_code), ...] per token
        """
        cleaned   = self._clean_unicode(text)
        cleaned   = self._normalize_whitespace(cleaned)
        expanded  = self._expand_abbreviations(cleaned)
        normalized = self._normalize_spelling(expanded)
        script    = self._detect_script(text)
        tags      = self._tag_tokens(normalized)

        return {
            "original":   text,
            "cleaned":    cleaned,
            "normalized": normalized,
            "lang_script": script,
            "lang_tags":  tags,
        }

    def normalize(self, text: str) -> str:
        """Quick single-value normalization (pipeline without tags)."""
        return self.process(text)["normalized"]

    # ── Internal steps ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at '{path}'. Using defaults.")
            return {}

    @staticmethod
    def _clean_unicode(text: str) -> str:
        """Normalize unicode characters and strip zero-width chars."""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  # zero-width
        text = re.sub(r"[^\x00-\x7F\u0900-\u097F\u0D00-\u0D7F]", " ", text)
        return text.strip()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s([?.!,])", r"\1", text)
        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand internet abbreviations (idk → I don't know, rn → right now, etc.)."""
        tokens = text.split()
        expanded = []
        for tok in tokens:
            lower = tok.lower().rstrip(".,!?")
            punct = tok[len(lower):]
            if lower in self.abbrevs:
                expanded.append(self.abbrevs[lower] + punct)
            else:
                expanded.append(tok)
        return " ".join(expanded)

    def _normalize_spelling(self, text: str) -> str:
        """Map noisy/variant spellings to canonical forms."""
        if not self.norm_map:
            return text
        tokens = text.split()
        result = []
        for tok in tokens:
            lower = tok.lower().rstrip(".,!?;:")
            punct = tok[len(lower):]
            canonical = self.norm_map.get(lower, lower)
            # Preserve original casing style
            if tok and canonical and tok[0].isupper():
                canonical = canonical[0].upper() + canonical[1:]
            result.append(canonical + punct)
        return " ".join(result)

    @staticmethod
    def _detect_script(text: str) -> str:
        """
        Detect the dominant script in the text.
        Returns: 'latin', 'devanagari', 'malayalam', or 'mixed'
        """
        latin = devanagari = malayalam = 0
        for ch in text:
            cp = ord(ch)
            if ch.isalpha():
                if cp <= 0x007F:
                    latin += 1
                elif _HI_LOW <= cp <= _HI_HIGH:
                    devanagari += 1
                elif _ML_LOW <= cp <= _ML_HIGH:
                    malayalam += 1
        total = latin + devanagari + malayalam
        if total == 0:
            return "unknown"
        dominant = max(latin, devanagari, malayalam)
        if dominant / total > 0.85:
            if latin == dominant:
                return "latin"
            if devanagari == dominant:
                return "devanagari"
            return "malayalam"
        return "mixed"

    @staticmethod
    def _tag_tokens(text: str) -> List[Tuple[str, str]]:
        """
        Assign a language tag to each token.
        Tags: EN (English), HI (Hindi/Hinglish), ML (Malayalam/Manglish), UNK

        Key insight: Hinglish AND Manglish both use Latin/Roman script.
        Script-voting alone labels everything EN. We must use lexicons first
        for Latin characters, reserving script-voting only for native
        Devanagari or Malayalam codepoints.
        """
        tags = []
        for token in text.split():
            clean = re.sub(r"[^a-zA-Z\u0900-\u097F\u0D00-\u0D7F]", "", token)
            lower = clean.lower()

            # ── Step 1: Detect native (non-Latin) script codepoints ───────────
            ml_chars = sum(1 for ch in clean if _ML_LOW <= ord(ch) <= _ML_HIGH)
            hi_chars = sum(1 for ch in clean if _HI_LOW <= ord(ch) <= _HI_HIGH)

            if ml_chars > 0 and ml_chars >= hi_chars:
                tags.append((token, "ML"))
                continue
            if hi_chars > 0:
                tags.append((token, "HI"))
                continue

            # ── Step 2: Lexicon lookup for Latin-script tokens ────────────────
            # Priority: ML → HI → EN   (ML most dialect-specific, EN most general)
            if lower in _ML_TOKENS:
                tags.append((token, "ML"))
            elif lower in _HI_TOKENS:
                tags.append((token, "HI"))
            elif lower in _EN_TOKENS:
                tags.append((token, "EN"))
            elif not lower:
                tags.append((token, "UNK"))   # punctuation-only token
            else:
                tags.append((token, "UNK"))   # unseen / ambiguous word

        return tags


# ── CLI for quick testing ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pp = TextPreprocessor()
    test_sentences = [
        "Enikku vayya da, full tired aanu",
        "Scene kya hai bhai?",
        "Bro njan innu varilla, mood illa",
        "Kal njan office varilla bro",
        "Set aayit poyi machane!",
        "Aiyyo njan poyi yaar",
        "Avan bahut scene aanu da",
    ]
    for s in test_sentences:
        r = pp.process(s)
        print(f"\nInput    : {r['original']}")
        print(f"Normalized: {r['normalized']}")
        print(f"Script   : {r['lang_script']}")
        print(f"Tags     : {r['lang_tags']}")
