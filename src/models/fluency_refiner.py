"""
fluency_refiner.py
==================
Post-processing layer to improve fluency and grammatical correctness of
model outputs before serving to the user.

Strategy:
  1. Grammar correction via language_tool_python (offline, no API needed)
  2. Contraction normalization ("I am not" → "I'm not")
  3. Redundant filler cleanup ("basically, basically...")
  4. Sentence capitalization enforcement

This module is OPTIONAL in the pipeline — disable it in config.yaml if
you find it over-corrects culturally accurate informal English.

Usage:
    from src.models.fluency_refiner import FluencyRefiner
    refiner = FluencyRefiner()
    output = refiner.refine("I am not having mood today")
    print(output)  # → "I'm not in the mood today."
"""

import re
from typing import List, Optional
from loguru import logger

# ── Optional grammar tool (fails gracefully if not installed) ─────────────────
try:
    import language_tool_python
    _LT_AVAILABLE = True
except ImportError:
    _LT_AVAILABLE = False
    logger.warning("language_tool_python not installed. Grammar correction disabled.")


# ── Contraction map (expansion → contraction) ─────────────────────────────────
CONTRACTION_MAP = {
    r"\bI am\b":         "I'm",
    r"\bI am not\b":     "I'm not",
    r"\bI will\b":       "I'll",
    r"\bI will not\b":   "I won't",
    r"\bI have\b":       "I've",
    r"\bI have not\b":   "I haven't",
    r"\bI do not\b":     "I don't",
    r"\bI did not\b":    "I didn't",
    r"\bI would\b":      "I'd",
    r"\bhe is\b":        "he's",
    r"\bshe is\b":       "she's",
    r"\bit is\b":        "it's",
    r"\bthey are\b":     "they're",
    r"\bwe are\b":       "we're",
    r"\byou are\b":      "you're",
    r"\bdo not\b":       "don't",
    r"\bdoes not\b":     "doesn't",
    r"\bdid not\b":      "didn't",
    r"\bcannot\b":       "can't",
    r"\bwill not\b":     "won't",
    r"\bwould not\b":    "wouldn't",
    r"\bshould not\b":   "shouldn't",
    r"\bcould not\b":    "couldn't",
    r"\bthat is\b":      "that's",
    r"\bwhat is\b":      "what's",
    r"\bwhere is\b":     "where's",
    r"\bthere is\b":     "there's",
    r"\blet us\b":       "let's",
    r"\bI am going to\b": "I'm gonna",
    r"\bgoing to\b":     "gonna",
    r"\bwant to\b":      "wanna",
}

# ── Known awkward literal patterns from mT5 output ────────────────────────────
LITERAL_PATTERN_FIXES = {
    # Common mistranslations from Hinglish/Manglish → correct idioms
    r"\bhaving mood\b":          "in the mood",
    r"\bnot having mood\b":      "not in the mood",
    r"\btension taking\b":       "stressing out",
    r"\bdo tension\b":           "stress out",
    r"\bscene is\b":             "situation is",
    r"\bis a scene\b":           "is causing drama",
    r"\bset went\b":             "worked out",
    r"\bannot able\b":           "can't",
    r"\bI am not able\b":        "I can't",
    r"\bfull tired\b":           "completely exhausted",
    r"\bfull happy\b":           "really happy",
    r"\bfull sad\b":             "really sad",
    r"\bfull bore\b":            "completely bored",
    r"\bcome cheyyum\b":         "will come",
    r"\bdo cheyyum\b":           "will do",
    r"\bgo cheyyum\b":           "will go",
    r"\bdo pannuven\b":          "will do it",
    r"\bI am gone\b":            "I'm done / I'm screwed",
    r"\boh no I am\b":           "oh no, I'm",
    r"\btomorrow not will\b":    "won't come tomorrow",
    r"\bnot understanding\b":    "not understanding",   # keep as-is
    r"\bmood not\b":             "not in the mood",
    r"\bbrother I\b":            "I",
    r"\bfriend I\b":             "I",
}

# ── Filler words that can appear redundantly ───────────────────────────────────
REDUNDANT_FILLERS = ["basically basically", "honestly honestly", "like like", "you know you know"]


class FluencyRefiner:
    """
    Post-processing refiner for MT5 translation outputs.
    Applies grammar correction, contraction normalization, and pattern fixes.
    """

    def __init__(self, use_grammar_tool: bool = True, language: str = "en-US"):
        self.lt_tool = None
        if use_grammar_tool and _LT_AVAILABLE:
            try:
                logger.info("Initializing LanguageTool (first launch downloads JRE data)...")
                self.lt_tool = language_tool_python.LanguageTool(language)
                logger.info("LanguageTool ready.")
            except Exception as e:
                logger.warning(f"LanguageTool init failed: {e}. Skipping grammar correction.")
        elif use_grammar_tool and not _LT_AVAILABLE:
            logger.warning("language_tool_python not available. Install it with: pip install language-tool-python")

    # ── Public API ─────────────────────────────────────────────────────────────

    def refine(self, text: str) -> str:
        """Apply full refinement pipeline to a single translation output."""
        if not text or not text.strip():
            return text
        text = self._fix_literal_patterns(text)
        text = self._apply_contractions(text)
        text = self._remove_redundant_fillers(text)
        text = self._fix_capitalization(text)
        text = self._fix_punctuation(text)
        if self.lt_tool:
            text = self._grammar_correct(text)
        return text.strip()

    def refine_batch(self, texts: List[str]) -> List[str]:
        """Refine a list of translation outputs."""
        return [self.refine(t) for t in texts]

    # ── Pipeline Steps ─────────────────────────────────────────────────────────

    @staticmethod
    def _fix_literal_patterns(text: str) -> str:
        """Replace known bad literal-translation patterns."""
        for pattern, replacement in LITERAL_PATTERN_FIXES.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _apply_contractions(text: str) -> str:
        """Convert formal expansions to natural contractions."""
        for pattern, contraction in CONTRACTION_MAP.items():
            text = re.sub(pattern, contraction, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _remove_redundant_fillers(text: str) -> str:
        """Remove doubled filler words."""
        for filler in REDUNDANT_FILLERS:
            word = filler.split()[0]
            text = re.sub(rf"\b{word}\s+{word}\b", word, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _fix_capitalization(text: str) -> str:
        """Ensure first character of the sentence is capitalized."""
        text = text.strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def _fix_punctuation(text: str) -> str:
        """Basic punctuation cleanup."""
        text = re.sub(r"\s([?.!,;:])", r"\1", text)   # remove space before punct
        text = re.sub(r"([?.!])\s*([?.!])+", r"\1", text)  # remove doubled end-punct
        text = re.sub(r"\s{2,}", " ", text)            # collapse multiple spaces
        if text and text[-1] not in ".?!":
            text += "."
        return text

    def _grammar_correct(self, text: str) -> str:
        """Apply LanguageTool grammar corrections (offline)."""
        try:
            matches = self.lt_tool.check(text)
            # Filter out style suggestions for informal English
            # (we want to correct grammar, NOT formality)
            grammar_matches = [
                m for m in matches
                if m.ruleId not in {
                    "EN_QUOTES", "COMMA_PARENTHESIS_WHITESPACE",
                    "WHITESPACE_RULE", "UPPERCASE_SENTENCE_START",
                    # Preserve informal tone markers
                    "GONNA", "WANNA", "INFORMAL_WORD",
                }
            ]
            corrected = language_tool_python.utils.correct(text, grammar_matches)
            return corrected
        except Exception as e:
            logger.debug(f"Grammar correction failed for '{text}': {e}")
            return text

    def __del__(self):
        """Clean up LanguageTool JVM thread."""
        if self.lt_tool:
            try:
                self.lt_tool.close()
            except Exception:
                pass


# ── Demo Examples ──────────────────────────────────────────────────────────────

DEMO_CASES = [
    ("I am not having mood today",             "I'm not in the mood today."),
    ("He is a scene bro",                      "He's causing drama, bro."),
    ("full tired aanu I am",                   "I'm completely exhausted."),
    ("Oh no I am gone yaar",                   "Oh no, I'm done, man."),
    ("I am not able come today",               "I can't come today."),
    ("Set went machane everything is okay",    "It worked out, dude, everything is okay."),
    ("tension taking why?",                    "Why are you stressing out?"),
    ("basically basically it went fine",       "Basically, it went fine."),
]


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    refiner = FluencyRefiner(use_grammar_tool=True)
    print(f"\n{'═'*60}")
    print("  FLUENCY REFINER DEMO")
    print(f"{'═'*60}")
    for raw, expected in DEMO_CASES:
        refined = refiner.refine(raw)
        print(f"\n  Raw      : {raw}")
        print(f"  Refined  : {refined}")
        print(f"  Expected : {expected}")
