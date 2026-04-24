"""
augmentation.py
===============
Data augmentation for Hinglish + Manglish translation dataset.

Strategies:
  A. Paraphrase augmentation  – synonym substitution via NLTK WordNet
  B. Noise injection          – character-level typos simulating real texting
  C. Sentence structure swap  – reordering tokens in code-mixed phrases

No external API needed (no Google Translate / back-translation API).

Usage:
    from src.data.augmentation import DataAugmentor
    aug = DataAugmentor()
    variants = aug.augment_sample(sample, n_paraphrases=3, noise_prob=0.15)
"""

import re
import json
import random
import string
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from loguru import logger

# Optional: NLTK for WordNet paraphrasing
try:
    import nltk
    from nltk.corpus import wordnet
    _NLTK_AVAILABLE = True
    # Ensure wordnet is downloaded
    try:
        wordnet.synsets("happy")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("NLTK not available. WordNet paraphrasing disabled.")


# ── Paraphrase seed bank (manual, culturally verified) ────────────────────────
# Maps natural English phrases → list of acceptable paraphrase variants
PARAPHRASE_BANK: Dict[str, List[str]] = {
    "I'm exhausted, man.": [
        "I'm totally wiped out, man.",
        "I'm completely drained, dude.",
        "I'm worn out, man.",
        "Man, I'm dead tired.",
        "I'm beat, bro.",
    ],
    "I'm not in the mood.": [
        "I don't feel like it.",
        "I'm not up for it.",
        "I'm not feeling it.",
        "I just don't feel like doing anything.",
        "I'm not feeling up to it.",
    ],
    "What's going on?": [
        "What's the scene?",
        "What's happening?",
        "What's up?",
        "What's the deal?",
        "What's the plan?",
        "What's the situation?",
    ],
    "I'm not coming today, bro.": [
        "I won't make it today, bro.",
        "I can't come today, man.",
        "Today's a no for me, bro.",
        "I'm skipping today, dude.",
        "Can't make it today, bro.",
    ],
    "He's creating drama.": [
        "He's stirring up trouble.",
        "He's making a scene.",
        "He's acting up.",
        "He's causing a fuss.",
        "He's being dramatic.",
    ],
    "It worked out perfectly.": [
        "Everything fell into place.",
        "It all sorted itself out.",
        "It came together perfectly.",
        "Things worked out great.",
        "It all worked out in the end.",
    ],
    "Oh no, I'm screwed.": [
        "Oh no, I'm in trouble.",
        "I'm done for.",
        "Oh man, I'm finished.",
        "I'm in deep trouble now.",
        "Well, I'm doomed.",
    ],
    "That's tough / I feel you.": [
        "That's rough.",
        "That sounds hard.",
        "That must be difficult.",
        "I get that, it's tough.",
        "That's really hard.",
    ],
    "I won't come to the office tomorrow, bro.": [
        "I'm skipping the office tomorrow, bro.",
        "I won't be at work tomorrow, man.",
        "I'm not heading to the office tomorrow, bro.",
        "Tomorrow I'm staying home, bro.",
        "No office for me tomorrow, dude.",
    ],
    "Let's go / Come on.": [
        "Let's head out.",
        "Come on, let's move.",
        "Let's do this.",
        "Alright, let's go.",
        "Let's get going.",
    ],
    "Relax / Take it easy.": [
        "Chill out.",
        "Easy there.",
        "Don't stress.",
        "Take it slow.",
        "Keep calm.",
    ],
    "I'm fed up.": [
        "I'm done with this.",
        "I've had enough.",
        "I'm over it.",
        "I'm so tired of this.",
        "I can't take this anymore.",
    ],
    "For sure / Definitely.": [
        "Absolutely.",
        "Without a doubt.",
        "100%.",
        "You bet.",
        "No question.",
    ],
    "He's impressive / He's cool.": [
        "He's awesome.",
        "He's got style.",
        "He's really something.",
        "He's quite the character.",
        "He's genuinely impressive.",
    ],
    "That's awesome / That's great.": [
        "That's brilliant.",
        "That's fantastic.",
        "That's amazing.",
        "That's superb.",
        "That's incredible.",
    ],
}

# ── Common EN words to target for synonym substitution ───────────────────────
_SYNONYM_TARGETS = {
    "tired", "exhausted", "bored", "happy", "sad", "good", "bad", "great",
    "quickly", "slowly", "really", "totally", "completely", "very", "quite",
    "done", "finished", "working", "going", "coming", "leaving", "staying",
}


class DataAugmentor:
    """
    Augments translation pairs via:
      - Bank-based paraphrasing (natural English side)
      - WordNet synonym substitution (English side)
      - Noise injection (code-mixed input side)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = self._load_config(config_path)
        aug_cfg = cfg.get("augmentation", {})
        self.n_paraphrases: int = aug_cfg.get("paraphrase_variants", 3)
        self.noise_prob: float = aug_cfg.get("noise_prob", 0.15)

        norm_path = Path(config_path).parent.parent / cfg.get("paths", {}).get(
            "normalization_map", "data/dictionaries/normalization_map.json"
        )
        self.noise_subs: Dict = {}
        if norm_path.exists():
            raw = json.loads(norm_path.read_text(encoding="utf-8"))
            self.noise_subs = raw.get("noise_patterns", {}).get("common_substitutions", {})

    # ── Public API ─────────────────────────────────────────────────────────────

    def augment_sample(
        self,
        sample: Dict,
        n_paraphrases: Optional[int] = None,
        noise_prob: Optional[float] = None,
    ) -> List[Dict]:
        """
        Returns n_paraphrases augmented copies of a sample.

        Input sample format:
          {
            "input":   "code-mixed sentence",
            "literal": "literal translation",
            "natural": "natural English (main target)",
            "emotion": "emotion tag",
            "category": "category label",
          }
        """
        n = n_paraphrases or self.n_paraphrases
        prob = noise_prob or self.noise_prob
        results = []

        paraphrases = self._paraphrase(sample["natural"], n)
        for i, para in enumerate(paraphrases):
            aug = deepcopy(sample)
            aug["natural"] = para
            aug["input"] = self._inject_noise(sample["input"], prob)
            aug["augmented"] = True
            aug["aug_strategy"] = "paraphrase"
            results.append(aug)

        return results

    def augment_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Augment a full list of samples. Returns original + augmented."""
        augmented = []
        for sample in dataset:
            augmented.append(sample)
            augmented.extend(self.augment_sample(sample))
        logger.info(
            f"Augmentation: {len(dataset)} → {len(augmented)} samples "
            f"({len(augmented) - len(dataset)} new)."
        )
        return augmented

    # ── Paraphrasing ───────────────────────────────────────────────────────────

    def _paraphrase(self, text: str, n: int) -> List[str]:
        """Generate n paraphrase variants of English text."""
        variants = []

        # Strategy 1: Paraphrase bank lookup (exact match)
        if text in PARAPHRASE_BANK:
            pool = PARAPHRASE_BANK[text]
            variants.extend(random.sample(pool, min(n, len(pool))))

        # Strategy 2: Partial bank match (substring)
        if len(variants) < n:
            for key, pool in PARAPHRASE_BANK.items():
                if key.rstrip(".?!").lower() in text.lower():
                    candidates = [c for c in pool if c not in variants]
                    if candidates:
                        variants.append(random.choice(candidates))
                    if len(variants) >= n:
                        break

        # Strategy 3: WordNet synonym substitution
        if len(variants) < n and _NLTK_AVAILABLE:
            wordnet_vars = self._wordnet_paraphrase(text, n - len(variants))
            variants.extend(wordnet_vars)

        # Strategy 4: Minor surface variations as fallback
        while len(variants) < n:
            variants.append(self._surface_vary(text, len(variants)))

        return variants[:n]

    @staticmethod
    def _wordnet_paraphrase(text: str, n: int) -> List[str]:
        """Substitute one word with a WordNet synonym to create variants."""
        tokens = text.split()
        variants = []
        attempts = 0
        while len(variants) < n and attempts < 20:
            attempts += 1
            idx = random.randint(0, len(tokens) - 1)
            word = tokens[idx].lower().strip(string.punctuation)
            if word not in _SYNONYM_TARGETS:
                continue
            synsets = wordnet.synsets(word, pos=wordnet.ADJ) or wordnet.synsets(word, pos=wordnet.ADV)
            if not synsets:
                synsets = wordnet.synsets(word)
            synonyms = set()
            for syn in synsets[:3]:
                for lemma in syn.lemmas():
                    syn_word = lemma.name().replace("_", " ")
                    if syn_word.lower() != word and len(syn_word.split()) == 1:
                        synonyms.add(syn_word)
            if synonyms:
                new_word = random.choice(list(synonyms))
                new_tokens = tokens[:]
                # Preserve punctuation
                punct = ""
                if tokens[idx] and tokens[idx][-1] in string.punctuation:
                    punct = tokens[idx][-1]
                new_tokens[idx] = new_word.capitalize() if tokens[idx][0].isupper() else new_word
                new_tokens[idx] += punct
                variant = " ".join(new_tokens)
                if variant != text and variant not in variants:
                    variants.append(variant)
        return variants

    @staticmethod
    def _surface_vary(text: str, attempt: int) -> str:
        """Minimal surface variation: contractions and filler words."""
        contractions = {
            "I am": "I'm", "I'm": "I am",
            "I will": "I'll", "I'll": "I will",
            "I have": "I've", "I've": "I have",
            "do not": "don't", "don't": "do not",
            "cannot": "can't", "can't": "cannot",
            "will not": "won't", "won't": "will not",
            "it is": "it's", "it's": "it is",
            "that is": "that's", "that's": "that is",
            "he is": "he's", "he's": "he is",
            "she is": "she's", "she's": "she is",
            "they are": "they're", "they're": "they are",
        }
        for src, tgt in contractions.items():
            if src in text:
                return text.replace(src, tgt, 1)
        fillers = ["honestly", "really", "you know", "like", "basically"]
        filler = fillers[attempt % len(fillers)]
        return f"{filler.capitalize()}, {text[0].lower() + text[1:]}"

    # ── Noise Injection ────────────────────────────────────────────────────────

    def _inject_noise(self, text: str, prob: float) -> str:
        """
        Simulate real texting noise:
          - Drop double consonants ('varilla' → 'varila')
          - Randomly delete/repeat characters
          - Character substitution from noise map
        """
        if prob <= 0:
            return text
        tokens = text.split()
        noisy = []
        for tok in tokens:
            if random.random() < prob and len(tok) > 2:
                noise_type = random.choice(["drop_double", "delete_char", "repeat_char"])
                if noise_type == "drop_double":
                    tok = self._drop_double_consonant(tok)
                elif noise_type == "delete_char":
                    idx = random.randint(1, len(tok) - 1)
                    tok = tok[:idx] + tok[idx + 1:]
                elif noise_type == "repeat_char":
                    idx = random.randint(0, len(tok) - 1)
                    tok = tok[:idx] + tok[idx] * 2 + tok[idx + 1:]
            noisy.append(tok)
        return " ".join(noisy)

    @staticmethod
    def _drop_double_consonant(token: str) -> str:
        """'varilla' → 'varila', 'cheyyum' → 'cheyum'"""
        consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
        result = []
        i = 0
        while i < len(token):
            result.append(token[i])
            if (i + 1 < len(token)
                    and token[i] == token[i + 1]
                    and token[i] in consonants):
                i += 2  # skip duplicate
            else:
                i += 1
        return "".join(result)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    aug = DataAugmentor()
    sample = {
        "input":    "Enikku vayya da, full tired aanu",
        "literal":  "I am not able brother, full tired is",
        "natural":  "I'm exhausted, man.",
        "emotion":  "fatigue",
        "category": "manglish_emotion",
    }
    variants = aug.augment_sample(sample, n_paraphrases=4, noise_prob=0.2)
    print(f"Original → {sample['natural']}")
    for i, v in enumerate(variants, 1):
        print(f"  Variant {i} input : {v['input']}")
        print(f"  Variant {i} target: {v['natural']}")
