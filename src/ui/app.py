"""
app.py  —  Streamlit UI
========================
Interactive frontend for the Hinglish + Manglish → English translation system.

Run:
  streamlit run src/ui/app.py
"""

import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import requests
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL     = "http://localhost:8000"
PAGE_TITLE  = "Hinglish + Manglish → English Translator"
PAGE_ICON   = "🌐"
ACCENT      = "#6C63FF"

# ── Page Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0f0f1a; color: #e0e0f0; }

  .hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }
  .hero-sub {
    font-size: 1.05rem; color: #9090b0; margin-bottom: 1.5rem;
  }

  .translation-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #6C63FF55;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    font-size: 1.18rem;
    font-weight: 600;
    color: #e0e0ff;
    line-height: 1.7;
    box-shadow: 0 4px 24px #6C63FF22;
    margin-top: 0.5rem;
  }

  .tag-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
  }
  .tag-EN { background: #1a3a2a; color: #48d080; border: 1px solid #48d08066; }
  .tag-HI { background: #2a1a3a; color: #c880ff; border: 1px solid #c880ff66; }
  .tag-ML { background: #1a2a3a; color: #48c8ff; border: 1px solid #48c8ff66; }
  .tag-UNK { background: #2a2a2a; color: #909090; border: 1px solid #90909066; }

  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }

  .example-btn { cursor: pointer; }

  .stTextArea textarea {
    background: #1a1a2e !important;
    color: #e0e0f0 !important;
    border: 1px solid #6C63FF55 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
  }
  .stButton button {
    background: linear-gradient(135deg, #6C63FF, #48CAE4) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
  }
  .stButton button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def call_translate(text: str, refine: bool = True) -> Optional[dict]:
    """Call the FastAPI backend."""
    try:
        resp = requests.post(
            f"{API_URL}/translate",
            json={"text": text, "refine_fluency": refine},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            st.error("⏳ The API is running but the model has not finished training yet! Please wait for `train.py` to finish.")
        else:
            st.error(f"API HTTP error: {e}")
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def call_health() -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return None


def lang_tag_html(tags: list) -> str:
    html = ""
    for tok, lang in tags:
        cls = f"tag-{lang}"
        html += f'<span class="tag-pill {cls}">{tok}</span> '
    return html


# ── Demo Examples ──────────────────────────────────────────────────────────────
EXAMPLES = [
    ("Manglish – Fatigue",      "Enikku vayya da, full tired aanu"),
    ("Hinglish – Question",     "Scene kya hai bhai?"),
    ("Mixed – Not Coming",      "Bro njan innu varilla, mood illa"),
    ("Mixed – Drama",           "Avan bahut scene aanu da"),
    ("Hinglish – Relief",       "Ekdum mast tha bhai, paisa vasool!"),
    ("Manglish – Panic",        "Aiyyo njan poyi, deadline miss ayi"),
    ("Manglish – Positive",     "Set aayit poyi machane!"),
    ("Mixed – Work",            "Kal njan office varilla bro"),
    ("Hinglish – Reassurance",  "Araam se bhai, tension mat le"),
    ("Codemix Grammar",         "I kal come cheyyum, confirm aanu"),
]


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    refine_fluency = st.checkbox("Fluency Refinement", value=True,
        help="Apply post-processing grammar correction")
    show_lang_tags = st.checkbox("Show Language Tags", value=True,
        help="Color-coded per-token language detection")

    st.markdown("---")
    st.markdown("### 🎯 Try an Example")
    chosen_example = st.selectbox(
        "Pick an example:",
        options=[e[0] for e in EXAMPLES],
    )

    st.markdown("---")
    # API Status
    health = call_health()
    if health:
        if health.get("status") == "ready":
            st.success(f"✅ Model Ready\n`{health['model']}`\nDevice: `{health['device']}`")
        else:
            st.warning("⚠️ Model not loaded yet.\nStart training first.")
    else:
        st.error("❌ API not running.\nStart with:\n```\nuvicorn src.api.main:app --reload\n```")

    st.markdown("---")
    st.markdown("""
**Categories covered:**
- Pure Manglish
- Pure Hinglish
- Mixed Hinglish + Manglish
- Slang normalization
- Emotion-heavy phrases
- Code-mixed grammar
    """)


# ── Main Content ───────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🌐 Hinglish + Manglish → English</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Culturally accurate translation — not literal word-for-word</div>', unsafe_allow_html=True)

# Tabs
tab_translate, tab_batch, tab_compare, tab_about = st.tabs([
    "🔤 Translate", "📦 Batch", "⚡ Compare", "ℹ️ About"
])


# ── TAB 1: TRANSLATE ──────────────────────────────────────────────────────────
with tab_translate:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Input")
        example_text = next(t for n, t in EXAMPLES if n == chosen_example)
        user_input = st.text_area(
            label="Enter code-mixed sentence:",
            value=example_text,
            height=130,
            placeholder='e.g. "Bro njan innu varilla, mood illa"',
            label_visibility="collapsed",
        )
        translate_btn = st.button("🚀 Translate", key="translate_btn", use_container_width=True)

    with col2:
        st.markdown("#### Translation")
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="translation-box" style="color:#555; font-style:italic;">Output will appear here...</div>',
            unsafe_allow_html=True,
        )

    if translate_btn and user_input.strip():
        with st.spinner("Translating..."):
            start = time.time()
            result = call_translate(user_input.strip(), refine=refine_fluency)
            elapsed = time.time() - start

        if result:
            with col2:
                placeholder.markdown(
                    f'<div class="translation-box">"{result["translation"]}"</div>',
                    unsafe_allow_html=True,
                )

            # Details
            st.markdown("---")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("Script Detected", result.get("lang_script", "—").upper())
            with d2:
                st.metric("Translation Time", f"{elapsed:.2f}s")
            with d3:
                st.metric("Tokens", len(result.get("lang_tags", [])))

            if show_lang_tags and result.get("lang_tags"):
                st.markdown("**Token Language Tags:**")
                st.markdown(
                    lang_tag_html(result["lang_tags"]),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<span class="tag-pill tag-EN">EN</span> English &nbsp; '
                    '<span class="tag-pill tag-HI">HI</span> Hinglish &nbsp; '
                    '<span class="tag-pill tag-ML">ML</span> Manglish',
                    unsafe_allow_html=True,
                )

            # Feedback
            st.markdown("---")
            st.markdown("**Was this translation accurate?**")
            fb1, fb2, fb3 = st.columns(3)
            with fb1:
                meaning_ok = st.checkbox("✅ Meaning preserved")
            with fb2:
                tone_ok = st.checkbox("✅ Tone correct")
            with fb3:
                natural_ok = st.checkbox("✅ Sounds natural")
            correction = st.text_input("Suggested correction (optional):", placeholder="Better English version...")
            if st.button("Submit Feedback", key="fb_btn"):
                try:
                    requests.post(f"{API_URL}/feedback", json={
                        "input": user_input,
                        "translation": result["translation"],
                        "meaning_ok": meaning_ok,
                        "tone_ok": tone_ok,
                        "natural_ok": natural_ok,
                        "correction": correction or None,
                    }, timeout=5)
                    st.success("Feedback saved — thank you!")
                except Exception:
                    st.info("API not available to save feedback.")
        else:
            with col2:
                placeholder.error("API not reachable. Is the server running?")


# ── TAB 2: BATCH ──────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Batch Translation (up to 50 sentences)")
    batch_input = st.text_area(
        "Enter one sentence per line:",
        height=200,
        placeholder="Enikku vayya da, full tired aanu\nScene kya hai bhai?\nKal njan office varilla bro",
        label_visibility="collapsed",
    )
    if st.button("Translate All", key="batch_btn"):
        lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one sentence.")
        elif len(lines) > 50:
            st.error("Maximum 50 sentences.")
        else:
            with st.spinner(f"Translating {len(lines)} sentences..."):
                try:
                    r = requests.post(
                        f"{API_URL}/translate/batch",
                        json={"texts": lines, "refine_fluency": refine_fluency},
                        timeout=120,
                    )
                    r.raise_for_status()
                    data = r.json()
                    results = data["results"]
                    st.markdown(f"**{len(results)} translations completed:**")
                    for i, res in enumerate(results, 1):
                        with st.expander(f"[{i}] {res['input'][:60]}..."):
                            st.markdown(f"**Input:** {res['input']}")
                            st.markdown(
                                f'<div class="translation-box">"{res["translation"]}"</div>',
                                unsafe_allow_html=True,
                            )
                            if show_lang_tags:
                                st.markdown(lang_tag_html(res["lang_tags"]), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Batch translation failed: {e}")


# ── TAB 3: COMPARE ────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("#### ❌ Literal vs ✅ Cultural Comparison")
    st.markdown("See how our system differs from literal word-for-word translation.")

    COMPARISON_PAIRS = [
        {
            "input":   "Enikku vayya da, full tired aanu",
            "literal": "I am not able brother, full tired is",
            "cultural":"I'm exhausted, man.",
            "tag":     "fatigue",
        },
        {
            "input":   "Scene kya hai?",
            "literal": "What is the scene?",
            "cultural": "What's going on?",
            "tag":     "curiosity",
        },
        {
            "input":   "Aiyyo njan poyi",
            "literal": "Oh no I am gone",
            "cultural": "Oh no, I'm screwed.",
            "tag":     "panic",
        },
        {
            "input":   "Avan scene aanu",
            "literal": "He is a scene",
            "cultural": "He's creating drama.",
            "tag":     "observation",
        },
        {
            "input":   "Set aayit poyi machane!",
            "literal": "Set went dude!",
            "cultural": "It all worked out perfectly, dude!",
            "tag":     "satisfaction",
        },
        {
            "input":   "Araam se bhai, tension mat le",
            "literal": "Relax brother, don't take tension",
            "cultural": "Chill out, bro, don't stress.",
            "tag":     "reassurance",
        },
    ]

    for pair in COMPARISON_PAIRS:
        with st.container():
            st.markdown(f"**Input:** `{pair['input']}`  — *{pair['tag']}*")
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"❌ Literal: *{pair['literal']}*")
            with c2:
                st.success(f"✅ Cultural: *{pair['cultural']}*")
            st.markdown("---")


# ── TAB 4: ABOUT ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## About This System

This system converts **code-mixed Hinglish + Manglish** text into
**culturally accurate, fluent English**, going beyond literal word-for-word translation.

### Architecture
```
Noisy Input
    ↓ Normalization (spelling, slang dict, script detection)
    ↓ mT5 Fine-Tuned (instruction-based, culturally trained)
    ↓ Fluency Refiner (grammar correction, contraction normalization)
    ↓ Culturally Accurate English
```

### Training Data Categories
| Category | Description |
|---|---|
| Pure Manglish | Malayalam-English code-mix |
| Pure Hinglish | Hindi-English code-mix |
| Mixed | Hinglish + Manglish together |
| Slang norm | `scene`, `set aay`, `poli`, `mass`, `vayya`... |
| Emotion-heavy | Fatigue, panic, frustration, joy |
| Code-mixed grammar | `"I kal come cheyyum"` patterns |

### Evaluation Metrics
- **BERTScore F1** — primary (semantic similarity)
- **BLEU** — lexical baseline
- **ROUGE-L** — recall-oriented
- **Human eval** — meaning, tone, naturalness

### Run Locally
```bash
# 1. Generate dataset
python src/data/dataset_builder.py

# 2. Train
python src/models/train.py --config config/config.yaml

# 3. Start API
uvicorn src.api.main:app --reload

# 4. Start UI (this app)
streamlit run src/ui/app.py
```
    """)
