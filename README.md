# 🌐 Hinglish + Manglish → Culturally Accurate English

> **Not literal translation — culturally aware translation**
>
> `"Scene kya hai?"` → ✅ `"What's going on?"` — NOT ❌ `"What is the scene?"`

---

## 🎯 What This System Does

Converts **code-mixed Hinglish + Manglish** text into **natural, fluent English** that preserves:
- Cultural intent and idiom meaning
- Social tone (casual / respectful / rude)
- Emotional subtext (fatigue, panic, hype, sarcasm)

---

## 🏗️ Architecture

```
Input (Noisy Hinglish / Manglish / Mixed)
        ↓
[1] Normalization
    • Spelling variants ("varila" → "varilla")
    • Slang dictionary lookup
    • Script & language detection (EN / HI / ML)
        ↓
[2] mT5 Fine-Tuned Translation (Instruction-Based)
    • Input:  "Translate the following Hinglish/Manglish sentence..."
    • Output: Culturally accurate English
        ↓
[3] Fluency Refiner (Post-Processing)
    • Grammar correction (LanguageTool)
    • Contraction normalization
    • Literal-pattern fixes
        ↓
Culturally Accurate English Output
```

---

## 📁 Project Structure

```
hinglish-manglish-english/
├── config/
│   └── config.yaml                 ← All hyperparameters
│
├── data/
│   ├── dictionaries/
│   │   ├── slang_dict.json         ← Hinglish + Manglish slang → English
│   │   └── normalization_map.json  ← Typo/noise normalization
│   ├── processed/                  ← Generated: train/val/test.json
│   └── raw/                        ← (Optional) real-world data
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py        ← Normalization pipeline
│   │   ├── dataset_builder.py      ← 3-layer synthetic dataset + seed data
│   │   └── augmentation.py         ← Paraphrase + noise augmentation
│   ├── models/
│   │   ├── train.py                ← mT5 fine-tuning (HuggingFace Trainer)
│   │   ├── evaluate.py             ← BLEU + BERTScore + ROUGE-L
│   │   └── fluency_refiner.py      ← Post-processing layer
│   ├── api/
│   │   ├── main.py                 ← FastAPI backend
│   │   └── schemas.py              ← Pydantic models
│   └── ui/
│       └── app.py                  ← Streamlit frontend
│
├── generate_dataset.py             ← Standalone dataset generation script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_dataset.py --verify
```
This creates `data/processed/train.json`, `val.json`, `test.json`
with ~900 instruction-formatted samples (100+ seed × augmentation).

### 3. Train the Model (Local — mt5-small)
```bash
python src/models/train.py --config config/config.yaml
```
Model saves to `outputs/model/` after training.

### 4. Train on Google Colab (mt5-base, GPU)
Upload the project to Colab and run the training notebook.
Enable GPU: `Runtime → Change runtime type → T4 GPU`
Then in `config.yaml`:
```yaml
model:
  base: "google/mt5-base"
training:
  fp16: true
  batch_size: 32
```

### 5. Evaluate
```bash
python src/models/evaluate.py \
    --model_path outputs/model \
    --test_data  data/processed/test.json \
    --output_csv results/evaluation.csv
```

### 6. Run the API
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Run the UI
```bash
streamlit run src/ui/app.py
```

---

## 📊 Dataset Design (3-Layer)

| Layer | Field | Purpose |
|---|---|---|
| Layer 1 | `literal` | Word-for-word (structural learning) |
| Layer 2 | `natural` | Culturally accurate English (**main target**) |
| Layer 3 | `emotion` + `category` | Auxiliary task / cultural tagging |

### Instruction Format
Every training sample is wrapped as:
```
Translate the following Hinglish/Manglish sentence into natural English preserving cultural meaning:
"{input_sentence}"
```

### Categories Covered

| Category | Examples |
|---|---|
| Manglish emotion | `vayya`, `maduthu`, `kashtam`, `aiyyo` |
| Manglish slang | `scene`, `set aay`, `poli`, `mass` |
| Hinglish question | `kya scene hai`, `kya baat hai` |
| Hinglish positive | `mast`, `dhamaal`, `jhakaas`, `paisa vasool` |
| Mixed (HI+ML) | `Kal njan office varilla bro` |
| Code-mixed grammar | `I kal come cheyyum` |
| Social tone | Respectful, casual, rude styles |
| Daily life | Food, weather, work, study |

---

## 🔑 Key Examples

| Input | ❌ Literal | ✅ Cultural |
|---|---|---|
| `Enikku vayya da, full tired aanu` | I am not able brother, full tired is | I'm exhausted, man. |
| `Scene kya hai?` | What is the scene? | What's going on? |
| `Aiyyo njan poyi` | Oh no I am gone | Oh no, I'm screwed. |
| `Avan scene aanu` | He is a scene | He's creating drama. |
| `Set aayit poyi machane!` | Set went dude! | It all worked out perfectly, dude! |
| `Araam se bhai, tension mat le` | Relax brother, don't take tension | Chill out, bro, don't stress. |

---

## 📏 Evaluation

> **BLEU score alone is insufficient for cultural translation.**
> A culturally correct output can score 0.3 BLEU while being semantically perfect.

| Metric | Role |
|---|---|
| **BERTScore F1** | Primary — semantic similarity |
| BLEU | Secondary — lexical baseline |
| ROUGE-L | Recall-oriented measure |
| Human eval CSV | Meaning / Tone / Naturalness ratings |

---

## 🧑‍🔬 Augmentation Strategy

| Strategy | Implementation |
|---|---|
| Paraphrase augmentation | Manual paraphrase bank + WordNet synonym substitution |
| Noise injection | Drop double consonants, random char delete/repeat |
| Back-translation | Disabled (no unreliable API) |

**No Google Translate / external API required.**

---

## 🚀 Scaling to mt5-base (Colab)

Change these in `config/config.yaml`:
```yaml
model:
  base: "google/mt5-base"     # 580M params
training:
  fp16: true                  # T4 GPU
  batch_size: 32
  epochs: 10
```

Expected improvement: BERTScore F1 ~0.72 → ~0.82+

---

## ⚠️ Hard Truths

| Challenge | Mitigation |
|---|---|
| Cultural meaning ≠ deterministic | Multiple valid outputs in dataset; BERTScore over BLEU |
| Manglish variability is extreme | Heavy normalization + noise injection training |
| Same phrase → different tones | Emotion tags track tone |
| Dataset quality > model size | 3-layer curated seed + augmentation |

---

## 📬 Feedback Loop

The UI and API collect human correction feedback to `data/feedback_log.jsonl`.
Use this data to iteratively improve the training set.

---

## 📄 License

MIT — free to use, modify, and extend.
