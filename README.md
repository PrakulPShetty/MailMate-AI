# MailMate AI

**Context-aware predictive text for professional email composition.**

MailMate is a local-first email writing assistant. As you type a draft, it
suggests the next word (or completes the word you are already typing) as inline
ghost text, and accepts the suggestion with `Tab`. Every suggestion is produced
by classical NLP and machine learning models running on the CPU — no GPU, no
cloud inference, no third-party API calls.

The project is a Natural Language Processing mini project built for academic
evaluation and as a reproducible reference implementation of a
statistical + supervised NLP pipeline.

---

## Highlights

- **Inline ghost text with `Tab` to accept** — suggestions render exactly where
  the cursor is, with instant local prefix filtering while the next request is
  in flight.
- **Context-aware ranking** — the email category predicted from the draft
  changes which words are suggested, so a leave request and a job application
  are completed with different vocabulary.
- **Fully inspectable models** — Laplace-smoothed N-gram tables, a WordNet tone
  lexicon, and a TF-IDF + Logistic Regression classifier are all stored in one
  119 KB artifact.
- **Sub-5 ms warm inference** — CPU-only, suitable for laptops and lab machines.
- **Graceful degradation** — the Studio reports `Backend Offline` instead of
  fabricating suggestions when the API is unreachable.

---

## The three NLP/ML engines

| Engine | Technique | Role in the product |
|---|---|---|
| **Language model** | Interpolated unigram / bigram / trigram model with Laplace (add-one) smoothing, weights λ = 0.2 / 0.3 / 0.5 | Estimates `P(next word ∣ context)` for every vocabulary candidate |
| **Tone model** | NLTK WordNet-expanded tone lexicon, pre-computed and normalized per tone (`professional`, `formal`, `polite`) | Scores how well a candidate fits the selected communication tone |
| **Context classifier** | TF-IDF (1–2 grams) + Logistic Regression over 7 email categories | Detects the intent of the draft and biases suggestions toward that category |

Candidates are ranked with a weighted blend of these signals plus a corpus
frequency prior:

```
score(w) = 0.50 · P_LM(w | context)
         + 0.25 · P_context(w | detected category)
         + 0.15 · tone(w, selected tone)
         + 0.10 · unigram_prior(w)
```

All weights are declared in `config.json` and validated against the model
artifact at backend startup.

---

## System architecture

```
       ┌─────────────────────────────────────────────────────────────┐
       │  Browser — Next.js 16 App Router (port 3000)                │
       │                                                             │
       │  /            Overview          /studio   Live composer      │
       │  /how-it-works Pipeline         /about    Credits            │
       │                                                             │
       │  Ghost-text textarea + Tab accept + Top-3 chips + telemetry │
       └───────────────────────────────┬─────────────────────────────┘
                                       │  fetch /api/*  (rewrite proxy)
                                       ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  Flask REST API (port 5000)                                 │
       │                                                             │
       │  GET  /health      GET  /categories                         │
       │  POST /predict     POST /classify                           │
       └───────────────────────────────┬─────────────────────────────┘
                                       │  in-process calls
                                       ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  NLP core — MailMate-AI/src/                                │
       │                                                             │
       │  preprocessing ─► language_model ─┐                         │
       │  context_classifier ──────────────┼─► predictor (ranking)   │
       │  tone_scorer ─────────────────────┘                         │
       └───────────────────────────────┬─────────────────────────────┘
                                       │  loaded once, eagerly
                                       ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  Artifact — MailMate-AI/models/mailmate_model.joblib        │
       │  vocab · n-gram counts · λ · vectorizer · classifier ·       │
       │  tone lexicon · scoring weights · config snapshot           │
       └─────────────────────────────────────────────────────────────┘
```

---

## Quantitative evaluation

Measured on the held-out 80/20 stratified test split of the 210-sample curated
corpus, and on the N-gram next-word task (45 sampled documents). Reproduced by
`MailMate-AI/notebooks/01_mailmate_nlp_training.ipynb`.

| Metric | Result |
|---|---|
| Context classifier accuracy (7 categories) | **100.00 %** |
| Context classifier weighted precision / recall / F1 | **100.00 %** |
| N-gram Top-3 next-word accuracy | **94.56 %** |
| N-gram Top-1 next-word accuracy | **77.86 %** |
| Language model perplexity (vocab = 261) | **40.04** |
| Warm CPU inference latency | **< 5 ms** |
| Model artifact size | **119 KB** |

The 100 % classifier accuracy reflects a small, highly distinctive curated
corpus where each category has its own vocabulary (e.g. *internship*,
*resume*, *medical leave*, *pull request*). On noisy real-world email, accuracy
would be lower; this is documented as a known limitation rather than presented
as a general benchmark.

---

## Repository structure

```
MailMate-AI/
├── src/                          Runtime NLP engine
│   ├── preprocessing.py          Lowercase + regex tokenization, suffix normalization
│   ├── language_model.py         Interpolated N-gram scorer with Laplace smoothing
│   ├── tone_scorer.py            WordNet tone lexicon lookup
│   ├── context_classifier.py     TF-IDF + Logistic Regression wrapper
│   ├── predictor.py              Candidate ranking and public predict()/classify() API
│   └── evaluation.py             Metric helpers
├── backend/
│   ├── app.py                    Flask API (health, categories, predict, classify)
│   └── loader.py                 Artifact loading, contract validation, singleton
├── tests/
│   └── test_api.py               23-case pytest API suite
├── frontend/                     Next.js 16 App Router UI
│   ├── app/                      Overview, How it works, About, Studio
│   ├── components/               Navbar, Footer
│   ├── lib/api.js                API client (AbortController-aware)
│   └── next.config.js            /api/* → localhost:5000 rewrite
├── models/.gitkeep               Artifact folder (mailmate_model.joblib is gitignored)
├── data/
│   ├── raw/.gitkeep              Raw corpus folder (contents gitignored)
│   └── processed/.gitkeep        Processed folder (labeled_emails.csv is gitignored)
├── notebooks/
│   └── 01_mailmate_nlp_training.ipynb   Trains and exports the artifact
└── verify_models.py              Standalone artifact verifier
config.json                       Model contract: version, categories, tones, weights
requirements.txt                  Pinned Python dependencies
```

Model binaries and processed datasets are intentionally excluded from version
control because they are regenerated deterministically by the notebook in
seconds. Empty folders are preserved with `.gitkeep` placeholders.

---

## Quick start

Full, copy-paste instructions for every operating system live in
**[`INSTRUCTIONS.md`](INSTRUCTIONS.md)**. The short version:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate the model artifact (gitignored in this public repo)
#    Run MailMate-AI/notebooks/01_mailmate_nlp_training.ipynb, then place:
#    MailMate-AI/models/mailmate_model.joblib

# 3. Install frontend dependencies
cd MailMate-AI/frontend && npm install && cd ../..

# 4. Terminal 1 — backend
python MailMate-AI/backend/app.py

# 5. Terminal 2 — frontend
cd MailMate-AI/frontend && npm run dev
```

Open <http://localhost:3000>, then <http://localhost:3000/studio> to use the
ghost-text composer.

**Verification**

```bash
python MailMate-AI/verify_models.py
python -m pytest MailMate-AI/tests/test_api.py -q
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status, model version, uptime |
| `GET` | `/categories` | Categories, tones, suggestion and length limits |
| `POST` | `/predict` | `{text, cursorPos, category, tone}` → ranked suggestions, detected category, tone score, latency |
| `POST` | `/classify` | `{text}` → category with per-class probabilities |

Example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Please find attached my res","tone":"professional"}'
```

```json
{
  "suggestions": [
    {"text": "resume",   "score": 0.5945, "source": "mixed"},
    {"text": "resolved", "score": 0.1705, "source": "mixed"},
    {"text": "respected","score": 0.1703, "source": "mixed"}
  ],
  "category": "job_application",
  "tone": {"label": "professional", "score": 0.2659},
  "latency_ms": 1.588
}
```

---

## Limitations

- N-gram models capture only one or two preceding tokens, not long-range context.
- Suggestions are restricted to the 261-token training vocabulary; unseen words
  map to `<UNK>` and are never suggested.
- Tone scoring is lexicon-based and hand-weighted, not learned.
- The corpus is 42 hand-written professional emails replicated 5× (210 rows) for
  balanced splits; it is not a large-scale corpus.
- Classifier probabilities are not calibrated — treat them as relative rankings.

---

## Academic credits

| | |
|---|---|
| **Author** | Prakul P Shetty |
| **SRN** | R24EI043 |
| **Institution** | REVA University |
| **Programme** | B.Tech, 5th Semester, Section IoT-A |
| **Course** | Natural Language Processing — Mini Project |
| **Academic year** | 2026–2027 |

Built with Python, scikit-learn, NLTK, Flask, and Next.js. All NLP models are
trained from scratch on a purpose-built professional email corpus.
