# MailMate AI

Context-aware predictive text generation for professional email composition.

**Course:** Natural Language Processing — Mini Project
**Student:** Prakul P Shetty (R24EI043)
**Semester:** 5th, Section IoT-A
**Academic Year:** 2026–2027

---

> **Status:** Feature complete through Phase 3. Full written documentation is
> finalized in **Phase 4**.
> See `INSTRUCTIONS.md` for the operator manual and workflow log.

---

## Project phases
- [x] Phase 1 — Project setup
- [x] Phase 2 — Core NLP (Colab)
- [x] Phase 3 — Backend + Frontend
- [ ] Phase 4 — Documentation

---

## What it does

MailMate suggests the next word, or completes the word you are typing, inside a
professional email draft. Suggestions come from four inspectable signals:

| Signal | Weight | Method |
|---|---|---|
| Language model | 0.50 | Interpolated unigram/bigram/trigram, Laplace smoothing |
| Context | 0.25 | TF-IDF + Logistic Regression over 7 email categories |
| Tone | 0.15 | WordNet-expanded lexicon (professional / formal / polite) |
| Frequency | 0.10 | Corpus unigram prior |

Everything runs locally on CPU. The shipped model artifact is ~119 KB, and warm
inference is under 4 ms.

---

## Quick start

Two terminals are required.

**1. Backend (Flask, port 5000)**

```bash
cd MailMate-AI
pip install -r ../requirements.txt
python backend/app.py
```

**2. Frontend (Next.js, port 3000)**

```bash
cd MailMate-AI/frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The Next.js rewrite forwards `/api/*` to Flask, so
no CORS configuration is needed in the browser.

Verify the model independently at any time:

```bash
python MailMate-AI/verify_models.py
pytest MailMate-AI/tests/test_api.py
```

---

## Repository layout

```
MailMate-AI/
├── src/            Runtime NLP engine (preprocessing, language model,
│                   tone scorer, context classifier, predictor, evaluation)
├── backend/        Flask REST API + artifact loader
├── tests/          Pytest API suite
├── frontend/       Next.js 16 App Router UI (4 pages)
├── models/         mailmate_model.joblib (v1.0.0)
├── notebooks/      Phase 2 Colab training notebook
├── data/           Raw corpus (ignored) and processed training CSV
└── verify_models.py
config.json         Runtime contract: version, categories, tones, weights
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, model version, uptime |
| GET | `/categories` | Categories, tones, length limits |
| POST | `/predict` | `{text, cursorPos, category, tone}` → ranked suggestions |
| POST | `/classify` | `{text}` → category with per-class probabilities |
