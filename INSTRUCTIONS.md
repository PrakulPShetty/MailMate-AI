# INSTRUCTIONS.md — MailMate AI

> Living document. Updated at the end of each phase.
> For academic report content, see the PDF submission.

---

## 1. Prerequisites
- Python 3.14 (local)
- Node.js 18+ (for Next.js frontend)
- No GPU required — classical NLP runs on CPU

---

## 2. One-Time Setup

### Backend (Python)
    cd MailMate-AI
    python -m venv venv
    venv\Scripts\activate       (Windows)
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt_tab')"

### Frontend (Next.js)
    cd MailMate-AI/frontend
    npm install

---

## 3. Running the Project

Two terminals required. Run from the INNER folder.

### Terminal 1 — Backend
    cd MailMate-AI
    venv\Scripts\activate
    python backend/app.py
    # Flask on http://localhost:5000

### Terminal 2 — Frontend
    cd MailMate-AI/frontend
    npm run dev
    # Next.js on http://localhost:3000

Open http://localhost:3000

### Production build (optional)
    cd MailMate-AI/frontend
    npm run build
    npm run start

### Verification
    python MailMate-AI/verify_models.py
    pytest MailMate-AI/tests/test_api.py

---

## 4. Phase Status

### Phase 1 — Project setup
- [x] Folder structure
- [x] .gitignore, requirements.txt, config.json
- [x] README.md and INSTRUCTIONS.md
- [x] src/ stubs
- [x] Next.js scaffold
- [x] Git repo pushed

### Phase 2 — Core NLP (Colab)
- [x] Notebook 01 — preprocessing, n-gram counts, classifier + evaluation
- [x] models/mailmate_model.joblib produced (v1.0.0, vocab 261)
- [x] data/processed/labeled_emails.csv produced (210 rows, 7 categories)
- [x] config.json updated to v1.0.0
- [x] models/ verified

### Phase 3 — Backend + Frontend
- [x] src/ prediction engine (preprocessing, LM, tone, classifier, predictor)
- [x] backend/loader.py, backend/app.py
- [x] frontend/ Next.js UI (Overview, How It Works, About, Studio)
- [x] verify_models.py
- [x] tests/test_api.py (23 passed)
- [x] Manual spot-check + full-stack proxy test

### Phase 4 — Documentation
- [ ] Codebase analysis
- [ ] README.md finalized
- [ ] INSTRUCTIONS.md finalized
- [ ] Report sections updated

---

## 5. Model Artifacts

Location: MailMate-AI/models/
Produced by: MailMate-AI/notebooks/01_mailmate_nlp_training.ipynb (Colab)
Shipped with project: yes (~119 KB)
Corpus shipped: no

---

## 6. Workflow Log

### Completed
<!-- Agent appends here at end of each phase -->
- Phase 1 — folder structure, config, docs, src/ stubs, Next.js scaffold committed.
  Removed duplicate `next.config.mjs` and `app/page.js`; pinned ML dependencies;
  verified flask/nltk/pytest install and NLTK `punkt_tab` resource.
- Phase 2 — Colab notebook `01_mailmate_nlp_training.ipynb` trained the n-gram LMs
  and TF-IDF classifier; artifacts (`models/mailmate_model.joblib`, `data/processed/
  labeled_emails.csv`, `config.json`) moved into the repo and verified: artifact
  loads, version 1.0.0, vocab 261, `classifier.classes_` === `config.categories`.
- Phase 3 — implemented the CPU prediction engine in `src/`, the Flask API
  (`backend/app.py` with `/health`, `/categories`, `/predict`, `/classify`),
  `verify_models.py`, a 23-case pytest suite, and the Next.js 16 UI (Overview,
  How It Works, About, Studio). Verified `npm run build`, `verify_models.py`,
  `pytest`, and a live `.joblib` → Flask → Next proxy request round trip.

### In Progress
<!-- Agent updates this -->

### Pending
<!-- Agent updates this -->

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Port 5000 in use | lsof -i :5000 or change Flask port |
| Port 3000 in use | Next.js uses 3001 automatically |
| ModuleNotFoundError | Activate venv first |
| models/ not found | Run verify_models.py |
| NLTK punkt_tab missing | python -c "import nltk; nltk.download('punkt_tab')" |
| Frontend 500 on /api/predict | Flask not running |
| npm install fails | Node version below 18 |

---

## 8. Known Limitations
- Bigram/trigram cannot capture long-range context
- Suggestions limited to the 261-token vocabulary seen in training; unknown
  words are mapped to `<UNK>` and never suggested
- Tone scoring is lexicon-based, not learned
- Training corpus is 42 hand-written professional email samples, replicated 5x
  (210 rows) for balanced splits — it is not a large-scale corpus
- Classifier confidence is not calibrated; treat `predict_proba` as a relative
  ranking rather than a probability estimate

---

## 9. Architecture Summary
- Frontend (Next.js) — UI only, calls /api/*
- Backend (Flask) — HTTP layer, imports src/predictor.py
- NLP core (src/) — trained in Phase 2
- Artifacts (models/) — frozen from Phase 2
