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
    python -c "import nltk; nltk.download('punkt')"

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

---

## 4. Phase Status

### Phase 1 — Project setup
- [ ] Folder structure
- [ ] .gitignore, requirements.txt, config.json
- [ ] README.md and INSTRUCTIONS.md
- [ ] src/ stubs
- [ ] Next.js scaffold
- [ ] Git repo pushed

### Phase 2 — Core NLP (Colab)
- [ ] Notebook 01 — corpus + preprocessing
- [ ] Notebook 02 — n-gram counts
- [ ] Notebook 03 — classifier + evaluation
- [ ] models/ verified

### Phase 3 — Backend + Frontend
- [ ] backend/loader.py, backend/app.py
- [ ] frontend/ Next.js UI
- [ ] tests/test_api.py
- [ ] Manual spot-check

### Phase 4 — Documentation
- [ ] Codebase analysis
- [ ] README.md finalized
- [ ] INSTRUCTIONS.md finalized
- [ ] Report sections updated

---

## 5. Model Artifacts

Location: MailMate-AI/models/
Produced by: MailMate-AI/notebooks/01-03 (Colab)
Shipped with project: yes (~30 MB)
Corpus shipped: no

---

## 6. Workflow Log

### Completed
<!-- Agent appends here at end of each phase -->

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
| NLTK punkt missing | python -c "import nltk; nltk.download('punkt')" |
| Frontend 500 on /api/predict | Flask not running |
| npm install fails | Node version below 18 |

---

## 8. Known Limitations
- Bigram/trigram cannot capture long-range context
- Suggestions limited to 15,000-word vocabulary
- Tone scoring is lexicon-based
- Corpus is EnronSent

---

## 9. Architecture Summary
- Frontend (Next.js) — UI only, calls /api/*
- Backend (Flask) — HTTP layer, imports src/predictor.py
- NLP core (src/) — trained in Phase 2
- Artifacts (models/) — frozen from Phase 2
