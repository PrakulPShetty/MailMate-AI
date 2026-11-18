# INSTRUCTIONS.md — MailMate AI

A complete local run guide for this repository. Everything runs on CPU; no GPU
or cloud account is required.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or newer | Tested on 3.10–3.14 |
| Node.js | 18 or newer | Required by Next.js 16 (npm ships with Node) |
| Git | any recent version | Only needed to clone the repository |

No GPU is used. Inference is pure Python/scikit-learn on the CPU.

---

## 2. First-time setup (when cloning this public repository)

### Step 1 — Clone and install Python packages

```bash
git clone https://github.com/PrakulPShetty/MailMate-AI.git
cd MailMate-AI
pip install -r requirements.txt
```

Optional but recommended, use a virtual environment first:

```bash
python -m venv venv
# Windows (Git Bash):  source venv/Scripts/activate
# macOS / Linux:       source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Generate the model artifact (one time, ~10 seconds)

The trained model and the processed dataset are **not committed** to this public
repository (they are gitignored binary/data artifacts), so a fresh clone has an
empty `MailMate-AI/models/` folder. Pick **one** of these two options:

**Option A — Run the training notebook (recommended, reproduces the results)**

1. Open `MailMate-AI/notebooks/01_mailmate_nlp_training.ipynb`
   in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Choose *Runtime → Run all*. The whole notebook finishes in seconds.
3. It writes three files:
   - `mailmate_model.joblib` — the model artifact
   - `labeled_emails.csv` — the processed training corpus
   - `config.json` — the runtime contract (already in the repo; keep the
     committed version unless the notebook reports a different version)
4. Copy the artifacts into the repository layout expected by the backend:

   ```
   MailMate-AI/models/mailmate_model.joblib
   MailMate-AI/data/processed/labeled_emails.csv
   config.json                      (repository root)
   ```

   The notebook's final cells mirror this exact folder structure, so a copy of
   its output folder can be dropped straight in.

**Option B — Supply an existing artifact**

If you already have `mailmate_model.joblib`, just place it at:

```
MailMate-AI/models/mailmate_model.joblib
```

The backend validates the artifact against `config.json` at startup, so both
must come from the same training run (`version` and `vocab_size` must match).

> The notebook also installs NLTK corpora (`punkt_tab`, `wordnet`, `omw-1.4`).
> Those are only required for **training**. Local inference needs no NLTK data
> downloads because the tone lexicon is pre-computed inside the artifact.

Sanity-check the artifact before continuing:

```bash
python MailMate-AI/verify_models.py
```

You should see `RESULT: ALL CHECKS PASSED`.

### Step 3 — Install frontend packages

```bash
cd MailMate-AI/frontend
npm install
cd ../..
```

---

## 3. Starting the website locally (2 terminals)

### Terminal 1 — Flask backend on port 5000

Run from the repository root:

```bash
python MailMate-AI/backend/app.py
```

Expected output:

```
Model artifact loaded from .../MailMate-AI/models/mailmate_model.joblib
 * Running on http://127.0.0.1:5000
```

Quick check: <http://localhost:5000/health>

### Terminal 2 — Next.js frontend on port 3000

```bash
cd MailMate-AI/frontend
npm run dev
```

Expected output:

```
▲ Next.js 16.4.6
- Local: http://localhost:3000
✓ Ready
```

### Open the app

| Page | URL | Purpose |
|---|---|---|
| Overview | <http://localhost:3000> | Project landing page |
| Studio | <http://localhost:3000/studio> | Live ghost-text email composer |
| How it works | <http://localhost:3000/how-it-works> | Pipeline walkthrough |
| About | <http://localhost:3000/about> | Credits and scope |

The Next.js rewrite forwards every `/api/*` request to
`http://localhost:5000/*`, so the browser needs no CORS configuration. In the
Studio, type an email: the top prediction appears as inline grey ghost text
(press `Tab` or `→` to accept), with the live ML category, tone score, latency
in milliseconds, and the Top-3 clickable chips all served by the Flask API.

If Flask is not running, `/studio` shows a **Backend Offline** badge and an
instruction notice instead of suggestions — it never fabricates predictions.

### Production build (optional)

```bash
cd MailMate-AI/frontend
npm run build
npm run start
```

---

## 4. Verification commands

Run both from the repository root:

```bash
# 1. Artifact contract + sample inference (no server needed)
python MailMate-AI/verify_models.py

# 2. Flask API test suite (health, categories, predict, classify, errors)
python -m pytest MailMate-AI/tests/test_api.py -q
```

Expected results:

```
RESULT: ALL CHECKS PASSED
23 passed
```

End-to-end spot check with the servers running (through the Next.js proxy):

```bash
curl http://localhost:3000/api/health
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Please find attached my res","tone":"professional"}'
```

---

## 5. Troubleshooting

| Problem | Cause / Fix |
|---|---|
| `ModelLoadError: model artifact not found` | `MailMate-AI/models/mailmate_model.joblib` is missing — complete Step 2 (run the notebook or copy the artifact). |
| `artifact validation failed: version mismatch` | `config.json` and the `.joblib` come from different training runs. Re-run the notebook and copy both outputs. |
| `ModuleNotFoundError: flask` / `sklearn` | Dependencies not installed, or the wrong virtual environment is active. Run `pip install -r requirements.txt` and re-activate the venv. |
| Port 5000 already in use | Stop the other process (`lsof -i :5000` on macOS/Linux, `netstat -ano \| findstr :5000` on Windows) or change the port in `MailMate-AI/backend/app.py` **and** the rewrite in `MailMate-AI/frontend/next.config.js`. |
| Port 3000 already in use | Next.js automatically falls back to 3001 — use the URL it prints. |
| `/studio` shows **Backend Offline** | Flask is not running on port 5000, or it crashed at startup — check Terminal 1. |
| Suggestions are empty for unknown words | The vocabulary is the 261 tokens seen in training. Words outside it map to `<UNK>` and are never suggested. |
| `npm install` fails | Node.js is older than 18, or a corporate proxy blocks the registry. |
| Notebook cannot download NLTK data | Run the notebook's first cell again, or execute `python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"`. |

---

## 6. Repository layout and tracked files

```
MailMate-AI/
├── src/                         Runtime NLP engine (portable, pathlib-based)
├── backend/                     Flask REST API + artifact loader
├── tests/                       Pytest API suite (23 tests)
├── frontend/                    Next.js 16 App Router UI
├── models/.gitkeep              Artifact folder (mailmate_model.joblib is gitignored)
├── data/
│   ├── raw/.gitkeep             Raw corpus folder (contents gitignored)
│   └── processed/.gitkeep       Processed folder (labeled_emails.csv is gitignored)
├── notebooks/                   Phase 2 training notebook (artifact generator)
└── verify_models.py             Standalone artifact verifier
config.json                      Runtime contract: version, categories, tones, weights
requirements.txt                 Pinned Python dependencies
```

Two placeholder files (`.gitkeep`) keep the empty `models/` and
`data/processed/` folders visible on GitHub. Your local `mailmate_model.joblib`
and `labeled_emails.csv` still exist on disk and keep working — they are simply
not pushed to the public repository.

All Python file paths resolve dynamically through
`pathlib.Path(__file__).resolve()`, so the project runs from any clone location
and on any operating system.
