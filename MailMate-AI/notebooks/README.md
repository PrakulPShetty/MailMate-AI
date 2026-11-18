# MailMate AI — NLP & Machine Learning Notebooks (Phase 2)

This directory contains the Google Colab training and evaluation notebook for **MailMate AI**, a context-aware predictive text generator for professional email communication.

## Notebook Overview

| File | Description | Execution Environment |
| :--- | :--- | :--- |
| `01_mailmate_nlp_training.ipynb` | End-to-end data preprocessing, WordNet lexical expansion, N-gram language modeling, TF-IDF + Logistic Regression context classification, quantitative evaluation, and artifact serialization. | Google Colab (CPU) |

---

## Methodology Implemented

1. **Corpus Preparation & Tokenization (Step 2):**
   - Built a balanced dataset of **210 professional email samples** across **7 communication categories** (`internship_followup`, `job_application`, `academic_communication`, `project_communication`, `meeting_request`, `leave_request`, `professional_followup`) and **3 tones** (`professional`, `formal`, `polite`).
   - Normalized text to lowercase and tokenized sentences using NLTK `word_tokenize` (`punkt_tab`).

2. **NLTK WordNet Corpus Integration (Step 3):**
   - **Lemmatization:** Applied `WordNetLemmatizer` to reduce inflected words to canonical root forms prior to TF-IDF feature extraction.
   - **Semantic Tone Lexicon:** Queried WordNet synsets (`wordnet.synsets`) to expand tone-specific vocabularies (`733` professional, `702` formal, and `635` polite terms) and pre-computed normalized weights into static dictionaries for sub-10ms CPU inference at runtime.

3. **Interpolated N-Gram Language Model (Step 4):**
   - Trained Unigram, Bigram, and Trigram frequency models over a **261-token vocabulary** (`456` unique bigrams, `484` unique trigrams).
   - Applied **Laplace (Add-One) Smoothing** to prevent zero-probability failures on unseen word combinations and **Linear Interpolation** ($\lambda_{\text{trigram}} = 0.50$, $\lambda_{\text{bigram}} = 0.30$, $\lambda_{\text{unigram}} = 0.20$).

4. **Supervised ML Context Classifier & Evaluation (Step 5):**
   - Extracted unigram/bigram **TF-IDF** vectors and trained a **Logistic Regression** classifier using an 80/20 stratified split (`random_state=42`).

---

## Quantitative Evaluation Results

| Metric | Measured Value |
| :--- | :--- |
| **ML Context Classifier Accuracy** | `100.00%` |
| **ML Weighted Precision / Recall / F1** | `100.00% / 100.00% / 100.00%` |
| **N-Gram Top-1 Next-Word Accuracy** | `77.86%` |
| **N-Gram Top-3 Next-Word Accuracy** | `94.56%` |
| **Language Model Perplexity** | `40.04` |

---

## Exported Artifacts (Step 6)

Running `01_mailmate_nlp_training.ipynb` generates and validates the following production files consumed by Phase 3:

- `../models/mailmate_model.joblib` (`119.04 KB`) — Serialized bundle containing the N-gram count tables, WordNet tone lexicon, fitted `TfidfVectorizer`, and trained `LogisticRegression` classifier (pinned to `scikit-learn==1.8.0`).
- `../data/processed/labeled_emails.csv` (`21.21 KB`) — Cleaned training dataset with category and tone labels.
- `../../config.json` (`0.52 KB`) — Synchronized runtime configuration (`version: "1.0.0"`, `vocab_size: 261`, scoring weights).

## Reproducing in Google Colab

1. Upload `01_mailmate_nlp_training.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially (**Runtime $\rightarrow$ Run all**). Note that Cell 1 pins `scikit-learn==1.8.0` and `joblib==1.5.3` to guarantee binary compatibility with the local Flask backend.
3. Download the exported `.joblib`, `.csv`, and `config.json` files from the Colab file browser into their respective repository paths.