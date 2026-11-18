import Link from "next/link";

export const metadata = {
  title: "How It Works — MailMate",
  description:
    "The end-to-end request lifecycle and the five-stage classical NLP pipeline behind MailMate's suggestions.",
};

const FLOW = [
  {
    step: "01",
    label: "Browser Editor",
    detail: "60 ms debounce, AbortController",
  },
  {
    step: "02",
    label: "Next.js /api Proxy",
    detail: "rewrite → localhost:5000",
  },
  {
    step: "03",
    label: "Flask REST API",
    detail: ":5000 · JSON validation",
  },
  {
    step: "04",
    label: "mailmate_model.joblib",
    detail: "119 KB · loaded once in RAM",
  },
  {
    step: "05",
    label: "Top-3 Ranked JSON",
    detail: "words, scores, category, tone",
  },
];

export default function HowItWorks() {
  return (
    <div className="container">
      <div className="section-head">
        <span className="kicker">Architecture</span>
        <h1 className="section-title" style={{ fontSize: 34 }}>
          From keystroke to ranked suggestion
        </h1>
        <p className="section-sub">
          A request never leaves the machine. The editor debounces input, the
          Next.js rewrite forwards it to Flask, and a single joblib bundle held
          in memory returns the ranking.
        </p>
      </div>

      <section aria-label="Request lifecycle">
        <div className="flow">
          {FLOW.map((node, index) => (
            <div key={node.step} style={{ display: "contents" }}>
              <div className="flow-node">
                <div className="flow-step">STEP {node.step}</div>
                <div className="flow-label">{node.label}</div>
                <div className="flow-detail">{node.detail}</div>
              </div>
              {index < FLOW.length - 1 && (
                <span className="flow-arrow" aria-hidden="true">
                  →
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="pipeline-title">
        <div className="section-head">
          <span className="eyebrow">Five stages</span>
          <h2 className="section-title" id="pipeline-title">
            The NLP &amp; ML pipeline
          </h2>
          <p className="section-sub">
            Each stage is a small, testable module in <code>MailMate-AI/src</code>.
            Parameters below are the values frozen into <code>v1.0.0</code>.
          </p>
        </div>

        <div className="stage-list">
          <article className="stage-card">
            <div className="stage-num">1</div>
            <div>
              <h3 className="stage-title">Text Normalization &amp; Mode Detection</h3>
              <p className="stage-body">
                Input is lowercased and tokenized with a single regex pass,
                <code> [a-z]+</code>, so no tokenizer model is loaded at request
                time. The caret position then decides the mode: a trailing space
                means <strong>next_word</strong> prediction, while a trailing run
                of letters means <strong>partial_word</strong> prefix completion.
                A vocabulary-constrained noun-lemma normalizer mirrors the
                WordNet lemmatizer used during training.
              </p>
              <div className="params">
                <span className="param">regex <b>[a-z]+</b></span>
                <span className="param">modes <b>next_word · partial_word</b></span>
                <span className="param">NLTK at runtime <b>none</b></span>
              </div>
            </div>
          </article>

          <article className="stage-card">
            <div className="stage-num">2</div>
            <div>
              <h3 className="stage-title">
                Interpolated N-Gram Language Model + Laplace Smoothing
              </h3>
              <p className="stage-body">
                Unigram, bigram, and trigram counts are blended so that unseen
                contexts never collapse to zero probability. Add-one smoothing
                handles the sparse tail; linear interpolation weights the trigram
                most heavily because the corpus is long enough for local context
                to be informative.
              </p>
              <div className="formula">
                P<sub>Laplace</sub>(w<sub>i</sub> | context) = ( Count(context, w
                <sub>i</sub>) + 1 ) / ( Count(context) + <em>V</em> )
              </div>
              <div className="formula">
                P = <em>0.20</em> · P<sub>uni</sub> + <em>0.30</em> · P<sub>bi</sub> +{" "}
                <em>0.50</em> · P<sub>tri</sub>
              </div>
              <div className="params">
                <span className="param">vocabulary <b>261 tokens</b></span>
                <span className="param">bigrams <b>456</b></span>
                <span className="param">trigrams <b>484</b></span>
                <span className="param">perplexity <b>40.04</b></span>
              </div>
            </div>
          </article>

          <article className="stage-card">
            <div className="stage-num">3</div>
            <div>
              <h3 className="stage-title">NLTK WordNet Corpus Tone Scoring</h3>
              <p className="stage-body">
                During training, each corpus word was expanded through WordNet
                synsets and weighted per tone, then normalized to{" "}
                <code>[0, 1]</code>. The result ships as frozen dictionaries, so
                runtime tone scoring is a single dict lookup per candidate — no
                corpus read, no synset traversal.
              </p>
              <div className="params">
                <span className="param">professional <b>733 words</b></span>
                <span className="param">formal <b>702 words</b></span>
                <span className="param">polite <b>635 words</b></span>
                <span className="param">lookup <b>O(1)</b></span>
              </div>
            </div>
          </article>

          <article className="stage-card">
            <div className="stage-num">4</div>
            <div>
              <h3 className="stage-title">
                TF-IDF + Logistic Regression Context Classification
              </h3>
              <p className="stage-body">
                Unigram and bigram TF-IDF features are extracted from lemmatized
                text and classified into seven professional email intents. The
                model is a multinomial logistic regression: 646 features, seven
                weight rows, and a softmax that exposes a calibrated ranking over
                categories rather than a single opaque label.
              </p>
              <div className="params">
                <span className="param">features <b>TF-IDF 1–2 grams</b></span>
                <span className="param">classifier <b>LogisticRegression</b></span>
                <span className="param">categories <b>7</b></span>
                <span className="param">accuracy <b>100.0%</b></span>
              </div>
            </div>
          </article>

          <article className="stage-card">
            <div className="stage-num">5</div>
            <div>
              <h3 className="stage-title">Multi-Factor Candidate Ranking</h3>
              <p className="stage-body">
                Every vocabulary candidate receives a blended score. The language
                model and corpus-frequency terms are normalized across the
                candidate set, so the four weights remain comparable. A shortlist
                pass keeps the classifier call to a single batched inference over
                24 candidates.
              </p>
              <div className="formula">
                Score = <em>0.50</em>(LM) + <em>0.25</em>(Context) +{" "}
                <em>0.15</em>(Tone) + <em>0.10</em>(Freq)
              </div>
              <div className="stage-cta">
                <Link href="/studio" className="btn btn-primary btn-sm">
                  Test Live in Studio <span aria-hidden="true">→</span>
                </Link>
              </div>
            </div>
          </article>
        </div>
      </section>
    </div>
  );
}
