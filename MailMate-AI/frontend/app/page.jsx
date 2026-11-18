import Link from "next/link";

const BENCHMARKS = [
  {
    value: "94.56%",
    label: "Top-3 Next-Word Accuracy",
    foot: "45 held-out sentences",
  },
  {
    value: "77.86%",
    label: "Top-1 Next-Word Accuracy",
    foot: "45 held-out sentences",
  },
  {
    value: "100.0%",
    label: "ML Category Classification",
    foot: "80/20 stratified split",
  },
  {
    value: "< 4 ms",
    label: "Local CPU Inference Latency",
    foot: "warm process, 261-token vocab",
  },
];

export default function Home() {
  return (
    <div className="container">
      <section className="hero">
        <div>
          <span className="kicker">Statistical NLP • WordNet Lexicon • CPU Inference</span>
          <h1 className="hero-title">
            Context-Aware Predictive Text for Professional Email Composition
          </h1>
          <p className="hero-sub">
            Real-time next-word and prefix completion powered by Laplace-smoothed
            N-gram language models, NLTK WordNet tone scoring, and TF-IDF context
            classification — running locally on CPU with sub-5ms latency.
          </p>
          <div className="hero-actions">
            <Link href="/studio" className="btn btn-primary">
              Open Email Studio <span aria-hidden="true">→</span>
            </Link>
            <Link href="/how-it-works" className="btn btn-secondary">
              Explore Architecture
            </Link>
          </div>
          <p className="hero-note">
            119 KB artifact · no external API calls · deterministic ranking
          </p>
        </div>

        <div className="preview">
          <div className="preview-bar">
            <span className="window-dots" aria-hidden="true">
              <span className="window-dot" />
              <span className="window-dot" />
              <span className="window-dot" />
            </span>
            <span className="preview-title">New message</span>
            <span className="preview-meta">draft</span>
          </div>

          <div className="preview-head">
            <div className="preview-field">
              <span className="preview-label">To</span>
              <span className="preview-value">internships@company.com</span>
            </div>
            <div className="preview-field">
              <span className="preview-label">Subject</span>
              <span className="preview-value">Internship application status</span>
            </div>
          </div>

          <div className="preview-body">
            <span className="preview-text">
              I am writing to follow up regarding my internship{" "}
            </span>
            <span className="ghost-text">application</span>
            <span className="keycap">Tab ⇥</span>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="benchmarks-title">
        <div className="section-head">
          <span className="eyebrow">Measured on the shipped artifact</span>
          <h2 className="section-title" id="benchmarks-title">
            Benchmarks, not estimates
          </h2>
          <p className="section-sub">
            Every figure below is reproduced from the Phase 2 evaluation run or
            from warm-process latency sampling on the local CPU.
          </p>
        </div>

        <div className="bench-grid">
          {BENCHMARKS.map((item) => (
            <div className="bench-card" key={item.label}>
              <div className="bench-value">{item.value}</div>
              <div className="bench-label">{item.label}</div>
              <div className="bench-foot">{item.foot}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="about-title">
        <div className="section-head">
          <span className="eyebrow">Design principle</span>
          <h2 className="section-title" id="about-title">
            Explainable by construction
          </h2>
          <p className="section-sub">
            Every suggestion is the output of four inspectable components: an
            interpolated N-gram probability, a WordNet tone weight, a logistic
            regression category probability, and a corpus frequency prior. No
            hidden layers, no remote inference, no prompt strings.
          </p>
        </div>
        <div className="hero-actions">
          <Link href="/how-it-works" className="btn btn-secondary">
            Read the pipeline
          </Link>
          <Link href="/about" className="btn btn-ghost">
            Project details
          </Link>
        </div>
      </section>
    </div>
  );
}
