export const metadata = {
  title: "About — MailMate",
  description:
    "Why MailMate uses an explainable, local classical NLP model instead of cloud LLM APIs, and the academic context behind the project.",
};

const CATEGORIES = [
  {
    name: "internship_followup",
    desc: "Checking application status, interview scheduling, candidacy updates.",
  },
  {
    name: "job_application",
    desc: "Submitting a résumé, applying for a role, aligning experience with a posting.",
  },
  {
    name: "academic_communication",
    desc: "Requesting feedback, clarifying submission guidelines, addressing faculty.",
  },
  {
    name: "project_communication",
    desc: "Progress updates, pull request reviews, integration and deployment notes.",
  },
  {
    name: "meeting_request",
    desc: "Proposing a slot, confirming availability, setting agendas.",
  },
  {
    name: "leave_request",
    desc: "Formal absence notification, medical leave, handover planning.",
  },
  {
    name: "professional_followup",
    desc: "Gentle reminders, confirming receipt, awaiting approval to proceed.",
  },
];

const STACK = [
  ["Language", "Python 3.14"],
  ["Lexical resource", "NLTK WordNet"],
  ["ML library", "Scikit-Learn 1.8.0"],
  ["API server", "Flask 3.1"],
  ["Frontend", "Next.js 16 (App Router)"],
  ["Artifact", "joblib 1.5.3 · 119 KB"],
];

const CREDITS = [
  [
    "Student",
    "Prakul P Shetty — SRN R24EI043",
  ],
  [
    "Programme",
    "5th Semester B.Tech CSE — IoT, Blockchain Technology & Cybersecurity",
  ],
  [
    "Institution",
    "School of Computer Science and Engineering, REVA University, Bengaluru",
  ],
  ["Faculty guide", "Prof. Arati Chabukswar"],
];

export default function About() {
  return (
    <div className="container">
      <div className="section-head">
        <span className="kicker">About this project</span>
        <h1 className="section-title" style={{ fontSize: 34 }}>
          Built for professional writing, not open-ended chat
        </h1>
        <p className="section-sub">
          MailMate is a classical NLP system: statistical language modelling,
          lexicon-driven tone scoring, and linear classification. It is
          deliberately small, inspectable, and offline.
        </p>
      </div>

      <section className="about-grid" aria-label="Problem and rationale">
        <div className="panel">
          <h2 className="panel-title">The problem</h2>
          <p>
            Generic next-word predictors are tuned for speed on chat text. In
            formal email they degrade quickly: the useful continuations are
            domain-specific (<em>application</em>, <em>availability</em>,{" "}
            <em>approval</em>), the register matters, and a suggestion that reads
            as informal can ruin a message to a recruiter or a professor.
          </p>
          <p>
            Professional writing is also highly patterned. A follow-up opens with
            a purpose clause, a meeting request ends with an availability
            question. That repetition is exactly what an interpolated N-gram
            model captures well, and what makes a 119 KB artifact sufficient.
          </p>
        </div>

        <div className="panel">
          <h2 className="panel-title">The engineering choice</h2>
          <p>
            A dependency-light CPU model was chosen over GPU neural generation
            and over hosted LLM APIs. Draft text stays on the machine — nothing
            is sent to a third party, so there is no data-residency question and
            no network dependency.
          </p>
          <p>
            The trade-off is explicit: no free-form generation, and no vocabulary
            outside the trained 261-token set. In return, ranking is
            deterministic, latency is sub-5ms, and every score can be traced back
            to a count table, a lexicon weight, or a logistic regression
            coefficient.
          </p>
        </div>
      </section>

      <section className="section" aria-labelledby="categories-title">
        <div className="section-head">
          <span className="eyebrow">Classification targets</span>
          <h2 className="section-title" id="categories-title">
            Seven supported email categories
          </h2>
          <p className="section-sub">
            The TF-IDF classifier returns a probability for each label; the
            studio shows the argmax while still letting you pin a category
            manually.
          </p>
        </div>

        <div className="cat-grid">
          {CATEGORIES.map((category) => (
            <div className="cat-card" key={category.name}>
              <div className="cat-name">{category.name}</div>
              <p className="cat-desc">{category.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="stack-title">
        <div className="section-head">
          <span className="eyebrow">Implementation</span>
          <h2 className="section-title" id="stack-title">
            Technology stack &amp; credits
          </h2>
        </div>

        <div className="stack-grid">
          <div className="panel">
            <h3 className="panel-title">Stack</h3>
            <div style={{ marginTop: 10 }}>
              {STACK.map(([key, value]) => (
                <div className="stack-row" key={key}>
                  <span className="stack-key">{key}</span>
                  <span className="stack-val">{value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <h3 className="panel-title">Academic credits</h3>
            <div className="credits">
              {CREDITS.map(([key, value]) => (
                <div className="credit-row" key={key}>
                  <span className="credit-key">{key}</span>
                  <span className="credit-val">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
