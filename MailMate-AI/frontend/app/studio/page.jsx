"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getCategories, getHealth, predict } from "../../lib/api";

const MAX_TEXT_LEN = 2000;
const DEBOUNCE_MS = 60;

const FALLBACK_CATEGORIES = [
  "academic_communication",
  "internship_followup",
  "job_application",
  "leave_request",
  "meeting_request",
  "professional_followup",
  "project_communication",
];

const CATEGORY_LABELS = {
  internship_followup: "Internship Follow-up",
  job_application: "Job Application",
  academic_communication: "Academic Communication",
  project_communication: "Project Communication",
  meeting_request: "Meeting Request",
  leave_request: "Leave Request",
  professional_followup: "Professional Follow-up",
};

const TONES = [
  { value: "professional", label: "Professional" },
  { value: "formal", label: "Formal" },
  { value: "polite", label: "Polite" },
];

const TEMPLATES = [
  {
    label: "Internship Follow-up",
    text: "I am writing to follow up regarding my internship ",
  },
  {
    label: "Job Application",
    text: "Please find attached my resume and cover letter for the ",
  },
  {
    label: "Leave Request",
    text: "I am writing to formally request medical leave for ",
  },
];

function formatCategory(value) {
  if (!value) return "Pending";
  return (
    CATEGORY_LABELS[value] ||
    value
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  );
}

function trailingPrefix(value) {
  const match = /([A-Za-z]+)$/.exec(value);
  return match ? match[1] : "";
}

function isNextWordMode(value) {
  if (!value) return true;
  if (/\s$/.test(value)) return true;
  return trailingPrefix(value) === "";
}

export default function StudioPage() {
  const [text, setText] = useState("");
  const [category, setCategory] = useState("auto");
  const [tone, setTone] = useState("professional");
  const [categoryOptions, setCategoryOptions] = useState(FALLBACK_CATEGORIES);
  const [suggestions, setSuggestions] = useState([]);
  const [detectedCategory, setDetectedCategory] = useState("");
  const [latency, setLatency] = useState(null);
  const [connected, setConnected] = useState(null);
  const [copied, setCopied] = useState(false);

  const textareaRef = useRef(null);
  const backdropRef = useRef(null);
  const abortRef = useRef(null);
  const lastPredictionRef = useRef(null);
  const pendingImmediateRef = useRef(null);
  const copiedTimerRef = useRef(null);

  // Connection + category metadata.
  useEffect(() => {
    let alive = true;
    getHealth()
      .then(() => alive && setConnected(true))
      .catch(() => alive && setConnected(false));
    getCategories()
      .then((data) => {
        if (alive && data && Array.isArray(data.categories) && data.categories.length) {
          setCategoryOptions(data.categories);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const runPrediction = useCallback(
    async (value) => {
      if (abortRef.current) abortRef.current.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const data = await predict(
          value,
          value.length,
          category === "auto" ? null : category,
          tone,
          controller.signal
        );
        if (controller.signal.aborted) return;

        const nextSuggestions = Array.isArray(data.suggestions)
          ? data.suggestions
          : [];
        setSuggestions(nextSuggestions);
        setDetectedCategory(data.category || "");
        setLatency(
          typeof data.latency_ms === "number" ? data.latency_ms : null
        );
        setConnected(true);
        lastPredictionRef.current = { text: value, suggestions: nextSuggestions };
      } catch (error) {
        if (error && error.name === "AbortError") return;
        setConnected(false);
        setSuggestions([]);
        setDetectedCategory("");
        setLatency(null);
      }
    },
    [category, tone]
  );

  useEffect(() => {
    if (!text.trim()) {
      setSuggestions([]);
      setDetectedCategory("");
      setLatency(null);
      lastPredictionRef.current = null;
      return undefined;
    }

    const immediate = pendingImmediateRef.current === text;
    if (immediate) pendingImmediateRef.current = null;

    const timer = setTimeout(() => {
      runPrediction(text);
    }, immediate ? 0 : DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [text, category, tone, runPrediction]);

  // Instant local filtering of the previous result while a request is in flight.
  const displayed = useMemo(() => {
    const previous = lastPredictionRef.current;
    if (!previous || previous.text === text) return suggestions;
    if (text.startsWith(previous.text)) {
      const added = text.slice(previous.text.length);
      if (/^[A-Za-z]+$/.test(added)) {
        const filtered = previous.suggestions.filter((item) =>
          item.text.toLowerCase().startsWith(added.toLowerCase())
        );
        if (filtered.length) return filtered;
      }
    }
    return suggestions;
  }, [text, suggestions]);

  const ghost = useMemo(() => {
    const top = displayed[0];
    if (!top) return "";
    if (isNextWordMode(text)) {
      const separator = text && !/\s$/.test(text) ? " " : "";
      return `${separator}${top.text}`;
    }
    const prefix = trailingPrefix(text);
    if (top.text.toLowerCase().startsWith(prefix.toLowerCase())) {
      return top.text.slice(prefix.length);
    }
    return "";
  }, [displayed, text]);

  const acceptSuggestion = useCallback(
    (word) => {
      if (!word) return;

      let next;
      if (isNextWordMode(text)) {
        const separator = text && !/\s$/.test(text) ? " " : "";
        next = `${text}${separator}${word} `;
      } else {
        const prefix = trailingPrefix(text);
        next = `${text.slice(0, text.length - prefix.length)}${word} `;
      }

      pendingImmediateRef.current = next;
      setText(next);
      requestAnimationFrame(() => {
        const element = textareaRef.current;
        if (element) {
          element.focus();
          element.setSelectionRange(next.length, next.length);
        }
      });
    },
    [text]
  );

  const handleKeyDown = (event) => {
    if (event.key === "Tab" && ghost) {
      event.preventDefault();
      acceptSuggestion(displayed[0].text);
      return;
    }
    if (
      event.key === "ArrowRight" &&
      ghost &&
      event.currentTarget.selectionStart === event.currentTarget.selectionEnd &&
      event.currentTarget.selectionStart === text.length
    ) {
      event.preventDefault();
      acceptSuggestion(displayed[0].text);
    }
  };

  const handleScroll = () => {
    if (backdropRef.current && textareaRef.current) {
      backdropRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  };

  const handleChange = (event) => {
    setText(event.target.value.slice(0, MAX_TEXT_LEN));
  };

  const applyTemplate = (value) => {
    pendingImmediateRef.current = value;
    setText(value);
    requestAnimationFrame(() => {
      const element = textareaRef.current;
      if (element) {
        element.focus();
        element.setSelectionRange(value.length, value.length);
      }
    });
  };

  const clearDraft = () => {
    setText("");
    setSuggestions([]);
    setDetectedCategory("");
    setLatency(null);
    textareaRef.current?.focus();
  };

  const copyEmail = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const statusLabel =
    connected === null ? "Checking" : connected ? "Connected" : "Backend Offline";

  return (
    <div className="container">
      <div className="studio-head">
        <div>
          <h1 className="studio-title">Email Studio</h1>
          <p className="studio-sub">
            Type a message and accept the inline suggestion with Tab. Scores are
            blended from the N-gram model, TF-IDF context classifier, WordNet tone
            lexicon, and corpus frequency.
          </p>
        </div>
      </div>

      <section className="studio-bar" aria-label="Composer controls">
        <div className="control-group">
          <label className="control-label" htmlFor="category">
            Email category
          </label>
          <select
            id="category"
            className="select"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="auto">Auto-Detect (ML)</option>
            {categoryOptions.map((value) => (
              <option key={value} value={value}>
                {formatCategory(value)}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <span className="control-label">Communication tone</span>
          <div className="segmented" role="group" aria-label="Communication tone">
            {TONES.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`seg${tone === option.value ? " is-active" : ""}`}
                aria-pressed={tone === option.value}
                onClick={() => setTone(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className="telemetry" aria-live="polite">
          <span className="tele-pill">
            <span
              className={`status-dot ${
                connected ? "dot-online" : "dot-offline"
              }`}
              aria-hidden="true"
            />
            {statusLabel}
          </span>
          <span className="tele-pill">
            Detected: <strong>{formatCategory(detectedCategory)}</strong>
          </span>
          <span className="tele-pill">
            <strong>{latency != null ? `${latency.toFixed(1)} ms` : "—"}</strong>
          </span>
        </div>
      </section>

      <section className="editor-shell" aria-label="Email draft">
        <div className="editor-chrome">
          <span className="window-dots" aria-hidden="true">
            <span className="window-dot" />
            <span className="window-dot" />
            <span className="window-dot" />
          </span>
          <span className="editor-chrome-label">New message</span>
          <span className="editor-chrome-right">
            {text.length} / {MAX_TEXT_LEN}
          </span>
        </div>

        <div className="editor-wrap">
          <div className="editor-backdrop" ref={backdropRef} aria-hidden="true">
            {text}
            {ghost ? <span className="editor-ghost">{ghost}</span> : null}
            {ghost ? <span className="keycap">Tab ⇥</span> : null}
          </div>
          <textarea
            ref={textareaRef}
            className="editor-input"
            value={text}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onScroll={handleScroll}
            maxLength={MAX_TEXT_LEN}
            spellCheck={false}
            aria-label="Email draft text"
            placeholder="Start writing an email…"
          />
        </div>

        <div className="editor-hint">
          <span>
            <kbd>Tab</kbd> accepts the inline suggestion
          </span>
          <span>
            <kbd>→</kbd> accepts when the cursor is at the end
          </span>
          <span>Suggestions update after a 60 ms debounce</span>
        </div>
      </section>

      <div className="suggest-bar" aria-live="polite">
        {displayed.length ? (
          displayed.map((item, index) => (
            <button
              key={`${item.text}-${index}`}
              type="button"
              className="chip"
              onClick={() => acceptSuggestion(item.text)}
              title={`Insert "${item.text}"`}
            >
              <span className="chip-rank">{index + 1}</span>
              <span className="chip-text">{item.text}</span>
              <span className="chip-score">{item.score.toFixed(3)}</span>
            </button>
          ))
        ) : (
          <span className="suggest-empty">
            {connected === false
              ? "No suggestions — the prediction service is unreachable."
              : text.trim()
                ? "No candidates for the current prefix."
                : "Suggestions appear as you type."}
          </span>
        )}
      </div>

      {connected === false ? (
        <div className="notice" role="status">
          Backend Offline — start the Flask API with{" "}
          <code>python MailMate-AI/backend/app.py</code> and reload this page.
        </div>
      ) : null}

      <section className="demo-row" aria-label="Demo controls">
        <span className="demo-label">Starters</span>
        {TEMPLATES.map((template) => (
          <button
            key={template.label}
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => applyTemplate(template.text)}
          >
            {template.label}
          </button>
        ))}
        <span className="spacer" />
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={copyEmail}
          disabled={!text}
        >
          {copied ? "Copied!" : "Copy Email"}
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={clearDraft}
          disabled={!text}
        >
          Clear
        </button>
      </section>
    </div>
  );
}
