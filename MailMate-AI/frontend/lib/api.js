// API client for the MailMate AI Flask backend.
// Requests go to /api/*, which next.config.js rewrites to http://localhost:5000/*.

const API_BASE = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const message =
      (payload && (payload.message || payload.error)) ||
      `Request failed with status ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

export async function getHealth() {
  return request("/health", { method: "GET", cache: "no-store" });
}

export async function getCategories() {
  return request("/categories", { method: "GET", cache: "no-store" });
}

export async function predict(text, cursorPos, category, tone, signal) {
  return request("/predict", {
    method: "POST",
    signal,
    body: JSON.stringify({
      text,
      cursorPos: cursorPos ?? null,
      category: category ?? null,
      tone: tone ?? null,
    }),
  });
}

export async function classify(text) {
  return request("/classify", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}
