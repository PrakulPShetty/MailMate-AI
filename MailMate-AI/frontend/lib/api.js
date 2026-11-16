// API client for MailMate AI backend.
// Calls go to /api/*, which Next.js rewrites to Flask at localhost:5000/*

export async function getCategories() {
  throw new Error('Not implemented — Phase 3');
}

export async function predict(text, cursorPos, category, tone) {
  throw new Error('Not implemented — Phase 3');
}

export async function classify(text) {
  throw new Error('Not implemented — Phase 3');
}

export async function getHealth() {
  throw new Error('Not implemented — Phase 3');
}
