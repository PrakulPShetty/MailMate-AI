# MailMate — Frontend

Next.js 16 App Router UI for the MailMate predictive text engine. All inference
happens in the Flask backend; this app only renders and forwards requests.

## Routes

| Route | Purpose |
|---|---|
| `/` | Overview: benchmarks, composer preview |
| `/how-it-works` | Request lifecycle and the five-stage NLP pipeline |
| `/about` | Rationale, categories, stack and credits |
| `/studio` | Interactive composer with inline ghost text |

## Development

```bash
npm install
npm run dev     # http://localhost:3000
```

The Flask API must be running on port 5000. `next.config.js` rewrites
`/api/:path*` to `http://localhost:5000/:path*`, so the browser only ever talks
to port 3000.

## Production build

```bash
npm run build
npm run start
```

Project-wide setup, phase tracking and troubleshooting live in the repository
root: `README.md` and `INSTRUCTIONS.md`.
