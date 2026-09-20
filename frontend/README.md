# AI Knowledge Inbox — Frontend

React + TypeScript + Vite + Tailwind UI for the AI Knowledge Inbox. It talks to
the FastAPI backend in `../backend` and covers the three flows: save notes/URLs,
list saved items, and ask questions that come back with an answer plus cited
source snippets.

## Prerequisites

- Node 18+ (built and tested on Node 22)
- The backend running on `http://127.0.0.1:8000` — see `../backend`

## Run

```bash
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api/*` to the backend
(`http://127.0.0.1:8000`), so no CORS setup is needed in development. Port 5173
is already on the backend's CORS allow-list if you prefer to call it directly.

## Scripts

- `npm run dev` — start the Vite dev server
- `npm run build` — type-check (`tsc -b`) and produce a production bundle in `dist/`
- `npm run preview` — serve the production build locally
- `npm run lint` — type-check only

## Configuration

`VITE_API_BASE` overrides where API calls go (default `/api`). To point at a
backend on another origin without the dev proxy:

```bash
# frontend/.env.local
VITE_API_BASE=http://127.0.0.1:8000
```

## How it maps to the backend

| UI                         | Endpoint       |
| -------------------------- | -------------- |
| Add note / URL             | `POST /ingest` |
| Saved items list           | `GET /items`   |
| Ask a question             | `POST /query`  |

The API client (`src/api/client.ts`) parses the backend's single error envelope
(`{ "error": { "code", "message", "details" } }`) into a typed `ApiError`, so
validation messages (e.g. a malformed URL) surface inline on the form.

## Structure

```
src/
  api/         client.ts (fetch + error handling), types.ts (contract types)
  components/  IngestForm, ItemsList, AskPanel, AnswerView, ui primitives
  hooks/       useItems (list + refresh)
  lib/         format helpers
  App.tsx      layout + shared items state
```

### Answer citations

`POST /query` returns an `answer` containing `[n]` markers that line up with the
`citation` field of each entry in `sources`. `AnswerView` tokenizes the answer,
renders each `[n]` as a clickable chip, and scrolls to / highlights the matching
source card. Bracketed numbers that don't map to a real source are left as plain
text.
```
