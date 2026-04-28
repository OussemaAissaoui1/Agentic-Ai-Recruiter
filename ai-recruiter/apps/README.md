# apps/ — frontends

Three Vite/React sub-apps + a static shell. Each sub-app builds into
`apps/static/<name>/`, which the unified FastAPI app mounts as a static
directory.

| Folder | Path served | API base |
|---|---|---|
| `shell/` | `/` | — (static landing) |
| `screening/` | `/screening/` | `/api/matching/*` |
| `interview/` | `/interview/` | `/api/nlp/*` + `/ws/vision` |
| `behavior/` | `/behavior/` | `/api/vision/*` + `/ws/vision` |

## Build

```bash
bash scripts/build_frontends.sh
```

This runs `npm ci && npm run build` in each sub-app. Outputs land in
`apps/static/<name>/` (gitignored).

## Local dev (per sub-app)

```bash
cd apps/screening    # or interview / behavior
npm install
npm run dev
```

Each Vite config proxies `/api` and `/ws` to `http://localhost:8000`, so
running uvicorn alongside `npm run dev` gives you the full app with
hot-reload on the frontend.

## Interview ↔ vision wiring

`apps/interview/src/vision/useVisionPipeline.js` is a small React hook
that:

1. POSTs `/api/vision/session/start` to open a behavioral session.
2. Opens a WebSocket to `/ws/vision?session_id=…`.
3. Streams 4 fps JPEG frames + 16 kHz PCM from the candidate's mic.
4. Receives the 4-D dimension scores and surfaces them via state.

`apps/interview/src/vision/BehavioralOverlay.jsx` is the floating
preview + dimension bars. After every recruiter question the interview
app fires `vision.markQuestion(text)` so the question shows up on the
behavioral timeline (visible in the report afterwards).
