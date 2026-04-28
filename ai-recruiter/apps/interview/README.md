# apps/interview/ — the live interview frontend

React/Vite SPA that drives the full interview experience.

- Streams generated questions + audio from `/api/nlp/stream` (SSE).
- Sends candidate answers back to the same endpoint.
- **In parallel**, opens a WebSocket to `/ws/vision` so the candidate's
  webcam + microphone feed the vision agent for real-time behavioral
  scoring. Per-frame results show up on a small floating overlay.
- After every question is finalised, fires
  `POST /api/vision/session/{id}/question` (via the vision hook) to
  annotate the behavioral timeline.

## Files

```
src/
├── app.jsx                   # full chat app (imports vision hook + overlay)
├── main.jsx                  # ReactDOM root
└── vision/
    ├── useVisionPipeline.js  # camera + mic + /ws/vision stream
    └── BehavioralOverlay.jsx # floating preview + 4-D score bars
```

## Running

Local dev (with the unified backend on :8000):

```bash
cd apps/interview
npm install
npm run dev          # http://localhost:3000  (proxies /api and /ws to :8000)
```

Production (served by the unified app):

```bash
bash scripts/build_frontends.sh   # from repo root
# unified app already serves /interview/
```

## Permissions

The browser will prompt for camera + mic the first time you open the
interview screen. Without them the vision overlay shows an error but
the chat still works.
