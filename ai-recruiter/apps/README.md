# apps/ — frontends

| Folder | Status | Notes |
|---|---|---|
| `hireflow/` | **active** — sole user-facing frontend | React 19 + TanStack Router SPA, mounted at `/` by `main.py` |
| `_legacy/` | archived | Old screening / interview / behavior / shell apps + `voice/demo`. Kept for reference only |

## HireFlow (active)

```bash
cd apps/hireflow
npm install
npm run build      # emits apps/hireflow/dist/, picked up by main.py
npm run dev        # or run dev with vite proxy → uvicorn on :8000
```

The Vite config proxies `/api`, `/ws`, `/avatar`, `/files` to
`http://127.0.0.1:8000` so `npm run dev` + `uvicorn main:app` gives you a
hot-reloading frontend against the real backend.

## Production layout

`main.py` mounts:

- `/assets/*` → `apps/hireflow/dist/assets/` (hashed bundles)
- everything else (non-API) → `apps/hireflow/dist/index.html` via SPA fallback

So any TanStack Router route (`/app`, `/c/apply/job-1`, `/c/interview/123`)
is served by the same `index.html` and resolved client-side.

## Backend touch points

The HireFlow client talks to:

- `/api/recruit/*` — jobs, applications, notifications, JD generation, interviews
- `/api/matching/{parse,rank,health}` — CV parse + scoring
- `/api/nlp/{session,chat,stream,tts}` — interview turns
- `/api/vision/session/start`, `/ws/vision` — behavioral analysis
- `/api/transcribe`, `/ws/transcribe` — speech-to-text
- `/avatar/{model.glb,visemes}` — 3D avatar
- `/api/status`, `/api/health` — readiness probes

## Legacy

`apps/_legacy/` contains the previous frontends (`screening/`, `shell/`,
`behavior/`, `static/` build outputs, and `agents/voice/demo/` moved here
from `agents/voice/`). They are no longer mounted. Delete the folder when
you're confident HireFlow has feature parity.
