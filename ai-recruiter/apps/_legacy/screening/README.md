# apps/screening/ — CV screening frontend

React/Vite/TS SPA for the matching agent. JD textarea + multi-CV upload +
optional GA constraints. Calls `/api/matching/{health, rank, extract-signals}`.

```bash
cd apps/screening
npm install
npm run dev          # http://localhost:5173
```

Production build → `apps/static/screening/` via
`bash scripts/build_frontends.sh`. Served by the unified app at
`/screening/`.
