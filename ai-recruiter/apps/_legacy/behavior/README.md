# apps/behavior/ — vision dashboard frontend

React/Vite SPA that opens a standalone vision session — useful for
debugging behavioral signals outside an interview. Talks to
`/api/vision/*` and `/ws/vision`.

```bash
cd apps/behavior
npm install
npm run dev          # http://localhost:3001
```

Production build → `apps/static/behavior/` via
`bash scripts/build_frontends.sh`. Served by the unified app at
`/behavior/`.
