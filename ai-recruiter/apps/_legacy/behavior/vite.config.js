import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Behavioral-analytics sub-app. Built into ../static/behavior, served by
// the unified FastAPI app at /behavior/. WebSocket → /ws/vision, HTTP →
// /api/vision/*.
export default defineConfig({
  plugins: [react()],
  base: '/behavior/',
  build: {
    outDir: '../static/behavior',
    emptyOutDir: true,
  },
  server: {
    port: 3001,
    proxy: {
      '/ws': { target: 'ws://localhost:8000', ws: true },
      '/api': { target: 'http://localhost:8000' },
    },
  },
})
