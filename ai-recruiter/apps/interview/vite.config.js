import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Interview sub-app. Built into ../static/interview, served at /interview/.
// Talks to /api/nlp/* and opens a parallel WebSocket to /ws/vision.
export default defineConfig({
  plugins: [react()],
  base: "/interview/",
  build: {
    outDir: "../static/interview",
    emptyOutDir: true,
  },
  server: {
    host: "0.0.0.0",
    port: 3000,
    proxy: {
      "/api": {
        target: process.env.BACKEND_URL || "http://localhost:8000",
        changeOrigin: true,
        secure: false,
      },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
