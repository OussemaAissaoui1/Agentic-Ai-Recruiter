import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// CV-screening sub-app. Built into ../static/screening, served by the
// unified FastAPI app at /screening/. All API calls go to /api/matching/*.
export default defineConfig({
  plugins: [react()],
  base: "/screening/",
  build: {
    outDir: "../static/screening",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
