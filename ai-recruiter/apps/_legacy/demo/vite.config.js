import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Served under /interview/ in the unified FastAPI app
  base: "/interview/",
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 3002,
    proxy: {
      "/api": {
        target: process.env.BACKEND_URL || "http://localhost:8002",
        changeOrigin: true,
        secure: false,
      },
      "/avatar": {
        target: process.env.BACKEND_URL || "http://localhost:8002",
        changeOrigin: true,
        secure: false,
      },
      "/ws": {
        target: process.env.BACKEND_URL || "http://localhost:8002",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
