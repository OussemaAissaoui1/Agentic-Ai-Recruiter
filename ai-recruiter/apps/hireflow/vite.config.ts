import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

const apiTarget = process.env.VITE_API_PROXY || "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    react(),
    tailwindcss(),
    tsconfigPaths(),
  ],
  server: {
    port: 5173,
    proxy: {
      "/api":      { target: apiTarget, changeOrigin: true },
      "/ws":       { target: apiTarget, changeOrigin: true, ws: true },
      "/avatar":   { target: apiTarget, changeOrigin: true },
      "/files":    { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
