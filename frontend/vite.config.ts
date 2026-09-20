import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server runs on 5173, which the backend's CORS allow-list already
// whitelists. /api is proxied to the FastAPI server so the frontend can use
// same-origin relative paths in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
