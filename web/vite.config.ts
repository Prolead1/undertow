import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const API_TARGET = "http://127.0.0.1:8050";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        ws: true,
      },
      "/openapi.json": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
  },
  test: {
    environment: "jsdom",
    globals: true,
    css: true,
    restoreMocks: true,
    clearMocks: true,
  },
});
