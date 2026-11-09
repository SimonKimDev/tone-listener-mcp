import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (proxyReq, request) => {
            // Rewrite only requests originating in this loopback dev UI, never foreign origins.
            if (request.headers.host === "127.0.0.1:5173"
                && request.headers.origin === "http://127.0.0.1:5173") {
              proxyReq.setHeader("Origin", "http://127.0.0.1:8765");
            }
          });
        },
      },
    },
  },
  build: {
    outDir: "../src/tone_listener_mcp/workbench/static",
    emptyOutDir: true,
  },
});
