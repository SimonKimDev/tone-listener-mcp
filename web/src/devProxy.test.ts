import { createServer as createHttpServer, request } from "node:http";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createServer, type ViteDevServer } from "vite";
import config from "../vite.config";

let backend: ReturnType<typeof createHttpServer>;
let vite: ViteDevServer;
let url: string;
let targetHost: string;

beforeAll(async () => {
  backend = createHttpServer((request, response) => {
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify({ host: request.headers.host, origin: request.headers.origin }));
  });
  await new Promise<void>((resolve) => backend.listen(0, "127.0.0.1", resolve));
  const address = backend.address();
  if (!address || typeof address === "string") throw new Error("Missing backend address.");
  targetHost = `127.0.0.1:${address.port}`;
  const proxy = config.server!.proxy!["/api"];
  if (!proxy || typeof proxy === "string") throw new Error("Expected guarded API proxy.");
  vite = await createServer({
    configFile: false,
    optimizeDeps: { noDiscovery: true, entries: [] },
    server: {
      host: "127.0.0.1", port: 0, strictPort: false,
      proxy: { "/api": { ...proxy, target: `http://${targetHost}` } },
      watch: null, hmr: false,
    },
  });
  await vite.listen();
  const viteAddress = vite.httpServer?.address();
  if (!viteAddress || typeof viteAddress === "string") throw new Error("Missing Vite address.");
  url = `http://127.0.0.1:${viteAddress.port}/api/analysis/preflight`;
});

afterAll(async () => {
  await vite?.close();
  if (backend) {
    backend.closeAllConnections();
    await new Promise<void>((resolve, reject) => backend.close((error) => error ? reject(error) : resolve()));
  }
});

describe("loopback development proxy", () => {
  it.each([
    ["127.0.0.1:5173", "http://127.0.0.1:5173", "http://127.0.0.1:8765", 200],
    ["127.0.0.1:5173", "https://foreign.example", "https://foreign.example", 200],
    ["127.0.0.1:5173", "null", "null", 200],
    ["localhost:5173", "http://127.0.0.1:5173", "http://127.0.0.1:5173", 200],
    ["foreign.example", "http://127.0.0.1:5173", null, 403],
  ])("checks host %s and origin %s before rewriting", async (host, origin, forwardedOrigin, status) => {
    const response = await new Promise<{ status: number | undefined; body: string }>((resolve, reject) => {
      const outgoing = request(url, { method: "POST", headers: { Host: host, Origin: origin } }, (incoming) => {
        let body = "";
        incoming.setEncoding("utf8");
        incoming.on("data", (chunk: string) => { body += chunk; });
        incoming.on("end", () => resolve({ status: incoming.statusCode, body }));
        incoming.on("error", reject);
      });
      outgoing.on("error", reject);
      outgoing.end();
    });
    expect(response.status).toBe(status);
    if (status === 200) {
      expect(JSON.parse(response.body)).toEqual({ host: targetHost, origin: forwardedOrigin });
    } else {
      expect(response.body).toContain("Blocked request");
    }
  });
});
