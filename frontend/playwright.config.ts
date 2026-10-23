import { defineConfig } from "@playwright/test";

// End-to-end tests run the real FastAPI server (serving the built frontend) and drive a real browser.
// Backend model mode is controlled by E2E_MODEL_BACKEND (default: demo, i.e. fake models; set "real" for real models).
const port = Number(process.env.E2E_PORT ?? 8765);
const backend = process.env.E2E_MODEL_BACKEND ?? "demo";
const tmp = process.env.E2E_TMP ?? "../data/e2e";

export default defineConfig({
  testDir: "./e2e",
  timeout: backend === "real" ? 600_000 : 120_000,
  expect: { timeout: backend === "real" ? 480_000 : 30_000 },
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel: process.env.E2E_BROWSER_CHANNEL ?? "chrome",
    headless: true,
    screenshot: "only-on-failure",
  },
  outputDir: "../data/e2e/test-results",
  webServer: {
    command: `rm -rf ${tmp} && mkdir -p ${tmp} && cd .. && RC_MODEL_BACKEND=${backend} RC_WORKER_MODE=inline ` +
      `RC_DATABASE_URL=sqlite:///frontend/${tmp}/e2e.db RC_STORAGE_ROOT=frontend/${tmp}/storage ` +
      `RC_DETECTOR_WEIGHTS=data/models/yolo11n.pt RC_LOG_JSON=false ` +
      `.venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port ${port}`,
    url: `http://127.0.0.1:${port}/api/v1/health/ready`,
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
