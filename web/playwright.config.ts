import { defineConfig, devices } from "@playwright/test";

const PORT = 18765;

export default defineConfig({
  expect: { timeout: 5_000 },
  forbidOnly: true,
  fullyParallel: false,
  outputDir: "test-results/viewer",
  projects: [
    {
      name: "desktop-chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: "narrow-chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
      },
    },
  ],
  reporter: [["line"]],
  retries: 0,
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    bypassCSP: true,
    colorScheme: "dark",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `../.venv/bin/python ../scripts/web/serve_viewer_fixture.py --port ${PORT} --web-root dist`,
    reuseExistingServer: false,
    stderr: "pipe",
    stdout: "pipe",
    timeout: 120_000,
    url: `http://127.0.0.1:${PORT}/api/v1/health`,
  },
  workers: 1,
});
