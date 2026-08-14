import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    coverage: {
      enabled: true,
      include: ["web/src/decision.ts"],
      provider: "v8",
      reporter: ["text"],
      thresholds: { statements: 80 },
    },
    environment: "jsdom",
    include: ["web/tests/**/*.test.{ts,tsx}"],
    setupFiles: ["web/tests/setup.ts"],
  },
});
