#!/usr/bin/env node

import process from "node:process";

import { chromium } from "@playwright/test";

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index === -1 || index + 1 >= process.argv.length) {
    throw new Error(`missing ${name}`);
  }
  return process.argv[index + 1];
}

const url = argument("--url");
const screenshot = argument("--screenshot");
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
  });
  const pageErrors = [];
  const consoleErrors = [];
  const externalRequests = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("request", (request) => {
    const destination = new URL(request.url());
    if (
      destination.protocol !== "data:" &&
      destination.hostname !== "127.0.0.1"
    ) {
      externalRequests.push(request.url());
    }
  });
  const response = await page.goto(url, { waitUntil: "networkidle" });
  if (response === null || !response.ok()) {
    throw new Error(`viewer returned ${response?.status() ?? "no response"}`);
  }
  const heading = page.getByRole("heading", {
    name: "Synthetic swapped lane-control associations",
  });
  await heading.waitFor({ state: "visible" });
  const decision = page.getByLabel("Persisted release decision");
  if (!(await decision.textContent())?.includes("Blocked infrastructure")) {
    throw new Error(
      "viewer did not render the truthful blocked release status",
    );
  }
  if (
    (await page
      .locator('[data-edge-type="control_applies_to_lane"]')
      .count()) === 0
  ) {
    throw new Error("viewer did not render lane-control graph edges");
  }
  if (!(await page.getByRole("checkbox", { name: "Candidate" }).isChecked())) {
    throw new Error("candidate comparison layer is not visible by default");
  }
  if (pageErrors.length || consoleErrors.length || externalRequests.length) {
    throw new Error(
      JSON.stringify({ pageErrors, consoleErrors, externalRequests }),
    );
  }
  await page.screenshot({ fullPage: true, path: screenshot });
  process.stdout.write(
    `${JSON.stringify({ state: "ACCEPTED", screenshot, url })}\n`,
  );
} finally {
  await browser.close();
}
