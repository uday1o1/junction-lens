import { expect, test } from "@playwright/test";
import path from "node:path";
import { pathToFileURL } from "node:url";

const axePath = require.resolve("axe-core/axe.min.js");

type AxeResult = Readonly<{
  violations: readonly Readonly<{
    id: string;
    impact: string | null;
    help: string;
  }>[];
}>;

test("opens the evidence report directly from disk without external assets", async ({
  page,
}, testInfo) => {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  const externalRequests: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol !== "file:" && url.protocol !== "data:") {
      externalRequests.push(request.url());
    }
  });
  const reportPath = path.resolve(
    "web/test-results/offline-report/REPORT.html",
  );

  await page.goto(pathToFileURL(reportPath).href);

  await expect(
    page.getByRole("heading", { name: "JunctionLens Evidence Report" }),
  ).toBeVisible();
  await expect(page.getByLabel("Persisted release status")).toHaveText(
    "FAIL_REGRESSION",
  );
  await expect(
    page.getByRole("heading", { name: "Gating cells" }),
  ).toBeVisible();
  await expect(
    page.getByText("control_edge_recall", { exact: true }),
  ).toBeVisible();
  await page.addScriptTag({ path: axePath });
  const accessibility = await page.evaluate(async () => {
    const axe = (
      window as unknown as Window & {
        axe: { run(root: Document): Promise<AxeResult> };
      }
    ).axe;
    return axe.run(document);
  });
  expect(accessibility.violations).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBe(dimensions.clientWidth);
  await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath("offline-report.png"),
  });
});
