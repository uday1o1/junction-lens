import { expect, test } from "@playwright/test";

const axePath = require.resolve("axe-core/axe.min.js");

type AxeResult = Readonly<{
  violations: readonly Readonly<{
    id: string;
    impact: string | null;
    help: string;
  }>[];
}>;

test("renders persisted graph evidence and passes browser accessibility checks", async ({
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
    if (url.protocol !== "data:" && url.hostname !== "127.0.0.1") {
      externalRequests.push(request.url());
    }
  });

  const response = await page.goto("/");
  expect(response?.headers()["content-security-policy"]).toContain(
    "script-src 'self'",
  );
  await expect(
    page.getByRole("heading", {
      name: "Synthetic intersection control regression",
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Persisted release decision")).toContainText(
    "Fail regression",
  );
  await expect(
    page.getByRole("article", { name: "Front left image restricted" }),
  ).toContainText("Image not exported");
  await expect(
    page.getByAltText(/Front center, synchronized source view/u),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /Bird's-eye lane and control graph/u }),
  ).toBeVisible();
  await expect(
    page.locator('[data-edge-type="lane_successor"]'),
  ).not.toHaveCount(0);
  await expect(
    page.locator('[data-edge-type="control_applies_to_lane"]'),
  ).not.toHaveCount(0);

  const candidate = page.getByRole("checkbox", { name: "Candidate" });
  await expect(candidate).toBeChecked();
  await candidate.uncheck();
  await expect(page.getByTestId("layer-candidate")).toHaveCount(0);
  await candidate.check();

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

  if (testInfo.project.name === "narrow-chromium") {
    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(dimensions.scrollWidth).toBe(dimensions.clientWidth);
    await expect(
      page.getByRole("navigation", { name: "Frame navigation" }),
    ).toBeVisible();
  }
  await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath("viewer.png"),
  });
});

test("navigates synchronized frames with buttons and keyboard", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("frame-00")).toBeVisible();

  const next = page.getByRole("button", { name: /Next frame/u });
  await next.click();
  await expect(page.getByText("frame-01")).toBeVisible();
  await expect(next).toBeDisabled();

  await page.keyboard.press("ArrowLeft");
  await expect(page.getByText("frame-00")).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Previous frame/u }),
  ).toBeDisabled();
  expect(pageErrors).toEqual([]);
});
