import axe from "axe-core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import { SCENE_DETAIL, SCENE_PAGE } from "./fixtures";

function response(value: object): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("thin evidence viewer", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(response(SCENE_PAGE))
        .mockResolvedValueOnce(response(SCENE_DETAIL)),
    );
    window.history.replaceState({}, "", "/");
  });

  afterEach(() => vi.unstubAllGlobals());

  it("renders the persisted decision, graph layers, and restricted camera state", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(
      screen.getByRole("heading", { name: "Loading registered scene" }),
    ).toBeVisible();
    expect(
      await screen.findByRole("heading", { name: SCENE_DETAIL.bundle.title }),
    ).toBeVisible();
    expect(screen.getByText("Fail regression")).toBeVisible();
    await user.click(screen.getByText("1 reason code"));
    expect(screen.getByText("GATE_REGRESSION_CI_BELOW_MARGIN")).toBeVisible();
    expect(
      screen.getByRole("article", { name: "Front left image restricted" }),
    ).toBeVisible();
    expect(
      screen.getByAltText(/Front center, synchronized source view/u),
    ).toBeVisible();
    expect(screen.getByTestId("layer-ground-truth")).toBeInTheDocument();
    expect(screen.getByTestId("layer-baseline")).toBeInTheDocument();
    expect(screen.getByTestId("layer-candidate")).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: "Candidate" }));
    expect(screen.queryByTestId("layer-candidate")).not.toBeInTheDocument();
  });

  it("navigates frames with semantic controls and arrow keys", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("heading", { name: SCENE_DETAIL.bundle.title });

    const next = screen.getByRole("button", { name: /Next frame/u });
    await user.click(next);
    expect(screen.getByText("frame-01")).toBeVisible();
    expect(next).toBeDisabled();

    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(screen.getByText("frame-00")).toBeVisible();
  });

  it("has no automated accessibility violations in the ready state", async () => {
    const { container } = render(<App />);
    await screen.findByRole("heading", { name: SCENE_DETAIL.bundle.title });

    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("renders stable empty and error states", async () => {
    vi.mocked(fetch)
      .mockReset()
      .mockResolvedValueOnce(
        response({
          schema_version: "junctionlens.api-artifact-page.v1",
          items: [],
        }),
      );
    const { rerender } = render(<App />);
    expect(
      await screen.findByRole("heading", {
        name: "The evidence viewer is ready",
      }),
    ).toBeVisible();

    vi.mocked(fetch)
      .mockReset()
      .mockResolvedValueOnce(new Response("", { status: 409 }));
    rerender(<App key="error" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 409");
    await waitFor(() =>
      expect(screen.getByText(/not recalculated/u)).toBeVisible(),
    );
  });
});
