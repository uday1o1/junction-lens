# JunctionLens Agent Instructions

## Authority

Read `BUILD_PLAN.md` completely before changing implementation files.
Treat `BUILD_PLAN.md` as the product and acceptance authority, subject only to an accepted ADR that resolves an evidenced contradiction or obsolete dependency.
Do not weaken, skip, or silently reinterpret a milestone gate.

## Engineering rules

- Never use the em dash.
- Preserve user changes and avoid destructive Git or filesystem operations.
- Never manually modify `CHANGELOG.md` files or files marked as generated unless explicitly asked.
- Put each full sentence on its own physical line when writing or substantially editing long Markdown files.
- Prefer quality, simplicity, robustness, scalability, and long-term maintainability over short-term speed.
- Reproduce bugs through the closest real caller path before fixing them.
- Test user-visible behavior through the public CLI, API, report, or browser path as applicable.
- Treat lint failures, test failures, and flaky tests as failures.
- Keep repository content project-facing and reusable.
- Never include credentials, private stakeholder context, personal paths, restricted datasets, model artifacts, engines, profiler reports, or working notes in Git.
- Use current official documentation for library, SDK, API, CLI, container, and platform behavior.

## Data and generated files

The repository must not download OpenLane-V2 data until the user has accepted all upstream terms through the repository acknowledgment workflow.
Dataset images, annotations, processed data, thumbnails, checkpoints, engines, profiles, and private evidence remain outside Git.
Generated protobuf files are build outputs unless an accepted language-specific tooling decision requires a checked-in generated file.
Do not edit generated outputs directly.

## Verification and Git

Run the repository-owned local checks before every commit and push.
Commit only a focused unit whose local gate has passed.
For target-qualified work, follow ADR 0001 and distinguish an implementation qualification commit from an acceptance evidence commit.
Use Conventional Commits and never add an agent as a co-author.
Push each verified focused commit to the existing private `origin` and confirm the remote commit matches before starting the next focused unit.

## Current documentation lookup

Use the `ctx7` CLI for current library, framework, SDK, API, CLI, and cloud-service documentation.
Resolve the official library first with `npx ctx7@latest library <name> "<question>"`, then fetch the selected version with `npx ctx7@latest docs <library-id> "<question>"`.
Use no more than three Context7 commands for one question.
Run Context7 outside the default sandbox and report quota failures instead of silently relying on remembered behavior.
