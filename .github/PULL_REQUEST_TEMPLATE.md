<!-- The title becomes the changelog line: `type: what changed`, with type one of
     feat, fix, perf, refactor, docs, types. e.g. `fix: Glass ignores shape rotation on Safari` -->

## What this changes

<!-- One or two sentences. Link the issue if there is one: Closes #123 -->

## How to see it

<!-- For anything visual, a screenshot or short recording before and after.
     For a bug fix, the composition that reproduced it. -->

## Checklist

- [ ] `pnpm lib:build` ran and the generated files it changed are committed
- [ ] `pnpm test` passes (snapshots updated on purpose, if any)
- [ ] `npx tsc --noEmit` passes in `packages/core`
- [ ] `pnpm --filter shaders-core lint:facade` passes
- [ ] New or changed props have a `default`, a `description` and `ui` metadata
- [ ] Doc comments follow `packages/core/docs/std/STYLE.md`
- [ ] No version bump or `CHANGELOG.md` edit (maintainers do that at release)
