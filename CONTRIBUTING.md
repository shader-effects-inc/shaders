# Contributing to Shaders

Thanks for helping. This guide covers how the repo is laid out, how to build and test it, and what a pull request needs so it can be merged.

## Code of conduct

This project follows the [Contributor Covenant](./CODE_OF_CONDUCT.md). By taking part you agree to it.

## Before you start

- **Bugs:** open an issue with the bug report form. A minimal reproduction (a preset link, a StackBlitz, or a short snippet) is the fastest route to a fix.
- **Features and new components:** open an issue first and describe what you want to build. It saves work on both sides if the design is agreed before the code exists.
- **Small fixes** (typos, doc comments, an obvious one-line bug) can go straight to a pull request.
- **Questions:** ask on [Discord](https://discord.gg/Mfqmb2jCQT).

## Setup

You need Node 22.20 and pnpm 10 (see `engines` in `package.json`).

```bash
git clone https://github.com/shader-effects-inc/shaders.git
cd shaders
pnpm install
pnpm lib:build   # builds every package and regenerates the framework components
pnpm test        # the engine's test suite
```

`pnpm lib:build` takes a minute the first time. After that, `pnpm --filter shaders dev` rebuilds the `shaders` package on change. To try a change in a real app, link the built package into it with `pnpm link ./packages/shaders` from that app, or point the app's `shaders` dependency at the folder.

## How the repo is laid out

| Path | What it is |
|---|---|
| `packages/core` | The engine: the TypeGPU renderer (`src/gpu`), the std vocabulary (`src/std`), and one folder per component under `src/shaders/<Name>/index.ts` |
| `packages/shaders` | The `shaders` npm package: the build that generates every framework entry point from core |
| `packages/vue`, `react`, `svelte`, `solid`, `js` | The framework bindings. Their component files are generated; edit the shader definition in core, not these |
| `packages/core/docs/std` | The primitives reference source, and `STYLE.md`, the guide for writing doc comments |

A component is a definition in core. The build reads every folder under `src/shaders/`, registers it, and generates the Vue, React, Svelte, Solid and JavaScript components from it. There's nothing to register by hand.

## Making a change

1. Make the change in `packages/core`.
2. Run `pnpm lib:build`. The build rewrites tracked files (the shader registry, each framework's component list, the exports maps). **Commit those changes with yours.** CI runs the build and fails the pull request if the working tree isn't clean afterwards.
3. Run `pnpm test`. Each component has a `shader-<name>.test.ts` under `packages/core/src/__tests__/gpu` that composes it and resolves the WGSL. If you changed what a shader emits on purpose, update its snapshot with `pnpm --filter shaders-core exec vitest run -u` and check the diff.
4. Typecheck the engine with `npx tsc --noEmit` from `packages/core`.
5. Lint with `pnpm --filter shaders-core lint:facade`. It enforces one rule worth knowing: shader code imports from `@coreroot/gpu/kit` and `@coreroot/gpu/porters`, never from `typegpu` directly.

### Adding a component

Start by copying the closest existing component of the same kind. A generator (draws from coordinates, like `Aurora`), a filter (changes the layers below it, like `Vignette`), a distortion (`Mirror`), or a shape effect (`Glass`). The neighbour shows the shape of the definition, the prop conventions, and which helpers to use.

Then:

- Give every prop a `default`, a `description`, and `ui` metadata (type, label, min and max for ranges). The design editor and the docs are generated from them.
- Write doc comments for readers who aren't graphics engineers. `packages/core/docs/std/STYLE.md` has the rules and a word list; the short version is one plain sentence saying what it produces, then only what changes how you'd use it.
- Add a `shader-<name>.test.ts` beside the others. The existing tests show the pattern: build a small tree, compose it, assert the body functions that should be present.
- Run `pnpm lib:build` and commit the generated files.
- Include a screenshot or a short recording in the pull request. Reviewers can't run every change on every GPU.

Renaming a component is a separate process (old names have to keep working for saved presets). Open an issue rather than renaming in a pull request.

### Code style

TypeScript in strict mode, two-space indentation, no semicolons, named imports. Match the file you're in. Inside a `'use gpu'` function body, don't reference `Math.*` (fold constants to module-level literals) and remember that integer-valued literals like `0.0` initialising a `let` transpile to `i32`; wrap them with `d.f32(0)`.

## Pull requests

- One change per pull request. A fix and a new feature are two pull requests.
- **The title is the changelog line.** Pull requests are squash-merged and the title becomes the commit, which `changelogen` turns into the release notes. Use the form `type: what changed`, where `type` is one of `feat`, `fix`, `perf`, `refactor`, `docs` or `types`. For example: `fix: Glass ignores shape rotation on Safari`.
- Fill in the pull request template. The checklist is the review in advance.
- Don't bump the version or edit `CHANGELOG.md`. Releases are prepared by maintainers with `pnpm release`, which writes both from the merged titles.
- Link the issue the change resolves, if there is one.

## Reporting a security issue

Don't open a public issue. Email simon@shaders.com with the details and we'll respond there.

## License

Shaders is [MIT licensed](./LICENSE). By contributing, you agree that your contributions are licensed the same way.
