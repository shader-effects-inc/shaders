# Agent-format example components

One component per feature of the agent format (`planning/AGENT_SHADER_FORMAT_PLAN.md`). They are
written exactly as an agent would write them: plain TypeScript importing `shaders/std` and
`shaders/std/wgsl`. Three checks run against them:

- `npx tsc -p examples/agent` type-checks them against the real std types.
- `src/__tests__/gpu/std-document.test.ts` parses each one without running it, evaluates the IR,
  and checks that the result matches running the file for real (the drift guard).
- Paste any of them into the editor's custom component tool to see it render.
