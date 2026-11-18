<h1 align="center">Shaders</h1>
<h3 align="center">
  Design WebGPU effects with your AI agent,<br />then ship them as components in your web projects
</h3>

<p align="center">
  <a href="https://www.npmjs.com/package/shaders" rel="noopener noreferrer nofollow"><img src="https://img.shields.io/npm/v/shaders?color=0368FF&label=version" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/shaders" rel="noopener noreferrer nofollow"><img src="https://img.shields.io/npm/dm/shaders?color=8D30FF&label=npm" alt="npm downloads per month"></a>
  <a target="_blank" rel="noopener noreferrer nofollow" href="https://www.jsdelivr.com/package/npm/shaders"><img alt="jsDelivr hits (npm)" src="https://img.shields.io/jsdelivr/npm/hm/shaders?logo=jsdeliver&color=FF4FBA"></a>
  <img alt="NPM License" src="https://img.shields.io/npm/l/shaders?color=FF2B6E">
</p>

<a href="https://shaders.com"><img alt="Shaders" src="https://shaders.com/og.jpg" /></a>

Shaders is a desktop app for designing interactive effects with your coding agent. You ask Claude Code, Cursor, Codex or any other MCP agent for what you want, watch it build on an infinite canvas, and adjust it by hand. When it looks right, your agent installs it in your project as ordinary component code.

That code runs on the engine in this repository: a WebGPU renderer and 200+ components (gradients, noise, glass, metal, light, distortions, transitions, blurs, cursor effects) that you nest, blend and mask like any other component. The engine and components are MIT licensed and published as the `shaders` npm package. You can use them without the app, and everything the app makes runs on them.

```bash
npm install shaders
```

## Contents

1. [Design with your agent](#design-with-your-agent)
2. [Use the components directly](#use-the-components-directly)
3. [The CLI](#the-cli)
4. [Writing Shaders code with an agent](#writing-shaders-code-with-an-agent)
5. [Build your own component](#build-your-own-component)
6. [Free and Pro](#free-and-pro)
7. [License](#license)
8. [Contribute](#contribute)

## Design with your agent

[Download Shaders](https://shaders.com/download) for macOS, Windows or Linux and create an account. Then connect your agent:

```bash
npx shaders mcp
```

The command finds the agents on your machine and adds the Shaders MCP server to each one. Sign in with the same account as the app when your agent asks. Keep the app open while your agent works, since it builds in the editor you have open.

Then tell it what you're designing, for example "Design three shader ideas for my hero background." Each idea gets its own frame on the canvas, so you can compare them side by side. Click any shader to see its layers and change colors, speed and other settings yourself, or ask your agent for changes in plain words. If the shader is for a specific place, like a hero or a pricing card, your agent can draw the page around it so you see it in context.

If nothing in the library fits, your agent can write a new component and give it props you can adjust in the editor. It can also add entrance, exit, hover and scroll animations.

Once a shader is ready, ask your agent to install it. It writes the shader as a component for your framework and adds it to your page, where you and your agent can keep editing it like the rest of your code.

[Getting started](https://shaders.com/docs/guide) · [Working with your agent](https://shaders.com/docs/guide/mcp)

## Use the components directly

`<Shader>` renders the canvas. Its children are layers, drawn top to bottom and blended on the GPU.

```jsx
import { Shader, LinearGradient, CursorTrail } from 'shaders/react'

<Shader className="w-full h-64">
  <LinearGradient colorA="#0f172a" colorB="#7c3aed" />
  <CursorTrail />
</Shader>
```

The same components and props work everywhere. Import them from `shaders/react`, `shaders/vue`, `shaders/svelte` or `shaders/solid`, which also covers projects built on those, like Next.js, Nuxt, Astro and Framer. Without a framework, use `createShader` from `shaders/js`:

```javascript
import { createShader } from 'shaders/js'

await createShader(document.getElementById('my-shader'), {
  components: [
    { type: 'LinearGradient', props: { colorA: '#0f172a', colorB: '#7c3aed' } },
    { type: 'CursorTrail' },
  ],
})
```

Everything is typed, props update reactively, and the components are safe to server render. Visitors need a browser with WebGPU; without it the canvas stays transparent and the rest of the page works as normal.

[Quickstart](https://shaders.com/docs/guide/quickstart) · [All 200+ components](https://shaders.com/docs/components), each with a live preview and every prop documented

## The CLI

The CLI links a codebase to your Shaders account, so the shaders you design land in your project as component files and stay in sync. It runs through `npx`, so there's nothing to install globally, and your agent can run it for you.

```bash
npx shaders                # guided menu: shows what's set up and offers the next step
npx shaders connect        # link this codebase to a Shaders project
npx shaders install        # pick shaders from that project; writes component files
npx shaders search "..."   # find a shader in the Pro library by describing it
npx shaders update         # pull in what changed in the editor
npx shaders open           # open your project in the Shaders app
npx shaders mcp            # add the Shaders MCP server to your agents
```

It detects your framework, writes to your components folder, and records what it installed in a lock file.

[CLI guide](https://shaders.com/docs/guide/cli)

## Writing Shaders code with an agent

For work in your codebase away from the canvas, give your agent the engine's skill: `npx shaders skill`, or `npx skills add shader-effects-inc/shaders`. The [skill](skills/shaders/SKILL.md) covers how layers stack and blend, masking, dynamic props, shape effects, every component, and writing new ones. `npx shaders connect` adds it for you. When the Shaders MCP is connected too, the skill tells your agent to design on the canvas and come back to the code once the shader is installed.

[shaders.com/llms.txt](https://shaders.com/llms.txt) indexes every docs page, and [shaders.com/llms-full.txt](https://shaders.com/llms-full.txt) has the full component reference with props, defaults and ranges.

[Agent skill guide](https://shaders.com/docs/guide/agent-skill)

## Build your own component

A component is a plain object passed to `defineShader`: a name, its props, and what to draw at each pixel. Mount it with `<CustomShader>` like any other component.

```ts
import { defineShader, wgsl, transformColor, transformPosition } from 'shaders/std'

export const Halo = defineShader({
  name: 'Halo',
  props: {
    color: { default: '#ffd166', transform: transformColor },
    center: { default: { x: 0.5, y: 0.5 }, transform: transformPosition },
    radius: { default: 0.6 },
  },
  paint: wgsl`
    let d = length((uv - center) * vec2f(aspect, 1.0)) / radius;
    return vec4f(color.rgb, 1.0 - smoothstep(0.8, 1.0, d));
  `,
})
```

You can write the pixel part in WGSL (stable) or compose it from the std primitives (experimental). This is also what your agent writes when it designs a custom component in the app.

[Custom components](https://shaders.com/docs/guide/custom-shaders) · [Primitives reference](https://shaders.com/docs/primitives)

## Free and Pro

The engine and components in this repository are free, with no account needed. The desktop app is free too: design on the canvas, export what you make, and let your agent make 200 edits a week. Reading shaders and installing code don't count toward that.

[Shaders Pro](https://shaders.com/pricing) is billed monthly or yearly and adds:

-   Unlimited agent edits (fair use applies)
-   A library of 1,000+ pro shaders to start from, which you and your agent can search and remix in the app or install with the CLI
-   Version history and larger uploads
-   Priority support and a Discord role

## License

-   The engine, every component and the framework bindings in this repository and the `shaders` npm package are [MIT licensed](./LICENSE).
-   The Shaders desktop app, the Shaders Pro library and other platform features at shaders.com are separate from this package and have their own [terms](https://shaders.com/license).

## Contribute

Issues and pull requests are welcome. Read the [contributing guide](./CONTRIBUTING.md) before starting, and for anything more than a bug fix, open an issue to discuss it before the PR.

The engine lives in `packages/core`, with one folder per component under `src/shaders/`. The framework packages are generated from it.

```bash
pnpm install
pnpm lib:build   # build every package (regenerates the registry and framework components)
pnpm test        # the engine's test suite
```

Say hello on [Discord](https://discord.gg/Mfqmb2jCQT).

---

Shaders © Shader Effects, Inc.
