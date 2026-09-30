# Shaders

<img alt="image" src="https://shaders.com/og.jpg" />

<p align="center">
  <a href="https://www.npmjs.com/package/shaders" rel="noopener noreferrer nofollow" ><img src="https://img.shields.io/npm/v/shaders?color=0368FF&label=version" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/shaders" rel="noopener noreferrer nofollow" ><img src="https://img.shields.io/npm/dm/shaders?color=8D30FF&label=npm" alt="npm downloads per month"></a>
  <a target="_blank" rel="noopener noreferrer nofollow" href="https://www.jsdelivr.com/package/npm/shaders"><img alt="jsDelivr hits (npm)" src="https://img.shields.io/jsdelivr/npm/hm/shaders?logo=jsdeliver&color=FF4FBA"></a>
  <a href="https://twitter.com/intent/user?screen_name=npm_i_shaders" target="_blank"><img alt="X (formerly Twitter) Follow" src="https://img.shields.io/twitter/follow/npm_i_shaders"></a>
</p>

## The design platform for web shaders

Shaders is the design platform for creating production-ready WebGPU effects, used by over 15,000 design engineers. Design visually on an infinite canvas at [shaders.com](https://shaders.com), then ship your work as declarative components for Vue / Nuxt, React / Next, Svelte, Solid, or vanilla JS — all from this single package.

**Try it at [shaders.com](https://shaders.com), or watch [video tutorials](https://shaders.com/resources)** — and join the community on [Discord](https://discord.gg/Mfqmb2jCQT).

## Write your own shader

Every component in this library is a plain TypeScript definition, and yours can be too. Define it with `defineShader` from `shaders/std`, then mount it with `<CustomShader>` from your framework's entry — no build plugin, no bundler config. The declarative parts (props, animated time, blend modes, masks, drivers, export) are handled by the engine; only the per-pixel math is yours, either composed from the std vocabulary (public alpha) or written as a raw WGSL body:

```ts
// halo.ts
import {defineShader, wgsl, transformColor, transformPosition} from 'shaders/std'

export const Halo = defineShader({
  name: 'Halo',
  animatedTime: {speed: 'speed'},
  props: {
    inner:  {default: '#ffd166', transform: transformColor},
    outer:  {default: '#0b132b', transform: transformColor},
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    radius: {default: 0.6},
    bands:  {default: 4},
    speed:  {default: 1},
  },
  paint: wgsl`
    let d = length((uv - center) * vec2f(aspect, 1.0)) / radius;
    let wave = 0.5 + 0.5 * cos(d * bands * 6.2831853 - time * 2.0);
    return vec4f(mix(outer.rgb, inner.rgb, wave * (1.0 - smoothstep(0.7, 1.0, d))), 1.0);
  `,
})
```

```tsx
import {Shader, CustomShader, Blur} from 'shaders/react'   // or shaders/vue, shaders/svelte, shaders/solid
import {Halo} from './halo'

<Shader>
  <Blur intensity={8}>
    <CustomShader src={Halo} radius={0.8} bands={6} />
  </Blur>
</Shader>
```

Inside a `wgsl` body, the identifiers you reference are bound for you: every prop by name (colors are `vec4f`, positions `vec2f`, numbers `f32`) plus `uv`, `time`, `aspect`, `viewport` and `pointer`. `paint:` makes a generator; `effect:` makes a filter, which also receives the child — as `child` for a per-pixel edit, or as `childTexture` + `childSampler` when the body samples neighbours (that reference is what makes it a gather filter with its own render pass). Return a `vec4f`.

Custom definitions also work by name in the framework-free path: `createShader(canvas, preset, {components: [Halo]})`.

---

Shaders © Shader Effects, Inc.

The engine, the component library, and the framework bindings in this repository and the `shaders` npm package are open source under the [MIT License](./LICENSE).

The design editor, presets, sections, and other platform features at [shaders.com](https://shaders.com) are separate from this package and have their own [terms](https://shaders.com/license).
