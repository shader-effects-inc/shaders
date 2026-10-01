# defineShader & wgsl

A shader is a plain object: a name, the props a user can set, and one field that says what it
draws. Hand it to `defineShader` and it becomes a component you can mount, nest, blend, mask
and animate like any shader in the library.

```ts
import {defineShader, transformColor, transformPosition, wgsl} from 'shaders/std'

export const Halo = defineShader({
  name: 'Halo',                      // what the editor and preset JSON call it
  props: {                           // the controls a user can set
    color: {default: '#ffd166', transform: transformColor},
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    radius: {default: 0.6},
  },
  paint: wgsl`                       // what it draws: a color for every pixel
    let d = length((uv - center) * vec2f(aspect, 1.0)) / radius;
    return vec4f(color.rgb, 1.0 - smoothstep(0.8, 1.0, d));
  `,
})
```

**One field says what kind of shader it is.** Give exactly one.

| Field | You get | What is inside it |
|---|---|---|
| `paint:` | a generator that paints from coordinates (gradients, noise, light) | nothing: it stands on its own |
| `effect:` | a filter over the layer nested inside it (tints, blurs, ripples) | the child layer, read as `child` or sampled as `childTexture` |
| `map:` | a distortion that moves the pixels of the layer inside it | the child layer |
| `shape:` | a 2D shape with fill and stroke | nothing |

**Two ways to write what it draws.** Compose it from std words (the rest of this reference)
and the engine compiles the composition to WGSL. A radial gradient between two color props is
one line:

```ts
paint: rampOver(dist.radial({center: p('center'), radius: p('radius'), aspect: 1, skew: 0}), pair(p('inner'), p('outer'), p('colorSpace')))
```

Or write the math yourself in a `wgsl` body: one function that returns a `vec4f` color. Your
props, `uv`, `time`, `aspect`, `viewport`, `pointer` and the child are bound by name, so there
is no setup to write. A filter that inverts whatever is inside it:

```ts
effect: wgsl`
  return vec4f(1.0 - child.rgb, child.a);
`
```

**Props are the controls.** Each has a `default`, usually a `transform` that says what kind of
value it is (`transformColor`, `transformPosition`, …), and optional `ui` metadata that tells
the editor how to show it. Blend modes, opacity, masks, transforms, dynamic props and code
export come with every shader; you never write them.

**Use it like any other shader.** Mount it with `<CustomShader src={Halo}>` in React, Vue,
Svelte or Solid. Call `registerShader(Halo)` when preset JSON should be able to name it as
`type: 'Halo'`.

```tsx
import {Shader, CustomShader, Blur} from 'shaders/react' // or shaders/vue, shaders/svelte, shaders/solid
import {Halo} from './halo'

<Shader>
  <Blur intensity={8}>
    <CustomShader src={Halo} radius={0.8} />
  </Blur>
</Shader>
```

## Reach for it when

| When | Use |
|---|---|
| a shader of your own, from a plain object | `defineShader` |
| per-pixel math you want to write by hand | `wgsl` in `paint:` or `effect:` |
| a generator that paints from coordinates | `defineShader` with `paint:` |
| a filter over the layer nested inside | `defineShader` with `effect:` holding a color word, `tintToward`, or a `wgsl` body that reads child |
| a filter that reads neighbouring pixels (blur, ripple, mosaic) | a `wgsl` body that samples childTexture, or `gather` |
| a distortion from one coordinate function | `defineShader` with `map:` |
| naming your shader in preset JSON for `createShader` | `registerShader`, or the components option |
| listing the custom shaders an app has registered | `getRegisteredShaders`, `onShaderRegistered` |

## Order

- defineShader
- wgsl
- registerShader
- unregisterShader
- getRegisteredShader
- getRegisteredShaders
- onShaderRegistered
- WgslBody

## Example

```ts
import {defineShader, wgsl, transformColor, transformPosition} from 'shaders/std'

// A generator: concentric color bands radiating from a draggable point, on its own clock.
export const Halo = defineShader({
  name: 'Halo',
  description: 'Concentric color bands radiating from a point.',
  animatedTime: {speed: 'speed'},
  props: {
    inner: {default: '#ffd166', transform: transformColor, ui: {type: 'color', label: 'Inner'}},
    outer: {default: '#0b132b', transform: transformColor, ui: {type: 'color', label: 'Outer'}},
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition, ui: {type: 'position', label: 'Center'}},
    radius: {default: 0.6, ui: {type: 'range', min: 0.1, max: 1.5, step: 0.01, label: 'Radius'}},
    bands: {default: 4, ui: {type: 'range', min: 1, max: 16, step: 1, label: 'Bands'}},
    speed: {default: 1, ui: {type: 'range', min: 0, max: 4, step: 0.1, label: 'Speed'}},
  },
  paint: wgsl`
    let q = (uv - center) * vec2f(aspect, 1.0);
    let d = length(q) / radius;
    let wave = 0.5 + 0.5 * cos(d * bands * 6.2831853 - time * 2.0);
    let fade = 1.0 - smoothstep(0.7, 1.0, d);
    return vec4f(mix(outer.rgb, inner.rgb, wave * fade), 1.0);
  `,
})
```
