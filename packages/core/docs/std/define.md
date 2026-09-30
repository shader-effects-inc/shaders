# defineShader & wgsl

Every shader in the library is a plain TypeScript object handed to `defineShader`, and yours
can be too. The object names the shader, lists its **props** (the attributes a user sets), and
carries one field that says what it draws: `paint:` for a generator that paints from
coordinates, `effect:` for a filter over the layer nested inside it, `map:` for a distortion,
`shape:` for a 2D shape. Blend modes, opacity, masks, transforms, prop drivers and code export
come from the engine; you only write the per-pixel part.

There are two ways to write that part. Compose it from **std words** (`rampOver` over a
`noiseField`, `tintToward` with a `radialMask`) and the engine compiles the composition to
WGSL for you. Or break out to a **`wgsl` body**: one WGSL function that returns a `vec4f`
color, where your props and `uv`, `time`, `aspect`, `viewport`, `pointer` and the child are
bound by name. Both give the same kind of definition, which mounts as `<CustomShader
src={Halo}>` in React, Vue, Svelte or Solid, or by name in preset JSON once registered:

```tsx
import {Shader, CustomShader, Blur} from 'shaders/react' // or shaders/vue, shaders/svelte, shaders/solid
import {Halo} from './halo'

<Shader>
  <Blur intensity={8}>
    <CustomShader src={Halo} radius={0.8} bands={6} />
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
| telling a raw body apart from a composed paint | `isWgslBody` |

## Order

- defineShader
- wgsl
- registerShader
- unregisterShader
- getRegisteredShader
- getRegisteredShaders
- onShaderRegistered
- isWgslBody
- WgslBody
- defineStd

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
