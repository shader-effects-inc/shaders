# Radiance

Words for light a shape gives off or blocks. Colored point lights strike the shape's
silhouette, the lit edges spill light into their surroundings, and the shape's body casts
shadows, so light stays out of a ring's hole and off the far side of a wall. They work inside
a shape-effect material, the `surface:` of `shapedSurface` from `materials`. The shape is a
**signed distance**, a function from a point to its distance from the edge, and every word
here marches rays against it.

`irradianceField` does the heavy work once: it gathers the spilled light into a texture in a
compute pass and only gathers again when a light or the shape changes, so a still scene
costs one texture read per pixel. Build its spec once, put its `compute` on your definition
and read `sample(params)` in your material. `shadowVisibility` is for light you compute
yourself at a point, such as the directly lit rim. `gatherIrradiance` is the same gather run
live per pixel, for when the light must change every frame. The light comes back as rgb.
Send it through `exposureTone` and `emissiveAlpha` for a glow that composites over what is
behind it, then mix in the body by `silhouette`.

## Reach for it when

| When | Use |
|---|---|
| colored lights spilling around a shape, with shadows | `irradianceField` |
| a light that the shape's body should block at one point | `shadowVisibility` |
| a gather that must change every frame | `gatherIrradiance` |

## Order

- irradianceField
- shadowVisibility
- gatherIrradiance

## Example

```ts
import {defineShader, listPropConfig, materials, radiance, compose, math, transformColor, transformPosition} from 'shaders/std'

const {shapedSurface, surfaceField, silhouette, exposureTone} = materials
const {irradianceField} = radiance
const {emissiveAlpha} = compose
const {local, mix, splat3, vec4} = math

// One spec for both halves: the compute gather and the fragment read.
const GATHER = {
  rays: {burst: 128, motion: 64, refine: 32, refineFrames: 12},
  steps: 32,
  shadowSteps: 12,
  resolution: 512,
  lights: {prop: 'lights', position: 'position', color: 'color', intensity: 'intensity'},
  reach: 'reach',
  wrap: 'wrap',
  lightRange: 'lightRange',
  shadowSoftness: 'shadowSoftness',
}

// Colored lights spilling around a dark shape, gathered once and read per pixel.
export const LitEdge = defineShader({
  name: 'LitEdge',
  role: 'shapeEffect',
  species: 'custom',
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    scale: {default: 1},
    rotation: {default: 0},
    lights: listPropConfig({
      maxItems: 6,
      minItems: 1,
      itemLabel: 'Light',
      item: {
        position: {kind: 'position', default: {x: 0.3, y: 0.3}, label: 'Position'},
        color: {kind: 'color', default: '#ffb347', label: 'Color'},
        intensity: {kind: 'number', default: 3, label: 'Intensity', min: 0, max: 8, step: 0.01},
      },
    }, {
      default: [
        {position: {x: 0.3, y: 0.3}, color: '#ffb347', intensity: 3},
        {position: {x: 0.72, y: 0.7}, color: '#7a5cff', intensity: 2},
      ],
      description: 'The point lights striking the shape',
      label: 'Lights',
      group: 'Lights',
    }),
    lightRange: {default: 1.2},
    reach: {default: 3},
    wrap: {default: 0},
    shadowSoftness: {default: 0.2},
    bodyColor: {default: '#000000', transform: transformColor},
    edgeSoftness: {default: 0.01},
    shape: {default: JSON.stringify({type: 'circleSDF', radius: 0.35}), ui: {type: 'shape', label: 'Shape'}},
    shapeSdfUrl: {default: '', compileTime: true},
    shapeType: {default: '', compileTime: true},
  },
  ...shapedSurface({
    stencil: 'centre',
    centreTap: 'fast',
    surface: (frame, params) => {
      const u = params.uniforms
      const field = surfaceField(frame, params)
      const spill = irradianceField(GATHER).sample(params) ?? splat3(0)
      const glow = emissiveAlpha(exposureTone(spill), 'glow')
      const cov = local(silhouette(field, u.edgeSoftness), 'cov')
      return vec4(mix(glow.member('rgb'), u.bodyColor.member('rgb'), cov), mix(glow.member('a'), 1, cov))
    },
  }),
  compute: (params) => irradianceField(GATHER).compute(params),
})
```
