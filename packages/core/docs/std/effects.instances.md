# Instances

Repeat the layer inside. `repeatInstances` draws many copies of it, laid out in a grid, around
a circle or along a line, each copy scaled, turned, faded and jittered a little more than the
last. It returns the fragment for `gpu.fragment`. The layout words (`grid`, `radial`, `line`)
say where the copies go, `byMode` lets one select prop switch between all three, and the
recipe's `variation` object says how the copies differ.

The definition sets `requiresRTT`, `requiresChild` and `wantsBoundsParams`: copies are cut from
the layer's own bounds and spread across this layer's bounding box. Radius, spacing and gaps
are fractions of the canvas, layout angles are in degrees, and the per-copy `rotation` and
`hueShift` are in radians (give those props a degrees-to-radians transform).

## Reach for it when

| When | Use |
|---|---|
| copies in rows and columns, brick-staggered or mirrored | `grid` |
| copies around a circle, or along an arc | `radial` |
| copies trailing off in one direction | `line` |
| one select prop that switches between layouts | `byMode` |
| the copies themselves, with their variation | `repeatInstances` |

## Order

- repeatInstances
- grid
- radial
- line
- byMode

## Example

```ts
import {defineShader, p, effects, crosses, transformBoolean} from 'shaders/std'

const {repeatInstances, radial} = effects.instances

const degToRad = (v: number) => (v * Math.PI) / 180

// Orbit: copies of the layer inside around a circle, each one smaller, fainter and hue-shifted.
export const Orbit = defineShader({
  name: 'Orbit',
  requiresRTT: true,
  requiresChild: true,
  wantsBoundsParams: true,
  props: {
    count: {default: 8},
    radius: {default: 0.3},
    startAngle: {default: 0},
    sweep: {default: 360},
    faceCenter: {default: false, transform: transformBoolean},
    instanceScale: {default: 0.9},
    instanceRotation: {default: 0, transform: degToRad},
    instanceOpacity: {default: 0.9},
    jitterPosition: {default: 0},
    jitterRotation: {default: 0},
    jitterScale: {default: 0},
    jitterOpacity: {default: 0},
    seed: {default: 0},
    phase: {default: 0},
    hueShift: {default: 30, transform: degToRad, recompile: crosses(0)},
    zOrder: {default: 'forward', transform: (v: string) => (v === 'backward' ? 1 : 0), compileTime: true},
    cropLeft: {default: 0},
    cropRight: {default: 0},
    cropTop: {default: 0},
    cropBottom: {default: 0},
  },
  gpu: {fragment: repeatInstances({
    source: {crop: {left: p('cropLeft'), right: p('cropRight'), top: p('cropTop'), bottom: p('cropBottom')}},
    placement: radial({count: p('count'), radius: p('radius'), startAngle: p('startAngle'), sweep: p('sweep'), faceCenter: p('faceCenter')}),
    variation: {
      scale: p('instanceScale'), rotation: p('instanceRotation'), opacity: p('instanceOpacity'),
      jitter: {position: p('jitterPosition'), rotation: p('jitterRotation'), scale: p('jitterScale'), opacity: p('jitterOpacity')},
      seed: p('seed'), phase: p('phase'),
    },
    hueShift: p('hueShift'),
    order: p('zOrder'),
  })},
})
```
