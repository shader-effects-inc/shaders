# Lens

Words for the color fringing a real lens leaves behind: the red, green and blue copies of
the layer inside pushed apart, or fanned out into a whole spectrum. Because they read the
child at several places per pixel, these are **gather effects**: the child is rendered to a
texture first and the effect samples it. `chromaticFan` is a sampling stage; hand it to
`gatherStack` (from `effects.blurs`) to turn it into the `effect:` of a filter definition,
optionally followed by more stages. `spectralLens` is a complete fragment for a `gpu:`
definition, with the lens warp, focus zones and grain built in.

Angles are in degrees and go through `transformAngle`. Strengths are 0–1.

## Reach for it when

| When | Use |
|---|---|
| a subtle RGB split along one direction | `chromaticFan` inside `gatherStack` |
| the full prism look: many color copies, a bulging lens, swirl and grain | `spectralLens` |

## Order

- chromaticFan
- spectralLens

## Example

```ts
import {defineShader, p, effects, transformAngle} from 'shaders/std'

const {chromaticFan} = effects.lens
const {gatherStack} = effects.blurs

// Chromatic aberration: red and blue pulled apart along an angle, green held in place.
export const Fringe = defineShader({
  name: 'Fringe',
  props: {
    strength: {default: 0.2, ui: {type: 'range', min: 0, max: 1, step: 0.01}},
    angle: {default: 0, transform: transformAngle, ui: {type: 'range', min: 0, max: 360, step: 1}},
    redOffset: {default: -1, ui: {type: 'range', min: -2, max: 2, step: 0.1}},
    greenOffset: {default: 0, ui: {type: 'range', min: -2, max: 2, step: 0.1}},
    blueOffset: {default: 1, ui: {type: 'range', min: -2, max: 2, step: 0.1}},
  },
  effect: gatherStack(chromaticFan({
    strength: p('strength'),
    angle: p('angle'),
    red: p('redOffset'),
    green: p('greenOffset'),
    blue: p('blueOffset'),
  }), []),
  missingChildMessage: 'Fringe needs a child to split.',
})
```
