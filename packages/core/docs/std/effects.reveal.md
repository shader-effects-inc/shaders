# Reveals & wipes

Words that make the layer inside disappear in a chosen order: a straight edge, a growing
circle, a clock sweep, a grid of blocks, a noise dissolve. Every wipe is the same effect,
`reveal`, fed a different **coverage**: a number per pixel from 0 to 1 that says when that
pixel goes (0 first, 1 last). As the `progress` prop climbs from 0 to 1, pixels whose
coverage falls below it fade out across a soft front `feather` wide.

The `coverage` words produce the ordering. Their modifiers reshape it: `.folded()` turns one
edge into two opening outward, `.tiled(n)` repeats it into strips, `.bands(n)` cuts it into
bands that vanish whole, and `.shuffled()` randomizes the band or cell order. Read a wipe from
the inside out: the coverage, its modifiers, then `reveal` with the three timing props.

Only the child's alpha changes, so a wipe composes over anything behind it and costs one
pointwise pass. `progress` has no clock of its own; drive it from a prop.

## Reach for it when

| When | Use |
|---|---|
| a straight edge sweeping across | `coverage.directional` |
| two doors opening from the middle | `coverage.directional` with `.folded()` |
| venetian blinds or random bars | `coverage.directional` with `.tiled(n)`, or `.bands(n).shuffled()` |
| a circle growing from a point | `coverage.radial` |
| rings popping outward | `coverage.radial` with `.bands(n)` |
| a clock hand sweeping around a point | `coverage.angular` |
| blocks dissolving, a checkerboard, or a lattice of diamonds | `coverage.cells` with `.shuffled()`, `.checker()` or `.diamond()` |
| a soft film dissolve | `coverage.noise` |
| the timing props every wipe shares | `reveal` |

## Order

- reveal
- coverage
- coverage.directional
- coverage.radial
- coverage.angular
- coverage.cells
- coverage.noise
- Coverage
- BandedCoverage
- CellCoverage

## Example

```ts
import {defineShader, p, effects, transformBoolean} from 'shaders/std'

const {reveal, coverage} = effects.reveal

// Random bars: a directional wipe cut into bars that vanish in a fixed shuffled order.
export const RandomBars = defineShader({
  name: 'RandomBars',
  props: {
    progress: {default: 0.5, ui: {type: 'range', min: 0, max: 1, step: 0.01}},
    angle: {default: 90, ui: {type: 'range', min: 0, max: 360, step: 1}},
    barCount: {default: 24, ui: {type: 'range', min: 2, max: 100, step: 1}},
    softness: {default: 0.1, ui: {type: 'range', min: 0, max: 1, step: 0.01}},
    invert: {default: false, transform: transformBoolean, ui: {type: 'checkbox'}},
  },
  effect: reveal({
    coverage: coverage.directional(p('angle')).bands(p('barCount')).shuffled(),
    progress: p('progress'),
    feather: p('softness'),
    invert: p('invert'),
  }),
  missingChildMessage: 'RandomBars needs a child to wipe.',
})
```
