# Stylize

Hand-made looks for the layer inside: a watercolor wash, a chalk drawing on a board, ASCII
characters, and copper-plate line engraving. Each look is a recipe of two or three words. A
**stage** is one piece of the look that reads the layer once it is rendered to a texture (a
wobbled sample position, a brush, a ring of edge samples, a character grid). Build a stage once
and hand the same one to every word that needs it, so they agree. The closing word (`wash`,
`chalkSketch`, `linework`) returns the effect for `effect:`. The ASCII look closes with
`glyphTint`, which returns a fragment for `gpu.fragment` because it also needs a character
atlas drawn on the CPU.

The output is straight alpha. Watercolor, chalk and ASCII keep the alpha of the layer inside;
engraving replaces the layer, so its definition sets `blendWithChildren: false`.

## Reach for it when

| When | Use |
|---|---|
| a painted, softened watercolor version of the layer | `wash` with `kuwahara`, `bleedWarp`, `paperGrain` |
| paint that bleeds past hard edges | `bleedWarp`, shared as the sample position by `kuwahara` and `wash` |
| a chalk drawing with outlines and cross-hatching | `chalkSketch` with `sobelTaps` |
| the layer redrawn as text characters | `charGrid`, `cellSample`, `glyphFor`, `glyphTint` |
| an engraved, banknote-style line drawing | `linework` |

## Order

- wash
- kuwahara
- bleedWarp
- paperGrain
- chalkSketch
- sobelTaps
- linework
- glyphTint
- glyphFor
- charGrid
- cellSample

## Example

```ts
import {defineShader, p, effects, crosses, transformColor} from 'shaders/std'

const {bleedWarp, kuwahara, paperGrain, wash} = effects.stylize

// One bleed stage, shared by the brush and the wash so the taps and the original sample agree.
const wet = bleedWarp({amount: p('bleed')})

// Watercolor: soft colour patches over paper grain, mixed back toward the original by strength.
export const Watercolor = defineShader({
  name: 'Watercolor',
  props: {
    radius: {default: 3, compileTime: true},
    bleed: {default: 1, recompile: crosses(0)},
    strength: {default: 1, recompile: crosses(0)},
    paper: {default: 0.35, recompile: crosses(0)},
    paperColor: {default: '#fbf7ec', transform: transformColor},
  },
  effect: wash({
    at: wet,
    brush: kuwahara({at: wet, radius: p('radius')}),
    grain: paperGrain({amount: p('paper')}),
    paper: p('paper'),
    paperColor: p('paperColor'),
    strength: p('strength'),
  }),
})
```
