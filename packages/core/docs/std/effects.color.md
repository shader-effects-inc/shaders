# Color effects

Words that recolor the layer inside them, one pixel at a time. Each returns a **pointwise
effect** for a filter definition's `effect:` field: it reads the child's color at this pixel
and returns the adjusted color, with alpha kept. Nothing is sampled from neighbouring
pixels, so these filters are cheap and stack freely. Pass props with `p('name')`; the units
are the ones the word names (degrees, 0–1 amounts, a multiplier around 1).

Most adjustments have a value that changes nothing (`saturate` at 1, `hueRotate` at 0,
`solarize` at strength 0). Declare it with `identityWhen` so the filter is skipped when it is
doing nothing. The looks that pick a code path from a select or checkbox (`tint`'s
`preserveLuminosity`, `duotone`'s `colorSpace`, `gradientMap`'s `palette`, `isolines`'
`source` and `colorMode`) need that prop declared `compileTime: true`.

## Reach for it when

| When | Use |
|---|---|
| more or less color | `saturate`, or `vibrance` to protect skin tones |
| a different hue | `hueRotate` |
| brighter, darker, punchier | `brightnessContrast`, `exposure` |
| black and white | `grayscale` |
| a negative or a darkroom flip of the highlights | `invert`, `solarize` |
| flat poster bands | `posterize` |
| a color wash over everything | `tint` |
| two or three colors mapped to brightness | `duotone`, `tritone` |
| a full gradient mapped to brightness, animated or not | `gradientMap` |
| film grain | `filmGrain` |
| contour lines traced from brightness or alpha | `isolines` |
| a measured film-stock lookup table in a `gpu:` fragment | `lutAtlasGrade` |

## Order

- saturate
- vibrance
- hueRotate
- brightnessContrast
- exposure
- grayscale
- invert
- solarize
- posterize
- tint
- duotone
- tritone
- gradientMap
- filmGrain
- isolines
- lutAtlasGrade

## Example

```ts
import {defineShader, p, effects, isZero, transformColor, transformColorSpace, colorSpaceOptions} from 'shaders/std'

const {duotone} = effects.color

// Two-color print: shadows in one color, highlights in another, mixed in the chosen color space.
export const Duotone = defineShader({
  name: 'Duotone',
  props: {
    colorA: {default: '#1a0b2e', transform: transformColor},
    colorB: {default: '#f9ed69', transform: transformColor},
    blend: {default: 0.5, ui: {type: 'range', min: 0, max: 1, step: 0.01}},
    colorSpace: {default: 'oklch', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
  },
  effect: duotone({colorA: p('colorA'), colorB: p('colorB'), blend: p('blend'), colorSpace: p('colorSpace')}),
  missingChildMessage: 'Duotone needs a child to recolor.',
})
```
