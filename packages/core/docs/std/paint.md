# Fields, palettes & ramps

The words every gradient and most textures are drawn with. A **field** is a number per pixel
(`Field`): the distance from a point, a noise value, a flow pattern. A **palette** turns a
number into a color (`Palette`): two colors, or a multi-stop ramp, mixed in the color space
your definition declares. `rampOver` joins the two and is the single most common line in a
generator.

Everything else here shapes the field before it meets the palette. `warped` bends the
coordinate so smooth noise turns liquid. `scaledVolume` gives noise a zoomable, aspect-correct
coordinate. `tone` and `threshold` reshape the value. `rings` and `tiles` repeat it. `share`
lets two words read one result. Fields nest, so read a recipe from the inside out: the
innermost word is sampled first and each wrapper changes what the next one sees.

The result is a paint: put it in the definition's `paint:` field.

## Reach for it when

| When | Use |
|---|---|
| a gradient of any shape, in a chosen color space | `rampOver` with a `dist` field and `pair` or `stops` |
| the field should flow like liquid | `warped` around a `noiseField`, inside `scaledVolume` |
| noise you can zoom evenly at any canvas size | `scaledVolume` |
| the value is too flat or too hot for the palette | `tone` |
| a signed pattern should become soft color bands | `threshold` over `flowField` |
| concentric or repeated bands of one field | `rings`, `tiles` |
| two words need the same field without computing it twice | `share` |
| a slow brightness breathing along a pattern | `pulse` inside `shimmered` |
| the palette value runs past 1 and must not seam | `wrapRamp`, `foldRamp` |
| a mesh-gradient look, colors scattered across the canvas | `scatterField` with `scatteredAnchors` |
| three or four colors blended by two values | `colorLadder3`, `quadBlend` |
| a rainbow from a hue number | `hueWheel` |

## Order

- rampOver
- pair
- stops
- standardPalette
- dist
- noiseField
- warped
- scaledVolume
- tone
- threshold
- rings
- tiles
- share
- flowField
- pulse
- shimmered
- wrapRamp
- foldRamp
- warpedPoint
- warpStep
- scatterField
- scatteredAnchors
- quadBlend
- colorLadder3
- hueWheel

## Example

```ts
import {defineShader, p, paint, transformColor, transformColorSpace, colorSpaceOptions, colorStopsPropConfig} from 'shaders/std'

const {rampOver, tone, scaledVolume, warped, noiseField, stops} = paint

// Liquid plasma: warped noise on a zoomable slab, tone-shaped, read through the stop ramp.
export const Plasma = defineShader({
  name: 'Plasma',
  animatedTime: {speed: 'speed'},
  props: {
    colorA: {default: '#7018be', transform: transformColor},
    colorB: {default: '#000000', transform: transformColor},
    stops: colorStopsPropConfig(),
    colorSpace: {default: 'oklch', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
    scale: {default: 2},
    warp: {default: 0.4},
    intensity: {default: 1.5},
    speed: {default: 1},
  },
  paint: rampOver(
    tone(
      scaledVolume(warped(noiseField('mx3'), {amount: p('warp'), amountScale: 4, timeScale: 0.125}), {scale: p('scale')}),
      {glow: p('intensity'), contrast: 1, balance: 50, invert: true},
    ),
    stops(p('colorSpace')),
  ),
})
```
