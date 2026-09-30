# Noise

Words for grainy, cloudy, cellular and streaked surfaces: the textures behind clouds, film
grain, stone, water and brushed metal. Read a noise recipe from the inside out, in four
steps. A **basis** is the raw pattern, one number per point (`gaborGrains`, `curlSpeed`,
`blueSpeckle`, or a `noiseField` from paint). A **framing** decides where on the canvas the
basis is sampled and how big it is: `seededPlane` for almost everything, `pixelGrid` for
patterns that live on pixels. A **tone** word turns the raw number into a 0–1 ramp position
with contrast and balance controls: `noiseTone` for a 0–1 basis, `signedTone` for a −1 to 1
one, or `unitized` to convert first. A **ramp** closes it: `rampOver` with `stops` or
`linearPair` gives the field color.

Each basis is either signed (−1 to 1) or unit (0 to 1); its comment says which. Signed
bases go through `unitized` or `signedTone`, unit bases straight into `noiseTone`. Bases
that move read the layer's clock, so declare `animatedTime: {speed: 'speed'}` on the
definition. `fractalNoise` and `worleyNoise` are complete paints that do all four steps
for you. The wave words (`waves`, `wavyLine`) are expressions rather than fields, for
recipes written as a `paint: (params) => …` function. `noiseRelief` is the one filter here:
it embosses the layer inside instead of painting.

## Reach for it when

| When | Use |
|---|---|
| soft drifting clouds or smoke | `seededPlane` around `evolving` around `noiseField`, then `noiseTone` |
| layered fractal detail with an angle | `fractalNoise` |
| cells, scales or a cracked-earth look | `worleyNoise` |
| Voronoi cells with a fill and border lines | `cellDistances` with `cellFill`, `cellBorders` and `borderOverlay` |
| brushed metal or wood grain | `gaborGrains` inside `unitized` |
| rippling water interference | `waveletBands` inside `unitized` |
| film grain or dither | `pixelGrid` around `blueSpeckle` |
| eroded terrain ridges | `erosionRidges` inside `unitized` |
| swirling flow, eddies and streams | `curlSpeed` |
| hairline scratches on a surface | `scratchStreaks` |
| a basis is −1 to 1 and the ramp wants 0–1 | `unitized` or `signedTone` |
| the standard contrast and balance sliders | `noiseTone` |
| a contrast slider where 1 means unchanged | `gainTone` |
| two colors with no color-space prop | `linearPair` |
| an organic wobble on an edge, radius or fill | `waves`, or `organicWaves` for a ready schedule |
| a wavy line, horizon or curtain path | `wavyLine` |
| flowing gradient ribbons between two points | `ribbons` closed by `overRgb` with `backToP3` and `tonePow` |
| embossing the layer inside as stone or cloth | `noiseRelief` with `reliefBases` |

## Order

- seededPlane
- evolving
- noiseTone
- unitized
- signedTone
- gainTone
- gaborGrains
- waveletBands
- curlSpeed
- erosionRidges
- scratchStreaks
- pixelGrid
- blueSpeckle
- scaledPlane
- linearPair
- fractalNoise
- worleyNoise
- transformWorleyMode
- transformWorleyDistance
- cellDistances
- cellFill
- cellBorders
- borderOverlay
- waves
- wavyLine
- organicWaves
- ribbons
- overRgb
- backToP3
- tonePow
- noiseRelief
- reliefBases

## Example

```ts
import {defineShader, p, noise, paint, transformColor, transformColorSpace, colorSpaceOptions} from 'shaders/std'

const {seededPlane, gaborGrains, unitized, noiseTone} = noise
const {rampOver, stops} = paint

// Brushed metal: oriented sine grains (a signed basis) framed on a zoomable plane,
// converted to 0–1, toned with contrast and balance, and read through the color ramp.
export const BrushedMetal = defineShader({
  name: 'BrushedMetal',
  animatedTime: {speed: 'speed'},
  props: {
    colorA: {default: '#f4f4f5', transform: transformColor},
    colorB: {default: '#27272a', transform: transformColor},
    colorSpace: {default: 'oklab', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
    scale: {default: 1.5},
    frequency: {default: 8},
    seed: {default: 0},
    contrast: {default: 0},
    balance: {default: 0},
    speed: {default: 0.5},
  },
  paint: rampOver(
    noiseTone(
      seededPlane(unitized(gaborGrains({frequency: p('frequency')})), {scale: p('scale'), seed: p('seed')}),
      {contrast: p('contrast'), balance: p('balance')},
    ),
    stops(p('colorSpace')),
  ),
})
```
