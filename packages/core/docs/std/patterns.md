# Patterns & tiling

Repeating patterns you can color and animate: checkers, stripes, zigzags, rings, falling
streaks, and tiled lattices of squares, hexagons, triangles, bricks, truchet arcs, woven
threads and isometric cubes, plus the print-screen filters (halftone dots, CMYK plates and
dithering). Most pattern words return a **mask**: a number per pixel that is 1 on the pattern
and 0 off it. The lattice words also return a per-cell **shade**, a brightness factor around 1
that `vary` multiplies into a fill so every tile reads slightly different.

The lattice words share one coordinate system: `cellFrame` sets how many cells fit down the
canvas height and how far the lattice is rotated, and `gridLines`, `hexLines`,
`triangleLines`, `truchetArcs`, `weaveThreads` and `isoCubeFaces` draw inside it. The
free-standing words (`checkerCells`, `stripeBands`, `ringWaves`, `dotLattice`,
`brickCourses`) bring their own. Color a mask with `strokeOver` (fill where 0, stroke where 1,
in a chosen color space) or hang it under one color as alpha with `withAlpha`; the result goes
in `paint:`. The halftone and dither words are filters over the layer inside them and go in
`effect:`.

## Reach for it when

| When | Use |
|---|---|
| a two-color checkerboard | `checkerCells` colored with `strokeOver` |
| stripes or chevrons that scroll | `stripeBands`, `zigzagBands` |
| rings rippling out from a point | `ringWaves` |
| rain or streaks falling across the canvas | `fallingStreaks` with `withAlpha` |
| a field of dots, optionally staggered or twinkling | `dotLattice` under `withAlpha` |
| grid lines, honeycombs or triangle lattices | `cellFrame` with `gridLines`, `hexLines`, `triangleLines` |
| every tile a slightly different brightness | `vary` with a lattice word's shade output |
| a brick wall | `brickCourses` |
| a maze of joined arcs | `truchetArcs` |
| a woven textile | `weaveThreads` |
| tumbling 3D cubes | `isoCubeFaces` with `times`, `mixOf`, `opaque` |
| a mapped size or thickness that should change per whole cell | `sampleMapsAtCellCentres` on the definition's mapSampleUVs field |
| a halftone dot screen over the layer inside | `dotScreen` |
| a four-color print with plate angles and misregistration | `cmykPress` with `inkPlate` |
| your own ink press inside a gather build | `dotScreenMask`, `inkTransmission` |
| one select prop that swaps whole effect recipes | `chosenBy` |
| a retro dither | `pixelGrid`, `quantise`, `ditherInks` |
| a color's alpha or RGB on its own | `alphaOf`, `rgbOf` |

## Order

- strokeOver
- withAlpha
- checkerCells
- stripeBands
- zigzagBands
- ringWaves
- fallingStreaks
- dotLattice
- cellFrame
- gridLines
- hexLines
- triangleLines
- brickCourses
- truchetArcs
- weaveThreads
- isoCubeFaces
- vary
- sampleMapsAtCellCentres
- times
- mixOf
- alphaOf
- rgbOf
- opaque
- dotScreen
- cmykPress
- inkPlate
- dotScreenMask
- inkTransmission
- chosenBy
- pixelGrid
- quantise
- ditherInks

## Example

```ts
import {defineShader, p, patterns, transformColor, transformColorSpace, transformAngle, colorSpaceOptions} from 'shaders/std'

const {cellFrame, hexLines, strokeOver, vary} = patterns

// A honeycomb: hexagon lines in a rotatable lattice, each cell's fill lightened or darkened at random.
const field = hexLines({
  frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}),
  thickness: p('thickness'),
  softness: p('softness'),
  variation: p('variation'),
})

export const Honeycomb = defineShader({
  name: 'Honeycomb',
  props: {
    fill: {default: '#101828', transform: transformColor},
    line: {default: '#f5d90a', transform: transformColor},
    colorSpace: {default: 'linear', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
    cells: {default: 8},
    thickness: {default: 1},
    rotation: {default: 0, transform: transformAngle},
    softness: {default: 0},
    variation: {default: 0.3},
  },
  paint: strokeOver({
    fill: vary(p('fill'), field.shade),
    stroke: p('line'),
    mask: field.lines,
    space: p('colorSpace'),
  }),
})
```
