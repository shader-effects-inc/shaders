# Warps

Words for the `map:` field. A **warp** is a component that moves the pixels of the layer
inside it (the child): for every pixel on screen, the map says where in the child to look.
Twisting, bulging, mirroring, tilting and wiping are all warps. Positions are in uv (0–1
across the canvas, y down) and angles are in degrees.

A definition with a `map:` is a warp, nothing else to declare. Most maps can look outside
the child, so the definition also says what happens there with `edges`: the default reads
an `edges` prop holding `'stretch'`, `'transparent'`, `'mirror'` or `'wrap'`; a number bakes
one mode in (`edges: 1` is transparent); `'none'` is for a map that never leaves the child.
`resample` picks the texture filter, `'catmullRom'` (the default, sharp under
magnification) or `'bilinear'`. Animated warps read the layer's clock, so declare
`animatedTime: {speed: 'speed'}`.

One word is not a map: `displaceByLayer` reads a second layer, so it goes in `effect:`.

## Reach for it when

| When | Use |
|---|---|
| twist the picture around a point | `twirl` |
| a magnifying glass or a pinch | `bulge` |
| reflect one half onto the other | `mirrorLine` |
| flip horizontally or vertically | `flip` with `edges: 'none'` |
| mirrored pie slices | `kaleidoscope` |
| a card tilted in 3D | `perspective` |
| pin the corners anywhere | `cornerPin` |
| a curved screen | `bend` |
| pull the picture out along one direction | `stretch` |
| a travelling ripple | `wave` |
| a slow liquid drift | `flowNoise` |
| unroll the picture into a circle, or roll it back | `toPolar`, `fromPolar` |
| rings that each spin on their own | `concentricRings` |
| a glitch of shifted bars | `barOffset` |
| a wipe transition that shreds the picture | `slicedSlide` with `edges: 1` |
| distortion driven by your own simulation | `liquidDisplace`, `gridCellDisplace` |
| another layer's pixels as a displacement map | `displaceByLayer` in `effect:` |

## Order

- twirl
- bulge
- mirrorLine
- flip
- kaleidoscope
- perspective
- cornerPin
- bend
- stretch
- wave
- flowNoise
- toPolar
- fromPolar
- concentricRings
- barOffset
- slicedSlide
- liquidDisplace
- gridCellDisplace
- displaceByLayer

## Example

```ts
import {defineShader, p, warps, transformPosition, transformEdges} from 'shaders/std'

// Twirl: twist the child around a draggable center, more the farther from it.
export const Twirl = defineShader({
  name: 'Twirl',
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    intensity: {default: 1, ui: {type: 'range', min: -5, max: 5, step: 0.1}},
    edges: {
      default: 'stretch',
      transform: transformEdges,
      compileTime: true,
      ui: {type: 'select', options: [{label: 'Stretch', value: 'stretch'}, {label: 'Transparent', value: 'transparent'}, {label: 'Mirror', value: 'mirror'}, {label: 'Wrap', value: 'wrap'}]},
    },
  },
  map: warps.twirl({center: p('center'), intensity: p('intensity')}),
})
```
