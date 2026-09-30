# Frames & coordinates

Frames are how a generator measures its pixels. Instead of raw uv you get named numbers
ready to use: the distance from a centre, how far along and across a line between two
points, or the axes of a chosen direction. `surfaceOf` is the starting point: the canvas's
`uv` (0–1, y down), its size in pixels and its aspect ratio. Every frame is aspect-corrected,
so 1 is the canvas height and a circle stays round on a wide canvas.

Build a frame from your position props with `p('name')`, call it with `params` and the
surface, and pull out the numbers you need. Shape them with `math` and hand the result to a
palette, a `light` word or `compose` to make the pixel's color. `direction` and
`directionFrame` take an angle in degrees, 0 pointing right and turning clockwise on screen.

## Reach for it when

| When | Use |
|---|---|
| the canvas uv, size and aspect at the top of a paint | `surfaceOf` |
| a pattern anchored to the whole canvas, not the layer's box | `rawSurfaceOf` |
| the distance from a draggable centre | `centredFrame` |
| the position along and across a line between two points | `segmentFrame` |
| a light or flow angle as a vector | `direction` |
| a grain, weave or streak running at an angle | `directionFrame` |
| a kaleidoscope | `sectorFold` |

## Order

- surfaceOf
- centredFrame
- segmentFrame
- direction
- directionFrame
- sectorFold
- rawSurfaceOf

## Example

```ts
import {defineShader, p, frames, math, transformColor, transformPosition} from 'shaders/std'

const {surfaceOf, segmentFrame} = frames
const {div, exp, mul, neg, smoothstep, sub, vec4} = math

// A bar of light between two draggable points: bright on the line, fading across it, feathered at both ends.
export const LightBar = defineShader({
  name: 'LightBar',
  props: {
    from: {default: {x: 0.2, y: 0.5}, transform: transformPosition},
    to: {default: {x: 0.8, y: 0.5}, transform: transformPosition},
    color: {default: '#7ad7ff', transform: transformColor},
    width: {default: 0.05},
    feather: {default: 0.1},
  },
  paint: (params) => {
    const u = params.uniforms
    const {along, across, length} = segmentFrame({from: p('from'), to: p('to')})(params, surfaceOf(params))
    const profile = exp(neg(div(mul(across, across), mul(u.width, u.width))))
    const ends = mul(smoothstep(0, u.feather, along), smoothstep(0, u.feather, sub(length, along)))
    const bar = mul(profile, ends)
    return vec4(mul(u.color.member('rgb'), bar), bar)
  },
})
```
