# Masks

A mask is coverage: a number from 0 to 1 at every pixel that says how much of something
shows there. Multiply it into a color's alpha to cut a shape out of a paint, or hand it to
an effect as its amount so the effect fades in across the canvas. Most masks start from a
distance, how far this pixel is from a center, a line or a shape's edge, and the mask word
turns that distance into an edge. `softDisc` and `softBand` give soft, styled edges for orbs
and curtains. `crispFill` and `crispStroke` give clean anti-aliased fills and outlines from a
signed distance, such as one of the `shape` words. `radialMask` is the ready-made one for
effects: 0 at a center, 1 past a radius, plugged straight into an effect's amount.

## Reach for it when

| When | Use |
|---|---|
| an effect that is strongest at the edges and clear in the middle | `radialMask` as the effect's amount |
| a soft glowing disc, orb or dot | `softDisc` |
| a clean filled shape from a signed distance | `crispFill` |
| an outline of a shape, or a drawn curve of a given width | `crispStroke` |
| a band along a line, sharp on one side and trailing off on the other | `softBand` |

## Order

- radialMask
- softDisc
- crispFill
- crispStroke
- softBand

## Example

```ts
import {defineShader, p, uniformOf, math, mask, frames, transformColor, transformPosition} from 'shaders/std'

const {add, sub, mul, div, clamp, local, vec4} = math

// Orb: a soft glowing disc with a crisp ring just outside it, over a transparent background.
export const Orb = defineShader({
  name: 'Orb',
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    color: {default: '#7dd3fc', transform: transformColor},
    ringColor: {default: '#ffffff', transform: transformColor},
    radius: {default: 0.2},
    softness: {default: 0.5},
  },
  paint: (params) => {
    const u = (name: string) => uniformOf(p(name), params)
    const surface = frames.surfaceOf(params)
    const {dist} = frames.centredFrame({center: p('center')})(params, surface)
    const px = div(1, surface.viewport.member('y')) // one device pixel, in uv
    const body = local(mask.softDisc({
      dist,
      radius: u('radius'),
      softness: {width: mul(u('softness'), 0.3), curve: add(mul(u('softness'), 2), 0.5)},
    }), 'body')
    // A signed distance to a circle a little larger than the orb: negative inside, zero on the ring.
    const ring = local(mask.crispStroke({distance: sub(dist, mul(u('radius'), 1.3)), width: 0.006, footprint: px}), 'ring')
    const rgb = add(mul(u('color').member('rgb'), body), mul(u('ringColor').member('rgb'), ring))
    return vec4(rgb, clamp(add(body, ring), 0, 1))
  },
})
```
