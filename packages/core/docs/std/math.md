# Math

The arithmetic you write your own paint with. Every word takes expressions or plain numbers
and returns an expression the GPU works out once per pixel: `mul(color, coverage)` fades a
color, `smoothstep(a, b, dist)` turns a distance into a soft mask, `mix(a, b, t)` blends two
colors. Props reach you as `params.uniforms.<name>` (or `uniformOf(p('name'), params)`), a
frame gives you the coordinate, and the words compose by nesting until you return a `vec4`
color from `paint:`. Wrap anything you read twice in `local` so it is computed once.

## Reach for it when

| When | Use |
|---|---|
| combining numbers, colors or coordinates | `add`, `sub`, `mul`, `div` |
| blending two colors or values by an amount | `mix` with `splat3` for colors |
| a soft edge from a distance | `smoothstep`; `step` for a hard one |
| keeping a value in range | `clamp`, `min`, `max` |
| a value you read more than once | `local` |
| the distance from a center or between points | `length`, `distance` |
| a pattern that repeats across the canvas | `fract` and `floor` on a scaled coordinate |
| a wave or a rotation over time | `sin`, `cos`, `rotate2` |
| an angle around a center | `atan2` |
| a smooth falloff or a bell shape | `exp`, `gaussBell` |
| a hinge that bends instead of creasing | `softPlus` |
| picking one of two values per pixel | `select` with `lt`, `gt`, `le`, `ge` |
| several motions that must not move in step | `hashPhase` |
| the final color | `vec4` from a `vec3` and an alpha |

## Order

- add
- sub
- mul
- div
- mix
- smoothstep
- clamp
- local
- vec4
- vec3
- vec2
- splat3
- length
- distance
- abs
- fract
- floor
- min
- max
- sin
- cos
- step
- pow
- exp
- sqrt
- neg
- select
- lt
- gt
- le
- ge
- atan2
- dot
- normalize
- rotate2
- gaussBell
- softPlus
- hashPhase
- sign
- ceil
- log
- exp2
- tan
- cross
- reflect
- refract
- float

## Example

```ts
import {defineShader, p, uniformOf, animatedTime, math, frames, transformColor, transformPosition} from 'shaders/std'

const {add, sub, mul, sin, smoothstep, mix, splat3, local, vec4} = math

// Rings: concentric waves spreading from a draggable center, blended between two colors.
export const Rings = defineShader({
  name: 'Rings',
  animatedTime: {speed: 'speed'},
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    colorA: {default: '#0b1026', transform: transformColor},
    colorB: {default: '#7dd3fc', transform: transformColor},
    frequency: {default: 24},
    softness: {default: 0.25},
    speed: {default: 1},
  },
  paint: (params) => {
    const u = (name: string) => uniformOf(p(name), params)
    const {dist} = frames.centredFrame({center: p('center')})(params, frames.surfaceOf(params))
    // A −1..1 wave along the distance, moved into 0..1 and computed once.
    const wave = local(add(mul(sin(sub(mul(dist, u('frequency')), animatedTime(params))), 0.5), 0.5), 'wave')
    // softness 0 is a hard ring, 0.5 a smooth gradient between rings.
    const band = smoothstep(sub(0.5, u('softness')), add(0.5, u('softness')), wave)
    const rgb = mix(u('colorA').member('rgb'), u('colorB').member('rgb'), splat3(band))
    return vec4(rgb, 1)
  },
})
```
