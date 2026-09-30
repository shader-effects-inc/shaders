# Context & invocation

The **context** is what every pixel knows about its surroundings: its coordinate (`uv`, 0–1
across the canvas with y down), the clock (`time`, in seconds), the canvas shape (`aspect`,
`viewportSize`) and the mouse (`pointer`). `ctx` names these as tokens you can pass to a word
or bind in a `wgsl` body's `inputs`.

The rest of this page is for a **hand-written builder**: a `paint:` or `effect:` given as a
function of the params instead of a composition or a `wgsl` body. Inside one, `paintFrame`
gives the coordinate and frame size to draw in, `uniformOf` reads a prop as a GPU value,
`resolveScalar` and `resolveArg` turn anything a word accepts (a prop ref, a `Scalar`, a
context token, a number) into a GPU value, and `pointwiseOp` wraps a compiled GPU function
as a new color word. Combine the results with the `math` words and return a `vec4` color.

## Reach for it when

| When | Use |
|---|---|
| the mouse, the clock or the canvas size as an input to a word | `ctx` |
| the coordinate to draw at in a hand-written paint | `paintFrame` |
| a prop's value inside a hand-written builder | `uniformOf` |
| a `Scalar` (a mask times a prop) inside a builder | `resolveScalar` |
| any value a word accepts, inside a filter builder | `resolveArg` |
| the same inside a warp's map factory | `resolveArgIn` |
| a new color word from a compiled GPU function | `pointwiseOp` |

## Order

- ctx
- ctx.uv
- ctx.time
- ctx.aspect
- ctx.pointer
- ctx.viewportSize
- ctx.logicalViewportSize
- paintFrame
- uniformOf
- resolveScalar
- resolveArg
- resolveArgIn
- pointwiseOp

## Example

```ts
import {defineShader, p, paintFrame, uniformOf, animatedTime, math, transformColor, transformPosition} from 'shaders/std'

const {sub, mul, add, vec2, vec4, length, smoothstep, mix, sin} = math

// A spotlight: a soft disc around a draggable point that breathes on the layer's own clock.
export const Spotlight = defineShader({
  name: 'Spotlight',
  animatedTime: {speed: 'speed'},
  props: {
    colorA: {default: '#0b1026', transform: transformColor},
    colorB: {default: '#ffd27a', transform: transformColor},
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    radius: {default: 0.4},
    speed: {default: 1},
  },
  paint: (params) => {
    const {uv} = paintFrame(params)
    const center = uniformOf(p('center'), params)
    const colorA = uniformOf(p('colorA'), params)
    const colorB = uniformOf(p('colorB'), params)
    // A position prop's y is stored flipped; undo it to compare with uv.
    const toCenter = sub(uv, vec2(center.member('x'), sub(1, center.member('y'))))
    const d = length(mul(toCenter, vec2(params.ctx.aspect, 1)))
    const breath = add(1, mul(0.1, sin(animatedTime(params))))
    const t = sub(1, smoothstep(0, mul(uniformOf(p('radius'), params), breath), d))
    return vec4(mix(colorA.member('rgb'), colorB.member('rgb'), t), 1)
  },
})
```
