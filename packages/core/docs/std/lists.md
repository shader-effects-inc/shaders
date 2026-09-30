# List props

A **list prop** holds any number of items of the same shape: lights, points, anchors. Declare
it with `listPropConfig`: each item is a small record of typed fields (`position`, `color`,
`number`, `boolean`) and the list has a `maxItems` cap. The editor renders it as an add and
remove list, every position field becomes a draggable handle, and any position can follow the
mouse. Adding or removing an item at runtime never rebuilds the shader.

Reading one takes a hand-written builder. `listOf(params, 'lights')` gives the live item count
and each item's fields by name, and `accumulate` adds a per-item value over all of them in a
real GPU loop. Anything the loop body reads that was computed outside it goes in `deps`, so
it is emitted once before the loop.

## Reach for it when

| When | Use |
|---|---|
| any number of lights, points or anchors as one prop | listPropConfig to declare it, `listOf` to read it |
| adding up light, weight or distance from every item | `accumulate` |
| one item's position, color or number at a loop index | `listOf` then at(i) |

## Order

- listOf
- accumulate

## Example

```ts
import {defineShader, p, listPropConfig, paintFrame, uniformOf, listOf, accumulate, math, transformColor} from 'shaders/std'

const {sub, mul, add, div, vec2, vec4, length, clamp, splat3, local} = math

// Point lights: each item in the list adds its colored glow over a background.
export const PointLights = defineShader({
  name: 'PointLights',
  props: {
    background: {default: '#0b1026', transform: transformColor},
    lights: listPropConfig({
      maxItems: 6,
      minItems: 1,
      itemLabel: 'Light',
      item: {
        position: {kind: 'position', default: {x: 0.5, y: 0.5}, label: 'Position'},
        color: {kind: 'color', default: '#ffb347', label: 'Color'},
        intensity: {kind: 'number', default: 1, label: 'Intensity', min: 0, max: 4, step: 0.01},
      },
    }, {
      default: [
        {position: {x: 0.3, y: 0.3}, color: '#ffb347', intensity: 1},
        {position: {x: 0.7, y: 0.7}, color: '#7a5cff', intensity: 1},
      ],
      description: 'The lights, each with its own position, color and brightness',
      label: 'Lights',
      group: 'Lights',
    }),
  },
  paint: (params) => {
    const {uv} = paintFrame(params)
    const lights = listOf(params, 'lights')
    const lit = accumulate(lights, (i) => {
      const light = lights.at(i)
      const pos = light.position('position')
      // A position field's y is stored flipped; undo it to compare with uv.
      const toLight = sub(uv, vec2(pos.member('x'), sub(1, pos.member('y'))))
      const d = local(length(mul(toLight, vec2(params.ctx.aspect, 1))), 'lightDist')
      return mul(light.color('color').member('rgb'), div(mul(light.number('intensity'), 0.05), add(0.01, mul(d, d))))
    }, {zero: 'vec3f', deps: [uv], hint: 'lightSum'})
    const base = uniformOf(p('background'), params).member('rgb')
    return vec4(clamp(add(base, lit), splat3(0), splat3(1)), 1)
  },
})
```
