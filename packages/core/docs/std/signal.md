# Signals

Live inputs a simulation reads every frame: where the pointer is and how fast it is moving.
You never read the numbers yourself. You hand a signal to a simulation step, such as the
splat step of a grid simulation, and the engine tracks the pointer, measures its speed in
uv per second, and feeds both in. `pointer` carries one choice, the teleport guard, which
stops a cursor that jumps into the canvas from slamming the field. `pointerSpeed` caps the
speed so a fast flick cannot overdrive it. Both are also exported at the top level of
`shaders/std`.

`cpuHash01` is the odd one out: a stable random number from a seed, worked out once in
JavaScript, for constants a paint needs per item rather than per pixel.

## Reach for it when

| When | Use |
|---|---|
| a simulation should be stirred where the cursor is | `pointer` as the splat step's position |
| stirring harder the faster the cursor moves | `pointerSpeed` as the splat step's amount |
| a cursor entering the canvas must not cause a splash | `pointer` with the teleport guard on (the default) |
| a stable random size, speed or offset per item, computed once | `cpuHash01` |

## Order

- pointer
- pointerSpeed
- cpuHash01

## Example

```ts
import {defineShader, p, simulate, op, signal, displaceBy, transformEdges} from 'shaders/std'

// A height field stirred at the pointer, harder the faster it moves, that settles when left alone.
const waves = simulate.grid({
  resolution: 128,
  history: 2,
  step: [
    op.wave({damping: p('decay')}),
    op.splat({at: signal.pointer({teleportGuard: 'on'}), amount: signal.pointerSpeed({max: 2}), radius: p('radius')}),
  ],
  derive: {displacement: op.gradient()},
  rest: {settlesWhen: 'derived-from-damping'},
})

// Ripples: the field's slope pushes the layer inside around, with a little color fringing.
export const Ripples = defineShader({
  name: 'Ripples',
  props: {
    intensity: {default: 10},
    decay: {default: 10},
    radius: {default: 0.5},
    chromaticSplit: {default: 1},
    edges: {default: 'stretch', transform: transformEdges, compileTime: true},
  },
  effect: displaceBy(waves.output('displacement'), {
    strength: p('intensity'),
    chromatic: p('chromaticSplit'),
    edges: p('edges'),
  }),
})
```
