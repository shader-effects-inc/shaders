# Simulate

Say "simulate this" and let the engine run it. `simulate.grid` declares a square grid of
numbers that changes a little every frame: you list the steps that advance it (`op.wave`
makes it ripple, `op.splat` presses the pointer into it) and name the outputs you want derived
from it (`op.gradient`, the slope at every cell). The engine allocates the state, keeps the
history a wave needs, tracks the pointer, clamps the frame clock, and stops dispatching once
the field has faded to nothing. There is no GPU code in the definition at all.

An **output** is a field an effect can read. Declare the simulation once at module scope, then
hand `waves.output('displacement')` to `displaceBy`, which bends the layer inside along the
slope with a chromatic split and edge handling. Today the engine implements exactly one recipe,
`[op.wave, op.splat]` with a derived `op.gradient`; a different configuration throws when the
shader is defined, never at render time.

When an effect needs its own GPU step (a reaction-diffusion rule, a fluid, a flock), the
lower-level families take over: `sim.grids`, `sim.fluids`, `sim.feedback`, `sim.agents`.

## Reach for it when

| When | Use |
|---|---|
| ripples that spread from the cursor and distort the layer inside | `simulate.grid` with `op.wave`, `op.splat`, `op.gradient`, read by `displaceBy` |
| the ripples should fade faster or ring longer | `op.wave` with a damping prop, 0 rings forever, 20 dies in frames |
| the brush should be bigger or press harder | `op.splat` with a radius prop and `pointerSpeed` |
| the simulation should sleep once still | the rest option on `simulate.grid` |
| a handle to the field for an effect | `GridSim` and its output method |

## Order

- simulate
- simulate.grid
- op
- op.wave
- op.splat
- op.gradient
- GridSim

## Example

```ts
import {defineShader, p, simulate, op, pointer, pointerSpeed, displaceBy, transformEdges} from 'shaders/std'

// A 128-cell wave field the pointer stirs. The engine keeps two frames of history for the wave,
// tracks the pointer (a jump into the canvas is not a stroke), and sleeps once the field has decayed.
const waves = simulate.grid({
  resolution: 128,
  history: 2,
  step: [
    op.wave({damping: p('decay')}),
    op.splat({at: pointer({teleportGuard: 'on'}), amount: pointerSpeed({max: 2}), radius: p('radius')}),
  ],
  derive: {displacement: op.gradient()},
  rest: {settlesWhen: 'derived-from-damping'},
})

// Ripples: the layer inside is displaced along the wave's slope, with a chromatic split at the crests.
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
