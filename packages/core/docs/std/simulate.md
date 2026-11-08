# Simulate

Say "simulate this" and let the engine run it. List a simulation in a definition's `inputs` and
read it in WGSL by name. `grid` is a field of cells advanced every frame by steps (`op.*`,
or your own `op.kernel`); `particles` is thousands of moving points drawn into a texture;
`fluid` is a real fluid solve carrying coloured dye. The engine allocates the state, swaps
it frame to frame, seeds it, tracks the pointer and clamps the clock. You write only what is
particular to your effect.

An **output** is a field an effect can read. Declare the simulation once at module scope, then
hand `waves.output('displacement')` to `displaceBy`, which bends the layer inside along the
slope with a chromatic split and edge handling (that path takes exactly `[op.wave, op.splat]` with a
derived `op.gradient`). Any other grid, particles or fluid goes in `inputs`.

When an effect needs its own GPU step (a reaction-diffusion rule, a fluid, a flock), the
lower-level families take over: `sim.grids`, `sim.fluids`, `sim.feedback`, `sim.agents`.

## Reach for it when

| When | Use |
|---|---|
| ripples that spread from the cursor and distort the layer inside | `grid` with `wave`, `splat`, `gradient`, read by `displaceBy` |
| the ripples should fade faster or ring longer | `wave` with a damping prop, 0 rings forever, 20 dies in frames |
| the brush should be bigger or press harder | `splat` with a radius prop and `pointerSpeed` |
| the simulation should sleep once still | the rest option on `grid` |
| a field, a swarm or a fluid read by name in WGSL | `grid`, `particles`, `fluid` in inputs |

## Order

- grid
- particles
- fluid

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
    op.splat({at: pointer(), amount: pointerSpeed({max: 2}), radius: p('radius')}),
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
