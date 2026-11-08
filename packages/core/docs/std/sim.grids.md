# Grid programs

A grid simulation is a fixed-size field of cells (a displacement per cell, two chemical
concentrations, a sort offset) that changes a little every frame. `gridSim` runs it: you
describe the frame as an ordered list of **stages** built with `stage.*`, and the engine runs that
list every frame, owning the clock, the frame-delta clamp, the ready gate, the settle gate and
the outputs the layer samples. A stage is bookkeeping (`stage.host`, `stage.values`, `stage.readyWhen`,
`stage.settle`) or work (`stage.pass`, `stage.cache`, `stage.seedOnce`, `stage.iterate`, `stage.sortPass`), and
every frame ends with `stage.publish`, which writes the state into the texture the fragment reads.

The build callback runs once per instance of the layer and returns the outputs, the clamp and
the stages; anything it creates (buffers, passes, trackers) belongs to that instance. A settled
field costs nothing: once `stage.settle` sees nothing driving it, frames are skipped and the last
published picture stays on screen.

Two shortcuts sit on top. The pointer fields in `effects.pointerFields` are whole grid programs
you spread as `...gridSim(springLatticeField({...}))` and read with a warp map. And for a small
field whose arithmetic reads better as JavaScript loops, `hostGridProgram` runs the same named
steps on the CPU and `hostFieldTexture` uploads the cells as a texture. When a wave field and a
displacement are all you need, declare them with `simulate.grid` instead.

## Reach for it when

| When | Use |
|---|---|
| liquid or fabric that ripples after the cursor pushes it | `gridSim` with `springLatticeField` and a warp map |
| a rule that must run N times per frame (reaction-diffusion) | `stage.iterate` after `stage.values`, then `stage.publish` |
| a sort that settles a little more every frame | `stage.sortPass`, with `stage.publish` reading its side |
| the state must be reset on the first frame or a seed change | `stage.seedOnce` |
| a pass that needs the child's picture, which arrives late | `stage.readyWhen` and `stage.cache` |
| the simulation should sleep once nothing drives it | `stage.settle` |
| this frame's values written before the passes read them | `stage.values` |
| the canvas size or aspect inside a stage | `trackedViewport` |
| a small field stepped in plain JavaScript | `hostGridProgram` with `hostStep`, published by `hostFieldTexture` |

## Order

- gridSim
- stage
- stage.values
- stage.pass
- stage.publish
- stage.iterate
- stage.seedOnce
- stage.sortPass
- stage.cache
- stage.readyWhen
- stage.settle
- stage.host
- trackedViewport
- hostGridProgram
- hostStep
- hostFieldTexture

## Example

```ts
import {defineShader, p, sim, effects, warps, transformEdges} from 'shaders/std'

const {gridSim} = sim.grids
const {springLatticeField} = effects.pointerFields

// Liquify: a spring lattice the cursor pushes, published as a displacement field the warp map reads.
// The pointer field packages the whole stage list (pointer tracking, values, the step passes, the
// settle gate, publish); this definition only names the props and the output.
export const Liquify = defineShader({
  name: 'Liquify',
  usesPointer: true,
  props: {
    intensity: {default: 10},
    stiffness: {default: 3},
    damping: {default: 3},
    radius: {default: 1},
    edges: {default: 'stretch', transform: transformEdges, compileTime: true},
  },
  ...gridSim(springLatticeField({
    stiffness: p('stiffness'), damping: p('damping'), radius: p('radius'),
    output: 'displacement',
  })),
  // Bend the incoming coordinate by the field, scaled by intensity, then handle the edges.
  map: warps.liquidDisplace({intensity: p('intensity'), output: 'displacement'}),
  // Before the field exists (no GPU yet), the warp passes the layer through untouched.
  uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement,
})
```
