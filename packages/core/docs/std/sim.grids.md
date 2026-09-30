# Grid programs

A grid simulation is a fixed-size field of cells (a displacement per cell, two chemical
concentrations, a sort offset) that changes a little every frame. `gridSim` runs it: you
describe the frame as an ordered list of **stages** built with `op.*`, and the engine runs that
list every frame, owning the clock, the frame-delta clamp, the ready gate, the settle gate and
the outputs the layer samples. A stage is bookkeeping (`op.host`, `op.values`, `op.readyWhen`,
`op.settle`) or work (`op.pass`, `op.cache`, `op.seedOnce`, `op.iterate`, `op.sortPass`), and
every frame ends with `op.publish`, which writes the state into the texture the fragment reads.

The build callback runs once per instance of the layer and returns the outputs, the clamp and
the stages; anything it creates (buffers, passes, trackers) belongs to that instance. A settled
field costs nothing: once `op.settle` sees nothing driving it, frames are skipped and the last
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
| a rule that must run N times per frame (reaction-diffusion) | `op.iterate` after `op.values`, then `op.publish` |
| a sort that settles a little more every frame | `op.sortPass`, with `op.publish` reading its side |
| the state must be reset on the first frame or a seed change | `op.seedOnce` |
| a pass that needs the child's picture, which arrives late | `op.readyWhen` and `op.cache` |
| the simulation should sleep once nothing drives it | `op.settle` |
| this frame's values written before the passes read them | `op.values` |
| the canvas size or aspect inside a stage | `trackedViewport` |
| a small field stepped in plain JavaScript | `hostGridProgram` with `hostStep`, published by `hostFieldTexture` |

## Order

- gridSim
- op
- op.values
- op.pass
- op.publish
- op.iterate
- op.seedOnce
- op.sortPass
- op.cache
- op.readyWhen
- op.settle
- op.host
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
