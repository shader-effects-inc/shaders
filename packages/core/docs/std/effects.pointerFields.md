# Pointer fields

Push fields the pointer stirs. Each word is a whole simulation: a grid of cells, each holding
how far the layer inside should shift there, that the cursor pushes and that settles back on
its own. Spread the word into a warp definition with `...sim.grids.gridSim(word)` and it
publishes the field under its `output` name; the warp's `map:` then reads it with
`warps.gridCellDisplace` (per cell, blocky) or `warps.liquidDisplace` (smoothly, scaled by an
intensity prop). The definition needs `usesPointer: true`.

`pointerSplatField` smears pixels along the cursor and lets them fade back; its grid size is a
prop, so pair it with `clampSplatGridSize` in a recompile rule. `springLatticeField` is a cloth
of springs on a fixed 64-cell grid that stretches, rings and settles like fabric. Both stop
computing once the field has settled, and both give a displacement in uv.

## Reach for it when

| When | Use |
|---|---|
| a smear that follows the cursor and fades | `pointerSplatField` |
| fabric or liquid that ripples after a push | `springLatticeField` |
| the grid-size prop's recompile rule | `clampSplatGridSize` |
| the range that rule holds the grid to | `SPLAT_MIN_GRID`, `SPLAT_MAX_GRID` |

## Order

- pointerSplatField
- springLatticeField
- clampSplatGridSize
- SPLAT_MIN_GRID
- SPLAT_MAX_GRID

## Example

```ts
import {defineShader, p, effects, sim, warps, recompileWhen, transformEdges} from 'shaders/std'

const {pointerSplatField, clampSplatGridSize} = effects.pointerFields

// Smear: the layer inside follows the cursor in coarse cells and eases back when it stops.
export const Smear = defineShader({
  name: 'Smear',
  usesPointer: true,
  props: {
    intensity: {default: 1},
    decay: {default: 3},
    radius: {default: 1},
    // Rebuild the simulation only when the cell count really changes, not on every slider step.
    gridSize: {default: 20, recompile: recompileWhen((prev, next) => clampSplatGridSize(prev as number) !== clampSplatGridSize(next as number))},
    edges: {default: 'stretch', transform: transformEdges, compileTime: true},
  },
  ...sim.grids.gridSim(pointerSplatField({
    gridSize: p('gridSize'), decay: p('decay'), intensity: p('intensity'), radius: p('radius'),
    output: 'displacement',
  })),
  map: warps.gridCellDisplace({gridSize: p('gridSize'), output: 'displacement'}),
  // Before the field exists (no GPU yet), the warp passes the layer through untouched.
  uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement,
})
```
