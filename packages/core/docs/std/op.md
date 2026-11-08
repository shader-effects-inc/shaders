# Grid steps

The steps a `grid` runs every frame, and the outputs it derives. Every cell holds a
`vec4f`; most steps work on `.x` and leave the other channels for you. Steps run in the order you
list them. `op.kernel` is your own step in WGSL, for anything the others do not cover.

## Reach for it when

| When | Use |
|---|---|
| ripples from the cursor | `wave` with `splat`, derive `gradient` |
| something spreads and fades | `diffuse`, `decay` |
| organic spots, stripes, mazes | `reactDiffuse` |
| a growing frost or crackle edge that can retreat | `growthFront`, with `occupancy` so fronts stop on contact |
| a cellular automaton (Life) | `automaton` |
| a temperature that spreads and cools | `heat` |
| your own rule | `kernel` |

## Order

- wave
- splat
- gradient
- diffuse
- decay
- reactDiffuse
- growthFront
- occupancy
- automaton
- heat
- kernel
