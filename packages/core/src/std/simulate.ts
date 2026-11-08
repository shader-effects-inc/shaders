/**
 * Simulations the engine runs for you. List one in a definition's `inputs` and read it in WGSL by
 * name. The engine owns the state, the frame-to-frame swap, the first-frame seed, the pointer, the
 * clock and the dispatch; you write only what is particular to your effect.
 *
 * - `simulate.grid`: a field of `vec4f` cells advanced by steps (`op.*` or your own kernels).
 * - `simulate.particles`: thousands of points that move, drawn into a texture.
 * - `simulate.fluid`: a real fluid solve carrying coloured dye.
 */
// Maintainer notes. Module namespace (`export * as simulate`), so each simulation's GPU builder
// is only bundled when its word is used. `grid` keeps returning a `GridSim` (CursorRipples'
// `displaceBy` reads `sim.config`), with the input builder attached under the build symbol.
import {GridSim, type GridSimConfig, type GridSimInit} from './sim'
import {withBuilder, type InputNode} from './inputs/types'
import {buildGrid} from './inputs/grid'

import {particlesInput, type ParticlesInput} from './inputs/particles'
import {fluidInput, type FluidInput} from './inputs/fluid'

/**
 * A grid of `vec4f` cells advanced once per frame by `step`, with named outputs in `derive`.
 * Read the grid in WGSL as `name(uv)` (bilinear between cell centres) and an output as
 * `name.output(uv)`.
 *
 * - `resolution`: cells per side of a square grid, or `[width, height]` (it does not follow the
 *   frame's shape: give `[w, h]` in the frame's proportions). Its cost is fixed, whatever the canvas size.
 * - `history: 2` lets steps read last frame's state with `readPrev` (a wave, or Verlet motion).
 * - `precision: 'full'` stores 32-bit cells, for positions and anything that moves by tiny amounts.
 * - `init` seeds every cell once: a WGSL body returning `vec4f` that sees `cell`, `cellUV` (the
 *   cell's centre, 0–1), `gridSize`, props, helpers and library functions.
 * - `step` runs in order every frame: `op.*` steps and your own `op.kernel` bodies. A kernel reads
 *   the grid (`read`, `readPrev`, `readLinear`), the built-ins (`dt` follows the layer's Speed),
 *   props, helpers and other inputs by name (`clicks(i)`, another grid's `field(uv)`).
 * - Revising the component restarts the grid from `init`; a capture continues it.
 *
 * @example
 * ```ts
 * inputs: {
 *   field: simulate.grid({
 *     resolution: 256, history: 2,
 *     step: [op.wave({damping: p('decay')}), op.splat({at: pointer(), amount: pointerSpeed({max: 2}), radius: p('radius')})],
 *     derive: {slope: op.gradient()},
 *   }),
 * },
 * effect: wgsl`return textureSample(childTexture, childSampler, uv + field.slope(uv).xy * intensity);`,
 * ```
 * @see op, displaceBy
 */
export function grid(config: GridSimInit): GridSim & InputNode {
    const full: GridSimConfig = {...config, history: config.history ?? 1, derive: config.derive ?? {}}
    if (full.history !== 1 && full.history !== 2) throw new Error('simulate.grid: history must be 1 or 2')
    const sim = new GridSim(full)
    return withBuilder(sim, (ctx) => buildGrid(ctx, full))
}

/**
 * Thousands of moving points drawn into a texture, read by name in WGSL.
 *
 * - `count` (number or prop) up to `max`; `mobile: {max}` caps it on phones.
 * - `forces`: data, e.g. `[{gravity: 0.3}, {drag: 0.5}, {cursor: 1}, {wind: 0.4}]`.
 * - `spawn`: `'scatter'` (default), `'ring'`, `'edge'`, `'atPointer'`, `{fromShape: source}` (a
 *   texture's opaque area, e.g. `samplePoints` or a layer prop), or a WGSL body returning a
 *   `Particle` (it sees `index` and `rand: vec4f`).
 * - `step`: WGSL run per particle per frame with `var p: Particle` in scope (and `index`, `rand`).
 *   Forces are applied before it; position integrates after it.
 * - `lifetime: {seconds, jitter}` respawns particles as they age out.
 * - `draw`: WGSL returning `Point(pos, sizePx, color)` per particle (it sees `p` and `index`).
 *   Without a state (no spawn, step or forces) it sees only `index`: the formula-driven mode.
 * - `splat`: `'point'` (1 px) or `'soft'` (a round footprint `size` pixels wide).
 * - `neighbors: {radius}` lets `step` read `crowd(p)` and `crowdSlope(p)`: how crowded a spot is.
 *
 * @example
 * ```ts
 * inputs: {
 *   motes: simulate.particles({count: 20000, forces: [{wind: 0.4}, {cursor: 1}, {drag: 0.6}], lifetime: {seconds: 6, jitter: 0.5}, splat: 'soft'}),
 * },
 * paint: wgsl`return motes(uv);`,
 * ```
 */
export function particles(opts: Parameters<typeof particlesInput>[0]): ParticlesInput {
    return particlesInput(opts)
}

/**
 * A fluid solve with coloured dye.
 *
 * - `velocity` / `dye.res`: grid sizes (cells on the long side). 256 / 512 is a good start.
 * - `dye.space: 'oklab'` carries the dye in OKLab, so colours mix cleanly instead of going muddy.
 * - `forces`: WGSL returning a `vec2f` acceleration for this cell (uv per second²). It sees
 *   `cellUV`, `velocity` (this cell's), `dye` (this cell's colour), and everything a body sees.
 * - `inject`: WGSL returning a `vec4f` of dye to add this frame at this cell (premultiplied: the
 *   alpha is how much). It sees `dye`, the cell's dye before this frame's addition (straight
 *   linear colour, alpha how much), so a spout can top up to a set thickness:
 *   `return vec4f(color.rgb, 1.0) * max(target - dye.a, 0.0)`. `isNew(period)` is true on the
 *   first frame of each `period`-second slot: use it to add a drop once instead of every frame.
 * - `init`: WGSL returning the dye's first state at this cell (straight linear colour, alpha how
 *   much), written once when the fluid starts: a pool that's already full.
 * - `stir`, `dissipation` and `drain.floor` take a number or a prop (`stir: p('smear')`), read
 *   live each frame.
 * - `stir`: how hard the pointer pushes the fluid (0 turns it off; default 1). A drag swirls it;
 *   for clicks, read `events('click')` in `inject` and add dye where they land.
 * - `boundary.floor: 'solid'` makes the bottom a wall dye piles up against (paint pooling);
 *   `boundary.walls: 'solid'` does the same at the sides. `drain.floor` removes that much dye per
 *   second near the floor so the canvas never fills.
 * - `dissipation`: how fast the flow's motion dies away (default 0.2; 0 keeps it swirling, higher
 *   feels thicker). The dye itself never fades; `drain` removes it.
 * - `pressureIterations`: solver accuracy (default 20); more keeps swirls tighter, at a cost.
 * - Read the dye as `ink(uv)`: straight-alpha colour, alpha how much dye is there (in OKLab mode it
 *   comes back as ordinary linear colour). `ink.velocity(uv).xy` is the flow, in uv per second.
 *   Shade from the dye's alpha (its slope gives a surface normal) for a thick, glossy look.
 * - The solve runs at a fixed size, so its cost doesn't grow with the canvas; `dt` follows Speed.
 *
 * @example
 * ```ts
 * inputs: {
 *   ink: simulate.fluid({
 *     velocity: 256, dye: {res: 512, space: 'oklab'}, boundary: {floor: 'solid'}, drain: {floor: 0.05},
 *     forces: wgsl`return vec2f(0.0, gravity * dye.a);`,
 *     inject: wgsl`
 *       let id = floor(time * drops);
 *       let at = vec2f(hash11(id), 0.08);
 *       let k = select(0.0, 1.0, isNew(1.0 / drops)) * (1.0 - smoothstep(0.0, 0.03, distance(cellUV, at)));
 *       return colors(hash11(id + 1.0)) * k;
 *     `,
 *   }),
 * },
 * paint: wgsl`return ink(uv);`,
 * ```
 */
export function fluid(opts: Parameters<typeof fluidInput>[0] = {}): FluidInput {
    return fluidInput(opts)
}
