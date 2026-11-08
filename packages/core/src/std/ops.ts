/**
 * Simulation steps, as data. Put them in a `simulate.grid` `step` list (they run in order, once
 * per frame) or `derive` map (named outputs computed from the grid). Every grid cell holds a
 * `vec4f`; most steps work on `.x` and leave the other channels for you.
 */
// Maintainer notes. Pure data constructors: each returns `{kind: 'op.<name>', …args}` and nothing
// else, so this module is free to import anywhere and every op serialises to the IR as is. The
// WGSL for each op lives with the grid builder (std/inputs/grid.ts), which is the only code that
// needs it.
import type {PropRef} from './values'
import type {PointerSignal, PointerSpeedSignal} from './signal'
import type {WgslBody} from './wgsl'

/** A number, or a prop read live each frame. */
export type OpArg = number | PropRef

/** Ripple like a water surface, fading by `damping` (0 rings forever, 20 dies in a few frames). Needs `history: 2`. */
export interface WaveOp {
    readonly kind: 'op.wave'
    readonly damping: PropRef | number
}
/** Press a soft round brush into `.x` wherever the pointer moves (`radius` 0.1–1, a fraction of the grid). */
export interface SplatOp {
    readonly kind: 'op.splat'
    readonly at: PointerSignal
    readonly amount: PointerSpeedSignal
    readonly radius: PropRef | number
}
/** The slope of `.x` at every cell, as `.xy`. A derived output. */
export interface GradientOp {
    readonly kind: 'op.gradient'
}
/** Spread every channel into the neighbours at `rate` per second (a blur, as a step). */
export interface DiffuseOp {
    readonly kind: 'op.diffuse'
    readonly rate: OpArg
}
/** Fade every channel toward 0, halving every `halfLife` seconds. */
export interface DecayOp {
    readonly kind: 'op.decay'
    readonly halfLife: OpArg
}
/** Gray-Scott reaction-diffusion on `.x` (A) and `.y` (B). `feed` and `kill` choose spots, stripes or mazes. */
export interface ReactDiffuseOp {
    readonly kind: 'op.reactDiffuse'
    readonly feed: OpArg
    readonly kill: OpArg
    readonly speed: OpArg
}
/** A cellular automaton on `.x` (0 or 1): born with `birth` live neighbours, survives with `survive`. Conway's Life is `{birth: [3], survive: [2, 3]}`. */
export interface AutomatonOp {
    readonly kind: 'op.automaton'
    readonly birth: readonly number[]
    readonly survive: readonly number[]
}
/** A temperature in `.x` that spreads (`spread`) and cools (`cool` per second). */
export interface HeatOp {
    readonly kind: 'op.heat'
    readonly spread: OpArg
    readonly cool: OpArg
}
/** A spreading boundary in `.x` (0 → 1) that creeps into neighbours at `rate` cells per second, ragged by `roughness`; `reverse` makes it retreat. Gate any effect by it. */
export interface GrowthFrontOp {
    readonly kind: 'op.growthFront'
    readonly rate: OpArg
    readonly roughness: OpArg
    readonly reverse: OpArg
}
/** First-come-wins claims in `.w`: an empty cell next to a claimed one takes its claim, so fronts from different seeds merge and stop instead of overwriting. */
export interface OccupancyOp {
    readonly kind: 'op.occupancy'
}
/** Your own step: a WGSL body run per cell, returning the cell's new `vec4f`. */
export interface CellKernelOp {
    readonly kind: 'op.kernel'
    readonly body: string
    readonly iterations: number
}

/** Any step a grid can run. */
export type GridStepOp = WaveOp | SplatOp | DiffuseOp | DecayOp | ReactDiffuseOp | AutomatonOp | HeatOp | GrowthFrontOp | OccupancyOp | CellKernelOp
/** Any output a grid can derive. */
export type GridDeriveOp = GradientOp | CellKernelOp

/**
 * A wave step: the grid ripples outward from any disturbance and fades by `damping` (a prop or
 * number, 0–20). Declare `history: 2` on the grid.
 *
 * @example
 * ```ts
 * op.wave({damping: p('decay')})
 * ```
 * @see splat
 */
export function wave(config: {damping: PropRef | number}): WaveOp {
    return {kind: 'op.wave', damping: config.damping}
}

/**
 * A brush step: a soft round push into `.x` at the pointer, scaled by how fast it moves.
 *
 * @example
 * ```ts
 * op.splat({at: pointer(), amount: pointerSpeed({max: 2}), radius: p('radius')})
 * ```
 * @see wave, pointer, pointerSpeed
 */
export function splat(config: {at: PointerSignal; amount: PointerSpeedSignal; radius: PropRef | number}): SplatOp {
    return {kind: 'op.splat', at: config.at, amount: config.amount, radius: config.radius}
}

/**
 * A derived output: the slope of `.x` at every cell as `.xy`, ready for a displacement.
 *
 * @example
 * ```ts
 * derive: {slope: op.gradient()}
 * ```
 */
export function gradient(): GradientOp {
    return {kind: 'op.gradient'}
}

/** Spread every channel into its neighbours at `rate` per second. @example `op.diffuse({rate: 0.5})` */
export function diffuse(config: {rate: OpArg}): DiffuseOp {
    return {kind: 'op.diffuse', rate: config.rate}
}

/** Fade every channel toward 0, halving every `halfLife` seconds. @example `op.decay({halfLife: 1.5})` */
export function decay(config: {halfLife: OpArg}): DecayOp {
    return {kind: 'op.decay', halfLife: config.halfLife}
}

/**
 * Gray-Scott reaction-diffusion on `.x` and `.y`. Seed `.y` with `init`. Classic settings:
 * spots `feed 0.035, kill 0.065`; mazes `0.029, 0.057`; coral `0.0545, 0.062`.
 *
 * @example
 * ```ts
 * op.reactDiffuse({feed: 0.055, kill: 0.062})
 * ```
 */
export function reactDiffuse(config: {feed: OpArg; kill: OpArg; speed?: OpArg}): ReactDiffuseOp {
    return {kind: 'op.reactDiffuse', feed: config.feed, kill: config.kill, speed: config.speed ?? 1}
}

/** A cellular automaton on `.x`. @example `op.automaton({birth: [3], survive: [2, 3]})` */
export function automaton(config: {birth: readonly number[]; survive: readonly number[]}): AutomatonOp {
    return {kind: 'op.automaton', birth: [...config.birth], survive: [...config.survive]}
}

/** A temperature in `.x` that spreads and cools. @example `op.heat({spread: 0.3, cool: 0.4})` */
export function heat(config: {spread: OpArg; cool: OpArg}): HeatOp {
    return {kind: 'op.heat', spread: config.spread, cool: config.cool}
}

/**
 * A spreading boundary in `.x` that creeps outward from seeded cells (`.x = 1` from `init` or a
 * splat) at `rate` cells per second. `reverse` (0 or 1, a prop is fine) makes it retreat.
 *
 * @example
 * ```ts
 * op.growthFront({rate: 6, roughness: 0.6, reverse: p('melt')})
 * ```
 */
export function growthFront(config: {rate: OpArg; roughness?: OpArg; reverse?: OpArg}): GrowthFrontOp {
    return {kind: 'op.growthFront', rate: config.rate, roughness: config.roughness ?? 0.5, reverse: config.reverse ?? 0}
}

/** First-come-wins claims in `.w`. Seed claims (any value > 0) with `init`. @example `op.occupancy()` */
export function occupancy(): OccupancyOp {
    return {kind: 'op.occupancy'}
}

/**
 * Your own grid step, run once per cell: return the cell's new `vec4f`. It reads `read(c)` (any
 * cell this step, clamped at the edges), `readLinear(p)` (the grid at any uv `p`, bilinear between
 * cell centres: for advection, `readLinear(cellUV - velocity * dt)`), `readPrev(c)` (the state at
 * the end of the frame before), `cell: vec2i`, `cellUV: vec2f` (the cell's centre), `gridSize: vec2i`,
 * plus props, built-ins (`dt` is the layer clock's step, so Speed scales it), inputs and library functions.
 * `iterations` repeats it within the frame, each pass reading the one before (Jacobi style).
 *
 * @example
 * ```ts
 * op.kernel(wgsl`
 *   let c = read(cell);
 *   let avg = (read(cell + vec2i(1, 0)) + read(cell - vec2i(1, 0))) * 0.5;
 *   return mix(c, avg, 0.5);
 * `)
 * ```
 */
export function kernel(body: WgslBody | string, opts: {iterations?: number} = {}): CellKernelOp {
    const text = typeof body === 'string' ? body : body.spec.body
    return {kind: 'op.kernel', body: text, iterations: Math.max(1, Math.min(64, Math.round(opts.iterations ?? 1)))}
}
