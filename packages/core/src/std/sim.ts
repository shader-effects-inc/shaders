/**
 * std/simulate — declare a simulation and let the engine run it.
 *
 * You describe the state (a square grid of numbers), the steps that advance it once per frame
 * (`op.wave`, `op.splat`), and the outputs you want derived from it (`op.gradient`). The engine
 * owns everything else: the state buffers, the frame-to-frame swap, the first-frame reset, the
 * pointer tracking, the frame clock, and stopping the work once the field has settled. An
 * output is a field an effect can read: `displaceBy(sim.output('name'), …)` bends the child by
 * it.
 *
 * Today one recipe is implemented: a damped wave field stirred by the pointer, read through its
 * gradient. A configuration outside that set throws when the shader is defined, never at render
 * time.
 */
// Maintainer notes. Declared state + step operators; the engine owns ping-pong, init,
// scheduling, rest/settle, and device lifecycle (the lowering wires the kit's runtime
// harnesses — `kit/waves.ts` is the first). Phase 0 slice: `simulate.grid` with the wave-field
// op set (`op.wave` + `op.splat` at the pointer, `derive: {…: op.gradient()}`). The op
// vocabulary grows per plan Phase 4; a configuration the lowering doesn't recognize throws at
// definition time, never silently degrades (see `lowerDisplaceBy` in `lower.ts` for the
// validation and the `waves.createWaveFieldSim` wiring).
import type {PropRef} from './values'
import type {PointerSignal, PointerSpeedSignal} from './signal'

// ── Ops ─────────────────────────────────────────────────────────────────────────────────

/**
 * A step that makes the grid ripple like a water surface, fading by `damping`.
 *
 * Needs `history: 2` on the grid, because a wave reads the two previous frames. `damping`
 * is a prop on a 0–20 scale: 0 rings forever, 20 dies in a few frames.
 */
// Damped `avg − prev` wave propagation (reads t−1 AND t−2). The damping prop maps to a
// per-step factor `1 − damping·0.004` in `kit/waves.ts` (DAMPING_SCALE).
export interface WaveOp {
    readonly kind: 'op.wave'
    readonly damping: PropRef
}

/**
 * A step that presses a soft round brush into the grid wherever the pointer moves.
 *
 * `at` is the pointer signal, `amount` the pointer-speed signal that scales the push (a still
 * pointer adds nothing), `radius` a prop on a 0.1–1 scale.
 */
// Gaussian brush injection at a positioned signal, scaled by an amount signal. The UI radius
// is scaled by 0.05 into field space by the lowering (`radiusScale`); the brush is truncated at
// 3σ and re-normalised to reach zero at its edge.
export interface SplatOp {
    readonly kind: 'op.splat'
    readonly at: PointerSignal
    readonly amount: PointerSpeedSignal
    readonly radius: PropRef
}

/**
 * A derived output: the slope of the grid at every cell, as a 2D vector.
 *
 * This is what a distortion wants: `displaceBy` moves each pixel along the slope.
 */
// Central-difference gradient of the field → an RG vector-field texture (a derive op).
export interface GradientOp {
    readonly kind: 'op.gradient'
}

/** Any step that advances the grid each frame. */
export type GridStepOp = WaveOp | SplatOp
/** Any output derived from the grid after the steps have run. */
export type GridDeriveOp = GradientOp

/**
 * The steps and outputs a grid simulation is built from.
 *
 * `op.wave` and `op.splat` go in `step` and run in order once per frame. `op.gradient` goes in
 * `derive`, named, and becomes an output an effect can read with `sim.output('name')`.
 *
 * @example
 * ```ts
 * step: [op.wave({damping: p('decay')}), op.splat({at: pointer({teleportGuard: 'on'}), amount: pointerSpeed({max: 2}), radius: p('radius')})],
 * derive: {displacement: op.gradient()},
 * ```
 * @tip The current engine implements exactly `[op.wave, op.splat]` with `op.gradient` derived. Other combinations throw at definition time.
 * @see simulate, GridSim
 */
export const op = {
    /**
     * A wave step: the grid ripples outward from any disturbance and fades by `damping` (a prop,
     * 0–20).
     *
     * @example
     * ```ts
     * op.wave({damping: p('decay')})
     * ```
     * @tip Declare `history: 2` on the grid. A wave needs the two previous frames.
     * @see op.splat
     */
    wave(config: {damping: PropRef}): WaveOp {
        return {kind: 'op.wave', damping: config.damping}
    },
    /**
     * A brush step: a soft round push into the grid at the pointer, scaled by how fast it
     * moves. `radius` is a prop on a 0.1–1 scale.
     *
     * @example
     * ```ts
     * op.splat({at: pointer({teleportGuard: 'on'}), amount: pointerSpeed({max: 2}), radius: p('radius')})
     * ```
     * @tip Use `pointer({teleportGuard: 'on'})`. The pointer entering the canvas is then not a stroke.
     * @see op.wave, pointer, pointerSpeed
     */
    splat(config: {at: PointerSignal; amount: PointerSpeedSignal; radius: PropRef}): SplatOp {
        return {kind: 'op.splat', at: config.at, amount: config.amount, radius: config.radius}
    },
    /**
     * A derived output: the grid's slope at every cell as a 2D vector, ready for `displaceBy`.
     *
     * @example
     * ```ts
     * derive: {displacement: op.gradient()}
     * ```
     * @see displaceBy
     */
    gradient(): GradientOp {
        return {kind: 'op.gradient'}
    },
} as const

// ── Grid simulation ─────────────────────────────────────────────────────────────────────

/**
 * What a grid simulation declares: its size, how many past frames it keeps, its steps, and its
 * named outputs.
 */
export interface GridSimConfig {
    /** Cells per side of the square grid. 128 is plenty for a ripple field. */
    resolution: number
    /** How many past frames the steps read. A wave needs 2. */
    history: number
    /** The steps that advance the grid, run in this order once per frame. */
    step: GridStepOp[]
    /** Named outputs an effect can read through `sim.output(name)`. */
    derive: Record<string, GridDeriveOp>
    /** Let the engine stop running the simulation once the field has faded to nothing. */
    // 'derived-from-damping': the settle window is computed from the damping value each frame
    // (see `createWaveFieldSim`); low damping keeps the sim running, high damping sleeps fast.
    rest?: {settlesWhen: 'derived-from-damping'}
}

/**
 * A handle to one named output of a simulation, the thing you hand to `displaceBy`.
 *
 * @see displaceBy, GridSim
 */
export interface SimOutputRef {
    readonly kind: 'simOutput'
    readonly sim: GridSim
    readonly output: string
}

/**
 * A declared grid simulation. Read its outputs with `output('name')` and pass them to an effect.
 *
 * @example
 * ```ts
 * effect: displaceBy(waves.output('displacement'), {strength: p('intensity'), chromatic: p('chromaticSplit'), edges: p('edges')})
 * ```
 * @tip Asking for an output that was not declared in `derive` throws with the list of names that were.
 * @see simulate, SimOutputRef
 */
export class GridSim {
    constructor(readonly config: GridSimConfig) {}

    output(name: string): SimOutputRef {
        if (!(name in this.config.derive)) {
            throw new Error(`std: simulation has no derived output '${name}' (declared: ${Object.keys(this.config.derive).join(', ')})`)
        }
        return {kind: 'simOutput', sim: this, output: name}
    }
}

/**
 * Declare a simulation the engine runs for you. `simulate.grid` gives a square grid of numbers
 * advanced once per frame by the steps you list.
 *
 * Declare it once at module scope, then read its outputs inside the definition. The engine
 * allocates the state, keeps the history, tracks the pointer and sleeps the simulation once it
 * has settled.
 *
 * @example
 * ```ts
 * const waves = simulate.grid({resolution: 128, history: 2, step: [op.wave({damping: p('decay')}), op.splat({at: pointer({teleportGuard: 'on'}), amount: pointerSpeed({max: 2}), radius: p('radius')})], derive: {displacement: op.gradient()}, rest: {settlesWhen: 'derived-from-damping'}})
 * ```
 * @tip Reach for `sim.grids`, `sim.fluids`, `sim.feedback` or `sim.agents` only when the effect needs its own GPU step. If a wave field and a displacement say it, stay here.
 * @see op, GridSim, displaceBy
 */
export const simulate = {
    /**
     * A square grid of numbers advanced once per frame by `step`, with named outputs in `derive`.
     *
     * @example
     * ```ts
     * const waves = simulate.grid({resolution: 128, history: 2, step: [op.wave({damping: p('decay')}), op.splat({at: pointer({teleportGuard: 'on'}), amount: pointerSpeed({max: 2}), radius: p('radius')})], derive: {displacement: op.gradient()}})
     * ```
     * @see op, GridSim
     */
    grid(config: GridSimConfig): GridSim {
        return new GridSim(config)
    },
} as const
