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
import type {GridDeriveOp, GridStepOp} from './ops'
import type {WgslBody} from './wgsl'

// ── Ops ─────────────────────────────────────────────────────────────────────────────────

// The step vocabulary lives in ./ops (pure data). `op` is its module namespace, so a bundler keeps
// only the ops a component uses.
export * as op from './ops'
export type {
    WaveOp, SplatOp, GradientOp, DiffuseOp, DecayOp, ReactDiffuseOp, AutomatonOp, HeatOp,
    GrowthFrontOp, OccupancyOp, CellKernelOp, GridStepOp, GridDeriveOp, OpArg,
} from './ops'

// ── Grid simulation ─────────────────────────────────────────────────────────────────────

/**
 * What a grid simulation declares: its size, how many past frames it keeps, its steps, and its
 * named outputs.
 */
export interface GridSimConfig {
    /** Cells per side of a square grid, or `[width, height]`. 128 is plenty for a ripple field. */
    resolution: number | readonly [number, number]
    /** How many past states the steps read (1 or 2). A wave needs 2. */
    history: number
    /**
     * How precisely each cell is stored (default 'half', 16-bit floats, about 3 significant
     * digits). 'full' stores 32-bit floats: use it when cells hold positions or anything that
     * moves by tiny amounts per frame (cloth, Verlet), where half precision rounds small steps to
     * nothing. Costs twice the memory; reads are exact but unfiltered-then-blended by hand.
     */
    precision?: 'half' | 'full'
    /** Each cell's starting value, as a WGSL body returning `vec4f` (it sees `cell`, `cellUV`). Default 0. */
    init?: WgslBody | string
    /** The steps that advance the grid, run in this order once per frame. */
    step: GridStepOp[]
    /** Named outputs an effect can read through `sim.output(name)`. */
    derive: Record<string, GridDeriveOp>
    /** Let the engine stop running the simulation once the field has faded to nothing. */
    // 'derived-from-damping': the settle window is computed from the damping value each frame
    // (see `createWaveFieldSim`); low damping keeps the sim running, high damping sleeps fast.
    rest?: {settlesWhen: 'derived-from-damping'}
}

/** What `simulate.grid` accepts: `history` defaults to 1, `derive` to none. */
export type GridSimInit = Omit<GridSimConfig, 'history' | 'derive'> & {history?: number; derive?: Record<string, GridDeriveOp>}

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
    readonly kind = 'simulate.grid' as const
    constructor(readonly config: GridSimConfig) {}

    /** The derived outputs, readable in WGSL as `name.output(uv)`. */
    get outputNames(): string[] {
        return Object.keys(this.config.derive)
    }

    /** 32-bit state can't go through a filtering sampler on every GPU: bodies read it exactly. */
    get unfilterable(): boolean {
        return this.config.precision === 'full'
    }

    output(name: string): SimOutputRef {
        if (!(name in this.config.derive)) {
            throw new Error(`std: simulation has no derived output '${name}' (declared: ${Object.keys(this.config.derive).join(', ')})`)
        }
        return {kind: 'simOutput', sim: this, output: name}
    }
}

export * as simulate from './simulate'
