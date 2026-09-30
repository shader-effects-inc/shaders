/**
 * std/sim/grids — a grid of cells advanced by a program of named stages.
 *
 * A grid simulation is a fixed-size field of cells (a displacement per cell, a chemical
 * concentration, a sort offset) that changes a little every frame. You describe the frame as
 * an ordered list of stages with `op.*`: read the pointer, write this frame's values, run a
 * GPU pass, repeat a pass N times, publish the result into the texture the layer samples. The
 * engine runs that list every frame, in order, and owns the clock, the frame-delta clamp, the
 * ready and settle gates, and the outputs the fragment reads. Once nothing has driven the
 * field for a while, `op.settle` lets the engine stop dispatching and the last picture stays.
 *
 * Two hosts: `gridSim` for fields stepped on the GPU, and `hostGridProgram` for small fields
 * stepped in plain JavaScript and uploaded as a texture with `hostFieldTexture`. When a wave
 * field and a displacement are all the effect needs, declare it with `simulate.grid` instead.
 */
// Maintainer notes. GridDistortion, Liquify, PixelSort and ReactionDiffusion each hand-wrote
// this frame program; `gridSim` owns the skeleton ONCE and each shader declares its program as
// named `op.*` stages — the kernels, layouts and constants stay in the shader file (the
// constants ARE the look):
//
//   ...gridSim((params, root) => ({
//       outputs: {displacement},
//       clampDt: 0.016,
//       stages: [
//           op.host('cursorVelocity', (f) => {...}),
//           op.settle({activeWhen: () => active, settleMs: () => ...}),
//           op.values('write', (f) => paramsU.write({...})),
//           op.pass(update),
//           op.publish(output),
//       ],
//   }))
//
// Frame-program phases (identical to every hand-written consumer):
//   ready gates → dt clock → host ticks → settle skips → emits (writes + dispatches), in stage order.
//
// ChromaFlow and PixelThrow are the CPU members of the family: their fields are advanced by host
// JS loops and uploaded as a data texture from inside the fragment builder (no compute hook).
// Their loops stay on the CPU — moving them to WGSL would re-derive the arithmetic, not relocate
// it — but the frame program becomes the same visible ordered stage list via `hostGridProgram` +
// `hostFieldTexture`.
//
// Compile-time vs runtime (C5): everything in a config is STRUCTURAL (pipelines, stage order —
// recompose to change); the values each stage writes per frame are the runtime surface, exactly
// as before. The build callback runs once per component instance, so stage state (parity ticks,
// settle clocks, velocity trackers) created inside it is per-instance.
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'

/** The GPU device handle the build callback receives, for creating the buffers and passes your stages use. */
// The TypeGPU root, via the contract (direct `typegpu` imports are restricted to src/gpu/).
export type GridRoot = NonNullable<GpuFragmentParams['gpu']>['root']

/** What the renderer reports each frame: the pointer in uv, the frame delta in seconds, the canvas size in pixels. */
export interface GridFrameParams {
    pointer?: {x: number; y: number}
    deltaTime?: number
    dimensions?: {width: number; height: number}
}

/** One frame, as every stage sees it: the clamped delta, the clock, and your props. */
export interface GridFrame {
    /** Seconds since the last frame, never more than `clampDt`. */
    dt: number
    /** Wall-clock milliseconds this frame. */
    // `Date.now()` — the settle gates' clock.
    now: number
    /** A numeric prop's current value, or `fallback` when it is not a number this frame. */
    num(key: string, fallback: number): number
    /** Any prop's current value. */
    getCpuValue(key: string): unknown
    frameParams: GridFrameParams
}

/**
 * One stage of the frame. Build them with `op.*` rather than by hand.
 *
 * Each frame runs every stage's `ready`, then every `tick`, then every `skip`, then every
 * `emit`, in list order.
 */
// Phases run in order: ready → tick → skip → emit.
export interface GridStage {
    /** Return false to run nothing this frame (an input the stage needs has not arrived yet). */
    // Frame gate: false → the whole frame returns null (a late-bound input is still pending).
    ready?(): boolean
    /** Bookkeeping before any gate: pointer smoothing, activity flags. */
    tick?(f: GridFrame): void
    /** Return true to skip the whole frame. The last published texture stays on screen. */
    skip?(f: GridFrame): boolean
    /** Add this stage's GPU passes and value writes to the frame. */
    emit?(f: GridFrame, nodes: ComputeStep[]): void
}

/** What a GPU grid simulation declares: its outputs, its frame-delta clamp, and its stages. */
export interface GridSimConfig {
    /** The textures the fragment samples, by the name it reads them under in `computeOutputs`. */
    // State/display textures, child RTT handles.
    outputs: Record<string, unknown>
    /** Receive the child's picture once it exists (it is allocated after the layer composes). */
    // Late input binding (child RTT → luma prepass, child-modulated kernels). Spread only if set.
    bindInputs?: (resolve: (key: string) => {texture: unknown} | undefined) => void
    /** Longest frame step accepted, in seconds. 0.016–0.05 keeps a tab switch from jumping the field. */
    clampDt: number
    /** What to do on a frame with no elapsed time: run anyway (default) or skip it. */
    // Liquify's guard.
    zeroDt?: 'run' | 'skip'
    /** The frame program, in order. */
    stages: GridStage[]
}

/**
 * A field of cells advanced every frame by an ordered list of stages. Spread it into a
 * definition next to a `gpu.fragment` that samples one of its outputs.
 *
 * The build callback runs once per instance of the layer, with the GPU device, and returns the
 * outputs, the frame-delta clamp and the stages. Return null to run nothing (a required child is
 * missing). Without a GPU the fragment should fall back to the child or to transparent.
 *
 * @example
 * ```ts
 * ...gridSim((params, root) => ({outputs: {state}, clampDt: 0.05, stages: [op.values('write', (f) => paramsU.write({dt: f.dt, radius: f.num('radius', 0.4)})), op.pass(update), op.publish(() => output)]}))
 * ```
 * @tip Everything created in the build callback (buffers, passes, trackers) belongs to that layer instance. Create nothing at module scope that holds per-frame state.
 * @see op, hostGridProgram, simulate, fluidSim, feedbackSim
 */
// The build callback runs at compute-node creation (per instance, after the no-device bail).
export function gridSim(
    build: (params: GpuFragmentParams, root: GridRoot) => GridSimConfig | null,
): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {gpu, getCpuValue} = params
            const root = gpu?.root
            if (!root) return null // GPU-free / no device: fragment falls back to its passthrough.
            const cfg = build(params, root)
            if (!cfg) return null

            const num = (key: string, fallback: number): number => {
                const v = getCpuValue(key)
                return typeof v === 'number' ? v : fallback
            }

            return {
                outputs: cfg.outputs as NonNullable<ReturnType<GpuComputeNode>>['outputs'],
                ...(cfg.bindInputs ? {bindInputs: cfg.bindInputs} : {}),
                getComputeNodes: (frameParams: unknown): ComputeStep[] | null => {
                    for (const s of cfg.stages) if (s.ready && !s.ready()) return null
                    const fp = frameParams as GridFrameParams
                    const dt = Math.min(fp.deltaTime ?? 0, cfg.clampDt)
                    if (cfg.zeroDt === 'skip' && dt <= 0) return null
                    const f: GridFrame = {dt, now: Date.now(), num, getCpuValue, frameParams: fp}
                    const nodes: ComputeStep[] = []
                    for (const s of cfg.stages) s.tick?.(f)
                    for (const s of cfg.stages) if (s.skip?.(f)) return null
                    for (const s of cfg.stages) s.emit?.(f, nodes)
                    return nodes
                },
            }
        },
    }
}

// ── Stage (op) factories ───────────────────────────────────────────────────────────────────────

/** A sort chain's stage, which also reports which of its two buffers holds the current result. */
// A parity-alternating sort chain's handle: which ping-pong side is current after this frame.
export interface SortPassStage extends GridStage {
    side(): 'A' | 'B'
}

/**
 * The stages a grid frame is built from. List them in `stages` in the order they should run.
 *
 * Bookkeeping: `host` (before any gate), `values` (write this frame's values), `readyWhen`
 * (wait for an input), `settle` (sleep once the field is still). Work: `pass` (one GPU pass),
 * `cache` (a pass picked per frame), `seedOnce` (a one-shot reset), `iterate` (a pass N times),
 * `sortPass` (a compare-swap chain), `publish` (write the result the layer samples).
 *
 * @example
 * ```ts
 * stages: [op.readyWhen(() => ready), op.values('write', (f) => paramsU.write({dt: f.dt})), op.iterate({count: () => 8, step: (i, nodes) => nodes.push(react.with(i % 2 ? groupBA : groupAB))}), op.publish(() => output)]
 * ```
 * @tip A `values` stage sits before the passes that read it. Stages run in list order.
 * @see gridSim
 */
export const op = {
    /**
     * A named step that runs before any gate: pointer smoothing, activity flags, anything a
     * later `settle` needs to know.
     *
     * @example
     * ```ts
     * op.host('cursor', (f) => { active = (f.frameParams.pointer?.x ?? 0.5) !== lastX })
     * ```
     * @see values, settle
     */
    // Runs in the tick phase. `name` is documentation only.
    host(name: string, run: (f: GridFrame) => void): GridStage {
        void name
        return {tick: run}
    },

    /**
     * A named step that writes this frame's values for the passes after it.
     *
     * @example
     * ```ts
     * op.values('write', (f) => paramsU.write({dt: f.dt, radius: f.num('radius', 0.4)}))
     * ```
     * @tip Read props here with `f.num('prop', fallback)` so a prop that is briefly not a number cannot reach the GPU as `undefined`.
     * @see host, pass
     */
    // Runs in the emit phase (per-frame uniform writes / prop derivation).
    values(name: string, run: (f: GridFrame) => void): GridStage {
        void name
        return {emit: (f) => run(f)}
    },

    /**
     * Sleep the simulation once nothing has driven it for `settleMs` of simulated time. Frames
     * are skipped and the last published texture stays on screen, so a still field costs nothing.
     *
     * `activeWhen` says whether something drove the field this frame. `settleMs` is how long the
     * field takes to fade after the last activity, in milliseconds.
     *
     * @example
     * ```ts
     * op.settle({activeWhen: () => moving, settleMs: (f) => 4000 / Math.max(f.num('decay', 1), 0.1)})
     * ```
     * @tip Simulated time, not wall-clock: a hidden tab does not count, so the field is still fading when the user comes back.
     * @see host
     */
    // Simulated, not wall-clock: a hidden tab stops rAF, so no frames step the field while the
    // user is away. Measured on the wall clock the gate would freeze the sim the moment the tab
    // came back with the field still fully visible (it never decayed). Only frames that actually
    // dispatch count toward the settle budget.
    settle(opts: {activeWhen: (f: GridFrame) => boolean; settleMs: (f: GridFrame) => number}): GridStage {
        /** Simulated ms stepped since the last active frame. */
        let simSinceActive = 0
        return {
            tick(f) {
                if (opts.activeWhen(f)) simSinceActive = 0
            },
            skip(f) {
                const skip = simSinceActive > opts.settleMs(f)
                if (!skip) simSinceActive += f.dt * 1000
                return skip
            },
        }
    },

    /**
     * Run nothing until `fn` returns true, typically once the child's picture has arrived
     * through `bindInputs`.
     *
     * @example
     * ```ts
     * op.readyWhen(() => lumaReady)
     * ```
     * @see cache
     */
    readyWhen(fn: () => boolean): GridStage {
        return {ready: fn}
    },

    /**
     * One GPU pass, the same every frame.
     *
     * @example
     * ```ts
     * op.pass(update)
     * ```
     * @see cache, iterate, publish
     */
    pass(step: ComputeStep): GridStage {
        return {emit: (_f, nodes) => nodes.push(step)}
    },

    /**
     * A GPU pass chosen per frame, for a pass whose input is rebound after the frame starts.
     *
     * @example
     * ```ts
     * op.cache(() => lumaPipeline)
     * ```
     * @see readyWhen, pass
     */
    // PixelSort's late-rebound child-luma snapshot.
    cache(step: (f: GridFrame) => ComputeStep): GridStage {
        return {emit: (f, nodes) => nodes.push(step(f))}
    },

    /**
     * The last stage: write the state into the texture the layer samples. Chosen per frame so it
     * can follow whichever buffer holds the current result.
     *
     * @example
     * ```ts
     * op.publish(() => outputPass.with(sort.side() === 'A' ? outputBgA : outputBgB))
     * ```
     * @see pass, sortPass
     */
    publish(step: (f: GridFrame) => ComputeStep): GridStage {
        return {emit: (f, nodes) => nodes.push(step(f))}
    },

    /**
     * A one-shot reset run before the rest of the frame whenever `staleWhen` says the state is
     * stale: on the first frame, or when a seed prop changes.
     *
     * @example
     * ```ts
     * op.seedOnce({pass: seed, staleWhen: () => needsSeed, onSeed: () => { needsSeed = false }})
     * ```
     * @see iterate
     */
    seedOnce(opts: {pass: ComputeStep; staleWhen: (f: GridFrame) => boolean; onSeed?: () => void}): GridStage {
        return {
            emit(f, nodes) {
                if (!opts.staleWhen(f)) return
                opts.onSeed?.()
                nodes.push(opts.pass)
            },
        }
    },

    /**
     * Run a step `count` times this frame. Your `step` adds the pass for iteration `i` and picks
     * which buffer it reads and writes.
     *
     * @example
     * ```ts
     * op.iterate({count: (f) => Math.round(f.num('speed', 8)), step: (i, nodes) => nodes.push(react.with(i % 2 ? groupBA : groupAB))})
     * ```
     * @tip A reaction-diffusion field reads smoother when you raise the iteration count than when you raise the step size.
     * @see pass, sortPass
     */
    // The caller's `step` handles its own ping-pong swap.
    iterate(opts: {count: (f: GridFrame) => number; step: (i: number, nodes: ComputeStep[], f: GridFrame) => void}): GridStage {
        return {
            emit(f, nodes) {
                const n = opts.count(f)
                for (let i = 0; i < n; i++) opts.step(i, nodes, f)
            },
        }
    },

    /**
     * A compare-and-swap sort that converges a little more every frame: `passes` swaps per frame,
     * alternating odd and even pairs and the two buffers. `side()` tells `publish` which buffer
     * holds the current order.
     *
     * @example
     * ```ts
     * const sort = op.sortPass({passes: () => 1 + Math.round(strength * 4), pass: (parity, side) => (parity ? swap1 : swap0).with(side === 'A' ? groupAB : groupBA)})
     * ```
     * @tip The alternation continues across frames instead of restarting, which is what makes the sort settle bit by bit.
     * @see publish, iterate
     */
    // Odd-even transposition sort chain. The parity tick PERSISTS across frames (harness policy —
    // a new frame continues the transposition sequence where the last one stopped, which is what
    // makes the sort converge "bit by bit").
    sortPass(opts: {passes: (f: GridFrame) => number; pass: (parity: 0 | 1, side: 'A' | 'B') => ComputeStep}): SortPassStage {
        let cur: 'A' | 'B' = 'A'
        let tick = 0
        return {
            emit(f, nodes) {
                const n = opts.passes(f)
                for (let i = 0; i < n; i++) {
                    nodes.push(opts.pass((tick % 2) as 0 | 1, cur))
                    cur = cur === 'A' ? 'B' : 'A'
                    tick++
                }
            },
            side: () => cur,
        }
    },
} as const

// ── Shared host parts ──────────────────────────────────────────────────────────────────────────

/**
 * The canvas size in pixels, kept current across resizes, in the two forms a stage wants.
 *
 * `tracked()` is the size at the last resize, for a pass that works in the canvas's own pixels.
 * `safe(fp)` prefers the size this frame reports and never returns less than 1, for an aspect
 * ratio.
 *
 * @example
 * ```ts
 * const viewport = trackedViewport(params)
 * op.values('write', (f) => { const s = viewport.safe(f.frameParams); paramsU.write({aspect: s.width / s.height}) })
 * ```
 * @see gridSim
 */
export function trackedViewport(params: GpuFragmentParams): {
    tracked(): {width: number; height: number}
    safe(fp: GridFrameParams): {width: number; height: number}
} {
    let curW = Math.max(1, Math.round(params.dimensions.width))
    let curH = Math.max(1, Math.round(params.dimensions.height))
    params.onResize(({width, height}) => {
        curW = Math.max(1, Math.round(width))
        curH = Math.max(1, Math.round(height))
    })
    return {
        tracked: () => ({width: curW, height: curH}),
        safe: (fp) => ({
            width: Math.max(1, fp.dimensions?.width ?? curW),
            height: Math.max(1, fp.dimensions?.height ?? curH),
        }),
    }
}

// ── CPU grid sims (ChromaFlow, PixelThrow) ─────────────────────────────────────────────────────
//
// These fields are advanced by host JS loops (the loop arithmetic is the shader's own — relocated,
// never re-derived) and uploaded as an rgba16float data texture the fragment samples. The program
// runner gives the loops the same visible named-stage structure the compute sims have.

/** One frame of a JavaScript-stepped grid: the clamped delta, the clock and the pointer in uv. */
export interface HostFrame {
    /** Seconds since the last frame, never more than `clampDt`. */
    // Wall-clock delta (these sims ignore the renderer's deltaTime).
    dt: number
    now: number
    pointer: {x: number; y: number}
    frameParams: unknown
}

/** One named JavaScript step. Return `'skip'` to end the frame early. */
export interface HostStep {
    name: string
    run(f: HostFrame): void | 'skip'
}

/**
 * One named step of a JavaScript-stepped grid. Return `'skip'` to end the frame here, which is
 * how an idle gate is written.
 *
 * @example
 * ```ts
 * hostStep('dtGuard', (f) => (f.dt <= 0 ? 'skip' : undefined))
 * ```
 * @see hostGridProgram
 */
export const hostStep = (name: string, run: (f: HostFrame) => void | 'skip'): HostStep => ({name, run})

/**
 * A frame program for a small grid stepped in plain JavaScript: the steps run in order every
 * frame until one returns `'skip'`. Hand the result to `onBeforeRender`.
 *
 * Use it when the field is small (a 32×32 flow grid) and the arithmetic reads better as loops.
 * Publish the cells with `hostFieldTexture` in the last step.
 *
 * @example
 * ```ts
 * onBeforeRender(hostGridProgram({clampDt: 0.016, steps: [hostStep('advect', (f) => advect(f.dt)), hostStep('publish', () => { pack(field.texData); field.upload() })]}))
 * ```
 * @see hostStep, hostFieldTexture, gridSim
 */
// Wall-clock dt (clamped), then each named step in order until one skips.
export function hostGridProgram(opts: {clampDt: number; steps: HostStep[]}): (fp: unknown) => void {
    let lastTime = Date.now()
    return (fp) => {
        const {pointer} = fp as {pointer: {x: number; y: number}}
        const now = Date.now()
        const dt = Math.min((now - lastTime) / 1000, opts.clampDt)
        lastTime = now
        const f: HostFrame = {dt, now, pointer, frameParams: fp}
        for (const s of opts.steps) if (s.run(f) === 'skip') return
    }
}

/**
 * The texture a JavaScript-stepped grid is read through. Pack the cells into `texData` (four
 * half-floats per cell, row by row), call `upload()`, and sample `texture` in the fragment.
 *
 * `size` is the grid's side in cells. The texture filters smoothly, so a coarse grid reads as
 * a continuous field.
 *
 * @example
 * ```ts
 * const field = hostFieldTexture(params, {size: 32, label: 'flow'})
 * ```
 * @tip Values go in as half-floats: convert each number with `toHalfFloat` before writing it into `texData`.
 * @see hostGridProgram
 */
// An rgba16float data texture (filterable — a filtering sampler rejects r32float, so cells are
// half-encoded before upload) registered as a media texture the fragment samples. `texData` is
// the upload staging buffer the publish step packs into.
export function hostFieldTexture(
    params: GpuFragmentParams,
    opts: {size: number; label: string},
): {texData: Uint16Array; texture: ReturnType<GpuFragmentParams['registerMediaTexture']>; upload(): void} {
    const texData = new Uint16Array(opts.size * opts.size * 4)
    const tex = params.createDataTexture({width: opts.size, height: opts.size, format: 'rgba16float', data: texData, label: opts.label})
    params.onCleanup(() => tex.destroy())
    const texture = params.registerMediaTexture(() => tex.texture)
    return {texData, texture, upload: () => tex.write(texData)}
}
