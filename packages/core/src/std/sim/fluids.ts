/**
 * std/sim/fluids — smoke, ink and fog that flow.
 *
 * A fluid simulation is a square grid of velocity plus a dye (density, age or color) that the
 * engine advances with a full fluid solve every frame. You bring the parts that make the look:
 * what injects dye and momentum (`splat` at a fixed source, `cursorRibbon` along a stroke),
 * any force applied before each solve (`ambientForce`), any correction after it
 * (`restoreToward`), how the field starts (`seededFieldInit`), and the per-frame values your
 * passes read. `fluidSim` owns the rest: the state buffers, the output texture, the solve
 * chain, the pointer tracking, the clock and the idle gate that sleeps a settled field.
 *
 * Spread `fluidSim(...)` into a definition and sample its output texture in the fragment. When
 * the effect only needs a wave field or a displacement, declare it with `simulate.grid` instead.
 */
// Maintainer notes. A fluid shader is a Stable-Fluids solve (scaffolds/fluids owns the kernels)
// wrapped in a frame program that was copy-pasted across every consumer: allocate the six state
// buffers, publish an output texture, write the per-frame params, run emission, run the solve
// chain, run the output pass. `fluidSim` owns that program ONCE; the shader keeps what is
// genuinely its own — its params struct, its emission/force/restore kernels (the constants ARE
// the look), and its `values` derivation — and declares them as named parts, so the definition
// reads as the pipeline it is:
//
//   ...fluidSim((params, root) => ({
//       resolution: N, layout, outputLayout, paramsSchema,
//       solver: {kernels, jacobiIters},
//       output: {kernel: outputKernel, key: 'smokeTexture'},
//       init: seededFieldInit({...}),            // Fog: fBm seed + silent warm-up
//       inject: [splat({kernel: splatKernel})],  // or cursorRibbon({...}) for stroke emitters
//       solve: [ambientForce(forceKernel), restoreToward(colorRestoreKernel)],
//       values: (f) => ({dt: f.dt, ...}),
//   }))
//
// Frame-program order (identical to every hand-written consumer):
//   [container pre-march] → [init + warm-up when stale] → emitter ticks → idle-gate skip →
//   params write → [container mask pass] → emitter dispatches → pre-solve stages →
//   curl…advect chain → post-solve stages → output.
//
// Compile-time vs runtime (C5): everything in the config is STRUCTURAL (kernels, layouts,
// resolution — recompose to change). The `values` object is the only runtime surface, exactly
// as before. The build callback runs once per component instance, so parts created inside it
// (trackers, color cycles, init latches) are per-instance state — never hoist a part to module
// scope.
import {d} from '../../gpu/kit/index'
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep, KitComputePipeline} from '../../gpu/compute'
import {createStateBuffer} from '../../gpu/compute'
import {
    createFluidKernelPass, createStableFluidsPasses, type StableFluidsKernels,
} from '../../gpu/scaffolds/fluids'
import {
    createIdleGate, createPointerVelocityTracker, pathStampRibbon,
    type PointerFrame, type PointerLike, type PointerVelocityTrackerOptions,
} from '../../gpu/kit/host/pointer'
/** The GPU device handle the build callback receives, for creating the resources your passes use. */
// The TypeGPU root, via the contract (direct `typegpu` imports are restricted to src/gpu/).
export type FluidRoot = NonNullable<GpuFragmentParams['gpu']>['root']

/** A pass function run once per grid cell, with the cell's column and row. */
// A 2D-dispatched TGSL kernel over the N×N grid.
export type FluidKernel = (cx: number, cy: number) => void

/** What the renderer reports each frame: the pointer in uv, the frame delta in seconds, the canvas size in pixels. */
export interface FluidFrameParams {
    pointer?: PointerLike
    deltaTime?: number
    dimensions?: {width: number; height: number}
}

/** One frame, as every part and `values` sees it: the clamped delta, the clocks, the pointer and your props. */
export interface FluidFrame {
    /** Seconds since the last frame, never more than `maxDeltaTime` (0.033 by default). */
    dt: number
    /** Wall-clock milliseconds this frame. */
    // `Date.now()` (wall-clock bookkeeping; the idle gates run on simulated `dt`).
    now: number
    /** The simulation's own clock: the sum of every `dt` so far. */
    // An init part may reset it (Fog's warm-up).
    readonly elapsed: number
    setElapsed(t: number): void
    /** The pointer this frame, when the config declared `pointer`. Null otherwise. */
    // The noun-owned tracker's sample. Ribbons own their own.
    ptr: PointerFrame | null
    /** A numeric prop's current value, or `fallback` when it is not a number this frame. */
    num(key: string, fallback: number): number
    /** Any prop's current value. */
    getCpuValue(key: string): unknown
    frameParams: FluidFrameParams
    /** Write the per-frame values directly, for a part that rewrites them between passes. */
    // Per-stamp rewrites inside ribbon thunks.
    writeParams(v: unknown): void
}

/** What a part receives when it builds its passes: the device, the grid size, the buffers and a pass builder. */
export interface FluidSetupCtx {
    params: GpuFragmentParams
    root: FluidRoot
    /** Cells per side of the grid. */
    n: number
    buffers: {
        velA: unknown; velB: unknown; dyeA: unknown; dyeB: unknown
        pressure: unknown; divergence: unknown; maskBuf?: unknown
    }
    /** The buffer holding the per-frame values, for a part with its own resources. */
    paramsBuffer: unknown
    /** The resource group every solver pass runs on. */
    fluidBindGroup: unknown
    /** Turn a pass function into a runnable pass, on the fluid's resources unless another group is given. */
    pass(kernel: FluidKernel, bindGroup?: unknown): KitComputePipeline
}

/** Something that injects into the field each frame. Build one with `splat` or `cursorRibbon`. */
export interface FluidEmitter<TParams = unknown> {
    setup?(ctx: FluidSetupCtx): void
    /** Bookkeeping before the idle gate: sample the pointer, drift a color cycle, mark activity. */
    tick?(f: FluidFrame): void
    /** Return true to skip the whole frame. The settled field stays on screen. */
    skip?(f: FluidFrame): boolean
    /** Add this frame's injection passes. They run before the solve. */
    emit?(f: FluidFrame, nodes: ComputeStep[], values: TParams): void
}

/** A pass that runs inside every solve, before it (`pre`) or after it (`post`). Build one with `ambientForce` or `restoreToward`. */
// Runs inside every solve, including an init part's warm-up steps.
export interface FluidSolveStage {
    stage: 'pre' | 'post'
    kernel: FluidKernel
}

/**
 * A force applied to the whole field before every solve: a turbulence field, a steady wind.
 *
 * @example
 * ```ts
 * solve: [ambientForce(turbulenceKernel)]
 * ```
 * @see restoreToward, fluidSim
 */
// Fog's turbulence field — dispatched before every solve chain.
export const ambientForce = (kernel: FluidKernel): FluidSolveStage => ({stage: 'pre', kernel})

/**
 * A correction applied after every solve: re-seed a color that the solve slowly blurs away.
 *
 * @example
 * ```ts
 * solve: [restoreToward(colorRestoreKernel)]
 * ```
 * @see ambientForce, fluidSim
 */
// Fog's color re-seed against numerical diffusion — after every solve chain.
export const restoreToward = (kernel: FluidKernel): FluidSolveStage => ({stage: 'post', kernel})

/** What an init part appends with: the frame's pass list, a queued values write, and a full solve. */
// `writeParams` returns a QUEUED thunk (device.queue order).
export interface FluidInitIo<TParams> {
    nodes: ComputeStep[]
    writeParams(v: TParams): ComputeStep
    solve(nodes: ComputeStep[], jacobiIters: number): void
}

/** How the field starts, and when it must start over. Build one with `seededFieldInit`. */
export interface FluidInitPart<TParams = unknown> {
    setup?(ctx: FluidSetupCtx): void
    /** Return true when the field must be (re)seeded this frame. */
    stale(f: FluidFrame): boolean
    run(f: FluidFrame, io: FluidInitIo<TParams>): void
}

/** What a container (a shape the fluid is confined to) contributes each frame. */
// SmokeFill's confining shape.
export interface FluidContainerParts {
    /** The pass that writes the inside/outside mask, run after the values write and before injection. */
    maskPass: KitComputePipeline
    /** Extra outputs for the fragment. */
    // The volumetric field texture.
    outputs?: Record<string, unknown>
    /** Passes that run first each frame, even on a frame too short to solve. */
    // The volumetric field pre-march. They run FIRST, and they are what a sub-millisecond frame
    // still returns — a pending march is never dropped.
    preFrame?(frameParams: unknown): ComputeStep[] | null
}

/** What a fluid simulation declares: its grid, its solver, its output, its parts and its per-frame values. */
export interface FluidSimConfig<TParams> {
    /** Cells per side of the grid. 256 is the usual choice. */
    // Structural — the kernels are baked against it.
    resolution: number
    /** The layout naming the fluid's buffers. It must use the solver's entry names. */
    // The scaffolds/fluids entry-name contract: velA, velB, dyeA, dyeB, pressure, divergence, params (+ maskBuf).
    layout: unknown
    /** The layout of the output pass: the dye buffer read-only plus the output texture. */
    // `dyeA` read-only + `outTex`.
    outputLayout: unknown
    /** The schema of the per-frame values your passes read. The engine owns the buffer. */
    paramsSchema: unknown
    /** The solver's passes and how many pressure iterations to run per solve. 10 is the norm. */
    solver: {kernels: StableFluidsKernels; jacobiIters: number}
    /** The pass that turns dye into the output texture, and the name the fragment reads it under. */
    output: {kernel: FluidKernel; key: string}
    /** How the values reach the GPU: written at once (default), or queued in order with the passes. Use `'thunk'` with `seededFieldInit`, whose warm-up queues its own writes. */
    write?: 'direct' | 'thunk'
    /** Track the pointer for a cursor force. Stroke emitters built with `cursorRibbon` track their own. */
    pointer?: PointerVelocityTrackerOptions
    /** A shape the fluid is confined to. */
    // Allocates `maskBuf`, dispatches its mask pass every frame.
    container?: {setup(ctx: FluidSetupCtx): FluidContainerParts}
    /** How the field starts. Omit to start empty. */
    init?: FluidInitPart<TParams>
    /** What injects dye and momentum each frame. */
    inject?: FluidEmitter<TParams>[]
    /** Passes run before and after every solve. */
    solve?: FluidSolveStage[]
    /** The values your passes read this frame, built from the clock, the pointer and your props. */
    // The per-frame params derivation — the one runtime surface.
    values(f: FluidFrame): TParams
    /** Longest frame step accepted, in seconds. Default 0.033. */
    maxDeltaTime?: number
}

const STATE_FORMAT = 'rgba16float' as const

/**
 * A fluid the engine solves every frame. Spread it into a definition next to a `gpu.fragment`
 * that samples the output texture under `output.key`.
 *
 * The build callback runs once per instance of the layer, with the GPU device, and returns the
 * grid size, the solver, the output pass, the parts and the `values` function. Everything
 * created inside it belongs to that instance. Without a GPU nothing runs and the fragment
 * should fall back to transparent.
 *
 * @example
 * ```ts
 * ...fluidSim(() => ({resolution: 256, layout: fluidLayout, outputLayout, paramsSchema: FluidParams, solver: {kernels: solverKernels, jacobiIters: 10}, output: {kernel: outputKernel, key: 'smokeTexture'}, pointer: {minDrag: 0.0005}, inject: [splat({kernel: splatKernel})], values: (f) => ({dt: f.dt, emitX: 0.5 * 256, emitY: 256, emitRad: f.num('emitRadius', 0.06) * 256, dyeFade: f.num('dissipation', 1), velFade: 0.2, curlStrength: f.num('detail', 30), colorDecay: 1})}))
 * ```
 * @tip Positions and radii in `values` are in grid cells, not uv: multiply a uv value by `resolution`.
 * @see splat, cursorRibbon, seededFieldInit, ambientForce, restoreToward, simulate
 */
// The build callback runs at COMPUTE-NODE creation (per instance, after the no-device bail), so
// parts and closures created inside it are instance state.
export function fluidSim<TParams>(
    build: (params: GpuFragmentParams, root: FluidRoot) => FluidSimConfig<TParams>,
): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {gpu, getCpuValue, registerComputeTexture, onCleanup} = params
            const root = gpu?.root
            if (!root) return null // GPU-free / no device: the fragment falls back to transparent.

            const cfg = build(params, root)
            const n = cfg.resolution
            const count = n * n

            const velA = createStateBuffer(root, d.vec4f, count)
            const velB = createStateBuffer(root, d.vec4f, count)
            const dyeA = createStateBuffer(root, d.vec4f, count)
            const dyeB = createStateBuffer(root, d.vec4f, count)
            const pressure = createStateBuffer(root, d.f32, count)
            const divergence = createStateBuffer(root, d.f32, count)
            const maskBuf = cfg.container ? createStateBuffer(root, d.vec4f, count) : undefined

            const outTex = root.createTexture({size: [n, n], format: STATE_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => outTex.destroy())
            const outputTexture = registerComputeTexture(outTex)

            const paramsU = root.createUniform(cfg.paramsSchema as never) as {buffer: unknown; write: (v: never) => void}
            const entries: Record<string, unknown> = {velA, velB, dyeA, dyeB, pressure, divergence, params: paramsU.buffer}
            if (maskBuf) entries.maskBuf = maskBuf
            const fluidBg = root.createBindGroup(cfg.layout as never, entries as never)
            const outputBg = root.createBindGroup(cfg.outputLayout as never, {dyeA, outTex} as never)

            const ctx: FluidSetupCtx = {
                params, root, n,
                buffers: {velA, velB, dyeA, dyeB, pressure, divergence, maskBuf},
                paramsBuffer: paramsU.buffer,
                fluidBindGroup: fluidBg,
                pass: (kernel, bindGroup = fluidBg) => createFluidKernelPass(root, kernel, {n, bindGroup}),
            }

            const containerParts = cfg.container?.setup(ctx)
            cfg.init?.setup?.(ctx)
            const emitters = cfg.inject ?? []
            for (const e of emitters) e.setup?.(ctx)
            const solvePre = (cfg.solve ?? []).filter((s) => s.stage === 'pre').map((s) => ctx.pass(s.kernel))
            const solvePost = (cfg.solve ?? []).filter((s) => s.stage === 'post').map((s) => ctx.pass(s.kernel))
            const solver = createStableFluidsPasses(root, cfg.solver.kernels, {n, bindGroup: fluidBg})
            const outputPass = ctx.pass(cfg.output.kernel, outputBg)

            const solveInto = (nodes: ComputeStep[], jacobiIters: number) => {
                nodes.push(...solvePre, ...solver.solveSteps({jacobiIters}), ...solvePost)
            }

            const tracker = cfg.pointer ? createPointerVelocityTracker(cfg.pointer) : null
            const maxDt = cfg.maxDeltaTime ?? 0.033
            let lastTime = Date.now()
            let elapsed = 0
            const num = (key: string, fallback: number): number => {
                const v = getCpuValue(key)
                return typeof v === 'number' ? v : fallback
            }
            const writeParams = (v: unknown) => paramsU.write(v as never)

            return {
                outputs: {[cfg.output.key]: outputTexture, ...(containerParts?.outputs ?? {})} as NonNullable<ReturnType<GpuComputeNode>>['outputs'],
                getComputeNodes: (frameParams: unknown): ComputeStep[] | null => {
                    const fp = frameParams as FluidFrameParams
                    const preNodes = containerParts?.preFrame ? containerParts.preFrame(frameParams) : null
                    const now = Date.now()
                    const dt = Math.min(fp.deltaTime ?? (now - lastTime) / 1000, maxDt)
                    lastTime = now
                    if (dt < 0.001) return preNodes
                    elapsed += dt

                    const f: FluidFrame = {
                        dt, now, frameParams: fp, num, getCpuValue, writeParams,
                        ptr: tracker ? tracker.update(fp.pointer, dt) : null,
                        get elapsed() { return elapsed },
                        setElapsed(t) { elapsed = t },
                    }

                    const nodes: ComputeStep[] = preNodes ? [...preNodes] : []
                    let ranInit = false
                    if (cfg.init?.stale(f)) {
                        ranInit = true
                        cfg.init.run(f, {
                            nodes,
                            writeParams: (v) => () => paramsU.write(v as never),
                            solve: solveInto,
                        })
                    }
                    for (const e of emitters) e.tick?.(f)
                    // An idle-gated frame must still submit a freshly-run init chain: the init
                    // part latched itself as seeded, so dropping the nodes here would leave the
                    // field permanently unseeded (an idle gate can skip the very first frame).
                    for (const e of emitters) if (e.skip?.(f)) return ranInit ? nodes : null

                    const v = cfg.values(f)
                    if ((cfg.write ?? 'direct') === 'thunk') nodes.push(() => paramsU.write(v as never))
                    else paramsU.write(v as never)
                    if (containerParts) nodes.push(containerParts.maskPass)
                    for (const e of emitters) e.emit?.(f, nodes, v)
                    solveInto(nodes, cfg.solver.jacobiIters)
                    nodes.push(outputPass)
                    return nodes
                },
            }
        },
    }
}

// ── Emitter parts ──────────────────────────────────────────────────────────────────────────────

/**
 * An emitter at a fixed source: one pass per frame that adds dye and momentum where your pass
 * function says (a cone from a point, a line along an edge).
 *
 * Give it the pass function, or a `setup` that builds the pass on resources of its own.
 *
 * @example
 * ```ts
 * inject: [splat({kernel: splatKernel})]
 * ```
 * @tip The source position and radius come from `values`, in grid cells.
 * @see cursorRibbon, fluidSim
 */
// Smoke's cone, SmokeFill's confined cone: one dispatch per frame.
export function splat<TParams = unknown>(
    opts: {kernel: FluidKernel} | {setup: (ctx: FluidSetupCtx) => KitComputePipeline},
): FluidEmitter<TParams> {
    let pass: KitComputePipeline | null = null
    return {
        setup(ctx) {
            pass = 'kernel' in opts ? ctx.pass(opts.kernel) : opts.setup(ctx)
        },
        emit(_f, nodes) {
            if (pass) nodes.push(pass)
        },
    }
}

/** This frame's ribbon of stamps along the stroke: spacing, cap, and how each stamp is written. */
export interface CursorRibbonSpec<TPrepared> {
    /** Distance between stamps, in uv. Half the brush radius reads continuous. */
    stepSize: number
    /** Most stamps in one frame, so a fast flick cannot run away. */
    maxSteps: number
    /** A per-stamp payload prepared in stamp order, with `t` from 0 to 1 along the stroke. */
    // Advanced at BUILD time in stamp order (InkFlow's color cycle).
    prepare?: (t: number) => TPrepared
    /** Write the values for one stamp at `posX, posY` (grid cells) right before its pass runs. */
    // Runs in a thunk right before that stamp's dispatch.
    write: (posX: number, posY: number, t: number, prepared: TPrepared) => void
}

/**
 * An emitter that paints along the pointer's stroke: a ribbon of stamps between last frame's
 * position and this one, so a fast flick leaves a continuous line, not dots.
 *
 * It tracks the pointer itself, marks the frame active when `activeWhen` says so, and sleeps
 * the simulation `fadeSeconds` after the last activity. `ribbon` describes this frame's stamps.
 * Read the pointer back with `.ptr()` and `.active()` inside `values`.
 *
 * @example
 * ```ts
 * const strokes = cursorRibbon<Params>({activeWhen: (ptr) => ptr.smoothSpeed > 0.005, fadeSeconds: (f) => decayFadeSeconds(255, f.num('dissipation', 0.4)), pass: {kernel: splatKernel}, ribbon: (f, ptr, values) => ({stepSize: Math.max(0.004, f.num('emitRadius', 0.07) * 0.5), maxSteps: 64, write: (emitX, emitY) => f.writeParams({...values, emitX, emitY})})})
 * ```
 * @tip Raise the tracker's `teleportGuard` when fast flicks should paint: the default treats a quarter-canvas jump as the cursor re-entering and emits nothing.
 * @see splat, fluidSim
 */
// A pointer-velocity tracker (teleport/drag policy as data), an idle gate that freezes the sim
// once the field has faded after the last stroke, and a per-frame ribbon of stamps interpolated
// along the drag path (`pathStampRibbon` — no dotted gaps on a fast flick). The part owns its
// tracker; SmokeFlow's `values` derives the cursor fields from `.ptr()` / `.active()`.
export function cursorRibbon<TParams = unknown, TPrepared = undefined>(opts: {
    /** Pointer tracking: smoothing, the teleport distance, the minimum drag. */
    tracker?: PointerVelocityTrackerOptions
    /** Whether the stroke emits this frame. A live frame keeps the simulation awake. */
    activeWhen: (ptr: PointerFrame, f: FluidFrame) => boolean
    /** Seconds after the last activity before the simulation may sleep. */
    fadeSeconds: (f: FluidFrame) => number
    /** Extra bookkeeping per frame before the idle gate. */
    // InkFlow's color-cycle time drift.
    onFrame?: (f: FluidFrame, ptr: PointerFrame) => void
    /** The stamp pass: a pass function, or a `setup` that builds one on resources of its own. */
    pass: {kernel: FluidKernel} | {setup: (ctx: FluidSetupCtx) => KitComputePipeline}
    /** This frame's stamps. */
    ribbon: (f: FluidFrame, ptr: PointerFrame, values: TParams) => CursorRibbonSpec<TPrepared>
}): FluidEmitter<TParams> & {ptr(): PointerFrame | null; active(): boolean} {
    const tracker = createPointerVelocityTracker(opts.tracker)
    const idle = createIdleGate()
    let pass: KitComputePipeline | null = null
    let ptr: PointerFrame | null = null
    let scale = 1
    let isActive = false
    return {
        setup(ctx) {
            scale = ctx.n
            pass = 'kernel' in opts.pass ? ctx.pass(opts.pass.kernel) : opts.pass.setup(ctx)
        },
        tick(f) {
            ptr = tracker.update(f.frameParams.pointer, f.dt)
            opts.onFrame?.(f, ptr)
            isActive = opts.activeWhen(ptr, f)
            if (isActive) idle.markActive()
        },
        skip(f) {
            const skip = idle.shouldSkip(opts.fadeSeconds(f))
            // Only frames that actually step the field count toward its decay budget.
            if (!skip) idle.tickFrame(f.dt)
            return skip
        },
        emit(f, nodes, values) {
            if (!isActive || !ptr || !pass) return
            const spec = opts.ribbon(f, ptr, values)
            pathStampRibbon(nodes, {
                fromX: ptr.prevX, fromY: ptr.prevY, dx: ptr.dx, dy: ptr.dy,
                dragDist: ptr.dragDist, stepSize: spec.stepSize, maxSteps: spec.maxSteps,
                scale, prepare: spec.prepare, write: spec.write, pass,
            })
        },
        ptr: () => ptr,
        active: () => isActive,
    }
}

// ── Init parts ─────────────────────────────────────────────────────────────────────────────────

/**
 * A field that starts full: seed it from a pass function, then run the solve silently
 * `warm.steps` times so the first visible frame already flows.
 *
 * Re-seeds whenever `seed` returns a new value. Each warm step advances a private clock by
 * `warm.dt` seconds and writes `warm.values(t)`. When it finishes, the simulation's clock
 * continues from where the warm-up ended.
 *
 * @example
 * ```ts
 * init: seededFieldInit({kernel: initKernel, seed: (f) => f.num('seed', 0), warm: {steps: 50, jacobiIters: 10, dt: 0.033, startTime: () => Math.random() * 100, values: (t, f) => ({...baseValues(f), time: t})}})
 * ```
 * @tip A random `startTime` makes two instances of the same layer drift differently.
 * @see fluidSim, ambientForce
 */
// Fog: when the seed key changes (or on first frame), dispatch the init kernel and run the whole
// solve `warm.steps` times with time-advancing param thunks, then hand the warm clock to the
// frame clock so ambient forces continue seamlessly.
export function seededFieldInit<TParams>(opts: {
    kernel: FluidKernel
    /** The value whose change forces a re-seed. */
    seed: (f: FluidFrame) => number
    warm: {
        /** How many silent solves to run. */
        steps: number
        jacobiIters: number
        /** Simulated seconds per warm step. */
        dt: number
        /** Where the warm clock starts. Randomize it so two instances differ. */
        startTime: () => number
        /** The full values for a warm step at simulated time `t`. `t = 0` is the seed write. */
        values: (t: number, f: FluidFrame) => TParams
    }
}): FluidInitPart<TParams> {
    let pass: KitComputePipeline | null = null
    let initialized = false
    let lastSeed = -1
    return {
        setup(ctx) {
            pass = ctx.pass(opts.kernel)
        },
        stale(f) {
            return !initialized || opts.seed(f) !== lastSeed
        },
        run(f, io) {
            initialized = true
            lastSeed = opts.seed(f)
            io.nodes.push(io.writeParams(opts.warm.values(0, f)))
            if (pass) io.nodes.push(pass)
            let warmTime = opts.warm.startTime()
            for (let w = 0; w < opts.warm.steps; w++) {
                const wt = warmTime
                warmTime += opts.warm.dt
                io.nodes.push(io.writeParams(opts.warm.values(wt, f)))
                io.solve(io.nodes, opts.warm.jacobiIters)
            }
            f.setElapsed(warmTime)
        },
    }
}
