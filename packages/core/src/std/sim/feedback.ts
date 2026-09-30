/**
 * std/sim/feedback — effects that remember the last frame.
 *
 * A feedback simulation keeps a picture of its own previous output and advances it by one
 * step of your rule each frame: read last frame's state, apply the rule, write the new state.
 * Temporal echoes, codec smears and decaying trails all live here. You bring one GPU step
 * function and the per-frame values it reads. The engine keeps the two state textures, swaps
 * them, keeps a display copy the layer samples, binds the child's picture once it exists, and
 * runs a speed-scaled clock.
 *
 * Spread `feedbackSim(...)` into a definition and sample `computeOutputs.display` in its
 * fragment. If the effect only needs a wave field or a displacement, declare it with
 * `simulate.grid` instead and skip the GPU code entirely.
 */
// Maintainer notes. The mechanics (ping-pong pair, display copy, late-bound child RTT, clamped
// speed-scaled clock) live in `scaffolds/feedbackSim` and are NOT re-implemented here — this
// noun owns the frame program every consumer (DataMosh, TimeTrail) hand-wrote around them:
//
//   ...feedbackSim((params, root) => ({
//       size: STATE_RES, format: 'rgba16float', speedProp: 'speed',
//       paramsSchema: MoshParams,
//       bindGroups: (ctx, read, write) => root.createBindGroup(layout, {...}),
//       step: moshKernel,                    // the compile-time-selected state-advance kernel
//       skip?: (t) => boolean,               // idle-gate policy as data
//       values: (t) => ({time: t.localTime, dt: t.dt, ...}),   // the one runtime surface
//   }))
//
// Frame-program order (identical to every hand-written consumer): scaffold tick (clock + swap
// bookkeeping) → skip gate → params write → one step dispatch on this orientation's bind groups.
//
// Compile-time vs runtime (C5): everything in the config is STRUCTURAL (kernel variant, layout
// wiring, resolution — recompose to change; the build callback re-runs). `values` is the only
// runtime surface, exactly as before. The build callback runs once per component instance, so
// anything created inside it (samplers, uniforms, compile-time kernel selection) is
// per-instance state — never hoist it to module scope.
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'
import {createFeedbackTrailSim, type SimBindContext, type SimSize, type SimSlot, type SimTick} from '../../gpu/scaffolds/feedbackSim'
import {createFluidKernelPass} from '../../gpu/scaffolds/fluids'

/** The GPU device handle the build callback receives, for creating the resources your step reads. */
// The TypeGPU root, via the contract (direct `typegpu` imports are restricted to src/gpu/).
export type FeedbackRoot = NonNullable<GpuFragmentParams['gpu']>['root']

/** A step function run once per state cell, with the cell's column and row. */
// A 2D-dispatched TGSL kernel over the state grid.
export type FeedbackKernel = (cx: number, cy: number) => void

/**
 * What you get when wiring your layout: the child's picture, the display texture, any extra
 * textures you declared, and the buffer holding your per-frame values.
 */
export type FeedbackBindContext = SimBindContext & {paramsBuffer: unknown}

/**
 * What a feedback simulation declares: the state's size and format, one step function, the
 * wiring that hands it the read and write sides, and the values it reads each frame.
 */
export interface FeedbackSimConfig<TParams, TGroups> {
    /** Cells per side of the square state texture. 512–768 reads sharp on most canvases. */
    // State + display resolution per axis (square — both current consumers' shape).
    size: number
    /** Storage precision. `rgba16float` is enough for color; `rgba32float` for exact positions. */
    format: 'rgba16float' | 'rgba32float'
    /** Extra textures that keep a previous and a current side, swapped together with the state. */
    // TimeTrail's prev-live copy. A `[1, 1]` stub fills a slot a disabled kernel variant never touches.
    extraSlots?: Record<string, SimSize>
    /** Extra textures with no previous side. */
    textures?: Record<string, SimSize>
    /** A prop that scales the clock. 0 freezes the simulation, 2 runs it twice as fast. */
    speedProp?: string
    /** Longest frame step accepted, in seconds. Default 0.05. Keeps a tab switch from jumping the state. */
    maxDeltaTime?: number
    /** The schema of the per-frame values your step reads. The engine owns the buffer. */
    paramsSchema: unknown
    /**
     * Wire your layout for one direction of the swap: `read` is last frame's state, `write` is
     * this frame's. Called twice, once per direction, after the child's picture exists.
     */
    bindGroups: (ctx: FeedbackBindContext, read: SimSlot, write: SimSlot) => TGroups
    /** Wire resources that do not depend on the swap direction. Called once. */
    staticGroups?: (ctx: FeedbackBindContext) => unknown
    /**
     * The rule: one step function that reads `read`, writes `write` and the display copy.
     * Pick the variant inside the build callback when a compile-time prop changes it.
     */
    // ONE kernel per step, compile-time variant selected in the build callback. Splitting a step
    // into several kernels is a different simulation (and different WGSL); a shader that needs a
    // multi-dispatch step should not ride this noun's default frame.
    step: FeedbackKernel
    /** Return true to skip this frame. The last state stays on screen and nothing swaps. */
    skip?: (t: SimTick<TGroups, unknown>) => boolean
    /** The values your step reads this frame, built from the clock and your props. */
    // The per-frame params derivation — the one runtime surface. Written before the dispatch.
    values: (t: SimTick<TGroups, unknown>) => TParams
    /** Extra outputs for the fragment beyond `childTexture` and `display`. */
    outputs?: Record<string, unknown>
}

/**
 * A simulation that advances a picture of its own last frame by one step of your rule, every
 * frame. Spread it into a definition next to a `gpu.fragment` that samples `display`.
 *
 * The build callback runs once per instance of the layer, so anything you create inside it
 * (resources, a kernel variant picked from a compile-time prop) belongs to that instance. The
 * simulation needs a child. Without one, or without a GPU, nothing runs and the fragment should
 * fall back to the child.
 *
 * @example
 * ```ts
 * ...feedbackSim((params, root) => ({size: 768, format: 'rgba16float', speedProp: 'speed', paramsSchema: decoder.Params, bindGroups: ({childTexture, display, paramsBuffer}, read, write) => root.createBindGroup(decoder.layout, {src: childTexture, prev: read.state, next: write.state, display, params: paramsBuffer} as never), step: decoder.kernel, values: ({dt, localTime}) => ({time: localTime, dt, intensity: params.getCpuValue('intensity') as number})}))
 * ```
 * @tip Read time from `localTime`, not the layer's clock. It stops when `speedProp` is 0 and every frame's step matches it.
 * @see fluidSim, gridSim, agentSim, simulate
 */
// Requires a child (these are all child-consuming echo/decoder effects): bails to `null`
// (fragment passthrough) without one, without a device, or when the scaffold cannot allocate.
export function feedbackSim<TParams, TGroups = unknown>(
    build: (params: GpuFragmentParams, root: FeedbackRoot) => FeedbackSimConfig<TParams, TGroups>,
): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {childNode, gpu} = params
            if (!childNode) return null
            const root = gpu?.root
            if (!root) return null // GPU-free resolve/tests: fragment falls back to passthrough.

            const cfg = build(params, root)
            const paramsU = root.createUniform(cfg.paramsSchema as never) as {buffer: unknown; write: (v: never) => void}

            const sim = createFeedbackTrailSim<TGroups, unknown>(params, {
                size: cfg.size,
                format: cfg.format,
                extraSlots: cfg.extraSlots,
                textures: cfg.textures,
                speedProp: cfg.speedProp,
                maxDeltaTime: cfg.maxDeltaTime,
                bindGroups: (ctx, read, write) => cfg.bindGroups({...ctx, paramsBuffer: paramsU.buffer}, read, write),
                staticGroups: cfg.staticGroups
                    ? (ctx) => cfg.staticGroups!({...ctx, paramsBuffer: paramsU.buffer})
                    : undefined,
            })
            if (!sim) return null

            // The step pipeline binds its groups per frame (`.with(groups)`), so it is built bare.
            const pipeline = createFluidKernelPass(root, cfg.step, {n: cfg.size, bindGroup: undefined})

            return {
                outputs: {
                    childTexture: sim.childTexture, display: sim.display, ...(cfg.outputs ?? {}),
                } as NonNullable<ReturnType<GpuComputeNode>>['outputs'],
                bindInputs: sim.bindInputs,
                getComputeNodes: (frameParams: unknown): ComputeStep[] | null =>
                    sim.tick(frameParams, (t) => {
                        if (cfg.skip?.(t)) return null
                        paramsU.write(cfg.values(t) as never)
                        return [pipeline.with(t.groups as never)]
                    }),
            }
        },
    }
}
