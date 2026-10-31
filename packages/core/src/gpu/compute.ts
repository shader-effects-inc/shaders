/**
 * Compute pipeline creation + per-frame dispatch.
 *
 * `root.createGuardedComputePipeline` gives us a bounds-guarded, thread-indexed dispatch
 * ([256] for 1D, [16,16] for 2D) — the bounds guard is provided by the pipeline, so kernels
 * need no manual `If(cy.lessThan(ch))` guard. This module wraps that primitive into a small
 * dispatcher that runs an ordered list of steps (pipelines + inline thunks) and exposes the
 * ping-pong / storage-buffer helpers the compute shaders need.
 */
import tgpu, {
    type TgpuRoot,
    type TgpuBindGroup,
    type TgpuBuffer,
    type TgpuGuardedComputePipeline,
    type StorageFlag,
} from 'typegpu'
import * as d from 'typegpu/data'
import type {GpuPassTimer, PassTimestampWrites} from './gpuTimer'
import type {AnyWgslData, WgslArray} from 'typegpu/data'

// Silence "imported but only used for its side-effect of registering the plugin meta" —
// tgpu is re-exported so downstream gpu/ modules can reach the same instance if needed.
void tgpu

/**
 * A dispatchable compute pipeline. Thin wrapper over a `TgpuGuardedComputePipeline` that
 * remembers an optional pre-bound dispatch size so the frame dispatcher can run it with no
 * arguments (`dispatch()`), while still allowing an explicit `dispatchThreads(w, h?)`.
 */
export interface KitComputePipeline {
    /** The underlying guarded pipeline (escape hatch for `.withTimestampWrites`, etc.). */
    readonly guarded: TgpuGuardedComputePipeline<number[]>
    /** Bind a (new) bind group, returning a new handle that shares the pre-bound size. */
    with(bindGroup: TgpuBindGroup): KitComputePipeline
    /** Time the next dispatch (see gpuTimer.ts), returning a handle that shares the pre-bound size. */
    withTimestampWrites(writes: PassTimestampWrites): KitComputePipeline
    /** Dispatch using the pre-bound size. Throws if none was supplied at creation. */
    dispatch(): void
    /** Dispatch an explicit thread count per dimension (1D/2D/3D). */
    dispatchThreads(...threads: number[]): void
}

/**
 * A single step in a compute program. Either a pipeline (dispatched with its pre-bound size)
 * or a plain function (executed inline — typically a batch of uniform writes performed
 * *between* dispatches). See `createComputeDispatcher` for the ordering contract.
 */
export type ComputeStep = KitComputePipeline | (() => void)

export interface GuardedComputeOptions {
    /**
     * Pre-bound dispatch size (threads per dimension). When present, `dispatch()` works with
     * no arguments — the common case for fixed-resolution compute (e.g. blur's 1024×640).
     */
    size?: [number] | [number, number] | [number, number, number]
    /** Bind group to bind immediately. Equivalent to calling `.with(bindGroup)` after. */
    bindGroup?: TgpuBindGroup
}

function wrapGuarded(
    guarded: TgpuGuardedComputePipeline<number[]>,
    size?: readonly number[],
): KitComputePipeline {
    return {
        guarded,
        with(bindGroup: TgpuBindGroup) {
            return wrapGuarded(guarded.with(bindGroup), size)
        },
        withTimestampWrites(writes: PassTimestampWrites) {
            return wrapGuarded(guarded.withTimestampWrites(writes), size)
        },
        dispatch() {
            if (!size) {
                throw new Error(
                    '[gpu] KitComputePipeline.dispatch(): no pre-bound size — pass opts.size to createGuardedCompute or call dispatchThreads(...) directly',
                )
            }
            guarded.dispatchThreads(...size)
        },
        dispatchThreads(...threads: number[]) {
            guarded.dispatchThreads(...threads)
        },
    }
}

/**
 * Thin wrapper over `root.createGuardedComputePipeline`. The callback is authored by the
 * caller as a `'use gpu'` arrow (transpiled where it is written, not here) taking up to 3
 * thread coordinates. The bounds guard and workgroup sizing come from the guarded pipeline,
 * so kernels need no manual `If(cy.lessThan(ch))` guard.
 *
 * @example
 * const pass = createGuardedCompute(root, (cx, cy) => { 'use gpu'; kernel(cx, cy) },
 *     { size: [1024, 640], bindGroup: bg })
 * pass.dispatch() // dispatches 1024×640 guarded threads
 */
export function createGuardedCompute(
    root: TgpuRoot,
    callback: (...threads: number[]) => void,
    opts: GuardedComputeOptions = {},
): KitComputePipeline {
    let guarded = root.createGuardedComputePipeline(callback)
    if (opts.bindGroup) guarded = guarded.with(opts.bindGroup)
    return wrapGuarded(guarded, opts.size)
}

export interface ComputeDispatcher {
    /**
     * Run an ordered list of compute steps this frame. Pipelines are dispatched in array
     * order; function entries execute inline at their position.
     *
     * **Ordering contract:** guarded pipelines submit their work to `device.queue`
     * immediately on `dispatch()`, and thunk entries (uniform writes via `buffer.write` /
     * `.patch`) also hit `device.queue` immediately. Because `device.queue` preserves
     * submission order, every thunk boundary forms an implicit ordering barrier: writes in a
     * thunk are guaranteed visible to pipelines dispatched *after* it, and reflect state as
     * of pipelines dispatched *before* it. This ordering matters for loops like CursorTrail's
     * decay → (write stamp uniform → dispatch stamp) × N. Batching several dispatches into one
     * command encoder for throughput is a frame-loop concern (frame.ts may pass an encoder via
     * the guarded pipeline's `.with(encoder)`); this dispatcher intentionally keeps the simple,
     * order-preserving default.
     */
    dispatch(steps: ComputeStep[], timer?: GpuPassTimer): void
}

/**
 * Create a dispatcher bound to a root. See {@link ComputeDispatcher.dispatch} for ordering.
 */
export function createComputeDispatcher(root: TgpuRoot): ComputeDispatcher {
    void root // reserved for future encoder-batched dispatch; kept for API symmetry
    return {
        dispatch(steps: ComputeStep[], timer?: GpuPassTimer) {
            for (const step of steps) {
                if (typeof step === 'function') {
                    step()
                    continue
                }
                // Profiling: each pipeline dispatch gets its own timestamp pair. Work a thunk
                // dispatches by itself is not timed.
                const writes = timer && typeof step.withTimestampWrites === 'function' ? timer.next() : null
                ;(writes ? step.withTimestampWrites(writes) : step).dispatch()
            }
        },
    }
}

export interface PingPong<TBindGroup> {
    /** The bind group for the current orientation (read source A / write target B, or swapped). */
    readonly current: TBindGroup
    /** Flip read/write orientation for the next dispatch. */
    swap(): void
}

/**
 * @deprecated Superseded by {@link createPingPongPair}, which fits the shapes real feedback
 * simulations have. This one shipped with zero consumers for three reasons, all worth recording
 * because they are the design brief for the replacement:
 *
 * 1. **One bind-group family only.** A sim almost never has just one pass per orientation.
 *    CursorRipples propagates over the buffer pair AND runs a gradient pass that reads whichever
 *    buffer was just written; ReactionDiffusion reacts N times then runs an output copy. This
 *    helper can express exactly one `make(read, write)` family, so those shaders had to keep
 *    their own orientation flag anyway — at which point the helper adds nothing.
 * 2. **No orientation query.** `current` hands back a bind group and nothing else, so a pass
 *    that needs "the resource the current orientation reads/wrote" (every output/display copy)
 *    cannot be built from it.
 * 3. **Eager construction.** The bind groups are built at call time. Every child-consuming sim
 *    (TimeTrail, DataMosh, KeyFrames, ReactionDiffusion-with-child) can only build its groups
 *    inside `bindInputs`, after the pass manager allocates the child RTT.
 *
 * Kept (not deleted) so the deprecation is visible at the call site if anything external adopted it.
 */
export function createPingPong<TTexture, TBindGroup>(
    texA: TTexture,
    texB: TTexture,
    makeBindGroup: (readTex: TTexture, writeTex: TTexture) => TBindGroup,
): PingPong<TBindGroup> {
    const readAwriteB = makeBindGroup(texA, texB)
    const readBwriteA = makeBindGroup(texB, texA)
    let flipped = false
    return {
        get current() {
            return flipped ? readBwriteA : readAwriteB
        },
        swap() {
            flipped = !flipped
        },
    }
}

/** One read/write orientation of a ping-pong pair. 0 = read `a` / write `b`; 1 = the reverse. */
export type PingPongOrientation = 0 | 1

/**
 * A ping-pong pair of simulation resources plus the bind groups built over them.
 *
 * `TSlot` is whatever one side of the pair is: a single storage buffer, a single storage texture,
 * or a RECORD of several resources that must swap in lockstep (TimeTrail swaps its history state
 * and its previous-live-frame copy together — one orientation flag, two texture pairs).
 */
export interface PingPongPair<TSlot> {
    /** The two sides, in their fixed allocation order. */
    readonly a: TSlot
    readonly b: TSlot
    /** 0 = currently reading `a` and writing `b`; 1 = the reverse. */
    orientation(): PingPongOrientation
    /** The side the current orientation READS (the live state going into this frame's pass). */
    readSource(): TSlot
    /** The side the current orientation WRITES (the state this frame's pass just produced). */
    writeTarget(): TSlot
    /** Flip orientation for the next dispatch. */
    swap(): void
    /**
     * Force orientation back to 0 (read `a`, write `b`). For a sim that RE-SEEDS: the seed kernel
     * writes one known side, so the next step has to read that side whatever the flag happened to
     * be (ReactionDiffusion re-seeds on every preset change).
     */
    reset(): void
    /**
     * Pre-build BOTH orientations of one bind-group family and return an accessor for the current
     * one. `make(read, write)` gets the sides in read/write order. Callable at any time — including
     * from inside `bindInputs`, which is the only place a child-consuming sim can build groups.
     */
    groups<T>(make: (read: TSlot, write: TSlot) => T): () => T
    /**
     * Pre-build one bind group PER SIDE (for a pass that touches a single side — a gradient,
     * display, or output copy). Returns accessors for the two roles, so the caller says which
     * side it means rather than tracking the flag: `.written()` for the side the current
     * orientation just wrote, `.read()` for the side it read.
     */
    perSide<T>(make: (side: TSlot) => T): {read: () => T; written: () => T}
}

/**
 * Ping-pong pair for feedback simulations: two sides of state, an orientation flag, and
 * pre-built bind groups for both orientations so swapping is a flag flip with no allocation.
 *
 * Works for storage buffers, storage textures, or records of either — the pair is generic over
 * the slot type and never looks inside it. See {@link createPingPong} (deprecated) for why the
 * first attempt at this didn't fit any consumer.
 *
 * @example // CursorRipples: wave propagation + a gradient pass reading the just-written buffer
 * const pp = createPingPongPair(bufferA, bufferB)
 * const propGroup = pp.groups((read, write) =>
 *     root.createBindGroup(propagateLayout, {readBuf: read, writeBuf: write, params}))
 * const gradGroup = pp.perSide((buf) => root.createBindGroup(gradientLayout, {srcBuf: buf, dispTex}))
 * const nodes = [propagate.with(propGroup()), gradient.with(gradGroup.written())]
 * pp.swap()
 *
 * @example // TimeTrail: two texture pairs swapping in lockstep
 * const pp = createPingPongPair({state: stateA, prevLive: liveA}, {state: stateB, prevLive: liveB})
 */
export function createPingPongPair<TSlot>(a: TSlot, b: TSlot): PingPongPair<TSlot> {
    let flipped = false
    const read = (): TSlot => (flipped ? b : a)
    const write = (): TSlot => (flipped ? a : b)
    return {
        a,
        b,
        orientation: () => (flipped ? 1 : 0),
        readSource: read,
        writeTarget: write,
        swap() {
            flipped = !flipped
        },
        reset() {
            flipped = false
        },
        groups<T>(make: (r: TSlot, w: TSlot) => T): () => T {
            const ab = make(a, b)
            const ba = make(b, a)
            return () => (flipped ? ba : ab)
        },
        perSide<T>(make: (side: TSlot) => T): {read: () => T; written: () => T} {
            const forA = make(a)
            const forB = make(b)
            return {
                read: () => (flipped ? forB : forA),
                written: () => (flipped ? forA : forB),
            }
        },
    }
}

/**
 * Storage buffer for particle / simulation state. Returns a fixed-length array buffer with
 * `storage` usage, bindable to a `{ storage, access: 'mutable' }` layout entry and
 * read/written on the GPU across frames
 * (CursorTrail, Smoke, FloatingParticles, …). Initialize / read back via `.write` / `.read`.
 *
 * @example const state = createStateBuffer(root, d.vec4f, particleCount)
 */
export function createStateBuffer<TElement extends AnyWgslData>(
    root: TgpuRoot,
    schema: TElement,
    count: number,
): TgpuBuffer<WgslArray<TElement>> & StorageFlag {
    // TypeGPU's `ValidateBufferSchema` / `AllowedUsages` conditional types don't resolve for a
    // generic element type, so the schema + usage args are cast through; the runtime call is a
    // plain `createBuffer(arrayOf(...)).$usage('storage')`.
    const buffer = root.createBuffer(d.arrayOf(schema, count) as never)
    return buffer.$usage('storage' as never) as unknown as TgpuBuffer<WgslArray<TElement>> & StorageFlag
}
