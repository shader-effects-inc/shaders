/**
 * Frame loop, throttling, deltaTime, synthetic frames, and `onSubmittedWorkDone` waits.
 *
 * This module holds the PURE, unit-testable pieces of the frame layer (spring physics,
 * easing curves, the texture-cap clamp, the visibility/60fps throttle gate, the shared-clock
 * time derivation) plus a thin RAF loop factory. The stateful render logic (drivers →
 * flush → compose → render → callbacks) lives in `index.ts`, which composes these helpers.
 *
 * We own the clock: shader `time` is `(performance.now() - timeOrigin) / 1000` when a shared
 * origin is set, else a delta-accumulated seconds counter — see {@link deriveElapsedTime}.
 *
 * SSR-safe: no window/navigator access at import time.
 */
import type {TgpuRoot} from 'typegpu'

// ═══════════════════════════════════════════════════════════════════════════════════════
// Spring-damper physics
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Largest integration step the spring is advanced by in one go, in seconds. The step below is
 * semi-implicit Euler, which is only stable while `dt·√stiffness` stays comfortably under 2 —
 * at the snappiest stiffness (200) that means dt ≲ 0.14s in theory and ~0.1s in practice.
 * The frame gate clamps the live delta to exactly 0.1s, and the 1 FPS off-screen throttle (or
 * the first frame back from a hidden tab) hands us that clamped 0.1s EVERY frame — at which
 * point the spring flips sign each step and grows without bound ("the shape spun like crazy
 * when I came back to the tab"). Sub-stepping at ≤ 1/60s keeps every frame in the stable
 * regime, and a normal 60 FPS delta takes exactly one step, so the on-screen feel is untouched.
 */
const SPRING_MAX_SUBSTEP = 1 / 60

/**
 * Spring-damper step for smooth mouse tracking. Returns `[newPosition, newVelocity]`.
 * When smoothing and momentum are both 0 the target is returned immediately (zero cost).
 * Large deltas are integrated in ≤ 1/60s sub-steps (see {@link SPRING_MAX_SUBSTEP}).
 */
export function applySpring(
    current: number,
    velocity: number,
    target: number,
    smoothing: number,
    momentum: number,
    dt: number,
): [number, number] {
    if (smoothing === 0 && momentum === 0) return [target, 0]
    const stiffness = 200 * Math.pow(0.01, smoothing) // 200 (snappy) → 2 (slow)
    const criticalDamping = 2 * Math.sqrt(stiffness)
    const damping = criticalDamping * (1 - momentum * 0.85)
    const steps = dt > SPRING_MAX_SUBSTEP ? Math.ceil(dt / SPRING_MAX_SUBSTEP) : 1
    const h = dt / steps
    let pos = current
    let vel = velocity
    for (let i = 0; i < steps; i++) {
        const force = stiffness * (target - pos)
        vel += (force - damping * vel) * h
        pos += vel * h
    }
    return [pos, vel]
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Easing curves for auto-animate
// ═══════════════════════════════════════════════════════════════════════════════════════

/** The easing names the auto-animate driver accepts. */
export type EasingName = 'linear' | 'quad' | 'expo' | 'bounce' | 'sine'

/** Uncharted-style multi-segment bounce ease. */
export function applyBounceEase(t: number): number {
    const n1 = 7.5625,
        d1 = 2.75
    if (t < 1 / d1) return n1 * t * t
    if (t < 2 / d1) {
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    }
    if (t < 2.5 / d1) {
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    }
    t -= 2.625 / d1
    return n1 * t * t + 0.984375
}

/** Maps a 0..1 phase through the named easing. Default is `sine`. */
export function applyEasing(t: number, easing: string): number {
    switch (easing) {
        case 'linear':
            return t
        case 'quad':
            return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
        case 'expo':
            if (t === 0) return 0
            if (t === 1) return 1
            return t < 0.5 ? Math.pow(2, 20 * t - 10) / 2 : (2 - Math.pow(2, -20 * t + 10)) / 2
        case 'bounce':
            return applyBounceEase(t)
        case 'sine':
        default:
            return (1 - Math.cos(Math.PI * t)) / 2
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Texture-cap clamp
// ═══════════════════════════════════════════════════════════════════════════════════════

export interface ClampEnv {
    /** Device `maxTextureDimension2D` (or the 8192 spec-minimum fallback pre-init). */
    maxTextureDim: number
    /** The DPR factor we back the canvas at. */
    pixelRatio: number
    /** Live viewport width in CSS px (window.innerWidth). */
    viewportWidth: number
    /** Live viewport height in CSS px (window.innerHeight). */
    viewportHeight: number
}

/**
 * Clamps requested CSS-pixel dimensions to a safe GPU buffer size while preserving aspect
 * ratio. The buffer (cssSize × pixelRatio) never exceeds the viewport or
 * `maxTextureDimension2D`. Pure (env injected) so it is golden-testable; `index.ts` wraps it
 * with the live window + DPR. The `-1` slack guards a borderline value from rounding over the
 * cap.
 */
export function clampToTextureCap(w: number, h: number, env: ClampEnv): {width: number; height: number} {
    const pr = env.pixelRatio || 1
    // GPU buffer = cssSize * pr, so the CSS-pixel cap is gpuMax / pr.
    const gpuCssCap = Math.max(1, Math.floor(env.maxTextureDim / pr) - 1)
    const capW = Math.min(w, env.viewportWidth, gpuCssCap)
    const capH = Math.min(h, env.viewportHeight, gpuCssCap)
    const scale = Math.min(capW / w, capH / h, 1)
    return {
        width: Math.max(1, Math.round(w * scale)),
        height: Math.max(1, Math.round(h * scale)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Shared-clock time derivation
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * The elapsed-seconds clock the shader `time` uniform reads. With a shared wall-clock origin
 * (synced multi-tile animation), time is `(now - origin)/1000` so every renderer sharing the
 * origin lands on the same value and self-heals across pauses. Without one, it accumulates
 * the (already clamped) per-frame delta — "seconds since this renderer started".
 */
export function deriveElapsedTime(
    sharedTimeOrigin: number | null,
    prevElapsed: number,
    deltaTime: number,
    now: number,
): number {
    if (sharedTimeOrigin != null) return (now - sharedTimeOrigin) / 1000
    return prevElapsed + deltaTime
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Frame throttle gate
// ═══════════════════════════════════════════════════════════════════════════════════════

/** Off-screen render interval (1 FPS) when not forced to full frame rate. */
export const OFF_SCREEN_FPS_INTERVAL = 1000
/**
 * On-screen minimum frame interval — a 60 FPS cap with 1ms jitter tolerance so a 60Hz
 * display hits a true 60fps without the aliasing that a cap exactly equal to the refresh
 * interval produces.
 */
export const MIN_FRAME_INTERVAL = 1000 / 60 - 1

export interface FrameGateState {
    /** Timestamp (performance.now ms) of the last rendered frame, or 0 if none yet. */
    lastRenderTime: number
    /** IntersectionObserver visibility — false throttles to 1 FPS. */
    isVisible: boolean
    /** When true, off-screen throttling is bypassed (hidden-canvas partner integrations). */
    forceFullFrameRate: boolean
    /**
     * Optional host-imposed minimum frame interval (ms) while ON-screen — the
     * per-renderer frame-rate cap (`setFrameRateCap`). A multi-tile canvas uses it
     * to demote tiny tiles to e.g. 15 FPS. 0/undefined = the default 60 FPS cap.
     * Never loosens the cap below MIN_FRAME_INTERVAL, and never affects the
     * off-screen 1 FPS throttle.
     */
    minInterval?: number
}

export interface FrameGateResult {
    /** Whether this frame should render (false = skip, throttled). */
    render: boolean
    /** Delta since the last frame, in seconds, clamped to 0.1. */
    deltaTime: number
}

/**
 * Decides whether the current tick should render and, if so, the clamped delta. Variable
 * frame rate: 1 FPS off-screen, up to 60 FPS on-screen (with the jitter-tolerant cap), delta
 * clamped to 0.1s. The synthetic-frame path bypasses this entirely (see {@link syntheticDelta});
 * it is only used by the RAF-driven live loop.
 */
export function frameGate(now: number, state: FrameGateState): FrameGateResult {
    const {lastRenderTime, isVisible, forceFullFrameRate, minInterval = 0} = state
    const deltaTime = lastRenderTime > 0 ? (now - lastRenderTime) / 1000 : 0.016
    if (lastRenderTime > 0) {
        const sinceLast = now - lastRenderTime
        if (!isVisible && !forceFullFrameRate && sinceLast < OFF_SCREEN_FPS_INTERVAL) {
            return {render: false, deltaTime}
        }
        if ((isVisible || forceFullFrameRate) && sinceLast < Math.max(MIN_FRAME_INTERVAL, minInterval)) {
            return {render: false, deltaTime}
        }
    }
    return {render: true, deltaTime: Math.min(deltaTime, 0.1)}
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// GPU idle wait
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Resolves when the device has finished all previously submitted work, via
 * `device.queue.onSubmittedWorkDone()`. Falls back to a one-frame timeout if the queue does
 * not expose the API (never expected on a real device; keeps the screenshot/export path from
 * hanging on a mock).
 */
export async function awaitGpuIdle(root: TgpuRoot): Promise<void> {
    try {
        const queue = root.device?.queue
        if (queue && typeof queue.onSubmittedWorkDone === 'function') {
            await queue.onSubmittedWorkDone()
            return
        }
    } catch {
        /* fall through to the timeout safety net */
    }
    await new Promise((resolve) => setTimeout(resolve, 16))
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Frame sequence orchestration (the canonical frame order, in one place)
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * The ordered per-frame steps. Kept as an interface + a pure runner so the canonical order is
 * declared once and unit-testable with mocks that record call order (no device needed).
 */
export interface FrameSequence {
    /** CPU driver springs / easing + animated-time advance (before anything is flushed). */
    updateDrivers(): void
    /** Ensure the desired composition is built + bound. Return false to skip (nothing to draw). */
    ensureComposition(): boolean
    /** onBeforeRender shader callbacks — they write uniforms, so they run before flush. */
    beforeRender(): void
    /** Coalesced uniform buffer patch for the bound composition. */
    flush(): void
    /**
     * Draw the composition currently selected by the pipeline cache (swap-when-ready). Return
     * `false` when nothing was drawn (a broken composition) so `markReady` is skipped.
     */
    render(): void | boolean
    /** Promote the pending composition once it has drawn a frame + fire onReady. */
    markReady(): void
    /** onAfterRender shader callbacks. */
    afterRender(): void
}

/**
 * Runs one frame's steps in the canonical order: drivers/time → [ensure composition] →
 * onBeforeRender → flush → render → markReady → onAfterRender. `ensureComposition` returning
 * false short-circuits the rest (no root composed yet). onBeforeRender precedes flush because
 * those callbacks write uniforms; in steady state `ensureComposition` is a cache-hit no-op.
 */
export function runFrameSequence(seq: FrameSequence): void {
    seq.updateDrivers()
    if (!seq.ensureComposition()) return
    seq.beforeRender()
    seq.flush()
    const drew = seq.render()
    // A render that drew nothing (a composition a custom body broke) must not report the
    // canvas ready — `onReady` promises a first composed frame.
    if (drew !== false) seq.markReady()
    seq.afterRender()
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// RAF loop factory
// ═══════════════════════════════════════════════════════════════════════════════════════

export interface FrameLoop {
    /** Start the loop (idempotent). No-op if already running. */
    start(): void
    /** Stop the loop and cancel any pending frame. */
    stop(): void
    /** Whether the loop is currently running. */
    readonly running: boolean
}

/**
 * A minimal `requestAnimationFrame` loop that invokes `tick` once per frame. The throttle
 * decision lives in `tick` itself (via {@link frameGate}); this factory only owns the RAF
 * scheduling + teardown. SSR-safe: `requestAnimationFrame` is resolved lazily at `start()`,
 * never at import.
 */
export function createFrameLoop(tick: () => void): FrameLoop {
    let rafId: number | null = null
    const raf = (cb: FrameRequestCallback): number =>
        typeof requestAnimationFrame === 'function' ? requestAnimationFrame(cb) : (setTimeout(() => cb(performance.now()), 16) as unknown as number)
    const cancel = (id: number): void => {
        if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(id)
        else clearTimeout(id as unknown as ReturnType<typeof setTimeout>)
    }
    const step = (): void => {
        tick()
        // Re-check running: `tick` (or a device-loss hook) may have called stop().
        if (rafId !== null) rafId = raf(step)
    }
    return {
        start(): void {
            if (rafId !== null) return
            rafId = raf(step)
        },
        stop(): void {
            if (rafId !== null) {
                cancel(rafId)
                rafId = null
            }
        },
        get running(): boolean {
            return rafId !== null
        },
    }
}
