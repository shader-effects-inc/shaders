/**
 * WebGPU availability probing + the library's quiet-by-default diagnostics channel.
 *
 * Shaders is WebGPU-only. On a browser/driver that can't provide a working device the
 * ONLY acceptable outcome is a transparent canvas and a silent console — a decorative
 * background must never break the page it decorates, and must never fill a partner's
 * console with errors they can do nothing about. Every "the GPU can't do this" path in
 * the renderer therefore funnels through here:
 *
 *   - {@link isWebGPUSupported} / {@link getWebGPUSupport} let hosts branch BEFORE mounting.
 *   - {@link GpuUnavailableError} carries a machine-readable {@link GpuFailureReason}.
 *   - {@link debugWarn} / {@link debugError} are no-ops unless debugging is explicitly
 *     switched on, so production consoles stay clean while developers keep a way in.
 *
 * SSR-safe: nothing here touches `navigator` / `localStorage` at import time.
 */

/**
 * Why the renderer can't (or can no longer) draw. Surfaced to hosts via
 * `setOnUnavailable` and to `createShader`'s `onError`.
 */
export type GpuFailureReason =
    /** The browser has no WebGPU at all (`navigator.gpu` is undefined). */
    | 'unsupported'
    /** WebGPU exists but no adapter was granted (blocklisted driver, headless, no GPU). */
    | 'no-adapter'
    /** An adapter exists but `requestDevice` failed or returned a dead device. */
    | 'no-device'
    /** Anything else threw while starting the renderer. */
    | 'init-failed'
    /** The device was lost after a successful start and no host recovery is wired up. */
    | 'device-lost'
    /** The GPU refused an allocation (`GPUOutOfMemoryError`). */
    | 'out-of-memory'
    /** Sustained uncaptured/internal device errors — the device is not usable. */
    | 'gpu-error'
    /** Repeated exceptions while composing or drawing. */
    | 'render-failed'
    /**
     * The composition needs more of some GPU resource than this device allows (today: more
     * uniform bytes than `maxUniformBufferBindingSize`). Unlike the other reasons this one is
     * about the CONTENT, not the browser — a smaller composition would run fine here.
     */
    | 'limit-exceeded'

/** An error whose cause is "this environment cannot run WebGPU", not "we have a bug". */
export class GpuUnavailableError extends Error {
    readonly reason: GpuFailureReason
    constructor(reason: GpuFailureReason, message?: string, options?: {cause?: unknown}) {
        super(message ?? `[gpu] WebGPU unavailable (${reason})`)
        this.name = 'GpuUnavailableError'
        this.reason = reason
        if (options?.cause !== undefined) (this as {cause?: unknown}).cause = options.cause
    }
}

/**
 * Structural test for {@link GpuUnavailableError}, used instead of `instanceof`.
 *
 * `instanceof` compares class identity, which silently breaks whenever the same source file
 * ends up as two module records — a bundler splitting `support` into its own chunk, or a
 * test/build alias resolving `./support` and `@coreroot/gpu/support` to different specifiers.
 * The failure mode is nasty: a definite "this device can't do it" verdict gets misread as a
 * generic error and retried. Duck-typing the shape is stable across all of that.
 */
export function isGpuUnavailableError(error: unknown): error is GpuUnavailableError {
    return (
        !!error &&
        typeof error === 'object' &&
        (error as {name?: unknown}).name === 'GpuUnavailableError' &&
        typeof (error as {reason?: unknown}).reason === 'string'
    )
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Debug channel
// ═══════════════════════════════════════════════════════════════════════════════════════

let debugOverride: boolean | null = null

/**
 * Turn Shaders' diagnostic logging on/off at runtime. Off by default — a page that can't
 * run WebGPU gets a transparent canvas and NOTHING in the console.
 *
 * Also enabled (without calling this) by either of:
 *   - `globalThis.__SHADERS_DEBUG__ = true`
 *   - `localStorage.setItem('shaders:debug', '1')`
 *
 * @param enabled `true`/`false` to force, `null` to fall back to the globals above.
 */
export function setShadersDebug(enabled: boolean | null): void {
    debugOverride = enabled
}

/** Whether diagnostic logging is currently switched on. */
export function isShadersDebug(): boolean {
    if (debugOverride !== null) return debugOverride
    const flag = (globalThis as {__SHADERS_DEBUG__?: unknown}).__SHADERS_DEBUG__
    if (flag !== undefined && flag !== null) return !!flag
    try {
        if (typeof localStorage !== 'undefined' && localStorage.getItem('shaders:debug') === '1') return true
    } catch {
        /* storage blocked (sandboxed iframe / privacy mode) — not a debug signal */
    }
    return false
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Custom WGSL diagnostics
// ═══════════════════════════════════════════════════════════════════════════════════════

let customWgslUsed = false

/**
 * Flipped the first time a `wgsl\`…\`` body (a user-authored shader) is composed. While set,
 * the renderer reports shader-compilation and composition failures with `console.error`
 * even when debugging is off: an author writing WGSL by hand needs the compiler's message,
 * and a blank canvas is not one. Library shaders never trip this, so nothing changes for
 * pages that only use the built-in components.
 */
export function markCustomWgslUsed(): void {
    customWgslUsed = true
}

/** Whether a user-authored WGSL body has been composed on this page. */
export function hasCustomWgsl(): boolean {
    return customWgslUsed
}

/** `console.error` that speaks when debugging is on OR a custom WGSL body is in play. */
export function authorError(...args: unknown[]): void {
    if (isShadersDebug() || customWgslUsed) console.error(...args)
}

/** `console.warn` that only speaks when debugging is on. */
export function debugWarn(...args: unknown[]): void {
    if (isShadersDebug()) console.warn(...args)
}

/** `console.error` that only speaks when debugging is on. */
export function debugError(...args: unknown[]): void {
    if (isShadersDebug()) console.error(...args)
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Availability
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Cheap synchronous check: does this browser expose the WebGPU API at all?
 *
 * `true` here does NOT guarantee a device can be created (a driver may be blocklisted, or
 * the adapter request may be denied) — it only rules out browsers with no WebGPU. Use it
 * to skip mounting shader UI entirely; use {@link getWebGPUSupport} when you need
 * certainty before committing to a layout.
 *
 * SSR → `false`.
 */
export function isWebGPUSupported(): boolean {
    return typeof navigator !== 'undefined' && !!(navigator as {gpu?: unknown}).gpu
}

/** Result of the full async support probe. */
export interface WebGPUSupportInfo {
    /** True only when an adapter was actually granted. */
    supported: boolean
    /** Why not, when `supported` is false. */
    reason?: GpuFailureReason
}

let probeCache: Promise<WebGPUSupportInfo> | null = null

/**
 * Full availability probe — asks the browser for an adapter and caches the answer for the
 * lifetime of the page. Requesting an adapter is cheap and side-effect-free (it does not
 * create a device), and browsers memoize it internally.
 *
 * Use this when a host needs to decide between a shader and a static fallback before
 * painting. The renderer itself only calls it on the failure path, to attribute a reason.
 */
export function getWebGPUSupport(): Promise<WebGPUSupportInfo> {
    if (probeCache) return probeCache
    probeCache = (async (): Promise<WebGPUSupportInfo> => {
        if (!isWebGPUSupported()) return {supported: false, reason: 'unsupported'}
        try {
            const adapter = await (navigator as unknown as {gpu: GPU}).gpu.requestAdapter({
                powerPreference: 'high-performance',
            })
            if (!adapter) return {supported: false, reason: 'no-adapter'}
            return {supported: true}
        } catch (error) {
            debugError('[gpu] adapter probe threw:', error)
            return {supported: false, reason: 'no-adapter'}
        }
    })()
    return probeCache
}

/** Testing seam — drop the cached probe result. */
export function __resetWebGPUSupportProbe(): void {
    probeCache = null
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Page-level circuit breaker
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Reasons that are facts about the PAGE, not about one renderer. Once any renderer hits
 * one of these, every other renderer on the page would hit it too — and on the memory
 * failures, each additional attempt actively makes things worse. That cascade (a page with
 * several shaders, each independently requesting a device and each independently
 * exhausting the same GPU) is what turns "one effect didn't load" into "the tab froze".
 *
 * Deliberately excludes `init-failed`, `render-failed` and `device-lost`: those can be
 * specific to one shader's pipeline or one tile's device, and must not blank the page.
 */
const PAGE_WIDE_REASONS: ReadonlySet<GpuFailureReason> = new Set<GpuFailureReason>([
    'unsupported',
    'no-adapter',
    'no-device',
    'out-of-memory',
    'gpu-error',
])

let gpuUnusableReason: GpuFailureReason | null = null

/**
 * Record a failure that rules out WebGPU for the whole page. No-op for reasons that are
 * specific to a single renderer.
 */
export function markGpuUnusable(reason: GpuFailureReason): void {
    if (gpuUnusableReason || !PAGE_WIDE_REASONS.has(reason)) return
    gpuUnusableReason = reason
}

/**
 * Why WebGPU is off the table for this page, or `null`. Renderers check this before asking
 * for a device — already-running renderers are untouched, we simply stop starting new ones.
 */
export function getGpuUnusableReason(): GpuFailureReason | null {
    return gpuUnusableReason
}

/** Testing seam — clear the page-level latch. */
export function __resetGpuUnusable(): void {
    gpuUnusableReason = null
}
