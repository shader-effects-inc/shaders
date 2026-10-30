/**
 * Device/root acquisition, feature requests, device-loss hook, per-device root cache
 * (WeakMap), and per-canvas context configuration.
 *
 * ## One device per page
 *
 * Unless a caller injects its own, every renderer on the page shares ONE lazily-created
 * GPUDevice — the "default root". The first `<Shader>` to mount creates it; every later one
 * adopts it, with no configuration from the developer.
 *
 * This matters more than it looks. A GPUDevice owns the browser's compiled-pipeline cache,
 * so a device per renderer means:
 *   - the same shader recompiles once per instance instead of once per page, and
 *   - an SPA route change destroys the device with the outgoing component, throwing the
 *     cache away, so the next route recompiles everything from scratch. On Windows, where
 *     Chrome compiles through HLSL and the graphics driver, that shows up as a main-thread
 *     stall of hundreds of milliseconds to seconds — visible as scroll freezing mid-navigation.
 *
 * Sharing one device also keeps a page well under browser limits on concurrent devices and
 * cuts adapter/device requests to one.
 *
 * The default device is deliberately never destroyed by a renderer's `cleanup()` — outliving
 * the components is the entire point (that's what preserves the cache across route changes).
 * It is released when the page goes away, or explicitly via {@link destroyDefaultGpuDevice}.
 */
import tgpu, {type TgpuRoot, warn} from 'typegpu'

import {GpuUnavailableError, debugWarn, getWebGPUSupport} from './support'

// TypeGPU's 'external-omitted' warning is a false positive for our composer: every fragment
// body's raw WGSL text includes a literal `in.uv` (composer.ts's fragment fn parameter, not a
// resolver external), which TypeGPU's generic external-resolution pass flags on every fragment
// pass it composes. Default OFF (this ships into arbitrary consumer builds — unset/unusual
// `NODE_ENV` should stay silent, not noisy) and only re-enabled when `NODE_ENV` PROVES we're in
// our own dev/test workflow — the same exact check TypeGPU uses for its own DEV/TEST flags
// (typegpu/shared/env.js), so this tracks their verbosity rather than fighting it.
if (process.env.NODE_ENV !== 'development' && process.env.NODE_ENV !== 'test') {
    warn.disable('external-omitted')
}

/**
 * Optional GPU features requested when we create our OWN device. `float32-filterable`
 * lets us linearly sample float RTT targets; `timestamp-query` powers first-class GPU
 * timing in the performance tracker. Requested as OPTIONAL so a device lacking
 * them still initializes. The shared-device path (createSharedDevice.ts) requests all
 * adapter features itself and is unaffected by this list.
 */
const OPTIONAL_FEATURES: GPUFeatureName[] = ['float32-filterable', 'timestamp-query']

const ADAPTER_OPTIONS: GPURequestAdapterOptions = {powerPreference: 'high-performance'}

/**
 * Per-stage limits a large composition outgrows at the WebGPU defaults: every text atlas,
 * image and render-to-texture layer one pass reads counts against 16 sampled textures. The
 * default device asks for whatever the adapter supports (most desktop GPUs allow far more),
 * which costs nothing and lets big compositions compile.
 */
const RAISED_LIMITS = [
    'maxSampledTexturesPerShaderStage',
    'maxSamplersPerShaderStage',
    'maxStorageTexturesPerShaderStage',
    'maxStorageBuffersPerShaderStage',
    'maxUniformBufferBindingSize',
] as const

/** The adapter's own values for {@link RAISED_LIMITS}; empty when it can't be asked. */
async function adapterLimits(): Promise<Record<string, number>> {
    try {
        const adapter = await navigator.gpu.requestAdapter(ADAPTER_OPTIONS)
        const limits = adapter?.limits as unknown as Record<string, unknown> | undefined
        const out: Record<string, number> = {}
        for (const name of RAISED_LIMITS) {
            const value = limits?.[name]
            if (typeof value === 'number') out[name] = value
        }
        return out
    } catch {
        return {}
    }
}

/**
 * One TgpuRoot per GPUDevice. Injected/shared devices (Projects multi-tile — many
 * renderers, one device) resolve to the same root; each renderer still configures its
 * own canvas context. Weak so a lost/destroyed device's root becomes collectable.
 */
const rootsByDevice = new WeakMap<GPUDevice, TgpuRoot>()

/**
 * Per-device set of device-loss callbacks. The single real `device.lost` handler is
 * attached once per device (when its root is first cached) and fans out to every
 * registered callback — so multiple renderers sharing one device each get notified
 * exactly once.
 */
const lostCallbacks = new WeakMap<GPUDevice, Set<(info: GPUDeviceLostInfo) => void>>()

/**
 * Devices whose `.lost` promise has resolved. A lost device must never be adopted:
 * WebGPU keeps its API callable but every call silently no-ops, so a renderer built on
 * it "works" and stays blank forever. This is exactly the shared-device recovery hole —
 * a device-loss rebuild (createShader) passes the caller's ORIGINAL `gpu.device` back
 * into `acquireRoot`, and re-adopting the corpse defeats the rebuild.
 */
const lostDevices = new WeakSet<GPUDevice>()

/**
 * The process-wide default root, promise-cached from the moment the first caller asks for
 * one. Caching the PROMISE (not the resolved value) is what makes a burst of `<Shader>`s
 * mounting in the same tick converge on a single device request instead of racing to create
 * one each.
 *
 * Also serves as the replacement any renderer lands on when an injected device dies: routing
 * every orphan onto the one live default is both simpler and better than minting a
 * replacement per dead device.
 *
 * Invalidated (not repaired) when its device is lost or the acquisition rejects, so the next
 * caller transparently creates a fresh one.
 */
let defaultRootPromise: Promise<RootContext> | null = null

/**
 * Default-root promises handed to {@link destroyDefaultGpuDevice}. Keyed by the promise rather
 * than the device because destruction has to be announced SYNCHRONOUSLY, and the device isn't
 * known until the promise resolves. See the note in `acquireDefaultRoot`.
 */
const destroyedDefaultRoots = new WeakSet<Promise<RootContext>>()

export interface AcquireRootOptions {
    /**
     * Inject an existing device (shared-device / Projects path). When present we adopt it
     * via `tgpu.initFromDevice` and reuse one cached root per device; `root.destroy()`
     * will NOT destroy an adopted device (see `createdDevice`).
     */
    device?: GPUDevice
    /**
     * The adapter that produced an injected device, if the caller has it. Stored on the
     * returned context for `getGPUContext()` reporting; never used to request a device.
     */
    adapter?: GPUAdapter
    /**
     * Called when the device is lost for a reason other than `'destroyed'`. Wiring point
     * only; the renderer decides whether to rebuild.
     */
    onDeviceLost?: (info: GPUDeviceLostInfo) => void
}

export interface RootContext {
    root: TgpuRoot
    device: GPUDevice
    adapter?: GPUAdapter
    /**
     * True only when this renderer EXCLUSIVELY owns the device and its `cleanup()` may
     * destroy it. Always false today: an injected device belongs to the caller, and the
     * default device is shared page-wide and must outlive any one renderer. Kept because it
     * is part of the public `RootContext` shape and guards the destroy path.
     */
    createdDevice: boolean
    /**
     * Unsubscribe this renderer's device-loss callback. MUST be called from `cleanup()`: the
     * default device outlives every renderer on it, so a missed release leaks the callback
     * (and later fires it on a torn-down renderer) for the life of the page.
     */
    release?: () => void
}

/**
 * Acquire a TgpuRoot: around an injected device when the caller supplies one, otherwise
 * around the shared page-wide default (see the module header). SSR-safe: no `navigator.gpu`
 * access happens until this is called.
 *
 * @throws {GpuUnavailableError} when a device can't be obtained — this renderer is
 *   WebGPU-only, there is no fallback.
 */
export async function acquireRoot(options: AcquireRootOptions = {}): Promise<RootContext> {
    if (options.device && !lostDevices.has(options.device)) {
        const device = options.device
        let root = rootsByDevice.get(device)
        if (!root) {
            root = tgpu.initFromDevice({device})
            rootsByDevice.set(device, root)
        }
        const release = wireDeviceLost(root, options.onDeviceLost)
        return {root, device, adapter: options.adapter, createdDevice: false, release}
    }

    // Either nothing was injected (the common path — every framework `<Shader>`), or the
    // injected device is already lost: a shared device that died while the page was hidden,
    // with the caller's `gpu` handle still pointing at the corpse. Adopting a lost device
    // yields a permanently-blank renderer, so both cases resolve to the live default root.
    if (options.device) {
        debugWarn('[gpu] injected GPU device was lost — falling back to the shared default device')
    }
    const ctx = await acquireDefaultRoot()
    const release = wireDeviceLost(ctx.root, options.onDeviceLost)
    // `createdDevice: false`: the default device is shared by every renderer on the page, so
    // no single renderer's cleanup may destroy it.
    return {root: ctx.root, device: ctx.device, adapter: ctx.adapter, createdDevice: false, release}
}

/**
 * Resolve the page-wide default root, creating it on first use.
 *
 * Concurrency and self-healing are the whole job here:
 *   - The promise is cached SYNCHRONOUSLY before the first await, so N renderers mounting in
 *     the same tick share one in-flight device request rather than making N.
 *   - A cached root whose device has since been lost is evicted and replaced. After every
 *     await the cache is re-read, so only the caller still holding the stale promise evicts
 *     it; everyone else adopts whatever a faster caller already installed.
 *   - A rejected acquisition is evicted too, so a transient failure doesn't poison the page.
 *     (Repeat attempts are separately suppressed by the renderer's page-level circuit
 *     breaker, so this can't turn into a retry storm.)
 */
async function acquireDefaultRoot(): Promise<RootContext> {
    for (;;) {
        const cached = defaultRootPromise
        if (!cached) break
        const ctx = await cached.catch(() => null)
        // The `destroyedDefaultRoots` half is DEFENSIVE, not load-bearing today. A concurrent
        // destroyDefaultGpuDevice() only learns the device once its own `.then` runs, but that
        // `.then` is attached straight to `cached` whereas this continuation sits behind the
        // `.catch()` above — an extra microtask hop — so destroy always marks `lostDevices`
        // first and the check below already catches it. That ordering is accidental: rewrite
        // this line as a try/catch around `await cached` and the hop disappears along with the
        // protection. The promise-level marker is set synchronously by destroy, so it holds
        // regardless of how this await is spelled.
        if (ctx && !destroyedDefaultRoots.has(cached) && !lostDevices.has(ctx.device)) return ctx
        if (defaultRootPromise === cached) defaultRootPromise = null
    }
    const pending = createOwnRoot()
    defaultRootPromise = pending
    try {
        return await pending
    } catch (error) {
        if (defaultRootPromise === pending) defaultRootPromise = null
        throw error
    }
}

/**
 * Destroy the shared default device and forget it, so the next acquire builds a fresh one.
 *
 * Nothing in the library calls this — the default device is meant to outlive components (see
 * the module header). It exists for tests and for hosts that genuinely want to reclaim GPU
 * memory when they know no shader will mount again.
 */
export function destroyDefaultGpuDevice(): void {
    const pending = defaultRootPromise
    defaultRootPromise = null
    if (!pending) return
    // Mark the PROMISE synchronously, before anything gets a chance to await, so a caller
    // already parked inside acquireDefaultRoot on this exact promise cannot resume and adopt a
    // device we are about to destroy. Today that caller is also saved by microtask ordering
    // (see the note there) — this makes the guarantee explicit instead of emergent.
    destroyedDefaultRoots.add(pending)
    void pending
        .then((ctx) => {
            // Bar the device everywhere, not just on the default path: an injected-device
            // caller can fall through to `lostDevices` too.
            lostDevices.add(ctx.device)
            try {
                ctx.root.destroy()
            } catch {
                /* already gone */
            }
        })
        .catch(() => {
            /* never resolved — nothing to destroy */
        })
}

/** Testing seam — drop the cached default root WITHOUT destroying it. */
export function __resetDefaultRoot(): void {
    defaultRootPromise = null
}

/**
 * Request a fresh adapter + device and wrap them in a root (the non-injected path).
 *
 * Every failure mode here is an ENVIRONMENT failure, not a bug, so it surfaces as a
 * {@link GpuUnavailableError} carrying a machine-readable reason. The renderer turns that
 * into a transparent canvas + one host callback, with nothing written to the console.
 */
async function createOwnRoot(): Promise<RootContext> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new GpuUnavailableError('unsupported', '[gpu] WebGPU is not available in this environment (navigator.gpu is undefined).')
    }
    const init = (requiredLimits: Record<string, number>) => tgpu.init({
        adapter: ADAPTER_OPTIONS,
        device: {optionalFeatures: OPTIONAL_FEATURES, requiredLimits},
    })
    const raised = await adapterLimits()
    let root: TgpuRoot
    try {
        try {
            root = await init(raised)
        } catch (error) {
            // Raised limits are read from an adapter request of our own; should the device's
            // adapter disagree, fall back to the WebGPU defaults rather than fail outright.
            if (!Object.keys(raised).length) throw error
            root = await init({})
        }
    } catch (error) {
        // Attribute the failure: `navigator.gpu` existing tells us nothing about whether an
        // adapter is actually obtainable (blocklisted driver, no GPU, denied in a sandboxed
        // iframe). The probe is cached, so this costs one adapter request per page at most —
        // and only ever on a path that has already failed.
        const info = await getWebGPUSupport()
        throw new GpuUnavailableError(info.supported ? 'no-device' : (info.reason ?? 'init-failed'), undefined, {cause: error})
    }
    // A device that is already lost the moment we get it (some software/fallback drivers do
    // this) would otherwise produce a renderer that silently no-ops forever. Treat it as an
    // unavailable environment right away.
    if (!root.device) {
        throw new GpuUnavailableError('no-device', '[gpu] adapter returned no device.')
    }
    rootsByDevice.set(root.device, root)
    return {root, device: root.device, createdDevice: true}
}

/**
 * Create and configure a WebGPU context for a canvas. Each canvas gets its own context
 * even when several share a device. Output is `premultiplied` by default; shaders
 * emit straight alpha, so the final pass premultiplies RGB by A. Pass overrides here to
 * change alpha/sRGB behaviour.
 */
export function configureCanvasContext(
    root: TgpuRoot,
    canvas: HTMLCanvasElement,
    options: Omit<Parameters<TgpuRoot['configureContext']>[0], 'canvas'> = {},
): GPUCanvasContext {
    return root.configureContext({
        canvas,
        alphaMode: 'premultiplied',
        ...options,
    })
}

/** Look up the cached root for an already-adopted device, if any (Projects multi-tile). */
export function getRootForDevice(device: GPUDevice): TgpuRoot | undefined {
    return rootsByDevice.get(device)
}

/**
 * Subscribe a renderer to this device's loss, returning an unsubscribe.
 *
 * The unsubscribe is load-bearing now that the default device is shared page-wide and
 * outlives every renderer on it. Without it each mount would leave its callback on the
 * device forever: an SPA that navigates twenty times accumulates twenty subscribers, and a
 * device loss then fires nineteen callbacks belonging to torn-down renderers.
 */
function wireDeviceLost(root: TgpuRoot, onDeviceLost?: (info: GPUDeviceLostInfo) => void): () => void {
    const device = root.device
    let callbacks = lostCallbacks.get(device)
    if (!callbacks) {
        callbacks = new Set()
        lostCallbacks.set(device, callbacks)
        // Attach the single real handler once per device; fan out to all callbacks.
        void root.device.lost.then((info) => {
            // Mark BEFORE the callbacks run (and for 'destroyed' too): a rebuild kicked
            // off by a callback re-enters acquireRoot with this same device, and must
            // see it as dead — never adoptable — whatever the loss reason was.
            lostDevices.add(device)
            if (info.reason === 'destroyed') return
            // Snapshot: a callback may unsubscribe (or subscribe) while we're fanning out.
            for (const cb of [...(lostCallbacks.get(device) ?? [])]) cb(info)
        })
    }
    if (!onDeviceLost) return () => {}
    const set = callbacks
    set.add(onDeviceLost)
    return () => set.delete(onDeviceLost)
}
