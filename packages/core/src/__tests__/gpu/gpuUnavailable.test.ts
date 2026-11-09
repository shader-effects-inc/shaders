import {describe, it, expect, vi, beforeEach, afterEach} from 'vitest'

/**
 * Graceful degradation when WebGPU can't run — the contract every browser without a working
 * WebGPU implementation must get:
 *
 *   1. `initialize()` RESOLVES (never rejects) and the canvas is left transparent (it was
 *      never drawn to, and we never paint over it).
 *   2. NOTHING is written to the console — not on the unsupported path, not per frame.
 *   3. The failure is reported once, with a machine-readable reason, via `setOnUnavailable`.
 *   4. Nothing keeps running afterwards: no RAF loop, no node registration, no retry on the
 *      re-entry points hosts call (visibility observers, colorSpace/toneMapping changes).
 *   5. Uncaptured device errors are CANCELLED (so the browser doesn't print
 *      "Uncaptured WebGPU error: …") and out-of-memory kills the renderer immediately —
 *      the reported hard-freeze came from continuing to submit work after an OOM.
 *
 * GPU-free: `typegpu` is mocked, so these run in plain jsdom.
 */

const mocks = vi.hoisted(() => ({
    /** What the mocked `tgpu.init` should do on the next call. */
    initBehaviour: 'ok' as 'ok' | 'throw',
    /** The device handed out by a successful `tgpu.init`. */
    device: null as unknown,
}))

// Only the device-acquisition entry points are stubbed — everything else must stay real,
// because importing the renderer pulls in the whole shader kit (`tgpu.fn`, `d`, `std`).
vi.mock('typegpu', async (importOriginal) => {
    const actual = (await importOriginal()) as {default: Record<string, unknown>}
    return {
        ...actual,
        default: {
            ...actual.default,
            initFromDevice: ({device}: {device: unknown}) => makeRoot(device),
            init: async () => {
                if (mocks.initBehaviour === 'throw') throw new Error('requestDevice failed')
                return makeRoot(mocks.device)
            },
        },
    }
})

/** A TgpuRoot stand-in: only the members the renderer touches before it gives up. */
function makeRoot(device: unknown) {
    return {
        device,
        configureContext: () => ({unconfigure: () => {}}),
        createTexture: () => ({}),
        destroy: () => {},
    }
}

/** A GPUDevice stand-in with a real EventTarget so uncapturederror dispatch is exercised. */
function makeDevice(limits: Record<string, number> = {}): GPUDevice & {
    dispatch: (error: unknown) => boolean
    lose: (reason: string) => void
} {
    const target = new EventTarget()
    let resolveLost!: (info: {reason: string; message: string}) => void
    const lost = new Promise<{reason: string; message: string}>((res) => {
        resolveLost = res
    })
    const device = {
        lost,
        lose: (reason: string) => resolveLost({reason, message: reason}),
        limits: {maxTextureDimension2D: 8192, maxUniformBufferBindingSize: 65536, ...limits},
        queue: {onSubmittedWorkDone: async () => {}},
        addEventListener: target.addEventListener.bind(target),
        removeEventListener: target.removeEventListener.bind(target),
        destroy: () => {},
        dispatch(error: unknown): boolean {
            // `cancelable` mirrors the real GPUUncapturedErrorEvent; dispatchEvent returns
            // false once a listener has called preventDefault(), which is exactly the signal
            // a browser uses to decide whether to print the error itself.
            const event = new Event('uncapturederror', {cancelable: true})
            ;(event as unknown as {error: unknown}).error = error
            return target.dispatchEvent(event)
        },
    }
    return device as unknown as GPUDevice & {dispatch: (error: unknown) => boolean; lose: (reason: string) => void}
}

/**
 * A canvas whose layout box is non-zero, inside a parent (the shape every framework host
 * renders) so initialize() gets past sizing and finds an element to observe for resize.
 */
function makeCanvas(): HTMLCanvasElement {
    const parent = document.createElement('div')
    const canvas = document.createElement('canvas')
    parent.appendChild(canvas)
    document.body.appendChild(parent)
    canvas.getBoundingClientRect = () =>
        ({width: 200, height: 100, top: 0, bottom: 100, left: 0, right: 200, x: 0, y: 0, toJSON: () => ({})}) as DOMRect
    return canvas
}

const settle = () => new Promise((r) => setTimeout(r, 0))

/**
 * Register a root + one ordinary node carrying a real uniform, so the frame sequence actually
 * reaches the composition build (uniform-store finalize onwards) rather than short-circuiting
 * on an empty tree.
 */
function mountOneNode(r: ReturnType<typeof shaderRendererGPU>): void {
    const passthrough = {name: 'Root', props: {}, fragment: (() => ({})) as never}
    const gen = {name: 'Gen', props: {}, fragment: (() => ({})) as never}
    r.registerNode('root', passthrough.fragment, null, null, {}, passthrough as never)
    r.registerNode('n0', gen.fragment, 'root', null, {speed: {value: 1}} as never, gen as never)
}

const {shaderRendererGPU} = await import('@coreroot/gpu/index')
const {__resetWebGPUSupportProbe, __resetGpuUnusable, isWebGPUSupported} = await import('@coreroot/gpu/support')
const {__resetDefaultRoot} = await import('@coreroot/gpu/root')

/** Spies on every console method, asserting the library stays silent by default. */
function silenceGuard() {
    return {
        log: vi.spyOn(console, 'log').mockImplementation(() => {}),
        info: vi.spyOn(console, 'info').mockImplementation(() => {}),
        warn: vi.spyOn(console, 'warn').mockImplementation(() => {}),
        error: vi.spyOn(console, 'error').mockImplementation(() => {}),
    }
}

function expectSilent(spies: ReturnType<typeof silenceGuard>): void {
    for (const [name, spy] of Object.entries(spies)) {
        expect(spy, `console.${name} was called: ${JSON.stringify(spy.mock.calls)}`).not.toHaveBeenCalled()
    }
}

beforeEach(() => {
    mocks.initBehaviour = 'ok'
    mocks.device = makeDevice()
    __resetWebGPUSupportProbe()
    __resetGpuUnusable()
    // The default GPU root is shared page-wide; drop it so each test starts from a clean
    // device rather than adopting one a previous test created.
    __resetDefaultRoot()
    ;(navigator as {gpu?: unknown}).gpu = {requestAdapter: async () => ({})}
})

afterEach(() => {
    vi.restoreAllMocks()
})

describe('isWebGPUSupported', () => {
    it('is false when the browser exposes no WebGPU', () => {
        delete (navigator as {gpu?: unknown}).gpu
        expect(isWebGPUSupported()).toBe(false)
    })

    it('is true when navigator.gpu exists', () => {
        expect(isWebGPUSupported()).toBe(true)
    })
})

describe('no WebGPU in this browser', () => {
    it('resolves initialize(), stays silent, and reports reason "unsupported"', async () => {
        delete (navigator as {gpu?: unknown}).gpu
        const spies = silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)

        await expect(r.initialize({canvas: makeCanvas()})).resolves.toBeUndefined()
        await settle()

        expect(onUnavailable).toHaveBeenCalledTimes(1)
        expect(onUnavailable.mock.calls[0][0]).toBe('unsupported')
        expect(r.getFailureReason()).toBe('unsupported')
        expect(r.isInitialized()).toBe(false)
        expectSilent(spies)
    })

    it('notifies a LATE subscriber (registered after initialize already failed)', async () => {
        delete (navigator as {gpu?: unknown}).gpu
        silenceGuard()
        const r = shaderRendererGPU()
        await r.initialize({canvas: makeCanvas()})

        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)
        await settle()

        expect(onUnavailable).toHaveBeenCalledWith('unsupported')
    })

    it('refuses to retry — hosts re-enter initialize on visibility and prop changes', async () => {
        delete (navigator as {gpu?: unknown}).gpu
        silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)

        const canvas = makeCanvas()
        await r.initialize({canvas})
        await r.initialize({canvas})
        await r.initialize({canvas})
        await settle()

        // One attempt, one notification — not one per re-entry.
        expect(onUnavailable).toHaveBeenCalledTimes(1)
    })

    it('accepts no nodes and starts no animation loop once it has given up', async () => {
        delete (navigator as {gpu?: unknown}).gpu
        silenceGuard()
        const raf = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(() => 1)
        const r = shaderRendererGPU()
        await r.initialize({canvas: makeCanvas()})

        r.registerNode('n1', (() => ({})) as never, null, null, {}, {name: 'X', props: {}, fragment: (() => ({})) as never} as never)
        r.startAnimation()

        expect(r.getNodeRegistry().nodes.size).toBe(0)
        expect(raf).not.toHaveBeenCalled()
    })
})

describe('WebGPU present but no device', () => {
    it('reports "no-adapter" when the adapter request comes back empty', async () => {
        mocks.initBehaviour = 'throw'
        ;(navigator as {gpu?: unknown}).gpu = {requestAdapter: async () => null}
        const spies = silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)

        await r.initialize({canvas: makeCanvas()})
        await settle()

        expect(onUnavailable).toHaveBeenCalledWith('no-adapter', expect.anything())
        expectSilent(spies)
    })

    it('reports "no-device" when an adapter exists but the device request fails', async () => {
        mocks.initBehaviour = 'throw'
        ;(navigator as {gpu?: unknown}).gpu = {requestAdapter: async () => ({})}
        const spies = silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)

        await r.initialize({canvas: makeCanvas()})
        await settle()

        expect(onUnavailable).toHaveBeenCalledWith('no-device', expect.anything())
        expectSilent(spies)
    })
})

describe('uncaptured device errors', () => {
    it('cancels the event so the browser does not print "Uncaptured WebGPU error"', async () => {
        const device = makeDevice()
        mocks.device = device
        silenceGuard()
        const r = shaderRendererGPU()
        await r.initialize({canvas: makeCanvas()})

        // dispatchEvent returns false only when a listener called preventDefault().
        const notCanceled = device.dispatch(new Error('validation: something'))
        expect(notCanceled).toBe(false)
        r.cleanup()
    })

    it('shuts down immediately on out-of-memory (the reported hard-freeze path)', async () => {
        const device = makeDevice()
        mocks.device = device
        const spies = silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)
        await r.initialize({canvas: makeCanvas()})

        // jsdom has no GPUOutOfMemoryError; stand one in so `instanceof` resolves the same
        // way it does in a browser.
        class FakeOOM extends Error {}
        ;(globalThis as {GPUOutOfMemoryError?: unknown}).GPUOutOfMemoryError = FakeOOM
        try {
            device.dispatch(new FakeOOM('Not enough memory left.'))
            await settle()
            expect(onUnavailable).toHaveBeenCalledWith('out-of-memory', expect.anything())
            expect(r.getFailureReason()).toBe('out-of-memory')
            expectSilent(spies)
        } finally {
            delete (globalThis as {GPUOutOfMemoryError?: unknown}).GPUOutOfMemoryError
        }
    })

    it('tolerates a handful of validation errors but gives up on a sustained flood', async () => {
        const device = makeDevice()
        mocks.device = device
        silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)
        await r.initialize({canvas: makeCanvas()})

        for (let i = 0; i < 8; i++) device.dispatch(new Error('validation error'))
        await settle()
        expect(onUnavailable).not.toHaveBeenCalled() // a transient hiccup must not blank a page

        for (let i = 0; i < 40; i++) device.dispatch(new Error('validation error'))
        await settle()
        expect(onUnavailable).toHaveBeenCalledWith('gpu-error', expect.anything())
    })

    it('stops listening after cleanup', async () => {
        const device = makeDevice()
        mocks.device = device
        silenceGuard()
        const r = shaderRendererGPU()
        await r.initialize({canvas: makeCanvas()})
        r.cleanup()

        expect(device.dispatch(new Error('late error'))).toBe(true) // nobody canceled it
        expect(r.getFailureReason()).toBeNull()
    })
})

describe('page-level circuit breaker', () => {
    /**
     * A page with several shaders must not have each of them independently rediscover that
     * the GPU is a dead end — on the memory failures every extra attempt deepens the pressure
     * that froze the tab in the first place.
     */
    it('a second renderer adopts the first one\'s page-wide verdict without touching the GPU', async () => {
        delete (navigator as {gpu?: unknown}).gpu
        const spies = silenceGuard()

        const first = shaderRendererGPU()
        await first.initialize({canvas: makeCanvas()})
        expect(first.getFailureReason()).toBe('unsupported')

        // Put WebGPU "back" — the second renderer must still stand down, because the verdict
        // is remembered for the page rather than re-derived per instance.
        ;(navigator as {gpu?: unknown}).gpu = {requestAdapter: async () => ({})}
        const second = shaderRendererGPU()
        const onUnavailable = vi.fn()
        second.setOnUnavailable(onUnavailable)
        await second.initialize({canvas: makeCanvas()})
        await settle()

        expect(second.getFailureReason()).toBe('unsupported')
        expect(onUnavailable.mock.calls[0][0]).toBe('unsupported')
        expectSilent(spies)
    })

    it('does NOT latch page-wide for a renderer-specific failure', async () => {
        mocks.device = makeDevice()
        silenceGuard()

        const first = shaderRendererGPU()
        await first.initialize({canvas: makeCanvas()})
        const exploding = {
            name: 'Exploding',
            props: {},
            fragment: () => {
                throw new Error('this one shader cannot compile here')
            },
        }
        first.registerNode('root', exploding.fragment as never, null, null, {}, exploding as never)
        for (let i = 0; i < 10; i++) await first.renderAndWait()
        expect(first.getFailureReason()).toBe('render-failed')

        // A sibling shader with a perfectly ordinary pipeline must still get to start.
        mocks.device = makeDevice()
        const second = shaderRendererGPU()
        await second.initialize({canvas: makeCanvas()})
        expect(second.getFailureReason()).toBeNull()
        expect(second.isInitialized()).toBe(true)
        second.cleanup()
    })
})

describe('shared default device lifecycle', () => {
    /**
     * Every renderer on the page now shares ONE device that outlives them all. That makes
     * unsubscribing on cleanup load-bearing: without it an SPA navigating repeatedly would
     * leave a device-loss subscriber per mount, and a real loss would then fire callbacks
     * against renderers that were torn down long ago.
     */
    it('a torn-down renderer stops hearing about device loss; a live one still does', async () => {
        const device = makeDevice()
        mocks.device = device
        silenceGuard()

        const unmounted = shaderRendererGPU()
        const unmountedSaw = vi.fn()
        unmounted.setOnUnavailable(unmountedSaw)
        await unmounted.initialize({canvas: makeCanvas()})
        unmounted.cleanup()

        const live = shaderRendererGPU()
        const liveSaw = vi.fn()
        live.setOnUnavailable(liveSaw)
        await live.initialize({canvas: makeCanvas()})

        // Both renderers were built on the SAME device — that's the whole point.
        expect(live.getInternalRenderer()?.device).toBe(device)

        device.lose('unknown')
        await settle()

        expect(liveSaw).toHaveBeenCalledWith('device-lost', expect.anything())
        expect(unmountedSaw).not.toHaveBeenCalled()
    })

    it('mounts sequentially onto one device — no per-renderer device request', async () => {
        mocks.device = makeDevice()
        silenceGuard()

        const first = shaderRendererGPU()
        await first.initialize({canvas: makeCanvas()})
        const firstDevice = first.getInternalRenderer()?.device
        first.cleanup() // route change: component unmounts

        const second = shaderRendererGPU()
        await second.initialize({canvas: makeCanvas()})

        // The device survived the unmount, so the browser's pipeline cache survives with it.
        expect(second.getInternalRenderer()?.device).toBe(firstDevice)
        second.cleanup()
    })
})

describe('frames that throw', () => {
    /**
     * The composition BUILD (uniform-store finalize → WGSL codegen → pipeline creation) runs
     * inside the frame sequence, so on a driver that can't complete it, it throws on every
     * single frame. Unguarded that escapes the animation-frame callback as an uncaught error —
     * one console entry per frame, forever, which is itself enough to lock up a tab.
     *
     * The stub root here has no `createBuffer`, so the build throws exactly where a real
     * incomplete/failing GPU backend would. What we assert is the POLICY: swallowed, counted,
     * and turned into one clean shutdown.
     */
    it('gives up after repeated build failures instead of logging once per frame', async () => {
        mocks.device = makeDevice()
        const spies = silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)
        await r.initialize({canvas: makeCanvas()})
        mountOneNode(r)

        // Well past MAX_CONSECUTIVE_RENDER_ERRORS, and not one of these may reach the console.
        for (let i = 0; i < 10; i++) await r.renderAndWait()
        await settle()

        expect(onUnavailable).toHaveBeenCalledWith('render-failed', expect.anything())
        expect(r.getFailureReason()).toBe('render-failed')
        expectSilent(spies)
    })

    /**
     * Every node's props share ONE uniform binding, so a composition's total prop footprint is
     * capped by `maxUniformBufferBindingSize`. Over that ceiling nothing can be bound at all,
     * and the raw symptom is a pile of validation errors behind a blank canvas that names no
     * cause. The preflight in `uniformStore.finalize` turns it into a precise verdict.
     *
     * A verdict is not a flaky frame: it must stop on the FIRST frame (retrying can only
     * produce the same answer) and keep its own reason rather than flattening to
     * 'render-failed'. Driven here by a device that reports an absurdly small cap.
     */
    it('reports limit-exceeded on the first frame when the uniform binding cap is blown', async () => {
        mocks.device = makeDevice({maxUniformBufferBindingSize: 16})
        const spies = silenceGuard()
        const r = shaderRendererGPU()
        const onUnavailable = vi.fn()
        r.setOnUnavailable(onUnavailable)
        await r.initialize({canvas: makeCanvas()})
        mountOneNode(r)

        await r.renderAndWait() // ONE frame — a definite verdict must not be retried
        await settle()

        expect(r.getFailureReason()).toBe('limit-exceeded')
        expect(onUnavailable).toHaveBeenCalledWith('limit-exceeded', expect.anything())
        expectSilent(spies)
    })

    it('keeps limit-exceeded out of the page-wide latch — it is about the composition, not the GPU', async () => {
        mocks.device = makeDevice({maxUniformBufferBindingSize: 16})
        silenceGuard()
        const tooBig = shaderRendererGPU()
        await tooBig.initialize({canvas: makeCanvas()})
        mountOneNode(tooBig)
        await tooBig.renderAndWait()
        expect(tooBig.getFailureReason()).toBe('limit-exceeded')

        // A smaller composition on the same page must still be allowed to start.
        mocks.device = makeDevice()
        const sibling = shaderRendererGPU()
        await sibling.initialize({canvas: makeCanvas()})
        expect(sibling.getFailureReason()).toBeNull()
        expect(sibling.isInitialized()).toBe(true)
        sibling.cleanup()
    })
})

