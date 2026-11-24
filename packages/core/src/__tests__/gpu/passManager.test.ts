import {describe, it, expect, vi, beforeEach} from 'vitest'
import {createPassManager} from '@coreroot/gpu/passManager'
import {setShadersDebug} from '@coreroot/gpu/support'
import {SHARED_SAMPLER_LAYOUT} from '@coreroot/gpu/composer'
import type {CompositionIR, FragmentSpec} from '@coreroot/gpu/composer'

/**
 * B5a passManager tests. GPU-free: `root` / `textureManager` / `dispatcher` are mocked (as in
 * uniformStore/compute tests). We assert:
 *   - IR execution ORDER (compute → RTT passes → final pass), recorded by mock pipelines,
 *   - RTT texture lifecycle (create / resize / dispose counts, textureCount),
 *   - the sampled-texture-limit assertion path (warn, not throw).
 */

// A shared call log so we can assert cross-resource ordering.
let calls: string[] = []

function mockPipeline(label: string) {
    const p: {
        with: () => typeof p
        withColorAttachment: (a: {view: {tag: string}}) => typeof p
        draw: () => void
    } = {
        with: () => p,
        withColorAttachment: (a) => {
            ;(p as unknown as {_view: string})._view = a.view.tag
            return p
        },
        draw: () => calls.push(`draw:${label}:${(p as unknown as {_view: string})._view}`),
    }
    return p
}

function mockRoot() {
    return {
        createBindGroup: vi.fn(() => ({tag: 'bindGroup'})),
        createRenderPipeline: vi.fn((desc: {fragment: {label: string}}) => mockPipeline(desc.fragment.label)),
        unwrap: vi.fn(() => ({createView: () => ({tag: 'rttView'})})),
    } as never
}

interface MockRenderTexture {
    texture: {tag: string}
    width: number
    height: number
    resize: ReturnType<typeof vi.fn>
    destroy: ReturnType<typeof vi.fn>
}

function mockTextureManager() {
    const created: MockRenderTexture[] = []
    const samplers = {linearClamp: {}, nearestClamp: {}, linearRepeat: {}, nearestRepeat: {}}
    const manager = {
        samplers,
        createRenderTexture: vi.fn((opts: {width: number; height: number; label?: string}) => {
            const baseTag = opts.label ?? 'tex'
            let gen = 0
            const rt: MockRenderTexture = {
                texture: {tag: baseTag},
                width: opts.width,
                height: opts.height,
                resize: vi.fn(function (this: MockRenderTexture, w: number, h: number) {
                    if (w === this.width && h === this.height) return false
                    this.width = w
                    this.height = h
                    // The real RenderTexture.resize destroys + recreates the backing GPU texture,
                    // giving it a NEW identity. Mirror that here so tests can catch a bind group
                    // that still references the pre-resize (now destroyed) texture.
                    this.texture = {tag: `${baseTag}#${++gen}`}
                    return true
                }),
                destroy: vi.fn(),
            }
            created.push(rt)
            return rt
        }),
        createMediaTexture: vi.fn(),
        createDataTexture: vi.fn(),
        createExternalTextureBinding: vi.fn(),
        get textureCount() {
            return created.filter((t) => !t.destroy.mock.calls.length).length
        },
        destroy: vi.fn(),
    }
    return {manager, created}
}

function mockDispatcher() {
    return {dispatch: vi.fn((steps: unknown[]) => calls.push(`compute:${steps.length}`))}
}

function frag(label: string, reads: string[]): FragmentSpec {
    return {
        body: '',
        externals: {},
        entry: {label} as never,
        reads,
        externalReads: [],
        usesSamplers: reads.length > 0,
        textureLayout: reads.length > 0 ? ({entries: reads} as never) : undefined,
    }
}

function makeIR(overrides: Partial<CompositionIR> = {}): CompositionIR {
    return {
        computeSteps: [],
        rttPasses: [{textureKey: 'rtt_0', fragment: frag('rtt0', [])}],
        finalPass: frag('final', ['rtt_0']),
        textures: [{key: 'rtt_0', kind: 'rtt'}],
        externalTextures: [],
        layouts: {uniforms: undefined, samplers: SHARED_SAMPLER_LAYOUT as never},
        onBeforeRender: [],
        onAfterRender: [],
        onResize: [],
        onCleanup: [],
        composedNodeIds: new Set(),
        ...overrides,
    }
}

beforeEach(() => {
    calls = []
})

describe('passManager — execution order', () => {
    it('dispatches compute, then RTT passes, then the final pass', () => {
        const root = mockRoot()
        const {manager} = mockTextureManager()
        const dispatcher = mockDispatcher()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher})

        const ir = makeIR({
            computeSteps: [{nodeId: 'c1', getComputeNodes: () => [(() => {}) as never]}],
        })
        pm.setComposition(ir, {tag: 'uniformBG'} as never, {width: 100, height: 80})
        pm.render({tag: 'canvas'}, {})

        expect(calls).toEqual(['compute:1', 'draw:rtt0:rttView', 'draw:final:canvas'])
    })

    it('skips compute steps that return null this frame', () => {
        const root = mockRoot()
        const {manager} = mockTextureManager()
        const dispatcher = mockDispatcher()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher})
        const ir = makeIR({computeSteps: [{nodeId: 'c1', getComputeNodes: () => null}]})
        pm.setComposition(ir, undefined, {width: 10, height: 10})
        pm.render({tag: 'canvas'}, {})
        expect(dispatcher.dispatch).not.toHaveBeenCalled()
        expect(calls).toEqual(['draw:rtt0:rttView', 'draw:final:canvas'])
    })

    it('repaintRtt re-executes only the RTT passes — no compute, no final pass', () => {
        // Post-resize warm-up: the renderer repaints the recreated (zero-initialized) RTT textures
        // so a compute step that reads a child RTT doesn't read black for one frame. Compute must
        // NOT dispatch here (a ping-pong simulation would double-step on resize frames).
        const root = mockRoot()
        const {manager} = mockTextureManager()
        const dispatcher = mockDispatcher()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher})
        const ir = makeIR({
            computeSteps: [{nodeId: 'c1', getComputeNodes: () => [(() => {}) as never]}],
        })
        pm.setComposition(ir, undefined, {width: 100, height: 80})
        pm.repaintRtt()
        expect(dispatcher.dispatch).not.toHaveBeenCalled()
        expect(calls).toEqual(['draw:rtt0:rttView'])
    })
})

describe('passManager — RTT texture lifecycle', () => {
    it('creates one texture per RTT boundary and tracks the live count', () => {
        const root = mockRoot()
        const {manager, created} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        const ir = makeIR({
            rttPasses: [
                {textureKey: 'rtt_0', fragment: frag('r0', [])},
                {textureKey: 'rtt_1', fragment: frag('r1', ['rtt_0'])},
            ],
            textures: [
                {key: 'rtt_0', kind: 'rtt'},
                {key: 'rtt_1', kind: 'rtt'},
            ],
        })
        pm.setComposition(ir, undefined, {width: 64, height: 64})
        expect(created).toHaveLength(2)
        expect(pm.textureCount).toBe(2)
        expect(pm.passCount).toBe(3) // 2 RTT + 1 final
    })

    it('resizes existing RTT textures in place (no realloc)', () => {
        const root = mockRoot()
        const {manager, created} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR(), undefined, {width: 32, height: 32})
        expect(pm.resize(200, 150)).toBe(true) // signals "textures reallocated" → caller repaints
        expect(created[0].resize).toHaveBeenCalledWith(200, 150)
        expect(created).toHaveLength(1) // not recreated
        // Same size again: nothing reallocated, so the caller can skip the warm-up repaint.
        expect(pm.resize(200, 150)).toBe(false)
    })

    it('rebuilds pass bind groups against the recreated textures after resize', () => {
        // Regression: `resize` destroys + recreates each RTT texture (new identity). If the pass
        // bind groups aren't rebuilt, the next submit binds a DESTROYED texture — WebGPU's
        // "Destroyed texture used in a submit" validation error (a blank frame). The final pass
        // here reads rtt_0, so its group-1 bind group must be rebuilt to reference the new texture.
        const root = mockRoot()
        const {manager, created} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR(), undefined, {width: 32, height: 32})

        const createBindGroup = root.createBindGroup as unknown as ReturnType<typeof vi.fn>
        // Count how many bind groups were built binding `tex` to the `rtt_0` entry.
        const boundTo = (tex: unknown): number =>
            createBindGroup.mock.calls.filter(([, entries]) => (entries as {rtt_0?: unknown})?.rtt_0 === tex).length

        const before = created[0].texture
        expect(boundTo(before)).toBeGreaterThan(0)

        pm.resize(200, 150)
        const after = created[0].texture
        expect(after).not.toBe(before) // destroy + recreate → new identity
        expect(boundTo(after)).toBeGreaterThan(0) // final pass rebound to the live texture
        expect(boundTo(before)).toBe(1) // no NEW bind group still references the destroyed one

        // The final pass still draws after resize (it has a valid bind group).
        pm.render({tag: 'canvas'}, {})
        expect(calls).toContain('draw:final:canvas')
    })

    it('disposes RTT textures the next composition no longer uses', () => {
        const root = mockRoot()
        const {manager, created} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(
            makeIR({
                rttPasses: [
                    {textureKey: 'rtt_0', fragment: frag('r0', [])},
                    {textureKey: 'rtt_1', fragment: frag('r1', ['rtt_0'])},
                ],
                textures: [
                    {key: 'rtt_0', kind: 'rtt'},
                    {key: 'rtt_1', kind: 'rtt'},
                ],
            }),
            undefined,
            {width: 32, height: 32},
        )
        // New composition uses only rtt_0.
        pm.setComposition(makeIR(), undefined, {width: 32, height: 32})
        const rtt1 = created[1]
        expect(rtt1.destroy).toHaveBeenCalled()
        expect(pm.textureCount).toBe(1)
    })

    it('dispose() destroys every live RTT texture', () => {
        const root = mockRoot()
        const {manager, created} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR(), undefined, {width: 16, height: 16})
        pm.dispose()
        expect(created[0].destroy).toHaveBeenCalled()
        expect(pm.textureCount).toBe(0)
        expect(pm.passCount).toBe(0)
    })
})

describe('passManager — texture limit assertion (log, not throw)', () => {
    it('warns when a pass samples more textures than the device ceiling', () => {
        // Library logging is silent in production (a page that can't run WebGPU must not
        // fill its host's console); the diagnostic still has to exist behind the debug flag.
        setShadersDebug(true)
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
        const root = mockRoot()
        const {manager} = mockTextureManager()
        const pm = createPassManager(root, {
            textureManager: manager as never,
            dispatcher: mockDispatcher(),
            maxSampledTextures: 2,
        })
        const ir = makeIR({
            rttPasses: [
                {textureKey: 'rtt_0', fragment: frag('r0', [])},
                {textureKey: 'rtt_1', fragment: frag('r1', [])},
                {textureKey: 'rtt_2', fragment: frag('r2', [])},
            ],
            textures: [
                {key: 'rtt_0', kind: 'rtt'},
                {key: 'rtt_1', kind: 'rtt'},
                {key: 'rtt_2', kind: 'rtt'},
            ],
            finalPass: frag('final', ['rtt_0', 'rtt_1', 'rtt_2']),
        })
        expect(() => pm.setComposition(ir, undefined, {width: 8, height: 8})).not.toThrow()
        expect(warn).toHaveBeenCalledWith(expect.stringContaining('device limit'))
        warn.mockRestore()
        setShadersDebug(null)
    })
})

describe('passManager — one submit per frame', () => {
    function rootWithDevice() {
        const encoders: Array<{label?: string}> = []
        const submits: unknown[][] = []
        const withArgs: unknown[] = []
        const pipeline = (label: string) => {
            const p: Record<string, unknown> = {}
            p.with = (x: unknown) => {
                withArgs.push(x)
                return p
            }
            p.withColorAttachment = (a: {view: {tag: string}}) => {
                p._view = a.view.tag
                return p
            }
            p.draw = () => calls.push(`draw:${label}:${p._view as string}`)
            return p
        }
        const device = {
            createCommandEncoder: vi.fn((desc: {label?: string}) => {
                const enc = {label: desc.label, beginRenderPass() {}, beginComputePass() {}, finish: () => ({buffer: desc.label})}
                encoders.push(enc)
                return enc
            }),
            queue: {submit: vi.fn((buffers: unknown[]) => submits.push(buffers))},
        }
        const root = {
            device,
            createBindGroup: vi.fn(() => ({tag: 'bindGroup'})),
            createRenderPipeline: vi.fn((desc: {fragment: {label: string}}) => pipeline(desc.fragment.label)),
            unwrap: vi.fn(() => ({createView: () => ({tag: 'rttView'})})),
        }
        return {root: root as never, encoders, submits, withArgs}
    }

    it('records the RTT and final passes into one encoder and submits once', () => {
        const {root, encoders, submits, withArgs} = rootWithDevice()
        const {manager} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR(), undefined, {width: 100, height: 80})
        pm.render({tag: 'canvas'}, {})
        expect(calls).toEqual(['draw:rtt0:rttView', 'draw:final:canvas'])
        expect(encoders).toHaveLength(1)
        expect(withArgs.filter((x) => x === encoders[0])).toHaveLength(2)
        expect(submits).toHaveLength(1)
    })

    it('runs compute (and the afterCompute flush) before the frame encoder exists', () => {
        const {root, encoders} = rootWithDevice()
        const {manager} = mockTextureManager()
        const pm = createPassManager(root, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR({computeSteps: [{nodeId: 'c1', getComputeNodes: () => [(() => {}) as never]}]}), undefined, {width: 10, height: 10})
        let encodersAtFlush = -1
        pm.render({tag: 'canvas'}, {}, () => {
            encodersAtFlush = encoders.length
        })
        expect(encodersAtFlush).toBe(0)
        expect(calls[0]).toBe('compute:1')
    })
})

describe('passManager — background compile', () => {
    it('compiles every pass pipeline and waits for all of them before reporting a failure', async () => {
        const settled: string[] = []
        let rejectFinal: (e: Error) => void = () => {}
        let resolveRtt: () => void = () => {}
        const root = {
            createBindGroup: vi.fn(() => ({tag: 'bindGroup'})),
            createRenderPipeline: vi.fn((desc: {fragment: {label: string}}) => {
                const p = mockPipeline(desc.fragment.label) as Record<string, unknown>
                p.initAsync = () => new Promise<void>((resolve, reject) => {
                    if (desc.fragment.label === 'final') rejectFinal = (e) => { settled.push('final'); reject(e) }
                    else resolveRtt = () => { settled.push('rtt0'); resolve() }
                })
                return p
            }),
            unwrap: vi.fn(() => ({createView: () => ({tag: 'rttView'})})),
        }
        const {manager} = mockTextureManager()
        const pm = createPassManager(root as never, {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR(), undefined, {width: 10, height: 10})
        const compiling = pm.compile()
        expect(compiling).not.toBeNull()
        let failure: unknown = null
        const done = compiling!.catch((e) => {
            failure = e
            settled.push('reported')
        })
        rejectFinal(new Error('bad wgsl'))
        await Promise.resolve()
        resolveRtt()
        await done
        expect(settled).toEqual(['final', 'rtt0', 'reported'])
        expect((failure as Error).message).toBe('bad wgsl')
    })

    it('has nothing to wait for when pipelines cannot compile in the background', () => {
        const {manager} = mockTextureManager()
        const pm = createPassManager(mockRoot(), {textureManager: manager as never, dispatcher: mockDispatcher()})
        pm.setComposition(makeIR(), undefined, {width: 10, height: 10})
        expect(pm.compile()).toBeNull()
    })
})
