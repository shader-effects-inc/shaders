import {describe, it, expect} from 'vitest'
import {shaderRendererGPU} from '@coreroot/gpu/index'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'

/**
 * B5b shaderRendererGPU tests — the runtime SHELL, GPU-free (no device). These exercise the
 * registration queue, parent→children integrity + renderOrder defaulting, updateUniformValue's
 * transform/compileTime recompose scheduling, the structural-hash trigger set, and the §2.3
 * public-API surface (incl. getRendererType==='webgpu' + the initialize({context}) rejection).
 * The full compose/render path needs a real GPU device and is not covered here.
 */

// A stub GPU shader definition — its `fragment` builder is never invoked in these tests (no
// composition is built without a device), so a no-op is fine.
function def(name: string): GpuShaderDefinition {
    return {name, props: {}, fragment: (() => ({})) as never}
}
const Root = def('Root')
const Gen = def('Generator')

// Minimal metadata (registerNode fills defaults). renderOrder is intentionally omitted unless
// specified, so registerNode's "max sibling + 1" defaulting is exercised.
const meta = (m: Partial<NodeMetadata> = {}): NodeMetadata =>
    ({blendMode: 'normal', opacity: undefined, ...m}) as NodeMetadata

describe('public API surface (§2.3)', () => {
    const r = shaderRendererGPU()
    const methods = [
        'initialize',
        'cleanup',
        'registerNode',
        'removeNode',
        'updateUniformValue',
        'updateNodeMetadata',
        'isInitialized',
        'resize',
        'startAnimation',
        'stopAnimation',
        'renderAndWait',
        'renderSyntheticFrame',
        'getPerformanceStats',
        'getLiveDriverValue',
        'getLiveScalarValue',
        'getNodeRegistry',
        'getRendererType',
        'getInternalRenderer',
        'setForceFullFrameRate',
        'setTimeOrigin',
        'setFrameLocked',
        'setOnReady',
        'setOnDeviceLost',
        'beginRecordingResolution',
        'setResolutionScale',
    ] as const

    for (const m of methods) {
        it(`exposes ${m}()`, () => {
            expect(typeof (r as Record<string, unknown>)[m]).toBe('function')
        })
    }

    it('getRendererType() is always "webgpu"', () => {
        expect(r.getRendererType()).toBe('webgpu')
    })
    it('getInternalRenderer() is null before init', () => {
        expect(r.getInternalRenderer()).toBeNull()
    })
    it('isInitialized() is false before init', () => {
        expect(r.isInitialized()).toBe(false)
    })
    it('exposes __testing helpers', () => {
        expect(typeof (r as never as {__testing: unknown}).__testing).toBe('object')
    })
})

describe('initialize rejects a WebGL context (WebGPU-only)', () => {
    it('throws a descriptive error when passed `context`', async () => {
        const r = shaderRendererGPU()
        const canvas = document.createElement('canvas')
        await expect(r.initialize({canvas, context: {} as never})).rejects.toThrow(/WebGPU-only/)
    })
})

describe('registration queue (pre-init)', () => {
    it('queues registrations while the renderer is not ready', () => {
        const r = shaderRendererGPU()
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta(), {}, Gen)
        expect(r.__testing.getPendingRegistrations().map((q) => q.id)).toEqual(['root', 'g1'])
        // Nothing entered the live registry yet.
        expect(r.__testing.getNodeRegistry().nodes.size).toBe(0)
    })

    it('a duplicate id updates the queued entry in place (no duplicate)', () => {
        const r = shaderRendererGPU()
        r.registerNode('g1', Gen.fragment, 'root', meta({renderOrder: 0}), {}, Gen)
        r.registerNode('g1', Gen.fragment, 'root', meta({renderOrder: 5}), {}, Gen)
        const q = r.__testing.getPendingRegistrations()
        expect(q).toHaveLength(1)
        expect(q[0].metadata?.renderOrder).toBe(5)
    })

    it('a null-fragment removal drops a queued registration', () => {
        const r = shaderRendererGPU()
        r.registerNode('g1', Gen.fragment, 'root', meta(), {}, Gen)
        expect(r.__testing.getPendingRegistrations()).toHaveLength(1)
        r.registerNode('g1', null, 'root', null, {}, Gen)
        expect(r.__testing.getPendingRegistrations()).toHaveLength(0)
    })

    it('processes the queue (in order) once made ready', () => {
        const r = shaderRendererGPU()
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta(), {}, Gen)
        r.__testing.setTestReady({width: 100, height: 100})
        expect(r.__testing.getPendingRegistrations()).toHaveLength(0)
        expect([...r.__testing.getNodeRegistry().nodes.keys()]).toEqual(['root', 'g1'])
    })
})

describe('parent→children index + renderOrder defaulting', () => {
    it('maintains parentToChildren and defaults renderOrder to max sibling + 1', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('a', Gen.fragment, 'root', meta(), {}, Gen) // no renderOrder → 0
        r.registerNode('b', Gen.fragment, 'root', meta(), {}, Gen) // → max(0)+1 = 1
        r.registerNode('c', Gen.fragment, 'root', meta(), {}, Gen) // → max(1)+1 = 2

        const nodes = r.__testing.getNodeRegistry().nodes
        expect(nodes.get('a')!.metadata.renderOrder).toBe(0)
        expect(nodes.get('b')!.metadata.renderOrder).toBe(1)
        expect(nodes.get('c')!.metadata.renderOrder).toBe(2)

        const p2c = r.__testing.getParentToChildren()
        expect([...(p2c.get('root') ?? [])].sort()).toEqual(['a', 'b', 'c'])
        expect([...(p2c.get(null) ?? [])]).toEqual(['root'])
    })

    it('an explicit renderOrder is respected', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('a', Gen.fragment, 'root', meta({renderOrder: 7}), {}, Gen)
        expect(r.__testing.getNodeRegistry().nodes.get('a')!.metadata.renderOrder).toBe(7)
    })

    it('removeNode prunes the node, its subtree, and the parent index', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('a', Gen.fragment, 'root', meta(), {}, Gen)
        r.registerNode('a-child', Gen.fragment, 'a', meta(), {}, Gen)
        r.removeNode('a')
        const nodes = r.__testing.getNodeRegistry().nodes
        expect(nodes.has('a')).toBe(false)
        expect(nodes.has('a-child')).toBe(false) // subtree removed
        expect(r.__testing.getParentToChildren().has('a')).toBe(false)
    })
})

describe('updateUniformValue — transform + compile-time recompose scheduling', () => {
    it('a value-only update writes the source of truth WITHOUT flagging recompose', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta(), {speed: {value: 1}}, Gen)
        r.__testing.clearStructuralDirty()
        r.updateUniformValue('g1', 'speed', 2)
        expect(r.__testing.isStructuralDirty()).toBe(false)
        expect(r.__testing.getNodeRegistry().nodes.get('g1')!.uniforms.speed.value).toBe(2)
    })

    it('a compileTime prop update flags a recompose', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta(), {mode: {value: 0, compileTime: true}}, Gen)
        r.__testing.clearStructuralDirty()
        r.updateUniformValue('g1', 'mode', 1)
        expect(r.__testing.isStructuralDirty()).toBe(true)
    })

    it('a compileTimeWhen prop update flags a recompose only when the predicate fires', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode(
            'g1',
            Gen.fragment,
            'root',
            meta(),
            {blur: {value: 0, compileTimeWhen: (prev, next) => (prev === 0) !== (next === 0)}},
            Gen,
        )
        r.__testing.clearStructuralDirty()
        // 0 → 3 crosses the zero boundary → recompose.
        r.updateUniformValue('g1', 'blur', 3)
        expect(r.__testing.isStructuralDirty()).toBe(true)

        r.__testing.clearStructuralDirty()
        // 3 → 5 stays non-zero → no recompose.
        r.updateUniformValue('g1', 'blur', 5)
        expect(r.__testing.isStructuralDirty()).toBe(false)
    })

    it('retains the prop transform on the node (applied by the store at build time)', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        const double = (v: unknown) => (v as number) * 2
        r.registerNode('g1', Gen.fragment, 'root', meta(), {gain: {value: 0.5, transform: double}}, Gen)
        r.updateUniformValue('g1', 'gain', 0.7)
        const u = r.__testing.getNodeRegistry().nodes.get('g1')!.uniforms.gain
        expect(u.value).toBe(0.7) // raw source of truth; transform runs in the uniform store
        expect(u.transform).toBe(double)
    })
})

describe('updateNodeMetadata + structural hash', () => {
    it('opacity changes update the source of truth; blendMode changes flip the structural hash', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta({blendMode: 'normal'}), {}, Gen)

        const h1 = r.__testing.computeStructuralHash()
        expect(typeof h1).toBe('string')

        r.updateNodeMetadata('g1', {blendMode: 'multiply'})
        const h2 = r.__testing.computeStructuralHash()
        expect(h2).not.toBe(h1)
        expect(r.__testing.getNodeRegistry().nodes.get('g1')!.metadata.blendMode).toBe('multiply')

        // Opacity 0.4 → 0.9 stays in the same non-zero bucket → hash unchanged.
        r.updateNodeMetadata('g1', {opacity: 0.4})
        const h3 = r.__testing.computeStructuralHash()
        r.updateNodeMetadata('g1', {opacity: 0.9})
        expect(r.__testing.computeStructuralHash()).toBe(h3)
    })

    // opacity<1 is a compile-time branch for EVERY node, not just uvRemap / acceptsUVContext ones:
    // composeSiblings emits no blend at all for a first child sitting at exactly 1, so that
    // composition never reads `_opacity`. If leaving 1 neither flags a recompose nor moves the hash,
    // the cached opacity-1 composition is reused and the slider does nothing.
    it('leaving opacity 1 recomposes and moves the hash for a plain (non-analytic) node', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta({opacity: 1}), {}, Gen)

        const atOne = r.__testing.computeStructuralHash()
        r.__testing.clearStructuralDirty()

        r.updateNodeMetadata('g1', {opacity: 0.5})
        expect(r.__testing.isStructuralDirty()).toBe(true)
        const partial = r.__testing.computeStructuralHash()
        expect(partial).not.toBe(atOne)

        // Back to 1 → crosses again, and lands on the original hash (a cache hit, not a rebuild).
        r.__testing.clearStructuralDirty()
        r.updateNodeMetadata('g1', {opacity: 1})
        expect(r.__testing.isStructuralDirty()).toBe(true)
        expect(r.__testing.computeStructuralHash()).toBe(atOne)

        // A same-bucket edit stays a pure uniform patch.
        r.updateNodeMetadata('g1', {opacity: 0.5})
        r.__testing.clearStructuralDirty()
        r.updateNodeMetadata('g1', {opacity: 0.25})
        expect(r.__testing.isStructuralDirty()).toBe(false)
    })

    it('computeStructuralHash returns null with no root node', () => {
        const r = shaderRendererGPU()
        expect(r.__testing.computeStructuralHash()).toBeNull()
    })
})

// Hosts mount their whole component tree before initialize() resolves a GPU device, so every
// node spends its first frames in the registration queue. A prop that changes in that window
// (a `visible` flag flipped from a mount effect / media query, a store hydrating) used to hit
// `nodes.get(id) === undefined` and be dropped for good, leaving the node stuck at its
// mount-time snapshot for the life of the page.
describe('updates arriving while a node is still queued', () => {
    it('replays metadata and uniform updates against the node once the queue flushes', () => {
        const r = shaderRendererGPU()
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta({visible: false}), {gain: {value: 1}}, Gen)

        r.updateNodeMetadata('g1', {visible: true})
        r.updateNodeMetadata('g1', {blendMode: 'multiply'})
        r.updateUniformValue('g1', 'gain', 0.25)

        r.__testing.setTestReady({width: 100, height: 100})

        const n = r.__testing.getNodeRegistry().nodes.get('g1')!
        expect(n.metadata.visible).toBe(true)
        expect(n.metadata.blendMode).toBe('multiply')
        expect(n.uniforms.gain.value).toBe(0.25)
    })

    it('drops parked updates for a node that unregisters before the flush', () => {
        const r = shaderRendererGPU()
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta({visible: false}), {}, Gen)
        r.updateNodeMetadata('g1', {visible: true})
        r.registerNode('g1', null, null, null, null as never)

        r.__testing.setTestReady({width: 100, height: 100})
        expect(r.__testing.getNodeRegistry().nodes.has('g1')).toBe(false)
    })

    it('drops parked updates when the node re-registers with a fresh snapshot before the flush', () => {
        const r = shaderRendererGPU()
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta({visible: false}), {}, Gen)
        r.updateNodeMetadata('g1', {visible: true})
        // Host re-registers g1 with current state — older parked updates are stale.
        r.registerNode('g1', Gen.fragment, 'root', meta({visible: false}), {}, Gen)

        r.__testing.setTestReady({width: 100, height: 100})
        expect(r.__testing.getNodeRegistry().nodes.get('g1')?.metadata.visible).toBe(false)
    })

    it('ignores an update for an id that was never registered', () => {
        const r = shaderRendererGPU()
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        expect(() => r.updateNodeMetadata('ghost', {visible: true})).not.toThrow()
        r.__testing.setTestReady({width: 100, height: 100})
        expect(r.__testing.getNodeRegistry().nodes.has('ghost')).toBe(false)
    })
})

describe('live driver readback (null when no driver)', () => {
    it('getLiveDriverValue / getLiveScalarValue return null for an undriven prop', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta(), {speed: {value: 1}}, Gen)
        expect(r.getLiveDriverValue('g1', 'speed')).toBeNull()
        expect(r.getLiveScalarValue('g1', 'speed')).toBeNull()
        expect(r.getLiveDriverValue('missing', 'x')).toBeNull()
    })
})

describe('driver state re-seed (cache-hit rebind path)', () => {
    // The design editor remounts every component on a structural edit, so each node re-registers
    // as a brand-new object with EMPTY driver state. When the resulting structural hash was
    // already built (undo/redo, removing a just-added layer), bindComposition reuses the cached
    // composition and buildFieldInits never runs — ensureDriverState is what re-seeds the state
    // there. Regression: dynamic props froze after add-then-remove until the next cache miss.
    const driven = () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta({
            maps: {
                speed: {type: 'mouse', axis: 'x', outputMin: 0, outputMax: 4},
                scale: {type: 'auto-animate', outputMin: 1, outputMax: 2},
                radius: {type: 'map', source: 'self', channel: 'alpha', inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 10},
                center: {type: 'mouse-position'},
            },
        } as never), {speed: {value: 1}, scale: {value: 1}, radius: {value: 0.5}, center: {value: {x: 0.5, y: 0.5}}}, Gen)
        return r
    }

    it('a freshly registered node carries empty driver state until seeded', () => {
        const r = driven()
        const node = r.__testing.getNodeRegistry().nodes.get('g1')!
        expect(Object.keys(node.mouseDriverState)).toHaveLength(0)
        expect(Object.keys(node.autoAnimateState)).toHaveLength(0)
        expect(Object.keys(node.mapValues)).toHaveLength(0)
    })

    it('ensureDriverState populates every driver type from metadata.maps', () => {
        const r = driven()
        r.__testing.ensureDriverState('g1')
        const node = r.__testing.getNodeRegistry().nodes.get('g1')!
        expect(Object.keys(node.mouseDriverState).sort()).toEqual(['center', 'speed'])
        expect(Object.keys(node.autoAnimateState)).toEqual(['scale'])
        expect(Object.keys(node.mapValues).sort()).toEqual(['radius', 'scale', 'speed'])
        expect(node.mapValues.radius).toMatchObject({inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 10})
        expect(node.mapValues.speed).toMatchObject({outputMin: 0, outputMax: 4})
    })

    it('is idempotent — re-seeding refreshes bounds without resetting spring state', () => {
        const r = driven()
        r.__testing.ensureDriverState('g1')
        const node = r.__testing.getNodeRegistry().nodes.get('g1')!
        node.mouseDriverState.speed.currentX = 0.42
        r.__testing.ensureDriverState('g1')
        expect(node.mouseDriverState.speed.currentX).toBe(0.42)
    })

    it('the CPU driver step produces live values once state exists (frozen before)', () => {
        const r = driven()
        // Before seeding: nothing to iterate — the frozen-dynamic-props symptom.
        r.__testing.stepMouseDrivers(0.016)
        expect(r.getLiveScalarValue('g1', 'speed')).toBeNull()
        r.__testing.ensureDriverState('g1')
        r.__testing.stepMouseDrivers(0.016)
        expect(r.getLiveScalarValue('g1', 'speed')).not.toBeNull()
        expect(r.getLiveDriverValue('g1', 'center')).not.toBeNull()
    })

    it('a maps-less node is a no-op', () => {
        const r = shaderRendererGPU()
        r.__testing.setTestReady({width: 100, height: 100})
        r.registerNode('root', Root.fragment, null, meta(), {}, Root)
        r.registerNode('g1', Gen.fragment, 'root', meta(), {speed: {value: 1}}, Gen)
        expect(() => r.__testing.ensureDriverState('g1')).not.toThrow()
        expect(Object.keys(r.__testing.getNodeRegistry().nodes.get('g1')!.mapValues)).toHaveLength(0)
    })
})

describe('setForceFullFrameRate returns the previous value', () => {
    it('toggles and reports the prior state', () => {
        const r = shaderRendererGPU()
        expect(r.setForceFullFrameRate(true)).toBe(false)
        expect(r.setForceFullFrameRate(false)).toBe(true)
    })
})
