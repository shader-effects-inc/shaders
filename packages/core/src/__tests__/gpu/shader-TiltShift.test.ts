import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type UniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import TiltShift from '@coreroot/shaders/TiltShift/index'
import {buildTiltShiftFillGraph, buildTiltShiftFillMapGraph, tiltShiftBlurAmount, intensityToRadius, INTENSITY_TO_RADIUS, VARIABLE_BLUR_DISPERSION_SPREAD} from '@coreroot/gpu/kit/blur'

/**
 * TiltShift port gate (Phase D3-A). GPU-free: a mock root answers the compute allocations so the
 * compute↔RTT↔fragment wiring runs inside `composeNodeTree` — a variable Gaussian whose per-pixel
 * radius map is FILLED by a focus-line-distance kernel (`tiltShiftFillBlurMap`), then the fragment
 * recomputes the same blur amount and mixes the sharp source with the blurred buffer (crisp focus).
 * Plus the fill kernel resolves to WGSL (D3 rules) and a CPU golden on the shared blur-amount body.
 */

function mockRoot() {
    const buffer = {patch: vi.fn(), write: vi.fn(), destroy: vi.fn(), $usage: vi.fn(function (this: unknown) {return buffer})}
    const texture = {$usage: vi.fn(function (this: unknown) {return texture}), destroy: vi.fn(), write: vi.fn()}
    const uniform = {buffer: {}, write: vi.fn(), patch: vi.fn()}
    const guarded = {with: vi.fn(function (this: unknown) {return guarded}), dispatchThreads: vi.fn()}
    return {
        createBuffer: vi.fn(() => buffer),
        createBindGroup: vi.fn(() => ({})),
        createTexture: vi.fn(() => texture),
        createUniform: vi.fn(() => uniform),
        createGuardedComputePipeline: vi.fn(() => guarded),
        device: {},
    } as never
}

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}
const RootContainer: GpuShaderDefinition = {
    name: 'Root',
    props: {} as never,
    fragment: ({childNode}: GpuFragmentParams): Expr => childNode ?? call(genBody, 'genBody', []),
}

interface NodeSpec {id: string; def: GpuShaderDefinition; parentId: string | null; props?: Record<string, unknown>; metadata?: Partial<NodeMetadata>}

function bridgeFieldInits(def: GpuShaderDefinition, props: Record<string, unknown>, id: string): FieldInit[] {
    const map = createGpuUniformsMap(def as never, props, id)
    const inits: FieldInit[] = []
    for (const [name, u] of Object.entries(map)) inits.push({name, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema})
    return inits
}
function defaultsFor(def: GpuShaderDefinition): Record<string, unknown> {
    const out: Record<string, unknown> = {}
    for (const [name, cfg] of Object.entries(def.props)) out[name] = (cfg as {default: unknown}).default
    return out
}
function buildRegistry(specs: NodeSpec[]): {registry: RegistryView; store: UniformStore; root: ReturnType<typeof mockRoot>} {
    const root = mockRoot()
    const store = createUniformStore(root, {systemSchema: SystemUniforms})
    const handlesById: Record<string, Record<string, {accessorPath: string; cpu: boolean; value: unknown}>> = {}
    for (const s of specs) {
        const props = {...defaultsFor(s.def), ...(s.props ?? {})}
        const propFields = bridgeFieldInits(s.def, props, s.id)
        const synthetic: FieldInit[] = [{name: '_opacity', schema: d.f32, initial: s.metadata?.opacity ?? 1}]
        handlesById[s.id] = store.defineNode(s.id, [...propFields, ...synthetic]) as never
    }
    store.defineSystem()
    store.finalize()
    const nodes = new Map<string, RegistryNode>()
    const childrenByParent = new Map<string, RegistryNode[]>()
    for (const s of specs) {
        nodes.set(s.id, {
            id: s.id, componentName: s.def.name, parentId: s.parentId, definition: s.def,
            metadata: {blendMode: 'normal', opacity: undefined, renderOrder: 0, ...s.metadata} as NodeMetadata,
            handles: handlesById[s.id],
        })
    }
    for (const s of specs) {
        if (s.parentId) {
            const arr = childrenByParent.get(s.parentId) ?? []
            arr.push(nodes.get(s.id)!)
            childrenByParent.set(s.parentId, arr)
        }
    }
    const rootNode = specs.find((s) => s.parentId === null)!
    const registry: RegistryView = {
        rootId: rootNode.id,
        getNode: (id) => nodes.get(id),
        getChildren: (parentId) => childrenByParent.get(parentId) ?? [],
        resolveCustomId: () => null,
        store,
    }
    return {registry, store, root}
}
const composeOpts = (root: unknown) => ({flipY: false, dimensions: {width: 800, height: 600}, gpu: {device: (root as {device: unknown}).device, root} as never})
const tree = () => [
    {id: 'root', def: RootContainer, parentId: null},
    {id: 'ts', def: TiltShift as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
    {id: 'gen', def: Generator, parentId: 'ts', metadata: {renderOrder: 0}},
] as NodeSpec[]

describe('TiltShift (a) compute → RTT input → mixed fragment composite', () => {
    it('RTTs the child, registers a compute-output texture, mixes sharp↔blurred by the focus band', () => {
        const {registry, root} = buildRegistry(tree())
        const ir = composeNodeTree(registry, composeOpts(root))

        expect(ir.computeSteps.length).toBe(1)
        expect(ir.rttPasses.length).toBe(1)
        const kinds = ir.textures.map((t) => t.kind).sort()
        expect(kinds).toContain('rtt')
        expect(kinds).toContain('compute')

        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/textureSample\(rtt_0/)
        expect(finalWgsl).toMatch(/textureSample\(compute_0/)
        expect(finalWgsl).toMatch(/tiltShiftBlurAmount/)
        expect(finalWgsl).toMatch(/mix\(/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('getComputeNodes returns the fill kernel + the two variable-blur passes (3 steps)', () => {
        const {registry, root} = buildRegistry(tree())
        const ir = composeNodeTree(registry, composeOpts(root))
        expect(ir.computeSteps[0].getComputeNodes({})?.length).toBe(3)
    })

    it('bindInputs resolves the child RTT key', () => {
        const {registry, root} = buildRegistry(tree())
        const ir = composeNodeTree(registry, composeOpts(root))
        let requestedKey: string | null = null
        ir.computeSteps[0].bindInputs?.((key) => {
            requestedKey = key
            return {texture: {}}
        })
        expect(requestedKey).toBe('rtt_0')
    })
})

describe('TiltShift (b) fragment fallback when compute is unavailable', () => {
    it('samples the child RTT sharp and unpremultiplies (no compute textures)', () => {
        const {registry} = buildRegistry(tree())
        const ir = composeNodeTree(registry, {flipY: false})
        expect(ir.computeSteps.length).toBe(0)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_0/)
        expect(finalWgsl).not.toMatch(/compute_0/)
    })
})

describe('TiltShift (c) fill kernel resolves (D3 rules)', () => {
    it('the blur-map fill kernel resolves with a storage textureStore + the shared blur-amount body', () => {
        const {layout, kernel} = buildTiltShiftFillGraph(1024, 640)
        const wgsl = tgpu.resolve([layout, kernel], {names: 'strict'})
        expect(wgsl).toMatch(/tiltShiftFillBlurMap/)
        expect(wgsl).toMatch(/tiltShiftBlurAmount/)
        expect(wgsl).toMatch(/textureStore/)
        expect(wgsl).toMatchSnapshot('fillBlurMap')
    })
})

describe('TiltShift (c2) map-driven fill kernel resolves (e0 compute-map interplay)', () => {
    it('the map-driven fill kernel samples the source texture, remaps, scales by the blur amount', () => {
        const {layout, kernel} = buildTiltShiftFillMapGraph(1024, 640, 'luminance')
        const wgsl = tgpu.resolve([layout, kernel], {names: 'strict'})
        expect(wgsl).toMatch(/tiltShiftFillBlurMapVariable/)
        expect(wgsl).toMatch(/tiltShiftBlurAmount/)
        expect(wgsl).toMatch(/textureLoad/) // reads the map source per pixel
        expect(wgsl).toMatch(/textureStore/)
        expect(wgsl).toMatchSnapshot('fillBlurMapVariable')
    })
    it('the alpha channel variant folds to sampling the source .w', () => {
        const {layout, kernel} = buildTiltShiftFillMapGraph(1024, 640, 'alpha')
        const wgsl = tgpu.resolve([layout, kernel], {names: 'strict'})
        expect(wgsl).toMatch(/textureLoad/)
        expect(wgsl).toMatch(/textureStore/)
    })
})

describe('TiltShift (c3) intensity map driver activates the variable fill path', () => {
    // Directly invoke the compute hook with a params object whose getMapInfo('intensity') resolves a
    // map driver (composer supplies this from node.metadata.maps). The hook must take the map branch:
    // fill kernel + variable H/V = 3 steps, and bindInputs must resolve BOTH the child RTT and the
    // map-source key. Mouse/auto/static intensity → getMapInfo null → the scalar path (also 3 steps
    // here, but no source key requested).
    function invokeCompute(withMap: boolean) {
        const requested: string[] = []
        const params = {
            childNode: {} as never,
            gpu: {root: mockRoot()} as never,
            dimensions: {width: 256, height: 256},
            convertToTexture: () => ({key: 'child', sample: () => ({})}) as never,
            registerComputeTexture: () => ({key: 'blurred', sample: () => ({})}) as never,
            getCpuValue: (prop: string) => (prop === 'center' ? {x: 0.5, y: 0.5} : prop === 'intensity' ? 50 : 0.3),
            getMapInfo: (prop: string) =>
                withMap && prop === 'intensity'
                    ? {
                          sourceTexture: {key: 'src'},
                          channel: 'luminance',
                          window: () => ({inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 100, curve: 0}),
                      }
                    : null,
            onCleanup: () => {},
            onResize: () => {},
        } as unknown as GpuFragmentParams
        const result = TiltShift.compute!(params) as {getComputeNodes: () => unknown[]; bindInputs?: (r: (k: string) => unknown) => void}
        result.bindInputs?.((key: string) => {
            requested.push(key)
            return {texture: {}}
        })
        return {steps: result.getComputeNodes().length, requested}
    }

    it('a `map` driver on intensity binds the map source and runs the variable fill kernel (3 steps)', () => {
        const {steps, requested} = invokeCompute(true)
        expect(steps).toBe(3) // fill kernel + variable H + variable V
        expect(new Set(requested).size).toBe(2) // child RTT + map source both resolved
        expect(requested).toContain('src')
    })

    it('no map driver → scalar path, the map source is never requested', () => {
        const {requested} = invokeCompute(false)
        expect(requested).not.toContain('src')
    })
})

describe('TiltShift (d) CPU golden — intensity→radius + focus-line blur amount', () => {
    it('intensityToRadius is the verbatim v1 factor (× 0.36)', () => {
        expect(INTENSITY_TO_RADIUS).toBe(0.36)
        expect(intensityToRadius(0)).toBe(0)
        expect(intensityToRadius(50)).toBeCloseTo(18, 10)
        expect(intensityToRadius(100)).toBeCloseTo(36, 10)
    })

    it('amount is 0 inside the focus band, ramps over falloff, saturates past it (angle 0, aspect 1)', () => {
        // center=(0.5, 0.5) authored → transformed centerY = 0.5; centerPos = (0.5, 0.5).
        // perpVector = (-sin0, cos0) = (0, 1). width 0.3 → focusWidth 0.15. falloff 0.3.
        const onLine = tiltShiftBlurAmount(0, d.vec2f(0.5, 0.5), 0.3, 0.3, d.vec2f(0.5, 0.5), 1)
        expect(onLine).toBeCloseTo(0, 6)
        // dist 0.4 → smoothstep(0.15, 0.45, 0.4) = 0.9259…
        const ramp = tiltShiftBlurAmount(0, d.vec2f(0.5, 0.5), 0.3, 0.3, d.vec2f(0.5, 0.9), 1)
        expect(ramp).toBeCloseTo(0.9259259, 5)
        // dist 0.5 > 0.45 → fully blurred
        const saturated = tiltShiftBlurAmount(0, d.vec2f(0.5, 0.5), 0.3, 0.3, d.vec2f(0.5, 1.0), 1)
        expect(saturated).toBeCloseTo(1, 6)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (d) Lens dials — dispersion gates the chromatic kernel variant at compose time (propValues),
// and both dials reach the variable blur's uniforms per frame through setDetail.
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('TiltShift (d) lens dials → variable-blur detail uniforms', () => {
    function invokeCompute(propValues: Record<string, number>) {
        const root = mockRoot() as unknown as {createUniform: {mock: {results: {value: {patch: {mock: {calls: unknown[][]}}}}[]}}}
        const params = {
            childNode: {} as never,
            gpu: {root} as never,
            dimensions: {width: 256, height: 256},
            propValues,
            convertToTexture: () => ({key: 'child', sample: () => ({})}) as never,
            registerComputeTexture: () => ({key: 'blurred', sample: () => ({})}) as never,
            getCpuValue: (prop: string) => (prop === 'center' ? {x: 0.5, y: 0.5} : prop === 'intensity' ? 50 : (propValues[prop] ?? 0.3)),
            getMapInfo: () => null,
            onCleanup: () => {},
            onResize: () => {},
        } as unknown as GpuFragmentParams
        const result = TiltShift.compute!(params) as {getComputeNodes: () => unknown[]}
        result.getComputeNodes()
        const patches = root.createUniform.mock.results.flatMap((r) => r.value.patch.mock.calls.map((c) => c[0]))
        return patches as Record<string, number>[]
    }

    it('dispersion 0.5 + jitter 0.2 → setDetail patches jitter 0.2 and the spread-scaled dispersion', () => {
        const patches = invokeCompute({dispersion: 0.5, jitter: 0.2})
        const detail = patches.find((p) => 'dispersion' in p)
        expect(detail).toEqual({jitter: 0.2, dispersion: 0.5 * VARIABLE_BLUR_DISPERSION_SPREAD})
    })

    it('dispersion 0 → no chromatic spread reaches the kernel (dispersion stays 0), jitter still pulls through', () => {
        const patches = invokeCompute({dispersion: 0, jitter: 0.4})
        const detail = patches.find((p) => 'dispersion' in p)
        expect(detail).toEqual({jitter: 0.4, dispersion: 0})
    })

    it('both dials at 0 → nothing to patch (the initial write already holds zeros)', () => {
        const patches = invokeCompute({dispersion: 0, jitter: 0})
        expect(patches.find((p) => 'dispersion' in p)).toBeUndefined()
    })
})
