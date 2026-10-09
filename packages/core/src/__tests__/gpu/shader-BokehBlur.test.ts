import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type UniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import BokehBlur from '@coreroot/shaders/BokehBlur/index'
import {
    bokehRadiusToPixels as radiusToPixels,
    BOKEH_RADIUS_SCALE as RADIUS_SCALE,
    BOKEH_TAP_COUNT as TAP_COUNT,
    BOKEH_SHAPES,
    bokehPolygonRadius,
    generateBokehTaps,
    buildBokehGraph,
    buildBokehMapGraph,
    type BokehMapChannel,
} from '@coreroot/gpu/kit/blur'

/**
 * BokehBlur port gate. GPU-free: the uniform store's root is mocked, and the mock ALSO answers the
 * compute allocations (createTexture / createUniform / createGuardedComputePipeline) so the single-pass
 * gather compute hook runs inside `composeNodeTree`, exercising the compute↔RTT↔fragment wiring:
 * `convertToTexture(childNode)` → child RTT, `registerComputeTexture(outputTexture)` →
 * a `kind:'compute'` sampleable buffer, `bindInputs` (late input binding), and the fragment that
 * samples the defocused buffer and unpremultiplies. Plus: the gather kernels resolve to WGSL, and CPU
 * goldens on the radius factor + quality tiers + polygon aperture.
 */

// ── mock root: real bindGroupLayouts, mocked device resources (enough for store + compute build) ─
function mockRoot() {
    const buffer = {
        patch: vi.fn(),
        write: vi.fn(),
        destroy: vi.fn(),
        $usage: vi.fn(function (this: unknown) {
            return buffer
        }),
    }
    const texture = {
        $usage: vi.fn(function (this: unknown) {
            return texture
        }),
        destroy: vi.fn(),
        write: vi.fn(),
    }
    const uniform = {buffer: {}, write: vi.fn(), patch: vi.fn()}
    const guarded = {
        with: vi.fn(function (this: unknown) {
            return guarded
        }),
        dispatchThreads: vi.fn(),
    }
    return {
        createBuffer: vi.fn(() => buffer),
        createBindGroup: vi.fn(() => ({})),
        createTexture: vi.fn(() => texture),
        createUniform: vi.fn(() => uniform),
        createGuardedComputePipeline: vi.fn(() => guarded),
        device: {},
    } as never
}

// A minimal generator: the content BokehBlur defocuses (RTT'd once as the child).
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

interface NodeSpec {
    id: string
    def: GpuShaderDefinition
    parentId: string | null
    props?: Record<string, unknown>
    metadata?: Partial<NodeMetadata>
    customId?: string
}

function bridgeFieldInits(def: GpuShaderDefinition, props: Record<string, unknown>, id: string): FieldInit[] {
    const map = createGpuUniformsMap(def as never, props, id)
    const inits: FieldInit[] = []
    for (const [name, u] of Object.entries(map)) {
        inits.push({name, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema})
    }
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
            id: s.id,
            componentName: s.def.name,
            parentId: s.parentId,
            customId: s.customId,
            definition: s.def,
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
    const customIds = new Map<string, string>()
    for (const s of specs) if (s.customId) customIds.set(s.customId, s.id)
    const registry: RegistryView = {
        rootId: rootNode.id,
        getNode: (id) => nodes.get(id),
        getChildren: (parentId) => childrenByParent.get(parentId) ?? [],
        resolveCustomId: (cid) => customIds.get(cid) ?? null,
        store,
    }
    return {registry, store, root}
}

const composeOpts = (root: unknown) => ({
    flipY: false,
    dimensions: {width: 800, height: 600},
    gpu: {device: (root as {device: unknown}).device, root} as never,
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (a) The compute↔RTT↔fragment wiring — BokehBlur > Generator
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('BokehBlur (a) compute → RTT input → fragment composite', () => {
    it('RTTs the child, registers a compute-output texture, and samples the defocused buffer', () => {
        const {registry, root} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'bokeh', def: BokehBlur as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'bokeh', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry, composeOpts(root))

        expect(ir.computeSteps.length).toBe(1)
        expect(ir.rttPasses.length).toBe(1)
        const kinds = ir.textures.map((t) => t.kind).sort()
        expect(kinds).toContain('rtt')
        expect(kinds).toContain('compute')

        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        // Defocused color sampled from the compute-output texture, then unpremultiplied.
        expect(finalWgsl).toMatch(/textureSample\(compute_0/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        // The RTT pass renders the child generator.
        const rttWgsl = tgpu.resolve([ir.rttPasses[0].fragment.entry], {names: 'strict'})
        expect(rttWgsl).toMatch(/genBody/)

        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('bindInputs resolves the child RTT key and binds the gather input', () => {
        const {registry, root} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'bokeh', def: BokehBlur as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'bokeh', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry, composeOpts(root))
        const spec = ir.computeSteps[0]

        let requestedKey: string | null = null
        const before = (root as unknown as {createBindGroup: {mock: {calls: unknown[]}}}).createBindGroup.mock.calls.length
        spec.bindInputs?.((key) => {
            requestedKey = key
            return {texture: {}}
        })
        const after = (root as unknown as {createBindGroup: {mock: {calls: unknown[]}}}).createBindGroup.mock.calls.length
        expect(requestedKey).toBe('rtt_0')
        expect(after).toBeGreaterThan(before)
    })

    it('getComputeNodes returns the single gather pass each frame', () => {
        const {registry, root} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'bokeh', def: BokehBlur as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'bokeh', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry, composeOpts(root))
        const steps = ir.computeSteps[0].getComputeNodes({})
        expect(steps?.length).toBe(1)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (b) Fragment fallback — no device (GPU-free): sharp passthrough, unpremultiplied
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('BokehBlur (b) fragment fallback when compute is unavailable', () => {
    it('samples the child RTT sharp and unpremultiplies (no compute textures)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'bokeh', def: BokehBlur as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'bokeh', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry, {flipY: false})
        expect(ir.computeSteps.length).toBe(0)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_0/)
        expect(finalWgsl).not.toMatch(/compute_0/)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (c) Gather kernels resolve to WGSL (single-pass scatter-as-gather)
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('BokehBlur (c) gather kernels resolve', () => {
    it('uniform-radius kernel resolves with a sampled textureLoad + storage textureStore + tap table', () => {
        const {layout, kernel} = buildBokehGraph(1024, 640, 64)
        const wgsl = tgpu.resolve([layout, kernel], {names: 'strict'})
        expect(wgsl).toMatch(/bokehGather/)
        expect(wgsl).toMatch(/textureLoad/)
        expect(wgsl).toMatch(/textureStore/)
        // The aperture tap table binds as a uniform array the loop indexes.
        expect(wgsl).toMatch(/array<vec4f, 64>/)
        // Cat-eye: the barrel-clip circle shifts by the catEye uniform; each tap weight carries the clip.
        expect(wgsl).toMatch(/catEye/)
        expect(wgsl).toMatch(/barrelShift/)
        expect(wgsl).toMatchSnapshot('bokehGather')
    })

    it('map-driven kernel resolves per channel and emits distinct WGSL (comptime channel branch)', () => {
        const channels: BokehMapChannel[] = ['luminance', 'luminanceInverted', 'alpha', 'alphaInverted']
        for (const channel of channels) {
            const {layout, kernel} = buildBokehMapGraph(1024, 640, 64, channel)
            const wgsl = tgpu.resolve([layout, kernel], {names: 'strict'})
            expect(wgsl).toMatch(/bokehGatherVariable/)
            expect(wgsl).toMatch(/textureStore/)
        }
        const lum = tgpu.resolve([buildBokehMapGraph(1024, 640, 32, 'luminance').kernel], {names: 'strict'})
        const alpha = tgpu.resolve([buildBokehMapGraph(1024, 640, 32, 'alpha').kernel], {names: 'strict'})
        expect(lum).not.toBe(alpha)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (d) Map-driven radius → variable gather — the compute-map interplay
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('BokehBlur (d) map-driven radius → variable gather', () => {
    const mapMeta = {
        renderOrder: 1,
        maps: {
            radius: {type: 'map', source: 'src', channel: 'luminance', inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 100, curve: 0},
        },
    } as unknown as Partial<NodeMetadata>

    function mapDrivenIr() {
        const {registry, root} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'src', def: Generator, parentId: 'root', customId: 'src', metadata: {renderOrder: 0}},
            {id: 'bokeh', def: BokehBlur as GpuShaderDefinition, parentId: 'root', metadata: mapMeta},
            {id: 'gen', def: Generator, parentId: 'bokeh', metadata: {renderOrder: 0}},
        ])
        return {ir: composeNodeTree(registry, composeOpts(root)), root}
    }

    it('registers TWO RTT boundaries (child + map source) and one compute node', () => {
        const {ir} = mapDrivenIr()
        expect(ir.computeSteps.length).toBe(1)
        expect(ir.rttPasses.length).toBe(2)
        const rttCount = ir.textures.filter((t) => t.kind === 'rtt').length
        expect(rttCount).toBe(2)
        expect(ir.textures.some((t) => t.kind === 'compute')).toBe(true)
    })

    it('bindInputs binds BOTH the child RTT input and the map-source RTT', () => {
        const {ir, root} = mapDrivenIr()
        const requested: string[] = []
        const before = (root as unknown as {createBindGroup: {mock: {calls: unknown[]}}}).createBindGroup.mock.calls.length
        ir.computeSteps[0].bindInputs?.((key) => {
            requested.push(key)
            return {texture: {}}
        })
        const after = (root as unknown as {createBindGroup: {mock: {calls: unknown[]}}}).createBindGroup.mock.calls.length
        expect(new Set(requested).size).toBe(2)
        expect(after).toBeGreaterThan(before)
    })

    it('getComputeNodes still returns a single pass (per-pixel radius folded into the gather)', () => {
        const {ir} = mapDrivenIr()
        const steps = ir.computeSteps[0].getComputeNodes({})
        expect(steps?.length).toBe(1)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (e) CPU goldens — radius factor + the aperture tap table
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('BokehBlur (e) CPU goldens', () => {
    it('radiusToPixels applies the RADIUS_SCALE factor', () => {
        expect(RADIUS_SCALE).toBe(0.8)
        expect(radiusToPixels(0)).toBe(0)
        expect(radiusToPixels(50)).toBeCloseTo(40, 10)
        expect(radiusToPixels(100)).toBeCloseTo(80, 10)
    })

    it('polygon-aperture radius: circle below 3 sides, inradius at hexagon edge midpoints', () => {
        expect(bokehPolygonRadius(1.234, 0)).toBe(1)
        expect(bokehPolygonRadius(1.234, 2)).toBe(1)
        // Hexagon: θ=0 is an edge-midpoint direction → the inradius cos(π/6); θ=π/6 is a vertex
        // direction → back out to the unit circle.
        expect(bokehPolygonRadius(0, 6)).toBeCloseTo(Math.cos(Math.PI / 6), 10)
        expect(bokehPolygonRadius(Math.PI / 6, 6)).toBeCloseTo(1, 10)
    })

    it('every shape fills the full table with unit-bounded taps and rim weights ≥ 1', () => {
        for (const shape of BOKEH_SHAPES) {
            const taps = generateBokehTaps(shape, 6, TAP_COUNT)
            expect(taps.length).toBe(TAP_COUNT)
            for (const tap of taps) {
                expect(Math.hypot(tap.x, tap.y)).toBeLessThanOrEqual(1 + 1e-6)
                expect(tap.rim).toBeGreaterThanOrEqual(1)
                expect(tap.rim).toBeLessThanOrEqual(1.6 + 1e-6)
            }
        }
    })

    it('is deterministic — same inputs produce an identical table', () => {
        expect(generateBokehTaps('heart', 6, 64)).toEqual(generateBokehTaps('heart', 6, 64))
    })

    it('blades taps follow the Vogel spiral warped by the polygon radius', () => {
        const taps = generateBokehTaps('blades', 6, 32)
        const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))
        const theta = 5 * GOLDEN_ANGLE
        const r = Math.sqrt((5 + 0.5) / 32) * bokehPolygonRadius(theta, 6)
        expect(taps[5].x).toBeCloseTo(-Math.cos(theta) * r, 10)
        expect(taps[5].y).toBeCloseTo(-Math.sin(theta) * r, 10)
    })

    it('ring keeps the donut hole empty', () => {
        const taps = generateBokehTaps('ring', 6, TAP_COUNT)
        for (const tap of taps) {
            expect(Math.hypot(tap.x, tap.y)).toBeGreaterThanOrEqual(0.44 - 1e-6)
        }
    })

    it('heart taps are asymmetric with the lobes mass in +y (visible disc = −taps → upright heart)', () => {
        const taps = generateBokehTaps('heart', 6, TAP_COUNT)
        const meanY = taps.reduce((sum, tap) => sum + tap.y, 0) / taps.length
        expect(meanY).toBeGreaterThan(0.05)
    })

    it('unknown shape values fall back to the circular aperture', () => {
        expect(generateBokehTaps('nonsense', 0, 32)).toEqual(generateBokehTaps('circle', 0, 32))
    })
})
