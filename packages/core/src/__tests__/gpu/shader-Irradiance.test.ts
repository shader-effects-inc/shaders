import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, collectStructuralHashInputs, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import Irradiance from '@coreroot/shaders/Irradiance/index'
import {
    buildIrradianceKernel, analyticField, svgField, volumetricField,
    irradianceAnalyticLayout, irradianceSvgLayout, irradianceFieldLayout, accumulateKernel, copyKernel,
    silhouetteSeedKernel, silhouetteStepKernel,
} from '@coreroot/gpu/scaffolds/radiance'

/**
 * Irradiance gate. A shape effect whose light field is gathered in a COMPUTE pass (std/paint/radiance
 * `irradianceField` over scaffolds/radiance: cone fan + shadow rays per texel of a fixed-res texture,
 * three field sources, dirty-key idle skip) and read bilinearly by the fragment, which adds the
 * full-res emitting edge (`nearestEdge` + `shadowVisibility`), the lit body (`geometricNormal`), the
 * stop palette walked on its own clock, exposure tone and the straight-alpha emissive output.
 */
const I = Irradiance as GpuShaderDefinition

function mockRoot() {
    const buffer = {patch: vi.fn(), write: vi.fn(), destroy: vi.fn(), $usage: vi.fn(function (this: unknown) {return buffer})}
    const texture = {$usage: vi.fn(function (this: unknown) {return texture}), destroy: vi.fn(), write: vi.fn()}
    const uniform = {buffer: {}, write: vi.fn(), patch: vi.fn()}
    const guarded = {with: vi.fn(function (this: unknown) {return guarded}), dispatchThreads: vi.fn()}
    return {
        createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({})), createTexture: vi.fn(() => texture),
        createUniform: vi.fn(() => uniform), createSampler: vi.fn(() => ({})),
        createGuardedComputePipeline: vi.fn(() => guarded), device: {},
        _uniform: uniform,
    } as never
}

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {'use gpu'; return d.vec4f(uv.x, uv.y, 0.5, 1.0)})
const RootContainer: GpuShaderDefinition = {name: 'Root', props: {} as never, fragment: ({childNode}: GpuFragmentParams): Expr => childNode ?? call(genBody, 'genBody', [])}

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
function buildRegistry(specs: NodeSpec[]): {registry: RegistryView; root: ReturnType<typeof mockRoot>} {
    const root = mockRoot()
    const store = createUniformStore(root, {systemSchema: SystemUniforms})
    const handlesById: Record<string, Record<string, {accessorPath: string; cpu: boolean; value: unknown}>> = {}
    for (const s of specs) {
        const props = {...defaultsFor(s.def), ...(s.props ?? {})}
        const propFields = bridgeFieldInits(s.def, props, s.id)
        handlesById[s.id] = store.defineNode(s.id, [...propFields, {name: '_opacity', schema: d.f32, initial: s.metadata?.opacity ?? 1}]) as never
    }
    store.defineSystem()
    store.finalize()
    const nodes = new Map<string, RegistryNode>()
    const childrenByParent = new Map<string, RegistryNode[]>()
    for (const s of specs) {
        nodes.set(s.id, {id: s.id, componentName: s.def.name, parentId: s.parentId, definition: s.def, metadata: {blendMode: 'normal', opacity: undefined, renderOrder: 0, ...s.metadata} as NodeMetadata, handles: handlesById[s.id]})
    }
    for (const s of specs) if (s.parentId) {
        const arr = childrenByParent.get(s.parentId) ?? []; arr.push(nodes.get(s.id)!); childrenByParent.set(s.parentId, arr)
    }
    const rootNode = specs.find((s) => s.parentId === null)!
    return {registry: {rootId: rootNode.id, getNode: (id) => nodes.get(id), getChildren: (p) => childrenByParent.get(p) ?? [], resolveCustomId: () => null, store}, root}
}
const composeOpts = (root: unknown) => ({flipY: false, dimensions: {width: 800, height: 600}, gpu: {device: (root as {device: unknown}).device, root} as never})
const tree = (props?: Record<string, unknown>) => [
    {id: 'root', def: RootContainer, parentId: null},
    {id: 'i', def: I, parentId: 'root', props, metadata: {renderOrder: 0}},
] as NodeSpec[]
const FRAME = {pointer: {x: 0.5, y: 0.5}, deltaTime: 0.016, dimensions: {width: 800, height: 600}}

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Irradiance (a) compute-gathered light field → fragment', () => {
    it('registers the irradiance texture; each frame = params write + gather + accumulate + copy + denoise H/V; a still scene converges then idles', () => {
        const {registry, root} = buildRegistry(tree(FLAT_SHAPE))
        const ir = composeNodeTree(registry, composeOpts(root))
        expect(ir.computeSteps.length).toBe(1)
        expect(ir.textures.map((t) => t.kind)).toContain('compute')
        const step = ir.computeSteps[0]
        expect(step.getComputeNodes(FRAME)?.length).toBe(6)
        // Two startup warm-up re-gathers (async inputs settling), then progressive refinement: a still
        // scene keeps folding new sample sets into the running mean…
        let refining = 0
        while (step.getComputeNodes(FRAME) !== null) refining++
        expect(refining).toBe(2 + 12)
        // …then idles: nothing feeding the gather changed → no dispatch at all.
        expect(step.getComputeNodes(FRAME)).toBeNull()
        expect(step.getComputeNodes(FRAME)).toBeNull()
    })

    it('ray schedule: a full burst on the first frame, small refinement gathers after, sample-weighted mean', () => {
        const {registry, root} = buildRegistry(tree(FLAT_SHAPE))
        const ir = composeNodeTree(registry, composeOpts(root))
        const step = ir.computeSteps[0]
        const uniform = (root as unknown as {_uniform: {write: ReturnType<typeof vi.fn>}})._uniform
        const run = () => { for (const n of step.getComputeNodes(FRAME) ?? []) if (typeof n === 'function') n() }
        run()
        const writes = uniform.write.mock.calls.map((c) => c[0] as Record<string, unknown>)
        expect(writes.find((w) => 'rays' in w)?.rays).toBe(128)
        expect(writes.find((w) => 'weight' in w)?.weight).toBe(1)
        // Warm-up frames re-gather at the motion budget (a fresh estimate each: weight 1)…
        run(); run()
        uniform.write.mockClear()
        // …then refinement folds small gathers into the running mean, sample-weighted.
        run()
        const refine = uniform.write.mock.calls.map((c) => c[0] as Record<string, unknown>)
        expect(refine.find((w) => 'rays' in w)?.rays).toBe(32)
        expect(refine.find((w) => 'weight' in w)?.weight).toBeCloseTo(32 / (64 + 32))
    })

    it('fragment samples the gathered field, adds the full-res rim + body, loops the light list', () => {
        const {registry, root} = buildRegistry(tree(FLAT_SHAPE))
        const ir = composeNodeTree(registry, composeOpts(root))
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl).toMatch(/textureSampleLevel\(compute_0/)
        expect(wgsl).toMatch(/analyticSdf_circleSDF/)
        expect(wgsl).toMatch(/sdfSpaceUV/)
        expect(wgsl).toMatch(/var rimVisibility/)
        expect(wgsl).toMatch(/bevelledFlatNormal/)
        // The light LIST: per-field lane arrays indexed in a runtime-count loop.
        expect(wgsl).toMatch(/lights_position\[/)
        expect(wgsl).toMatch(/lights_color\[/)
        expect(wgsl).toMatch(/u32\(uniforms\.n_i\.lightsCount\)/)
        expect(wgsl).toMatch(/silhouetteAlpha/)
        expect(wgsl).not.toMatch(/coneCover/) // the gather lives in compute, not the fragment
        expect(wgsl).toMatchSnapshot('final-pass')
    })

})

describe('Irradiance (b) fragment fallback when compute is unavailable', () => {
    it('no compute textures (GPU-free resolve): rim + body still resolve, no gathered field read', () => {
        const {registry} = buildRegistry(tree(FLAT_SHAPE))
        const ir = composeNodeTree(registry, {flipY: false})
        expect(ir.computeSteps.length).toBe(0)
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl).not.toMatch(/compute_0/)
        expect(wgsl).toMatch(/var rimVisibility/)
    })

    it('shadows compile out of the rim when disabled', () => {
        const {registry} = buildRegistry(tree({...FLAT_SHAPE, shadows: false}))
        const ir = composeNodeTree(registry, {flipY: false})
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl).not.toMatch(/var rimVisibility/)
    })
})

describe('Irradiance (c) gather kernels resolve (D3 rules)', () => {
    const opts = {steps: 24, shadowSteps: 12, namePrefix: 'irrTest'}

    it('analytic source: the baked SDF fn per texel, the cone loop, shadow rays, textureStore', () => {
        const k = buildIrradianceKernel(irradianceAnalyticLayout, analyticField('polygonSDF'), opts)
        const wgsl = tgpu.resolve([k], {names: 'strict'})
        expect(wgsl).toMatch(/irrTestGather/)
        expect(wgsl).toMatch(/irrTestVisibility/)
        expect(wgsl).toMatch(/polygonSdf|analyticSdf/)
        expect(wgsl).toMatch(/textureStore/)
        expect(wgsl).toMatch(/rays/) // the per-frame ray budget is a uniform loop bound
    })

    it('SVG source samples the SDF texture bilinearly (textureSampleLevel) and continues it beyond the square', () => {
        const k = buildIrradianceKernel(irradianceSvgLayout, svgField, {...opts, shadowSteps: 0})
        const wgsl = tgpu.resolve([k], {names: 'strict'})
        expect(wgsl).toMatch(/textureSampleLevel/)
        expect(wgsl).toMatch(/irradianceBeyondSquare/)
        expect(wgsl).not.toMatch(/irrTestVisibility\(/) // shadow rays compiled out
    })

    it('volumetric source samples the jump-flooded silhouette distance field', () => {
        const k = buildIrradianceKernel(irradianceFieldLayout, volumetricField, opts)
        const wgsl = tgpu.resolve([k], {names: 'strict'})
        expect(wgsl).toMatch(/textureSampleLevel/)
        expect(wgsl).toMatch(/silhouetteTex/)
    })

    it('silhouette jump-flood kernels resolve (seed reads the marched field; step propagates seeds)', () => {
        const seed = tgpu.resolve([silhouetteSeedKernel], {names: 'strict'})
        expect(seed).toMatch(/silhouetteSeed/)
        expect(seed).toMatch(/textureLoad/)
        const step = tgpu.resolve([silhouetteStepKernel], {names: 'strict'})
        expect(step).toMatch(/silhouetteStep/)
        expect(step).toMatch(/textureStore/)
    })

    it('progressive accumulation + copy kernels resolve', () => {
        expect(tgpu.resolve([accumulateKernel], {names: 'strict'})).toMatch(/irradianceAccumulate/)
        expect(tgpu.resolve([copyKernel], {names: 'strict'})).toMatch(/irradianceCopy/)
    })
})

describe('Irradiance (d) volumetric (3D) shape path', () => {
    it('a sphere3D shape pre-marches the field ahead of the gather; fragment routes the vf sampler', () => {
        const {registry, root} = buildRegistry(tree({shape: JSON.stringify({type: 'sphere3D', radius: 0.35}), shapeType: 'sphere3D'}))
        const ir = composeNodeTree(registry, composeOpts(root))
        expect(ir.computeSteps.length).toBe(1)
        expect(ir.textures.filter((t) => t.kind === 'compute').length).toBe(2) // marched field + irradiance
        // First frame: field-march thunk + the silhouette jump-flood (params write + seed, then
        // 9 halving steps of write + pass for 512²) + params write + gather + accumulate + copy +
        // denoise H/V.
        const nodes = ir.computeSteps[0].getComputeNodes(FRAME)
        expect(nodes?.length).toBe(1 + (2 + 2 * 9) + 6)
        // Run the thunks: the silhouette passes must write the ABI's field names (vfSpanX…, step).
        const uniform = (root as unknown as {_uniform: {write: ReturnType<typeof vi.fn>}})._uniform
        uniform.write.mockClear()
        for (const n of nodes ?? []) if (typeof n === 'function') n()
        const silWrites = uniform.write.mock.calls.map((c) => c[0] as Record<string, unknown>).filter((w) => 'step' in w)
        expect(silWrites.length).toBe(1 + 9)
        for (const w of silWrites) {
            expect(typeof w.vfSpanX).toBe('number')
            expect(typeof w.vfOriginX).toBe('number')
            expect(typeof w.res).toBe('number')
        }
        expect(silWrites.map((w) => w.step)).toEqual([0, 256, 128, 64, 32, 16, 8, 4, 2, 1])
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl).toMatch(/fieldSampleArg|vfSample/)
    })
})

describe('Irradiance (e) structural hash', () => {
    it('registers in the structural hash surface (no clocks: color is per light, static)', () => {
        expect(I.animatedTime).toBeUndefined()
        expect(I.extraAnimatedTimes).toBeUndefined()
        const {registry} = buildRegistry(tree(FLAT_SHAPE))
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Irradiance')
    })
})
