import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import SmokeFill, {makeMaskKernel, svgMaskKernel, vfMaskKernel, splatKernel, jacobiKernel, outputKernel} from '@coreroot/shaders/SmokeFill/index'

/**
 * SmokeFill port gate (W7-B) — Stable-Fluids CONFINED to a shape. A mask kernel fills a per-cell
 * SDF-membership buffer (body-callable buildAnalyticSdfFn, shapeType baked), the fluid gates on it,
 * and the fragment adds a crisp SDF edge mask. GPU-free.
 */

function mockRoot() {
    const buffer = {patch: vi.fn(), write: vi.fn(), destroy: vi.fn(), $usage: vi.fn(function (this: unknown) {return buffer})}
    const texture = {$usage: vi.fn(function (this: unknown) {return texture}), destroy: vi.fn(), write: vi.fn()}
    const uniform = {buffer: {}, write: vi.fn(), patch: vi.fn()}
    const guarded = {with: vi.fn(function (this: unknown) {return guarded}), dispatchThreads: vi.fn()}
    return {
        createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({})), createTexture: vi.fn(() => texture),
        createUniform: vi.fn(() => uniform), createGuardedComputePipeline: vi.fn(() => guarded), device: {},
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
    {id: 'sf', def: SmokeFill as GpuShaderDefinition, parentId: 'root', props, metadata: {renderOrder: 0}},
] as NodeSpec[]

const FRAME = {pointer: {x: 0.5, y: 0.5}, deltaTime: 0.016, dimensions: {width: 800, height: 600}}

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('SmokeFill (a) shape-confined fluid → SDF-masked fragment', () => {
    it('registers a compute texture; ordered dispatch = mask + fluid (21 steps); fragment masks by SDF', () => {
        const {registry, root} = buildRegistry(tree(FLAT_SHAPE))
        const ir = composeNodeTree(registry, composeOpts(root))
        expect(ir.computeSteps.length).toBe(1)
        expect(ir.textures.map((t) => t.kind)).toContain('compute')
        // mask + splat + curl + vorticity + divergence (5) + 10 jacobi + gradSub/advectVel/copyVel/advectDens/copyDens/output (6).
        expect(ir.computeSteps[0].getComputeNodes(FRAME)?.length).toBe(21)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/-1024/) // the inline SDF-distance → crisp shape-edge clamp
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/mixColors/)
        expect(finalWgsl).toMatch(/textureSample\(compute_0/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('SmokeFill (b) fragment fallback when compute is unavailable', () => {
    it('transparent with no compute textures (GPU-free resolve)', () => {
        const {registry} = buildRegistry(tree(FLAT_SHAPE))
        const ir = composeNodeTree(registry, {flipY: false})
        expect(ir.computeSteps.length).toBe(0)
        expect(tgpu.resolve([ir.finalPass.entry], {names: 'strict'})).not.toMatch(/compute_0/)
    })
})

describe('SmokeFill (c) kernels resolve (D3 rules)', () => {
    it('mask (analytic SDF call) / splat / jacobi / output kernels resolve', () => {
        const mask = tgpu.resolve([makeMaskKernel('circleSDF')], {names: 'strict'})
        expect(mask).toMatch(/smokeFillMask/)
        expect(tgpu.resolve([splatKernel], {names: 'strict'})).toMatch(/smokeFillSplat/)
        expect(tgpu.resolve([jacobiKernel], {names: 'strict'})).toMatch(/smokeFillJacobi/)
        expect(tgpu.resolve([outputKernel], {names: 'strict'})).toMatch(/textureStore/)
    })

    it('SVG mask kernel resolves — samples the SDF texture (textureLoad) instead of an analytic fn (e0)', () => {
        const svg = tgpu.resolve([svgMaskKernel], {names: 'strict'})
        expect(svg).toMatch(/smokeFillSvgMask/)
        expect(svg).toMatch(/textureLoad/) // reads the uploaded SVG SDF texture per grid cell
        // Writes the same inside/dist mask the fluid passes read — no analytic sdf call in this variant.
        expect(svg).toMatch(/textureStore|maskBuf|=/)
    })

    it('volumetric mask kernel resolves — loads the pre-marched field texture per grid cell', () => {
        const vf = tgpu.resolve([vfMaskKernel], {names: 'strict'})
        expect(vf).toMatch(/smokeFillVfMask/)
        expect(vf).toMatch(/textureLoad/) // reads the rgba32float field over its aspect-fit domain
    })
})

describe('SmokeFill (d) volumetric (3D) shape path', () => {
    it('a sphere3D shape pre-marches the field ahead of the fluid; fragment routes the vf sampler', () => {
        const {registry, root} = buildRegistry(tree({shape: JSON.stringify({type: 'sphere3D', radius: 0.35}), shapeType: 'sphere3D'}))
        const ir = composeNodeTree(registry, composeOpts(root))
        expect(ir.computeSteps.length).toBe(1)
        // Both compute textures registered: the marched field + the smoke output.
        expect(ir.textures.filter((t) => t.kind === 'compute').length).toBe(2)
        // First frame: 1 field-march thunk + mask + fluid (21) = 22 ordered nodes.
        expect(ir.computeSteps[0].getComputeNodes(FRAME)?.length).toBe(22)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/fieldSampleArg/) // volumetric sampler feeds the crisp shape mask
        expect(finalWgsl).toMatch(/sdfSpaceUV/) // the inline SDF-distance shape mask rides the sdf-space UV
    })
})
