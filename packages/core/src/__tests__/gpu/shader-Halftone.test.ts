import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type UniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, collectStructuralHashInputs, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import Halftone from '@coreroot/shaders/Halftone/index'
import {halftoneChannelK, halftoneChannelC, halftoneTransmission, halftoneClassic} from '@coreroot/gpu/kit/patternPaints'

/**
 * Halftone port gate (Phase D2-C). RTT stylization with blendWithChildren:true + compileTime `style`
 * (classic single-plate / cmyk four-plate). Both modes convertToTexture + unpremultiply. Leaf-sibling
 * test proves the over-child composite; CPU golden the pure CMYK-channel + transmission + classic math.
 */
function mockRoot() {
    const buffer = {patch: vi.fn(), write: vi.fn(), destroy: vi.fn(), $usage: vi.fn(function (this: unknown) { return buffer })}
    return {createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({}))} as never
}

const genBody = tgpu.fn([d.f32, d.vec2f], d.vec4f)((seed, uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, seed, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {seed: {default: 0.5}} as never,
    fragment: ({uniforms, ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [uniforms.seed, ctx.uv]),
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
}

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

function buildRegistry(specs: NodeSpec[]): {registry: RegistryView; store: UniformStore} {
    const store = createUniformStore(mockRoot(), {systemSchema: SystemUniforms})
    const handlesById: Record<string, Record<string, {accessorPath: string; cpu: boolean; value: unknown}>> = {}
    for (const s of specs) {
        const props = {...defaultsFor(s.def), ...(s.props ?? {})}
        const propFields = bridgeFieldInits(s.def, props, s.id)
        const synthetic: FieldInit[] = [{name: '_opacity', schema: d.f32, initial: s.metadata?.opacity ?? 1}]
        handlesById[s.id] = store.defineNode(s.id, [...propFields, ...synthetic]) as never
    }
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
    const root = specs.find((s) => s.parentId === null)!
    const registry: RegistryView = {
        rootId: root.id,
        getNode: (id) => nodes.get(id),
        getChildren: (parentId) => childrenByParent.get(parentId) ?? [],
        resolveCustomId: () => null,
        store,
    }
    return {registry, store}
}

const H = Halftone as GpuShaderDefinition

describe('Halftone (a) classic style, shipped-v1 replace over a preceding sibling (blendWithChildren:false parity)', () => {
    it('RTTs + unpremultiplies the child, modulates a dot plate, REPLACES the sibling (shipped v1 parity)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'gen', def: Generator, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'ht', def: H, parentId: 'root', metadata: {renderOrder: 1}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBeGreaterThanOrEqual(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        // The sibling is drawn once, into the RTT the filter reads: at full opacity the filter
        // replaces it outright, so the final pass doesn't evaluate it again underneath.
        expect(finalWgsl).not.toMatch(/genBody/)
        expect(tgpu.resolve([ir.rttPasses[0].fragment.entry], {names: 'strict'})).toMatch(/genBody/)
        expect(finalWgsl).toMatch(/halftoneClassic/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        expect(finalWgsl).toMatchSnapshot('halftone-classic')
    })
})

describe('Halftone (b) cmyk style', () => {
    const buildCmyk = () => buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'gen', def: Generator, parentId: 'root', metadata: {renderOrder: 0}},
        {id: 'ht', def: H, parentId: 'root', props: {style: 'cmyk', misprint: 0.005}, metadata: {renderOrder: 1}},
    ])

    it('emits the four-plate offset sampling + channel separation + transmission', () => {
        const ir = composeNodeTree(buildCmyk().registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/halftonePlateUV/)
        expect(finalWgsl).toMatch(/halftoneChannelK/)
        expect(finalWgsl).toMatch(/halftonePlateGrid/)
        expect(finalWgsl).toMatch(/halftoneTransmission/)
        expect(finalWgsl).not.toMatch(/halftoneClassic/)
        expect(finalWgsl).toMatchSnapshot('halftone-cmyk')
    })

    it('style is part of the structural (recompile) hash', () => {
        const build = (style: string) => {
            const {registry} = buildRegistry([
                {id: 'root', def: RootContainer, parentId: null},
                {id: 'gen', def: Generator, parentId: 'root', metadata: {renderOrder: 0}},
                {id: 'ht', def: H, parentId: 'root', props: {style}, metadata: {renderOrder: 1}},
            ])
            return collectStructuralHashInputs(registry).join('\n')
        }
        expect(build('classic')).not.toBe(build('cmyk'))
    })
})

describe('Halftone (c) CPU golden — CMYK channels, transmission, classic', () => {
    it('channelK / channelC reproduce the v1 separation', () => {
        const s = d.vec4f(0.2, 0.4, 0.9, 1.0)
        const k = 1 - Math.max(Math.max(0.2, 0.4), 0.9) // 0.1
        expect(halftoneChannelK(s) as number).toBeCloseTo(k, 6)
        const invK = Math.max(1 - k, 0.0001)
        expect(halftoneChannelC(s) as number).toBeCloseTo((1 - 0.2 - k) / invK, 6)
    })

    it('transmission fades white toward the ink by mask × ink alpha', () => {
        const ink = d.vec4f(0, 1, 1, 0.5) // cyan, half opacity
        const o = halftoneTransmission(ink, 1.0) as {x: number; y: number; z: number}
        // mix(vec3(1), (0,1,1), 1*0.5) = (0.5, 1, 1)
        expect(o.x).toBeCloseTo(0.5, 6)
        expect(o.y).toBeCloseTo(1, 6)
        expect(o.z).toBeCloseTo(1, 6)
    })

    it('classic modulates the child color + alpha by the dot pattern', () => {
        // Deterministic: at frequency where gridUV lands at cell centre the dot is fully covered.
        const out = halftoneClassic(d.vec4f(0.5, 0.5, 0.5, 1.0), d.vec2f(0.5, 0.5), 1.0, 45, 100) as {x: number; y: number; z: number; w: number}
        // Output rgb == child.rgb * dotPattern, alpha == child.a * dotPattern → rgb/a ratio holds.
        expect(out.w).toBeGreaterThanOrEqual(0)
        expect(out.w).toBeLessThanOrEqual(1)
        expect(out.x).toBeCloseTo(0.5 * out.w, 6)
    })
})
