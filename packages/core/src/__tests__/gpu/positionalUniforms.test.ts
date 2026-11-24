import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, schemaSignature, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, collectStructuralHashInputs, structuralNodeOrder, call} from '@coreroot/gpu/composer'
import {structuralHash} from '@coreroot/gpu/pipelineCache'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import Glow from '@coreroot/shaders/Glow/index'
import Swirl from '@coreroot/shaders/Swirl/index'
import Mirror from '@coreroot/shaders/Mirror/index'
import Vignette from '@coreroot/shaders/Vignette/index'

/**
 * Shared compile, phase 2: uniform struct members are named by tree position (`n_0`, `n_1`, …),
 * so two compositions with the same structure generate byte-identical WGSL whatever their layer
 * ids, custom ids and registration order, and their id-free share keys match.
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

const rootBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Root: GpuShaderDefinition = {
    name: 'Root',
    props: {} as never,
    fragment: ({childNode, ctx}: GpuFragmentParams): Expr => childNode ?? call(rootBody, 'rootBody', [ctx.uv]),
}

interface Spec {id: string; def: GpuShaderDefinition; parentId: string | null; customId?: string; metadata?: Partial<NodeMetadata>}

function fieldsFor(def: GpuShaderDefinition, id: string): FieldInit[] {
    const props: Record<string, unknown> = {}
    for (const [name, cfg] of Object.entries(def.props)) props[name] = (cfg as {default: unknown}).default
    const inits: FieldInit[] = []
    for (const [name, u] of Object.entries(createGpuUniformsMap(def as never, props, id))) {
        inits.push({name, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema})
    }
    return [...inits, {name: '_opacity', schema: d.f32, initial: 1}]
}

/** Registers nodes in `registration` order but names members by tree position, as the renderer does. */
function compose(specs: Spec[]) {
    const root = mockRoot()
    const nodes = new Map<string, RegistryNode>()
    const children = new Map<string, RegistryNode[]>()
    for (const s of specs) {
        nodes.set(s.id, {
            id: s.id, componentName: s.def.name, parentId: s.parentId, definition: s.def,
            metadata: {blendMode: 'normal', opacity: undefined, renderOrder: 0, ...s.metadata, id: s.customId} as NodeMetadata,
            handles: {},
        })
    }
    for (const s of specs) if (s.parentId) children.set(s.parentId, [...(children.get(s.parentId) ?? []), nodes.get(s.id)!])
    const store = createUniformStore(root, {systemSchema: SystemUniforms})
    const registry: RegistryView = {
        rootId: specs.find((s) => s.parentId === null)!.id,
        getNode: (id) => nodes.get(id),
        getChildren: (pid) => [...(children.get(pid) ?? [])],
        resolveCustomId: (cid) => specs.find((s) => s.customId === cid)?.id ?? null,
        store,
    }
    const order = structuralNodeOrder(registry)
    const position = new Map(order.map((id, i) => [id, i]))
    order.forEach((id, i) => {
        const s = specs.find((x) => x.id === id)!
        nodes.get(id)!.handles = store.defineNode(id, fieldsFor(s.def, id), `n_${i}`) as never
    })
    store.defineSystem()
    store.finalize()
    const ir = composeNodeTree(registry, {flipY: false, dimensions: {width: 800, height: 600}, gpu: {device: {}, root} as never})
    const wgsl = [...ir.rttPasses.map((p) => p.fragment.entry), ir.finalPass.entry]
        .map((entry) => tgpu.resolve([entry], {names: 'strict'}))
    const shareKey = structuralHash(collectStructuralHashInputs(registry, {}, (id) => String(position.get(id))))
    const ownKey = structuralHash(collectStructuralHashInputs(registry, {}))
    return {wgsl, shareKey, ownKey}
}

const tree = (ids: {root: string, glow: string, gen: string, swirl: string, mirror: string, vig: string}, mask: string): Spec[] => [
    {id: ids.root, def: Root, parentId: null},
    {id: ids.glow, def: Glow as GpuShaderDefinition, parentId: ids.root, metadata: {renderOrder: 0}},
    {id: ids.gen, def: Root, parentId: ids.glow, metadata: {renderOrder: 0}},
    {id: ids.swirl, def: Swirl as GpuShaderDefinition, parentId: ids.root, customId: mask, metadata: {renderOrder: 1}},
    {id: ids.mirror, def: Mirror as GpuShaderDefinition, parentId: ids.root, metadata: {renderOrder: 2, mask: {source: mask, type: 'alpha'} as never}},
    {id: ids.vig, def: Vignette as GpuShaderDefinition, parentId: ids.root, metadata: {renderOrder: 3}},
]

describe('positional uniform names (shared compile phase 2)', () => {
    const a = compose(tree({root: 'r1', glow: 'g-1', gen: 'x.9', swirl: 'sw', mirror: 'mi', vig: 'v'}, 'shape'))
    // Different ids, a different custom id for the mask source, registered in reverse order
    const b = compose(tree({root: 'R', glow: 'glowAbc', gen: 'gen_77', swirl: 'q', mirror: 'z', vig: 'vg2'}, 'logo').reverse())

    it('generates identical WGSL for the same structure under different ids', () => {
        expect(a.wgsl.length).toBeGreaterThan(1)
        expect(b.wgsl).toEqual(a.wgsl)
        expect(a.wgsl.join('\n')).toMatch(/n_0\b/)
        expect(a.wgsl.join('\n')).not.toMatch(/n_g_1|n_glowAbc/)
    })

    it('matches on the id-free share key but not on the per-renderer key', () => {
        expect(b.shareKey).toBe(a.shareKey)
        expect(b.ownKey).not.toBe(a.ownKey)
    })

    it('separates structures that differ only in which node a mask reads', () => {
        const specs = tree({root: 'r1', glow: 'g-1', gen: 'x.9', swirl: 'sw', mirror: 'mi', vig: 'v'}, 'shape')
        specs[5] = {...specs[5], customId: 'other'}
        specs[4] = {...specs[4], metadata: {...specs[4].metadata, mask: {source: 'other', type: 'alpha'} as never}}
        expect(compose(specs).shareKey).not.toBe(a.shareKey)
    })
})

describe('uniform layout signature (shared compile)', () => {
    const signatureOf = (fields: FieldInit[]) => {
        const store = createUniformStore(mockRoot(), {systemSchema: SystemUniforms})
        store.defineNode('a', fields, 'n_0')
        store.defineSystem()
        store.finalize()
        return schemaSignature(store.schema)
    }
    const f = (name: string, schema: unknown = d.f32): FieldInit => ({name, schema: schema as never, initial: 0})

    it('is the same for the same fields and differs when one is added, retyped or moved', () => {
        const base = signatureOf([f('speed'), f('tint', d.vec4f)])
        // Spelled out member by member, not just the struct's name
        expect(base).toMatch(/n_0:\{speed:f32,tint:vec4f/)
        expect(signatureOf([f('speed'), f('tint', d.vec4f)])).toBe(base)
        expect(signatureOf([f('speed'), f('tint', d.vec4f), f('extra')])).not.toBe(base)
        expect(signatureOf([f('speed'), f('tint', d.vec3f)])).not.toBe(base)
        expect(signatureOf([f('tint', d.vec4f), f('speed')])).not.toBe(base)
    })
})
