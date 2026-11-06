import {describe, it, expect, vi, beforeAll, afterAll} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type UniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, type ComposeOptions} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import Text from '@coreroot/shaders/Text/index'
import {orientedBoxField} from '@coreroot/std/effects/stylize'

/**
 * Text port gate (Phase D4-A). Text rasters a glyph run to a media texture and samples it at the
 * rotated box-local UV. GPU-free resolve/snapshot asserts the media sample + the placement body;
 * the half-extents are extraFields (uniforms.halfW / halfH). The raster (document/canvas/font) work
 * is deferred behind setTimeout(0). CPU golden checks the orientedBoxField placement math (center maps to
 * the box centre). Orientation note: v2 drops v1's `localV.oneMinus()` — the media texture is
 * top-left origin (copyExternalImageToTexture) like VideoTexture's external, so no flip is needed
 * (empirically confirmed by the smoke PNG).
 */

beforeAll(() => vi.useFakeTimers())
afterAll(() => vi.useRealTimers())

function mockRoot() {
    const buffer = {
        patch: vi.fn(),
        write: vi.fn(),
        destroy: vi.fn(),
        $usage: vi.fn(function (this: unknown) {
            return buffer
        }),
    }
    return {createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({}))} as never
}

const composeOptions: ComposeOptions = {
    createMediaTexture: (opts) => ({texture: {}, width: opts.width, height: opts.height, write: vi.fn(), unwrap: vi.fn(() => ({})), destroy: vi.fn()}) as never,
    gpu: {device: {queue: {}} as never, root: {} as never},
}

const RootContainer: GpuShaderDefinition = {
    name: 'Root',
    props: {} as never,
    fragment: ({childNode}: GpuFragmentParams): Expr => childNode ?? ({_emit: () => 'vec4f(0.0)'} as Expr),
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
    for (const [name, u] of Object.entries(map)) {
        inits.push({name, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema})
    }
    // extraFields (halfW/halfH) — the renderer registers these on the node struct; mirror it here.
    for (const [name, cfg] of Object.entries(def.extraFields ?? {})) {
        inits.push({name, schema: cfg.schema as never, initial: cfg.initial as never})
    }
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
            id: s.id,
            componentName: s.def.name,
            parentId: s.parentId,
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

function resolveFinal(props?: Record<string, unknown>): {wgsl: string; ir: ReturnType<typeof composeNodeTree>} {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'txt', def: Text as GpuShaderDefinition, parentId: 'root', props, metadata: {renderOrder: 0}},
    ])
    const ir = composeNodeTree(registry, composeOptions)
    return {wgsl: tgpu.resolve([ir.finalPass.entry], {names: 'strict'}), ir}
}

describe('Text (a) glyph atlas sampling + placement', () => {
    it('samples its rectangle of the glyph atlas at the placement body, reads half-extents', () => {
        const {wgsl, ir} = resolveFinal()
        const mediaTex = ir.textures.filter((t) => t.kind === 'media')
        expect(mediaTex.length).toBe(1)
        expect(mediaTex[0].key).toBe('media_0')
        expect(ir.externalTextures.length).toBe(0)
        expect(ir.rttPasses.length).toBe(0)
        expect(wgsl).toMatch(/orientedBoxField/)
        // Half-extents and the atlas rectangle read as uniforms (extraFields).
        expect(wgsl).toMatch(/\.halfW/)
        expect(wgsl).toMatch(/\.halfH/)
        expect(wgsl).toMatch(/\.glyphRect/)
        // Coverage: explicit level 0 (no mips), only inside the glyph box, area filter or cubic.
        expect(wgsl).toMatch(/textureSampleLevel\(media_0/)
        expect(wgsl).not.toMatch(/textureSample\(media_0/)
        expect(wgsl).toMatch(/if \(covInside\w* > 0\.5\)/)
        expect(wgsl).toMatch(/catmullRomAxisWeights/)
        expect(wgsl).toMatch(/dpdx\(/)
        // Edge contrast shaping by the text colour's lightness.
        expect(wgsl).toMatch(/fn shapeCoverage/)
        expect(wgsl).toMatchSnapshot('final-pass')
    })

    it('shares one atlas texture (one binding) across every Text layer', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'a', def: Text as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'b', def: Text as GpuShaderDefinition, parentId: 'root', props: {text: 'Second'}, metadata: {renderOrder: 1}},
            {id: 'c', def: Text as GpuShaderDefinition, parentId: 'root', props: {text: 'Third'}, metadata: {renderOrder: 2}},
        ])
        const ir = composeNodeTree(registry, composeOptions)
        expect(ir.textures.filter((t) => t.kind === 'media').map(t => t.key)).toEqual(['media_0'])
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl.match(/textureSampleLevel\(media_0/g)!.length).toBeGreaterThan(3)
    })
})

describe('Text (b) CPU golden — placement (orientedBoxField)', () => {
    // center is the TRANSFORMED prop (transformPosition stores (x, 1-y)); the body recovers y via 1-center.y.
    it('maps the box centre to (0.5, 0.5) local, inside', () => {
        const f = orientedBoxField(d.vec2f(0.5, 0.5), d.f32(1), d.vec2f(0.5, 0.5), d.f32(0), d.f32(0.25), d.f32(0.1)) as d.v3f
        expect(f.x).toBeCloseTo(0.5, 6)
        expect(f.y).toBeCloseTo(0.5, 6)
        expect(f.z).toBeCloseTo(1, 6)
    })
    it('maps a point half a box-width to the right to localU = 1 (box edge)', () => {
        const f = orientedBoxField(d.vec2f(0.75, 0.5), d.f32(1), d.vec2f(0.5, 0.5), d.f32(0), d.f32(0.25), d.f32(0.1)) as d.v3f
        expect(f.x).toBeCloseTo(1, 5)
        expect(f.y).toBeCloseTo(0.5, 6)
    })
    it('is transparent (mask 0) outside the box', () => {
        const f = orientedBoxField(d.vec2f(0.95, 0.5), d.f32(1), d.vec2f(0.5, 0.5), d.f32(0), d.f32(0.25), d.f32(0.1)) as d.v3f
        expect(f.z).toBeCloseTo(0, 6)
    })
})
