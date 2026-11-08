import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import {blend} from '@coreroot/gpu/kit'
import TimeTrail, {timeTrailKernel} from '@coreroot/shaders/TimeTrail/index'
import {frameDiffMask, hueRotateTurns} from '@coreroot/gpu/scaffolds/gridKernels'

/**
 * TimeTrail port gate — an Echo-style temporal-trail feedback SIMULATION (the DataMosh family).
 * Two rgba16float ping-pong state textures + a display copy + a prev-live ping-pong pair for the
 * Motion source's frame difference; the child RTT is late-bound (bindInputs — the physical texture
 * is allocated after composition). Motion source (default): frame-difference-gated stamp, ghosts
 * composite OVER the live frame (works on opaque children); Alpha source: coverage stamp, crisp
 * live frame composites over the trail. GPU-free: a mock root answers allocations; with no device
 * the fragment falls back to a live passthrough.
 */

function mockRoot() {
    const buffer = {patch: vi.fn(), write: vi.fn(), destroy: vi.fn(), $usage: vi.fn(function (this: unknown) {return buffer})}
    const texture = {$usage: vi.fn(function (this: unknown) {return texture}), destroy: vi.fn(), write: vi.fn(), props: {format: 'rgba16float'}}
    const uniform = {buffer: {}, write: vi.fn(), patch: vi.fn()}
    const guarded = {with: vi.fn(function (this: unknown) {return guarded}), dispatchThreads: vi.fn()}
    return {
        createBuffer: vi.fn(() => buffer),
        createBindGroup: vi.fn(() => ({})),
        createTexture: vi.fn(() => texture),
        createSampler: vi.fn(() => ({resourceType: 'sampler'})),
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
function buildRegistry(specs: NodeSpec[]): {registry: RegistryView; root: ReturnType<typeof mockRoot>} {
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
    return {registry, root}
}
const composeOpts = (root: unknown) => ({flipY: false, dimensions: {width: 800, height: 600}, gpu: {device: (root as {device: unknown}).device, root} as never})
const tree = () => [
    {id: 'root', def: RootContainer, parentId: null},
    {id: 'tt', def: TimeTrail as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}},
    {id: 'gen', def: Generator, parentId: 'tt', metadata: {renderOrder: 0}},
] as NodeSpec[]

const FRAME = {deltaTime: 0.016, pointer: {x: 0.5, y: 0.5}, dimensions: {width: 800, height: 600}}

describe('TimeTrail (a) feedback compute → display copy → crisp-over-trail composite', () => {
    it('RTTs the child, registers a display (compute) texture, samples both + the composite fn', () => {
        const {registry, root} = buildRegistry(tree())
        const ir = composeNodeTree(registry, composeOpts(root))
        expect(ir.computeSteps.length).toBe(1)
        expect(ir.rttPasses.length).toBe(1)
        const kinds = ir.textures.map((t) => t.kind).sort()
        expect(kinds).toContain('rtt')
        expect(kinds).toContain('compute')
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/unpremultiply/) // straight-alpha in/out around the kit blend
        expect(finalWgsl).toMatch(/textureSample\(compute_0/) // the trail display copy
        expect(finalWgsl).toMatch(/textureSample\(rtt_0/) // the crisp live child
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('is child-dependent → exposes bindInputs; getComputeNodes yields 1 step once inputs are bound', () => {
        const {registry, root} = buildRegistry(tree())
        const ir = composeNodeTree(registry, composeOpts(root))
        const step = ir.computeSteps[0]
        expect(step.bindInputs).toBeDefined()
        // Before the child RTT is bound the pass is a no-op (nothing to read).
        expect(step.getComputeNodes(FRAME)).toBeNull()
        step.bindInputs!((key: string) => ({texture: {key}}))
        expect(step.getComputeNodes(FRAME)?.length).toBe(1)
    })
})

describe('TimeTrail (b) fragment fallback when compute is unavailable', () => {
    it('passthrough (unpremultiplied) with no compute textures', () => {
        const {registry} = buildRegistry(tree())
        const ir = composeNodeTree(registry, {flipY: false})
        expect(ir.computeSteps.length).toBe(0)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/unpremultiply/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_0/)
        expect(finalWgsl).not.toMatch(/compute_0/)
    })
})

describe('TimeTrail (c) compileTime tint mode branch', () => {
    it('Color mode emits the luma-weighted tint recolor; None does not', () => {
        // The colorize stage is declaration-site algebra: a `tinted` local recolors the trail by
        // its luminance toward the tint uniform.
        const color = tree(); color[1].props = {tintMode: 'color'}
        const {registry: r1, root: root1} = buildRegistry(color)
        expect(tgpu.resolve([composeNodeTree(r1, composeOpts(root1)).finalPass.entry], {names: 'strict'})).toMatch(/tinted/)
        const {registry: r2, root: root2} = buildRegistry(tree())
        expect(tgpu.resolve([composeNodeTree(r2, composeOpts(root2)).finalPass.entry], {names: 'strict'})).not.toMatch(/tinted/)
    })
})

describe('TimeTrail (d) kernel + helpers resolve (compute-simulation rules)', () => {
    it('the feedback kernel resolves with textureStore + the named step', () => {
        const wgsl = tgpu.resolve([timeTrailKernel], {names: 'strict'})
        expect(wgsl).toMatch(/timeTrailStep/)
        expect(wgsl).toMatch(/textureStore/)
        // Motion source machinery: frame-difference mask + the prev-live copy (3 stores total).
        expect(wgsl).toMatch(/smoothstep/)
        expect((wgsl.match(/textureStore/g) ?? []).length).toBe(3)
        expect(wgsl).toMatchSnapshot('timeTrailKernel')
    })

    it('both trail sources compose (compile-time select)', () => {
        for (const trailSource of ['motion', 'alpha']) {
            const specs = tree()
            specs[1].props = {trailSource}
            const {registry, root} = buildRegistry(specs)
            expect(composeNodeTree(registry, composeOpts(root)).computeSteps.length).toBe(1)
        }
    })

    it('trailBlend routes through the kit blend fns — multiply bakes multiply; add bakes linearDodge', () => {
        const mul = tree(); mul[1].props = {trailBlend: 'multiply'}
        const {registry: r1, root: root1} = buildRegistry(mul)
        expect(tgpu.resolve([composeNodeTree(r1, composeOpts(root1)).finalPass.entry], {names: 'strict'})).toMatch(/multiply/)
        const add = tree(); add[1].props = {trailBlend: 'add'}
        const {registry: r2, root: root2} = buildRegistry(add)
        expect(tgpu.resolve([composeNodeTree(r2, composeOpts(root2)).finalPass.entry], {names: 'strict'})).toMatch(/linearDodge/)
    })
})

describe('TimeTrail (e) CPU goldens — composite order + mask + hue rotation', () => {
    // The fragment composites with the kit blend fns (straight-alpha in, premultiplied out).
    // These goldens pin the ORDER wiring: Motion = blendFn(live, trail, trailOpacity);
    // Alpha = blendFn(trail·op, live, 1).
    it('Motion order: the ghost trail shows OVER an opaque live frame; trailOpacity 0 → pure live', () => {
        const live = d.vec4f(0.2, 0.4, 0.6, 1) // straight
        const trail = d.vec4f(1.0, 0.0, 0.0, 0.5) // straight half-alpha red ghost
        const out = blend.normal(live, trail, 1) as unknown as {x: number; y: number; z: number; w: number}
        // trail over live (premult out): rgb = trail·0.5 + live·1·0.5 → (0.6, 0.2, 0.3), alpha 1.
        expect(out.w).toBeCloseTo(1, 5)
        expect(out.x).toBeCloseTo(0.6, 5)
        expect(out.y).toBeCloseTo(0.2, 5)
        expect(out.z).toBeCloseTo(0.3, 5)
        const off = blend.normal(live, trail, 0) as unknown as {x: number; y: number; z: number}
        expect(off.x).toBeCloseTo(0.2, 5)
        expect(off.y).toBeCloseTo(0.4, 5)
        expect(off.z).toBeCloseTo(0.6, 5)
    })
    it('Motion order: multiply lays dark ink ghosts onto the live frame', () => {
        const live = d.vec4f(0.8, 0.8, 0.8, 1)
        const trail = d.vec4f(0.2, 0.2, 0.2, 1) // dark opaque ghost
        const out = blend.multiply(live, trail, 1) as unknown as {x: number}
        expect(out.x).toBeCloseTo(0.16, 5) // 0.8 · 0.2 — darker than both
    })
    it('Alpha order: opaque live fully occludes the trail (normal blend, live as overlay)', () => {
        const live = d.vec4f(0.2, 0.4, 0.6, 1)
        const trail = d.vec4f(0.9, 0.1, 0.1, 1)
        const out = blend.normal(trail, live, 1) as unknown as {x: number; y: number; z: number; w: number}
        expect(out.x).toBeCloseTo(0.2, 5)
        expect(out.y).toBeCloseTo(0.4, 5)
        expect(out.z).toBeCloseTo(0.6, 5)
        expect(out.w).toBeCloseTo(1, 5)
    })
    it('motion mask (kit frameDiffMask): static sheds nothing, changed stamps fully', () => {
        const a = d.vec4f(0.3, 0.5, 0.7, 1)
        expect(frameDiffMask(a, a, 0.06)).toBeCloseTo(0, 6)
        const moved = d.vec4f(0.9, 0.1, 0.2, 1)
        expect(frameDiffMask(moved, a, 0.06)).toBeCloseTo(1, 6)
        // alpha-only change (a shape's edge passing) also triggers.
        const ghosted = d.vec4f(0.3, 0.5, 0.7, 0.5)
        expect(frameDiffMask(ghosted, a, 0.06)).toBeCloseTo(1, 6)
    })
    it('hue rotation (kit hueRotateTurns) by 0 is identity; by 1 turn returns to the same color', () => {
        const rgb = d.vec3f(0.8, 0.2, 0.1)
        const same = hueRotateTurns(rgb, 0) as unknown as {x: number; y: number; z: number}
        expect(same.x).toBeCloseTo(0.8, 4)
        expect(same.y).toBeCloseTo(0.2, 4)
        expect(same.z).toBeCloseTo(0.1, 4)
        const full = hueRotateTurns(rgb, 1) as unknown as {x: number; y: number; z: number}
        expect(full.x).toBeCloseTo(0.8, 4)
        expect(full.y).toBeCloseTo(0.2, 4)
        expect(full.z).toBeCloseTo(0.1, 4)
    })
    it('hueRotateTurns rotates by whole-hue fractions (radians hue, turns amount)', () => {
        // red + 1/3 turn → green, + 2/3 turn → blue (the old fract-on-radians squashed every hue into [0, 1) rad).
        const g = hueRotateTurns(d.vec3f(1, 0, 0), 1 / 3) as unknown as {x: number; y: number; z: number}
        expect(g.x).toBeCloseTo(0, 3)
        expect(g.y).toBeCloseTo(1, 3)
        expect(g.z).toBeCloseTo(0, 3)
        const b = hueRotateTurns(d.vec3f(1, 0, 0), 2 / 3) as unknown as {x: number; y: number; z: number}
        expect(b.x).toBeCloseTo(0, 3)
        expect(b.y).toBeCloseTo(0, 3)
        expect(b.z).toBeCloseTo(1, 3)
    })
    // The Color-mode colorize stage is declaration-site algebra now — its math (luma-weighted
    // tint, premultiplied) is pinned by the tint-branch WGSL assert + the final-pass snapshot.
})
