import {describe, it, expect, vi} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type UniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, collectStructuralHashInputs, call} from '@coreroot/gpu/composer'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import Dither from '@coreroot/shaders/Dither/index'
import {ditherBayerQuad, ditherPeriodic, ditherOrderedResult, ditherFloydSteinberg, ditherComposeCustom, ditherComposeSource, ditherLuma} from '@coreroot/gpu/kit/patternPaints'

/**
 * Dither port gate (Phase D2-C). RTT stylization, blendWithChildren:true, compileTime pattern (7)
 * + colorMode (2). Periodic patterns computed procedurally (no DataTexture); Floyd-Steinberg uses a
 * local-array serpentine diffusion body fed by 64 builder-sampled luminances (arrayExpr). Leaf-sibling
 * test proves over-child composite; CPU golden the pure math incl. the full FS diffusion.
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

const DI = Dither as GpuShaderDefinition
const resolveWith = (props: Record<string, unknown>): {ir: ReturnType<typeof composeNodeTree>; wgsl: string} => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'gen', def: Generator, parentId: 'root', metadata: {renderOrder: 0}},
        {id: 'di', def: DI, parentId: 'root', props, metadata: {renderOrder: 1}},
    ])
    const ir = composeNodeTree(registry)
    return {ir, wgsl: tgpu.resolve([ir.finalPass.entry], {names: 'strict'})}
}

describe('Dither (a) default (bayer4, custom), shipped-v1 replace over a preceding sibling (blendWithChildren:false parity)', () => {
    it('RTTs + unpremultiplies the child, dithers, REPLACES the sibling (shipped v1 parity)', () => {
        const {ir, wgsl} = resolveWith({})
        expect(ir.rttPasses.length).toBeGreaterThanOrEqual(1)
        // The sibling is drawn once, into the RTT the filter reads: at full opacity the filter
        // replaces it outright, so the final pass doesn't evaluate it again underneath.
        expect(wgsl).not.toMatch(/genBody/)
        expect(tgpu.resolve([ir.rttPasses[0].fragment.entry], {names: 'strict'})).toMatch(/genBody/)
        expect(wgsl).toMatch(/ditherPeriodic/)
        expect(wgsl).toMatch(/ditherComposeCustom/)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
        expect(wgsl).toMatch(/textureSample\(rtt_/)
        expect(wgsl).toMatchSnapshot('dither-bayer4-custom')
    })
})

describe('Dither (b) compile-time pattern branches', () => {
    it('bayer2/4/8 + clusteredDot use the periodic body', () => {
        for (const p of ['bayer2', 'bayer4', 'bayer8', 'clusteredDot']) {
            expect(resolveWith({pattern: p}).wgsl).toMatch(/ditherPeriodic/)
        }
    })
    it('blueNoise / whiteNoise use their hash bodies (not periodic)', () => {
        expect(resolveWith({pattern: 'blueNoise'}).wgsl).toMatch(/ditherBlueNoise/)
        expect(resolveWith({pattern: 'whiteNoise'}).wgsl).toMatch(/ditherWhiteNoise/)
    })
    it('floydSteinberg emits the diffusion body + a 64-element luminance array', () => {
        const {wgsl} = resolveWith({pattern: 'floydSteinberg'})
        expect(wgsl).toMatch(/ditherFloydSteinberg/)
        expect(wgsl).toMatch(/array<f32, 64>/)
        expect(wgsl).not.toMatch(/ditherPeriodic/)
        expect(wgsl).toMatchSnapshot('dither-floydsteinberg')
    })
    it('pattern is part of the structural (recompile) hash', () => {
        const build = (pattern: string) => {
            const {registry} = buildRegistry([
                {id: 'root', def: RootContainer, parentId: null},
                {id: 'gen', def: Generator, parentId: 'root', metadata: {renderOrder: 0}},
                {id: 'di', def: DI, parentId: 'root', props: {pattern}, metadata: {renderOrder: 1}},
            ])
            return collectStructuralHashInputs(registry).join('\n')
        }
        expect(build('bayer4')).not.toBe(build('floydSteinberg'))
    })
})

describe('Dither (c) colorMode branch', () => {
    it('source colorMode uses the source-compose body, not custom', () => {
        const {wgsl} = resolveWith({colorMode: 'source'})
        expect(wgsl).toMatch(/ditherComposeSource/)
        expect(wgsl).not.toMatch(/ditherComposeCustom/)
    })
})

describe('Dither (d) CPU golden — pattern math + FS diffusion', () => {
    it('bayer quad + periodic bayer2 reproduce the v1 formula', () => {
        // bQ(1,1) = 1*3 + 1*2 - 1*1*4 = 1
        expect(ditherBayerQuad(1, 1) as number).toBeCloseTo(1, 6)
        // coord (1,1) → tx=1,ty=1 → bayer2 = bQ(1,1)/4 = 0.25
        const p = ditherPeriodic(d.vec2f(1, 1)) as {x: number}
        expect(p.x).toBeCloseTo(0.25, 6)
    })

    it('clustered-dot lookup matches the v1 matrix at a couple of cells', () => {
        // cM[0][0]=12 → 0.75; cM[3][2]=9 → 0.5625.
        expect((ditherPeriodic(d.vec2f(0, 0)) as {w: number}).w).toBeCloseTo(0.75, 6)
        expect((ditherPeriodic(d.vec2f(2, 3)) as {w: number}).w).toBeCloseTo(0.5625, 6)
    })

    it('ordered result = step(adjustedDither, biased luminance)', () => {
        // ditherValue 0.5, spread 1 → adjusted 0.5; luminance 0.7, threshold 0.5 → biased 0.7 ≥ 0.5 → 1
        expect(ditherOrderedResult(0.5, 0.7, 0.5, 1) as number).toBe(1)
        // biased below threshold → 0
        expect(ditherOrderedResult(0.9, 0.1, 0.5, 1) as number).toBe(0)
    })

    it('compose bodies match v1', () => {
        const custom = ditherComposeCustom(d.vec4f(0, 0, 0, 0), d.vec4f(1, 1, 1, 1), 1) as {x: number; w: number}
        expect(custom.x).toBeCloseTo(1, 6)
        expect(custom.w).toBeCloseTo(1, 6)
        const source = ditherComposeSource(d.vec4f(0.5, 0.5, 0.5, 0.8), 0) as {x: number; w: number}
        expect(source.x).toBeCloseTo(0.5 * 0.3, 6) // ditherResult 0 → dark
        expect(source.w).toBeCloseTo(0.8, 6)
    })

    it('ditherFloydSteinberg matches an independent serpentine transcription', () => {
        // Reference FS over an 8x8 block, transcribed from v1.
        const fsGolden = (lum: number[], lic: number, threshold: number, spread: number): number => {
            const B = 8
            const thr = 0.5 - (threshold - 0.5)
            const err = new Array(B * B).fill(0)
            const q = new Array(B * B).fill(0)
            for (let y = 0; y < B; y++) {
                const ltr = y % 2 === 0
                for (let xi = 0; xi < B; xi++) {
                    const x = ltr ? xi : B - 1 - xi
                    const idx = y * B + x
                    const val = lum[idx] + err[idx]
                    const qv = val >= thr ? 1 : 0
                    q[idx] = qv
                    const e = (val - qv) * spread
                    const dir = ltr ? 1 : -1
                    const xr = x + dir
                    if (xr >= 0 && xr < B) err[y * B + xr] += e * (7 / 16)
                    if (y + 1 < B) {
                        const xbl = x - dir
                        const xbr = x + dir
                        if (xbl >= 0 && xbl < B) err[(y + 1) * B + xbl] += e * (3 / 16)
                        err[(y + 1) * B + x] += e * (5 / 16)
                        if (xbr >= 0 && xbr < B) err[(y + 1) * B + xbr] += e * (1 / 16)
                    }
                }
            }
            return q[lic]
        }

        // Exhaustive equivalence: EVERY one of the 64 cells, across several luminance fields and
        // threshold/spread combos, must match the reference. This locks the rolling two-row error
        // buffer + scalar-result rewrite (FIX-G) to the flat 8×8 err/q scan byte-for-byte — the scan
        // visits cells in the identical order and accumulates identical error terms, so the guard is
        // that the working-set reduction never perturbs a single cell.
        const ramp = Array.from({length: 64}, (_, i) => (i % 8) / 7)
        const diag = Array.from({length: 64}, (_, i) => (Math.floor(i / 8) + (i % 8)) / 14)
        const pseudo = Array.from({length: 64}, (_, i) => ((i * 2654435761) % 1000) / 1000) // deterministic hash-ish
        for (const lum of [ramp, diag, pseudo]) {
            for (const [threshold, spread] of [[0.5, 1.0], [0.3, 1.0], [0.7, 0.5], [0.5, 0.0]]) {
                for (let lic = 0; lic < 64; lic++) {
                    const out = ditherFloydSteinberg(lum, lic, threshold, spread) as number
                    expect(out).toBe(fsGolden(lum, lic, threshold, spread))
                }
            }
        }
    })

    it('ditherLuma is Rec.601 luminance × alpha', () => {
        expect(ditherLuma(d.vec4f(0.2, 0.5, 0.9, 0.5)) as number).toBeCloseTo((0.2 * 0.299 + 0.5 * 0.587 + 0.9 * 0.114) * 0.5, 6)
    })
})
