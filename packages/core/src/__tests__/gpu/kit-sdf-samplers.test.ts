import {describe, it, expect, vi} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import * as sdf3d from '@coreroot/gpu/kit/sdf3d'
import {createSvgSdfSampler, createSvgSdfSamplerWithGradients, createAnalyticSdfSampler, buildAnalyticSdfFn} from '@coreroot/gpu/kit/sdf'
import {expr} from '@coreroot/gpu/composer'
import {Expr, type GpuFragmentParams, type KitTexture, type EmitContext} from '@coreroot/gpu/contract'

/**
 * kit SDF-sampler subsystem gates (the shape-effect enablers). GPU-free:
 *   (a) CPU-golden the `createAnalytic3dSdfSetup` shape-JSON → MarchParams resolver (a mock root
 *       captures the uniform writes; auto-animate sub-props resolve over time).
 *   (b) resolve-snapshot a baked analytic-3D sdfFn + its march + the volumetric kernel.
 *   (c) resolve-snapshot the fn-arg field samplers (the consumer-wireable sampler form).
 *   (d) construct the flat SVG samplers + the analytic-2D sampler over STUB params/accessors and
 *       assert they emit a texture-sample / primitive-call Expr.
 * The device-side allocators need a GPU, so — like kit/blur — they are tsc-checked only, not run.
 */

// ── mock root: captures the last MarchParams write; answers the compute allocations ─────────────
function mockRoot() {
    let lastMarch: unknown = null
    const uniform = {
        buffer: {},
        write: (v: unknown) => {
            lastMarch = v
        },
        patch: vi.fn(),
    }
    const texture = {
        $usage: vi.fn(function (this: unknown) {
            return texture
        }),
        destroy: vi.fn(),
        write: vi.fn(),
    }
    const guarded = {
        with: vi.fn(function (this: unknown) {
            return guarded
        }),
        dispatchThreads: vi.fn(),
    }
    const root = {
        createUniform: vi.fn(() => uniform),
        createTexture: vi.fn(() => texture),
        createBindGroup: vi.fn(() => ({})),
        createGuardedComputePipeline: vi.fn(() => guarded),
        device: {},
        getLastMarch: () => lastMarch as Record<string, unknown> | null,
    }
    return root
}

// A minimal EmitContext for standalone Expr emission (the stubs below only need `external`).
const mockCtx: EmitContext = {
    external: (_v: unknown, hint: string) => hint,
    statement: () => {},
    freshLocal: (hint: string) => hint,
    memo: (_k: string, f: () => string) => f(),
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// (a) CPU-golden — createAnalytic3dSdfSetup update resolver
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('kit/sdf3d createAnalytic3dSdfSetup (a) CPU-golden resolver', () => {
    it('writes MarchParams from a static cube3D config (size, rotation, bounding radius)', () => {
        const root = mockRoot()
        // Fully explicit (cube3D's defaults are rotX 25 / rotY 35 — spelled out here to golden them).
        const cfg = {type: 'cube3D', sizeX: 0.3, sizeY: 0.3, sizeZ: 0.3, rounding: 0.05, rotX: 45, rotY: 0, rotZ: 0}
        const setup = sdf3d.createAnalytic3dSdfSetup(root as never, 'cube3D', cfg, () => cfg)
        setup.update({deltaTime: 0})

        const v = root.getLastMarch()!
        expect(v.pA).toBeCloseTo(0.3, 6) // sizeX
        expect(v.pB).toBeCloseTo(0.3, 6) // sizeY
        expect(v.pC).toBeCloseTo(0.3, 6) // sizeZ
        expect(v.pD).toBeCloseTo(0.05, 6) // rounding

        const rad = (45 * Math.PI) / 180
        const rot = v.rot as Record<string, number>
        expect(rot.cx).toBeCloseTo(Math.cos(rad), 6)
        expect(rot.sx).toBeCloseTo(Math.sin(rad), 6)
        expect(rot.cy).toBeCloseTo(1, 6) // rotY = 0
        expect(rot.sy).toBeCloseTo(0, 6)
        expect(rot.cz).toBeCloseTo(1, 6) // rotZ = 0

        // rBound = shape3dBoundingRadius(cfg) + 0.02 (the setup's pad).
        const expectedRB = sdf3d.shape3dBoundingRadius(cfg) + 0.02
        expect(v.rBound).toBeCloseTo(expectedRB, 6)
        expect(expectedRB).toBeCloseTo(Math.hypot(0.3, 0.3, 0.3) + 0.05 + 0.02, 6)

        // Aspect-fit domain: rPad = rBound + 0.05, span = 2·rPad, origin = 0.5 − rPad.
        const rPad = expectedRB + 0.05
        expect(v.spanX).toBeCloseTo(rPad * 2, 6)
        expect(v.originX).toBeCloseTo(0.5 - rPad, 6)
    })

    it('resolves an auto-animate sub-prop over time (radius moves as the clock advances)', () => {
        const root = mockRoot()
        const cfg = {
            type: 'sphere3D',
            radius: {type: 'auto-animate', speed: 1, mode: 'loop', easing: 'linear', outputMin: 0.2, outputMax: 0.5},
            rotX: 0,
        }
        const setup = sdf3d.createAnalytic3dSdfSetup(root as never, 'sphere3D', cfg, () => cfg)

        setup.update({deltaTime: 0}) // elapsed 0 → phase 0 → radius = outputMin
        const r0 = root.getLastMarch()!.pA as number
        expect(r0).toBeCloseTo(0.2, 6)

        setup.update({deltaTime: 1}) // elapsed 1 → globalT 0.2 → linear → 0.2 + 0.2·0.3
        const r1 = root.getLastMarch()!.pA as number
        expect(r1).toBeCloseTo(0.26, 6)
        expect(r1).toBeGreaterThan(r0)
    })

    it('setActiveRes re-writes the uniform without advancing the auto clock', () => {
        const root = mockRoot()
        const cfg = {
            type: 'sphere3D',
            radius: {type: 'auto-animate', speed: 1, mode: 'loop', easing: 'linear', outputMin: 0.2, outputMax: 0.5},
        }
        const setup = sdf3d.createAnalytic3dSdfSetup(root as never, 'sphere3D', cfg, () => cfg)
        setup.update({deltaTime: 1})
        const before = root.getLastMarch()!.pA as number
        setup.setActiveRes(512)
        const v = root.getLastMarch()!
        expect(v.activeRes).toBe(512)
        expect(v.pA).toBeCloseTo(before, 9) // clock not advanced → radius unchanged
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (b) Resolve gate — baked analytic-3D sdfFn + march + volumetric kernel
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('kit/sdf3d createAnalytic3dSdfSetup (b) baked sdfFn resolves', () => {
    it('cube3D sdfFn + march + kernel resolve to WGSL (snapshot)', () => {
        const root = mockRoot()
        const setup = sdf3d.createAnalytic3dSdfSetup(root as never, 'cube3D', {type: 'cube3D'}, () => ({type: 'cube3D'}))
        const march = sdf3d.buildRaymarchedFieldFn(setup.sdfFn, 'raw')
        const kernel = sdf3d.buildVolumetricFieldKernel(setup.layout, march)
        const wgsl = tgpu.resolve([setup.sdfFn, march, kernel], {names: 'strict'})
        expect(typeof wgsl).toBe('string')
        expect(wgsl).toMatch(/fn/)
        expect(wgsl).toMatch(/textureStore/)
        expect(wgsl).toMatchSnapshot()
    })

    it('metaballs3D sdfFn (reads mb0..mb7) resolves', () => {
        const root = mockRoot()
        const setup = sdf3d.createAnalytic3dSdfSetup(root as never, 'metaballs3D', {type: 'metaballs3D'}, () => ({type: 'metaballs3D'}))
        const wgsl = tgpu.resolve([setup.sdfFn], {names: 'strict'})
        expect(wgsl).toMatch(/fn/)
    })

    it('metaballs3D ball count: active balls orbit, surplus balls park outside the bounding sphere', () => {
        const mbAt = (v: Record<string, unknown>, i: number) => v[`mb${i}`] as {x: number; y: number; z: number}

        // Default (no `balls` key) → 4 active at the original i·π/2 phasing, 4 parked.
        const root4 = mockRoot()
        const cfg4 = {type: 'metaballs3D'}
        sdf3d.createAnalytic3dSdfSetup(root4 as never, 'metaballs3D', cfg4, () => cfg4).update({deltaTime: 0})
        const v4 = root4.getLastMarch()!
        const spread = sdf3d.SHAPE3D_DEFAULTS.metaballs3D.spread
        // ball 1 at elapsed 0: ang = π/2, wob = 0.75 + 0.25·sin(1.7)
        const wob1 = 0.75 + 0.25 * Math.sin(1.7)
        expect(mbAt(v4, 1).x).toBeCloseTo(spread * Math.cos(Math.PI / 2) * wob1, 6)
        expect(mbAt(v4, 1).z).toBeCloseTo(spread * Math.sin(Math.PI / 2) * wob1, 6)
        for (const i of [4, 5, 6, 7]) expect(mbAt(v4, i).y).toBe(99)

        // balls: 6 → six active (even 2π/6 spacing), two parked.
        const root6 = mockRoot()
        const cfg6 = {type: 'metaballs3D', balls: 6}
        sdf3d.createAnalytic3dSdfSetup(root6 as never, 'metaballs3D', cfg6, () => cfg6).update({deltaTime: 0})
        const v6 = root6.getLastMarch()!
        expect(mbAt(v6, 5).y).not.toBe(99)
        const wob5 = 0.75 + 0.25 * Math.sin(5 * 1.7)
        expect(mbAt(v6, 5).x).toBeCloseTo(spread * Math.cos(5 * (Math.PI / 3)) * wob5, 6)
        for (const i of [6, 7]) expect(mbAt(v6, i).y).toBe(99)

        // balls: 1 → a lone ball, everything else parked.
        const root1 = mockRoot()
        const cfg1 = {type: 'metaballs3D', balls: 1}
        sdf3d.createAnalytic3dSdfSetup(root1 as never, 'metaballs3D', cfg1, () => cfg1).update({deltaTime: 0})
        const v1 = root1.getLastMarch()!
        expect(mbAt(v1, 0).y).not.toBe(99)
        for (const i of [1, 2, 3, 4, 5, 6, 7]) expect(mbAt(v1, i).y).toBe(99)
    })

    it('SVG-3D extrusion sdfFn (samples the sdfSource DATA texture) + march + kernel resolve (snapshot)', () => {
        const root = mockRoot()
        const params = svgStubParams(root)
        const setup = sdf3d.createSvg3dSdfSetup(params, '', () => ({type: 'svgExtrude3D', depth: 0.12, bevel: 0.02}))
        const march = sdf3d.buildRaymarchedFieldFn(setup.sdfFn, 'none')
        const kernel = sdf3d.buildVolumetricFieldKernel(setup.layout, march)
        const wgsl = tgpu.resolve([setup.sdfFn, march, kernel], {names: 'strict'})
        expect(wgsl).toMatch(/textureLoad/) // the extrusion samples the 2D field
        expect(wgsl).toMatch(/textureStore/) // the kernel writes the volumetric field
        expect(wgsl).toMatchSnapshot()
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (c) Resolve gate — the fn-arg field samplers (the consumer-wireable sampler form)
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('kit/sdf3d buildFieldSampleGraphArgs (c) fn-arg samplers resolve', () => {
    it('bicubic + analytic-gradient + bilinear + texelSpan resolve to WGSL with textureLoad (snapshot)', () => {
        const {fieldSampleArg, fieldSampleGradArg, fieldSampleFastArg, texelSpanFn} = sdf3d.buildFieldSampleGraphArgs()
        const wgsl = tgpu.resolve([fieldSampleArg, fieldSampleGradArg, fieldSampleFastArg, texelSpanFn], {names: 'strict'})
        expect(typeof wgsl).toBe('string')
        expect(wgsl).toMatch(/textureLoad/)
        expect(wgsl).toMatchSnapshot()
    })

    it('the analytic depth slope resolves: 16 texel loads of .a, a B-spline derivative, a vec2f', () => {
        const {fieldDepthSlopeArg} = sdf3d.buildFieldSampleGraphArgs()
        const wgsl = tgpu.resolve([fieldDepthSlopeArg], {names: 'strict'})
        expect(wgsl).toMatch(/fn fieldDepthSlopeArg\([^)]*\) -> vec2f/)
        expect(wgsl.match(/textureLoad\(/g)?.length).toBe(16)
        expect(wgsl).toMatch(/\.a;/)
        expect(wgsl).toMatchSnapshot()
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (d) Construction gate — flat SVG samplers + analytic-2D sampler over stubs
// ═══════════════════════════════════════════════════════════════════════════════════════

// Stub GpuFragmentParams: enough for the SVG data/media texture lifecycle + a fragment sample.
function stubParams(): GpuFragmentParams {
    const mediaTexture = {
        texture: {__mock: 'tgpuTexture'},
        width: 512,
        height: 512,
        write: vi.fn(),
        unwrap: () => ({}) as never,
        destroy: vi.fn(),
    }
    const kitTexture: KitTexture = {
        key: 'media_0',
        sample: (uv: Expr) => new Expr((ctx) => `textureSample(tex.$.media_0, samp.$.linearClamp, ${uv._emit(ctx)})`),
        sampleLevel: (uv: Expr) => new Expr((ctx) => `textureSampleLevel(tex.$.media_0, samp.$.linearClamp, ${uv._emit(ctx)}, 0.0)`),
        accessor: () => new Expr(() => 'tex.$.media_0'),
        dimensions: () => new Expr(() => 'vec2f(512.0, 512.0)'),
    }
    return {
        createDataTexture: () => mediaTexture as never,
        createMediaTexture: () => mediaTexture as never,
        registerMediaTexture: () => kitTexture,
        onBeforeRender: () => {},
        onCleanup: () => {},
    } as unknown as GpuFragmentParams
}

// SVG-3D setup stub: adds a gpu.root (for the shared MarchParams uniform) to the flat stub.
function svgStubParams(root: ReturnType<typeof mockRoot>): GpuFragmentParams {
    return {...(stubParams() as object), gpu: {root, device: {}}} as unknown as GpuFragmentParams
}

describe('kit/sdf flat SVG samplers (d) construct + emit a texture sample', () => {
    // Explicit-LOD (`textureSampleLevel`): the flat SVG field is tapped inside material
    // `guarded` branches (non-uniform control flow), where implicit-derivative
    // `textureSample` is a WGSL validation error — the bug custom shapes hit on Nebula.
    it('createSvgSdfSampler returns a closure that emits an explicit-LOD sample', () => {
        const sampler = createSvgSdfSampler(stubParams(), '')
        const text = sampler(expr('in.uv'))._emit(mockCtx)
        expect(text).toContain('textureSampleLevel')
        expect(text).toContain('in.uv')
    })

    it('createSvgSdfSamplerWithGradients returns a closure that emits an explicit-LOD sample', () => {
        const sampler = createSvgSdfSamplerWithGradients(stubParams(), '')
        const text = sampler(expr('in.uv'))._emit(mockCtx)
        expect(text).toContain('textureSampleLevel')
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (e) createVolumetricFieldComputeNode — CPU orchestration (dirty-key + SampleParams writes)
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('kit/sdf3d createVolumetricFieldComputeNode (e) compute orchestration', () => {
    function computeNodeParams(root: ReturnType<typeof mockRoot>) {
        const extraFields: Record<string, number> = {}
        const computeKit: KitTexture = {
            key: 'compute_0',
            sample: (uv: Expr) => uv,
            accessor: () => new Expr(() => 'tex.$.compute_0'),
            dimensions: () => new Expr(() => 'vec2f(0.0, 0.0)'),
        }
        const params = {
            gpu: {root, device: {}},
            getCpuValue: (k: string) => (k === 'shapeSdfUrl' ? '' : k === 'shapeType' ? 'cube3D' : k === 'scale' ? 1 : undefined),
            registerComputeTexture: vi.fn(() => computeKit),
            setExtraField: (name: string, value: number | number[]) => {
                extraFields[name] = value as number
            },
            onCleanup: () => {},
            dimensions: {width: 800, height: 600},
        } as unknown as GpuFragmentParams
        return {params, extraFields}
    }

    it('builds an analytic-3D field pre-march, dirty-keys, and publishes SampleParams', () => {
        const root = mockRoot()
        const {params, extraFields} = computeNodeParams(root)
        const shape = {type: 'cube3D', sizeX: 0.3}
        const node = sdf3d.createVolumetricFieldComputeNode(params, () => shape)!
        expect(node).not.toBeNull()
        expect((params.registerComputeTexture as unknown as {mock: {calls: unknown[]}}).mock.calls.length).toBe(1)

        // Frame 1: state changed (from init) → returns the dispatch step + writes the 6 SampleParams.
        const step1 = node.getComputeNodes({deltaTime: 0})
        expect(step1?.length).toBe(1)
        expect(typeof step1![0]).toBe('function')
        expect(extraFields._vfSpanX).toBeGreaterThan(0)
        expect(extraFields._vfRBound).toBeCloseTo(sdf3d.shape3dBoundingRadius(shape) + 0.02, 6)
        expect(extraFields._vfActiveRes).toBeGreaterThan(0)

        // Frame 2: identical static shape → dirty-key matches → no re-march.
        const step2 = node.getComputeNodes({deltaTime: 0})
        expect(step2).toBeNull()
    })

    it('returns null for a flat (2D) shape type', () => {
        const root = mockRoot()
        const {params} = computeNodeParams(root)
        // shapeType resolves to cube3D via getCpuValue, but the shape config type wins only when
        // getCpuValue('shapeType') is empty — here force a flat type via getShapeConfig + a params
        // whose shapeType is a 2D SDF.
        const flatParams = {
            ...(params as object),
            getCpuValue: (k: string) => (k === 'shapeSdfUrl' ? '' : k === 'shapeType' ? 'starSDF' : k === 'scale' ? 1 : undefined),
        } as unknown as GpuFragmentParams
        expect(sdf3d.createVolumetricFieldComputeNode(flatParams, () => ({type: 'starSDF'}))).toBeNull()
    })

    it('returns null with no device (GPU-free)', () => {
        const noGpu = {gpu: undefined} as unknown as GpuFragmentParams
        expect(sdf3d.createVolumetricFieldComputeNode(noGpu, () => ({type: 'cube3D'}))).toBeNull()
    })
})

describe('kit/sdf createAnalyticSdfSampler (d) construct + resolve', () => {
    it('emits a call to the shape-specific analytic fn', () => {
        const sampler = createAnalyticSdfSampler('starSDF', {radius: expr('0.3'), sides: expr('5.0'), innerRatio: expr('0.5')})
        const text = sampler(expr('in.uv'))._emit(mockCtx)
        expect(text).toContain('analyticSdf_starSDF')
        expect(text).toContain('in.uv')
    })

    it('the underlying analytic fns transpile to WGSL for representative shapes (snapshot)', () => {
        const fns = ['circleSDF', 'polygonSDF', 'starSDF', 'heartSDF', 'arcSDF', 'trapezoidSDF', 'parallelogramSDF'].map((t) =>
            buildAnalyticSdfFn(t),
        )
        const wgsl = tgpu.resolve(fns, {names: 'strict'})
        expect(wgsl).toMatch(/fn/)
        expect(wgsl).toMatchSnapshot()
    })
})
