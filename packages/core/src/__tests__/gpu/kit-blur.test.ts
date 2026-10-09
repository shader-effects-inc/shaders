import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {
    buildFixedWeights,
    buildTruncatedWeights,
    buildFixedBlurGraph,
    buildVariableBlurGraph,
    DEFAULT_COMPUTE_WIDTH,
    DEFAULT_COMPUTE_HEIGHT,
    gaussianTapWeights,
    unrolledTapGather,
    mapSourceScalar,
    applyRemapWindow,
    remapWindowValues,
    MAP_SOURCE_DIM_FIELDS,
    REMAP_WINDOW_FIELDS,
    buildBloomExtractGraph,
    buildBloomExtractMapGraph,
    bloomExtractSoftKnee,
    aspectAwareComputeRes,
} from '@coreroot/gpu/kit/blur'

/**
 * B4 kit/blur tests.
 *  1. CPU golden-value: weight generation (fixed + truncated) vs independently hand-computed
 *     values and invariants, for radius 4 and radius 24 (§8.1 layer 3 — mechanical enforcement
 *     of mathematical equivalence for the shared weight math).
 *  2. GPU-free WGSL snapshots of both kernels of both blur variants via `tgpu.resolve` (§8.1
 *     layer 2) — resolves the inner kernel fns, which is the documented GPU-free path since the
 *     guarded pipelines themselves need a device.
 */

const HALF = 24 // DEFAULT_HALF_KERNEL

function approx(a: number, b: number, eps = 1e-9) {
    return Math.abs(a - b) < eps
}

describe('buildFixedWeights — full symmetric normalized Gaussian', () => {
    it('radius 4 (9 taps): normalized, symmetric, unimodal', () => {
        const w = buildFixedWeights(4)
        expect(w).toHaveLength(9)
        // sum == 1
        expect(approx(w.reduce((a, b) => a + b, 0), 1, 1e-12)).toBe(true)
        // symmetry
        for (let i = 0; i < 4; i++) expect(approx(w[i], w[8 - i])).toBe(true)
        // center is the max
        expect(w[4]).toBeGreaterThan(w[3])
        // neighbour/center ratio == exp(-1/(2σ²)) with σ = 4/3
        const sigma = 4 / 3
        expect(approx(w[5] / w[4], Math.exp(-1 / (2 * sigma * sigma)), 1e-12)).toBe(true)
    })

    it('radius 24 (49 taps): normalized, symmetric, ratio matches σ = 8', () => {
        const w = buildFixedWeights(24)
        expect(w).toHaveLength(49)
        expect(approx(w.reduce((a, b) => a + b, 0), 1, 1e-12)).toBe(true)
        for (let i = 0; i < 24; i++) expect(approx(w[i], w[48 - i])).toBe(true)
        const sigma = 24 / 3
        expect(approx(w[25] / w[24], Math.exp(-1 / (2 * sigma * sigma)), 1e-12)).toBe(true)
    })
})

describe('buildTruncatedWeights — 3σ active-half truncation + renormalization', () => {
    it('radius 4 → sigma 2.0: activeHalf 6, hand-computed center weight, zero outside window', () => {
        const sigma = 2.0 // sigmaH for pixelRadius 4 (pixelRadius * 0.5)
        const {weights, activeHalf} = buildTruncatedWeights(HALF, sigma)
        expect(weights).toHaveLength(49)
        expect(activeHalf).toBe(6) // ceil(3 * 2.0) = 6, clamped into [1, 24]

        // Independent hand computation of the unnormalized Gaussian over the active window.
        let total = 0
        for (let i = -6; i <= 6; i++) total += Math.exp(-(i * i) / (2 * sigma * sigma))
        const center = 1 / total
        const edge = Math.exp(-(6 * 6) / (2 * sigma * sigma)) / total

        expect(approx(weights[HALF], center, 1e-9)).toBe(true) // index 24 = center
        expect(approx(weights[HALF + 6], edge, 1e-9)).toBe(true) // last active tap
        // Everything beyond the active window is exactly zero.
        for (let i = 7; i <= 24; i++) {
            expect(weights[HALF + i]).toBe(0)
            expect(weights[HALF - i]).toBe(0)
        }
        // Renormalized over the truncated support.
        expect(approx(weights.reduce((a, b) => a + b, 0), 1, 1e-9)).toBe(true)
        // Symmetric within the active window.
        for (let i = 1; i <= 6; i++) expect(approx(weights[HALF + i], weights[HALF - i])).toBe(true)
    })

    it('radius 24 → sigma 12.0: activeHalf clamped to 24 (full kernel), all taps nonzero', () => {
        const sigma = 12.0 // sigmaH for pixelRadius 24
        const {weights, activeHalf} = buildTruncatedWeights(HALF, sigma)
        expect(activeHalf).toBe(24) // ceil(36)=36 clamped to 24
        expect(approx(weights.reduce((a, b) => a + b, 0), 1, 1e-12)).toBe(true)
        for (let i = 0; i < 49; i++) expect(weights[i]).toBeGreaterThan(0)
        // neighbour/center ratio == exp(-1/(2σ²))
        expect(approx(weights[HALF + 1] / weights[HALF], Math.exp(-1 / (2 * sigma * sigma)), 1e-12)).toBe(true)
    })

    it('tiny sigma clamps activeHalf to at least 1', () => {
        const {activeHalf, weights} = buildTruncatedWeights(HALF, 0.001)
        expect(activeHalf).toBe(1)
        expect(approx(weights.reduce((a, b) => a + b, 0), 1, 1e-9)).toBe(true)
    })
})

describe('WGSL resolve snapshots (GPU-free)', () => {
    it('fixed Gaussian — horizontal + vertical kernels', () => {
        const g = buildFixedBlurGraph(HALF, DEFAULT_COMPUTE_WIDTH, DEFAULT_COMPUTE_HEIGHT)
        expect(tgpu.resolve([g.hLayout, g.kernelH], {names: 'strict'})).toMatchSnapshot('fixed-horizontal')
        expect(tgpu.resolve([g.vLayout, g.kernelV], {names: 'strict'})).toMatchSnapshot('fixed-vertical')
    })

    it('variable Gaussian — horizontal + vertical kernels (runtime jitter uniform, one load per tap)', () => {
        const g = buildVariableBlurGraph(HALF, DEFAULT_COMPUTE_WIDTH, DEFAULT_COMPUTE_HEIGHT)
        const h = tgpu.resolve([g.hLayout, g.kernelH], {names: 'strict'})
        const v = tgpu.resolve([g.vLayout, g.kernelV], {names: 'strict'})
        // The comb dither is scaled by the `jitter` uniform (exactly the undithered comb at 0).
        expect(h).toMatch(/\* \(\*p\)\.jitter\)/)
        expect(v).toMatch(/\* params\.jitter\)/)
        expect(h).not.toMatch(/dispersion\)/)
        expect(h).toMatchSnapshot('variable-horizontal')
        expect(v).toMatchSnapshot('variable-vertical')
    })

    it('variable Gaussian (chromatic) — three loads per tap, red/blue radii spread by dispersion', () => {
        const g = buildVariableBlurGraph(HALF, DEFAULT_COMPUTE_WIDTH, DEFAULT_COMPUTE_HEIGHT, {chromatic: true})
        const h = tgpu.resolve([g.hLayout, g.kernelH], {names: 'strict'})
        const v = tgpu.resolve([g.vLayout, g.kernelV], {names: 'strict'})
        expect(h).toMatch(/variableBlurHChromatic/)
        expect(v).toMatch(/variableBlurVChromatic/)
        expect(h).toMatch(/\(1f \+ \(\*p\)\.dispersion\)/)
        expect(h).toMatch(/\(1f - \(\*p\)\.dispersion\)/)
        // Blur loop: three loads per tap (plus the passthrough branch's single load).
        expect((h.match(/textureLoad\(input/g) ?? []).length).toBe(4)
        expect(h).toMatchSnapshot('variable-horizontal-chromatic')
        expect(v).toMatchSnapshot('variable-vertical-chromatic')
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// Phase 11 additions: tap gather, map-radius plumbing, bloom extract.
// ═══════════════════════════════════════════════════════════════════════════════════════

describe('gaussianTapWeights — CPU golden + the bit-exactness contract', () => {
    it('reproduces the hand-written (sigma 0.8, 32 tap) table BIT-for-bit', () => {
        // The three motion blurs shipped this literal expression. The weights are emitted into the
        // WGSL at full double precision, so anything short of bit-equality moves their snapshots.
        const raw = Array.from({length: 32}, (_, i) => {
            const t = (i / 31 - 0.5) * 2
            return Math.exp((-0.5 * t * t) / 0.64)
        })
        const sum = raw.reduce((a, b) => a + b)
        const expected = raw.map((w) => w / sum)
        const actual = gaussianTapWeights(32, 0.8)
        expect(actual).toHaveLength(32)
        for (let i = 0; i < 32; i++) expect(Object.is(actual[i], expected[i])).toBe(true)
    })

    it('normalized, symmetric, peaked at the centre', () => {
        const w = gaussianTapWeights(9, 1)
        expect(approx(w.reduce((a, b) => a + b, 0), 1, 1e-12)).toBe(true)
        for (let i = 0; i < 4; i++) expect(approx(w[i], w[8 - i])).toBe(true)
        expect(w[4]).toBeGreaterThan(w[3])
    })

    it('σ² is rounded to 15 significant digits, so 0.8 divides by exactly 0.64', () => {
        // `0.8 * 0.8` is 0.6400000000000001 — the rounding is what makes the migration Gate A.
        const rounded = gaussianTapWeights(3, 0.8)
        const naive = (() => {
            const raw = [-1, 0, 1].map((t) => Math.exp((-0.5 * t * t) / (0.8 * 0.8)))
            const sum = raw.reduce((a, b) => a + b)
            return raw.map((w) => w / sum)
        })()
        expect(Object.is(rounded[0], naive[0])).toBe(false)
    })
})

describe('unrolledTapGather — flat weighted sum, no control flow', () => {
    /** A minimal Expr stand-in that records the algebra instead of emitting WGSL. */
    type Rec = {readonly s: string; mul(w: number): Rec; add(o: Rec): Rec; member(c: string): Rec}
    const rec = (s: string): Rec => ({
        s,
        mul: (w: number) => rec(`(${s} * ${w})`),
        add: (o: Rec) => rec(`(${s} + ${o.s})`),
        member: (c: string) => rec(`${s}.${c}`),
    })

    it('left-folds in tap order (float addition is not associative)', () => {
        const out = unrolledTapGather({
            weights: [0.25, 0.5, 0.25],
            tapCoord: (i) => rec(`uv${i}`) as never,
            sample: (coord) => rec(`tex(${(coord as unknown as Rec).s})`) as never,
        }) as unknown as Rec
        expect(out.s).toBe('(((tex(uv0) * 0.25) + (tex(uv1) * 0.5)) + (tex(uv2) * 0.25))')
    })

    it('a component selector reduces each tap before weighting (DropShadow alpha path)', () => {
        const out = unrolledTapGather({
            weights: [1, 2],
            tapCoord: (i) => rec(`uv${i}`) as never,
            sample: (coord) => rec(`tex(${(coord as unknown as Rec).s})`) as never,
            component: 'a',
        }) as unknown as Rec
        expect(out.s).toBe('((tex(uv0).a * 1) + (tex(uv1).a * 2))')
    })
})

describe('map-radius plumbing', () => {
    it('the field groups spread in the order the fill kernels declared them', () => {
        expect(Object.keys(MAP_SOURCE_DIM_FIELDS)).toEqual(['inputWidth', 'inputHeight'])
        expect(Object.keys(REMAP_WINDOW_FIELDS)).toEqual(['inputMin', 'inputMax', 'outputMin', 'outputMax', 'curve'])
    })

    it('mapSourceScalar memoizes per channel and emits one distinct fn each', () => {
        const channels = ['luminance', 'luminanceInverted', 'alpha', 'alphaInverted'] as const
        for (const c of channels) expect(mapSourceScalar(c)).toBe(mapSourceScalar(c))
        const wgsl = channels.map((c) => tgpu.resolve([mapSourceScalar(c)], {names: 'strict'}))
        expect(new Set(wgsl).size).toBe(4)
        expect(wgsl[0]).toMatch(/dot\(sample\.xyz/)
        expect(wgsl[2]).toMatch(/sample\.w/)
        // Names are distinct so two channels can coexist in one tree without colliding.
        expect(wgsl[0]).toMatch(/mapSourceScalar_luminance/)
        expect(wgsl[3]).toMatch(/mapSourceScalar_alphaInverted/)
    })

    it('applyRemapWindow is CPU-executable and matches the fragment path', () => {
        const remap = (raw: number, inMin: number, inMax: number, outMin: number, outMax: number, curve: number) => {
            const range = Math.max(inMax - inMin, 0.0001)
            const n = Math.min(1, Math.max(0, (raw - inMin) / range))
            const eased = Math.pow(n, Math.pow(2, -curve * 2))
            return outMin + (outMax - outMin) * eased
        }
        const cases: [number, number, number, number, number, number][] = [
            [0.5, 0, 1, 0, 100, 0],
            [0.25, 0, 1, 10, 50, 1],
            [0.9, 0.2, 0.8, 0, 36, -1],
            [0.5, 0.5, 0.5, 0, 10, 0], // zero-width input window → guarded divide
        ]
        for (const c of cases) {
            expect(applyRemapWindow(...c) as unknown as number).toBeCloseTo(remap(...c), 4)
        }
    })

    it('remapWindowValues carries exactly the five window fields', () => {
        const w = {inputMin: 0.1, inputMax: 0.9, outputMin: 2, outputMax: 8, curve: -0.5}
        expect(remapWindowValues({...w, extra: 1} as never)).toEqual(w)
    })
})

describe('bloom extract graphs (GPU-free)', () => {
    it('uniform-radius extract resolves, names its kernel, writes bright + radius', () => {
        const g = buildBloomExtractGraph(1024, 640, 'testBloomExtract')
        const wgsl = tgpu.resolve([g.layout, g.kernel], {names: 'strict'})
        expect(wgsl).toMatch(/fn testBloomExtract/)
        expect(wgsl).toMatch(/bloomExtractSoftKnee/)
        expect((wgsl.match(/textureStore/g) ?? []).length).toBe(2)
        expect(wgsl).toMatchSnapshot('bloom-extract-uniform')
    })

    it('map-driven extract adds the source tap and the shared remap', () => {
        const g = buildBloomExtractMapGraph(1024, 640, 'testBloomExtractMap', 'alpha')
        const wgsl = tgpu.resolve([g.layout, g.kernel], {names: 'strict'})
        expect(wgsl).toMatch(/fn testBloomExtractMap/)
        expect(wgsl).toMatch(/mapSourceScalar_alpha/)
        expect(wgsl).toMatch(/applyRemapWindow/)
        expect(wgsl).toMatchSnapshot('bloom-extract-map')
    })

    it('bloomExtractSoftKnee area-averages a 4×4 footprint and soft-knees the threshold', () => {
        const wgsl = tgpu.resolve([bloomExtractSoftKnee], {names: 'strict'})
        expect((wgsl.match(/for \(/g) ?? []).length).toBe(2) // the 4×4 grid
        expect(wgsl).toMatch(/0\.0625f/) // 1/16, the area average
        expect(wgsl).toMatch(/dot\(sample\.xyz/) // luminance-normalized mask
    })
})

describe('aspectAwareComputeRes', () => {
    it('caps the LONGER edge and derives the other from the aspect', () => {
        expect(aspectAwareComputeRes(1920, 1080)).toEqual({width: 1024, height: 576})
        expect(aspectAwareComputeRes(1080, 1920)).toEqual({width: 576, height: 1024})
        expect(aspectAwareComputeRes(1000, 1000)).toEqual({width: 1024, height: 1024})
    })

    it('never returns a degenerate size (a canvas can initialize at zero height)', () => {
        expect(aspectAwareComputeRes(1, 0).width).toBeGreaterThanOrEqual(8)
        expect(aspectAwareComputeRes(1, 0).height).toBeGreaterThanOrEqual(8)
        expect(aspectAwareComputeRes(1, 100000)).toEqual({width: 8, height: 1024})
    })
})
