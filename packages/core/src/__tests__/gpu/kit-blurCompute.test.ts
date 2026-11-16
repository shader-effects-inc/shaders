import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {buildBokehPrefilterGraph} from '@coreroot/gpu/kit/blur'
import {
    buildMotionBlurComputeGraph,
    motionBlurStreakCompositeFn,
    motionBlurComputeRes,
    motionBlurSpectrumTint,
    spectrumRgb,
    MOTION_BLUR_COMPUTE_LONG_EDGE,
    type MotionBlurPathKind,
} from '@coreroot/gpu/kit/motionBlur'

/**
 * The reduced-resolution blur passes: the compute-backed motion blur (Zoom/Linear/AngularBlur) and
 * BokehBlur's prefilter. GPU-free gates: every kernel transpiles and resolves, each path kind
 * reaches its own trajectory fn, and the CPU-side sizing and spectrum mirror their references.
 */
describe('motion blur compute — kernels resolve', () => {
    const cases: [MotionBlurPathKind, RegExp][] = [
        ['linear', /linearBlurTapCoord/],
        ['orbit', /angularBlurTapCoord/],
        ['zoom', /zoomBlurSpiralTapCoord/],
    ]
    for (const [kind, trajectory] of cases) {
        it(`${kind}: resolves, walks its own trajectory, samples bilinearly and stores`, () => {
            const graph = buildMotionBlurComputeGraph(kind, 960, 540)
            const wgsl = tgpu.resolve([graph.kernel], {names: 'strict'})
            expect(wgsl).toMatch(trajectory)
            expect(wgsl).toMatch(/motionBlurDetailField/)
            expect(wgsl).toMatch(/motionBlurDetailTapWeight/)
            expect(wgsl).toMatch(/textureSampleLevel/)
            expect(wgsl).toMatch(/textureStore/)
            // The adaptive tap loop, not an unrolled comb.
            expect(wgsl).toMatch(/for\s*\(/)
        })
    }
})

describe('motion blur compute — fragment composite resolves', () => {
    for (const kind of ['linear', 'orbit', 'zoom'] as MotionBlurPathKind[]) {
        it(`${kind}: full-res short gather fading into the buffer`, () => {
            const wgsl = tgpu.resolve([motionBlurStreakCompositeFn(kind) as never], {names: 'strict'})
            expect(wgsl).toMatch(new RegExp(`fn ${kind}MotionBlurComposite`))
            expect(wgsl).toMatch(/textureSampleLevel\(tex, samp/)
            expect(wgsl).toMatch(/smoothstep\(8/)
            expect(wgsl).toMatch(/mix\(/)
        })
    }
})

describe('motion blur compute — buffer size', () => {
    it('is half the canvas, the long edge capped, aspect kept', () => {
        expect(motionBlurComputeRes(1920, 1080)).toEqual({width: 960, height: 540})
        const uhd = motionBlurComputeRes(3840, 2160)
        expect(uhd.width).toBe(MOTION_BLUR_COMPUTE_LONG_EDGE)
        expect(uhd.height).toBe(576)
        expect(motionBlurComputeRes(400, 800)).toEqual({width: 200, height: 400})
        expect(motionBlurComputeRes(4, 4)).toEqual({width: 8, height: 8})
    })
})

describe('motion blur compute — spectral tint', () => {
    it('matches the baked CPU spectrum (spectrumRgb − 1)', () => {
        for (const hue of [0, 0.1, 0.25, 0.4, 0.6, 0.75]) {
            const [r, g, b] = spectrumRgb(hue)
            const out = motionBlurSpectrumTint(hue) as unknown as {x: number; y: number; z: number}
            expect(out.x).toBeCloseTo(r - 1, 5)
            expect(out.y).toBeCloseTo(g - 1, 5)
            expect(out.z).toBeCloseTo(b - 1, 5)
        }
    })
})

describe('motion blur compute — every tap count covers the spectrum', () => {
    it('each channel gets dispersion weight for 4–32 taps (fewer starves a channel: 2 taps → no green)', () => {
        for (let n = 4; n <= 32; n++) {
            const sum = [0, 0, 0]
            for (let i = 0; i < n; i++) {
                const rgb = spectrumRgb((i / (n - 1)) * 0.75)
                for (let c = 0; c < 3; c++) sum[c] += rgb[c]
            }
            for (let c = 0; c < 3; c++) expect(sum[c]).toBeGreaterThan(0.2)
        }
        // The failure this guards against.
        const two = [spectrumRgb(0), spectrumRgb(0.75)]
        expect(two[0][1] + two[1][1]).toBe(0)
    })
})

describe('bokeh prefilter — kernel resolves', () => {
    it('box-filters the child with four bilinear reads into the compute-res buffer', () => {
        const graph = buildBokehPrefilterGraph(1024, 576)
        const wgsl = tgpu.resolve([graph.kernel], {names: 'strict'})
        expect((wgsl.match(/textureSampleLevel\(/g) ?? []).length).toBe(4)
        expect(wgsl).toMatch(/textureStore/)
    })
})

// Keep `d` referenced for the vector casts above (the kit's fns run on the CPU in tests).
void d
