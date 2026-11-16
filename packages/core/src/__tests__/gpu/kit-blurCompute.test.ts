import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {buildBokehPrefilterGraph} from '@coreroot/gpu/kit/blur'
import {
    motionBlurAdaptiveGatherFn,
    motionBlurSpectrumTint,
    spectrumRgb,
    type MotionBlurPathKind,
} from '@coreroot/gpu/kit/motionBlur'

/**
 * The cost-adaptive blur paths: the adaptive motion-blur gather (Zoom/Linear/AngularBlur) and
 * BokehBlur's prefilter. GPU-free gates: every fn transpiles and resolves, each path kind reaches
 * its own trajectory fn, and the GPU spectrum mirrors the baked CPU table.
 */
describe('motion blur adaptive gather — resolves', () => {
    const cases: [MotionBlurPathKind, RegExp][] = [
        ['linear', /linearBlurTapCoord/],
        ['orbit', /angularBlurTapCoord/],
        ['zoom', /zoomBlurSpiralTapCoord/],
    ]
    for (const [kind, trajectory] of cases) {
        it(`${kind}: walks its own trajectory at full resolution, as many taps as the streak is long`, () => {
            const wgsl = tgpu.resolve([motionBlurAdaptiveGatherFn(kind) as never], {names: 'strict'})
            expect(wgsl).toMatch(new RegExp(`fn ${kind}MotionBlurAdaptive`))
            expect(wgsl).toMatch(trajectory)
            expect(wgsl).toMatch(/motionBlurDetailField/)
            expect(wgsl).toMatch(/motionBlurDetailTapWeight/)
            // Explicit-LOD samples of the child in a loop (legal in non-uniform flow), clamped to
            // the unrolled gather's 4..32 tap budget.
            expect(wgsl).toMatch(/textureSampleLevel\(tex, samp/)
            expect(wgsl).toMatch(/for\s*\(/)
            expect(wgsl).toMatch(/clamp\(\(?ceil\(lengthPx\) \+ 1\.0\)?, 4\.0, 32\.0\)/)
        })
    }
})

describe('motion blur adaptive gather — spectral tint', () => {
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

describe('motion blur adaptive gather — every tap count covers the spectrum', () => {
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
