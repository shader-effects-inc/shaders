import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import ZoomBlur from '@coreroot/shaders/ZoomBlur/index'
import {
    zoomBlurTapCoord, zoomBlurSpiralTapCoord, MOTION_BLUR_WEIGHTS, MOTION_BLUR_TAP_COUNT,
    MOTION_BLUR_SPECTRUM, MOTION_BLUR_SPECTRUM_REVERSED, MOTION_BLUR_SPIRAL_TWIST, spectrumRgb,
} from '@coreroot/gpu/kit/motionBlur'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * ZoomBlur port gate (W7-D) — an RTT multi-tap FILTER (requiresRTT/requiresChild, no uvRemap), the
 * AngularBlur twin. It RTTs the child and accumulates 32 weighted Gaussian samples scaled radially
 * from `center`, then unpremultiplies (RTT premultiplied — Twirl trap #2). v1's JS `for` loop unrolls
 * at build; this builder unrolls 32 `texture.sample()` calls, each tap's scaled coord from a shared
 * body fn (`1 + radius·(tap/31)`, radius = intensity·0.01). CPU-goldened.
 */
const ZB = ZoomBlur as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

describe('ZoomBlur (a) RTT multi-tap filter path', () => {
    it('RTTs the child, unrolls 32 taps of zoomBlurTapCoord, unpremultiplies', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'zb', def: ZB, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'zb', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/zoomBlurTapCoord/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        expect((finalWgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('ZoomBlur (b) weights', () => {
    it('normalized Gaussian weights sum to 1 over 32 taps', () => {
        expect(MOTION_BLUR_WEIGHTS.length).toBe(32)
        expect(MOTION_BLUR_WEIGHTS.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    })
})

describe('ZoomBlur (c) CPU golden — scaled tap coordinate', () => {
    it('scales the (aspect-corrected) offset from center by 1 + intensity·0.01·(tap/31), un-corrects', () => {
        // center transformed = (cx, 1 - cy_authored); the body flips y back via 1 - center.y.
        const cases = [
            {center: [0.5, 0.5] as const, intensity: 30, uv: [0.7, 0.6] as const, aspect: 1.5, tap: 0},
            {center: [0.5, 0.5] as const, intensity: 100, uv: [0.2, 0.3] as const, aspect: 1.0, tap: 31},
            {center: [0.3, 0.4] as const, intensity: 50, uv: [0.6, 0.9] as const, aspect: 2.0, tap: 16},
        ]
        for (const {center, intensity, uv, aspect, tap} of cases) {
            const cpX = center[0]
            const cpY = 1 - center[1]
            const radius = intensity * 0.01
            const scale = 1 + radius * (tap / 31)
            const acdX = (uv[0] - cpX) * aspect
            const acdY = uv[1] - cpY
            const ex = acdX / scale / aspect + cpX
            const ey = acdY / scale + cpY
            const out = zoomBlurTapCoord(
                d.vec2f(center[0], center[1]), intensity, d.vec2f(uv[0], uv[1]), aspect, tap,
            ) as unknown as {x: number; y: number}
            expect(out.x).toBeCloseTo(ex, 5)
            expect(out.y).toBeCloseTo(ey, 5)
        }
        // tap 0 → scale 1 → the identity sample (aspect multiply/divide cancel) → returns uv unchanged.
        const id = zoomBlurTapCoord(d.vec2f(0.5, 0.5), 30, d.vec2f(0.7, 0.6), 1.5, 0) as unknown as {x: number; y: number}
        expect(id.x).toBeCloseTo(0.7, 6)
        expect(id.y).toBeCloseTo(0.6, 6)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// (d) Detail dials — every dial defaults to 0 and the plain gather above is what compiles then
// (snapshot (a) is the parity gate). Any non-zero dial switches to the detail gather: the same
// 32 taps, per-channel tap weights, spectral tints, normalised at the end.
// ═══════════════════════════════════════════════════════════════════════════════════════
describe('ZoomBlur (d) detail gather', () => {
    function detailWgsl(props: Record<string, unknown>): string {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'zb', def: ZB, parentId: 'root', props, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'zb', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        return tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
    }

    it('dispersion alone switches to the detail gather: field + 32 weighted taps + resolve, plain zoom coords', () => {
        const wgsl = detailWgsl({dispersion: 0.6, falloff: 0.5})
        expect(wgsl).toMatch(/motionBlurDetailField/)
        expect(wgsl).toMatch(/motionBlurDetailTapWeight/)
        expect(wgsl).toMatch(/motionBlurDetailResolve/)
        expect(wgsl).toMatch(/zoomBlurTapCoord/)
        expect(wgsl).not.toMatch(/zoomBlurSpiralTapCoord/)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
        expect((wgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect((wgsl.match(/motionBlurDetailTapWeight\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT + 1) // 32 calls + the fn decl
        expect(wgsl).toMatchSnapshot('final-pass-dispersion')
    })

    it('spiral routes the taps through the spiral coordinate fn', () => {
        const wgsl = detailWgsl({spiral: 0.4})
        expect(wgsl).toMatch(/zoomBlurSpiralTapCoord/)
        expect(wgsl).toMatch(/motionBlurDetailField/)
    })

    it('every dial at 0 compiles the plain gather (no detail fns)', () => {
        const wgsl = detailWgsl({dispersion: 0, falloff: 0, focus: 0, bias: 0, highlights: 0, spiral: 0, jitter: 0})
        expect(wgsl).not.toMatch(/motionBlurDetail/)
        expect(wgsl).toMatch(/zoomBlurTapCoord/)
    })
})

describe('ZoomBlur (e) CPU goldens — spectrum table + spiral coordinate', () => {
    it('the spectrum runs red → violet over the 32 taps, and the reversed table mirrors it', () => {
        expect(MOTION_BLUR_SPECTRUM.length).toBe(MOTION_BLUR_TAP_COUNT)
        const [r0, g0, b0] = MOTION_BLUR_SPECTRUM[0].map((v) => v + 1)
        expect(r0).toBeCloseTo(1, 6)
        expect(g0).toBeCloseTo(0, 6)
        expect(b0).toBeCloseTo(0, 6)
        const [r31, g31, b31] = MOTION_BLUR_SPECTRUM[31].map((v) => v + 1)
        expect(b31).toBeCloseTo(1, 6)
        expect(g31).toBeCloseTo(0, 6)
        expect(r31).toBeGreaterThan(0.4) // violet carries some red
        // Green peaks mid-streak and is absent at both ends, so a white edge passes through yellow
        // and cyan on the way from red to violet.
        const greens = MOTION_BLUR_SPECTRUM.map(([, g]) => g + 1)
        expect(greens[15]).toBeCloseTo(1, 6)
        expect(greens[0]).toBeLessThan(0.01)
        expect(greens[31]).toBeLessThan(0.01)
        for (let i = 0; i < MOTION_BLUR_TAP_COUNT; i++) {
            expect(MOTION_BLUR_SPECTRUM_REVERSED[i]).toEqual(MOTION_BLUR_SPECTRUM[31 - i])
        }
    })

    it('spectrumRgb matches the spectral-lens hue wheel at the primaries', () => {
        expect(spectrumRgb(0)).toEqual([1, 0, 0])
        expect(spectrumRgb(1 / 3).map((v) => Math.round(v * 1e6) / 1e6)).toEqual([0, 1, 0])
        expect(spectrumRgb(2 / 3).map((v) => Math.round(v * 1e6) / 1e6)).toEqual([0, 0, 1])
    })

    it('the spiral tap coordinate equals the plain zoom coordinate at spiral 0', () => {
        const cases = [
            {center: [0.5, 0.5] as const, intensity: 30, uv: [0.7, 0.6] as const, aspect: 1.5, tap: 0},
            {center: [0.3, 0.4] as const, intensity: 50, uv: [0.6, 0.9] as const, aspect: 2.0, tap: 16},
            {center: [0.5, 0.5] as const, intensity: 100, uv: [0.2, 0.3] as const, aspect: 1.0, tap: 31},
        ]
        for (const {center, intensity, uv, aspect, tap} of cases) {
            const plain = zoomBlurTapCoord(d.vec2f(center[0], center[1]), intensity, d.vec2f(uv[0], uv[1]), aspect, tap) as unknown as {x: number; y: number}
            const spiral = zoomBlurSpiralTapCoord(d.vec2f(center[0], center[1]), intensity, d.vec2f(uv[0], uv[1]), aspect, tap, 0) as unknown as {x: number; y: number}
            expect(spiral.x).toBeCloseTo(plain.x, 6)
            expect(spiral.y).toBeCloseTo(plain.y, 6)
        }
    })

    it('a non-zero spiral rotates the offset around the centre by spiral·radius·(tap/31)·twist', () => {
        const center = d.vec2f(0.5, 0.5)
        const out = zoomBlurSpiralTapCoord(center, 100, d.vec2f(0.8, 0.5), 1.0, 31, 1) as unknown as {x: number; y: number}
        // radius 1, along 1, spiral 1 → angle = twist; scale 2. Offset (0.3, 0) rotated then /2.
        const angle = MOTION_BLUR_SPIRAL_TWIST
        expect(out.x).toBeCloseTo(0.5 + (0.3 * Math.cos(angle)) / 2, 5)
        expect(out.y).toBeCloseTo(0.5 + (0.3 * Math.sin(angle)) / 2, 5)
    })
})
