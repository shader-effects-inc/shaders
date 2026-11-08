import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import * as cm from '@coreroot/gpu/kit/colorMixing'

/**
 * Golden-value + round-trip + resolve gates for kit/colorMixing (the highest-precision-
 * sensitivity kit module). TGSL fns execute on CPU as plain JS (DualFn) at f32 precision, so
 * assertions use tolerances. References are derived from the original colorMixing.ts formulas.
 */

const near = (actual: d.v3f | d.v4f, expected: number[], eps: number) => {
    const a = [...actual]
    expect(a.length).toBe(expected.length)
    for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(a[i] - expected[i])).toBeLessThan(eps)
    }
}

// Sample colors in P3-linear (the space transformColor produces)
const GRAY = () => d.vec3f(0.5, 0.5, 0.5)
const WARM = () => d.vec3f(0.7, 0.3, 0.15)
const COOL = () => d.vec3f(0.12, 0.4, 0.75)
const WHITE = () => d.vec3f(1, 1, 1)
const BLACK = () => d.vec3f(0, 0, 0)

describe('colorMixing — P3 ↔ sRGB round-trip', () => {
    it('sRGBToP3(p3ToSRGB(c)) ≈ c', () => {
        for (const c of [GRAY(), WARM(), COOL(), WHITE()]) {
            near(cm.sRGBToP3(cm.p3ToSRGB(c)), [...c], 1e-4)
        }
    })
    it('p3ToSRGB(white) stays white (rows sum ~1)', () => {
        near(cm.p3ToSRGB(WHITE()), [1, 1, 1], 1e-3)
    })
})

describe('colorMixing — OKLab / OKLCh', () => {
    it('rgbToOklab(white) ≈ (1, 0, 0)', () => {
        near(cm.rgbToOklab(WHITE()), [1, 0, 0], 1e-4)
    })
    it('rgbToOklab(black) ≈ (0, 0, 0)', () => {
        near(cm.rgbToOklab(BLACK()), [0, 0, 0], 1e-6)
    })
    it('oklabToRgb(rgbToOklab(c)) ≈ c', () => {
        for (const c of [GRAY(), WARM(), COOL()]) {
            near(cm.oklabToRgb(cm.rgbToOklab(c)), [...c], 1e-4)
        }
    })
    it('oklchToOklab(oklabToOklch(lab)) ≈ lab', () => {
        const lab = cm.rgbToOklab(WARM())
        near(cm.oklchToOklab(cm.oklabToOklch(lab)), [...lab], 1e-4)
    })
})

describe('colorMixing — HSL / HSV / Lab / LCh round-trips', () => {
    it('hslToRgb(rgbToHsl(c)) ≈ c', () => {
        for (const c of [GRAY(), WARM(), COOL()]) {
            near(cm.hslToRgb(cm.rgbToHsl(c)), [...c], 1e-4)
        }
    })
    it('hsvToRgb(rgbToHsv(c)) ≈ c', () => {
        for (const c of [GRAY(), WARM(), COOL()]) {
            near(cm.hsvToRgb(cm.rgbToHsv(c)), [...c], 1e-4)
        }
    })
    it('labToRgb(rgbToLab(c)) ≈ c', () => {
        for (const c of [GRAY(), WARM(), COOL()]) {
            near(cm.labToRgb(cm.rgbToLab(c)), [...c], 1e-4)
        }
    })
    it('lchToLab(labToLch(lab)) ≈ lab', () => {
        const lab = cm.rgbToLab(COOL())
        near(cm.lchToLab(cm.labToLch(lab)), [...lab], 1e-3)
    })
})

describe('colorMixing — mixColors alpha-weighted blending', () => {
    const A = () => d.vec4f(0.2, 0.4, 0.6, 1.0)
    const B = () => d.vec4f(0.8, 0.6, 0.4, 0.5)

    it('linear mix at t=0.5 matches hand-computed weighted average', () => {
        // weightA = 1·0.5 = 0.5, weightB = 0.5·0.5 = 0.25, total = 0.75
        // rgb = (A·0.5 + B·0.25)/0.75, alpha = 0.75
        near(cm.mixColors(A(), B(), 0.5, 0), [0.4, 0.4666667, 0.5333333, 0.75], 1e-5)
    })
    it('linear mix at t=0 returns colorA, at t=1 returns colorB', () => {
        near(cm.mixColors(A(), B(), 0, 0), [0.2, 0.4, 0.6, 1.0], 1e-5)
        near(cm.mixColors(A(), B(), 1, 0), [0.8, 0.6, 0.4, 0.5], 1e-5)
    })
    it('mixing a color with itself is identity in every space (t=0.35)', () => {
        const c = d.vec4f(0.55, 0.3, 0.7, 1.0)
        for (const mode of [0, 1, 2, 3, 4, 5]) {
            near(cm.mixColors(c, c, 0.35, mode), [0.55, 0.3, 0.7, 1.0], 2e-3)
        }
    })
    it('both endpoints transparent → safeWeight guard keeps output finite (alpha 0)', () => {
        const ta = d.vec4f(0.5, 0.5, 0.5, 0)
        const tb = d.vec4f(0.2, 0.2, 0.2, 0)
        const out = [...cm.mixColors(ta, tb, 0.5, 0)]
        expect(out.every((v) => Number.isFinite(v))).toBe(true)
        expect(out[3]).toBeCloseTo(0, 6)
    })
})

describe('colorMixing — CPU mirror agrees with GPU forward conversion', () => {
    it('convertP3ToMixSpaceCPU ≈ convertP3ToMixSpace for OKLCh/OKLAB/HSL/HSV/LCH', () => {
        const c = COOL()
        for (const mode of [1, 2, 3, 4, 5]) {
            const cpu = cm.convertP3ToMixSpaceCPU(c.x, c.y, c.z, mode)
            const gpu = cm.convertP3ToMixSpace(c, mode)
            near(gpu, cpu, 1e-3)
        }
    })
})

describe('colorMixing — resolve gate', () => {
    it('all conversions + a mix variant resolve to WGSL (snapshot)', () => {
        const wgsl = tgpu.resolve(
            [
                cm.p3ToSRGB, cm.sRGBToP3,
                cm.rgbToOklab, cm.oklabToRgb, cm.oklabToOklch, cm.oklchToOklab,
                cm.selectRGBBySector, cm.rgbToHsl, cm.hslToRgb, cm.rgbToHsv, cm.hsvToRgb,
                cm.rgbToLab, cm.labToRgb, cm.labToLch, cm.lchToLab,
                cm.mixColorsLinear, cm.mixColorsOklch,
            ],
            {names: 'strict'},
        )
        expect(typeof wgsl).toBe('string')
        expect(wgsl).toMatch(/fn/)
        expect(wgsl).toMatchSnapshot()
    })
})

describe('colorMixing — hue interpolation (shortest arc + achromatic endpoints)', () => {
    const DEG = 180 / Math.PI
    const wrapDeg = (h: number) => ((h % 360) + 360) % 360
    /** OKLCh hue (degrees, 0..360) + chroma of a P3-linear color. */
    const oklchOf = (c: d.v3f | d.v4f) => {
        const lab = cm.rgbToOklab(cm.p3ToSRGB(d.vec3f(c.x, c.y, c.z)))
        return {h: wrapDeg(Math.atan2(lab.z, lab.y) * DEG), C: Math.hypot(lab.y, lab.z)}
    }
    /** HSL/HSV-style hue (degrees) of an sRGB-ish triple. */
    const rgbHueDeg = (c: d.v3f | d.v4f) => wrapDeg(cm.rgbToHsl(cm.p3ToSRGB(d.vec3f(c.x, c.y, c.z))).x * DEG)

    it('mixHueShortest wraps the difference into [-π, π)', () => {
        const a = (350 / DEG), b = (10 / DEG)
        // halfway between 350° and 10° is 0°, not 180°
        expect(wrapDeg(cm.mixHueShortest(a, 1, b, 1, 0.5, 1, 0.001) * DEG)).toBeCloseTo(0, 3)
        // a quarter of the way from 10° to 350° goes DOWN through 5°
        expect(wrapDeg(cm.mixHueShortest(b, 1, a, 1, 0.25, 1, 0.001) * DEG)).toBeCloseTo(5, 3)
        // weight fraction = wB / sw (alpha-weighted), endpoints exact
        expect(cm.mixHueShortest(1, 1, 2, 1, 0, 1, 0.001)).toBeCloseTo(1, 6)
        expect(wrapDeg(cm.mixHueShortest(1, 1, 2, 1, 1, 1, 0.001) * DEG)).toBeCloseTo(2 * DEG, 3)
    })
    it('mixHueShortest: an achromatic endpoint borrows the other hue', () => {
        expect(cm.mixHueShortest(3, 0, 1, 1, 0.5, 1, 0.001)).toBeCloseTo(1, 6)
        expect(cm.mixHueShortest(1, 1, -2.5, 0, 0.5, 1, 0.001)).toBeCloseTo(1, 6)
    })

    // Green (OKLCh h≈142°) and cyan (h≈195°) straddle atan2's ±π seam: the naive average lands
    // near -10° (pink). The shortest arc keeps the midpoint between them.
    const GREEN = () => d.vec4f(0, 1, 0, 1)
    const CYAN = () => d.vec4f(0, 1, 1, 1)
    it('OKLCh mix across the ±π seam stays between the endpoint hues (scalar + preconverted + stops)', () => {
        const hG = oklchOf(GREEN()).h
        const hC = oklchOf(CYAN()).h
        const check = (out: d.v4f) => {
            const h = oklchOf(out).h
            expect(h).toBeGreaterThan(Math.min(hG, hC))
            expect(h).toBeLessThan(Math.max(hG, hC))
        }
        check(cm.mixColors(GREEN(), CYAN(), 0.5, 1))
        const ca = cm.convertP3ToMixSpace(d.vec3f(0, 1, 0), 1)
        const cb = cm.convertP3ToMixSpace(d.vec3f(0, 1, 1), 1)
        check(cm.mixPreconvertedColors(ca, cb, 1, 1, 0.5, 1))
        const step = cm.mixPreconvertedInSpaceVariants[1](ca, 1, cb, 1, 0.5)
        const back = cm.convertMixSpaceToP3(step.conv, 1)
        check(d.vec4f(back, step.alpha))
    })
    it('LCh mix across the ±π seam stays between the endpoint hues', () => {
        const out = cm.mixColors(GREEN(), CYAN(), 0.5, 5)
        expect(out.y).toBeGreaterThan(out.x) // green-cyan, not a pink/red midpoint
        const h = oklchOf(out).h
        expect(h).toBeGreaterThan(120)
        expect(h).toBeLessThan(220)
    })
    it('HSL/HSV red ↔ magenta takes the short way (rose), not through green', () => {
        const RED = () => d.vec4f(1, 0, 0, 1)
        const MAGENTA = () => d.vec4f(1, 0, 1, 1)
        for (const mode of [3, 4]) {
            const out = cm.mixColors(RED(), MAGENTA(), 0.5, mode)
            const h = rgbHueDeg(out)
            expect(Math.min(h, 360 - h)).toBeLessThan(40) // ~330°, i.e. within the red→magenta arc
            expect(h).toBeGreaterThan(300)
            for (const v of [out.x, out.y, out.z]) expect(v).toBeGreaterThan(-1e-3) // no negative channel
        }
    })
    it('HSL/HSV round-trip holds for red→magenta hues (negative raw hue)', () => {
        const rose = d.vec3f(1, 0, 0.5)
        near(cm.hslToRgb(cm.rgbToHsl(rose)), [1, 0, 0.5], 1e-4)
        near(cm.hsvToRgb(cm.rgbToHsv(rose)), [1, 0, 0.5], 1e-4)
    })
    it('white ↔ blue keeps blue\'s hue in every cylindrical space (no junk hue from white)', () => {
        const WHITE4 = () => d.vec4f(1, 1, 1, 1)
        const BLUE4 = () => d.vec4f(0, 0, 1, 1)
        const labHue = (c: d.v3f | d.v4f) => {
            const lab = cm.rgbToLab(cm.p3ToSRGB(d.vec3f(c.x, c.y, c.z)))
            return wrapDeg(Math.atan2(lab.z, lab.y) * DEG)
        }
        const hBlueRgb = rgbHueDeg(BLUE4())
        // Hue is held in the MIX space's own hue (CIE LCh's blue hue is not OKLCh's).
        expect(Math.abs(oklchOf(cm.mixColors(WHITE4(), BLUE4(), 0.5, 1)).h - oklchOf(BLUE4()).h)).toBeLessThan(1)
        expect(Math.abs(labHue(cm.mixColors(WHITE4(), BLUE4(), 0.5, 5)) - labHue(BLUE4()))).toBeLessThan(1)
        for (const mode of [3, 4]) {
            const out = cm.mixColors(WHITE4(), BLUE4(), 0.5, mode)
            expect(Math.abs(rgbHueDeg(out) - hBlueRgb)).toBeLessThan(1)
        }
    })
    it('OKLAB and linear modes are unchanged (no hue channel)', () => {
        const a = d.vec4f(0.2, 0.4, 0.6, 1), b = d.vec4f(0.8, 0.6, 0.4, 1)
        near(cm.mixColors(a, b, 0.5, 0), [0.5, 0.5, 0.5, 1], 1e-5)
        expect(cm.mixPreconvertedInSpaceVariants[2]).toBe(cm.mixPreconvertedInSpace)
    })
})
