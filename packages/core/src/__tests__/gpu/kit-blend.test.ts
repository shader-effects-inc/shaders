import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import * as blend from '@coreroot/gpu/kit/blend'
import * as cm from '@coreroot/gpu/kit/colorMixing'

/**
 * Golden-value + resolve gates for kit/blend. TGSL fns execute on CPU (DualFn) at f32; the
 * reference below is an independent f64 transcription of the original blendModes/*.ts formulas.
 */

type Vec3 = [number, number, number]
const near = (actual: d.v4f, expected: number[], eps: number) => {
    const a = [...actual]
    for (let i = 0; i < 4; i++) expect(Math.abs(a[i] - expected[i])).toBeLessThan(eps)
}

// ── Reference (f64, from original formulas) ──
const lum = (c: Vec3) => c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
const step = (e: number, x: number) => (x >= e ? 1 : 0)
const mix = (a: number, b: number, t: number) => a + (b - a) * t
const len = (c: Vec3) => Math.hypot(c[0], c[1], c[2])
const norm = (c: Vec3): Vec3 => {
    const l = len(c)
    return [c[0] / l, c[1] / l, c[2] / l]
}

// Component-wise blend color formulas (base, overlay channel scalars)
const componentModes: Record<string, (b: number, o: number) => number> = {
    normal: (_b, o) => o,
    multiply: (b, o) => b * o,
    screen: (b, o) => 1 - (1 - b) * (1 - o),
    linearDodge: (b, o) => b + o,
    overlay: (b, o) => mix(2 * b * o, 1 - 2 * (1 - b) * (1 - o), step(0.5, b)),
    difference: (b, o) => Math.abs(b - o),
    colorDodge: (b, o) => mix(b, Math.min(b / (1 - o), 1), step(0.001, 1 - o)),
    exclusion: (b, o) => b + o - 2 * b * o,
    darken: (b, o) => Math.min(b, o),
    lighten: (b, o) => Math.max(b, o),
    colorBurn: (b, o) => mix(0, Math.max(1 - (1 - b) / (o + 0.0001), 0), step(0.0001, o)),
    linearBurn: (b, o) => Math.max(b + o - 1, 0),
    softLight: (b, o) =>
        mix(2 * b * o + b * b * (1 - 2 * o), 2 * b * (1 - o) + Math.sqrt(b) * (2 * o - 1), step(0.5, o)),
    hardLight: (b, o) => mix(2 * b * o, 1 - 2 * (1 - b) * (1 - o), step(0.5, o)),
}
// Whole-vector (luminance-based) blend colors
const chromaModes: Record<string, (B: Vec3, O: Vec3) => Vec3> = {
    color: (B, O) => {
        const dLum = lum(B) - lum(O)
        return [O[0] + dLum, O[1] + dLum, O[2] + dLum]
    },
    luminosity: (B, O) => {
        const r = lum(O) / (lum(B) + 0.0001)
        return [B[0] * r, B[1] * r, B[2] * r]
    },
    hue: (B, O) => {
        const bl = lum(B)
        const bc: Vec3 = [B[0] - bl, B[1] - bl, B[2] - bl]
        const oc: Vec3 = [O[0] - lum(O), O[1] - lum(O), O[2] - lum(O)]
        const bcl = len(bc)
        const od = norm(oc)
        return [od[0] * bcl + bl, od[1] * bcl + bl, od[2] * bcl + bl]
    },
    saturation: (B, O) => {
        const bl = lum(B)
        const bc: Vec3 = [B[0] - bl, B[1] - bl, B[2] - bl]
        const oc: Vec3 = [O[0] - lum(O), O[1] - lum(O), O[2] - lum(O)]
        const bd = norm(bc)
        const ocl = len(oc)
        return [bd[0] * ocl + bl, bd[1] * ocl + bl, bd[2] * ocl + bl]
    },
}

const overComp = (base: number[], blended: Vec3, oa: number): number[] => {
    const bw = base[3] * (1 - oa)
    const fa = oa + bw
    return [blended[0] * oa + base[0] * bw, blended[1] * oa + base[1] * bw, blended[2] * oa + base[2] * bw, fa]
}
const refBlend = (mode: string, base: number[], overlay: number[], op: number): number[] => {
    const B3: Vec3 = [base[0], base[1], base[2]]
    const O3: Vec3 = [overlay[0], overlay[1], overlay[2]]
    const oa = overlay[3] * op
    const blended: Vec3 = componentModes[mode]
        ? [componentModes[mode](B3[0], O3[0]), componentModes[mode](B3[1], O3[1]), componentModes[mode](B3[2], O3[2])]
        : chromaModes[mode](B3, O3)
    return overComp(base, blended, oa)
}

const ALL = Object.keys(componentModes).concat(Object.keys(chromaModes))
const SAFE_AT_EXTREMES = ['normal', 'multiply', 'screen', 'linearDodge', 'overlay', 'difference', 'exclusion', 'darken', 'lighten', 'linearBurn', 'hardLight']

describe('blend — 18 standard modes vs reference', () => {
    const SAMPLE_A = {base: [0.2, 0.4, 0.6, 1.0], overlay: [0.7, 0.2, 0.3, 1.0], op: 1.0}
    const SAMPLE_B = {base: [0.15, 0.55, 0.85, 0.8], overlay: [0.6, 0.35, 0.1, 0.6], op: 0.7}

    for (const mode of ALL) {
        it(`${mode}`, () => {
            for (const s of [SAMPLE_A, SAMPLE_B]) {
                const fn = (blend.blendModes as Record<string, typeof blend.normal>)[mode]
                const got = fn(d.vec4f(...(s.base as [number, number, number, number])), d.vec4f(...(s.overlay as [number, number, number, number])), s.op)
                near(got, refBlend(mode, s.base, s.overlay, s.op), 2e-4)
            }
        })
    }

    it('holds at extreme channel values (0/1) for well-defined modes', () => {
        const base = [0, 0, 0, 1]
        const overlay = [1, 1, 1, 1]
        for (const mode of SAFE_AT_EXTREMES) {
            const fn = (blend.blendModes as Record<string, typeof blend.normal>)[mode]
            const got = fn(d.vec4f(0, 0, 0, 1), d.vec4f(1, 1, 1, 1), 1.0)
            near(got, refBlend(mode, base, overlay, 1.0), 1e-5)
        }
    })
})

describe('blend — hand-computed cross-checks (guards shared transcription errors)', () => {
    // base=(0.2,0.4,0.6,1) overlay=(0.5,0.5,0.5,1) opacity=1 → result = blended color, alpha 1
    const B = () => d.vec4f(0.2, 0.4, 0.6, 1)
    const O = () => d.vec4f(0.5, 0.5, 0.5, 1)
    it('multiply = base·overlay', () => near(blend.multiply(B(), O(), 1), [0.1, 0.2, 0.3, 1], 1e-6))
    it('screen = 1-(1-b)(1-o)', () => near(blend.screen(B(), O(), 1), [0.6, 0.7, 0.8, 1], 1e-6))
    it('difference = |b-o|', () => near(blend.difference(B(), O(), 1), [0.3, 0.1, 0.1, 1], 1e-6))
    it('exclusion = b+o-2bo', () => near(blend.exclusion(B(), O(), 1), [0.5, 0.5, 0.5, 1], 1e-6))
    it('darken = min', () => near(blend.darken(B(), O(), 1), [0.2, 0.4, 0.5, 1], 1e-6))
    it('linearBurn = max(b+o-1,0)', () => near(blend.linearBurn(B(), O(), 1), [0, 0, 0.1, 1], 1e-6))
    it('opacity=0 passes base through', () => near(blend.multiply(B(), O(), 0), [0.2, 0.4, 0.6, 1], 1e-6))
})

describe('blend — normal-oklch / normal-oklab (color-space over-composite)', () => {
    const oklchRef = (B: number[], O: number[], op: number, space: 'oklch' | 'oklab'): number[] => {
        const oa = O[3] * op
        const bw = B[3] * (1 - oa)
        const fwd = (v: d.v3f) => (space === 'oklch' ? cm.oklabToOklch(cm.rgbToOklab(cm.p3ToSRGB(v))) : cm.rgbToOklab(cm.p3ToSRGB(v)))
        const back = (v: d.v3f) => (space === 'oklch' ? cm.sRGBToP3(cm.oklabToRgb(cm.oklchToOklab(v))) : cm.sRGBToP3(cm.oklabToRgb(v)))
        const bC = fwd(d.vec3f(B[0], B[1], B[2]))
        const oC = fwd(d.vec3f(O[0], O[1], O[2]))
        const blended = d.vec3f(oC.x * oa + bC.x * bw, oC.y * oa + bC.y * bw, oC.z * oa + bC.z * bw)
        const rgb = back(blended)
        return [rgb.x, rgb.y, rgb.z, oa + bw]
    }
    const B = [0.2, 0.4, 0.6, 1]
    const O = [0.7, 0.2, 0.3, 1]

    it('full replace (opacity 1, both opaque) ≈ overlay', () => {
        near(blend.normalOklch(d.vec4f(...(B as [number, number, number, number])), d.vec4f(...(O as [number, number, number, number])), 1), [0.7, 0.2, 0.3, 1], 2e-3)
        near(blend.normalOklab(d.vec4f(...(B as [number, number, number, number])), d.vec4f(...(O as [number, number, number, number])), 1), [0.7, 0.2, 0.3, 1], 2e-3)
    })
    it('self-blend ≈ base', () => {
        const c = d.vec4f(0.35, 0.55, 0.25, 1)
        near(blend.normalOklch(c, c, 1), [0.35, 0.55, 0.25, 1], 2e-3)
        near(blend.normalOklab(c, c, 1), [0.35, 0.55, 0.25, 1], 2e-3)
    })
    it('mid opacity matches color-space composite reference', () => {
        near(blend.normalOklch(d.vec4f(...(B as [number, number, number, number])), d.vec4f(...(O as [number, number, number, number])), 0.5), oklchRef(B, O, 0.5, 'oklch'), 1e-5)
        near(blend.normalOklab(d.vec4f(...(B as [number, number, number, number])), d.vec4f(...(O as [number, number, number, number])), 0.5), oklchRef(B, O, 0.5, 'oklab'), 1e-5)
    })
})

describe('blend — applyBlendMode dispatcher', () => {
    it('dispatches to the named mode', () => {
        const B = d.vec4f(0.2, 0.4, 0.6, 1)
        const O = d.vec4f(0.5, 0.5, 0.5, 1)
        near(blend.applyBlendMode(B, O, 'multiply', 1), [...blend.multiply(B, O, 1)] as number[], 1e-9)
    })
    it('falls back to normal for an unknown mode', () => {
        const B = d.vec4f(0.2, 0.4, 0.6, 1)
        const O = d.vec4f(0.5, 0.5, 0.5, 0.5)
        near(blend.applyBlendMode(B, O, 'bogus' as blend.BlendMode, 1), [...blend.normal(B, O, 1)] as number[], 1e-9)
    })
})

describe('blend — resolve gate', () => {
    it('representative modes resolve to WGSL (snapshot)', () => {
        const wgsl = tgpu.resolve(
            [blend.multiply, blend.overlay, blend.hue, blend.softLight, blend.normalOklch],
            {names: 'strict'},
        )
        expect(wgsl).toMatch(/fn/)
        expect(wgsl).toMatchSnapshot()
    })
})

describe('blend — normalOklch hue takes the shortest arc', () => {
    it('cyan over green at half opacity stays green-cyan (not a pink midpoint across the ±π seam)', () => {
        const out = blend.normalOklch(d.vec4f(0, 1, 0, 1), d.vec4f(0, 1, 1, 1), 0.5)
        const lab = cm.rgbToOklab(cm.p3ToSRGB(d.vec3f(out.x, out.y, out.z)))
        const h = ((Math.atan2(lab.z, lab.y) * 180) / Math.PI + 360) % 360
        expect(h).toBeGreaterThan(140)
        expect(h).toBeLessThan(200)
        expect(out.y).toBeGreaterThan(out.x)
    })
    it('white over blue keeps blue\'s hue (achromatic side borrows it)', () => {
        const hue = (c: {x: number; y: number; z: number}) => {
            const lab = cm.rgbToOklab(cm.p3ToSRGB(d.vec3f(c.x, c.y, c.z)))
            return ((Math.atan2(lab.z, lab.y) * 180) / Math.PI + 360) % 360
        }
        const out = blend.normalOklch(d.vec4f(0, 0, 1, 1), d.vec4f(1, 1, 1, 1), 0.5)
        expect(Math.abs(hue(out) - hue(d.vec4f(0, 0, 1, 1)))).toBeLessThan(1)
    })
})
