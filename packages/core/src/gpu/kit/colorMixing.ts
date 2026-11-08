/**
 * Color-space conversions + alpha-weighted mixing.
 *
 * Six mix spaces: 0 linear RGB · 1 OKLCh · 2 OKLAB · 3 HSL · 4 HSV · 5 LCH. See CLAUDE.md
 * "Color Space Blending".
 *
 * Conversions are `tgpu.fn` shells (resolvable + snapshot-testable + reusable as externals,
 * e.g. by kit/blend.ts's OKLCh/OKLab blends). `mixColors` is a compile-time dispatcher over
 * six pre-transpiled `mixColors*` variants — the `colorSpaceMode` branch stays on the CPU
 * builder.
 *
 * NOTE: the kit's `select`/`oneMinus` helpers (kit/index.ts) are plain JS and cannot
 * be called INSIDE a `'use gpu'` body — use `std.*` directly there. `std.select` takes WGSL
 * order `(ifFalse, ifTrue, cond)`.
 */
import {d, std, tgpu} from './index'

// ─── Constants ───────────────────────────────────────────────────────────────
const EPSILON = 0.0001 // Prevent division by zero
const PI_OVER_3 = 1.0471975512 // π/3, used for HSL/HSV hue conversions
const PI_F = 3.14159265359 // π (pre-folded — no Math.* inside 'use gpu' bodies)
const TWO_PI = 6.28318530718 // 2π
const INV_TWO_PI = 0.15915494309 // 1/(2π)
const INV_SIX = 0.16666666667 // 1/6 — hue sector wrap
// Below these chroma/saturation values a colour is treated as achromatic: its hue is
// meaningless (atan2 noise), so a hue mix takes the OTHER endpoint's hue instead.
const OKLCH_ACHROMATIC = 0.001 // OKLCh chroma (chromatic range ~0..0.4)
const LCH_ACHROMATIC = 0.25 // CIE LCh chroma (chromatic range ~0..130)
const HS_ACHROMATIC = 0.001 // HSL/HSV RGB chroma (max−min, 0..1)
const LAB_DELTA = 6.0 / 29.0 // CIE Lab delta threshold
const LAB_KAPPA = 500.0 // CIE Lab a* scaling
const LAB_LAMBDA = 200.0 // CIE Lab b* scaling
const LAB_DELTA3 = LAB_DELTA * LAB_DELTA * LAB_DELTA // (6/29)³
const LAB_OFFSET = 4.0 / 29.0
const LAB_FWD_SLOPE = 1.0 / (3.0 * LAB_DELTA * LAB_DELTA) // 1/(3δ²) — forward f(t)
const LAB_INV_SLOPE = 3.0 * LAB_DELTA * LAB_DELTA // 3δ² — inverse f(t)

// ─── P3 ↔ sRGB (linear) ──────────────────────────────────────────────────────

/** P3-linear → sRGB-linear (D65). */
export const p3ToSRGB = tgpu.fn([d.vec3f], d.vec3f)((p3) => {
    'use gpu'
    const r = p3.x * 1.2249401 - p3.y * 0.2249404 - p3.z * 0.0
    const g = p3.x * -0.0420569 + p3.y * 1.0420571 + p3.z * 0.0
    const b = p3.x * -0.0196376 - p3.y * 0.0786361 + p3.z * 1.0982735
    return d.vec3f(r, g, b)
})

/** sRGB-linear → P3-linear (inverse of p3ToSRGB). */
export const sRGBToP3 = tgpu.fn([d.vec3f], d.vec3f)((srgb) => {
    'use gpu'
    const r = srgb.x * 0.8224621 + srgb.y * 0.1775380 + srgb.z * 0.0
    const g = srgb.x * 0.0331941 + srgb.y * 0.9668058 + srgb.z * 0.0
    const b = srgb.x * 0.0170826 + srgb.y * 0.0723974 + srgb.z * 0.9105199
    return d.vec3f(r, g, b)
})

// ─── OKLab ↔ RGB / OKLCh ──────────────────────────────────────────────────────

/** sRGB-linear → OKLab (Björn Ottosson). Sign-preserving cube root for negative lobes. */
export const rgbToOklab = tgpu.fn([d.vec3f], d.vec3f)((rgb) => {
    'use gpu'
    const r = rgb.x
    const g = rgb.y
    const b = rgb.z

    const l = r * 0.4122214708 + g * 0.5363325363 + b * 0.0514459929
    const m = r * 0.2119034982 + (g * 0.6806995451 + b * 0.1073969566)
    const s = r * 0.0883024619 + g * 0.2817188376 + b * 0.6299787005

    // sign-preserving cube root (pow returns NaN for negative bases)
    const l_ = std.sign(l) * std.pow(std.abs(l), 1.0 / 3.0)
    const m_ = std.sign(m) * std.pow(std.abs(m), 1.0 / 3.0)
    const s_ = std.sign(s) * std.pow(std.abs(s), 1.0 / 3.0)

    return d.vec3f(
        l_ * 0.2104542553 + m_ * 0.7936177850 - s_ * 0.0040720468,
        l_ * 1.9779984951 - m_ * 2.4285922050 + s_ * 0.4505937099,
        l_ * 0.0259040371 + m_ * 0.7827717662 - s_ * 0.8086757660,
    )
})

/** OKLab → sRGB-linear. */
export const oklabToRgb = tgpu.fn([d.vec3f], d.vec3f)((lab) => {
    'use gpu'
    const L = lab.x
    const a = lab.y
    const b = lab.z

    const l_ = L + a * 0.3963377774 + b * 0.2158037573
    const m_ = L - a * 0.1055613458 - b * 0.0638541728
    const s_ = L - a * 0.0894841775 - b * 1.2914855480

    const l = std.pow(l_, 3.0)
    const m = std.pow(m_, 3.0)
    const s = std.pow(s_, 3.0)

    return d.vec3f(
        l * 4.0767416621 - m * 3.3077115913 + s * 0.2309699292,
        l * -1.2684380046 + m * 2.6097574011 - s * 0.3413193965,
        l * -0.0041960863 - m * 0.7034186147 + s * 1.7076147010,
    )
})

/** OKLab → OKLCh (cylindrical). */
export const oklabToOklch = tgpu.fn([d.vec3f], d.vec3f)((lab) => {
    'use gpu'
    const L = lab.x
    const a = lab.y
    const b = lab.z
    const C = std.sqrt(a * a + b * b)
    const h = std.atan2(b, a) // two-parameter form for correct quadrant
    return d.vec3f(L, C, h)
})

/** OKLCh → OKLab. */
export const oklchToOklab = tgpu.fn([d.vec3f], d.vec3f)((lch) => {
    'use gpu'
    const L = lch.x
    const C = lch.y
    const h = lch.z
    return d.vec3f(L, C * std.cos(h), C * std.sin(h))
})

// ─── HSL / HSV sector helper ──────────────────────────────────────────────────

/**
 * Select RGB components for a hue sector (0-5), shared by HSL/HSV → RGB.
 *   0:[0,1)→(c,x,0) 1:[1,2)→(x,c,0) 2:[2,3)→(0,c,x) 3:[3,4)→(0,x,c) 4:[4,5)→(x,0,c) 5:[5,6)→(c,0,x)
 */
export const selectRGBBySector = tgpu.fn([d.f32, d.f32, d.f32], d.vec3f)((c, x, sector) => {
    'use gpu'
    let r = d.f32(0)
    let g = d.f32(0)
    let b = d.f32(0)
    if (sector < 1.0) {
        r = c
        g = x
    } else if (sector < 2.0) {
        r = x
        g = c
    } else if (sector < 3.0) {
        g = c
        b = x
    } else if (sector < 4.0) {
        g = x
        b = c
    } else if (sector < 5.0) {
        r = x
        b = c
    } else {
        r = c
        b = x
    }
    return d.vec3f(r, g, b)
})

// ─── HSL ↔ RGB ────────────────────────────────────────────────────────────────

/** sRGB-linear → HSL (hue in radians, 0..2π). */
export const rgbToHsl = tgpu.fn([d.vec3f], d.vec3f)((rgb) => {
    'use gpu'
    const r = rgb.x
    const g = rgb.y
    const b = rgb.z

    const maxVal = std.max(std.max(r, g), b)
    const minVal = std.min(std.min(r, g), b)
    const delta = maxVal - minVal

    const l = (maxVal + minVal) * 0.5

    // saturation: delta / (1 - |2L - 1|)
    const satDenom = std.max(1.0 - std.abs(l * 2.0 - 1.0), EPSILON)
    const s = delta / satDenom

    const isRMax = maxVal === r
    const isGMax = maxVal === g
    const hueR = (g - b) / std.max(delta, EPSILON)
    const hueG = 2.0 + (b - r) / std.max(delta, EPSILON)
    const hueB = 4.0 + (r - g) / std.max(delta, EPSILON)
    const hue = std.select(std.select(hueB, hueG, isGMax), hueR, isRMax)

    const h = hue * PI_OVER_3
    return d.vec3f(h, s, l)
})

/** HSL → sRGB-linear. */
export const hslToRgb = tgpu.fn([d.vec3f], d.vec3f)((hsl) => {
    'use gpu'
    const h = hsl.x
    const s = hsl.y
    const l = hsl.z

    const c = s * (1.0 - std.abs(l * 2.0 - 1.0))
    // Wrap into [0, 6): rgbToHsl/rgbToHsv emit hue in [-π/3, 5π/3) and hue mixes can leave
    // [0, 2π) — std.mod is truncated, so a negative hPrime would give a negative x.
    const hRaw = h / PI_OVER_3
    const hPrime = hRaw - std.floor(hRaw * INV_SIX) * 6.0
    const x = c * (1.0 - std.abs(std.mod(hPrime, 2.0) - 1.0))
    const m = l - c * 0.5
    const sector = std.floor(hPrime)
    return selectRGBBySector(c, x, sector).add(m)
})

// ─── HSV ↔ RGB ────────────────────────────────────────────────────────────────

/** sRGB-linear → HSV (hue in radians). */
export const rgbToHsv = tgpu.fn([d.vec3f], d.vec3f)((rgb) => {
    'use gpu'
    const r = rgb.x
    const g = rgb.y
    const b = rgb.z

    const maxVal = std.max(std.max(r, g), b)
    const minVal = std.min(std.min(r, g), b)
    const delta = maxVal - minVal

    const v = maxVal
    const s = delta / std.max(maxVal, EPSILON)

    const isRMax = maxVal === r
    const isGMax = maxVal === g
    const hueR = (g - b) / std.max(delta, EPSILON)
    const hueG = 2.0 + (b - r) / std.max(delta, EPSILON)
    const hueB = 4.0 + (r - g) / std.max(delta, EPSILON)
    const hue = std.select(std.select(hueB, hueG, isGMax), hueR, isRMax)

    const h = hue * PI_OVER_3
    return d.vec3f(h, s, v)
})

/** HSV → sRGB-linear. */
export const hsvToRgb = tgpu.fn([d.vec3f], d.vec3f)((hsv) => {
    'use gpu'
    const h = hsv.x
    const s = hsv.y
    const v = hsv.z

    const c = v * s
    // Wrap into [0, 6): rgbToHsl/rgbToHsv emit hue in [-π/3, 5π/3) and hue mixes can leave
    // [0, 2π) — std.mod is truncated, so a negative hPrime would give a negative x.
    const hRaw = h / PI_OVER_3
    const hPrime = hRaw - std.floor(hRaw * INV_SIX) * 6.0
    const x = c * (1.0 - std.abs(std.mod(hPrime, 2.0) - 1.0))
    const m = v - c
    const sector = std.floor(hPrime)
    return selectRGBBySector(c, x, sector).add(m)
})

// ─── CIE Lab ↔ RGB / LCh ──────────────────────────────────────────────────────

/** sRGB-linear → CIE Lab (D65). */
export const rgbToLab = tgpu.fn([d.vec3f], d.vec3f)((rgb) => {
    'use gpu'
    const r = rgb.x
    const g = rgb.y
    const b = rgb.z

    // linear RGB → XYZ (D65)
    const x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    const y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    const z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    // normalize by D65 white point
    const xn = x / 0.95047
    const yn = y / 1.00000
    const zn = z / 1.08883

    // f(t) = t > δ³ ? t^(1/3) : t/(3δ²) + 4/29
    const fx = std.select(xn * LAB_FWD_SLOPE + LAB_OFFSET, std.pow(xn, 1.0 / 3.0), xn > LAB_DELTA3)
    const fy = std.select(yn * LAB_FWD_SLOPE + LAB_OFFSET, std.pow(yn, 1.0 / 3.0), yn > LAB_DELTA3)
    const fz = std.select(zn * LAB_FWD_SLOPE + LAB_OFFSET, std.pow(zn, 1.0 / 3.0), zn > LAB_DELTA3)

    const L = fy * 116.0 - 16.0
    const a = (fx - fy) * LAB_KAPPA
    const bLab = (fy - fz) * LAB_LAMBDA
    return d.vec3f(L, a, bLab)
})

/** CIE Lab → sRGB-linear. */
export const labToRgb = tgpu.fn([d.vec3f], d.vec3f)((lab) => {
    'use gpu'
    const L = lab.x
    const a = lab.y
    const b = lab.z

    const fy = (L + 16.0) / 116.0
    const fx = a / LAB_KAPPA + fy
    const fz = fy - b / LAB_LAMBDA

    // inverse f(t): t > δ ? t³ : (t - 4/29)·3δ²
    const xn = std.select((fx - LAB_OFFSET) * LAB_INV_SLOPE, std.pow(fx, 3.0), fx > LAB_DELTA)
    const yn = std.select((fy - LAB_OFFSET) * LAB_INV_SLOPE, std.pow(fy, 3.0), fy > LAB_DELTA)
    const zn = std.select((fz - LAB_OFFSET) * LAB_INV_SLOPE, std.pow(fz, 3.0), fz > LAB_DELTA)

    // denormalize by D65 white point
    const x = xn * 0.95047
    const y = yn * 1.00000
    const z = zn * 1.08883

    // XYZ → linear RGB (D65)
    const r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    const g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    const bRgb = x * 0.0556434 - y * 0.2040259 + z * 1.0572252
    return d.vec3f(r, g, bRgb)
})

/** CIE Lab → LCh (cylindrical). */
export const labToLch = tgpu.fn([d.vec3f], d.vec3f)((lab) => {
    'use gpu'
    const L = lab.x
    const a = lab.y
    const b = lab.z
    const C = std.sqrt(a * a + b * b)
    const h = std.atan2(b, a)
    return d.vec3f(L, C, h)
})

/** LCh → CIE Lab. */
export const lchToLab = tgpu.fn([d.vec3f], d.vec3f)((lch) => {
    'use gpu'
    const L = lch.x
    const C = lch.y
    const h = lch.z
    return d.vec3f(L, C * std.cos(h), C * std.sin(h))
})

// ─── Alpha-weighted mixing ────────────────────────────────────────────────────

/** One alpha-weighted mix step in the working space: (a·wA + b·wB) / safeWeight. */
const awMix = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec3f)((a, b, wA, wB, sw) => {
    'use gpu'
    return a.mul(wA).add(b.mul(wB)).div(sw)
})

/**
 * Hue (radians) for one alpha-weighted mix step: interpolates along the SHORTEST arc (the
 * difference is wrapped into [-π, π)), and an achromatic endpoint (chroma/saturation below
 * `achromatic`) borrows the other endpoint's hue so grey/white/black never drag in a junk hue.
 * Same weight fraction as awMix (wB / sw). Exported for CPU golden tests (DualFn).
 */
export const mixHueShortest = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (hA, cA, hB, cB, wB, sw, achromatic) => {
        'use gpu'
        const hueA = std.select(hA, hB, cA < achromatic)
        const hueB = std.select(hB, hueA, cB < achromatic)
        const diff = hueB - hueA
        const wrapped = diff - std.floor((diff + PI_F) * INV_TWO_PI) * TWO_PI
        return hueA + wrapped * (wB / sw)
    },
)

/** awMix for (L, C, h) cylindrical spaces (OKLCh): L/C alpha-weighted, h shortest-arc. */
const awMixOklch = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec3f)((a, b, wA, wB, sw) => {
    'use gpu'
    const m = awMix(a, b, wA, wB, sw)
    return d.vec3f(m.x, m.y, mixHueShortest(a.z, a.y, b.z, b.y, wB, sw, OKLCH_ACHROMATIC))
})

/** awMix for CIE LCh (L, C, h): L/C alpha-weighted, h shortest-arc. */
const awMixLch = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec3f)((a, b, wA, wB, sw) => {
    'use gpu'
    const m = awMix(a, b, wA, wB, sw)
    return d.vec3f(m.x, m.y, mixHueShortest(a.z, a.y, b.z, b.y, wB, sw, LCH_ACHROMATIC))
})

/**
 * awMix for HSL (h, s, l): s/l alpha-weighted, h shortest-arc. Achromatic is judged on the RGB
 * chroma s·(1−|2l−1|), not raw s — near white/black HSL saturation amplifies float noise.
 */
const awMixHsl = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec3f)((a, b, wA, wB, sw) => {
    'use gpu'
    const m = awMix(a, b, wA, wB, sw)
    const chromaA = a.y * (1.0 - std.abs(a.z * 2.0 - 1.0))
    const chromaB = b.y * (1.0 - std.abs(b.z * 2.0 - 1.0))
    return d.vec3f(mixHueShortest(a.x, chromaA, b.x, chromaB, wB, sw, HS_ACHROMATIC), m.y, m.z)
})

/** awMix for HSV (h, s, v): s/v alpha-weighted, h shortest-arc; achromatic judged on chroma s·v. */
const awMixHsv = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec3f)((a, b, wA, wB, sw) => {
    'use gpu'
    const m = awMix(a, b, wA, wB, sw)
    return d.vec3f(mixHueShortest(a.x, a.y * a.z, b.x, b.y * b.z, wB, sw, HS_ACHROMATIC), m.y, m.z)
})

/**
 * Six pre-transpiled `mixColors` variants — full round-trip mix for each color space.
 * Alpha weighting ensures transparent colors (alpha=0) never contribute their RGB; output
 * alpha is the weighted sum (standard alpha interpolation). Signature matches the composer's
 * expectation: (colorA: v4f P3-linear, colorB: v4f P3-linear, t: f32) → v4f P3-linear.
 */
export const mixColorsLinear = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((colorA, colorB, t) => {
    'use gpu'
    const weightA = colorA.w * (1.0 - t)
    const weightB = colorB.w * t
    const totalWeight = weightA + weightB
    const safeWeight = std.max(totalWeight, 0.001)
    const rgb = awMix(colorA.xyz, colorB.xyz, weightA, weightB, safeWeight)
    return d.vec4f(rgb, totalWeight)
})

export const mixColorsOklch = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((colorA, colorB, t) => {
    'use gpu'
    const weightA = colorA.w * (1.0 - t)
    const weightB = colorB.w * t
    const totalWeight = weightA + weightB
    const safeWeight = std.max(totalWeight, 0.001)
    const oklchA = oklabToOklch(rgbToOklab(p3ToSRGB(colorA.xyz)))
    const oklchB = oklabToOklch(rgbToOklab(p3ToSRGB(colorB.xyz)))
    const mixed = awMixOklch(oklchA, oklchB, weightA, weightB, safeWeight)
    const rgb = sRGBToP3(oklabToRgb(oklchToOklab(mixed)))
    return d.vec4f(rgb, totalWeight)
})

export const mixColorsOklab = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((colorA, colorB, t) => {
    'use gpu'
    const weightA = colorA.w * (1.0 - t)
    const weightB = colorB.w * t
    const totalWeight = weightA + weightB
    const safeWeight = std.max(totalWeight, 0.001)
    const oklabA = rgbToOklab(p3ToSRGB(colorA.xyz))
    const oklabB = rgbToOklab(p3ToSRGB(colorB.xyz))
    const mixed = awMix(oklabA, oklabB, weightA, weightB, safeWeight)
    const rgb = sRGBToP3(oklabToRgb(mixed))
    return d.vec4f(rgb, totalWeight)
})

export const mixColorsHsl = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((colorA, colorB, t) => {
    'use gpu'
    const weightA = colorA.w * (1.0 - t)
    const weightB = colorB.w * t
    const totalWeight = weightA + weightB
    const safeWeight = std.max(totalWeight, 0.001)
    const hslA = rgbToHsl(p3ToSRGB(colorA.xyz))
    const hslB = rgbToHsl(p3ToSRGB(colorB.xyz))
    const mixed = awMixHsl(hslA, hslB, weightA, weightB, safeWeight)
    const rgb = sRGBToP3(hslToRgb(mixed))
    return d.vec4f(rgb, totalWeight)
})

export const mixColorsHsv = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((colorA, colorB, t) => {
    'use gpu'
    const weightA = colorA.w * (1.0 - t)
    const weightB = colorB.w * t
    const totalWeight = weightA + weightB
    const safeWeight = std.max(totalWeight, 0.001)
    const hsvA = rgbToHsv(p3ToSRGB(colorA.xyz))
    const hsvB = rgbToHsv(p3ToSRGB(colorB.xyz))
    const mixed = awMixHsv(hsvA, hsvB, weightA, weightB, safeWeight)
    const rgb = sRGBToP3(hsvToRgb(mixed))
    return d.vec4f(rgb, totalWeight)
})

export const mixColorsLch = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((colorA, colorB, t) => {
    'use gpu'
    const weightA = colorA.w * (1.0 - t)
    const weightB = colorB.w * t
    const totalWeight = weightA + weightB
    const safeWeight = std.max(totalWeight, 0.001)
    const lchA = labToLch(rgbToLab(p3ToSRGB(colorA.xyz)))
    const lchB = labToLch(rgbToLab(p3ToSRGB(colorB.xyz)))
    const mixed = awMixLch(lchA, lchB, weightA, weightB, safeWeight)
    const rgb = sRGBToP3(labToRgb(lchToLab(mixed)))
    return d.vec4f(rgb, totalWeight)
})

/** mixColors variants keyed by colorSpaceMode (0 linear · 1 OKLCh · 2 OKLAB · 3 HSL · 4 HSV · 5 LCH). */
export const mixColorsVariants = {
    0: mixColorsLinear,
    1: mixColorsOklch,
    2: mixColorsOklab,
    3: mixColorsHsl,
    4: mixColorsHsv,
    5: mixColorsLch,
} as const

/**
 * Mix two P3-linear colors in the given space. `colorSpaceMode` is a compile-time JS number
 * (part of the pipeline hash), so this is a CPU/builder-level dispatcher — it selects the
 * matching pre-transpiled variant. Do NOT call this inside a `'use gpu'` body; reference
 * `mixColorsVariants[mode]` (a `tgpu.fn`) there instead.
 */
export const mixColors = (colorA: d.v4f, colorB: d.v4f, t: number, colorSpaceMode: number): d.v4f => {
    const variant = mixColorsVariants[colorSpaceMode as keyof typeof mixColorsVariants] ?? mixColorsLinear
    return variant(colorA, colorB, t)
}

// ─── Hoisted forward/back conversion (multi-stop gradient optimization) ───────
// The forward P3 → mix-space conversion is pixel-invariant, so gradient shaders that blend
// many samples convert fixed endpoints ONCE, interpolate in-space, then back-convert once.
// Same math as the round-trip inside mixColors, split at the mix boundary.

/** Forward half of a color-space blend: P3-linear → working space for `colorSpaceMode`. */
export const convertP3ToMixSpace = (p3: d.v3f, colorSpaceMode: number): d.v3f => {
    switch (colorSpaceMode) {
        case 1: return oklabToOklch(rgbToOklab(p3ToSRGB(p3)))
        case 2: return rgbToOklab(p3ToSRGB(p3))
        case 3: return rgbToHsl(p3ToSRGB(p3))
        case 4: return rgbToHsv(p3ToSRGB(p3))
        case 5: return labToLch(rgbToLab(p3ToSRGB(p3)))
        default: return p3 // Mode 0: linear RGB stays in P3
    }
}

/** Back half of a color-space blend: working space → P3-linear. Pair with convertP3ToMixSpace. */
export const convertMixSpaceToP3 = (mixed: d.v3f, colorSpaceMode: number): d.v3f => {
    switch (colorSpaceMode) {
        case 1: return sRGBToP3(oklabToRgb(oklchToOklab(mixed)))
        case 2: return sRGBToP3(oklabToRgb(mixed))
        case 3: return sRGBToP3(hslToRgb(mixed))
        case 4: return sRGBToP3(hsvToRgb(mixed))
        case 5: return sRGBToP3(labToRgb(lchToLab(mixed)))
        default: return mixed // Mode 0: already P3
    }
}

// ─── Preconverted-endpoint mixing (chaining) ──────────────────────────────────

/** Working-space step packed with its accumulated alpha, for chaining preconverted mixes. */
export const MixStep = d.struct({conv: d.vec3f, alpha: d.f32})

/**
 * One alpha-weighted mix step performed IN the working space, for chaining (identical math to
 * the mix inside mixColors, minus the back-conversion). Feed one step's conv/alpha into the
 * next; call `convertMixSpaceToP3` once on the final conv.
 */
export const mixPreconvertedInSpace = tgpu.fn([d.vec3f, d.f32, d.vec3f, d.f32, d.f32], MixStep)(
    (convA, alphaA, convB, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        return MixStep({conv: awMix(convA, convB, weightA, weightB, safeWeight), alpha: totalWeight})
    },
)

/**
 * Hue-aware in-space steps for the cylindrical spaces (same contract as mixPreconvertedInSpace):
 * the hue channel takes the shortest arc and an achromatic endpoint borrows the other hue.
 * `mixPreconvertedInSpace` itself stays a plain per-channel mix — correct for linear RGB / OKLab
 * only; pick the step for a compile-time mode via `mixPreconvertedInSpaceVariants`.
 */
export const mixPreconvertedInSpaceOklch = tgpu.fn([d.vec3f, d.f32, d.vec3f, d.f32, d.f32], MixStep)(
    (convA, alphaA, convB, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        return MixStep({conv: awMixOklch(convA, convB, weightA, weightB, safeWeight), alpha: totalWeight})
    },
).$name('mixPreconvertedInSpaceOklch')

export const mixPreconvertedInSpaceHsl = tgpu.fn([d.vec3f, d.f32, d.vec3f, d.f32, d.f32], MixStep)(
    (convA, alphaA, convB, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        return MixStep({conv: awMixHsl(convA, convB, weightA, weightB, safeWeight), alpha: totalWeight})
    },
).$name('mixPreconvertedInSpaceHsl')

export const mixPreconvertedInSpaceHsv = tgpu.fn([d.vec3f, d.f32, d.vec3f, d.f32, d.f32], MixStep)(
    (convA, alphaA, convB, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        return MixStep({conv: awMixHsv(convA, convB, weightA, weightB, safeWeight), alpha: totalWeight})
    },
).$name('mixPreconvertedInSpaceHsv')

export const mixPreconvertedInSpaceLch = tgpu.fn([d.vec3f, d.f32, d.vec3f, d.f32, d.f32], MixStep)(
    (convA, alphaA, convB, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        return MixStep({conv: awMixLch(convA, convB, weightA, weightB, safeWeight), alpha: totalWeight})
    },
).$name('mixPreconvertedInSpaceLch')

/** In-space step keyed by colorSpaceMode (0 linear · 1 OKLCh · 2 OKLAB · 3 HSL · 4 HSV · 5 LCH). */
export const mixPreconvertedInSpaceVariants = {
    0: mixPreconvertedInSpace,
    1: mixPreconvertedInSpaceOklch,
    2: mixPreconvertedInSpace,
    3: mixPreconvertedInSpaceHsl,
    4: mixPreconvertedInSpaceHsv,
    5: mixPreconvertedInSpaceLch,
} as const

/**
 * Six pre-transpiled variants of the preconverted-endpoint mix + back-conversion.
 * convA/convB are endpoints already in the working space (see convertP3ToMixSpace); alphas are
 * read per pixel; result is P3-linear. Mode 0 endpoints are raw P3 — no back-conversion.
 */
export const mixPreconvertedLinear = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec4f)(
    (convA, convB, alphaA, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        const mixed = awMix(convA, convB, weightA, weightB, safeWeight)
        return d.vec4f(mixed, totalWeight)
    },
)

export const mixPreconvertedOklch = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec4f)(
    (convA, convB, alphaA, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        const mixed = awMixOklch(convA, convB, weightA, weightB, safeWeight)
        return d.vec4f(sRGBToP3(oklabToRgb(oklchToOklab(mixed))), totalWeight)
    },
)

export const mixPreconvertedOklab = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec4f)(
    (convA, convB, alphaA, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        const mixed = awMix(convA, convB, weightA, weightB, safeWeight)
        return d.vec4f(sRGBToP3(oklabToRgb(mixed)), totalWeight)
    },
)

export const mixPreconvertedHsl = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec4f)(
    (convA, convB, alphaA, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        const mixed = awMixHsl(convA, convB, weightA, weightB, safeWeight)
        return d.vec4f(sRGBToP3(hslToRgb(mixed)), totalWeight)
    },
)

export const mixPreconvertedHsv = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec4f)(
    (convA, convB, alphaA, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        const mixed = awMixHsv(convA, convB, weightA, weightB, safeWeight)
        return d.vec4f(sRGBToP3(hsvToRgb(mixed)), totalWeight)
    },
)

export const mixPreconvertedLch = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.vec4f)(
    (convA, convB, alphaA, alphaB, t) => {
        'use gpu'
        const weightA = alphaA * (1.0 - t)
        const weightB = alphaB * t
        const totalWeight = weightA + weightB
        const safeWeight = std.max(totalWeight, 0.001)
        const mixed = awMixLch(convA, convB, weightA, weightB, safeWeight)
        return d.vec4f(sRGBToP3(labToRgb(lchToLab(mixed))), totalWeight)
    },
)

export const mixPreconvertedVariants = {
    0: mixPreconvertedLinear,
    1: mixPreconvertedOklch,
    2: mixPreconvertedOklab,
    3: mixPreconvertedHsl,
    4: mixPreconvertedHsv,
    5: mixPreconvertedLch,
} as const

/**
 * Mix two preconverted endpoints (see convertP3ToMixSpace) in `colorSpaceMode` and convert the
 * result to P3-linear. CPU/builder-level dispatcher (mode is compile-time), same as mixColors.
 */
export const mixPreconvertedColors = (
    convA: d.v3f,
    convB: d.v3f,
    alphaA: number,
    alphaB: number,
    t: number,
    colorSpaceMode: number,
): d.v4f => {
    const variant = mixPreconvertedVariants[colorSpaceMode as keyof typeof mixPreconvertedVariants] ?? mixPreconvertedLinear
    return variant(convA, convB, alphaA, alphaB, t)
}

// ─── CPU-side conversions (plain JS) ─────────────────────────────────────────
// Used when endpoint colors are plain uniforms: convert once per frame on the CPU. Mirrors the
// forward GPU conversions exactly so hoisted gradients halve their per-pixel transcendental cost.

const EPS_CPU = 0.0001
const LAB_DELTA_CPU = 6.0 / 29.0

const p3ToSRGBCPU = (r: number, g: number, b: number): [number, number, number] => [
    r * 1.2249401 - g * 0.2249404,
    r * -0.0420569 + g * 1.0420571,
    r * -0.0196376 - g * 0.0786361 + b * 1.0982735,
]

const rgbToOklabCPU = (r: number, g: number, b: number): [number, number, number] => {
    const l = Math.cbrt(r * 0.4122214708 + g * 0.5363325363 + b * 0.0514459929)
    const m = Math.cbrt(r * 0.2119034982 + g * 0.6806995451 + b * 0.1073969566)
    const s = Math.cbrt(r * 0.0883024619 + g * 0.2817188376 + b * 0.6299787005)
    return [
        l * 0.2104542553 + m * 0.7936177850 - s * 0.0040720468,
        l * 1.9779984951 - m * 2.4285922050 + s * 0.4505937099,
        l * 0.0259040371 + m * 0.7827717662 - s * 0.8086757660,
    ]
}

// Shared hue/sector math for HSL & HSV — hue in radians, matching the GPU path
const rgbHueCPU = (r: number, g: number, b: number, maxVal: number, delta: number): number => {
    const den = Math.max(delta, EPS_CPU)
    const hue = maxVal === r ? (g - b) / den
        : maxVal === g ? 2 + (b - r) / den
        : 4 + (r - g) / den
    return hue * (Math.PI / 3)
}

const rgbToHslCPU = (r: number, g: number, b: number): [number, number, number] => {
    const maxVal = Math.max(r, g, b)
    const minVal = Math.min(r, g, b)
    const delta = maxVal - minVal
    const l = (maxVal + minVal) * 0.5
    const s = delta / Math.max(1 - Math.abs(2 * l - 1), EPS_CPU)
    return [rgbHueCPU(r, g, b, maxVal, delta), s, l]
}

const rgbToHsvCPU = (r: number, g: number, b: number): [number, number, number] => {
    const maxVal = Math.max(r, g, b)
    const delta = maxVal - Math.min(r, g, b)
    const s = delta / Math.max(maxVal, EPS_CPU)
    return [rgbHueCPU(r, g, b, maxVal, delta), s, maxVal]
}

const rgbToLabCPU = (r: number, g: number, b: number): [number, number, number] => {
    const x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    const y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    const z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    const delta3 = LAB_DELTA_CPU ** 3
    const f = (t: number) => t > delta3 ? Math.cbrt(t) : t / (3 * LAB_DELTA_CPU * LAB_DELTA_CPU) + 4 / 29
    const fx = f(x), fy = f(y), fz = f(z)
    return [fy * 116 - 16, (fx - fy) * 500, (fy - fz) * 200]
}

/**
 * CPU mirror of the per-space forward conversion used inside mixColors. Input: P3-linear RGB.
 * Output: the triplet in the mix space for `colorSpaceMode` (1 OKLCh · 2 OKLAB · 3 HSL · 4 HSV
 * · 5 LCH). Mode 0 needs no conversion — keep using mixColors directly for linear RGB.
 */
export const convertP3ToMixSpaceCPU = (
    r: number, g: number, b: number, colorSpaceMode: number,
): [number, number, number] => {
    const [sr, sg, sb] = p3ToSRGBCPU(r, g, b)
    switch (colorSpaceMode) {
        case 1: {
            const [L, a, bb] = rgbToOklabCPU(sr, sg, sb)
            return [L, Math.sqrt(a * a + bb * bb), Math.atan2(bb, a)]
        }
        case 2: return rgbToOklabCPU(sr, sg, sb)
        case 3: return rgbToHslCPU(sr, sg, sb)
        case 4: return rgbToHsvCPU(sr, sg, sb)
        case 5: {
            const [L, a, bb] = rgbToLabCPU(sr, sg, sb)
            return [L, Math.sqrt(a * a + bb * bb), Math.atan2(bb, a)]
        }
        default: return [r, g, b]
    }
}

/**
 * Mix a color's rgb toward a target color by a scalar amount, preserving the source alpha —
 * the GPU body behind std's `tintToward` colorOp (linear-RGB; the colorSpace-aware path is
 * `mixColorsVariants`). Pure — CPU-executable as a DualFn for golden tests.
 */
export const mixToward = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((color, target, amount) => {
    'use gpu'
    const rgb = d.vec3f(color.x, color.y, color.z)
    const targetRgb = d.vec3f(target.x, target.y, target.z)
    const mixed = std.mix(rgb, targetRgb, d.vec3f(amount))
    return d.vec4f(mixed.x, mixed.y, mixed.z, color.w)
})
