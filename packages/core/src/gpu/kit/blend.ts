/**
 * The 20 blend modes. Each mode is a
 * `tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)` `(base, overlay, opacity)` so the composer can
 * reference them as externals in its blend folds.
 *
 * Every mode shares the straight-alpha "over" compositing tail via the shared `overComposite`
 * helper. The OKLCh/OKLab "normal" variants composite in their color space and reuse
 * kit/colorMixing's conversion fns.
 *
 * Modes that branch on a threshold express `step` via `std.select` inside the `step3` helper
 * rather than a `mix(a, b, step(e, x))` call: typegpu@0.11.9 has no CPU implementation of `step`
 * on vectors (needed for the golden-value gate). `step3` is mathematically identical to WGSL
 * `step` (returns 1.0 where x ≥ edge, else 0.0).
 */
import {d, std, tgpu} from './index'
import {mixHueShortest, oklabToOklch, oklabToRgb, oklchToOklab, p3ToSRGB, rgbToOklab, sRGBToP3} from './colorMixing'

/** OKLCh chroma below which a colour is achromatic (its hue is noise) — matches colorMixing. */
const OKLCH_ACHROMATIC = 0.001

/** Luminance (ITU-R BT.709). */
export const luminance = tgpu.fn([d.vec3f], d.f32)((rgb) => {
    'use gpu'
    return rgb.x * 0.2126 + rgb.y * 0.7152 + rgb.z * 0.0722
})

/** WGSL `step(edge, x)` via select (1.0 where x ≥ edge, else 0.0); CPU-executable (see header). */
const step3 = tgpu.fn([d.vec3f, d.vec3f], d.vec3f)((edge, x) => {
    'use gpu'
    return std.select(d.vec3f(0.0), d.vec3f(1.0), std.ge(x, edge))
})

/**
 * Straight-alpha "over" compositing tail shared by every blend mode:
 *   finalAlpha = overlayAlpha + base.a·(1 - overlayAlpha)
 *   rgb        = blended·overlayAlpha + base.rgb·(base.a·(1 - overlayAlpha))
 */
const overComposite = tgpu.fn([d.vec4f, d.vec3f, d.f32], d.vec4f)((base, blended, overlayAlpha) => {
    'use gpu'
    const baseWeight = base.w * (1.0 - overlayAlpha)
    const finalAlpha = overlayAlpha + baseWeight
    const rgb = blended.mul(overlayAlpha).add(base.xyz.mul(baseWeight))
    return d.vec4f(rgb, finalAlpha)
})

/**
 * Premultiplied-alpha → straight-alpha. RTT boundaries produce premultiplied data (the "over"
 * composite accumulates RGB weighted by alpha), so effects that SAMPLE an RTT and return into the
 * blend pipeline (which expects straight alpha) must convert back. No-op at alpha 1. Used by the
 * uvRemap-distortion / RTT-filter effects (Twirl, Bulge, Liquify, …).
 *   straightRGB = alpha > 0.001 ? rgb / alpha : vec3(0); returns vec4(straightRGB, alpha).
 */
export const unpremultiplyAlpha = tgpu.fn([d.vec4f], d.vec4f)((color) => {
    'use gpu'
    const alpha = color.w
    // std.select arg order is (ifFalse, ifTrue, cond): straight rgb where alpha > 0.001, else 0.
    const straightRGB = std.select(d.vec3f(0.0, 0.0, 0.0), color.xyz.div(alpha), alpha > 0.001)
    return d.vec4f(straightRGB, alpha)
})

export const normal = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    return overComposite(base, overlay.xyz, overlayAlpha)
})

export const normalOklch = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const baseWeight = base.w * (1.0 - overlayAlpha)
    const finalAlpha = overlayAlpha + baseWeight
    const baseOklch = oklabToOklch(rgbToOklab(p3ToSRGB(base.xyz)))
    const overlayOklch = oklabToOklch(rgbToOklab(p3ToSRGB(overlay.xyz)))
    const blendedOklch = overlayOklch.mul(overlayAlpha).add(baseOklch.mul(baseWeight))
    // Hue is an angle: take the shortest arc between base and overlay (by the overlay's share of
    // the total weight), and an achromatic side borrows the other side's hue.
    const hue = mixHueShortest(baseOklch.z, baseOklch.y, overlayOklch.z, overlayOklch.y, overlayAlpha,
        std.max(finalAlpha, 0.001), OKLCH_ACHROMATIC)
    const rgb = sRGBToP3(oklabToRgb(oklchToOklab(d.vec3f(blendedOklch.x, blendedOklch.y, hue))))
    return d.vec4f(rgb, finalAlpha)
})

export const normalOklab = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const baseWeight = base.w * (1.0 - overlayAlpha)
    const finalAlpha = overlayAlpha + baseWeight
    const baseOklab = rgbToOklab(p3ToSRGB(base.xyz))
    const overlayOklab = rgbToOklab(p3ToSRGB(overlay.xyz))
    const blendedOklab = overlayOklab.mul(overlayAlpha).add(baseOklab.mul(baseWeight))
    const rgb = sRGBToP3(oklabToRgb(blendedOklab))
    return d.vec4f(rgb, finalAlpha)
})

export const multiply = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const blended = base.xyz.mul(overlay.xyz)
    return overComposite(base, blended, overlayAlpha)
})

export const screen = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const blended = std.sub(1.0, std.sub(1.0, base.xyz).mul(std.sub(1.0, overlay.xyz)))
    return overComposite(base, blended, overlayAlpha)
})

export const linearDodge = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    // Linear Dodge (Add): base + overlay. Intentionally NOT clamped — scene-referred pipeline
    // lets light accumulate past white and roll off via tone mapping.
    const overlayAlpha = overlay.w * opacity
    const blended = base.xyz.add(overlay.xyz)
    return overComposite(base, blended, overlayAlpha)
})

export const overlay = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const mult = base.xyz.mul(overlay.xyz).mul(2.0)
    const scr = std.sub(1.0, std.sub(1.0, base.xyz).mul(std.sub(1.0, overlay.xyz)).mul(2.0))
    const blended = std.mix(mult, scr, step3(d.vec3f(0.5), base.xyz))
    return overComposite(base, blended, overlayAlpha)
})

export const difference = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const blended = std.abs(base.xyz.sub(overlay.xyz))
    return overComposite(base, blended, overlayAlpha)
})

export const colorDodge = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // base / (1 - overlay), guarded against division by zero via step on (1 - overlay)
    const oneMinusOverlay = std.sub(1.0, overlay.xyz)
    const dodged = std.mix(
        base.xyz,
        std.min(base.xyz.div(oneMinusOverlay), d.vec3f(1.0)),
        step3(d.vec3f(0.001), oneMinusOverlay),
    )
    return overComposite(base, dodged, overlayAlpha)
})

export const exclusion = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // base + overlay - 2·base·overlay
    const blended = base.xyz.add(overlay.xyz).sub(base.xyz.mul(overlay.xyz).mul(2.0))
    return overComposite(base, blended, overlayAlpha)
})

export const color = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // Take color from overlay, luminance from base
    const baseLum = luminance(base.xyz)
    const overlayLum = luminance(overlay.xyz)
    const lumDiff = baseLum - overlayLum
    const blended = overlay.xyz.add(lumDiff)
    return overComposite(base, blended, overlayAlpha)
})

export const luminosity = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // Take luminance from overlay, color from base
    const baseLum = luminance(base.xyz)
    const overlayLum = luminance(overlay.xyz)
    const lumRatio = overlayLum / (baseLum + 0.0001)
    const blended = base.xyz.mul(lumRatio)
    return overComposite(base, blended, overlayAlpha)
})

export const darken = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const blended = std.min(base.xyz, overlay.xyz)
    return overComposite(base, blended, overlayAlpha)
})

export const lighten = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const blended = std.max(base.xyz, overlay.xyz)
    return overComposite(base, blended, overlayAlpha)
})

export const colorBurn = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // 1 - (1 - base) / overlay; black where overlay is 0
    const burned = std.max(std.sub(1.0, std.sub(1.0, base.xyz).div(overlay.xyz.add(0.0001))), d.vec3f(0.0))
    const blended = std.mix(d.vec3f(0.0), burned, step3(d.vec3f(0.0001), overlay.xyz))
    return overComposite(base, blended, overlayAlpha)
})

export const linearBurn = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // base + overlay - 1 (clamped to 0)
    const blended = std.max(base.xyz.add(overlay.xyz).add(-1.0), d.vec3f(0.0))
    return overComposite(base, blended, overlayAlpha)
})

export const softLight = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // Pegtop formula
    const mult = base.xyz.mul(overlay.xyz).mul(2.0)
    const baseSquared = base.xyz.mul(base.xyz)
    const softDark = mult.add(baseSquared.mul(std.sub(1.0, overlay.xyz.mul(2.0))))
    const baseSqrt = std.sqrt(base.xyz)
    const softLightVal = base.xyz.mul(std.sub(1.0, overlay.xyz)).mul(2.0).add(baseSqrt.mul(overlay.xyz.mul(2.0).sub(1.0)))
    const blended = std.mix(softDark, softLightVal, step3(d.vec3f(0.5), overlay.xyz))
    return overComposite(base, blended, overlayAlpha)
})

export const hardLight = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    const mult = base.xyz.mul(overlay.xyz).mul(2.0)
    const scr = std.sub(1.0, std.sub(1.0, base.xyz).mul(std.sub(1.0, overlay.xyz)).mul(2.0))
    const blended = std.mix(mult, scr, step3(d.vec3f(0.5), overlay.xyz))
    return overComposite(base, blended, overlayAlpha)
})

export const hue = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // Overlay hue (chromaticity direction), base saturation intensity + luminance
    const baseLum = luminance(base.xyz)
    const overlayLum = luminance(overlay.xyz)
    const baseChromaticity = base.xyz.sub(baseLum)
    const overlayChromaticity = overlay.xyz.sub(overlayLum)
    const baseChromaLength = std.length(baseChromaticity)
    // normalize(0) is NaN (poisons overComposite even at overlayAlpha 0 — NaN·0 = NaN);
    // an achromatic overlay has no hue, so fall back to zero chroma (gray at baseLum).
    const overlayChromaLength = std.length(overlayChromaticity)
    const overlayChromaDir = std.select(
        d.vec3f(0.0),
        overlayChromaticity.div(overlayChromaLength),
        overlayChromaLength > 0.0001,
    )
    const blended = overlayChromaDir.mul(baseChromaLength).add(baseLum)
    return overComposite(base, blended, overlayAlpha)
})

export const saturation = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)((base, overlay, opacity) => {
    'use gpu'
    const overlayAlpha = overlay.w * opacity
    // Overlay saturation (chroma intensity), base hue direction + luminance
    const baseLum = luminance(base.xyz)
    const overlayLum = luminance(overlay.xyz)
    const baseChromaticity = base.xyz.sub(baseLum)
    const overlayChromaticity = overlay.xyz.sub(overlayLum)
    // Same NaN guard as hue: an achromatic base has no chroma direction → stay gray.
    const baseChromaLength = std.length(baseChromaticity)
    const baseChromaDir = std.select(
        d.vec3f(0.0),
        baseChromaticity.div(baseChromaLength),
        baseChromaLength > 0.0001,
    )
    const overlayChromaLength = std.length(overlayChromaticity)
    const blended = baseChromaDir.mul(overlayChromaLength).add(baseLum)
    return overComposite(base, blended, overlayAlpha)
})

/**
 * All 20 blend modes keyed by the exact mode names used by the design editor / presets.
 * The composer references these as externals.
 */
export const blendModes = {
    normal,
    'normal-oklch': normalOklch,
    'normal-oklab': normalOklab,
    multiply,
    screen,
    linearDodge,
    overlay,
    difference,
    colorDodge,
    exclusion,
    color,
    luminosity,
    darken,
    lighten,
    colorBurn,
    linearBurn,
    softLight,
    hardLight,
    hue,
    saturation,
} as const

export type BlendMode = keyof typeof blendModes

/**
 * Blend two colors using the named mode — CPU/builder-level dispatcher. `blendMode` is
 * compile-time (part of the pipeline hash), so selection stays on the CPU. Inside a `'use gpu'`
 * body, reference `blendModes[mode]` (a `tgpu.fn`) instead.
 */
export const applyBlendMode = (
    base: d.v4f,
    overlay: d.v4f,
    blendMode: BlendMode = 'normal',
    opacity: number = 1.0,
): d.v4f => {
    const fn = blendModes[blendMode] ?? blendModes.normal
    return fn(base, overlay, opacity)
}
