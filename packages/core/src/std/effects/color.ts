/**
 * std/effects/color — color filters: words that recolor the layer inside them, one pixel at a time.
 *
 * Each word returns a pointwise effect for a filter definition's `effect:` field. It reads
 * the child's color at this pixel and returns the adjusted color, alpha kept. Pass props with
 * `p('name')`; numbers are in the units named on each word. The adjustment words have a value
 * that changes nothing (`saturate` at 1, `hueRotate` at 0); declare it with `identityWhen` so
 * the filter is skipped when it does nothing.
 */
// Maintainer notes: each word wraps one pointwise color op from kit/colorOps (the GPU body) into
// a PointwiseEffect. The authored shader file passes prop bindings, the word owns the wiring,
// including builder-level compile-time work (Tint's body pair, the tone filters' color-space
// mixes, the gradient map's palette branch).
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {formatFloat} from '../../gpu/contract'
import {call, expr, vec4, floatE, ZERO} from '../../gpu/composer'
import {colorOps, colorMixing, tgpu, d, std, tonemap} from '../../gpu/kit/index'
import type {PointwiseEffect} from '../types'
import type {PropRef} from '../values'
import {pointwiseOp, resolveArg, uniformOf, type ArgSpec} from '../invoke'
import type {FilterParams} from '../../gpu/scaffolds/pointwiseFilter'

/**
 * Rotate the child's hue around the color wheel.
 *
 * `shift` is in degrees. 0 leaves the color unchanged, 180 swaps every color for its
 * complement, 360 comes back around. Alpha kept.
 *
 * @example
 * ```ts
 * effect: hueRotate(p('shift'))
 * ```
 * @tip Declare `identityWhen: isZero('shift')` so the filter is skipped at 0.
 * @see saturate, vibrance, tint
 */
export function hueRotate(shift: ArgSpec): PointwiseEffect {
    // Rodrigues rotation of RGB around the achromatic (1,1,1) axis; the deg→rad fold lives in the body.
    return pointwiseOp(colorOps.hueRotate, 'hueRotate', [shift])
}

/**
 * Scale the child's saturation.
 *
 * `intensity` is a multiplier: 0 is grayscale, 1 leaves the color unchanged, values above 1
 * (up to about 3) push colors past their original saturation.
 *
 * @example
 * ```ts
 * effect: saturate(p('intensity'))
 * ```
 * @tip Declare `identityWhen: isValue('intensity', 1)`; the no-change value is 1, not 0.
 * @see vibrance, grayscale
 */
export function saturate(intensity: ArgSpec): PointwiseEffect {
    // Rec.709 luminance-weighted mix between gray and the color.
    return pointwiseOp(colorOps.saturate, 'saturate', [intensity])
}

/**
 * Boost muted colors while leaving already-vivid ones mostly alone.
 *
 * `intensity` is an adjustment around 0 (about −2 to 2): 0 leaves the color unchanged,
 * positive values add saturation to dull pixels, negative values drain it.
 *
 * @example
 * ```ts
 * effect: vibrance(p('intensity'))
 * ```
 * @tip Declare `identityWhen: isZero('intensity')`.
 * @see saturate
 */
export function vibrance(intensity: ArgSpec): PointwiseEffect {
    // Selective saturation: an adjustment around zero, unlike saturate's multiplier.
    return pointwiseOp(colorOps.vibrance, 'vibrance', [intensity])
}

/**
 * Adjust the child's brightness and contrast together.
 *
 * Both are sliders from −1 to 1 with 0 as no change. `brightness` adds to every channel.
 * `contrast` stretches colors away from mid-gray (positive) or flattens them toward it
 * (negative).
 *
 * @example
 * ```ts
 * effect: brightnessContrast({brightness: p('brightness'), contrast: p('contrast')})
 * ```
 * @tip Declare `identityWhen: allOf(isZero('brightness'), isZero('contrast'))`.
 * @see exposure
 */
export function brightnessContrast(slots: {brightness: ArgSpec; contrast: ArgSpec}): PointwiseEffect {
    // `(rgb - 0.5) * (contrast + 1) + 0.5 + brightness`; the `+ 1` is folded in the body so raw 0 is identity.
    return pointwiseOp(colorOps.brightnessContrast, 'brightnessContrast', [slots.brightness, slots.contrast])
}

/**
 * Multiply the child's brightness, like opening a camera's aperture.
 *
 * `gain` of 1 leaves the color unchanged, 0.5 halves it, 2 doubles it. Blacks stay black.
 * Values pushed above 1 are not clamped, so a glow or bloom filter placed after this one can
 * pick them up.
 *
 * @example
 * ```ts
 * effect: exposure(p('exposure'))
 * ```
 * @tip Declare `identityWhen: isValue('exposure', 1)`; the no-change value is 1.
 * @see brightnessContrast
 */
export function exposure(gain: ArgSpec): PointwiseEffect {
    // Pure scalar gain, deliberately unclamped (HDR-safe: values above 1 survive to the tone-mapping shoulder).
    return pointwiseOp(colorOps.exposure, 'exposure', [gain])
}

/**
 * Turn the child black and white, weighted the way the eye sees brightness. Alpha kept.
 *
 * @example
 * ```ts
 * effect: grayscale()
 * ```
 * @see saturate, duotone
 */
export function grayscale(): PointwiseEffect {
    // Rec.709 luminance weights.
    return pointwiseOp(colorOps.grayscale, 'grayscale', [])
}

/**
 * Flip every color of the child to its negative. Alpha kept.
 *
 * @example
 * ```ts
 * effect: invert()
 * ```
 * @see solarize
 */
export function invert(): PointwiseEffect {
    return pointwiseOp(colorOps.invert, 'invert', [])
}

/**
 * Invert only the bright parts of the child, the darkroom solarization look.
 *
 * `threshold` (0–1) is the brightness above which a pixel flips. `strength` (0–1) fades the
 * flipped result over the original: 0 is untouched, 1 is fully solarized.
 *
 * @example
 * ```ts
 * effect: solarize({threshold: p('threshold'), strength: p('strength')})
 * ```
 * @tip Declare `identityWhen: isZero('strength')`.
 * @see invert, posterize
 */
export function solarize(slots: {threshold: ArgSpec; strength: ArgSpec}): PointwiseEffect {
    // Threshold is on Rec.601 luminance.
    return pointwiseOp(colorOps.solarize, 'solarize', [slots.threshold, slots.strength])
}

/**
 * Reduce the child to a few flat color levels, like a screen-printed poster.
 *
 * `steps` is the number of levels per channel; 2 to 20 is the useful range. Fewer steps give
 * bolder bands.
 *
 * @example
 * ```ts
 * effect: posterize(p('levels'))
 * ```
 * @see solarize, isolines
 */
export function posterize(steps: ArgSpec): PointwiseEffect {
    // floor(c * steps) / steps per channel.
    return pointwiseOp(colorOps.posterize, 'posterize', [steps])
}

/**
 * Wash the child toward one color.
 *
 * `amount` is 0–1: 0 leaves the child alone, 1 replaces its color entirely. While the
 * boolean `preserveLuminosity` prop is on, the result is scaled back to the child's original
 * brightness, so the tint changes hue without darkening. Declare that prop
 * `compileTime: true` with `transform: (v) => (v ? 1 : 0)`. Alpha kept.
 *
 * @example
 * ```ts
 * effect: tint({color: p('color'), amount: p('amount'), preserveLuminosity: p('preserveLuminosity')})
 * ```
 * @tip Reach for `tintToward` instead when you do not need the brightness-preserving mode.
 * @see tintToward, duotone, hueRotate
 */
export function tint(slots: {color: PropRef; amount: ArgSpec; preserveLuminosity: PropRef}): PointwiseEffect {
    // `preserveLuminosity` picks a compile-time body PAIR; only the chosen variant emits. The
    // preserving body rescales the tinted output back to the original Rec.601 brightness.
    return {
        kind: 'pointwise',
        body: (propValues) =>
            Number(propValues[slots.preserveLuminosity.name]) > 0
                ? {fn: colorOps.tintPreserveLuma, hint: 'tintPreserveLuma'}
                : {fn: colorOps.tintPlain, hint: 'tintPlain'},
        args: (params) => [resolveArg(slots.color, params), resolveArg(slots.amount, params)],
    }
}

/**
 * Sprinkle film grain over the child, heaviest in the shadows.
 *
 * `strength` (0–1) sets how visible the grain is. `bias` (0–10) concentrates it in darker
 * areas; 0 grains everything evenly. While the boolean `animated` prop is on the pattern
 * changes every frame; off, it freezes. The definition must declare the grain's clock:
 * `extraFields: {animTime: {schema: schema.f32, initial: 0}}`. Alpha kept.
 *
 * @example
 * ```ts
 * effect: filmGrain({strength: p('strength'), bias: p('bias'), animated: p('animated')})
 * ```
 * @tip Declare `identityWhen: isZero('strength')`.
 * @see posterize
 */
export function filmGrain(slots: {strength: ArgSpec; bias: ArgSpec; animated: PropRef}): PointwiseEffect {
    // The grain drift is CPU-accumulated into the `animTime` extraField each frame
    // (`animTime += deltaTime * animated * 10`, frozen when `animated` is off); this word owns
    // the per-frame clock and the GPU read.
    return {
        kind: 'pointwise',
        body: {fn: colorOps.filmGrain, hint: 'filmGrain'},
        args: (params) => [
            params.ctx.uv,
            params.ctx.viewportSize,
            resolveArg(slots.strength, params),
            resolveArg(slots.bias, params),
            params.uniforms.animTime,
        ],
        // `animated` is a boolean → getCpuValue coerced `> 0`. A closure holds the running total
        // (reset on recompose).
        setup: (params) => {
            let animTimeAcc = 0
            params.onBeforeRender(({deltaTime}) => {
                const on = ((params.getCpuValue(slots.animated.name) as number) > 0) ? 1 : 0
                animTimeAcc += deltaTime * on * 10.0
                params.setExtraField('animTime', animTimeAcc)
            })
        },
    }
}

// Compile-time enum reads for the isolines branches (applied by the bridge, so propValues carry
// the mapped number — these stay robust to a raw string).
const sourceModeOf = (raw: unknown): number => (typeof raw === 'number' ? raw : raw === 'alpha' ? 1 : 0)
const colorModeOf = (raw: unknown): number => (typeof raw === 'number' ? raw : raw === 'custom' ? 1 : 0)

/**
 * Draw topographic contour lines over the child, traced from its brightness or its alpha.
 *
 * `levels` is how many contours (2–30). `lineWidth` is in pixels. `softness` (0–1) blurs the
 * line edges. `gamma` crowds the contours toward the highlights (below 1) or the shadows
 * (above 1). The boolean `invert` prop flips the source. `source` (`'luminance' | 'alpha'`)
 * and `colorMode` (`'source' | 'custom'`) are select props declared `compileTime: true`: in
 * source mode the lines take the child's own colors over transparent, in custom mode they take
 * `lineColor` over `backgroundColor`.
 *
 * @example
 * ```ts
 * effect: isolines({source: p('source'), levels: p('levels'), lineWidth: p('lineWidth'), softness: p('softness'), gamma: p('gamma'), invert: p('invert'), colorMode: p('colorMode'), lineColor: p('lineColor'), backgroundColor: p('backgroundColor')})
 * ```
 * @tip The lines replace the child rather than sit over it; set `blendWithChildren: false` in the definition.
 * @see posterize, duotone
 */
export function isolines(slots: {
    source: PropRef
    levels: ArgSpec
    lineWidth: ArgSpec
    softness: ArgSpec
    gamma: ArgSpec
    invert: ArgSpec
    colorMode: PropRef
    lineColor: PropRef
    backgroundColor: PropRef
}): PointwiseEffect {
    // Compile-time `source` body pair, banded with fwidth anti-aliasing. Compile-time `colorMode`
    // picks the palette: 'custom' draws lineColor over backgroundColor; 'source' draws the child's
    // own colors (alpha 1) over transparent.
    return {
        kind: 'pointwise',
        // Compile-time source channel: luminance dot OR alpha read — only the chosen body emits.
        body: (propValues) =>
            sourceModeOf(propValues[slots.source.name]) === 1
                ? {fn: colorOps.isolinesFromAlpha, hint: 'isolinesFromAlpha'}
                : {fn: colorOps.isolinesFromLuma, hint: 'isolinesFromLuma'},
        args: (params) =>
            [slots.levels, slots.lineWidth, slots.softness, slots.gamma, slots.invert].map((s) => resolveArg(s, params)),
        compose: (lineMask, params) => {
            // Compile-time color mode: source → line=child.rgb@α1 over transparent; custom → props.
            const custom = colorModeOf(params.propValues[slots.colorMode.name]) === 1
            const bg = custom ? uniformOf(slots.backgroundColor, params) : ZERO
            const line = custom ? uniformOf(slots.lineColor, params) : vec4(params.childNode.member('rgb'), floatE(1))
            return call(colorOps.contourCompose, 'contourCompose', [bg, line, lineMask])
        },
    }
}

/** The compile-time color-space mix variant for a colorSpace prop binding. */
function mixVariantFor(colorSpace: PropRef, params: FilterParams) {
    const mode = (params.propValues[colorSpace.name] as number) ?? 0
    return colorMixing.mixColorsVariants[mode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
}

/**
 * Repaint the child in two colors: `colorA` for the shadows, `colorB` for the highlights.
 *
 * `blend` (0–1) is the brightness where the two meet. `colorSpace` is a select prop with
 * `transform: transformColorSpace` and `compileTime: true`; it picks the space the colors
 * mix in (`'oklch'` for vivid, `'linear'` for plain). Alpha kept.
 *
 * @example
 * ```ts
 * effect: duotone({colorA: p('colorA'), colorB: p('colorB'), blend: p('blend'), colorSpace: p('colorSpace')})
 * ```
 * @see tritone, gradientMap, grayscale
 */
export function duotone(slots: {colorA: PropRef; colorB: PropRef; blend: ArgSpec; colorSpace: PropRef}): PointwiseEffect {
    // The luminance→t body runs first; the two-color mix happens here in the compile-time color space.
    return pointwiseOp(colorOps.duotoneT, 'duotoneT', [slots.blend], {
        compose: (t, params) => {
            const variant = mixVariantFor(slots.colorSpace, params)
            const duotoneColor = call(variant, 'mixColors', [uniformOf(slots.colorA, params), uniformOf(slots.colorB, params), t])
            return vec4(duotoneColor.member('rgb'), params.childNode.member('a'))
        },
    })
}

/**
 * Repaint the child in three colors: `colorA` for the shadows, `colorB` for the midtones,
 * `colorC` for the highlights.
 *
 * `blendMid` (0–1) is the brightness the midtone color sits at. `colorSpace` is a select prop
 * with `transform: transformColorSpace` and `compileTime: true`; it picks the space the colors
 * mix in. Alpha kept.
 *
 * @example
 * ```ts
 * effect: tritone({colorA: p('colorA'), colorB: p('colorB'), colorC: p('colorC'), blendMid: p('blendMid'), colorSpace: p('colorSpace')})
 * ```
 * @see duotone, gradientMap
 */
export function tritone(slots: {colorA: PropRef; colorB: PropRef; colorC: PropRef; blendMid: ArgSpec; colorSpace: PropRef}): PointwiseEffect {
    // The luminance→factors body runs first; the three mixes happen here in the compile-time color space.
    return pointwiseOp(colorOps.tritoneFactors, 'tritoneFactors', [slots.blendMid], {
        compose: (factors, params) => {
            const shadowToMid = factors.member('x')
            const midToHighlight = factors.member('y')
            const finalT = factors.member('z')
            const variant = mixVariantFor(slots.colorSpace, params)
            const colorA = uniformOf(slots.colorA, params)
            const colorB = uniformOf(slots.colorB, params)
            const colorC = uniformOf(slots.colorC, params)
            // Shadows→midtones, then midtones→highlights, then blend the two by luminance position.
            const lowerBlend = call(variant, 'mixColors', [colorA, colorB, shadowToMid])
            const upperBlend = call(variant, 'mixColors', [colorB, colorC, midToHighlight])
            const finalColor = call(variant, 'mixColors', [lowerBlend, upperBlend, finalT])
            return vec4(finalColor.member('rgb'), params.childNode.member('a'))
        },
    })
}

// Inigo Quilez cosine palettes: color(t) = a + b*cos(2π*(c*t + d))
// https://iquilezles.org/articles/palettes/
const PALETTES: Record<number, {a: number[]; b: number[]; c: number[]; d: number[]}> = {
    0: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.333, 0.667] },   // Rainbow
    1: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.1, 0.2] },       // Sunset
    2: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.3, 0.2, 0.2] },       // Ocean
    3: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.8, 0.9, 0.3] },       // Fire
    4: { a: [0.5, 0.5, 0.5], b: [0.25, 0.25, 0.25], c: [1, 1, 1], d: [0.0, 0.25, 0.5] },   // Pastel
    5: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [2, 1, 1], d: [0.5, 0.2, 0.25] }       // Neon
}

// The `palette` prop is a compile-time cpu-only string; map string → index at builder level.
const PALETTE_INDEX: Record<string, number> = {rainbow: 0, sunset: 1, ocean: 2, fire: 3, pastel: 4, neon: 5, custom: 6}

/** A vec3f literal Expr (there is no `vec3` builder factory; build the raw WGSL, decimal-safe). */
const vec3E = (v: number[]): Expr => expr(`vec3f(${formatFloat(v[0])}, ${formatFloat(v[1])}, ${formatFloat(v[2])})`)

/**
 * Recolor the child by its brightness through a gradient, the Photoshop gradient map.
 *
 * `palette` is a select prop declared `compileTime: true`: one of the built-in gradients
 * `'rainbow' | 'sunset' | 'ocean' | 'fire' | 'pastel' | 'neon'`, or `'custom'` to use
 * `colorLow`, `colorMid` and `colorHigh` (shadows, midtones, highlights) mixed in
 * `colorSpace`. `blackPoint` and `whitePoint` (0–1) clip the input brightness. `contrast`
 * (0–3) steepens the mapping. `speed` scrolls the gradient over time; 0 holds it still.
 * `strength` (0–1) fades the result over the original. Alpha kept.
 *
 * @example
 * ```ts
 * effect: gradientMap({palette: p('palette'), colorLow: p('colorLow'), colorMid: p('colorMid'), colorHigh: p('colorHigh'), speed: p('speed'), contrast: p('contrast'), blackPoint: p('blackPoint'), whitePoint: p('whitePoint'), strength: p('strength'), colorSpace: p('colorSpace')})
 * ```
 * @tip Declare `identityWhen: isZero('strength')`.
 * @see duotone, tritone
 */
export function gradientMap(slots: {
    palette: PropRef
    colorLow: PropRef
    colorMid: PropRef
    colorHigh: PropRef
    speed: PropRef
    contrast: ArgSpec
    blackPoint: ArgSpec
    whitePoint: ArgSpec
    strength: PropRef
    colorSpace: PropRef
}): PointwiseEffect {
    // Remap luminance through black/white points and contrast, then color it: a built-in cosine
    // palette, or (palette 'custom') a cyclic low→mid→high ramp mixed in the compile-time color
    // space, scrolled over time by speed, and blended back over the original by strength.
    return pointwiseOp(colorOps.gradientMapT, 'gradientMapT', [slots.blackPoint, slots.whitePoint, slots.contrast], {
        // The luminance→t body runs first; everything after it is builder-level: a compile-time
        // palette branch, and for custom colors the per-segment mixes in the compile-time color space.
        compose: (t, params): Expr => {
            const {childNode, propValues, ctx} = params
            const strength = uniformOf(slots.strength, params)
            // Global clock; phase = speed × time — animates the gradient scroll.
            const phase = uniformOf(slots.speed, params).mul(ctx.time)

            // `palette` is a compile-time string → JS-branch the whole fragment (recomposes on change).
            const paletteIdx = PALETTE_INDEX[(propValues[slots.palette.name] as string) ?? 'rainbow'] ?? 0
            if (paletteIdx !== 6) {
                const p = PALETTES[paletteIdx] ?? PALETTES[0]
                const mapped = call(colorOps.gradientMapCosine, 'gradientMapCosine', [t, phase, vec3E(p.a), vec3E(p.b), vec3E(p.c), vec3E(p.d)])
                return call(colorOps.gradientMapCompose, 'gradientMapCompose', [childNode, mapped, strength])
            }

            // Custom 3-stop cyclic ramp (low → mid → high → low) in the compile-time color space.
            const variant = mixVariantFor(slots.colorSpace, params)
            const phaseData = call(colorOps.gradientMapCustomPhase, 'gradientMapCustomPhase', [t, phase])
            const fr = phaseData.member('x')
            const i = phaseData.member('y')
            const colorLow = uniformOf(slots.colorLow, params)
            const colorMid = uniformOf(slots.colorMid, params)
            const colorHigh = uniformOf(slots.colorHigh, params)
            const seg0 = call(variant, 'mixColors', [colorLow, colorMid, fr])
            const seg1 = call(variant, 'mixColors', [colorMid, colorHigh, fr])
            const seg2 = call(variant, 'mixColors', [colorHigh, colorLow, fr])
            const mappedColor = call(colorOps.gradientMapSelect, 'gradientMapSelect', [seg0, seg1, seg2, i])
            return call(colorOps.gradientMapCompose, 'gradientMapCompose', [childNode, mappedColor.member('rgb'), strength])
        },
    })
}

// ── 3D-LUT atlas grade ──────────────────────────────────────────────────────────────────────
//
// A measured N³ color LUT (HaldCLUT convention: sRGB-in/sRGB-out) uploaded as an rgba8unorm 2D
// atlas — blue slices side by side along x (width N·N, height N). A lookup is two hardware-
// bilinear taps (the blue slice below and above) mixed by the blue fraction: the classic
// 2D-atlas trilinear. Sample coords are half-texel inset so bilinear never bleeds across slices.

interface LutAtlasFns {
    uv: ReturnType<typeof makeLutAtlasUvFn>
    apply: ReturnType<typeof makeLutAtlasApplyFn>
}

function makeLutAtlasUvFn(size: number) {
    const LUT_MAX = size - 1
    const INV_ATLAS_W = 1 / (size * size)
    const INV_LUT_N = 1 / size
    /**
     * Atlas UV for one blue slice. The linear input is clamped to the LUT domain and
     * sRGB-encoded first; `hi` selects the slice below (0) or above (1) the encoded blue value.
     * Red/green land on hardware bilinear inside the slice.
     */
    return tgpu.fn([d.vec4f, d.f32], d.vec2f)((color, hi) => {
        'use gpu'
        const enc = tonemap.linearToSrgb(std.clamp(color.xyz, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0)))
        const slice = enc.z * d.f32(LUT_MAX)
        const s = std.clamp(std.floor(slice) + hi, d.f32(0), d.f32(LUT_MAX))
        const u = (s * d.f32(size) + 0.5 + enc.x * d.f32(LUT_MAX)) * d.f32(INV_ATLAS_W)
        const v = (0.5 + enc.y * d.f32(LUT_MAX)) * d.f32(INV_LUT_N)
        return d.vec2f(u, v)
    }).$name('lutAtlasUv')
}

function makeLutAtlasApplyFn(size: number) {
    const LUT_MAX = size - 1
    /**
     * Complete the trilinear grade: mix the two slice taps by the blue fraction (in sRGB, the
     * LUT's own space), decode back to linear, and blend toward the graded color by `strength`.
     * Alpha preserved; colors above 1.0 grade at their clamped value (the LUT domain is SDR).
     */
    return tgpu.fn([d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec4f)(
        (color, tapLo, tapHi, strength) => {
            'use gpu'
            const enc = tonemap.linearToSrgb(std.clamp(color.xyz, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0)))
            const slice = enc.z * d.f32(LUT_MAX)
            const f = slice - std.floor(slice)
            const graded = tonemap.srgbToLinear(std.mix(tapLo.xyz, tapHi.xyz, d.vec3f(f, f, f)))
            const rgb = std.mix(color.xyz, graded, d.vec3f(strength, strength, strength))
            return d.vec4f(rgb.x, rgb.y, rgb.z, color.w)
        },
    ).$name('lutAtlasApply')
}

const lutAtlasFnCache = new Map<number, LutAtlasFns>()

/**
 * @internal The per-size LUT atlas fn pair (size is a compile-time constant baked into the
 * bodies), cached so every grade of the same size shares one pair.
 */
export function lutAtlasFnsFor(size: number): LutAtlasFns {
    let fns = lutAtlasFnCache.get(size)
    if (!fns) {
        fns = {uv: makeLutAtlasUvFn(size), apply: makeLutAtlasApplyFn(size)}
        lutAtlasFnCache.set(size, fns)
    }
    return fns
}

/**
 * Grade a color through a measured 3D color lookup table (LUT), the film-stock look.
 *
 * Returns a stage `(color, params) => gradedColor` for a `gpu:` fragment, not an `effect:`.
 * `select` is a select prop declared `compileTime: true` that names the table; `decode(key)`
 * returns that table's bytes as RGBA, `size` levels per channel laid out as `size` square
 * slices side by side (red across, green down, one slice per blue level), sRGB in and out.
 * `fallback` is the key used when the prop is unset. `strength` (0–1) fades the graded color
 * over the original. Alpha kept; colors brighter than white grade as white.
 *
 * @example
 * ```ts
 * const grade = lutAtlasGrade({select: p('stock'), strength: p('strength'), size: 17, decode: decodeStockLut, fallback: 'portrait400', label: 'film-lut'})
 * // then, inside the definition's gpu.fragment: grade(childColor, params)
 * ```
 * @tip Changing `select` rebuilds the shader; keep the table list short.
 * @see tint, gradientMap
 */
export function lutAtlasGrade(slots: {
    select: PropRef
    strength: PropRef
    size: number
    decode: (key: string) => Uint8Array
    fallback: string
    label: string
}): (color: Expr, params: FilterParams | GpuFragmentParams) => Expr {
    // `select` keys `decode` (the CPU-side LUT bytes for the baked selection), decoded once per
    // composition into an rgba8unorm slice atlas (HaldCLUT convention); `strength` blends toward
    // the graded color.
    const fns = lutAtlasFnsFor(slots.size)
    return (color, params) => {
        const {propValues, uniforms, createDataTexture, registerMediaTexture, onCleanup} = params as GpuFragmentParams
        const key = (propValues[slots.select.name] as string) ?? slots.fallback
        const lutTex = createDataTexture({
            width: slots.size * slots.size, height: slots.size, format: 'rgba8unorm',
            data: slots.decode(key), label: `${slots.label}-${key}`,
        })
        onCleanup(() => lutTex.destroy())
        const lutKit = registerMediaTexture(() => lutTex.texture)

        const tapLo = lutKit.sample(call(fns.uv, 'lutAtlasUvLo', [color, floatE(0)]), 'linearClamp')
        const tapHi = lutKit.sample(call(fns.uv, 'lutAtlasUvHi', [color, floatE(1)]), 'linearClamp')
        return call(fns.apply, 'lutAtlasApply', [color, tapLo, tapHi, uniforms[slots.strength.name]])
    }
}
