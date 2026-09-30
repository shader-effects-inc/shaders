/**
 * std/paint/fields — fields, palettes and the ramp that joins them.
 *
 * A field is a number per pixel: a distance from a point, a noise value, a flow pattern. A
 * palette turns that number into a color. `rampOver` evaluates a field at each pixel and
 * looks the result up in a palette, which is how every gradient and most textures are drawn.
 * The other words shape a field before it meets the palette: `warped` makes it flow,
 * `scaledVolume` lets you zoom it, `tone` and `threshold` reshape the value, `rings` and
 * `tiles` repeat it. Fields nest, so read a recipe from the inside out.
 */
// Maintainer notes (not part of the reference):
//  - `rampOver` owns the UV idiom (`uvContext ?? ctx.uv`) and the effective-viewport aspect, and
//    hands the field a vec2 UV coordinate. Coordinate-changing combinators (`scaledVolume`) may
//    hand their inner field a different coordinate type (vec3).
//  - Structural slots (`edges`, a palette's `space`) are compile-time props: read as CPU values,
//    JS-branched, only the selected WGSL path is emitted.
//  - Fields are pure per composition; `share` binds a WGSL local so two consumers do not
//    re-derive the same pattern.
//  - The GPU bodies live in the kit (`kit/gradientPaints.ts`, `kit/fields.ts`, `kit/tone.ts`).
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {abs, clamp, fract, local, mul, sub, vec2, vec3} from '../math'
import {call, floatE, vec4, asLocal} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {gradientPaints, colorStops, colorMixing, fields as fieldsKit, noise as noiseKit, noiseColor} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {resolveScalar, uniformOf, type ArgSpec} from '../invoke'
import {transformColorSpace} from '../../utilities/transformations'
import {packScatterAnchors} from '../../utilities/scatterAnchors'

/** What a generator's `paint:` field takes: a function that returns the color at this pixel. */
export type Paint = (params: GpuFragmentParams) => Expr

/**
 * A number per pixel, computed from a coordinate.
 *
 * Distance fields (`dist.*`) and `flowField` take a uv coordinate (0–1 across the canvas, y
 * down). Noise fields take a 3D coordinate, so wrap them in `scaledVolume`.
 */
export type Field = (params: GpuFragmentParams, coord: Expr) => Expr

/**
 * A color for a field value between 0 and 1.
 *
 * Build one with `pair` (two colors) or `stops` (a multi-stop ramp), then hand it to `rampOver`.
 */
export type Palette = (t: Expr, params: GpuFragmentParams) => Expr

/**
 * A number input: a prop `p('name')`, a plain number, or a prop remapped as `{base, add, mul}`.
 *
 * The remapped form resolves to `(base + add) * mul`. Use it when a slider's range is not the
 * unit the word wants, such as a 0–100 slider that should become a small bias.
 *
 * @example
 * ```ts
 * threshold(field, {low: 0.3, high: 0.7, bias: {base: p('blend'), add: -50, mul: 0.006}})
 * ```
 */
export type FieldScalar = ArgSpec | {base: ArgSpec; add?: number; mul?: number}

// ── Resolution helpers ──────────────────────────────────────────────────────────────────

function baseArg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

function fArg(spec: FieldScalar, params: GpuFragmentParams): Expr {
    if (typeof spec === 'object' && spec !== null && 'base' in spec) {
        let e = baseArg(spec.base, params)
        if (spec.add !== undefined) e = e.add(spec.add)
        if (spec.mul !== undefined) e = e.mul(spec.mul)
        return e
    }
    return baseArg(spec, params)
}

/** The composed UV + effective viewport every paint evaluates against. */
function frame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

function structural(ref: PropRef, params: GpuFragmentParams): unknown {
    return params.propValues[ref.name]
}

// ── Palettes ────────────────────────────────────────────────────────────────────────────

/**
 * @internal
 * Compile-time color-space mode reader — robust to a raw string (a preset loaded before the
 * prop transform ran) as well as the bridge-mapped number.
 */
export function colorSpaceModeOf(raw: unknown): number {
    if (typeof raw === 'number') return raw
    if (typeof raw === 'string') return transformColorSpace(raw)
    return 0
}

/**
 * @internal
 * Pick the color-mix body for the compile-time color space bound by `space` (linear when
 * absent). The mode is a compile-time prop value, so it JS-branches to a specialised variant
 * and participates in the structural hash.
 */
export function mixColorsIn(space: PropRef | undefined, params: GpuFragmentParams) {
    const mode = space ? colorSpaceModeOf(params.propValues[space.name]) : 0
    return colorMixing.mixColorsVariants[mode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
}

/**
 * A three-color blend: `a` to `b` by the first value, then on to `c` by the second.
 *
 * Returns a function you call with two 0–1 values and the params. Both mixes happen in the
 * color space named by the `space` prop (mark it `compileTime`).
 *
 * @example
 * ```ts
 * paint: (params) => colorLadder3({a: p('colorA'), b: p('colorB'), c: p('colorC'), space: p('colorSpace')})(t1, t2, params)
 * ```
 * @see pair, quadBlend
 */
export function colorLadder3(slots: {a: PropRef; b: PropRef; c: PropRef; space: PropRef}) {
    return (t1: Expr, t2: Expr, params: GpuFragmentParams): Expr => {
        const variant = mixColorsIn(slots.space, params)
        const cAB = call(variant, 'mixColors', [uniformOf(slots.a, params), uniformOf(slots.b, params), t1])
        return call(variant, 'mixColors', [cAB, uniformOf(slots.c, params), t2])
    }
}

/**
 * A palette of two color props, mixed in the color space named by the `space` prop.
 *
 * Mark the `space` prop `compileTime`. Field value 0 gives `a`, 1 gives `b`.
 *
 * @example
 * ```ts
 * paint: rampOver(dist.radial({center: p('center'), radius: p('radius'), aspect: 1, skew: 0}), pair(p('colorA'), p('colorB'), p('colorSpace')))
 * ```
 * @see stops, standardPalette
 */
export function pair(a: PropRef, b: PropRef, space: PropRef): Palette {
    return (t, params) => {
        const spaceMode = colorSpaceModeOf(structural(space, params))
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        return call(variant, 'mixColors', [uniformOf(a, params), uniformOf(b, params), t])
    }
}

/**
 * The standard multi-stop palette, read from the `stops`, `colorA` and `colorB` props.
 *
 * Declare `stops: colorStopsPropConfig()` plus `colorA` and `colorB` on the definition. When
 * the user has set two or more stops the ramp uses them, otherwise it mixes `colorA` to
 * `colorB`. Both mix in the color space named by the `space` prop (mark it `compileTime`).
 *
 * @example
 * ```ts
 * paint: rampOver(dist.linear({from: p('start'), to: p('end'), angle: p('angle')}), stops(p('colorSpace')), {edges: p('edges')})
 * ```
 * @tip The prop names are fixed: `stops`, `colorA`, `colorB`. Use `pair` when yours differ.
 * @see pair, standardPalette
 */
export function stops(space: PropRef): Palette {
    // The multi-stop path reads the fixed `colorsArray`/`positionsArray`/`convertedColorsArray`/
    // `stopCount` uniforms that `colorStopsPropConfig` packs; the fallback is the colorA/colorB pair.
    return (t, params) => {
        const spaceMode = colorSpaceModeOf(structural(space, params))
        const stopCount = (params.propValues.stopCount as number) ?? 0
        if (stopCount > 1) {
            return colorStops.mixColorStopsRuntime(
                t,
                {
                    colorsArray: params.uniforms.colorsArray,
                    positionsArray: params.uniforms.positionsArray,
                    convertedColorsArray: params.uniforms.convertedColorsArray,
                    stopCount: params.uniforms.stopCount,
                },
                spaceMode,
            )
        }
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        return call(variant, 'mixColors', [params.uniforms.colorA, params.uniforms.colorB, t])
    }
}

// ── rampOver ────────────────────────────────────────────────────────────────────────────

// Edge mode constants (mirror transformEdges: 0=stretch, 1=transparent, 2=mirror, 3=wrap).
const TRANSPARENT = 1
const MIRROR = 2
const WRAP = 3

/**
 * A paint that colors each pixel by looking its field value up in a palette.
 *
 * Pass an `edges` prop (values `stretch`, `transparent`, `mirror`, `wrap`, marked
 * `compileTime`) to choose what happens where the field runs past 0–1: hold the end colors,
 * fade to transparent, bounce back, or repeat. Without it the value goes to the palette as
 * is, which is right for fields that already stay in range such as `rings` or `tone`.
 *
 * @example
 * ```ts
 * paint: rampOver(dist.linear({from: p('start'), to: p('end'), angle: p('angle')}), stops(p('colorSpace')), {edges: p('edges')})
 * ```
 * @tip This is the whole recipe for a gradient: a `dist.*` field, a palette, done.
 * @see dist, pair, stops, rings, tiles
 */
export function rampOver(field: Field, palette: Palette, opts?: {edges?: PropRef}): Paint {
    return (params) => {
        const {uv} = frame(params)
        const t = field(params, uv)

        if (!opts?.edges) return palette(t, params)

        // Edge-mode finalT (compile-time branch), then clamp for the color lookup.
        const edgeMode = (structural(opts.edges, params) as number) ?? 0
        let finalT: Expr
        if (edgeMode === MIRROR) finalT = call(gradientPaints.gradientEdgeMirror, 'gradientEdgeMirror', [t])
        else if (edgeMode === WRAP) finalT = call(gradientPaints.gradientEdgeWrap, 'gradientEdgeWrap', [t])
        else if (edgeMode === TRANSPARENT) finalT = t
        else finalT = call(gradientPaints.gradientClamp01, 'gradientClamp01', [t])
        const clampedT = call(gradientPaints.gradientClamp01, 'gradientClamp01', [finalT])

        const color = palette(clampedT, params)

        // Transparent edge mode fades alpha to 0 outside the [0,1] band (RGB unchanged).
        if (edgeMode === TRANSPARENT) {
            const cut = call(gradientPaints.gradientTransparentAlpha, 'gradientTransparentAlpha', [t])
            return vec4(color.member('rgb'), color.member('a').mul(cut))
        }
        return color
    }
}

// ── Metric-distance fields ──────────────────────────────────────────────────────────────

/**
 * Distance fields: the shape of a gradient, as a number that grows away from a line or point.
 *
 * Each takes position props made with `transformPosition` and returns 0 at the start and 1
 * at the end, unbounded beyond. Feed one to `rampOver`.
 *
 * @see rampOver, rings, tiles
 */
export const dist = {
    /**
     * A straight gradient from one point to another.
     *
     * 0 at `from`, 1 at `to`. `angle` in degrees turns the whole gradient about the midpoint.
     *
     * @example
     * ```ts
     * paint: rampOver(dist.linear({from: p('start'), to: p('end'), angle: p('angle')}), stops(p('colorSpace')), {edges: p('edges')})
     * ```
     * @see rampOver
     */
    linear(slots: {from: PropRef; to: PropRef; angle: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.gradientRawT, 'gradientRawT', [
                uniformOf(slots.from, params), uniformOf(slots.to, params), fArg(slots.angle, params), uv, viewport,
            ])
        }
    },

    /**
     * A round or elliptical gradient spreading out from a center.
     *
     * 0 at `center`, 1 at `radius` (a fraction of the canvas height). `aspect` 1 is a circle,
     * larger values squeeze it horizontally. `skew` in degrees tilts the ellipse.
     *
     * @example
     * ```ts
     * paint: rampOver(rings(dist.radial({center: p('center'), radius: p('radius'), aspect: p('aspect'), skew: p('skewAngle')}), p('repeat')), stops(p('colorSpace')))
     * ```
     * @see rings
     */
    radial(slots: {center: PropRef; radius: FieldScalar; aspect: FieldScalar; skew: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.radialDist, 'radialDist', [
                uniformOf(slots.center, params), fArg(slots.skew, params), fArg(slots.aspect, params), fArg(slots.radius, params), uv, viewport,
            ])
        }
    },

    /**
     * A sweep around a center, like a clock hand: 0 to 1 once around.
     *
     * The sweep starts at the left of the center and runs counter-clockwise. `rotation` in
     * degrees moves the start. Wrap it in `tiles` to repeat the sweep.
     *
     * @example
     * ```ts
     * paint: rampOver(tiles(dist.conic({center: p('center'), rotation: p('rotation')}), p('repeat')), stops(p('colorSpace')))
     * ```
     * @see tiles
     */
    conic(slots: {center: PropRef; rotation: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.conicSweep, 'conicSweep', [
                uniformOf(slots.center, params), fArg(slots.rotation, params), uv, viewport,
            ])
        }
    },

    /**
     * A diamond-shaped gradient spreading out from a center.
     *
     * 0 at `center`, 1 at `size` (a fraction of the canvas height). `roundness` 0 is a sharp
     * diamond and 1 is a square. `rotation` in degrees tilts it.
     *
     * @example
     * ```ts
     * paint: rampOver(rings(dist.diamond({center: p('center'), size: p('size'), rotation: p('rotation'), roundness: p('roundness')}), p('repeat')), stops(p('colorSpace')))
     * ```
     * @see rings
     */
    diamond(slots: {center: PropRef; size: FieldScalar; rotation: FieldScalar; roundness: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.diamondDist, 'diamondDist', [
                uniformOf(slots.center, params), fArg(slots.size, params), fArg(slots.rotation, params), fArg(slots.roundness, params), uv, viewport,
            ])
        }
    },

    /**
     * A turning spiral stripe: 0 on the background, 1 on the stroke.
     *
     * It spins on the layer's clock, so declare `animatedTime: {speed: 'speed'}`. `scale` sets
     * how many turns fit, `width` (0–2) the stroke thickness, `falloff` (0–1) thins the stroke
     * away from the center, `softness` (0–1) blurs its edge. Unlike the other distance fields
     * this one is already 0–1, so pair it with a palette of background and stroke colors.
     *
     * @example
     * ```ts
     * paint: rampOver(dist.spiral({center: p('center'), scale: p('scale'), width: p('strokeWidth'), falloff: p('strokeFalloff'), softness: p('softness')}), pair(p('background'), p('stroke'), p('colorSpace')))
     * ```
     */
    spiral(slots: {center: PropRef; scale: FieldScalar; width: FieldScalar; falloff: FieldScalar; softness: FieldScalar}): Field {
        // Stays a fused kit body: its anti-aliasing reads `fwidth` of the spiral parameter, so the
        // offset must be derived inside the same fragment-scope function the derivatives sample.
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.spiralMask, 'spiralMask', [
                uniformOf(slots.center, params),
                fArg(slots.scale, params),
                fArg(slots.width, params),
                fArg(slots.falloff, params),
                fArg(slots.softness, params),
                animatedTime(params),
                uv,
                viewport,
            ])
        }
    },
}

// ── Field combinators ───────────────────────────────────────────────────────────────────

/**
 * Repeats a field as concentric bands.
 *
 * `count` above 1 restarts the field every 1/count, so a radial gradient becomes rings. At 1
 * or below the field is clamped to 0–1, and values under 1 stretch the gradient further out.
 *
 * @example
 * ```ts
 * paint: rampOver(rings(dist.radial({center: p('center'), radius: p('radius'), aspect: 1, skew: 0}), p('repeat')), stops(p('colorSpace')))
 * ```
 * @tip Each band restarts at the first color rather than bouncing back. For a mirrored repeat, pass `edges` to `rampOver` instead.
 * @see tiles, rampOver
 */
export function rings(field: Field, count: FieldScalar): Field {
    // `count` is a runtime uniform: the kit body selects clamp vs fract per pixel.
    return (params, coord) => call(gradientPaints.repeatRings, 'repeatRings', [field(params, coord), fArg(count, params)])
}

/**
 * Repeats a field `count` times, always wrapping.
 *
 * Made for sweeps: `tiles(dist.conic(...), 4)` gives four sectors. Any count works, including
 * fractions.
 *
 * @example
 * ```ts
 * paint: rampOver(tiles(dist.conic({center: p('center'), rotation: p('rotation')}), p('repeat')), stops(p('colorSpace')))
 * ```
 * @see rings
 */
export function tiles(field: Field, count: FieldScalar): Field {
    return (params, coord) => call(gradientPaints.repeatWrap, 'repeatWrap', [field(params, coord), fArg(count, params)])
}

/**
 * Reshapes a 0–1 field with glow, contrast and balance controls, then flips it.
 *
 * `glow` is a spread control: 5 leaves the field alone, lower values pull the bright parts
 * into tight cores, higher values spread them into a wash. `contrast` scales around the
 * middle, 1 is neutral. `balance` is 0–100 with 50 neutral. `invert` is always true: the
 * result is flipped so the palette's first color lands where the field is high.
 *
 * @example
 * ```ts
 * paint: rampOver(tone(scaledVolume(noiseField('mx3'), {scale: p('scale')}), {glow: p('intensity'), contrast: p('contrast'), balance: p('balance'), invert: true}), stops(p('colorSpace')))
 * ```
 * @tip Raising `balance` pushes more of the canvas toward the first color.
 * @see threshold, noiseField
 */
export function tone(field: Field, opts: {glow: FieldScalar; contrast: FieldScalar; balance: FieldScalar; invert: true}): Field {
    // Kit body: pow(v, 5 / glow) → (v − 0.5) · contrast + 0.5 → + (balance / 100 − 0.5) → 1 − v.
    // Currently the one glow-gamma inverted shape the kit tone library implements for unit fields.
    return (params, coord) => call(gradientPaints.plasmaTone, 'toneRemap', [
        field(params, coord), fArg(opts.glow, params), fArg(opts.contrast, params), fArg(opts.balance, params),
    ])
}

/**
 * Turns a signed field (−1 to 1) into a soft 0–1 mask.
 *
 * The field is first mapped to 0–1 and shifted by `bias`, then eased from 0 at `low` to 1 at
 * `high`. Bring `low` and `high` together for a hard edge.
 *
 * @example
 * ```ts
 * paint: rampOver(threshold(flowField({detail: p('detail')}), {low: 0.3, high: 0.7, bias: {base: p('blend'), add: -50, mul: 0.006}}), stops(p('colorSpace')))
 * ```
 * @see flowField, tone
 */
export function threshold(field: Field, opts: {low: number; high: number; bias?: FieldScalar}): Field {
    return (params, coord) => call(gradientPaints.unitThreshold, 'unitThreshold', [
        field(params, coord),
        opts.bias !== undefined ? fArg(opts.bias, params) : floatE(0),
        floatE(opts.low),
        floatE(opts.high),
    ])
}

/**
 * Evaluates a field once and lets several words read the same result.
 *
 * Wrap a field in `share` when it feeds two places, such as a `threshold` and a `pulse`.
 *
 * @example
 * ```ts
 * const pattern = share(flowField({detail: p('detail')}))
 * // then: threshold(pattern, ...) and pulse(pattern, ...)
 * ```
 * @tip One result per layer, whatever coordinate it is asked for. Do not use it to sample the same field at two offsets.
 * @see flowField, pulse
 */
export function share(field: Field): Field {
    // The first read binds a WGSL local; later reads reuse it. Cached per composition (params).
    const cache = new WeakMap<GpuFragmentParams, Expr>()
    return (params, coord) => {
        let bound = cache.get(params)
        if (!bound) {
            bound = asLocal(field(params, coord), 'field')
            cache.set(params, bound)
        }
        return bound
    }
}

// ── Noise fields ────────────────────────────────────────────────────────────────────────

/** Noise bases usable as a {@link noiseField} — unit-range ([0,1]) samples over a vec3 coordinate. */
const NOISE_BASES = {
    /** MaterialX 3D noise, unit-biased. Hash-based (bit-cast) → GPU-only. */
    mx3: {fn: () => gradientPaints.unitNoise3, hint: 'unitNoise3'},
    /** 3D Perlin gradient noise, ~[0,1]. Hash-based (bit-cast) → GPU-only. */
    perlin: {fn: () => noiseKit.perlin13, hint: 'perlin13'},
    /** 3D value noise — soft blocky cells (Quilez permutation), [0,1]. Pure float. */
    value: {fn: () => noiseKit.value13, hint: 'value13'},
    /** RAW MaterialX 3D noise, signed [-1,1] — pair with a signed tone. GPU-only. */
    mx3signed: {fn: () => noiseKit.mxNoiseFloat3, hint: 'mxNoiseFloat3'},
} as const

/**
 * The noise flavors `noiseField` can sample.
 *
 * `mx3` is smooth all-purpose noise, `perlin` classic gradient noise, `value` softer and
 * blockier. All three return 0–1. `mx3signed` returns −1 to 1 for use with `threshold`.
 */
export type NoiseBasis = keyof typeof NOISE_BASES

/**
 * A field of smooth noise in the chosen flavor.
 *
 * Noise needs a 3D coordinate, so put it inside `scaledVolume` (and `warped` for motion).
 *
 * @example
 * ```ts
 * paint: rampOver(scaledVolume(noiseField('mx3'), {scale: p('scale')}), stops(p('colorSpace')))
 * ```
 * @see scaledVolume, warped, tone
 */
export function noiseField(basis: NoiseBasis): Field {
    const entry = NOISE_BASES[basis]
    return (_params, coord) => call(entry.fn(), entry.hint, [coord])
}

/**
 * Makes a field flow like liquid by bending its coordinate with moving noise.
 *
 * `amount` sets how far the coordinate is pushed (0 is off), `amountScale` multiplies it so
 * a 0–1 slider can reach further. The motion runs on the layer's clock times `timeScale`, so
 * declare `animatedTime: {speed: 'speed'}`. Needs a 3D coordinate: put it inside `scaledVolume`.
 *
 * @example
 * ```ts
 * scaledVolume(warped(noiseField('mx3'), {amount: p('warp'), amountScale: 4, timeScale: 0.125}), {scale: p('scale')})
 * ```
 * @see noiseField, scaledVolume, warpedPoint
 */
export function warped(field: Field, opts: {amount: FieldScalar; amountScale?: number; timeScale?: number}): Field {
    // Two-level Inigo-Quilez domain warp over MaterialX noise; the memoized 'warpDomain' instance
    // is shared with `warpedPoint`.
    const warpFn = fieldsKit.domainWarp2(noiseKit.mxNoiseFloat3, {name: 'warpDomain'})
    return (params, coord) => {
        let amount = fArg(opts.amount, params)
        if (opts.amountScale !== undefined) amount = amount.mul(opts.amountScale)
        let time = animatedTime(params)
        if (opts.timeScale !== undefined) time = time.mul(opts.timeScale)
        return field(params, call(warpFn, 'warpDomain', [coord, time, amount]))
    }
}

/**
 * Gives a noise field a zoomable, aspect-correct coordinate.
 *
 * Turns the pixel's uv into a 3D point so `noiseField` and `warped` can read it. `scale` zooms
 * exponentially: 1 is neutral and each step up magnifies about 2.7 times. A 0–4 slider works
 * well.
 *
 * @example
 * ```ts
 * paint: rampOver(scaledVolume(noiseField('perlin'), {scale: p('scale')}), stops(p('colorSpace')))
 * ```
 * @see noiseField, warped
 */
export function scaledVolume(field: Field, opts: {scale: FieldScalar}): Field {
    // Aspect-corrected z = 0 slab, zoomed by e^(scale − 1).
    return (params, uv) => {
        const {viewport} = frame(params)
        return field(params, call(gradientPaints.volumeDomain, 'volumeDomain', [uv, viewport, fArg(opts.scale, params)]))
    }
}

// ── Flow pattern + shimmer ──────────────────────────────────────────────────────────────

/**
 * A swirling, animated wave pattern from −1 to 1.
 *
 * `detail` sets how fine the swirls are. It moves on the layer's clock, so declare
 * `animatedTime: {speed: 'speed'}`. Feed it to `threshold` for color bands or to `pulse` for
 * a shimmer.
 *
 * @example
 * ```ts
 * paint: rampOver(threshold(flowField({detail: p('detail')}), {low: 0.3, high: 0.7}), stops(p('colorSpace')))
 * ```
 * @see threshold, pulse, share
 */
export function flowField(slots: {detail: FieldScalar}): Field {
    // Three nested sin/cos layers, each domain-warping the previous. Reads the raw coordinate (no
    // aspect correction — the pattern is viewport-relative by design).
    return (params, coord) => call(gradientPaints.flowLayers, 'flowLayers', [coord, fArg(slots.detail, params), animatedTime(params)])
}

/**
 * A gentle brightness pulse that follows a pattern field, hovering around 1.
 *
 * The result swings between 1 − `depth` and 1 + `depth`, so `depth` around 0.02 is subtle.
 * `speed` is how fast it breathes, `span` how much the pattern staggers the phase. Use it as
 * the gain in `shimmered`.
 *
 * @example
 * ```ts
 * paint: shimmered(rampOver(threshold(pattern, {low: 0.3, high: 0.7}), stops(p('colorSpace'))), pulse(pattern, {speed: 2.5, span: 8, depth: 0.015}))
 * ```
 * @see shimmered, flowField
 */
export function pulse(field: Field, opts: {speed: number; span: number; depth: number}): Field {
    // sin(t · speed + pattern · span) · depth + 1
    return (params, coord) => call(gradientPaints.shimmerPulse, 'shimmerPulse', [
        field(params, coord), animatedTime(params), floatE(opts.speed), floatE(opts.span), floatE(opts.depth),
    ])
}

/**
 * A rainbow color for a hue value: 0 is red, cycling through the spectrum and back every 1.
 *
 * Takes any number and wraps it. Returns an rgb color with no alpha.
 *
 * @example
 * ```ts
 * const rgb = hueWheel(math.add(math.mul(angle, 0.42), animatedTime(params)))
 * ```
 */
export function hueWheel(h: Expr): Expr {
    // The classic six-segment hue wheel (Prism's fan, ColorWheel-class effects).
    const x = local(mul(fract(h), 6), 'hue6')
    return vec3(
        clamp(sub(abs(sub(x, 3)), 1), 0, 1),
        clamp(sub(2, abs(sub(x, 2))), 0, 1),
        clamp(sub(2, abs(sub(x, 4))), 0, 1),
    )
}

/**
 * Multiplies a paint's color and alpha by a gain field, for a slow shimmer over a pattern.
 *
 * The gain is read at the raw canvas uv. Build it with `pulse`.
 *
 * @example
 * ```ts
 * paint: shimmered(rampOver(field, stops(p('colorSpace'))), pulse(field, {speed: 2.5, span: 8, depth: 0.015}))
 * ```
 * @see pulse
 */
export function shimmered(paint: Paint, gain: Field): Paint {
    return (params) => paint(params).mul(gain(params, params.ctx.uv))
}

// ── Scatter fields, ramp wrapping, and the standard palette ─────────────────────────────

/**
 * A mesh-gradient field: up to 8 drifting points scattered across the canvas, blended by nearness.
 *
 * Each point carries a fixed 0–1 value and the result is their distance-weighted average, so
 * a palette lookup gives soft patches of color. `at` is an aspect-corrected coordinate (x runs
 * 0 to the aspect ratio, y 0–1). `count` is 1–8, `seed` reshuffles the layout, `drift` (0–1)
 * sets how far points wander on `time`, `softness` sharpens the patches as it rises. Pass
 * `anchors` from `scatteredAnchors` to skip recomputing the points per pixel.
 *
 * @example
 * ```ts
 * const t = scatterField({at: uv, count: uniformOf(p('count'), params), seed, drift: uniformOf(p('drift'), params), aspect, time, softness: 3})
 * return standardPalette(t, params)
 * ```
 * @see scatteredAnchors, wrapRamp, standardPalette
 */
export function scatterField(opts: {
    at: Expr
    count: Expr | number
    seed: Expr | number
    drift: Expr | number
    aspect: Expr | number
    time: Expr | number
    softness: Expr | number
    /** The points precomputed each frame by `scatteredAnchors`. When given, `drift`, `aspect` and `time` are ignored here. */
    anchors?: Expr
}): Expr {
    // Golden-spiral points blended by inverse distance; `softness` is the IDW exponent and `count`
    // gates the fixed-8 GPU loop at runtime. `anchors` is an `array<vec4f, 4>` extraField holding
    // the 8 anchors packed 4-per-vec4.
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    if (opts.anchors) {
        return call(fieldsKit.scatterFieldAnchored, 'scatterField', [
            opts.at, e(opts.count), e(opts.seed), opts.anchors, e(opts.softness),
        ])
    }
    return call(fieldsKit.scatterField, 'scatterField', [
        opts.at, e(opts.count), e(opts.seed), e(opts.drift), e(opts.aspect), e(opts.time), e(opts.softness),
    ])
}

/**
 * Computes the `scatterField` points once per frame instead of once per pixel.
 *
 * Declare a field on the definition, `extraFields: {meshAnchors: {schema: schema.arrayOf(schema.vec4f, 4), initial: new Array(16).fill(0)}}`,
 * name it in `field`, and pass the result as `anchors` to `scatterField`. The points follow
 * the layer's clock, so declare `animatedTime`.
 *
 * @example
 * ```ts
 * const anchors = scatteredAnchors(params, {count: p('count'), seed: p('seed'), drift: p('drift'), field: 'meshAnchors'})
 * const t = scatterField({at: uv, count: uniformOf(p('count'), params), seed, drift: 0, aspect: 1, time: 0, softness: 3, anchors})
 * ```
 * @see scatterField
 */
export function scatteredAnchors(params: GpuFragmentParams, slots: {count: PropRef; seed: PropRef; drift: PropRef; field: string}): Expr {
    // Registers a per-frame writer for the pixel-invariant half of the field (8 anchor positions
    // from count/seed/drift/aspect/animated time) and returns the field's GPU accessor. Reads the
    // node's live `_animTime` (the same accumulator the GPU warp reads), so anchors and warp stay
    // on one clock. Dirty-keyed: a static composition (speed 0, no edits) writes nothing. In a
    // GPU-free composition (tests) `_animTime` is unavailable and the field keeps its initial value.
    let aspect = params.dimensions.width / Math.max(params.dimensions.height, 1e-6)
    params.onResize(({width, height}) => { aspect = width / Math.max(height, 1e-6) })
    let lastKey = ''
    params.onBeforeRender(() => {
        const t = params.getCpuValue('_animTime')
        const count = params.getCpuValue(slots.count.name)
        const seed = params.getCpuValue(slots.seed.name)
        const drift = params.getCpuValue(slots.drift.name)
        if (typeof t !== 'number' || typeof count !== 'number' || typeof seed !== 'number' || typeof drift !== 'number') return
        const key = `${count}|${seed}|${drift}|${aspect}|${t}`
        if (key === lastKey) return
        lastKey = key
        params.setExtraField(slots.field, packScatterAnchors(count, seed, drift, aspect, t))
    })
    return params.uniforms[slots.field]
}

/**
 * Runs a palette value back and forth `cycles` times so a ramp repeats without a seam.
 *
 * At `cycles` 1 the value passes through unchanged. Where the bands get thinner than a pixel
 * they fade to the palette's middle instead of flickering. Use it on the value before the
 * palette lookup, in a `paint:` function.
 *
 * @example
 * ```ts
 * return standardPalette(wrapRamp({t: field, cycles: math.add(1, math.mul(uniformOf(p('wrapping'), params), 6))}), params)
 * ```
 * @tip Not for use inside a `guarded` branch. Reach for `foldRamp` there.
 * @see foldRamp, scatterField
 */
export function wrapRamp(opts: {t: Expr; cycles: Expr | number}): Expr {
    // Triangle wave with a sub-pixel fade to 0.5. Fragment-only (fwidth).
    return call(gradientPaints.rampWrap, 'rampWrap', [opts.t, typeof opts.cycles === 'number' ? floatE(opts.cycles) : opts.cycles])
}

/**
 * Folds any number into 0–1 as a seamless zigzag: up from 0 to 1, back down, every 2 units.
 *
 * Scale the value first to set the cycle length. Safe anywhere, including inside a `guarded`
 * branch.
 *
 * @example
 * ```ts
 * const rgb = stops(p('colorSpace'))(foldRamp(phase), params).member('rgb')
 * ```
 * @see wrapRamp
 */
export function foldRamp(t: Expr): Expr {
    // The derivative-free cousin of `wrapRamp`: no sub-pixel fade, so it is legal in non-uniform
    // control flow where `fwidth` is not.
    return call(gradientPaints.gradientEdgeMirror, 'gradientEdgeMirror', [t])
}

/**
 * A 3D position pushed around by moving noise, for reading several fields at one warped spot.
 *
 * `at` is the 3D point to warp, `time` drives the motion, `amount` how far it is pushed.
 * `levels` 2 (default) is the full warp, 1 is a cheaper single pass for low-power devices.
 * Use `warped` instead when one field is enough.
 *
 * @example
 * ```ts
 * const q = warpedPoint({at: math.vec3(x, y, seed), time: math.mul(t, 0.3), amount: 0.5, levels: isMobileGpuViewport() ? 1 : 2})
 * ```
 * @see warped, warpStep
 */
export function warpedPoint(opts: {at: Expr; time: Expr | number; amount: Expr | number; levels?: 1 | 2}): Expr {
    // Two Inigo-Quilez warp levels over MaterialX noise; same memoized 'warpDomain' instance as `warped`.
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    if (opts.levels === 1) {
        // The mobile tier: one warp level (2 noise reads instead of 4), same offsets and look data.
        const warp1 = fieldsKit.domainWarp1(noiseKit.mxNoiseFloat3, {name: 'warpDomain1'})
        return call(warp1, 'warpDomain1', [opts.at, e(opts.time), e(opts.amount)])
    }
    const warpFn = fieldsKit.domainWarp2(noiseKit.mxNoiseFloat3, {name: 'warpDomain'})
    return call(warpFn, 'warpDomain', [opts.at, e(opts.time), e(opts.amount)])
}

/**
 * One step of a chained warp: pushes a 2D position by two noise reads and hands them back.
 *
 * Returns a vec4: `xy` is the warped position, `zw` the two noise values, useful for shading
 * later. Chain steps by feeding one step's `xy` into the next. `scale` sets the noise
 * frequency, `z` is a time or seed slice, `offsets` two fixed 2D offsets that keep the reads
 * apart, and the push distance is `reach` times `strength`.
 *
 * @example
 * ```ts
 * const level1 = warpStep({at: uv, scale: 1.2, z: math.mul(t, 0.5), offsets: [[5.2, 1.3], [1.7, 9.2]], reach: 0.4, strength: uniformOf(p('warp'), params)})
 * const level2 = warpStep({at: level1.member('xy'), scale: 2.4, z: t, offsets: [[8.3, 2.8], [2.1, 6.4]], reach: 0.25, strength: uniformOf(p('warp'), params)})
 * ```
 * @see warpedPoint, quadBlend
 */
export function warpStep(opts: {
    at: Expr
    scale: number
    z: Expr | number
    offsets: [[number, number], [number, number]]
    reach: number
    strength: Expr | number
}): Expr {
    // One chained Inigo-Quilez level (displaces the CHAINED position, unlike domainWarp2 which
    // offsets the original); `offsets` are the caller's look data.
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    return call(fieldsKit.chainedWarpStep, 'chainedWarpStep', [
        opts.at, floatE(opts.scale), e(opts.z),
        vec2(opts.offsets[0][0], opts.offsets[0][1]),
        vec2(opts.offsets[1][0], opts.offsets[1][1]),
        floatE(opts.reach), e(opts.strength),
    ])
}

/**
 * The color for a 0–1 value from the standard palette props: `colorA`, `colorB`, `stops`, `colorSpace`.
 *
 * Declare those four props (`stops: colorStopsPropConfig()`, `colorSpace` marked `compileTime`).
 * It has the shape of a `Palette`, so it drops straight into `rampOver`, or call it yourself
 * inside a `paint:` function.
 *
 * @example
 * ```ts
 * paint: rampOver(scaledVolume(noiseField('mx3'), {scale: p('scale')}), standardPalette)
 * ```
 * @see stops, pair
 */
export function standardPalette(t: Expr, params: GpuFragmentParams): Expr {
    // The coordinate is consumed once — the stops loop takes it as a fn parameter.
    return noiseColor.mixStopsOrColorsExpr(params, t)
}

/**
 * A four-color blend on a square: `a` to `b` across the top, `c` to `d` across the bottom, then top to bottom.
 *
 * Returns a function you call with two 0–1 values (across, then down) and the params. Mixes
 * in the color space named by the `space` prop (mark it `compileTime`). For any space other
 * than linear, declare `extraFields` named `convA`, `convB`, `convC`, `convD`, each
 * `{schema: schema.vec3f, initial: [0, 0, 0]}`.
 *
 * @example
 * ```ts
 * const base = quadBlend({a: p('colorA'), b: p('colorB'), c: p('colorC'), d: p('colorD'), space: p('colorSpace')})(t1, t2, params)
 * ```
 * @see colorLadder3, pair
 */
export function quadBlend(slots: {a: PropRef; b: PropRef; c: PropRef; d: PropRef; space: PropRef}) {
    // For a non-linear space the four endpoint conversions are pixel-invariant, so a per-frame
    // driver preconverts them (dirty-keyed) into the convA–convD vec3 extraFields and the GPU mixes
    // the preconverted values with a single back-conversion.
    return (t1: Expr, t2: Expr, params: GpuFragmentParams): Expr => {
        const {uniforms} = params
        const colorSpaceMode = colorSpaceModeOf(params.propValues[slots.space.name])
        const colorA = uniformOf(slots.a, params)
        const colorB = uniformOf(slots.b, params)
        const colorC = uniformOf(slots.c, params)
        const colorD = uniformOf(slots.d, params)
        if (colorSpaceMode !== 0) {
            let lastKey = ''
            params.onBeforeRender(() => {
                const cols = ([slots.a.name, slots.b.name, slots.c.name, slots.d.name]).map(
                    (prop) => params.getCpuValue(prop) as {x: number; y: number; z: number} | undefined,
                )
                if (cols.some((c) => !c)) return
                const key = cols.map((c) => `${c!.x},${c!.y},${c!.z}`).join('|')
                if (key === lastKey) return
                lastKey = key
                const fields = ['convA', 'convB', 'convC', 'convD'] as const
                cols.forEach((c, i) => {
                    const conv = colorMixing.convertP3ToMixSpaceCPU(c!.x, c!.y, c!.z, colorSpaceMode)
                    params.setExtraField(fields[i], [conv[0], conv[1], conv[2]])
                })
            })
            const row1 = asLocal(call(colorMixing.mixPreconvertedInSpace, 'mixInSpace', [
                uniforms.convA, colorA.member('a'), uniforms.convB, colorB.member('a'), t1,
            ]), 'flowRow1')
            const row2 = asLocal(call(colorMixing.mixPreconvertedInSpace, 'mixInSpace', [
                uniforms.convC, colorC.member('a'), uniforms.convD, colorD.member('a'), t1,
            ]), 'flowRow2')
            const variant = colorMixing.mixPreconvertedVariants[colorSpaceMode as keyof typeof colorMixing.mixPreconvertedVariants] ?? colorMixing.mixPreconvertedLinear
            return call(variant, 'mixPreconvertedColors', [
                row1.member('conv'), row2.member('conv'), row1.member('alpha'), row2.member('alpha'), t2,
            ])
        }
        const row1 = asLocal(call(colorMixing.mixColorsLinear, 'mixColors', [colorA, colorB, t1]), 'flowRow1')
        const row2 = asLocal(call(colorMixing.mixColorsLinear, 'mixColors', [colorC, colorD, t1]), 'flowRow2')
        return call(colorMixing.mixColorsLinear, 'mixColors', [row1, row2, t2])
    }
}
