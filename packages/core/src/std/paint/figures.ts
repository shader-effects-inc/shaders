/**
 * std/paint/figures — drawn strokes: a straight segment and a sine wave.
 *
 * Each word binds your props and returns something a generator's `paint:` field takes. The
 * stroke is drawn in one color prop with an antialiased edge, transparent everywhere else,
 * so it layers over whatever sits behind it.
 */
// Maintainer notes (not part of the reference):
//  - Built from the kit's figure/stroke bodies (`kit/shapePaints.ts`).
//  - Only genuinely generic drawing figures live here (a stroked segment, a sine band). LOOKS do
//    not: Blob's recipe lives in its shader file, composed from std words (std is how we draw,
//    the shader file is the stylesheet).
//  - The UV idiom — standalone renders read `ctx.uv`; wrapped by a UV-propagating parent the
//    composed `uvContext` wins; strokes use the effective viewport (resize-fit box).
//  - The single-color strokes end in the shared color tail `vec4(color.rgb, color.a * mask)`.
//  - Animated words read the per-node accumulated time themselves; the `animatedTime:`
//    declaration stays on the definition.
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, vec4, floatE, animatedTime} from '../../gpu/porters'
import {shapePaints} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {uniformOf, resolveScalar, paintFrame, type ArgSpec} from '../invoke'

/** What a generator's `paint:` field takes: a function that returns the color at this pixel. */
export type Paint = (params: GpuFragmentParams) => Expr

// ── Shared resolution helpers ───────────────────────────────────────────────────────────

/** Resolve a scalar-ish slot (prop, scalar graph, ctx token, or number literal) to an Expr. */
function arg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

/** The single-color stroke tail: the stroke color with alpha scaled by the coverage mask. */
function maskedColor(color: PropRef, mask: Expr, params: GpuFragmentParams): Expr {
    const c = uniformOf(color, params)
    return vec4(c.member('rgb'), c.member('a').mul(mask))
}

// ── Stroked segment ─────────────────────────────────────────────────────────────────────

/** The inputs to `strokedSegment`. */
export interface StrokedSegmentSlots {
    /** The two endpoints, position props made with `transformPosition`. */
    from: PropRef
    to: PropRef
    /** Stroke thickness as a fraction of the canvas height (0.01 is a hairline). */
    width: ArgSpec
    /** Line style as a number: 0 solid, 1 dashed, 2 dotted. */
    style: ArgSpec
    /** Dash length (caps included) and the gap between dashes or dots, as fractions of the canvas height. */
    dashLength: ArgSpec
    gapLength: ArgSpec
    /** Cap shape at each end as a number: 0 square, 1 rounded. */
    capStart: ArgSpec
    capEnd: ArgSpec
    /** The stroke color prop. */
    color: PropRef
}

/**
 * A straight line between two points, solid, dashed or dotted, with square or round ends.
 *
 * Dashes and dots are spaced so the pattern lands exactly on both endpoints. The edge stays
 * antialiased at any thickness. Style and cap inputs are numbers, so give a select prop a
 * `transform` that maps its labels to them, or pass the number directly.
 *
 * @example
 * ```ts
 * paint: strokedSegment({from: p('pointA'), to: p('pointB'), width: p('thickness'), style: 1, dashLength: p('dashLength'), gapLength: p('gapLength'), capStart: 1, capEnd: 1, color: p('color')})
 * ```
 * @see sineStroke
 */
export function strokedSegment(slots: StrokedSegmentSlots): Paint {
    // Coincident endpoints render a single cap-shaped point.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const mask = call(shapePaints.lineMask, 'lineMask', [
            uv,
            viewport,
            uniformOf(slots.from, params),
            uniformOf(slots.to, params),
            arg(slots.width, params),
            arg(slots.style, params),
            arg(slots.dashLength, params),
            arg(slots.gapLength, params),
            arg(slots.capStart, params),
            arg(slots.capEnd, params),
        ])
        return maskedColor(slots.color, mask, params)
    }
}

// ── Sine stroke ─────────────────────────────────────────────────────────────────────────

/** The inputs to `sineStroke`. */
export interface SineStrokeSlots {
    /** The wave's center, a position prop made with `transformPosition`, and its rotation in degrees. */
    position: PropRef
    angle: ArgSpec
    /** Waves per canvas height along the line, and wave height as a fraction of the canvas height. */
    frequency: ArgSpec
    amplitude: ArgSpec
    /** Band thickness (0–2) and edge softness (0–1). */
    thickness: ArgSpec
    softness: ArgSpec
    /** The stroke color prop. */
    color: PropRef
}

/**
 * A wavy line that travels along itself, drawn as a soft band.
 *
 * The wave is centered on `position` and turned by `angle`. It moves on the layer's clock,
 * so declare `animatedTime: {speed: 'speed'}`.
 *
 * @example
 * ```ts
 * paint: sineStroke({position: p('position'), angle: p('angle'), frequency: p('frequency'), amplitude: p('amplitude'), thickness: p('thickness'), softness: p('softness'), color: p('color')})
 * ```
 * @see strokedSegment
 */
export function sineStroke(slots: SineStrokeSlots): Paint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const mask = call(shapePaints.sineWaveMask, 'sineWaveMask', [
            uv,
            viewport,
            arg(slots.angle, params),
            uniformOf(slots.position, params),
            arg(slots.frequency, params),
            arg(slots.amplitude, params),
            arg(slots.thickness, params),
            arg(slots.softness, params),
            animatedTime(params),
        ])
        return maskedColor(slots.color, mask, params)
    }
}
