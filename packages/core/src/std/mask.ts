/**
 * std/mask — Coverage: a 0..1 number per pixel that says how much of something shows there.
 *
 * Multiply a mask into a color's alpha to cut a shape out of a paint, or hand it to an
 * effect as its amount so the effect fades in across the canvas. Most masks start from a
 * distance: how far this pixel is from a center, a line or a shape's edge. `softDisc` and
 * `softBand` give soft, styled edges. `crispFill` and `crispStroke` give clean shape edges
 * from a signed distance. `radialMask` is the ready-made one for effects.
 */
// Maintainer notes (not part of the reader-facing reference):
// - Mask nouns: scalar coverage fields usable inside pointwise effects without forcing RTT
//   (Vignette is the proof case). GPU bodies live in the kit (`kit/fields.ts`); the Scalar
//   constructors capture data only.
// - The Expr-level coverage words (`softBand`, `softDisc`, `crispFill`, `crispStroke`) live
//   here too — concept-level masks any paint can shape with.
import type {Expr} from '../gpu/contract'
import {abs, add, mul, neg, pow, smoothstep, sub} from './math'
import type {PropRef} from './values'
import {Scalar} from './values'

/**
 * A soft band around a line, sharp on one side and long on the other.
 *
 * `distance` is the signed distance from the line: negative below it, positive above.
 * Coverage rises from 0 to 1 as the distance crosses `below.from` to `below.to`, then falls
 * back to 0 as it crosses `above.from` (default 0) to `above.to`. The shape under curtains,
 * horizons and underlines.
 *
 * @example
 * ```ts
 * const curtain = mask.softBand({distance: sub(uv.member('y'), pathY), below: {from: -0.04, to: 0.015}, above: {to: height}})
 * ```
 * @see softDisc, crispStroke
 */
export function softBand(opts: {
    distance: Expr
    below: {from: Expr | number; to: Expr | number}
    above: {from?: Expr | number; to: Expr | number}
}): Expr {
    return mul(
        smoothstep(opts.below.from, opts.below.to, opts.distance),
        sub(1, smoothstep(opts.above.from ?? 0, opts.above.to, opts.distance)),
    )
}

/**
 * A filled disc with a soft, shaped edge.
 *
 * `dist` is the distance from the disc's center and `radius` its size, both in the same
 * units. The edge fades over `softness.width` either side of the radius, and
 * `softness.curve` bends that fade: 1 is even, higher values pull it tighter toward the
 * rim. The look of blobs, orbs and glow dots.
 *
 * @example
 * ```ts
 * const orb = mask.softDisc({dist, radius: params.uniforms.size, softness: {width: 0.08, curve: 1.5}})
 * ```
 * @see crispFill, softBand
 */
export function softDisc(opts: {
    dist: Expr
    radius: Expr | number
    softness: {width: Expr | number; curve: Expr | number}
}): Expr {
    // A smoothstep band of half-width `softness.width` around `radius`, raised through
    // `softness.curve`. The soft cousin of the kit's crisp `aa.discCoverage` — this one is a
    // LOOK dial, not an anti-aliasing footprint.
    const {width, curve} = opts.softness
    return pow(sub(1, smoothstep(sub(opts.radius, width), add(opts.radius, width), opts.dist)), curve)
}

/**
 * A clean fill of a shape: 1 inside, 0 outside.
 *
 * `distance` is a signed distance, negative inside the shape, such as one from the `shape`
 * words. `footprint` is how wide the edge blends. One pixel in the distance's units gives an
 * anti-aliased edge. A wider value gives a deliberate soft edge.
 *
 * @example
 * ```ts
 * const bar = mask.crispFill({distance: shape.roundedRect(x, y, w, h, rounding), footprint: px})
 * ```
 * @tip One pixel in uv is `div(1, viewport.member('y'))`, with `viewport` from `frames.surfaceOf`.
 * @see crispStroke, softDisc
 */
export function crispFill(opts: {distance: Expr; footprint: Expr | number}): Expr {
    const half = mul(opts.footprint, 0.5)
    return sub(1, smoothstep(neg(half), half, opts.distance))
}

/**
 * A clean outline of a shape, or a drawn curve, of a given width.
 *
 * `distance` is a signed distance to the shape's edge or to the curve. The stroke is
 * centered on the edge and `width` wide, in the distance's units. `footprint` blends the
 * stroke's own edges, as in `crispFill`.
 *
 * @example
 * ```ts
 * const outline = mask.crispStroke({distance: shape.circle(x, y, radius), width: params.uniforms.lineWidth, footprint: px})
 * ```
 * @see crispFill
 */
export function crispStroke(opts: {distance: Expr; width: Expr | number; footprint: Expr | number}): Expr {
    // crispFill on |distance| − width/2.
    return crispFill({distance: sub(abs(opts.distance), mul(opts.width, 0.5)), footprint: opts.footprint})
}

/**
 * An amount that is 0 near a center and grows to 1 with distance, for an effect's strength.
 *
 * Coverage stays 0 inside `radius` and reaches 1 at `radius + falloff`, measured in uv with
 * the aspect corrected so the ring is round. Every slot takes a prop: `center` is a position
 * prop, `radius` and `falloff` are numbers in uv. Chain `.times(p('intensity'))` to scale
 * it.
 *
 * @example
 * ```ts
 * effect: tintToward(p('color'), {amount: radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity'))})
 * ```
 * @tip Also exported at the top level of `shaders/std`, so `radialMask` works without the `mask.` prefix.
 * @see softDisc
 */
export function radialMask(slots: {center: PropRef; radius: PropRef; falloff: PropRef}): Scalar {
    // `center` binds a position prop (transformPosition-stored). Body: `fields.radialFalloffMask`.
    return new Scalar({kind: 'radialMask', center: slots.center, radius: slots.radius, falloff: slots.falloff})
}
