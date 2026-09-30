/**
 * std/frames — coordinate frames.
 *
 * A frame is the coordinate system a generator measures its pixels in. Instead of raw uv you
 * get named numbers ready to use: the distance from a centre, how far along and across a line
 * between two points, or the axes of a chosen direction. Start a `paint:` with `surfaceOf`,
 * feed it to a frame built from your position props, and shape the result with math and
 * `paint` words.
 */
// Maintainer notes. Every frame binds its outputs to WGSL locals so downstream algebra reads
// one evaluation. Frames own the house conventions: the `uvContext ?? ctx.uv` generator idiom,
// aspect correction (x scaled by aspect, the D-1 centre-scaled rule), and the transformPosition
// storage convention (positions arrive as `(x, 1−y)`; frames recover the authored y).
import type {Expr, GpuFragmentParams} from '../gpu/contract'
import type {PropRef} from './values'
import {uniformOf} from './invoke'
import {add, atan2, cos, div, dot, fract, local, max, min, mul, neg, sin, sqrt, sub, vec2, length as vlen} from './math'
import {constants} from '../gpu/kit/index'

const DEG_TO_RAD = constants.DEG_TO_RAD

/**
 * The canvas a generator paints on: its `uv`, its `viewport` size in pixels and its `aspect`.
 *
 * `uv` runs 0–1 across the layer, y down. `aspect` is width over height. When a parent fits
 * the layer into a box, `viewport` and `aspect` describe that box. Every frame takes this as
 * its second argument.
 *
 * @example
 * ```ts
 * paint: (params) => { const {uv, viewport, aspect} = surfaceOf(params); … }
 * ```
 * @see rawSurfaceOf, centredFrame, segmentFrame
 */
export function surfaceOf(params: GpuFragmentParams): {uv: Expr; viewport: Expr; aspect: Expr} {
    // The generator idiom: a UV-propagating parent supplies `uvContext`; the fitted box, if any,
    // arrives as `effectiveViewportSize`.
    const uv = params.uvContext ?? params.ctx.uv
    const viewport = params.effectiveViewportSize ?? params.ctx.viewportSize
    const aspect = local(div(viewport.member('x'), viewport.member('y')), 'aspect')
    return {uv, viewport, aspect}
}

/**
 * The canvas like `surfaceOf`, but always measured against the whole canvas, never a fitted box.
 *
 * Use it when a pattern should stay anchored to the canvas as the layer is resized.
 *
 * @example
 * ```ts
 * const {delta, dist} = centredFrame({center: p('center')})(params, rawSurfaceOf(params))
 * ```
 * @see surfaceOf
 */
export function rawSurfaceOf(params: GpuFragmentParams): {uv: Expr; viewport: Expr; aspect: Expr} {
    // Blob's organic disc anchors to the raw canvas rather than a resize-fit box. Same
    // `uvContext ?? ctx.uv` idiom as `surfaceOf`; only the viewport differs.
    const uv = params.uvContext ?? params.ctx.uv
    const viewport = params.ctx.viewportSize
    const aspect = local(div(viewport.member('x'), viewport.member('y')), 'aspect')
    return {uv, viewport, aspect}
}

/** What `centredFrame` gives you for each pixel. */
export interface CentredFrame {
    /** The vector from the centre to the pixel, aspect-corrected (in canvas-height units). */
    delta: Expr
    /** The distance from the centre, in canvas-height units. */
    dist: Expr
}

/**
 * A frame centred on a position prop: each pixel's offset from the centre and its distance.
 *
 * Pass `{center: p('center')}` where `center` is a position prop. Call the result with
 * `params` and a surface. Distances are aspect-corrected, so 1 is the canvas height and a
 * circle stays round on a wide canvas.
 *
 * @example
 * ```ts
 * const {dist} = centredFrame({center: p('center')})(params, surfaceOf(params))
 * ```
 * @tip `dist` is in canvas-height units: a radius of 0.5 reaches the top and bottom edges from the middle.
 * @see segmentFrame, surfaceOf, rawSurfaceOf
 */
export function centredFrame(slots: {center: PropRef}) {
    // The centre is a transformPosition value (stored `(x, 1−y)`); the centre is scaled by the
    // aspect, not the UV alone (the D-1 centre-scaled rule), so it lands where the handle sits.
    return (params: GpuFragmentParams, surface: {uv: Expr; aspect: Expr}): CentredFrame => {
        const center = uniformOf(slots.center, params)
        const {uv, aspect} = surface
        const delta = local(vec2(
            sub(mul(uv.member('x'), aspect), mul(center.member('x'), aspect)),
            sub(uv.member('y'), sub(1, center.member('y'))),
        ), 'centred')
        const dist = local(vlen(delta), 'centredDist')
        return {delta, dist}
    }
}

/** What `segmentFrame` gives you for each pixel. */
export interface SegmentFrame {
    /** How far the pixel is along the line from `from` toward `to` (0 at `from`), in canvas-height units. */
    along: Expr
    /** The signed distance from the line: positive below a left-to-right line, negative above. */
    across: Expr
    /** The length of the line from `from` to `to`, in canvas-height units (never less than 0.0001). */
    length: Expr
}

/**
 * A frame along the line between two position props: how far each pixel is along it and across it.
 *
 * Pass `{from: p('from'), to: p('to')}` and call the result with `params` and a surface.
 * `along` is 0 at `from` and equals `length` at `to`. `across` is the signed distance from the
 * line. All three are aspect-corrected, in canvas-height units.
 *
 * @example
 * ```ts
 * const {along, across, length} = segmentFrame({from: p('from'), to: p('to')})(params, surfaceOf(params))
 * ```
 * @tip Divide `along` by `length` for a 0–1 position along the line, the natural gradient parameter.
 * @see centredFrame, directionFrame, surfaceOf
 */
export function segmentFrame(slots: {from: PropRef; to: PropRef}) {
    // Positions are transformPosition values (stored `(x, 1−y)`); both endpoints are recovered
    // and aspect-corrected before the projection. `across` is the 2D cross product with the
    // unit direction, so its sign follows the screen's y-down handedness.
    return (params: GpuFragmentParams, surface: {uv: Expr; aspect: Expr}): SegmentFrame => {
        const from = uniformOf(slots.from, params)
        const to = uniformOf(slots.to, params)
        const {uv, aspect} = surface

        const cx = mul(from.member('x'), aspect)
        const cy = sub(1, from.member('y'))
        const rel = local(vec2(sub(mul(uv.member('x'), aspect), cx), sub(uv.member('y'), cy)), 'rel')
        const toRel = local(
            vec2(sub(mul(to.member('x'), aspect), cx), sub(sub(1, to.member('y')), cy)),
            'toRel',
        )
        const len = local(max(vlen(toRel), 0.0001), 'segLen')
        const dir = local(div(toRel, len), 'segDir')

        const along = local(dot(rel, dir), 'along')
        const across = local(
            add(mul(rel.member('x'), neg(dir.member('y'))), mul(rel.member('y'), dir.member('x'))),
            'across',
        )
        return {along, across, length: len}
    }
}

/**
 * A direction in degrees as a unit vector.
 *
 * 0° points right and angles turn clockwise on screen, so 90° points down. Use it wherever a
 * word wants a light or flow direction as a vector.
 *
 * @example
 * ```ts
 * const light = direction(params.uniforms.lightAngle, 'light')
 * ```
 * @see directionFrame
 */
export function direction(angleDeg: Expr | number, hint = 'dir'): Expr {
    // `(cos θ, sin θ)` bound to one local; clockwise because screen y grows downward.
    const rad = mul(angleDeg, DEG_TO_RAD)
    return local(vec2(cos(rad), sin(rad)), hint)
}

/** What `directionFrame` gives you. */
export interface DirectionFrame {
    /** The unit vector pointing along the direction. */
    tangent: Expr
    /** The unit vector at right angles to it. */
    perp: Expr
    /** Project a 2D point into the frame: its distance `along` the direction and `across` it. */
    coordsOf(p: Expr): {along: Expr; across: Expr}
}

/**
 * The two axes of a direction given in degrees, for patterns that run along it.
 *
 * `tangent` points along the direction, `perp` at right angles to it. `coordsOf(point)`
 * returns that point's `along` and `across` distances in the frame. Brush grain, weave,
 * streaks and flow all live in one of these.
 *
 * @example
 * ```ts
 * const grain = directionFrame(params.uniforms.brushAngle, 'brush'); const {along, across} = grain.coordsOf(delta)
 * ```
 * @see direction, segmentFrame
 */
export function directionFrame(angleDeg: Expr | number, hint = 'grain'): DirectionFrame {
    // `perp` is the tangent turned by +90° in the same clockwise-on-screen sense as `direction`.
    const tangent = direction(angleDeg, `${hint}T`)
    const perp = local(vec2(neg(tangent.member('y')), tangent.member('x')), `${hint}P`)
    return {
        tangent,
        perp,
        coordsOf(p: Expr) {
            return {
                along: local(dot(p, tangent), 'along'),
                across: local(dot(p, perp), 'across'),
            }
        },
    }
}

/**
 * Fold a point into one mirrored wedge around a centre: the kaleidoscope fold.
 *
 * `sectorAngle` is the wedge's angle in radians (2π divided by the number of facets). `center`
 * is where the wedges meet, 0.5 by default. Returns the folded point as an offset from the
 * centre. Sample anything at that point and it repeats and mirrors around the centre.
 *
 * @example
 * ```ts
 * const folded = sectorFold(uv.member('x'), uv.member('y'), math.div(2 * Math.PI, params.uniforms.facets))
 * ```
 * @see directionFrame, centredFrame
 */
export function sectorFold(x: Expr, y: Expr, sectorAngle: Expr, center = 0.5): Expr {
    // The kaleidoscope domain fold (Crystal): polar around `center`, wrap the angle into one
    // sector, mirror at the sector's half, back to Cartesian.
    const PI = 3.141592653589793
    const dx = local(sub(x, center), 'kfDx')
    const dy = local(sub(y, center), 'kfDy')
    const r = local(sqrt(add(mul(dx, dx), mul(dy, dy))), 'kfR')
    const theta = add(atan2(dy, dx), PI)
    const inSector = local(mul(fract(div(theta, sectorAngle)), sectorAngle), 'kfInSector')
    const folded = local(min(inSector, sub(sectorAngle, inSector)), 'kfFolded')
    return vec2(mul(r, cos(folded)), mul(r, sin(folded)))
}
