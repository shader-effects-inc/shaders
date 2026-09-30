/**
 * std/shape — the 2D shapes.
 *
 * Each word is a shape as a signed distance: pass a point `(x, y)` and the shape's sizes and
 * you get back how far that point is from the shape's edge, negative inside and positive
 * outside. Sizes are in uv, where 1 is the canvas height. A shape becomes a component through
 * the `shape:` field of `defineShader`: its `distance` reads the shape-local point (already
 * centred on the `center` prop, aspect-corrected and rotated, y down) plus `u('prop')` for each
 * size prop, and returns one of these words. The engine adds the fill, stroke, softness and
 * color props, the bounding box and the pixel mask. See the Vesica, Star and Heart shaders for
 * complete examples.
 */
// Maintainer notes. These are `clip-path`-style primitives over the analytic SDF kit. Each
// function takes the shape-local coordinates plus its dimensions (expressions or numbers) and
// returns the signed distance. Anything shape-specific (Circle's radius→edge halving, Arc's
// degrees→radians) is plain algebra at the declaration site, not a variant here. The pixel
// mask and fill/stroke mixing live in the sdfShape scaffold (`strokeMaskFromSdf`).
import type {Expr} from '../gpu/contract'
import {call, floatE} from '../gpu/composer'
import {sdf} from '../gpu/kit/index'

type Val = Expr | number

const toE = (v: Val): Expr => (typeof v === 'number' ? floatE(v) : v)

const field =
    (fn: unknown, hint: string) =>
    (...args: Val[]): Expr =>
        call(fn, hint, args.map(toE))

/**
 * A circle of a given radius.
 *
 * `radius` is in uv (1 = the canvas height). Returns the signed distance to the edge,
 * negative inside.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => circle(x, y, u('radius')), bounds: {size: {width: 'radius', height: 'radius'}}, …}
 * ```
 * @tip Blend a sharp shape toward this one with `math.mix` to round it off (the Polygon shader does).
 * @see ellipse, ring, polygon
 */
export const circle = field(sdf.circleSdf, 'circleSdf') as (x: Val, y: Val, radius: Val) => Expr

/**
 * An ellipse with separate horizontal and vertical radii.
 *
 * `radiusX` and `radiusY` are the half-widths along each axis, in uv.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => ellipse(x, y, u('radiusX'), u('radiusY')), …}
 * ```
 * @see circle, roundedRect
 */
export const ellipse = field(sdf.ellipseSdf, 'ellipseSdf') as (x: Val, y: Val, radiusX: Val, radiusY: Val) => Expr

/**
 * A rectangle with rounded corners.
 *
 * `width` and `height` are half-extents (the distance from the centre to each side), in uv.
 * `rounding` is the corner radius. Pass 0 for a sharp rectangle.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => roundedRect(x, y, u('width'), u('height'), u('rounding')), …}
 * ```
 * @tip A `width` of 0.5 spans the full canvas height, not half of it.
 * @see parallelogram, trapezoid, cross
 */
export const roundedRect = field(sdf.roundedRectSdf, 'roundedRectSdf') as (x: Val, y: Val, width: Val, height: Val, rounding: Val) => Expr

/**
 * A regular polygon with any number of sides.
 *
 * `radius` is the distance from the centre to the middle of each side (not to the corners), in
 * uv. `sides` is the side count, 3 for a triangle and up. A flat side faces right at rotation 0.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => polygon(x, y, u('radius'), u('sides')), …}
 * ```
 * @tip `math.mix(polygon(…), circle(…), u('rounding'))` morphs the corners away.
 * @see star, flower, circle
 */
export const polygon = field(sdf.polygonSdf, 'polygonSdf') as (x: Val, y: Val, radius: Val, sides: Val) => Expr

/**
 * A star with straight sides and sharp points.
 *
 * `outerRadius` is the distance from the centre to the tips, in uv. `sides` is the number of
 * points. `innerRatio` (0–1) is the inner corner radius as a fraction of the tip radius.
 * 0.382 gives the classic five-point star. One tip points right at rotation 0.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => star(x, y, u('radius'), u('sides'), u('innerRatio')), …}
 * ```
 * @see flower, polygon
 */
export const star = field(sdf.starSdf, 'starSdf') as (x: Val, y: Val, outerRadius: Val, sides: Val, innerRatio: Val) => Expr

/**
 * A flower with pointed petals and V-shaped valleys between them.
 *
 * `outerRadius` is the distance from the centre to a petal tip, in uv. `sides` is the petal
 * count. `innerRatio` (0–1) is the valley radius as a fraction of the tip radius.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => flower(x, y, u('radius'), u('sides'), u('innerRatio')), …}
 * ```
 * @see star, polygon
 */
export const flower = field(sdf.flowerSdf, 'flowerSdf') as (x: Val, y: Val, outerRadius: Val, sides: Val, innerRatio: Val) => Expr

/**
 * A heart, point down at rotation 0.
 *
 * `radius` is the size it is fitted into, in uv.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => heart(x, y, u('radius')), bounds: {size: {width: 'radius', height: 'radius'}}, …}
 * ```
 * @see teardrop, circle
 */
export const heart = field(sdf.heartSdf, 'heartSdf') as (x: Val, y: Val, radius: Val) => Expr

/**
 * A plus-shaped cross with four equal arms.
 *
 * `size` is the arm half-length (centre to an arm's end), `thickness` the arm half-width, both
 * in uv. `rounding` rounds the corners. Rotate it 45° for an X.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => cross(x, y, u('radius'), u('thickness'), u('rounding')), …}
 * ```
 * @see roundedRect
 */
export const cross = field(sdf.crossSdf, 'crossSdf') as (x: Val, y: Val, size: Val, thickness: Val, rounding: Val) => Expr

/**
 * A ring: a circular band around a centre.
 *
 * `radius` is the distance to the middle of the band, `thickness` half the band's width, both
 * in uv. The full band is twice `thickness` wide.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => ring(x, y, u('radius'), u('thickness')), rotatable: false, …}
 * ```
 * @tip Rotationally symmetric, so declare `rotatable: false` and the rotation prop is left out.
 * @see circle, arc, crescent
 */
export const ring = field(sdf.ringSdf, 'ringSdf') as (x: Val, y: Val, radius: Val, thickness: Val) => Expr

/**
 * A vesica: the lens where two equal circles overlap.
 *
 * `radius` is each circle's radius, in uv. `spread` (0–1) is how far apart the circles sit as
 * a fraction of the radius. 0 is a full circle, 1 an infinitely thin lens.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => vesica(x, y, u('radius'), u('spread')), …}
 * ```
 * @see ellipse, crescent
 */
export const vesica = field(sdf.vesicaSdf, 'vesicaSdf') as (x: Val, y: Val, radius: Val, spread: Val) => Expr

/**
 * A crescent: a disc with a smaller disc bitten out of it.
 *
 * `outerRadius` is the full disc's radius, in uv. `innerRatio` (0–1) is the bite's radius as a
 * fraction of it. `offset` is how far the bite's centre is shifted sideways, in uv.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => crescent(x, y, u('radius'), u('innerRatio'), u('offset')), …}
 * ```
 * @see ring, vesica
 */
export const crescent = field(sdf.crescentSdf, 'crescentSdf') as (x: Val, y: Val, outerRadius: Val, innerRatio: Val, offset: Val) => Expr

/**
 * A trapezoid: a four-sided shape with a flat top and bottom of different widths.
 *
 * `topWidth` and `bottomWidth` are half-widths of the top and bottom edges as seen on screen,
 * `height` the half-height, all in uv. Equal widths give a rectangle.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => trapezoid(x, y, u('topWidth'), u('bottomWidth'), u('height')), …}
 * ```
 * @see parallelogram, roundedRect
 */
export const trapezoid = field(sdf.trapezoidSdf, 'trapezoidSdf') as (x: Val, y: Val, topWidth: Val, bottomWidth: Val, height: Val) => Expr

/**
 * A teardrop: a round bulb tapering to a sharp point, point up at rotation 0.
 *
 * `radius` is the bulb's radius and `height` the distance from the bulb's centre to the point,
 * both in uv.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => teardrop(x, y, u('radius'), u('height')), …}
 * ```
 * @see heart, ellipse
 */
export const teardrop = field(sdf.teardropSdf, 'teardropSdf') as (x: Val, y: Val, radius: Val, height: Val) => Expr

/**
 * A parallelogram: a rectangle with its top edge slid sideways.
 *
 * `width` and `height` are half-extents, in uv. `skew` is how far the top edge is shifted
 * horizontally, in uv. 0 gives a rectangle.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => parallelogram(x, y, u('width'), u('height'), u('skew')), …}
 * ```
 * @see trapezoid, roundedRect
 */
export const parallelogram = field(sdf.parallelogramSdf, 'parallelogramSdf') as (x: Val, y: Val, width: Val, height: Val, skew: Val) => Expr

/**
 * A pie wedge: a filled sector of a circle.
 *
 * `radius` is the sector's radius, in uv. `halfAngle` is half the opening angle, in radians.
 * Multiply an aperture prop in degrees by `APERTURE_TO_HALF_ANGLE` to get it. The wedge opens
 * downward at rotation 0.
 *
 * @example
 * ```ts
 * shape: {distance: ({x, y}, u) => arc(x, y, u('radius'), math.mul(u('aperture'), APERTURE_TO_HALF_ANGLE)), …}
 * ```
 * @see ring, circle, APERTURE_TO_HALF_ANGLE
 */
export const arc = field(sdf.arcSdf, 'arcSdf') as (x: Val, y: Val, radius: Val, halfAngle: Val) => Expr

/**
 * The factor that turns an opening angle in degrees into the half-angle `arc` takes.
 *
 * Multiply a 0–360 aperture prop by it. It equals π/360.
 *
 * @example
 * ```ts
 * distance: ({x, y}, u) => arc(x, y, u('radius'), math.mul(u('aperture'), APERTURE_TO_HALF_ANGLE))
 * ```
 * @see arc
 */
export const APERTURE_TO_HALF_ANGLE = sdf.APERTURE_TO_HALFANGLE
