/**
 * 2D signed-distance-field primitives (`'use gpu'` TGSL). Every function takes UV-space offsets
 * `(dx, dy)` from the shape center plus the shape's params and returns a signed distance: negative
 * inside, positive outside, zero on the boundary. The caller applies rotation externally (see
 * `shapeLocalCoords`).
 *
 * Each primitive is an exported DualFn so the golden tests can CPU-evaluate it. The shape shaders
 * (Circle/Ellipse/Cross/Heart, Arc/Star/…) call these — plus the shared `shapeLocalCoords` +
 * `strokeMaskFromSdf` shape helpers below — from their `'use gpu'` fragment bodies.
 *
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 * NOT HERE: the SVG-SDF *sampler* wiring (`createSvgSdfSampler` /
 * `createSvgSdfSamplerWithGradients` / `createSdfDataTexture`) and the CPU-resolved shape sub-prop
 * drivers are consumed only by the shape-effect shaders (Glass, Crystal, LiquidMetal, Frost, …)
 * and depend on `kit/sdf3d.ts` and the contract hook for "a shader creates + samples a DataTexture
 * as a KitTexture" (the fn-arg path). `getSdfContentBounds` lives in `utilities/sdfBounds.ts`; this
 * module only shares `SVG_SDF_SIZE` + `loadSdfFromUrl` with it / the SVG path.
 */
import {tgpu, d, std} from './index'
import {toHalfFloat} from './dataEncoding'
import {debugWarn} from '../support'
import {mixColorsVariants, mixColorsLinear} from './colorMixing'
import {call, floatE, vec4} from '../composer'
import type {Expr, GpuFragmentParams, KitTexture, GpuMediaTexture} from '../contract'
import type {SurfacePatternMode} from './sdf3d'
import {generateSvgSdfField, type SvgSdfField} from './svgSdfGenerate'
import {isMobileGpuViewport} from '../../utilities/device'

// Must match generateSdf.ts OUTPUT_SIZE (baked SVG SDF field resolution).
export const SVG_SDF_SIZE = 512

// Math constants as plain number literals — the pattern for values referenced INSIDE a
// `'use gpu'` body (kit/colorMixing's PI_OVER_3 etc.). A `Math.PI` member access inside a body is
// NOT proven to fold, so these are spelled out.
const PI = 3.141592653589793 //  Math.PI
const TWO_PI = 6.283185307179586 //  2 * Math.PI
const SQRT2_OVER_4 = 0.35355339059327373 //  Math.SQRT2 / 4
const DEG_TO_RAD = 0.017453292519943295 //  Math.PI / 180

// ─── Analytic 2D SDF primitives ───────────────────────────────────────────────
// Sign convention: negative = inside, positive = outside, zero = boundary.

/** Circle SDF. `radius` = circle radius in UV space. */
export const circleSdf = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((dx, dy, radius) => {
    'use gpu'
    return std.sqrt(dx * dx + dy * dy) - radius
})

/**
 * Regular polygon SDF. `radius` = inradius (center → nearest edge midpoint), `sides` = side
 * count. Nearest-sector half-plane formula: exact at edge midpoints and vertices for any convex
 * polygon. Caller applies rotation externally.
 */
export const polygonSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, radius, sides) => {
    'use gpu'
    const len = std.sqrt(dx * dx + dy * dy)
    const angle = std.atan2(dy, dx) // atan2(y, x)
    const sectorAngle = TWO_PI / sides
    const sectorIdx = std.floor(angle / sectorAngle + 0.5) // nearest sector
    const canonicalAngle = angle - sectorIdx * sectorAngle // angle within sector
    return len * std.cos(canonicalAngle) - radius
})

/**
 * Flower SDF (N-petalled). `outerRadius` = center→petal-tip, `sides` = petals, `innerRatio` =
 * valley radius / outer radius. Triangular radial wave → distinct tips, clean V-shaped valleys.
 * Caller applies rotation externally.
 */
export const flowerSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, outerRadius, sides, innerRatio) => {
    'use gpu'
    const angle = std.atan2(dy, dx)
    const len = std.sqrt(dx * dx + dy * dy)
    const innerRadius = outerRadius * innerRatio
    // Normalize angle: increments by 1 per sector of 2π/n.
    const tAngle = (angle * sides) / TWO_PI
    // Triangular wave: 1 at tips (integer tAngle), 0 at valleys (half-integer tAngle).
    const tFrac = tAngle - std.floor(tAngle)
    const t = std.abs(tFrac * 2.0 - 1.0)
    const boundaryR = innerRadius + (outerRadius - innerRadius) * t
    return len - boundaryR
})

/**
 * Star SDF (N-pointed) — exact segment-based star polygon. `outerRadius` = center→tip, `sides` =
 * points, `innerRatio` = inner vertex radius / outer radius. True Euclidean SDF: flat sides,
 * sharp corners. Caller applies rotation externally.
 */
export const starSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, outerRadius, sides, innerRatio) => {
    'use gpu'
    const innerRadius = outerRadius * innerRatio
    const len = std.sqrt(dx * dx + dy * dy)
    const angle = std.atan2(dy, dx)

    // Fold angle into the canonical sector centered at the nearest tip.
    const sectorAngle = TWO_PI / sides
    const sectorIdx = std.floor(angle / sectorAngle + 0.5)
    const bn = angle - sectorIdx * sectorAngle // bn ∈ [-π/n, π/n]

    // Cartesian coords in folded sector: tip points in +py direction.
    const fpx = std.abs(len * std.sin(bn)) // fold to right half-sector
    const fpy = len * std.cos(bn)

    // Sector half-angle an = π/n.
    const an = PI / sides
    const sinAn = std.sin(an)
    const cosAn = std.cos(an)

    // Boundary segment: outer tip O=(0, outerRadius) → inner vertex I=(innerRadius·sinAn, innerRadius·cosAn).
    const ex = innerRadius * sinAn // I.x
    const ey = innerRadius * cosAn - outerRadius // I.y - O.y (edge direction)

    // Point relative to the outer tip.
    const qx = fpx
    const qy = fpy - outerRadius

    // Project onto segment, clamp to [0, 1].
    const dotQE = qx * ex + qy * ey
    const dotEE = ex * ex + ey * ey
    const t = std.clamp(dotQE / dotEE, 0.0, 1.0)

    // Distance to the nearest point on the segment.
    const nx = fpx - ex * t
    const ny = fpy - (outerRadius + ey * t)
    const dist = std.sqrt(nx * nx + ny * ny)

    // Sign: cross(edge, q) — negative inside the star, positive outside.
    const cross = ex * qy - ey * qx
    return dist * std.sign(cross)
})

/**
 * Ring (annulus) SDF. `radius` = center→ring centerline, `thickness` = ring half-width. Inside =
 * the ring material. Rotationally symmetric (no rotation needed).
 */
export const ringSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, radius, thickness) => {
    'use gpu'
    const len = std.sqrt(dx * dx + dy * dy)
    return std.abs(len - radius) - thickness
})

/**
 * Cross (plus-sign) SDF — exact SDF of a horizontal ∪ vertical rectangle. `size` = arm
 * half-length, `thickness` = arm half-width, `rounding` = corner rounding offset. Caller applies
 * rotation externally.
 */
export const crossSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, size, thickness, rounding) => {
    'use gpu'
    const px = std.abs(dx)
    const py = std.abs(dy)
    const dHoriz = std.max(px - size, py - thickness)
    const dVert = std.max(py - size, px - thickness)
    return std.min(dHoriz, dVert) - rounding
})

/**
 * Rounded Rectangle SDF — exact SDF of a rectangle with uniformly rounded corners. `width`/
 * `height` = half-extents, `rounding` = corner radius. Caller applies rotation externally.
 */
export const roundedRectSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, width, height, rounding) => {
    'use gpu'
    const qx = std.abs(dx) - width + rounding
    const qy = std.abs(dy) - height + rounding
    const mqx = std.max(qx, 0.0)
    const mqy = std.max(qy, 0.0)
    const outerDist = std.sqrt(mqx * mqx + mqy * mqy)
    const innerDist = std.min(std.max(qx, qy), 0.0)
    return outerDist + innerDist - rounding
})

/**
 * Ellipse SDF (scaled-circle approximation — exact at the boundary). `radiusX`/`radiusY` =
 * semi-axes. The `min(a,b)` correction keeps |gradient| ≈ 1 near the edge (where smoothstep /
 * Glass refraction care about accuracy). Caller applies rotation externally.
 */
export const ellipseSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, radiusX, radiusY) => {
    'use gpu'
    const nx = dx / radiusX
    const ny = dy / radiusY
    const k = std.sqrt(nx * nx + ny * ny)
    return (k - 1.0) * std.min(radiusX, radiusY)
})

/**
 * Vesica (lens) SDF — intersection of two overlapping circles. `radius` = circle radius,
 * `spread` = half-distance between centers as a fraction of radius (0 → single circle, 1 →
 * infinitely thin lens). Caller applies rotation externally.
 */
export const vesicaSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, radius, spread) => {
    'use gpu'
    const px = std.abs(dx)
    const py = std.abs(dy)
    const dd = radius * spread
    const b = std.sqrt(std.max(radius * radius - dd * dd, 0.0))
    const cond = (py - b) * dd > px * b
    const dist1 = std.sqrt(px * px + (py - b) * (py - b))
    const dist2 = std.sqrt((px + dd) * (px + dd) + py * py) - radius
    return std.select(dist2, dist1, cond)
})

/**
 * Crescent (moon) SDF — outer circle with an inner circle subtracted (iq's moon SDF).
 * `outerRadius` = outer circle radius, `innerRatio` = inner radius / outer radius, `offset` =
 * horizontal distance between the two centers. Caller applies rotation externally.
 */
export const crescentSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, outerRadius, innerRatio, offset) => {
    'use gpu'
    const py = std.abs(dy)
    const ra = outerRadius
    const rb = outerRadius * innerRatio
    const off = offset

    const a = (ra * ra - rb * rb + off * off) / std.max(off * 2.0, 0.001)
    const bSq = std.max(ra * ra - a * a, 0.0)
    const b = std.sqrt(bSq)

    const lhs = off * (dx * b - py * a)
    const rhs = off * off * std.max(b - py, 0.0)

    const d1x = dx - a
    const d1y = py - b
    const dist1 = std.sqrt(d1x * d1x + d1y * d1y)

    const lenP = std.sqrt(dx * dx + py * py)
    const innerX = dx - off
    const lenInner = std.sqrt(innerX * innerX + py * py)
    const dist2 = std.max(lenP - ra, (lenInner - rb) * -1.0)

    return std.select(dist2, dist1, lhs > rhs)
})

/**
 * Trapezoid SDF — exact SDF of an axis-aligned trapezoid centered at the origin. `r1` = bottom
 * half-width, `r2` = top half-width, `he` = half-height. Caller applies rotation externally.
 */
export const trapezoidSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, r1, r2, he) => {
    'use gpu'
    const px = std.abs(dx)

    const k1x = r2
    const k1y = he
    const k2x = r2 - r1
    const k2y = he * 2.0

    const clampWidth = std.select(r2, r1, dy < 0.0)
    const cax = px - std.min(px, clampWidth)
    const cay = std.abs(dy) - he

    const diffX = k1x - px
    const diffY = k1y - dy
    const dotNum = diffX * k2x + diffY * k2y
    const dotDen = std.max(k2x * k2x + k2y * k2y, 0.0001)
    const t = std.clamp(dotNum / dotDen, 0.0, 1.0)
    const cbx = px - k1x + k2x * t
    const cby = dy - k1y + k2y * t

    // inside = (cbx < 0) AND (cay < 0). Expressed without `&&` (unproven in TGSL bodies) as a
    // nested select — identical result, since the inner test is only reached when cbx < 0.
    const sInner = std.select(d.f32(1), d.f32(-1), cay < 0.0)
    const s = std.select(d.f32(1), sInner, cbx < 0.0)

    const dca = cax * cax + cay * cay
    const dcb = cbx * cbx + cby * cby
    return s * std.sqrt(std.min(dca, dcb))
})

/**
 * Heart SDF (iq). `radius` scales the heart to fit within that radius. Works in iq's heart space
 * (point near origin, lobes around y≈1.2); screen-y is downward so `dy` is negated to put the
 * point at the bottom and lobes at the top, then a recenter + √-fit scale. Caller applies
 * rotation externally.
 */
export const heartSdf = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((dx, dy, radius) => {
    'use gpu'
    const S = radius / 0.75
    const px = std.abs(dx) / S
    const py = (dy * -1.0) / S + 0.6
    // Branch A: outer lobes (px + py > 1).
    const ax = px - 0.25
    const ay = py - 0.75
    const distA = std.sqrt(ax * ax + ay * ay) - SQRT2_OVER_4
    // Branch B: cusp + point.
    const b1y = py - 1.0
    const dot1 = px * px + b1y * b1y
    const m = std.max(px + py, 0.0) * 0.5
    const b2x = px - m
    const b2y = py - m
    const dot2v = b2x * b2x + b2y * b2y
    const distB = std.sqrt(std.min(dot1, dot2v)) * std.sign(px - py)
    return std.select(distB, distA, px + py > 1.0) * S
})

/**
 * Arc / pie sector SDF (iq's sdPie) — a filled circular wedge opening ±`halfAngle` about +y.
 * `radius` = sector radius, `halfAngle` = half the aperture (radians). Caller applies rotation
 * externally.
 */
export const arcSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, radius, halfAngle) => {
    'use gpu'
    const px = std.abs(dx)
    const cx = std.sin(halfAngle)
    const cy = std.cos(halfAngle)
    const l = std.sqrt(px * px + dy * dy) - radius
    const clamped = std.clamp(px * cx + dy * cy, 0.0, radius)
    const mx = px - cx * clamped
    const my = dy - cy * clamped
    const m = std.sqrt(mx * mx + my * my)
    return std.max(l, m * std.sign(cy * px - cx * dy))
})

/**
 * Teardrop SDF — a 2D rounded cone: a bulb of radius `radius` at the base tapering to a sharp
 * point a distance `h` away. `dy` negated so the bulb sits at the bottom, point up (screen-y is
 * down). Exact. Caller applies rotation externally.
 */
export const teardropSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, radius, h) => {
    'use gpu'
    const mid = (h - radius) * 0.5
    const qx = std.abs(dx)
    const qy = dy * -1.0 + mid
    const hSafe = std.max(h, 0.0001)
    const b = radius / hSafe // (r1 - r2)/h with r2 = 0
    const a = std.sqrt(std.max(1.0 - b * b, 0.0))
    const k = qx * (b * -1.0) + qy * a
    const cap0 = std.sqrt(qx * qx + qy * qy) - radius
    const qyTip = qy - h
    const cap1 = std.sqrt(qx * qx + qyTip * qyTip)
    const body = qx * a + qy * b - radius
    return std.select(std.select(body, cap1, k > a * h), cap0, k < 0.0)
})

/**
 * Parallelogram SDF (iq). `wi` = half-width, `he` = half-height, `sk` = horizontal skew of the
 * top edge. Exact. Caller applies rotation externally.
 */
export const parallelogramSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((dx, dy, wi, he, sk) => {
    'use gpu'
    const ex = sk
    const ey = he
    const flip1 = dy < 0.0
    const px0 = std.select(dx, dx * -1.0, flip1)
    const py0 = std.select(dy, dy * -1.0, flip1)
    const wx0 = px0 - ex
    const wy = py0 - ey
    const wx = wx0 - std.clamp(wx0, wi * -1.0, wi)
    const dX = wx * wx + wy * wy
    const dY = wy * -1.0
    const s = px0 * ey - py0 * ex
    const flip2 = s < 0.0
    const px = std.select(px0, px0 * -1.0, flip2)
    const py = std.select(py0, py0 * -1.0, flip2)
    const vx0 = px - wi
    const dotve = vx0 * ex + py * ey
    const dotee = std.max(ex * ex + ey * ey, 0.0001)
    const tt = std.clamp(dotve / dotee, -1.0, 1.0)
    const vx = vx0 - ex * tt
    const vy = py - ey * tt
    const finalX = std.min(dX, vx * vx + vy * vy)
    const finalY = std.min(dY, wi * he - std.abs(s))
    return std.sqrt(finalX) * std.sign(finalY * -1.0)
})

// ─── Shared shape-rendering helpers ────────────────────────────────────────────
// The aspect-correct/rotate preamble and the stroke/softness mask that every analytic SHAPE shader
// (Circle/Ellipse/Cross/Heart, …) repeats, so the per-shader body is just "local coords → sdf →
// mask". Shared once here (like kit/edges' Expr-level builders) to keep the shape fleet DRY.

/**
 * Aspect-correct + rotate a screen UV into shape-local space centered on `center`. The shape
 * preamble: aspect-correct X (circular shapes stay circular), flip `center.y` back
 * (transformPosition stores `(x, 1 - y)`, so `1 - center.y` recovers the authored y), then rotate
 * the delta by `rotation` degrees. Pass `rotation = 0` for shapes with no rotation (the cos/sin
 * fold to identity). Returns the rotated `(dx, dy)` offset to feed an SDF primitive.
 */
export const shapeLocalCoords = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32], d.vec2f)((center, rotation, uv, aspect) => {
    'use gpu'
    const aspectUV = d.vec2f(uv.x * aspect, uv.y)
    const centerPos = d.vec2f(center.x * aspect, 1.0 - center.y)
    const dx = aspectUV.x - centerPos.x
    const dy = aspectUV.y - centerPos.y
    const rotRad = rotation * DEG_TO_RAD
    const cosR = std.cos(rotRad)
    const sinR = std.sin(rotRad)
    const rdx = dx * cosR + dy * sinR
    const rdy = dy * cosR - dx * sinR
    return d.vec2f(rdx, rdy)
})

/** The stroke/softness mask a shape's fragment builder mixes fill + stroke color with. */
export const ShapeMask = d.struct({overallMask: d.f32, strokeBlend: d.f32})

/**
 * Fill coverage + stroke blend factor from an SDF value.
 * `overallMask` fades the whole shape out at its outer edge (softness-wide smoothstep); `strokeBlend`
 * fades the fill color into the stroke color at the inner edge (0 when there's no stroke). Stroke
 * boundaries are expressed in SDF space (boundary at 0) — `strokePosition` 0=outside / 1=center /
 * 2=inside — so it works for any primitive (circles feed `distance - circleEdge`, which is exactly
 * `circleSdf`, so the same helper covers Circle too).
 */
export const strokeMaskFromSdf = tgpu.fn([d.f32, d.f32, d.f32, d.f32], ShapeMask)((dist, softness, strokeThickness, strokePosition) => {
    'use gpu'
    const hasStroke = strokeThickness > 0.0
    const isInside = strokePosition >= 1.5
    // In the `isInside === false` branch this is `isCenter` (strokePosition already < 1.5).
    const isCenterOrAbove = strokePosition >= 0.5
    const halfThickness = strokeThickness * 0.5
    const negThickness = strokeThickness * -1.0
    const negHalf = halfThickness * -1.0

    const strokeInner = std.select(std.select(d.f32(0), negHalf, isCenterOrAbove), negThickness, isInside)
    const strokeOuter = std.select(std.select(strokeThickness, halfThickness, isCenterOrAbove), d.f32(0), isInside)

    const overallMask = 1.0 - std.smoothstep(strokeOuter - softness, strokeOuter, dist)
    const blend = std.smoothstep(strokeInner - softness, strokeInner, dist)
    const strokeBlend = std.select(d.f32(0), blend, hasStroke)
    return ShapeMask({overallMask, strokeBlend})
})

/**
 * The shape fragment TAIL, at the Expr/builder level: take a `ShapeMask` (from
 * `strokeMaskFromSdf`) plus the fill and stroke colors, blend fill → stroke by `strokeBlend` in
 * the given color space, and fade the whole shape by `overallMask`.
 *
 * `colorSpaceMode` is a COMPILE-TIME JS number (the `colorSpace` prop is `compileTime: true`), so
 * the variant is selected here in JS and only that variant's math is emitted — this is a builder,
 * not something to call from a `'use gpu'` body. The unknown-mode fallback is linear, matching
 * every shape's inline `?? mixColorsLinear`.
 *
 * Every analytic 2D shape shader shares this tail; `scaffolds/sdfShape.ts` calls it for the fleet,
 * and a shape that needs a custom fragment can call it directly instead of re-deriving the mix.
 */
export function fillStrokeColor(mask: Expr, fill: Expr, stroke: Expr, colorSpaceMode: number): Expr {
    const variant = mixColorsVariants[colorSpaceMode as keyof typeof mixColorsVariants] ?? mixColorsLinear
    const blended = call(variant, 'mixColors', [fill, stroke, mask.member('strokeBlend')])
    return vec4(blended.member('rgb'), blended.member('a').mul(mask.member('overallMask')))
}

// ─── SVG SDF binary loading (CPU code) ─────────────────

/**
 * Fetch a pre-computed SDF binary from a URL and fill `sdfData` in place. Supports two formats
 * (detected by byte length):
 *   - compact: Uint16, two bytes per pixel — decode [0, 65535] → [-1, 1]
 *   - legacy:  Float32, four bytes per pixel — copy directly
 *
 * The SVG-shape path uploads `sdfData` into an r16float DataTexture via
 * `textureManager.createDataTexture` (r16float is always linear-filterable; r32float requires the
 * optional `float32-filterable` feature), then samples it through the fn-arg / KitTexture path.
 */
export async function loadSdfFromUrl(url: string, sdfData: Float32Array): Promise<void> {
    const response = await fetch(url, {mode: 'cors'})
    if (!response.ok) throw new Error(`SDF fetch failed: HTTP ${response.status}`)
    const buffer = await response.arrayBuffer()
    const pixelCount = SVG_SDF_SIZE * SVG_SDF_SIZE

    if (buffer.byteLength === pixelCount * 2) {
        // New compact format: Uint16 encoded, decode [0, 65535] → [-1, 1].
        const uint16 = new Uint16Array(buffer)
        const n = Math.min(uint16.length, sdfData.length)
        for (let i = 0; i < n; i++) sdfData[i] = uint16[i] / 32767.5 - 1.0
    } else if (buffer.byteLength === pixelCount * 4) {
        // Legacy format: raw Float32.
        const floats = new Float32Array(buffer)
        const n = Math.min(floats.length, sdfData.length)
        for (let i = 0; i < n; i++) sdfData[i] = floats[i]
    } else {
        // Unexpected size — the URL likely points to an SVG or other non-SDF file rather than a
        // generated .bin. Fail clearly instead of a cryptic "multiple of 4" RangeError.
        throw new Error(
            `SDF binary has unexpected size (${buffer.byteLength} bytes); expected ` +
                `${pixelCount * 2} (compact) or ${pixelCount * 4} (legacy). ` +
                `The URL may point to a non-SDF file (e.g. the raw SVG).`,
        )
    }
}

// ─── SVG-SDF samplers (data-texture backed) ────
//
// FORMAT CHOICE (r16float, not r32float): the kit KitTexture `.sample()` emits `textureSample(…)`
// with a FILTERING sampler (linearClamp), and WebGPU rejects a filtering sampler bound to a
// non-filterable texture at pipeline-creation time. `r16float`/`rgba16float` are ALWAYS
// linear-filterable; `r32float`/`rgba32float` need the OPTIONAL `float32-filterable` feature, so
// binding one under a filtering sampler would be a hard validation error, not a soft degrade. The
// field lives in [-1, 1] where half precision (~1e-3) shifts refraction well under a device pixel.
// The trade is that raw floats must be half-encoded before upload (`toHalfFloat`), since
// `createDataTexture` takes r16float texel bytes (Uint16), not Float32.

// `toHalfFloat` lives in the shared kit/dataEncoding.ts (used by every sampled data texture, not
// just SVG SDFs).

/**
 * The shared single-channel SDF data texture: an `r16float` `SVG_SDF_SIZE²` field initialised to
 * 0.5 (outside — invisible until the async load lands), with the async `.bin` load + on-ready
 * re-upload scheduled via `params.onBeforeRender` (a `needsUpdate` flag), and disposal via
 * `params.onCleanup`. Returned as `{kitTexture, texture}` so the FLAT sampler samples `kitTexture`
 * in the fragment while the SVG-3D setup binds the raw `texture` into its compute march kernel.
 */
// ─── Generated SVG fields (no .bin) ────
//
// When the shape config carries the SVG itself (`svgUrl`, written by the shape picker alongside
// `shapeSdfUrl`), the field is generated on the GPU from the SVG (`svgSdfGenerate.ts`) at a finer
// resolution than the 512² `.bin`; the `.bin` stays the fallback when the SVG can't be fetched or
// decoded. Both builders below hand out that one texture (`.r` distance, `.g/.b` the baked slope),
// so every consumer binds it exactly like the data texture it replaces.

/** Generated-field resolution: 2048² on desktop (~2.5 ms GPU), 1024² on phones. */
export function generatedSvgSdfSize(): number {
    return isMobileGpuViewport() ? 1024 : 2048
}

/** The SVG URL a shape config carries, or '' (string or object config). */
export function svgUrlOf(params: GpuFragmentParams): string {
    const cfg = parseShapeConfigValue(params.getCpuValue?.('shape'))
    return typeof cfg.svgUrl === 'string' ? cfg.svgUrl : ''
}

/** A generated field for this node's SVG, or null when there is no SVG URL or no GPU. */
function generatedSvgField(params: GpuFragmentParams, shapeSdfUrl: string): SvgSdfField | null {
    const root = params.gpu?.root
    const svgUrl = svgUrlOf(params)
    if (!root || !svgUrl) return null
    const field = generateSvgSdfField(root as never, {svgUrl, sdfUrl: shapeSdfUrl}, generatedSvgSdfSize(), loadSdfFromUrl, SVG_SDF_SIZE)
    params.onCleanup(() => field.destroy())
    field.ready.then((source) => {
        if (source !== 'svg') debugWarn(`[SDF] Could not generate from the SVG; ${source === 'bin' ? 'using the .bin fallback' : 'no field'}:`, svgUrl)
    })
    return field
}

/** A generated field dressed as the data texture it replaces. */
function asMediaTexture(params: GpuFragmentParams, field: SvgSdfField): GpuMediaTexture {
    const root = params.gpu!.root as unknown as {unwrap: (t: unknown) => GPUTexture}
    return {
        texture: field.texture,
        width: field.size,
        height: field.size,
        write: () => {},
        unwrap: () => root.unwrap(field.texture),
        generateMipmaps: () => {},
        destroy: () => field.destroy(),
    }
}

export function createSdfDataTexture(
    params: GpuFragmentParams,
    shapeSdfUrl: string,
): {kitTexture: KitTexture; texture: GpuMediaTexture; getVersion: () => number} {
    const generated = generatedSvgField(params, shapeSdfUrl)
    if (generated) {
        return {
            kitTexture: params.registerMediaTexture(() => generated.texture),
            texture: asMediaTexture(params, generated),
            getVersion: generated.getVersion,
        }
    }
    const count = SVG_SDF_SIZE * SVG_SDF_SIZE
    // CPU field mirror (used to re-encode after the async load); half-encoded upload buffer.
    const sdfData = new Float32Array(count).fill(0.5)
    const texData = new Uint16Array(count)
    const halfHalf = toHalfFloat(0.5)
    for (let i = 0; i < count; i++) texData[i] = halfHalf

    const texture = params.createDataTexture({
        width: SVG_SDF_SIZE,
        height: SVG_SDF_SIZE,
        format: 'r16float',
        data: texData,
        label: 'svg-sdf',
    })
    params.onCleanup(() => texture.destroy())

    let needsUpload = false
    let version = 0
    if (shapeSdfUrl) {
        loadSdfFromUrl(shapeSdfUrl, sdfData)
            .then(() => {
                for (let i = 0; i < count; i++) texData[i] = toHalfFloat(sdfData[i])
                needsUpload = true
            })
            .catch((err) => debugWarn('[SDF] Failed to load SDF texture:', err))
    }
    // Defer the GPU write to the render loop (needsUpdate-on-ready scheduling). Bump `version` on
    // each upload so the SVG-3D setup's state key re-marches once when the load lands.
    params.onBeforeRender(() => {
        if (needsUpload) {
            needsUpload = false
            texture.write(texData)
            version++
        }
    })

    const kitTexture = params.registerMediaTexture(() => texture.texture)
    return {kitTexture, texture, getVersion: () => version}
}

/**
 * Flat SVG-SDF sampler. Backs the shape-effect shaders over a custom uploaded SVG shape. `.r` of
 * the returned vec4 is the signed distance (RedFormat → `.r`, matching what the effects read).
 * Returns `(uv) => sample` for use exactly like the analytic sampler.
 */
export function createSvgSdfSampler(params: GpuFragmentParams, shapeSdfUrl: string): SvgSdfSampler {
    const {kitTexture} = createSdfDataTexture(params, shapeSdfUrl)
    // Explicit-LOD: material bodies tap the field inside `guarded` branches (non-uniform flow),
    // where implicit-derivative textureSample is a WGSL validation error. Single mip → identical.
    return Object.assign((uv: Expr) => kitTexture.sampleLevel(uv), {kitTexture})
}

/** An SVG-SDF sampler that also carries the field texture it reads (`.r` = signed distance). */
export type SvgSdfSampler = ((uv: Expr) => Expr) & {kitTexture: KitTexture}

/**
 * Flat SVG-SDF sampler with BAKED forward-difference gradients. An `rgba16float` field: `.r` =
 * distance, `.g/.b` = the per-texel forward differences `(f(uv+ε)−f(uv))/ε` in x/y, precomputed
 * once on the CPU when the SDF loads (the field is static per URL). Effects that take finite
 * differences of the flat field read `.g/.b` of their single centre tap instead of paying two
 * extra texture samples. `.a` = 1 (matching the RedFormat sampler's implicit alpha). RGBA needs 4
 * channels, so this uses a `createMediaTexture` (`rgba16float`) — `createDataTexture` is
 * single-channel only.
 */
export function createSvgSdfSamplerWithGradients(params: GpuFragmentParams, shapeSdfUrl: string): SvgSdfSampler {
    // The generated field already carries the baked slope in `.g/.b`.
    const generated = generatedSvgField(params, shapeSdfUrl)
    if (generated) {
        const kitTexture = params.registerMediaTexture(() => generated.texture)
        return Object.assign((uv: Expr) => kitTexture.sampleLevel(uv), {kitTexture})
    }
    const size = SVG_SDF_SIZE
    const count = size * size
    const sdfData = new Float32Array(count).fill(0.5)
    const texData = new Uint16Array(count * 4)
    const halfHalf = toHalfFloat(0.5)
    const halfOne = toHalfFloat(1)
    // Initialised to 0.5 (outside) like the plain sampler — invisible until load.
    for (let i = 0; i < count; i++) {
        texData[i * 4] = halfHalf
        texData[i * 4 + 3] = halfOne
    }

    const texture = params.createMediaTexture({
        width: size,
        height: size,
        format: 'rgba16float',
        label: 'svg-sdf-grad',
    })
    params.onCleanup(() => texture.destroy())

    // CPU mirror of the GPU bilinear fetch with clamp-to-edge addressing, in texel coordinates
    // (t = uv·size − 0.5; a texel centre is t = x).
    const sampleField = (tx: number, ty: number): number => {
        const x0 = Math.floor(tx)
        const y0 = Math.floor(ty)
        const fx = tx - x0
        const fy = ty - y0
        const cl = (v: number) => Math.min(Math.max(v, 0), size - 1)
        const x0c = cl(x0),
            x1c = cl(x0 + 1),
            y0c = cl(y0),
            y1c = cl(y0 + 1)
        return (
            sdfData[y0c * size + x0c] * (1 - fx) * (1 - fy) +
            sdfData[y0c * size + x1c] * fx * (1 - fy) +
            sdfData[y1c * size + x0c] * (1 - fx) * fy +
            sdfData[y1c * size + x1c] * fx * fy
        )
    }

    const EPS_UV = 0.01 // matches the effects' finite-difference EPS
    const epsTexels = EPS_UV * size // 5.12 texels
    const bake = () => {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const i = y * size + x
                const f0 = sdfData[i]
                const gx = (sampleField(x + epsTexels, y) - f0) / EPS_UV
                const gy = (sampleField(x, y + epsTexels) - f0) / EPS_UV
                texData[i * 4] = toHalfFloat(f0)
                texData[i * 4 + 1] = toHalfFloat(gx)
                texData[i * 4 + 2] = toHalfFloat(gy)
            }
        }
    }

    let needsUpload = false
    if (shapeSdfUrl) {
        loadSdfFromUrl(shapeSdfUrl, sdfData)
            .then(() => {
                bake()
                needsUpload = true
            })
            .catch((err) => debugWarn('[SDF] Failed to load SDF texture:', err))
    }
    params.onBeforeRender(() => {
        if (needsUpload) {
            needsUpload = false
            texture.write(texData)
        }
    })

    const kitTexture = params.registerMediaTexture(() => texture.texture)
    return Object.assign((uv: Expr) => kitTexture.sampleLevel(uv), {kitTexture})
}

// ─── Compile-time analytic 2D SDF sampler ─
//
// CONSUMER OWNS THE UNIFORMS: a kit builder can't own uniforms, so the CONSUMING shape-effect
// shader declares the shape sub-props (as `props` or `extraFields`) and passes their GPU accessor
// Exprs in via `sub`. The shape type is dispatched at BUILD TIME (compile-time JS branch), so the
// emitted WGSL carries only the selected primitive's math. Returns `(uv) => vec4(dist, 0, 0, 1)`,
// plug-compatible with the SVG sampler. 3D shape types do NOT come through here — they need a
// compute pre-march, so the consumer routes them through `createVolumetricFieldComputeNode`
// (kit/sdf3d.ts).

/** The shape sub-prop accessor Exprs a consumer supplies to `createAnalyticSdfSampler`. Missing
 *  keys default to 0.0; a shape only reads the sub-props its formula needs (see the switch). */
export interface AnalyticSubProps {
    radius?: Expr
    sides?: Expr
    rounding?: Expr
    innerRatio?: Expr
    rotation?: Expr
    height?: Expr
    offset?: Expr
    aperture?: Expr
}

/**
 * Aperture (full angle, degrees) → half-angle (radians): × π/360. Pre-folded module const, because
 * `Math.*` inside a `'use gpu'` body is unavailable.
 *
 * Exported for the Arc shader, whose body needs the same conversion. A body cannot read
 * `sdf.APERTURE_TO_HALFANGLE` directly (a member access inside `'use gpu'` is not proven to fold),
 * so Arc aliases it to a module-scope local at import time and references THAT.
 */
export const APERTURE_TO_HALFANGLE = 0.008726646259971648 //  Math.PI / 360

/**
 * Build the compile-time `tgpu.fn` for `shapeType`: `(uv, radius, sides, rounding, innerRatio,
 * rotation, height, offset, aperture) => vec4(dist, 0, 0, 1)`. One primitive is selected by a
 * build-time JS branch (`shapeType` is a plain string), so only its math is emitted. Rotation is
 * applied to (dx, dy) once here — circle/ring are rotation-invariant so passing the rotated coords
 * is identical to the unrotated fast path. Exported (over the closure) so it is directly resolvable.
 */
export function buildAnalyticSdfFn(shapeType: string) {
    return tgpu.fn(
        [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
        d.vec4f,
    )((uv, r, n, round, inner, rot, h, off, ap) => {
        'use gpu'
        const dx = uv.x - 0.5
        const dy = uv.y - 0.5
        const rotRad = rot * DEG_TO_RAD
        const cosR = std.cos(rotRad)
        const sinR = std.sin(rotRad)
        const rdx = dx * cosR + dy * sinR
        const rdy = dy * cosR - dx * sinR

        let dist = d.f32(0)
        if (shapeType === 'polygonSDF') dist = std.mix(polygonSdf(rdx, rdy, r, n), circleSdf(rdx, rdy, r), round)
        else if (shapeType === 'flowerSDF') dist = flowerSdf(rdx, rdy, r, n, inner)
        else if (shapeType === 'ringSDF') dist = ringSdf(rdx, rdy, r, inner)
        else if (shapeType === 'crossSDF') dist = crossSdf(rdx, rdy, r, inner, round)
        else if (shapeType === 'starSDF') dist = starSdf(rdx, rdy, r, n, inner)
        else if (shapeType === 'roundedRectSDF') dist = roundedRectSdf(rdx, rdy, r, h, round)
        else if (shapeType === 'ellipseSDF') dist = ellipseSdf(rdx, rdy, r, h)
        else if (shapeType === 'vesicaSDF') dist = vesicaSdf(rdx, rdy, r, inner)
        else if (shapeType === 'crescentSDF') dist = crescentSdf(rdx, rdy, r, inner, off)
        else if (shapeType === 'trapezoidSDF') dist = trapezoidSdf(rdx, rdy, inner, r, h)
        else if (shapeType === 'heartSDF') dist = heartSdf(rdx, rdy, r)
        else if (shapeType === 'arcSDF') dist = arcSdf(rdx, rdy, r, ap * APERTURE_TO_HALFANGLE)
        else if (shapeType === 'teardropSDF') dist = teardropSdf(rdx, rdy, r, h)
        else if (shapeType === 'parallelogramSDF') dist = parallelogramSdf(rdx, rdy, r, h, off)
        else dist = circleSdf(rdx, rdy, r)
        return d.vec4f(dist, 0.0, 0.0, 1.0)
    }).$name(`analyticSdf_${shapeType}`)
}

/**
 * Build the compile-time analytic 2D SDF sampler for `shapeType`. `patternMode` is accepted for
 * signature parity with the 3D path but has no effect on flat 2D shapes (they never fill the
 * surface-locked `.g/.b` pattern coords); it is documented here so a consumer can pass it
 * uniformly.
 */
export function createAnalyticSdfSampler(
    shapeType: string,
    sub: AnalyticSubProps,
    _patternMode: SurfacePatternMode = 'none',
): (uv: Expr) => Expr {
    void _patternMode
    const radius = sub.radius ?? floatE(0)
    const sides = sub.sides ?? floatE(0)
    const rounding = sub.rounding ?? floatE(0)
    const innerRatio = sub.innerRatio ?? floatE(0)
    const rotation = sub.rotation ?? floatE(0)
    const height = sub.height ?? floatE(0)
    const offset = sub.offset ?? floatE(0)
    const aperture = sub.aperture ?? floatE(0)
    const fn = buildAnalyticSdfFn(shapeType)
    return (uv: Expr) => call(fn, `analyticSdf_${shapeType}`, [uv, radius, sides, rounding, innerRatio, rotation, height, offset, aperture])
}

// ─── CPU shape-JSON → analytic sub-prop drivers (onBeforeRender uniform updates, flat-2D) ─────────
//
// The eight sub-props are re-read from the shape JSON each frame via `onBeforeRender`. A kit builder
// can't own uniforms, so the CONSUMING shape-effect shader DECLARES these as `extraFields` (spread
// `ANALYTIC_SDF_EXTRA_FIELDS` into its definition), calls `driveAnalyticSubProps(params,
// getShapeConfig)` in its `fragment` to wire the per-frame `setExtraField` writes, and passes the
// returned Exprs into `createAnalyticSdfSampler`. Shared by Glass / Frost / Neon / Emboss (the FLAT
// path). The per-frame fallback aliases (`width`/`bottomWidth`,
// `thickness`/`spread`/`topWidth`/`topRatio`, `skew`) resolve to the same sub-props.

/** The eight analytic shape sub-prop fields a flat shape-effect shader declares as `extraFields`
 *  (spread into the definition). Driven each frame by `driveAnalyticSubProps`. These are generic
 *  defaults; the JSON drives the real values on the first `onBeforeRender`, before the first
 *  render. */
export const ANALYTIC_SDF_EXTRA_FIELDS: Record<string, {schema: typeof d.f32; initial: number}> = {
    _saRadius: {schema: d.f32, initial: 0.35},
    _saSides: {schema: d.f32, initial: 6},
    _saRounding: {schema: d.f32, initial: 0},
    _saInnerRatio: {schema: d.f32, initial: 0.4},
    _saRotation: {schema: d.f32, initial: 0},
    _saHeight: {schema: d.f32, initial: 0.25},
    _saOffset: {schema: d.f32, initial: 0.2},
    _saAperture: {schema: d.f32, initial: 270},
}

/**
 * Wire the per-frame CPU resolution of the analytic shape sub-props from the shape JSON and return
 * the `AnalyticSubProps` accessor Exprs for `createAnalyticSdfSampler`. The consumer MUST declare
 * `...ANALYTIC_SDF_EXTRA_FIELDS` in its definition's `extraFields`. `getShapeConfig` returns the
 * live `shape` prop value (JSON string or object) — pass `() => params.getCpuValue('shape')`.
 */
/** The CPU-resolved analytic shape sub-props (the JSON → sub-prop mapping, aliases included) — the
 *  values `driveAnalyticSubProps` publishes as `_sa*` extraFields, for a consumer whose OWN compute
 *  params carry them. */
export function analyticSubPropValues(cfg: Record<string, unknown>): {
    radius: number; sides: number; rounding: number; innerRatio: number
    rotation: number; height: number; offset: number; aperture: number
} {
    const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)
    return {
        radius: num(cfg.radius, num(cfg.width, num(cfg.bottomWidth, 0.35))),
        sides: num(cfg.sides, 6),
        rounding: num(cfg.rounding, 0),
        innerRatio: num(cfg.innerRatio, num(cfg.thickness, num(cfg.spread, num(cfg.topWidth, num(cfg.topRatio, 0.4))))),
        rotation: num(cfg.rotation, 0),
        height: num(cfg.height, 0.25),
        offset: num(cfg.offset, num(cfg.skew, 0.2)),
        aperture: num(cfg.aperture, 270),
    }
}

/** Parse the live `shape` prop (JSON string or object) into its config record. */
export function parseShapeConfigValue(raw: unknown): Record<string, unknown> {
    if (raw && typeof raw === 'object') return raw as Record<string, unknown>
    if (typeof raw === 'string') { try { return JSON.parse(raw) } catch { return {} } }
    return {}
}

export function driveAnalyticSubProps(params: GpuFragmentParams, getShapeConfig: () => unknown): AnalyticSubProps {
    let lastShapeJson = ''
    let lastCfg: Record<string, unknown> = parseShapeConfigValue(getShapeConfig())
    const write = () => {
        const s = analyticSubPropValues(lastCfg)
        params.setExtraField('_saRadius', s.radius)
        params.setExtraField('_saSides', s.sides)
        params.setExtraField('_saRounding', s.rounding)
        params.setExtraField('_saInnerRatio', s.innerRatio)
        params.setExtraField('_saRotation', s.rotation)
        params.setExtraField('_saHeight', s.height)
        params.setExtraField('_saOffset', s.offset)
        params.setExtraField('_saAperture', s.aperture)
    }
    params.onBeforeRender(() => {
        const raw = getShapeConfig()
        if (raw && typeof raw === 'object') {
            lastCfg = raw as Record<string, unknown>
        } else if (typeof raw === 'string') {
            if (raw !== lastShapeJson) {
                lastShapeJson = raw
                try { lastCfg = JSON.parse(raw) } catch { return }
            }
        }
        write()
    })
    const u = params.uniforms
    return {
        radius: u._saRadius, sides: u._saSides, rounding: u._saRounding, innerRatio: u._saInnerRatio,
        rotation: u._saRotation, height: u._saHeight, offset: u._saOffset, aperture: u._saAperture,
    }
}
