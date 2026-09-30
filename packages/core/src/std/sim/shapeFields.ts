/**
 * std/sim/shapeFields — the shape a swarm is held inside.
 *
 * A containment force (`force.containment`) needs one thing from a shape: the signed distance
 * to its surface at any 3D point, negative inside and positive outside. `shapeField` builds
 * that distance function from whatever shape the layer was given: a flat shape or an SVG
 * outline given some depth, a true 3D solid, or an SVG lifted into 3D. One integrator then
 * serves every shape kind.
 *
 * Coordinates are shape-local: the shape sits in a cube from −0.5 to 0.5 with y up, and the
 * layer's `center`, `scale` and `rotation` are applied elsewhere.
 */
// Maintainer notes. Baked 3D signed-distance FIELD parts for the volume-swarm family
// (Particles is the consumer, see `resolveShapeField` there). Layout contracts follow the
// family convention: factories take the consumer's layouts and reference entries by REQUIRED
// NAMES (documented per factory).
import {tgpu, d, std, sdf3d} from '../../gpu/kit/index'
import type {ShapeField3} from './agentForces'

/** Rounded z-extrusion of a flat signed distance (the standard opExtrusion). */
const extrudeZ = tgpu.fn([d.f32, d.f32], d.f32)((d2, wz) => {
    'use gpu'
    const ox = std.max(d2, 0.0)
    const oz = std.max(wz, 0.0)
    return std.min(std.max(d2, wz), 0.0) + std.sqrt(ox * ox + oz * oz)
}).$name('shapeFieldExtrudeZ')

/** The analytic 2D SDF menu's call shape (`sdf.buildAnalyticSdfFn`). */
type AnalyticSdf2d = (
    uv: d.v2f, radius: number, sides: number, rounding: number, innerRatio: number,
    rotation: number, height: number, offset: number, aperture: number,
) => {readonly x: number}

/** Params view for the extruded flat-shape fields: the CPU-resolved analytic sub-prop bundle
 *  (the driveAnalyticSubProps convention) plus the extrusion half-depth. */
interface ExtrudedAnalyticParamsView {
    readonly saRadius: number
    readonly saSides: number
    readonly saRounding: number
    readonly saInnerRatio: number
    readonly saRotation: number
    readonly saHeight: number
    readonly saOffset: number
    readonly saAperture: number
    readonly halfDepth: number
}

/** The per-axis sin/cos rotation bundle the sdf3d MarchParams carry. */
interface RotSinCosView {
    readonly cx: number
    readonly sx: number
    readonly cy: number
    readonly sy: number
    readonly cz: number
    readonly sz: number
}

/** A layout whose MarchParams uniform carries the shape rotation (the sdf3d setups). */
interface RotatedFieldLayout {
    readonly $: {readonly params: {readonly rot: RotSinCosView}}
}

/**
 * Flat analytic shape extruded in z: the baked 2D SDF evaluated in shape-local uv (y up →
 * v down), sub-props read live from `params.sa*`, rounded-extruded over `params.halfDepth`.
 */
function analyticExtruded(
    layout: {readonly $: {readonly params: ExtrudedAnalyticParamsView}},
    sdf2d: AnalyticSdf2d,
): ShapeField3 {
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        const prm = layout.$.params
        const uv2 = d.vec2f(p3.x + 0.5, 0.5 - p3.y)
        const d2 = sdf2d(uv2, prm.saRadius, prm.saSides, prm.saRounding, prm.saInnerRatio, prm.saRotation, prm.saHeight, prm.saOffset, prm.saAperture).x
        return extrudeZ(d2, std.abs(p3.z) - prm.halfDepth)
    }).$name('shapeFieldAnalyticExtruded') as ShapeField3
}

/**
 * Flat SVG outline extruded in z (nearest texel read — plenty for containment forces).
 * Beyond the SDF texture's ±0.5 footprint the clamped field is flat (or, under a feature
 * touching the border, even reads INSIDE) — which strands strays and draws plumb-line
 * columns — so the field is extended with the exact distance to the footprint box, keeping
 * magnitude + direction valid everywhere the swarm can roam.
 */
function svgExtruded(
    layout: {readonly $: {readonly params: {readonly halfDepth: number}}},
    svgLayout: {readonly $: {readonly sdfSource: d.Infer<d.WgslTexture2d<d.F32>>}},
    cfg: {size: number},
): ShapeField3 {
    const SIZE = cfg.size
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        const prm = layout.$.params
        const sizeM1 = SIZE - 1
        const tx = std.clamp(d.i32((p3.x + 0.5) * SIZE), 0, sizeM1)
        const ty = std.clamp(d.i32((0.5 - p3.y) * SIZE), 0, sizeM1)
        const d2 = std.textureLoad(svgLayout.$.sdfSource, d.vec2u(d.u32(tx), d.u32(ty)), 0).x
        const exX = std.max(std.abs(p3.x) - 0.5, 0.0)
        const exY = std.max(std.abs(p3.y) - 0.5, 0.0)
        const d2e = d2 + std.sqrt(exX * exX + exY * exY)
        return extrudeZ(d2e, std.abs(p3.z) - prm.halfDepth)
    }).$name('shapeFieldSvgExtruded') as ShapeField3
}

/**
 * True analytic 3D volume: the setup's baked SDF sampled at the rotated point — rotation
 * (and any CPU-animated sub-props) arrive through the setup's MarchParams uniform, so the
 * field moves under the swarm exactly like the marched render would.
 */
function analytic3d(layout: RotatedFieldLayout, baked: ShapeField3): ShapeField3 {
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        return baked(sdf3d.rotateVec3(p3, layout.$.params.rot))
    }).$name('shapeFieldAnalytic3d') as ShapeField3
}

/**
 * SVG lifted into 3D (the bevel-extruding SVG-3D setup's SDF), rotated by MarchParams and —
 * like the flat SVG field — extended beyond the SDF texture's footprint with the exact
 * distance to the footprint box (the clamped field is flat out there: zero gradient, and a
 * border-touching feature can even read inside).
 */
function svgLifted3d(layout: RotatedFieldLayout, baked: ShapeField3): ShapeField3 {
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        const ps = sdf3d.rotateVec3(p3, layout.$.params.rot)
        const exX = std.max(std.abs(ps.x) - 0.5, 0.0)
        const exY = std.max(std.abs(ps.y) - 0.5, 0.0)
        return baked(ps) + std.sqrt(exX * exX + exY * exY)
    }).$name('shapeFieldSvgLifted3d') as ShapeField3
}

/**
 * The signed-distance function of the shape a swarm lives in, for `force.containment`.
 *
 * Four builders, one per shape source. Each returns a `ShapeField3`: a GPU function from a
 * shape-local 3D point to a signed distance, negative inside. `analyticExtruded` takes a flat
 * shape (circle, polygon, star…) and gives it a depth from `params.halfDepth`. `svgExtruded`
 * does the same for an SVG outline uploaded as a distance texture. `analytic3d` wraps a true
 * 3D solid so it rotates under the swarm. `svgLifted3d` wraps an SVG lifted into 3D.
 *
 * @example
 * ```ts
 * force.containment(simLayout, {field: shapeField.analyticExtruded(simLayout, buildAnalyticSdfFn(shapeType)), gradEps: 0.02, wall: {featherFrom: -0.1, featherTo: 0.01, base: 1.1, springK: 22}, recall: {from: 1.5, to: 1.9, k: 9}, homing: {from: 0.12, to: 0.45, k: 5}, entrainment: {featherHalf: 0.05}})
 * ```
 * @tip Pick the builder at bake time from the shape props and hand ONE field to the integrator. The swarm never needs to know which kind it is.
 * @see force, integrator
 */
export const shapeField = {
    /**
     * A flat shape given depth: the 2D distance of a circle, polygon or star, extruded in z by
     * `params.halfDepth`. The shape's sub-props are read live from `params.sa*`.
     *
     * @example
     * ```ts
     * shapeField.analyticExtruded(simLayout, buildAnalyticSdfFn('hexagon'))
     * ```
     * @see svgExtruded, analytic3d
     */
    // Layout needs `params.{saRadius, saSides, saRounding, saInnerRatio, saRotation, saHeight,
    // saOffset, saAperture, halfDepth}` — the driveAnalyticSubProps bundle, written per frame.
    analyticExtruded,
    /**
     * An SVG outline given depth: the outline's distance texture, extruded in z by
     * `params.halfDepth`. `size` is the texture's side in texels.
     *
     * @example
     * ```ts
     * shapeField.svgExtruded(simLayout, svgSdfLayout, {size: 512})
     * ```
     * @tip Outside the texture's footprint the distance keeps growing, so strays still find their way back.
     * @see analyticExtruded, svgLifted3d
     */
    // Layout needs `params.halfDepth`; `svgLayout` needs `sdfSource` (a sampled f32 texture).
    svgExtruded,
    /**
     * A true 3D solid (sphere, torus, knot…) from its baked distance function, rotated by the
     * shape's live rotation so the swarm follows the surface as it turns.
     *
     * @example
     * ```ts
     * shapeField.analytic3d(setup3d.layout, setup3d.sdfFn as ShapeField3)
     * ```
     * @see analyticExtruded, svgLifted3d
     */
    // `layout.$.params.rot` is the sdf3d MarchParams sin/cos bundle; `baked` is the setup's sdfFn.
    analytic3d,
    /**
     * An SVG outline lifted into a 3D solid, rotated by the shape's live rotation.
     *
     * @example
     * ```ts
     * shapeField.svgLifted3d(setup3d.layout, setup3d.sdfFn as ShapeField3)
     * ```
     * @see svgExtruded, analytic3d
     */
    // Same MarchParams rotation as analytic3d, plus the footprint-box extension of svgExtruded.
    svgLifted3d,
} as const
