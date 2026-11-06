/**
 * Reconstruction filters for resampling an RTT.
 *
 * WHY THIS EXISTS. Every distortion has two code paths: the analytic one, where its UV remap is
 * folded into the coordinate the content underneath is generated at (no resampling at all, so
 * nothing to filter), and the RTT one, where the composed child is rasterised to a texture first
 * and then re-sampled at the remapped coordinate. On the RTT path the hardware `linearClamp`
 * sampler gives a single bilinear tap, and bilinear MAGNIFICATION of a high-contrast edge — text,
 * a logo, a clip-mask boundary — reconstructs it as a piecewise-linear ramp across the texel grid.
 * That reads as faceted, stair-stepped "pixellation": the artifact is the filter, not the
 * resolution.
 *
 * Catmull-Rom fixes the reconstruction: a C1, interpolating cubic whose negative lobes keep edges
 * looking crisp instead of smeared. The 9-tap form below is the standard Jimenez trick — the 4×4
 * cubic footprint is evaluated as 3×3 BILINEAR taps by placing the middle tap at the
 * weight-weighted centroid of the inner two texels, so the hardware filter does a third of the
 * work for us (16 texel fetches → 9 bilinear fetches).
 *
 * Per-axis positions and weights are separate DualFns so both are golden-testable on the CPU; the
 * Expr-level builder memoises all four into locals, so each is evaluated ONCE per use rather than
 * re-emitted per tap.
 *
 * The negative lobes can undershoot below zero on a hard edge. The result is clamped at 0 because
 * the RTT holds PREMULTIPLIED alpha and a negative alpha would blow up the unpremultiply divide
 * downstream. Overshoot ABOVE the neighbourhood max is deliberately left alone — that is the
 * edge acutance Catmull-Rom is here for, and the final tonemap handles the range.
 */
import {tgpu, d, std} from './index'
import type {Expr, KitTexture, SharedSamplerName} from '../contract'
import {Expr as ExprClass} from '../contract'

/**
 * Catmull-Rom tap POSITIONS for one axis, in normalised (UV) space: `(pos0, pos12, pos3)`.
 * `pos12` is offset off the texel centre so one bilinear tap covers the inner texel pair with the
 * correct ratio. `uv` is that axis's UV; `texSize` that axis's size in texels.
 */
export const catmullRomAxisPositions = tgpu.fn([d.f32, d.f32], d.vec3f)((uv, texSize) => {
    'use gpu'
    const samplePos = uv * texSize
    // Texel centre of the second of the four taps (0.5 offsets are texel centres).
    const texPos1 = std.floor(samplePos - 0.5) + 0.5
    const f = samplePos - texPos1
    const w1 = 1.0 + f * f * (-2.5 + 1.5 * f)
    const w2 = f * (0.5 + f * (2.0 - 1.5 * f))
    // Guarded: w1 + w2 is ~1 across f in [0,1), but never divide blind.
    const w12 = std.max(w1 + w2, 1e-6)
    const offset12 = w2 / w12
    return d.vec3f((texPos1 - 1.0) / texSize, (texPos1 + offset12) / texSize, (texPos1 + 2.0) / texSize)
})

/**
 * Catmull-Rom tap WEIGHTS for one axis: `(w0, w1 + w2, w3)`, matching
 * `catmullRomAxisPositions`. Sums to 1 (the cubic is a partition of unity), so a 3×3 product
 * of two axes' weights also sums to 1 — no renormalisation needed.
 */
export const catmullRomAxisWeights = tgpu.fn([d.f32, d.f32], d.vec3f)((uv, texSize) => {
    'use gpu'
    const samplePos = uv * texSize
    const texPos1 = std.floor(samplePos - 0.5) + 0.5
    const f = samplePos - texPos1
    const w0 = f * (-0.5 + f * (1.0 - 0.5 * f))
    const w1 = 1.0 + f * f * (-2.5 + 1.5 * f)
    const w2 = f * (0.5 + f * (2.0 - 1.5 * f))
    const w3 = f * f * (-0.5 + 0.5 * f)
    return d.vec3f(w0, w1 + w2, w3)
})

/** WGSL identifier overrides for {@link bSplineTapSetup}'s emitted fn and struct. */
export interface BSplineTapNames {
    /** WGSL name of the setup fn. */
    fnName?: string
    /** WGSL name of the returned struct. */
    structName?: string
}

const bSplineTapCache = new Map<string, ReturnType<typeof buildBSplineTapSetup>>()

function buildBSplineTapSetup(gridSize: number, name: string, structName: string) {
    const Taps = d
        .struct({uvA: d.vec2f, uvB: d.vec2f, uvC: d.vec2f, uvD: d.vec2f, w: d.vec4f})
        .$name(structName)
    const setup = tgpu.fn([d.vec2f], Taps)((uv) => {
        'use gpu'
        const gf = d.f32(gridSize)
        const tx = uv.x * gf - 0.5
        const ty = uv.y * gf - 0.5
        const pxf = std.floor(tx)
        const pyf = std.floor(ty)
        const fx = tx - pxf
        const fy = ty - pyf
        // Cubic B-spline weights: ((1−t)³, 3t³−6t²+4, −3t³+3t²+3t+1, t³)/6.
        const ofx = 1.0 - fx
        const wx0 = ofx * ofx * ofx * 0.16666667
        const wx1 = fx * fx * fx * 0.5 - fx * fx + 0.66666667
        const wx3 = fx * fx * fx * 0.16666667
        const wx2 = 1.0 - wx0 - wx1 - wx3
        const ofy = 1.0 - fy
        const wy0 = ofy * ofy * ofy * 0.16666667
        const wy1 = fy * fy * fy * 0.5 - fy * fy + 0.66666667
        const wy3 = fy * fy * fy * 0.16666667
        const wy2 = 1.0 - wy0 - wy1 - wy3
        // Collapse into two bilinear taps per axis (g0/g1 ≥ 1/6, division is safe).
        const g0x = wx0 + wx1
        const g1x = wx2 + wx3
        const g0y = wy0 + wy1
        const g1y = wy2 + wy3
        const h0x = pxf - 1.0 + wx1 / g0x
        const h1x = pxf + 1.0 + wx3 / g1x
        const h0y = pyf - 1.0 + wy1 / g0y
        const h1y = pyf + 1.0 + wy3 / g1y
        return Taps({
            uvA: d.vec2f((h0x + 0.5) / gf, (h0y + 0.5) / gf),
            uvB: d.vec2f((h1x + 0.5) / gf, (h0y + 0.5) / gf),
            uvC: d.vec2f((h0x + 0.5) / gf, (h1y + 0.5) / gf),
            uvD: d.vec2f((h1x + 0.5) / gf, (h1y + 0.5) / gf),
            w: d.vec4f(g0x * g0y, g1x * g0y, g0x * g1y, g1x * g1y),
        })
    }).$name(name)
    return {Taps, setup}
}

/**
 * Cubic B-SPLINE tap setup for a square texture of `gridSize` texels, as four hardware-bilinear
 * taps plus their weights (weights sum to 1).
 *
 * Where Catmull-Rom above is an INTERPOLATING filter for reconstructing image detail, the B-spline
 * is an APPROXIMATING one: it does not pass through the texel values, it smooths them, and it is
 * C1 — the reason to reach for it is a low-resolution FIELD whose derivative is visible. Hardware
 * bilinear is only C0, so a displacement field sampled bilinearly kinks at every cell border and a
 * sharp edge dragged by the field renders each kink as a ~1-cell scallop. PixelThrow's flow grid is
 * the exemplar.
 *
 * The 4×4 cubic footprint collapses to 2 bilinear fetches per axis (offsets h0/h1 with weights
 * g0/g1) — the classic GPU Gems trick. Sampling is the CALLER's job: take the four tap UVs through
 * whatever sampler and edge mode it wants, and weight by `w.x/.y/.z/.w` in that order.
 *
 * This is a factory, not a plain fn, because `gridSize` folds into the body as a literal (C3): call
 * it once per grid size at module scope and reuse the result. Results are memoised per
 * `gridSize|fnName|structName`. Both names land verbatim in the WGSL, so a migrating shader passes
 * the names it already emitted and its snapshot does not move.
 */
export function bSplineTapSetup(gridSize: number, names: BSplineTapNames = {}) {
    const fnName = names.fnName ?? 'bSplineTapSetup'
    const structName = names.structName ?? 'BSplineTaps'
    const key = `${gridSize}|${fnName}|${structName}`
    const hit = bSplineTapCache.get(key)
    if (hit) return hit
    const built = buildBSplineTapSetup(gridSize, fnName, structName)
    bSplineTapCache.set(key, built)
    return built
}

/** Unique suffix per builder call so two bicubic samples in one pass get distinct locals. */
let bicubicCounter = 0

/**
 * Expr-level Catmull-Rom sample of `tex` at `uv` — a drop-in replacement for `tex.sample(uv)`
 * on the RTT path of a distortion that can scale content up.
 *
 * Costs 9 bilinear taps instead of 1, so use it where reconstruction quality is the point (a
 * magnifying distortion resampling composed content) and keep the plain bilinear sample
 * everywhere content is moved around at ~1:1.
 *
 * Taps reach up to 1.5 texels beyond `uv`, and they go through the SAME sampler as `uv` itself
 * — with `linearClamp` that means the footprint clamps at the texture border rather than
 * wrapping, which is why the callers below only use this for the clamp/transparent edge modes.
 *
 * `range` bounds the cubic's ringing. `'positive'` (default) only kills undershoot below zero,
 * which is all an HDR premultiplied RTT needs — overshoot there is the edge acutance we want, and
 * the tonemap deals with the range. `'unit'` also clips overshoot above 1, for an LDR source whose
 * channels are consumed as-is (a glyph atlas's alpha, where >1 would over-brighten the blend).
 */
export const sampleCatmullRomExpr = (
    tex: KitTexture,
    uv: Expr,
    sampler: SharedSamplerName = 'linearClamp',
    range: 'positive' | 'unit' = 'positive',
): Expr => {
    const id = bicubicCounter++
    return new ExprClass((ctx) =>
        ctx.memo(`catmullRom:${id}`, () => {
            const uvL = ctx.freshLocal('crUV')
            const sizeL = ctx.freshLocal('crSize')
            const pxL = ctx.freshLocal('crPosX')
            const pyL = ctx.freshLocal('crPosY')
            const wxL = ctx.freshLocal('crWx')
            const wyL = ctx.freshLocal('crWy')
            const outL = ctx.freshLocal('crOut')

            // Emit the UV / texture-size / per-axis position+weight locals ONCE, then reference
            // them from all nine taps.
            ctx.statement(`let ${uvL} = ${uv._emit(ctx)};`)
            ctx.statement(`let ${sizeL} = ${tex.dimensions()._emit(ctx)};`)
            const posFn = ctx.external(catmullRomAxisPositions, 'catmullRomAxisPositions')
            const wFn = ctx.external(catmullRomAxisWeights, 'catmullRomAxisWeights')
            ctx.statement(`let ${pxL} = ${posFn}(${uvL}.x, ${sizeL}.x);`)
            ctx.statement(`let ${pyL} = ${posFn}(${uvL}.y, ${sizeL}.y);`)
            ctx.statement(`let ${wxL} = ${wFn}(${uvL}.x, ${sizeL}.x);`)
            ctx.statement(`let ${wyL} = ${wFn}(${uvL}.y, ${sizeL}.y);`)

            ctx.statement(`var ${outL} = vec4f(0.0, 0.0, 0.0, 0.0);`)
            const axis = ['x', 'y', 'z'] as const
            for (const yi of [0, 1, 2]) {
                for (const xi of [0, 1, 2]) {
                    const tapUV = new ExprClass(
                        () => `vec2f(${pxL}.${axis[xi]}, ${pyL}.${axis[yi]})`,
                    )
                    const weight = `(${wxL}.${axis[xi]} * ${wyL}.${axis[yi]})`
                    ctx.statement(`${outL} += ${tex.sample(tapUV, sampler)._emit(ctx)} * ${weight};`)
                }
            }
            // Bound the cubic's ringing — see the `range` note above.
            ctx.statement(
                range === 'unit'
                    ? `${outL} = clamp(${outL}, vec4f(0.0, 0.0, 0.0, 0.0), vec4f(1.0, 1.0, 1.0, 1.0));`
                    : `${outL} = max(${outL}, vec4f(0.0, 0.0, 0.0, 0.0));`,
            )
            return outL
        }),
    )
}

/** Unique suffix per coverage-sample call so two in one pass get distinct locals. */
let coverageCounter = 0

/**
 * Glyph coverage (an f32, 0–1) from a single-channel atlas at `uv`, filtered for how big an
 * atlas texel is on screen — the sampler behind crisp text.
 *
 * - Denser than the screen (the normal case: glyph rasters are supersampled), the pixel covers
 *   several texels, and the right filter is their AREA: four bilinear taps spaced a quarter of
 *   the pixel's footprint apart average exactly the texels it covers (a box filter), so every
 *   edge gets its true coverage at any sub-pixel position or rotation.
 * - Magnified past one texel per pixel (a distortion scaling text up), the area is less than a
 *   texel and the job flips to reconstruction: the 9-tap Catmull-Rom above keeps edges acute.
 *
 * Everything reads mip level 0 explicitly (glyph atlases have no mips — an implicit-LOD sample
 * of a mipped supersampled raster drops two levels and throws the supersampling away), which
 * also makes it legal inside a branch: outside the glyph box (`inside` ≤ 0.5) no texel is read
 * at all, so a small label costs almost nothing on a large frame. The footprint is measured
 * with derivatives BEFORE the branch, where they're valid.
 */
export const sampleCoverageExpr = (
    tex: KitTexture,
    uv: Expr,
    inside: Expr,
    sampler: SharedSamplerName = 'linearClamp',
): Expr => {
    const id = coverageCounter++
    return new ExprClass((ctx) =>
        ctx.memo(`coverage:${id}`, () => {
            const uvL = ctx.freshLocal('covUV')
            const sizeL = ctx.freshLocal('covSize')
            const footL = ctx.freshLocal('covFootprint')
            const insideL = ctx.freshLocal('covInside')
            const outL = ctx.freshLocal('coverage')
            ctx.statement(`let ${uvL} = ${uv._emit(ctx)};`)
            ctx.statement(`let ${insideL} = ${inside._emit(ctx)};`)
            ctx.statement(`let ${sizeL} = ${tex.dimensions()._emit(ctx)};`)
            // Atlas texels per screen pixel, the larger of the two screen axes (rotation-proof)
            ctx.statement(`let ${footL} = max(length(dpdx(${uvL} * ${sizeL})), length(dpdy(${uvL} * ${sizeL})));`)
            ctx.statement(`var ${outL} = 0.0;`)
            const tap = (u: string) => `${tex.sampleLevel(new ExprClass(() => u), sampler)._emit(ctx)}.r`
            ctx.statement(`if (${insideL} > 0.5) {`)
            ctx.statement(`if (${footL} >= 1.0) {`)
            // Box: taps at ±¼ footprint (capped at 4 texels, past which a box this small aliases anyway)
            const offL = ctx.freshLocal('covOffset')
            ctx.statement(`let ${offL} = vec2f(min(${footL}, 4.0) * 0.25) / ${sizeL};`)
            // (WGSL has no unary plus: the positive taps name the offset bare)
            const taps = [['-', '-'], ['', '-'], ['-', ''], ['', '']]
                .map(([sx, sy]) => tap(`${uvL} + vec2f(${sx}${offL}.x, ${sy}${offL}.y)`))
            ctx.statement(`${outL} = 0.25 * (${taps.join(' + ')});`)
            ctx.statement(`} else {`)
            const pxL = ctx.freshLocal('covPosX')
            const pyL = ctx.freshLocal('covPosY')
            const wxL = ctx.freshLocal('covWx')
            const wyL = ctx.freshLocal('covWy')
            const posFn = ctx.external(catmullRomAxisPositions, 'catmullRomAxisPositions')
            const wFn = ctx.external(catmullRomAxisWeights, 'catmullRomAxisWeights')
            ctx.statement(`let ${pxL} = ${posFn}(${uvL}.x, ${sizeL}.x);`)
            ctx.statement(`let ${pyL} = ${posFn}(${uvL}.y, ${sizeL}.y);`)
            ctx.statement(`let ${wxL} = ${wFn}(${uvL}.x, ${sizeL}.x);`)
            ctx.statement(`let ${wyL} = ${wFn}(${uvL}.y, ${sizeL}.y);`)
            const axis = ['x', 'y', 'z'] as const
            const terms: string[] = []
            for (const yi of [0, 1, 2]) {
                for (const xi of [0, 1, 2]) {
                    terms.push(`${tap(`vec2f(${pxL}.${axis[xi]}, ${pyL}.${axis[yi]})`)} * (${wxL}.${axis[xi]} * ${wyL}.${axis[yi]})`)
                }
            }
            ctx.statement(`${outL} = clamp(${terms.join(' + ')}, 0.0, 1.0);`)
            ctx.statement(`}`)
            ctx.statement(`}`)
            return outL
        }),
    )
}
