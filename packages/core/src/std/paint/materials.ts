/**
 * std/paint/materials — words that shade a surface so a shape reads as glass, metal, plastic
 * or stone.
 *
 * A material is a normal, a light and a response. `shapedSurface` turns a shape prop into a
 * surface you can shade: it hands your `surface` function a frame holding the shape's distance
 * field and placement coordinates. From there a material reads as a recipe: a normal
 * (`geometricNormal`, then `nudgeNormal` for relief), a light (`keyLightAt`, `pointLightFrom`),
 * a response (`lambert`, `sharpGlint`, `schlickFresnel`) and a finish (`neutralTone`,
 * `silhouette`). The same words light any field with a slope, not only shapes: `fdSlope` of a
 * noise field feeds `nudgeNormal`, and `keyLightAt` with `lambert` shades it.
 */
// Maintainer notes — the shape-effect spine.
//
// Every SDF shape-effect material (Glass, Chrome, LiquidMetal, Plastic, Crystal, Frost, Water, …)
// used to repeat the same ~60–100 lines of plumbing: the `VOLUMETRIC_FIELD_EXTRA_FIELDS` +
// `ANALYTIC_SDF_EXTRA_FIELDS` spreads, the `createVolumetricFieldComputeNode` compute hook, the
// `resolveShapeFieldSampler` routing (compute-marched volumetric field → flat SVG → analytic),
// the `sdfSpaceUV` placement, and the neighbour-tap stencil. `shapedSurface` is that spine as
// ONE spread noun: the definition spreads its `{extraFields, compute, gpu}` halves
// (`...shapedSurface({...})` — the gaussianBlur/ComputeBackedEffect precedent) and supplies the
// MATERIAL as the `surface:` slot, which receives the resolved `SurfaceFrame` (placement UV,
// field taps, samplers, flags, child textures) and returns the shaded color Expr.
//
// Config axes are declared, not copy-pasted:
//  - `pattern` / `chord`      — forwarded to the volumetric pre-march AND the sampler routing,
//                               so the two paths can never disagree.
//  - `stencil`                — the field-tap frame the material reads: `'none'` (the material
//                               samples for itself — Glass/ThinFilm/Neon/Emboss), `'centre'`
//                               (placement + centre tap only — Heatmap/LightEdge), a forward
//                               2-neighbour stencil at a per-material eps (the metals), or the
//                               5-tap central stencil (Hologram).
//  - `placeUV`                — optional pre-placement displacement of the screen UV (Hologram's
//                               beam wobble); the default is `ctx.uv`.
//  - `child: 'required'`      — resolve the child RTT (compute-created when present, else
//                               `convertToTexture`) and return transparent without a child.
//
// Material-specific prepasses (Glass's frosted blur) are NOT axes here: a consumer wraps the
// returned `compute` half and merges its own compute node's outputs — the fragment picks the
// compute-created `childTexture` up automatically, and anything else rides `params.computeOutputs`.
//
// Sign convention throughout: the viewer looks along +z, so a normal that faces the viewer has
// negative z, and a light on the viewer's side has negative z (`keyLightAt` elevation −0.6 … −0.9).
// `light.shine`/`light.domeNormal` use the opposite (+z) convention — do not mix them with these
// words without flipping.
import type {EmitContext, GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {Expr} from '../../gpu/contract'
import {call, floatE, ZERO} from '../../gpu/composer'
import {effects, lighting, materialParts, noise, sdf, sdf3d, tonemap} from '../../gpu/kit/index'
import {
    add, clamp as clampE, div, dot as dotE, exp as expE, exp2, float, fract as fractE, length as lengthE, local, max as maxE,
    mix as mixE, mul, neg, normalize as normalizeE, pow as powE, sin as sinE, smoothstep as smoothstepE, splat3,
    sub, vec2 as vec2E, vec3 as vec3E,
} from '../math'
import type {DirectionFrame} from '../frames'

const {sdfSpaceUV, offsetUV} = effects.glass
const {ANALYTIC_SDF_EXTRA_FIELDS} = sdf
const {createVolumetricFieldComputeNode, resolveShapeFieldSampler, VOLUMETRIC_FIELD_EXTRA_FIELDS} = sdf3d

// ─── Conditional region ───────────────────────────────────────────────────────────────────────
//
// Expr algebra is pure expressions — it cannot express the `if (outside) return transparent`
// early-exit the material bodies rely on for their perf (a shape usually covers a fraction of the
// canvas). `guarded` restores that as vocabulary: a REAL WGSL branch whose interior statements
// (including every `local()` hoist emitted while serialising `inner`) land inside the branch, so
// pixels failing `cond` pay only the condition. NOTE: promote-to-math candidate — this is
// universal Expr flow control, not a materials concern.

let guardCounter = 0

/**
 * Evaluates an expression only where a condition holds, and a cheap fallback everywhere else.
 *
 * Use it to skip a material's whole body for pixels outside the shape. Any value you build
 * inside `inner` and also read after the guard must be listed in `deps`.
 *
 * @example
 * ```ts
 * return guarded(insideShape(field.sdf, field.pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), vec4(0, 0, 0, 0), 'plastic')
 * ```
 * @tip Values first made inside the guard are not visible after it. Put shared locals in `deps`.
 * @see insideShape, silhouette
 */
export function guarded(cond: Expr, inner: Expr, fallback: Expr, hint = 'guarded', deps: Expr[] = []): Expr {
    // Emits `var x = fallback; if (cond) { …inner hoists…; x = inner; }`. Values hoisted BEFORE
    // the guard (emitted by `cond`/`fallback`, or by earlier expressions) stay in the enclosing
    // scope and remain visible inside; values first used inside are declared inside and must not
    // be re-read outside. A `local()` shared between the guarded interior and any later
    // expression MUST be listed in `deps` — that emits it in the enclosing scope first (a hoist
    // whose first use is inside the branch would otherwise be block-scoped and unreachable
    // afterwards).
    const id = guardCounter++
    return new Expr((ctx) =>
        ctx.memo(`guard:${id}`, () => {
            const name = ctx.freshLocal(hint)
            const condText = cond._emit(ctx)
            const fallbackText = fallback._emit(ctx)
            for (const dep of deps) dep._emit(ctx)
            const stmts: string[] = []
            const scoped: EmitContext = {
                external: (v, h) => ctx.external(v, h),
                statement: (w) => stmts.push(w),
                freshLocal: (h) => ctx.freshLocal(h),
                memo: (k, f) => ctx.memo(k, f),
            }
            const innerText = inner._emit(scoped)
            ctx.statement(`var ${name} = ${fallbackText};`)
            ctx.statement(`if (${condText}) {\n  ${stmts.join('\n  ')}\n  ${name} = ${innerText};\n}`)
            return name
        }),
    )
}

// reflect/refract/exp2 PROMOTED to std/math (2026-09-01) — re-exported here so existing
// material recipes keep their import surface.
export {reflect, refract, exp2} from '../math'

/**
 * True inside the shape and up to two pixels outside it.
 *
 * Pass the signed distance and pixel size from `surfaceField`. The two-pixel margin keeps the
 * anti-aliased edge inside the shaded region. The usual condition for `guarded`.
 *
 * @example
 * ```ts
 * guarded(insideShape(field.sdf, field.pxH), shaded, vec4(0, 0, 0, 0))
 * ```
 * @see guarded, silhouette
 */
export function insideShape(sdf: Expr, pxH: Expr): Expr {
    // The exact complement of the kit's `outsideShape` early-exit test.
    const outside = call(lighting.outsideShape, 'outsideShape', [sdf, pxH])
    return new Expr((ctx) => `(!(${outside._emit(ctx)}))`)
}

// Maintainer notes on SurfaceFrame members: `surf0.x` on 3D shapes is −chord/2 (minus half the
// depth the view passes through), not a 2D distance. `gradSampler` is bilinear on the volumetric
// path unless `gradSampler: 'same'`. `texelSampler` serves the voxel G-buffer's face/cell channels
// and aliases `sampler` on the flat paths. `childTexture` is compute-created when a consumer
// prepass made one (Glass's frosted blur).
/**
 * What `shapedSurface` hands your `surface` function: the shape's distance field, sampled and
 * ready to shade.
 *
 * Read the taps through `surfaceField` rather than directly. `volumetric` tells you at build
 * time whether the shape is a 3D one (a sphere, a cube, an extruded SVG) or a flat one.
 */
export interface SurfaceFrame {
    /** Placement coordinates: the shape's own 0–1 square after center, scale and rotation. Absent with `stencil: 'none'`. */
    sdfUV?: Expr
    /** The field at this pixel. `.x` is the signed distance (negative inside), `.y/.z` pattern coordinates, `.w` depth. */
    surf0?: Expr
    /** The field one step right and one step down, when the stencil took them. */
    surfX?: Expr
    surfY?: Expr
    /** The field one step left and one step up. Only with `stencil: {kind: 'central'}`. */
    surfL?: Expr
    surfD?: Expr
    /** Sample the field at any placement coordinate, at full quality. */
    sampler: (uv: Expr) => Expr
    /** Sample the field at any placement coordinate, cheaply. Use it for neighbour taps. */
    gradSampler: (uv: Expr) => Expr
    /** Sample one texel without filtering, for discrete data packed into the field. */
    texelSampler: (uv: Expr) => Expr
    /** True when the shape is 3D or extruded. Known at build time, so you can branch in TypeScript. */
    volumetric: boolean
    /** `volumetric` as a 0/1 number, for use inside an expression. */
    volFlag: Expr
    /** True when `.y/.z` already hold the field's gradients instead of pattern coordinates. Skip your own slope taps. */
    bakedGradients: boolean
    /** The layer inside, as a texture. Present with `child: 'required'`. */
    childTexture?: KitTexture
}

// Maintainer notes on ShapedSurfaceSpec: `bakedGradients` routing is `ShapeFieldSamplerOptions` in
// kit/sdf3d. `centreTap: 'fast'` exists because the metals' bicubic-per-pixel is 16 texel loads at
// every canvas pixel on a 3D shape.
/** The options `shapedSurface` takes. Only `surface` is required. */
export interface ShapedSurfaceSpec {
    /** Bake surface coordinates for 3D shapes so a pattern sticks to them (default 'none'). 'raw' follows the surface without seams, 'triplanar' projects from three sides. */
    pattern?: 'none' | 'raw' | 'triplanar'
    /** How thickness is measured through a 3D shape (default 'span'). 'firstLobe' measures the front wall only. */
    chord?: 'span' | 'firstLobe'
    /** Quality of the neighbour taps (default 'fast'). 'same' uses the full-quality sampler for them too. */
    gradSampler?: 'fast' | 'same'
    /** Let the field carry its own gradients in `.y/.z` (default 'none'). Check `frame.bakedGradients` before taking slopes. */
    bakedGradients?: 'none' | 'volumetric' | 'all'
    /** Which taps to take for you (default a right-and-down pair, `eps` 0.01 in placement units). 'none' takes none, 'centre' this pixel only, 'central' adds left and up. */
    stencil?: 'none' | 'centre' | {kind: 'forward' | 'central'; eps: number}
    /**
     * Which sampler takes the centre tap (default 'quality'). 'fast' is cheaper on 3D shapes
     * when the centre tap only gates coverage and the material takes its own quality tap
     * where it needs one.
     */
    centreTap?: 'quality' | 'fast'
    /** Displace the canvas uv before placement, to wobble or warp the whole shape. */
    placeUV?: (params: GpuFragmentParams) => Expr
    /** Require the layer inside and expose it as `frame.childTexture`. Without a child the effect draws nothing. */
    child?: 'required'
    /** Your material: shade the frame and return the color at this pixel. */
    surface: (frame: SurfaceFrame, params: GpuFragmentParams) => Expr
}

/** The three fields `shapedSurface` returns. Spread them into your definition. */
export interface ShapedSurfaceEffect {
    extraFields: Record<string, {schema: import('typegpu/data').AnyWgslData; initial: number | number[]}>
    compute: GpuComputeNode
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/**
 * Turns a shape prop into a surface your material shades.
 *
 * Spread the result into a definition that declares the `center`, `scale`, `rotation` and
 * `shape` props (copy them from any shape effect). Your `surface` function receives a
 * `SurfaceFrame` and returns the color at this pixel. Flat shapes, SVG shapes and 3D shapes
 * all arrive through the same frame.
 *
 * @example
 * ```ts
 * ...shapedSurface({
 *   pattern: 'raw',
 *   surface: (frame, params) => {
 *     const u = params.uniforms
 *     const field = surfaceField(frame, params, {scale: u.scale})
 *     const n = geometricNormal(frame, field, {bevelWidth: u.bevelWidth, bevelShape: u.bevelShape})
 *     const key = keyLightAt(direction(u.lightAngle), -0.7)
 *     const rgb = mul(u.color.member('rgb'), lambert(n, key.member('L'), {wrap: 0.3}))
 *     return guarded(insideShape(field.sdf, field.pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), vec4(0, 0, 0, 0))
 *   },
 * }),
 * ```
 * @tip Start from `surfaceField`, end with `guarded` around `insideShape`. Everything between is the look.
 * @see surfaceField, geometricNormal, silhouette, guarded
 */
export function shapedSurface(spec: ShapedSurfaceSpec): ShapedSurfaceEffect {
    const pattern = spec.pattern ?? 'none'
    const chord = spec.chord ?? 'span'
    const stencil = spec.stencil ?? {kind: 'forward' as const, eps: 0.01}

    // Volumetric SDF pre-march (3D / SVG-extrude shapes) — dirty-keyed; flat shapes and
    // GPU-free resolve return null (the fragment falls back to the flat-SVG/analytic sampler).
    const compute: GpuComputeNode = (params) =>
        createVolumetricFieldComputeNode(params, () => params.getCpuValue('shape'), pattern, chord)

    const fragment = (params: GpuFragmentParams): Expr => {
        const {childNode, computeOutputs, convertToTexture, uniforms, ctx} = params

        let childTexture: KitTexture | undefined
        if (spec.child === 'required') {
            if (!childNode) return ZERO
            childTexture = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
        }

        // Sampler routing — the SAME pattern mode as the pre-march, by construction.
        const {sampler, gradSampler, texelSampler, volumetric, bakedGradients} = resolveShapeFieldSampler(params, {
            patternMode: pattern,
            gradSampler: spec.gradSampler ?? 'fast',
            bakedGradients: spec.bakedGradients ?? 'none',
        })

        const frame: SurfaceFrame = {
            sampler, gradSampler, texelSampler, volumetric, bakedGradients,
            volFlag: floatE(volumetric ? 1 : 0),
            childTexture,
        }

        if (stencil !== 'none') {
            const screenUV = spec.placeUV ? spec.placeUV(params) : ctx.uv
            frame.sdfUV = local(call(sdfSpaceUV, 'sdfSpaceUV', [uniforms.center, uniforms.scale, uniforms.rotation, screenUV, ctx.aspect]), 'sdfUV')
            frame.surf0 = (spec.centreTap === 'fast' ? gradSampler : sampler)(frame.sdfUV)
            if (stencil !== 'centre') {
                const {kind, eps} = stencil
                frame.surfX = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(eps), floatE(0)]))
                if (kind === 'central') {
                    frame.surfL = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(-eps), floatE(0)]))
                }
                frame.surfY = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(0), floatE(eps)]))
                if (kind === 'central') {
                    frame.surfD = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(0), floatE(-eps)]))
                }
            }
        }

        return spec.surface(frame, params)
    }

    return {
        extraFields: {...VOLUMETRIC_FIELD_EXTRA_FIELDS, ...ANALYTIC_SDF_EXTRA_FIELDS},
        compute,
        gpu: {fragment},
    }
}

// ─── Material vocabulary ────────────────────────────────────────────────────────────────────────
//
// The shared anatomy of a shaded shape-effect material, as words. A material file should read as
// a recipe over these — field basics → geometric normal → (relief → tilt) → view → lighting →
// tint — with only its own constants and composition at the declaration site.

// `sdf` is the centre tap's `.x`, divided by `scale` when `surfaceField` was given one.
/** The field values a material reads, bound once: the taps, the signed distance and the pixel size. */
export interface SurfaceField {
    /** The field at this pixel, and one step right and one step down when the stencil took them. */
    s0: Expr
    sX?: Expr
    sY?: Expr
    /** Signed distance to the shape's edge: negative inside, zero on the edge. */
    sdf: Expr
    /** One pixel, as a fraction of the canvas height. */
    pxH: Expr
    /** Canvas width divided by height. */
    aspect: Expr
}

/**
 * The first line of a material: the field taps, the signed distance and the pixel size, bound once.
 *
 * Pass `scale` (the shape's scale prop) to divide the distance by it, as the library's metals
 * and plastics do, so edge softness and pixel tests track the shape's size. Leave it out for
 * `continuedField`.
 *
 * @example
 * ```ts
 * const field = surfaceField(frame, params, {scale: params.uniforms.scale})
 * ```
 * @see shapedSurface, geometricNormal, silhouette, insideShape
 */
export function surfaceField(
    frame: SurfaceFrame,
    params: GpuFragmentParams,
    opts: {scale?: Expr} = {},
): SurfaceField {
    const {ctx} = params
    const s0 = local(frame.surf0!, 's0')
    const pxH = local(div(1, ctx.viewportSize.member('y')), 'pxH')
    const sdf = local(opts.scale ? div(s0.member('x'), opts.scale) : s0.member('x'), 'sdf')
    const aspect = div(ctx.viewportSize.member('x'), ctx.viewportSize.member('y'))
    return {
        s0,
        sX: frame.surfX ? local(frame.surfX, 'sX') : undefined,
        sY: frame.surfY ? local(frame.surfY, 'sY') : undefined,
        sdf,
        pxH,
        aspect,
    }
}

/**
 * The surface normal of the shape, whichever kind of shape is active.
 *
 * A 3D shape takes its normal from the marched depth. A flat shape takes a normal that tilts
 * away from the face near the edge, over `bevelWidth` (in the units of `field.sdf`) with the
 * profile `bevelShape` (0 round, 1 machined). `sharpness` clamps the 3D slope (default 5, lower
 * is calmer). `eps` must match the stencil step (default 0.01). Needs a stencil with neighbour taps.
 *
 * @example
 * ```ts
 * const n = geometricNormal(frame, field, {bevelWidth: u.bevelWidth, bevelShape: u.bevelShape})
 * ```
 * @tip Add relief with `nudgeNormal` or `tiltNormal` after this, never before.
 * @see marchedNormal, nudgeNormal, tiltNormal, fieldSlope
 */
export function geometricNormal(
    frame: SurfaceFrame,
    field: SurfaceField,
    opts: {bevelWidth: Expr; bevelShape: Expr; sharpness?: number; eps?: number; hint?: string},
): Expr {
    // Resolved at build time: marched volumetric shapes take the field-tap normal; flat shapes
    // take the gradient normal tilted by the bevel profile.
    const eps = float(opts.eps ?? 0.01)
    const hint = opts.hint ?? 'nGeo'
    if (frame.volumetric) {
        return local(
            call(lighting.volumetricNormal, 'volumetricNormal', [field.s0, field.sX!, field.sY!, eps, float(opts.sharpness ?? 5)]),
            hint,
        )
    }
    const rimT = clampE(div(neg(field.sdf), maxE(opts.bevelWidth, 0.003)), 0, 1)
    return local(
        call(lighting.bevelledFlatNormal, 'bevelledFlatNormal', [
            call(lighting.fieldGradient, 'fieldGradient', [field.s0, field.sX!, field.sY!, eps]),
            call(effects.bevel.bevelSin, 'bevelSin', [rimT, opts.bevelShape]),
        ]),
        hint,
    )
}

/**
 * Coordinates for a pattern that sticks to the shape's surface.
 *
 * On a flat shape they are the placement coordinates centred on the shape (0,0 at its centre).
 * On a 3D shape they follow the surface as it rotates, when the effect declares
 * `pattern: 'raw'` or `'triplanar'`.
 *
 * @example
 * ```ts
 * const pat = surfacePattern(frame, field)
 * ```
 * @see shapedSurface, rotationSensor, surfaceNoiseAt
 */
export function surfacePattern(frame: SurfaceFrame, field: SurfaceField, opts: {uv?: Expr} = {}): Expr {
    return local(call(lighting.patternCoords, 'patternCoords', [opts.uv ?? frame.sdfUV!, field.s0, frame.volFlag]), 'pat')
}

/**
 * The direction the shape's distance grows in: a 2D vector pointing outward at this pixel.
 *
 * Its length is about 1 on a flat shape. `eps` is the tap step (default 0.01) and must match
 * the stencil step.
 *
 * @example
 * ```ts
 * const grad = fieldSlope(field)
 * ```
 * @see nearestEdge, geometricNormal, tiltAlong
 */
export function fieldSlope(field: SurfaceField, eps = 0.01): Expr {
    return local(call(lighting.fieldGradient, 'fieldGradient', [field.s0, field.sX!, field.sY!, float(eps)]), 'grad')
}

/**
 * The direction the viewer looks along at this pixel, for reflections that sweep across a face.
 *
 * A unit 3D vector pointing into the canvas. `fov` sets how much it leans off-axis toward
 * the edges (0.6 is the usual value, 0.55 flatter).
 *
 * @example
 * ```ts
 * const view = viewRay(params, field)
 * ```
 * @see grazingOf, reflect, studioSoftboxes
 */
export function viewRay(params: GpuFragmentParams, field: SurfaceField, fov = 0.6): Expr {
    return local(call(lighting.perspectiveViewRay, 'perspectiveViewRay', [params.ctx.uv, field.aspect, float(fov)]), 'viewI')
}

/**
 * How edge-on the surface is to the viewer: 0 face-on, 1 grazing.
 *
 * The input the fresnel words take when you have a view ray. Without one, use `grazingFlat`.
 *
 * @example
 * ```ts
 * const grazing = grazingOf(n, viewRay(params, field))
 * ```
 * @see grazingFlat, fresnelBoost, schlickFresnel
 */
export function grazingOf(n: Expr, view: Expr): Expr {
    return local(sub(1, clampE(neg(dotE(n, view)), 0, 1)), 'grazing')
}

/**
 * Tilts a normal along and across a direction, for brushed or woven relief.
 *
 * `frame` is a `directionFrame` (from `frames`). `along` and `across` are slope amounts; a
 * few tenths is a visible relief.
 *
 * @example
 * ```ts
 * const n = tiltNormal(nGeo, brush, {across: reliefAcross, along: reliefAlong})
 * ```
 * @see nudgeNormal, tiltAlong, grainNoise
 */
export function tiltNormal(n: Expr, frame: DirectionFrame, slopes: {along?: Expr; across?: Expr}, hint = 'n'): Expr {
    const terms = (axis: 'x' | 'y'): Expr[] => {
        const out: Expr[] = []
        if (slopes.across) out.push(mul(frame.perp.member(axis), slopes.across))
        if (slopes.along) out.push(mul(frame.tangent.member(axis), slopes.along))
        return out
    }
    const sum = (base: Expr, extra: Expr[]): Expr => extra.reduce((acc, e) => add(acc, e), base)
    return local(
        normalizeE(vec3E(sum(n.member('x'), terms('x')), sum(n.member('y'), terms('y')), n.member('z'))),
        hint,
    )
}

/**
 * The brightness of a five-light studio seen in a reflected direction.
 *
 * Pass the x and y of a reflected view ray. `rotation` is a `direction()` vector that turns
 * the studio. `keyRadius` sizes the main light (0.16 crisp, 0.42 soft), `drift` slides it,
 * `strength` scales the whole bank and `skyGain` the floor-to-sky gradient behind it (0.3–0.4).
 *
 * @example
 * ```ts
 * const env = studioSoftboxes(R.member('x'), R.member('y'), {rotation: direction(u.envRotation), keyRadius: 0.3, drift: 0, strength: u.environment, skyGain: 0.38})
 * ```
 * @tip Smear it with `smearAlong` for a brushed look.
 * @see smearAlong, reflect, viewRay
 */
export function studioSoftboxes(x: Expr, y: Expr, opts: {
    /** A `direction()` vector that rotates the studio. */
    rotation: Expr
    keyRadius: Expr | number
    drift: Expr | number
    strength: Expr | number
    skyGain: number
}): Expr {
    // BrushedMetal smears it (skyGain 0.38, animated drift); CarbonFiber takes one crisp tap
    // (skyGain 0.3, keyRad 0.16, drift 0). The five banks are shared; only skyGain differs.
    const toE = (v: Expr | number): Expr => (typeof v === 'number' ? float(v) : v)
    return call(materialParts.studio5Softbox, 'studio5Softbox', [
        x, y, opts.rotation.member('x'), opts.rotation.member('y'),
        toE(opts.keyRadius), toE(opts.drift), toE(opts.strength), float(opts.skyGain),
    ])
}

/**
 * One octave of smooth signed noise along and across a direction, for grain.
 *
 * `coords` come from `directionFrame(...).coordsOf(p)`. `freq` sets the feature density per
 * axis (a high across value gives fine streaks). Returns roughly −1 to 1. Sum octaves and
 * apply gains at the call site.
 *
 * @example
 * ```ts
 * const sheen = grainNoise(brush.coordsOf(pat), {freq: [1.3, 6.5], drift: driftA})
 * ```
 * @see surfaceNoise, tiltNormal, smearAlong
 */
export function grainNoise(
    coords: {along: Expr; across: Expr},
    opts: {freq: [Expr | number, Expr | number]; offset?: [number, number]; drift?: Expr},
): Expr {
    let x: Expr = mul(coords.along, opts.freq[0])
    let y: Expr = mul(coords.across, opts.freq[1])
    if (opts.offset?.[0]) x = add(x, opts.offset[0])
    if (opts.offset?.[1]) y = add(y, opts.offset[1])
    if (opts.drift) x = add(x, opts.drift)
    return call(noise.mxNoiseFloat2, 'mxNoiseFloat2', [vec2E(x, y)])
}

/**
 * Blurs any 2D lookup along a direction, for streaked reflections and motion smear.
 *
 * Samples `taps` points (default 9) spread `spread` either side of `center` along the unit
 * vector `axis`, weighted by a bell curve, and sums them. `sample` receives an x and a y.
 *
 * @example
 * ```ts
 * const env = smearAlong({center: R, axis: brush.tangent, spread: 0.3}, (x, y) => studioSoftboxes(x, y, studio))
 * ```
 * @see studioSoftboxes, grainNoise
 */
export function smearAlong(
    opts: {center: Expr; axis: Expr; spread: Expr; taps?: number; sigma?: number; hint?: string},
    sample: (x: Expr, y: Expr) => Expr,
): Expr {
    // Gaussian weights baked at build time and normalized; `sigma` in taps.
    const taps = opts.taps ?? 9
    const sigma = opts.sigma ?? 2.1
    const half = (taps - 1) / 2
    const raw = Array.from({length: taps}, (_, i) => Math.exp(-((i - half) ** 2) / (2 * sigma * sigma)))
    const sum = raw.reduce((a, b) => a + b, 0)
    const weights = raw.map((w) => w / sum)

    const step = local(div(opts.spread, half), 'step')
    let acc: Expr = float(0)
    for (let i = 0; i < taps; i++) {
        const off = local(mul(step, float(i - half)), 'off')
        acc = add(acc, mul(
            sample(
                add(opts.center.member('x'), mul(opts.axis.member('x'), off)),
                add(opts.center.member('y'), mul(opts.axis.member('y'), off)),
            ),
            float(weights[i]),
        ))
    }
    return local(acc, opts.hint ?? 'smear')
}

/**
 * A 3D light direction from a `direction()` vector and an elevation.
 *
 * `z` is negative for a light on the viewer's side of the surface (−0.6 raking to −0.9
 * frontal). Use it where a word wants a unit light vector. `keyLightAt` also gives the half
 * vector for highlights.
 *
 * @example
 * ```ts
 * const L = lightVec3(direction(u.lightAngle), -0.7)
 * ```
 * @see keyLightAt, anisoSpecular, lambert
 */
export function lightVec3(dir: Expr, z: number, hint = 'L'): Expr {
    return local(normalizeE(vec3E(dir.member('x'), dir.member('y'), float(z))), hint)
}

/**
 * Turns roughness and anisotropy sliders (0–1 each) into the two lobe widths `anisoSpecular` takes.
 *
 * Higher roughness widens the highlight. Higher anisotropy stretches it along the grain and
 * squeezes it across.
 *
 * @example
 * ```ts
 * alphas: wardAlphas(u.roughness, u.anisotropy)
 * ```
 * @see anisoSpecular
 */
export function wardAlphas(roughness: Expr, anisotropy: Expr): {along: Expr; across: Expr} {
    // The house mapping: base width mix(0.05, 0.45, roughness); along ×(1 + 7·aniso), across
    // ×(1 − 0.55·aniso) floored at 0.02.
    const baseA = local(mixE(0.05, 0.45, roughness), 'baseA')
    return {
        along: mul(baseA, add(1, mul(anisotropy, 7))),
        across: maxE(mul(baseA, sub(1, mul(anisotropy, 0.55))), 0.02),
    }
}

/**
 * A highlight stretched along a grain, as on brushed metal or carbon fibre.
 *
 * `tangent` is the 2D grain direction, `light` a unit 3D light vector, `view` the view ray
 * and `alphas` the widths from `wardAlphas`. Pass `gain` or scale the result yourself.
 *
 * @example
 * ```ts
 * const spec = anisoSpecular({normal: n, tangent: brush.tangent, light: lightVec3(direction(u.lightAngle), -0.7), view, alphas: wardAlphas(rough, aniso), gain: u.specular})
 * ```
 * @see wardAlphas, lightVec3, sharpGlint, dualLobeGlint
 */
export function anisoSpecular(opts: {
    normal: Expr
    /** The 2D grain direction. */
    tangent: Expr
    /** A unit 3D light direction (see `lightVec3`). */
    light: Expr
    view: Expr
    alphas: {along: Expr; across: Expr}
    gain?: Expr | number
}): Expr {
    // Ward anisotropic lobe, unit gain, stretched along `tangent`.
    const spec = call(lighting.wardAnisotropicSpecular, 'wardAnisotropicSpecular', [
        call(lighting.WardSpecularInput, 'WardSpecularInput', [
            opts.normal,
            vec3E(opts.tangent.member('x'), opts.tangent.member('y'), 0),
            opts.light,
            opts.view,
            opts.alphas.along,
            opts.alphas.across,
        ]),
    ])
    return opts.gain === undefined ? spec : mul(spec, opts.gain)
}

/**
 * Tilts a normal along one 2D direction by a slope amount.
 *
 * `axis` is a unit 2D vector such as `direction()` or `fieldSlope`. A few tenths is a
 * visible tilt.
 *
 * @example
 * ```ts
 * const n = tiltAlong(nGeo, weaveDir, mul(crown, 0.3))
 * ```
 * @see tiltNormal, nudgeNormal
 */
export function tiltAlong(n: Expr, axis: Expr, amount: Expr, hint = 'n'): Expr {
    return local(normalizeE(vec3E(
        add(n.member('x'), mul(axis.member('x'), amount)),
        add(n.member('y'), mul(axis.member('y'), amount)),
        n.member('z'),
    )), hint)
}

/**
 * A multiplier that brightens grazing edges: 1 face-on, up to `1 + amount` at the rim.
 *
 * `grazing` comes from `grazingOf` or `grazingFlat`. `power` (default 3) sets how tightly
 * the lift hugs the rim.
 *
 * @example
 * ```ts
 * const lit = mul(env, fresnelBoost(grazingOf(n, view), {amount: 0.6}))
 * ```
 * @see schlickFresnel, grazingOf, grazingFlat
 */
export function fresnelBoost(grazing: Expr, opts: {power?: number; amount: Expr | number}): Expr {
    // `1 + grazing^power · amount`.
    return add(1, mul(powE(grazing, float(opts.power ?? 3)), opts.amount))
}

/**
 * Maps a lighting value (0 dark, 1 light) onto two colors.
 *
 * Takes color props and returns rgb. The final tint of a material lit in grey.
 *
 * @example
 * ```ts
 * const rgb = tintRamp(lit, u.darkColor, u.lightColor)
 * ```
 * @see neutralTone, cosineRainbow
 */
export function tintRamp(t: Expr, dark: Expr, light: Expr): Expr {
    return mixE(dark.member('rgb'), light.member('rgb'), splat3(t))
}

/**
 * The shape's coverage at this pixel: 1 inside, 0 outside, soft across the edge.
 *
 * `edgeSoftness` is the softness prop; at 0 the edge is still anti-aliased over 1.5 pixels.
 * Use it as the alpha of the color you return.
 *
 * @example
 * ```ts
 * return guarded(insideShape(field.sdf, field.pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), vec4(0, 0, 0, 0))
 * ```
 * @see insideShape, guarded
 */
export function silhouette(field: SurfaceField, edgeSoftness: Expr): Expr {
    return call(materialParts.silhouetteAlpha, 'silhouetteAlpha', [field.sdf, edgeSoftness, field.pxH])
}

/**
 * Compresses bright color back into range without shifting its hue.
 *
 * The last step of a metal or glass look that adds reflection on top of a base color.
 * `exposureTone` is the cousin for emitted light.
 *
 * @example
 * ```ts
 * const rgb = neutralTone(mul(env, u.tint.member('rgb')))
 * ```
 * @see exposureTone
 */
export function neutralTone(rgb: Expr): Expr {
    // Khronos neutral filmic shoulder.
    return call(tonemap.neutral, 'tonemapNeutral', [rgb])
}

/**
 * Tilts a normal by a slope in x and y, turning a height field into relief.
 *
 * Feed it the `dx`/`dy` of `fdSlope` or `perlinSlope`, scaled by how strong you want the
 * relief. Works on a flat `vec3(0, 0, -1)` normal as well as a shape's normal.
 *
 * @example
 * ```ts
 * const n = nudgeNormal(vec3(0, 0, -1), mul(relief.dx, 0.08), mul(relief.dy, 0.08))
 * ```
 * @see fdSlope, perlinSlope, tiltNormal, geometricNormal
 */
export function nudgeNormal(n: Expr, dx: Expr, dy: Expr, hint = 'n'): Expr {
    // `normalize(n.x + dx, n.y + dy, n.z)`. Callers fold their relief into ONE dx/dy each; a
    // multi-term sum re-associated through this add would change float rounding.
    return local(call(materialParts.nudgeNormal, 'nudgeNormal', [n, dx, dy]), hint)
}

/**
 * The surface normal of a 3D shape, from the marched depth.
 *
 * Only meaningful when `frame.volumetric` is true. `geometricNormal` chooses between this
 * and the flat-shape normal for you. `sharpness` clamps the slope (default 5; 4 is calmer,
 * 6 crisper).
 *
 * @example
 * ```ts
 * const n = frame.volumetric ? marchedNormal(field, 4) : flatNormal
 * ```
 * @see geometricNormal
 */
export function marchedNormal(field: SurfaceField, sharpness = 5, hint = 'nGeo'): Expr {
    return local(call(lighting.volumetricNormal, 'volumetricNormal', [field.s0, field.sX!, field.sY!, float(0.01), float(sharpness)]), hint)
}

/**
 * A key light from an angle: the light direction `.L` and the half vector `.H` for highlights.
 *
 * `dir` is a `direction()` vector (from `frames`). `elevation` is how far the light sits
 * toward the viewer: −0.6 raking, −0.9 frontal. Read `.L` with `member('L')` for `lambert`
 * and `.H` for a glint.
 *
 * @example
 * ```ts
 * const key = keyLightAt(direction(u.lightAngle), -0.7)
 * ```
 * @tip Elevation is negative in this engine. A positive value lights the surface from behind.
 * @see keyLightXY, lightVec3, lambert, dualLobeGlint, sharpGlint
 */
export function keyLightAt(dir: Expr, elevation: number, hint = 'kl'): Expr {
    // Each material picks its own eye-tuned elevation (the fleet uses −0.6 … −0.9).
    return keyLightXY(dir.member('x'), dir.member('y'), elevation, hint)
}

/**
 * A key light from explicit x and y components, for a second or a crossed light.
 *
 * Same result as `keyLightAt`: `.L` and `.H`.
 *
 * @example
 * ```ts
 * const fill = keyLightXY(mul(ly, -1), lx, -0.4, 'fill')
 * ```
 * @see keyLightAt
 */
export function keyLightXY(x: Expr, y: Expr, elevation: number, hint = 'kl'): Expr {
    return local(call(materialParts.keyLight, 'keyLight', [x, y, float(elevation)]), hint)
}

/**
 * How edge-on the surface is to a straight-on viewer: 0 face-on, 1 grazing.
 *
 * Use it when the material has no `viewRay`. The fresnel input for plastics, frost and water.
 *
 * @example
 * ```ts
 * const grazing = grazingFlat(n)
 * ```
 * @see grazingOf, schlickFresnel, fresnelBoost
 */
export function grazingFlat(n: Expr): Expr {
    return local(call(materialParts.grazingView, 'grazingView', [n]), 'grazing')
}

/**
 * A highlight whose tightness follows a sharpness slider.
 *
 * `ndh` is the dot of the normal and the key light's `.H`. `sharpness` runs 0–1 and slides
 * the exponent from `cfg.x` to `cfg.y` and the gain from `cfg.z` to `cfg.w`. Pass `cfg` as
 * `vec4(soft exponent, sharp exponent, soft gain, sharp gain)`.
 *
 * @example
 * ```ts
 * const spec = sharpGlint(dot(n, key.member('H')), u.sharpness, vec4(60, 900, 0.4, 1.4))
 * ```
 * @see dualLobeGlint, anisoSpecular, keyLightAt
 */
export function sharpGlint(ndh: Expr, sharpness: Expr, cfg: Expr): Expr {
    // LiquidMetal passes (60, 900, 0.4, 1.4), Water (40, 600, 0.3, 1.3).
    return call(materialParts.sharpGlint, 'sharpGlint', [ndh, sharpness, cfg])
}

/**
 * The slope of smooth noise at a point, for wavy or molten relief.
 *
 * Returns a 2D slope scaled by `gain` and clamped to ±4, so relief never spikes into seams.
 * Feed its x and y to `nudgeNormal`.
 *
 * @example
 * ```ts
 * const hg = perlinSlope(pat, 0.62)
 * const n = nudgeNormal(nGeo, mul(hg.member('x'), -0.5), mul(hg.member('y'), -0.5))
 * ```
 * @see nudgeNormal, fdSlope, flowWarp
 */
export function perlinSlope(q: Expr, gain: Expr | number): Expr {
    // Chrome's pressed-metal waviness passes gain 1, LiquidMetal's molten relief 0.62.
    return call(materialParts.clampedPerlinGrad, 'clampedPerlinGrad', [q, typeof gain === 'number' ? float(gain) : gain])
}

/**
 * How much material the view passes through at this pixel: 0 at the edge, up to about 1.2 deep inside.
 *
 * Multiply by a density prop and feed `beerLambert` for glass, ice and water.
 *
 * @example
 * ```ts
 * const thickness = mul(opticalThickness(field, frame), u.density)
 * ```
 * @see beerLambert
 */
export function opticalThickness(field: SurfaceField, frame: SurfaceFrame): Expr {
    // Chord on marched shapes, a rim-depth proxy on flat ones.
    return call(materialParts.opticalThickness, 'opticalThickness', [field.sdf, frame.volFlag])
}

/**
 * How much light survives a thickness of tinted material, per channel.
 *
 * `absorb` is an rgb absorption; higher absorbs more. Returns rgb transmission 0–1.
 * Multiply the color behind by it.
 *
 * @example
 * ```ts
 * const transmit = beerLambert(mul(opticalThickness(field, frame), u.density), absorb)
 * ```
 * @see opticalThickness
 */
export function beerLambert(thickness: Expr, absorb: Expr): Expr {
    // `exp(−thickness · absorb)` per channel.
    return call(materialParts.beerLambert, 'beerLambert', [thickness, absorb])
}

/**
 * A slowly folding 2D offset to add to pattern coordinates, for molten or wind-blown motion.
 *
 * `warpInput` is the pattern coordinate scaled by about 0.6, `flowT` the layer's clock times a
 * speed. Add the result to the coordinate before sampling noise.
 *
 * @example
 * ```ts
 * const q = add(pat, mul(flowWarp(mul(pat, 0.6), flowT), u.warp))
 * ```
 * @see surfaceNoiseAt, perlinSlope
 */
export function flowWarp(warpInput: Expr, flowT: Expr): Expr {
    // Two incommensurate sin·cos products; shared verbatim by LiquidMetal and Water.
    return call(materialParts.flowWarpOffset, 'flowWarpOffset', [warpInput, flowT])
}

/**
 * Smooth signed noise at a 2D point, roughly −1 to 1, with features about one unit wide.
 *
 * The basis for grain, crumple and frost. Scale the inputs to set the feature size.
 *
 * @example
 * ```ts
 * const grain = surfaceNoise(mul(pat.member('x'), 140), mul(pat.member('y'), 140))
 * ```
 * @see surfaceNoiseAt, valueNoise, hashNoise, grainNoise
 */
export function surfaceNoise(x: Expr | number, y: Expr | number): Expr {
    return call(noise.mxNoiseFloat2, 'mxNoiseFloat2', [vec2E(x, y)])
}

/**
 * `surfaceNoise` at a 2D point you already hold.
 *
 * @example
 * ```ts
 * const n = surfaceNoiseAt(mul(pat, 6))
 * ```
 * @see surfaceNoise, volumeNoiseAt, cellNoiseAt
 */
export function surfaceNoiseAt(p: Expr): Expr {
    return call(noise.mxNoiseFloat2, 'mxNoiseFloat2', [p])
}

/**
 * Smooth signed noise at a 3D point, roughly −1 to 1.
 *
 * For anything sampled through a body rather than across its surface: gas, smoke,
 * inclusions, density.
 *
 * @example
 * ```ts
 * const density = volumeNoiseAt(vec3(pos.member('x'), pos.member('y'), t))
 * ```
 * @see surfaceNoiseAt
 */
export function volumeNoiseAt(p: Expr): Expr {
    return call(noise.mxNoiseFloat3, 'mxNoiseFloat3', [p])
}

/**
 * A smooth rainbow color from a value that wraps every 1.
 *
 * `phases` offset the red, green and blue cycles; `[0, 1/3, 2/3]` is the full spectrum. The
 * color behind holographic foil and thin-film iridescence.
 *
 * @example
 * ```ts
 * const rgb = cosineRainbow(mul(grazing, 3), [0, 1 / 3, 2 / 3])
 * ```
 * @see tintRamp
 */
export function cosineRainbow(t: Expr, phases: [number, number, number]): Expr {
    // IQ cosine palette. Phases are arguments because the fleet's copies differ in the 4th decimal.
    return call(lighting.cosinePalette, 'cosinePalette', [t, float(phases[0]), float(phases[1]), float(phases[2])])
}

/**
 * Cellular noise at a 2D point: 0 at each cell's seed point, rising toward the cell walls.
 *
 * The value is a squared distance in cell units. Use it for facets and cracks, or as the
 * field `fdSlope` differentiates for faceted relief.
 *
 * @example
 * ```ts
 * const facet = fdSlope(cellNoiseAt, mul(pat, 6), 0.002)
 * ```
 * @see fdSlope, surfaceNoiseAt, pointStars
 */
export function cellNoiseAt(p: Expr): Expr {
    // Worley F1, jitter 1.
    return call(noise.mxWorleyNoiseFloat2Pub, 'mxWorleyNoiseFloat2Pub', [p, float(1)])
}

/**
 * The value and slope of any field at a point: `value`, `dx` and `dy`.
 *
 * `sampleAt` is the field as a function of a 2D point, `eps` the tap distance in that point's
 * units (a few thousandths in uv). The bridge from a height field to a normal: feed `dx` and
 * `dy` to `nudgeNormal`.
 *
 * @example
 * ```ts
 * const relief = fdSlope((at) => heightField(params, at), uv, 0.0035)
 * ```
 * @tip Three taps of the field per pixel. Keep the field cheap or `share` it.
 * @see nudgeNormal, perlinSlope, cellNoiseAt
 */
export function fdSlope(
    sampleAt: (p: Expr) => Expr,
    p: Expr,
    eps: number,
    hint = 'fd',
): {value: Expr; dx: Expr; dy: Expr} {
    const value = local(sampleAt(p), `${hint}0`)
    return {
        value,
        dx: local(div(sub(sampleAt(add(p, vec2E(eps, 0))), value), eps), `${hint}X`),
        dy: local(div(sub(sampleAt(add(p, vec2E(0, eps))), value), eps), `${hint}Y`),
    }
}

/**
 * Random 0–1 per point, with no smoothness between neighbours.
 *
 * For grain, flicker and per-pixel jitter. Feed it large coordinates so every pixel differs.
 *
 * @example
 * ```ts
 * const grain = hashNoise(mul(uv.member('x'), 991.3), mul(uv.member('y'), 787.7))
 * ```
 * @see interleavedNoise, valueNoise, sensorGrain
 */
export function hashNoise(x: Expr | number, y: Expr | number): Expr {
    return call(noise.hash12, 'hash12', [vec2E(x, y)])
}

/**
 * A 0–1 dither pattern over device pixels where neighbours differ as much as possible.
 *
 * Better than `hashNoise` for jittering sample positions, which would otherwise clump into
 * visible grain. `pixel` is the pixel coordinate (uv times the viewport size). `clock`
 * shifts the pattern each frame.
 *
 * @example
 * ```ts
 * const jitter = interleavedNoise(mul(uv, viewport), frame)
 * ```
 * @see hashNoise, sensorGrain
 */
export function interleavedNoise(pixel: Expr, clock: Expr | number = 0): Expr {
    // Jimenez 2014 interleaved gradient noise; `clock` advances by the golden ratio per unit so
    // successive frames decorrelate.
    const cell = add(mul(pixel.member('x'), 0.06711056), mul(pixel.member('y'), 0.00583715))
    return fractE(add(mul(52.9829189, fractE(cell)), mul(clock, 0.61803398875)))
}

/**
 * Smooth noise 0–1 at a 2D point, softer and blockier than `surfaceNoise`.
 *
 * @example
 * ```ts
 * const wobble = sub(valueNoise(mul(x, 4), mul(y, 3)), 0.5)
 * ```
 * @see surfaceNoise, hashNoise
 */
export function valueNoise(x: Expr | number, y: Expr | number): Expr {
    return call(noise.value12, 'value12', [vec2E(x, y)])
}

/**
 * How far a 3D shape has rotated, as a 2D pan you can move content by.
 *
 * Zero on flat shapes and on any shape at rest, growing as the shape turns. Constant across
 * the frame, so it pans a star field or a backdrop without shearing it. Needs `pattern: 'raw'`.
 *
 * @example
 * ```ts
 * const pan = rotationSensor(frame)
 * ```
 * @see shellRefract, surfacePattern
 */
export function rotationSensor(frame: SurfaceFrame): {x: Expr; y: Expr} {
    // One field tap at the placement centre reads the surface-locked pattern coords of the
    // centre hit; flat shapes gate to zero through `volFlag`.
    const s0C = local(frame.gradSampler(vec2E(0.5, 0.5)), 's0C')
    return {
        x: local(mul(neg(s0C.member('y')), frame.volFlag), 'rotSenseX'),
        y: local(mul(neg(s0C.member('z')), frame.volFlag), 'rotSenseY'),
    }
}

/**
 * Coordinates for what sits inside a glass shell, bent where the shell curves.
 *
 * Returns `x` and `y` centred on the shape and displaced by the normal's tilt times
 * `amount`, plus the unbent `centered` pair for anything fixed to the container. The bend is
 * strongest at bevels and silhouettes, where real glass bends its interior.
 *
 * @example
 * ```ts
 * const interior = shellRefract(frame, n, u.bend)
 * ```
 * @see rotationSensor, refract
 */
export function shellRefract(frame: SurfaceFrame, n: Expr, amount: Expr): {
    x: Expr
    y: Expr
    centered: {x: Expr; y: Expr}
} {
    // `n.xy` is ~0 on a flat face and large at bevels and silhouettes — the cheap
    // single-interface cousin of a full refraction trace.
    const cx = local(sub(frame.sdfUV!.member('x'), 0.5), 'cx')
    const cy = local(sub(frame.sdfUV!.member('y'), 0.5), 'cy')
    return {
        x: local(add(cx, mul(n.member('x'), amount)), 'bentX'),
        y: local(add(cy, mul(n.member('y'), amount)), 'bentY'),
        centered: {x: cx, y: cy},
    }
}

/**
 * How mirror-like a surface is at this pixel: `r0` face-on, rising to `r0 + gain` at the rim.
 *
 * `grazing` comes from `grazingOf` or `grazingFlat`. Glass is `{r0: 0.04, gain: 0.96}`. Lower
 * the gain to keep the rim from turning into a full mirror over an interior.
 *
 * @example
 * ```ts
 * const fresnel = schlickFresnel(grazingOf(n, view), {gain: 0.6})
 * ```
 * @see fresnelBoost, grazingOf, reflect
 */
export function schlickFresnel(grazing: Expr, opts: {r0?: number; gain: Expr | number; power?: number}): Expr {
    // `r0 + grazing^power · gain`, power 5 by default.
    return add(opts.r0 ?? 0.04, mul(powE(grazing, float(opts.power ?? 5)), opts.gain))
}

/**
 * A brightness factor that shimmers over time, different for each element.
 *
 * Ranges from `1 − 2·amount` to 1. `phase` is a per-element value (a noise or a hash) that
 * keeps the elements out of step; `rate` is how fast, `spread` how differently they flicker.
 *
 * @example
 * ```ts
 * shimmer: (v) => twinkle(t, v, {amount: 0.3, rate: 1.7, spread: 40})
 * ```
 * @see pointStars
 */
export function twinkle(t: Expr, phase: Expr, opts: {amount: Expr; rate: number; spread: number}): Expr {
    // `1 − amount + sin(t·rate + phase·spread)·amount`.
    return add(sub(1, opts.amount), mul(sinE(add(mul(t, opts.rate), mul(phase, opts.spread))), opts.amount))
}

/**
 * A plane of sparse glinting points: stars, glitter, dust.
 *
 * `p` is the plane coordinate; scale it to set density. Returns the glint intensity, the slow
 * per-point `variation` field, and `rgb` when `tint` is given. Radii are in the cellular
 * noise's squared-distance units, so they are small (0.004–0.02).
 *
 * @example
 * ```ts
 * const stars = pointStars(mul(pat, 40), {variationFreq: 0.13, radius: [0.004, 0.013], keep: [0.35, 0.75], brightness: [0.3, 1.2], gain: u.stars, tint: {from: [0.72, 0.82, 1], to: [1, 0.92, 0.82], gain: 1.6}})
 * ```
 * @see twinkle, cellNoiseAt
 */
export function pointStars(p: Expr, opts: {
    /** Frequency of the per-point variation field relative to `p` (slow, about 0.1). */
    variationFreq: number
    /** Point radius as `[base, jitter]`: base plus variation times jitter. */
    radius: [number, number]
    /** The band of the variation field that keeps a point at all. Narrower is sparser. */
    keep: [number, number]
    /** Brightness as `[base, gain]`: base plus variation squared times gain. */
    brightness: [number, number]
    /** A per-point shimmer factor built from the variation field (see `twinkle`). */
    shimmer?: (variation: Expr) => Expr
    /** Overall gain. */
    gain: Expr
    /** Color the glints: variation mixes `from` to `to`, scaled by the glint times `gain`. */
    tint?: {from: [number, number, number]; to: [number, number, number]; gain: number}
    hint?: string
}): {glint: Expr; variation: Expr; rgb?: Expr} {
    // Worley feature points lit within a tiny radius, gated to a sparse subset by a slow
    // variation field; the worley metric is squared distance in cell units.
    const hint = opts.hint ?? 'stars'
    const variation = local(add(mul(surfaceNoiseAt(mul(p, opts.variationFreq)), 0.5), 0.5), `${hint}V`)
    const shimmer = opts.shimmer ? opts.shimmer(variation) : float(1)
    const glint = local(mul(mul(mul(
        smoothstepE(add(opts.radius[0], mul(variation, opts.radius[1])), 0.0005, cellNoiseAt(p)),
        smoothstepE(opts.keep[0], opts.keep[1], variation)),
        mul(add(opts.brightness[0], mul(mul(variation, variation), opts.brightness[1])), shimmer),
    ), opts.gain), hint)
    const rgb = opts.tint
        ? mul(
            mixE(vec3E(...opts.tint.from), vec3E(...opts.tint.to), splat3(variation)),
            splat3(mul(glint, opts.tint.gain)),
        )
        : undefined
    return {glint, variation, rgb}
}

// ─── Placement-space points, the field beyond its texture, and positional light ──────────────
//
// The words a shape effect needs to light its surroundings from a POINT (a cursor, an anchor)
// rather than a direction: bring a position prop into the field's own coordinates, keep the
// distance field honest far from the shape, find the boundary a pixel's light comes from, and
// evaluate a point light against a normal.

/**
 * A position prop in the shape's own coordinates, so you can compare it with the field.
 *
 * Takes a `transformPosition` prop and applies the same center, scale and rotation the shape
 * has. Distances from it are in placement units, not canvas units.
 *
 * @example
 * ```ts
 * const lightUV = placementPoint(params, params.uniforms.lightPosition)
 * ```
 * @see pointLightFrom, nearestEdge
 */
export function placementPoint(params: GpuFragmentParams, position: Expr, hint = 'placed'): Expr {
    const {uniforms, ctx} = params
    // transformPosition stores `(x, 1 − y)`; sdfSpaceUV takes a screen UV.
    const screenUV = vec2E(position.member('x'), sub(1, position.member('y')))
    return local(call(sdfSpaceUV, 'sdfSpaceUV', [uniforms.center, uniforms.scale, uniforms.rotation, screenUV, ctx.aspect]), hint)
}

/**
 * The distance field extended past the edge of an SVG shape's texture.
 *
 * An SVG field stops at its 0–1 square, so a far-reaching glow or shadow would see a plateau
 * beyond it. Returns the extended distance here, an `outward` push to add to any slope taken
 * near the border, and `at(uv)` for the distance anywhere. Pass an unscaled `surfaceField`.
 * Flat and 3D shapes pass through unchanged.
 *
 * @example
 * ```ts
 * const far = continuedField(frame, field, params)
 * ```
 * @see nearestEdge, fieldSlope
 */
export function continuedField(
    frame: SurfaceFrame,
    field: SurfaceField,
    params: GpuFragmentParams,
): {sdf: Expr; outward: Expr; at: (uv: Expr) => Expr} {
    // The continuation is a sphere-trace-safe LOWER bound — `max(beyond, border − beyond)`, the
    // shape being inside the square — not `border + beyond`, which is an upper bound a march
    // overshoots through. `at` uses the fast neighbour-quality taps (the sampler for ray marches
    // and shadow rays). Analytic fields need nothing.
    const flatSvg = !frame.volumetric && Boolean(params.propValues.shapeSdfUrl)
    if (!flatSvg) return {sdf: field.sdf, outward: vec2E(0, 0), at: (uv) => frame.gradSampler(uv).member('x')}
    const uv = frame.sdfUV!
    const beyond = (p: Expr): Expr => sub(p, clampE(p, vec2E(0, 0), vec2E(1, 1)))
    const outward = local(beyond(uv), 'outward')
    const continued = (border: Expr, dist: Expr): Expr => maxE(border, maxE(dist, sub(border, dist)))
    return {
        sdf: local(continued(field.sdf, lengthE(outward)), 'sdfFar'),
        outward,
        at: (p) => {
            const dist = lengthE(beyond(p))
            return continued(frame.gradSampler(p).member('x'), dist)
        },
    }
}

/**
 * The closest point on the shape's edge to this pixel, and the outward normal there.
 *
 * Where a rim glow or contact shadow comes from for pixels outside the shape. `slope` is
 * `fieldSlope`; its length is guarded.
 *
 * @example
 * ```ts
 * const edge = nearestEdge(frame.sdfUV!, field.sdf, fieldSlope(field))
 * ```
 * @see fieldSlope, pointLightFrom, continuedField
 */
export function nearestEdge(sdfUV: Expr, sdf: Expr, slope: Expr, hint = 'edge'): {point: Expr; normal: Expr} {
    const normal = local(div(slope, maxE(lengthE(slope), 0.0001)), `${hint}N`)
    const point = local(sub(sdfUV, mul(normal, maxE(sdf, 0))), `${hint}P`)
    return {point, normal}
}

/**
 * A point light as seen from a point on the shape: the 3D light vector `L`, the 2D direction `toLight` and the `distance`.
 *
 * `position` is in placement coordinates (see `placementPoint`). `height` lifts the light
 * toward the viewer, in the same units. Feed `L` to `lambert` and `distance` to `inverseSquare`.
 *
 * @example
 * ```ts
 * const onBody = pointLightFrom({position: lightUV, height: u.lightHeight}, frame.sdfUV!)
 * ```
 * @see placementPoint, lambert, inverseSquare, keyLightAt
 */
export function pointLightFrom(
    light: {position: Expr; height: Expr | number},
    at: Expr,
    hint = 'pl',
): {L: Expr; toLight: Expr; distance: Expr} {
    // `L` follows the material convention (z negative = toward the viewer, matching `lightVec3`).
    const delta = local(sub(light.position, at), `${hint}D`)
    const distance = local(lengthE(delta), `${hint}Dist`)
    const toLight = local(div(delta, maxE(distance, 0.0001)), `${hint}Dir`)
    const L = local(normalizeE(vec3E(delta.member('x'), delta.member('y'), neg(light.height))), `${hint}L`)
    return {L, toLight, distance}
}

/**
 * How much a light falls on a surface: 0 facing away, 1 facing it.
 *
 * `normal` and `light` are unit vectors, both 2D or both 3D. `wrap` lets light reach around
 * the shape: 0 is a hard terminator, 1 lights it all the way round.
 *
 * @example
 * ```ts
 * const diffuse = lambert(n, key.member('L'), {wrap: 0.3})
 * ```
 * @see keyLightAt, pointLightFrom, dualLobeGlint
 */
export function lambert(normal: Expr, light: Expr, opts: {wrap?: Expr | number} = {}): Expr {
    const ndl = dotE(normal, light)
    if (opts.wrap === undefined) return maxE(ndl, 0)
    return clampE(div(add(ndl, opts.wrap), add(1, opts.wrap)), 0, 1)
}

/**
 * How a point light fades with distance: 1 at the source, half at `range`.
 *
 * @example
 * ```ts
 * const lit = mul(lambert(n, onBody.L), inverseSquare(onBody.distance, u.lightRange))
 * ```
 * @see pointLightFrom, lambert
 */
export function inverseSquare(distance: Expr, range: Expr | number): Expr {
    const r = div(distance, range)
    return div(1, add(1, mul(r, r)))
}

/**
 * Turns light that piles past 1 into a soft burn toward white, like film.
 *
 * Apply it to summed emitted light before output. `neutralTone` is the cousin for a lit,
 * tinted surface.
 *
 * @example
 * ```ts
 * const glow = exposureTone(add(irradiance, emitter))
 * ```
 * @see neutralTone, sensorGrain
 */
export function exposureTone(rgb: Expr): Expr {
    // `1 − e^(−rgb)`.
    return sub(1, expE(neg(rgb)))
}

/**
 * Adds a fine per-pixel brightness jitter, so rendered light reads as photographed.
 *
 * Each pixel is scaled by up to half of `amount` either way. `pixel` is the pixel coordinate
 * (uv times the viewport size); `clock` re-rolls the grain each frame.
 *
 * @example
 * ```ts
 * const rgb = sensorGrain(glow, {amount: 0.08, pixel: mul(uv, viewport), clock: t})
 * ```
 * @see hashNoise, interleavedNoise, exposureTone
 */
export function sensorGrain(rgb: Expr, opts: {amount: Expr | number; pixel: Expr; clock: Expr | number}): Expr {
    // `rgb · (1 + (hash − 0.5) · amount)`.
    const n = hashNoise(add(opts.pixel.member('x'), mul(opts.clock, 37.1)), add(opts.pixel.member('y'), mul(opts.clock, 17.3)))
    return mul(rgb, add(1, mul(sub(n, 0.5), opts.amount)))
}

/**
 * A highlight with a tight hot core and a wide dim halo.
 *
 * `ndh` is the dot of the normal and the key light's `.H`. `core` and `halo` are
 * `[exponent, gain]` pairs. `softness` (0–1, 0.5 neutral) widens or tightens both lobes.
 *
 * @example
 * ```ts
 * const spec = dualLobeGlint(dot(n, key.member('H')), {core: [140, 1.3], halo: [22, 0.14], softness: u.highlightSoftness, gain: u.highlight})
 * ```
 * @see sharpGlint, keyLightAt, lambert
 */
export function dualLobeGlint(ndh: Expr, opts: {
    /** `[exponent, gain]` of the hot core lobe. */
    core: [number, number]
    /** `[exponent, gain]` of the wide halo lobe. */
    halo: [number, number]
    softness?: Expr
    gain: Expr
}): Expr {
    // softness 0 is ~4× sharper than the authored exponents, 1 is ~4× softer.
    const widen = opts.softness ? local(exp2(mixE(2, -2, opts.softness)), 'lobeWiden') : undefined
    const exponent = (e: number): Expr => (widen ? mul(e, widen) : float(e))
    return mul(add(
        mul(powE(ndh, exponent(opts.core[0])), opts.core[1]),
        mul(powE(ndh, exponent(opts.halo[0])), opts.halo[1]),
    ), opts.gain)
}
