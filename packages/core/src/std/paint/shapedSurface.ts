/**
 * The surface spine: shade whichever shape the user picks (a flat shape, an SVG, a 3D shape) with
 * one surface function. The engine supplies the shape's field at this pixel and around it.
 */
// Split out of ./materials with direct imports (no kit namespace barrels), so a component that
// uses only the spine bundles only the spine and the shape-field machinery it needs.
import type {GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import type {Expr} from '../../gpu/contract'
import {call, floatE, ZERO} from '../../gpu/composer'
import {sdfSpaceUV, offsetUV} from '../../gpu/kit/effects/glass'
import {ANALYTIC_SDF_EXTRA_FIELDS} from '../../gpu/kit/sdf'
import {createVolumetricFieldComputeNode, resolveShapeFieldSampler, VOLUMETRIC_FIELD_EXTRA_FIELDS} from '../../gpu/kit/sdf3d'
import type {PortableFieldSampler} from '../../gpu/kit/sdf3d'
import {local} from '../math'

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
    /** The field in a form a separate WGSL fn can sample anywhere (a custom body's `shapeAt(p)`). */
    portable?: PortableFieldSampler
    /** 3D shapes only: the marched depth's slope at any placement coordinate, `vec2` (smooth and
     *  exact, unlike screen derivatives of `.w`). Unreliable within ~2 texels of the silhouette. */
    depthSlope?: (uv: Expr) => Expr
    /** 3D shapes only: one field texel, in placement units. */
    texelSpan?: Expr
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
        const {sampler, gradSampler, texelSampler, volumetric, bakedGradients, portable, depthSlope, texelSpan} = resolveShapeFieldSampler(params, {
            patternMode: pattern,
            gradSampler: spec.gradSampler ?? 'fast',
            bakedGradients: spec.bakedGradients ?? 'none',
        })

        const frame: SurfaceFrame = {
            sampler, gradSampler, texelSampler, volumetric, bakedGradients, portable, depthSlope, texelSpan,
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
