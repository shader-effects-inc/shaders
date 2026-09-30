/**
 * std/effects/lens — lens effects: color fringing and spectral splits of the layer inside.
 *
 * `chromaticFan` is a sampling stage: pass it to `effects.blurs.gatherStack` to build the
 * `effect:` of a gather filter, one that reads the child as a texture and samples it at
 * several places per pixel. `spectralLens` is a whole fragment for a `gpu:` definition.
 * Angles are in degrees, strengths 0–1.
 */
// Maintainer notes: the rgbSplit family (see blurs.ts) reads the child RTT at per-channel offset
// UVs and recombines one channel from each tap. This module carries the directional per-channel
// variant: a fan of three taps along one angle, each channel with its own signed offset
// multiplier. Taps are premultiplied; the gather species owns the unpremultiply tail.
import type {GpuFragmentParams, EmitContext} from '../../gpu/contract'
import {Expr, formatFloat} from '../../gpu/contract'
import {call, vec4, expr, ZERO} from '../../gpu/composer'
import {tgpu, d, std, blend} from '../../gpu/kit/index'
import * as lensParts from '../../gpu/kit/lensParts'
import {resolveArg, type ArgSpec} from '../invoke'
import type {PropRef} from '../values'
import type {SampleStage} from './blurs'

// ── Kernel ──────────────────────────────────────────────────────────────────────────────────
//
// The child sample UV for one channel. `angle` is the TRANSFORMED prop value (degrees,
// transformAngle normalises to 0–360), so `std.radians(angle)` recovers the direction. Aspect-
// correct X on the direction so the angle reads visually correct on any canvas; strength scales
// 0–1 to a 0–0.1 UV offset. Called three times (one channelOffset per channel).
/** @internal The per-channel sample coordinate behind `chromaticFan`. */
export const chromaticOffsetUV = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (uv, angle, strength, aspect, channelOffset) => {
        'use gpu'
        const angleRad = std.radians(angle)
        const direction = d.vec2f(std.cos(angleRad) / aspect, std.sin(angleRad))
        const scaledStrength = strength * 0.1
        const offsetBase = direction.mul(scaledStrength)
        return uv.add(offsetBase.mul(channelOffset))
    })

/**
 * Split the child into red, green and blue copies pushed apart along one direction.
 *
 * `angle` is in degrees (0 pushes along x); give the prop `transform: transformAngle`.
 * `strength` is 0–1, where 1 moves a channel by a tenth of the canvas height. `red`, `green`
 * and `blue` multiply that shift per channel, usually −1, 0 and 1. Returns a sampling stage;
 * wrap it in `gatherStack` to make the effect.
 *
 * @example
 * ```ts
 * effect: gatherStack(chromaticFan({strength: p('strength'), angle: p('angle'), red: p('redOffset'), green: p('greenOffset'), blue: p('blueOffset')}), [])
 * ```
 * @tip Alpha comes from the green copy, so keep `green` at 0 to hold the layer's outline in place.
 * @see gatherStack, spectralLens
 */
export function chromaticFan(opts: {
    strength: ArgSpec
    angle: ArgSpec
    red: ArgSpec
    green: ArgSpec
    blue: ArgSpec
}): SampleStage {
    // Three per-channel offset taps along `angle` (each channel's signed multiplier × `strength`)
    // recombined R/G/B, alpha from the green tap. Not a uvRemap candidate: three DIFFERENT sample
    // UVs can't fold into one coordinate.
    return (params) => {
        const tapUV = (offset: ArgSpec): Expr =>
            call(chromaticOffsetUV, 'chromaticOffsetUV', [
                params.ctx.uv, resolveArg(opts.angle, params), resolveArg(opts.strength, params),
                params.ctx.aspect, resolveArg(offset, params),
            ])
        const redSample = params.texture.sample(tapUV(opts.red))
        const greenSample = params.texture.sample(tapUV(opts.green))
        const blueSample = params.texture.sample(tapUV(opts.blue))
        return vec4(redSample.member('r'), greenSample.member('g'), blueSample.member('b'), greenSample.member('a'))
    }
}

// ── Spectral lens ───────────────────────────────────────────────────────────────────────────

/** The props `spectralLens` reads. Every field is a prop ref (`p('name')`); ranges are the useful slider ranges. */
export interface SpectralLensSlots {
    /** Position prop (`transform: transformPosition`) the split, focus, warp and swirl pivot around. */
    center: PropRef
    /** 0–1, how far apart the color copies are pushed; 0 is off. */
    spread: PropRef
    /** −1 to 1, slides the copies toward one end of the spread; 0 spaces them evenly. */
    bias: PropRef
    /** Direction of the spread in degrees (`transform: transformAngle`). */
    angle: PropRef
    /** 0–1, bends the spread from a straight line (0) to a burst out from the center (1). */
    perspective: PropRef
    /** Number of color copies, 2–50; more is smoother and costlier. */
    count: PropRef
    /** 0–1, how strongly each copy takes its own color from the spectrum; 0 keeps the original color. */
    dispersion: PropRef
    /** −1 to 1, where the dispersion lands: −1 only near the center, 1 only toward the edges. */
    dispersionShift: PropRef
    /** 0–1, turns the dispersion colors around the hue wheel; 1 is a full turn. */
    dispersionColor: PropRef
    /** 0–1, fades the split out in a round zone at the center. */
    focusCenter: PropRef
    /** 0–1, fades the split out toward the edges; 1 restores the original image there. */
    focusEdges: PropRef
    /** −1 to 1, rotates the copies around the center by an angle that grows along the spread. */
    swirl: PropRef
    /** 0–1, scatters the spread direction with noise; 0 is off. */
    noise: PropRef
    /** 0–1, how fine the direction noise is. */
    noiseFrequency: PropRef
    /** 0–1, shifts the noise pattern (a seed). */
    noiseOffset: PropRef
    /** −1 to 1, fisheye bulge (positive) or pincushion pinch (negative). */
    lensBulge: PropRef
    /** 0–1, squeezes the image toward a circular outline. */
    lensCircle: PropRef
    /** 0–1, breaks up the copy edges with grain. */
    grainMixer: PropRef
    /** 0–1, black-and-white grain laid over the result. */
    grainOverlay: PropRef
}

/**
 * Fan the child out into many color-shifted copies through a warped lens, the spectral-lens look.
 *
 * A whole fragment for a `gpu:` definition; the child is read as a texture. `count` copies
 * spread along `angle` around `center`, `dispersion` colors them across the spectrum, the
 * focus slots fade the split near the center and the edges, `lensBulge` and `lensCircle`
 * warp the whole image, and `swirl`, `noise` and the grain slots scatter it. Every slot is a
 * prop ref; the field list on `SpectralLensSlots` gives units and ranges.
 *
 * @example
 * ```ts
 * gpu: {fragment: spectralLens({center: p('center'), spread: p('spread'), bias: p('bias'), angle: p('angle'), perspective: p('perspective'), count: p('count'), dispersion: p('dispersion'), dispersionShift: p('dispersionShift'), dispersionColor: p('dispersionColor'), focusCenter: p('focusCenter'), focusEdges: p('focusEdges'), swirl: p('swirl'), noise: p('noise'), noiseFrequency: p('noiseFrequency'), noiseOffset: p('noiseOffset'), lensBulge: p('lensBulge'), lensCircle: p('lensCircle'), grainMixer: p('grainMixer'), grainOverlay: p('grainOverlay')})}
 * ```
 * @tip Every pixel samples the child `count` times, and phones cap `count` at 24. Keep it low on large canvases.
 * @see chromaticFan
 */
export function spectralLens(slots: SpectralLensSlots): (params: GpuFragmentParams) => Expr {
    // Split the child into shifting chromatic layers (a runtime-count gather fan) with
    // barrel/pincushion lens warp, optional circular crop, swirl, noise scatter and film grain.
    // Composed from the kit's lens statement parts — lensGeometry → spreadAxis → spectralFan →
    // grainOverlay, all appending to one statement stream (they share locals via the fresh
    // prefix). The fan loop is raw WGSL because per-iteration texture sampling can't fold into
    // the Expr graph; the accumulated result is premultiplied → straight alpha on the way out
    // like every RTT filter.
    return (params) => {
        const {uniforms, childNode, ctx, convertToTexture} = params
        if (!childNode) return ZERO

        const childTex = convertToTexture(childNode)
        const texKey = childTex.key

        return new Expr((ec: EmitContext) => {
            const em: lensParts.LensEmitFrame = {
                p: ec.freshLocal('lens'),
                stmts: [],
                U: (name: string): string => uniforms[slots[name as keyof SpectralLensSlots].name]._emit(ec),
                L: (n: number): string => formatFloat(n),
                noiseIdx: {value: 0},
            }

            lensParts.lensGeometryStmts(em, ctx.uv._emit(ec), ctx.aspect._emit(ec))
            lensParts.spreadAxisStmts(em)
            lensParts.spectralFanStmts(em, texKey)
            lensParts.grainOverlayStmts(em)

            for (const s of em.stmts) ec.statement(s)

            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [expr(`${em.p}_out`)])._emit(ec)
        })
    }
}
