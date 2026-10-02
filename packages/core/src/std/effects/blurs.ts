/**
 * Blurs, glows, sharpening, pixelation, shadows, retro screens and other effects that read the
 * layer inside them (the child) as a texture. Each word goes in a filter definition's `effect:`
 * field, or spreads into a definition when it needs its own compute pass. Finished effects
 * (`gaussianBlur`, `motionBlur`, `bloom`, `pixelate`, …) are one call. Stacked looks (a CRT, a
 * page peel, fluted glass) are built with `gatherStack` from a sampling stage, overlay stages
 * and, where several stages share one geometry, a frame.
 */
// Maintainer notes.
// Nearly all of this module rides the gather pipeline: the child renders to a texture, taps are
// premultiplied, and the species owns the RTT boundary and the premultiplied → straight
// unpremultiply tail (a stack that finishes in straight alpha passes `{resultAlpha: 'straight'}`
// to `gatherStack`). Multi-part recipes are composed IN THE SHADER FILE from the stage parts
// exported here — a `SampleStage`, ordered `OverlayStage`s, and, where several stages read one
// geometry, a shared `GatherFrame`. The GPU math lives in the kit's `motionBlur`/`warpMaps`
// modules; these parts only bind props and context.
import type {Expr, GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {call, floatE, mixExpr, vec4} from '../../gpu/composer'
import {asLocal, expr} from '../../gpu/composer'
import {formatFloat, animatedTime, createGuardedCompute, ZERO} from '../../gpu/porters'
import {blur as blurKit, motionBlur as motionBlurKit} from '../../gpu/kit/index'
import {blend, sampling, warpMaps, edges, tgpu, d, std, constants} from '../../gpu/kit/index'
import type {RttFilterParams} from '../../gpu/scaffolds/rttFilter'
import {resolveArg, uniformOf, type ArgSpec} from '../invoke'
import type {GatherEffect} from '../types'
import type {PropRef} from '../values'
import * as math from '../math'

// ── Gather pipeline ─────────────────────────────────────────────────────────────────────────
//
// A gather effect as a STACK: one sampling stage produces the initial color from the child RTT,
// then overlay stages transform it in order. Each stage is a named part; an effect noun reads as
// the recipe of its stages.

/**
 * The first stage of a stacked effect: reads the child as a texture and returns the starting
 * color at this pixel.
 *
 * Pass one to `gatherStack` as its first argument. The overlay stages then edit its result in order.
 * @see gatherStack, OverlayStage, childTap
 */
export type SampleStage = (params: RttFilterParams) => Expr

/**
 * One editing stage of a stacked effect: takes the color so far and returns the edited color.
 *
 * Stages run in the order you list them in `gatherStack`.
 * @see gatherStack, SampleStage
 */
export type OverlayStage = (color: Expr, params: RttFilterParams) => Expr

/**
 * Assemble an effect from one sampling stage and an ordered list of overlay stages.
 *
 * The child renders to a texture, the sampling stage reads it, and each overlay edits the result
 * in turn. Put the result in a filter's `effect:` field. Pass `{resultAlpha: 'straight'}` only
 * when the stack already returns straight (not premultiplied) alpha, as `crispTap` and
 * `refractedTaps` do.
 *
 * @example
 * ```ts
 * effect: gatherStack(rgbSplit({amount: p('colorShift')}), [scanlines({frequency: p('lines'), intensity: 0.3}), opaque()])
 * ```
 * @tip An empty overlay list is fine: `gatherStack(stage, [])` wraps a lone sampling stage.
 * @see SampleStage, OverlayStage, GatherFrame
 */
export function gatherStack(
    sample: SampleStage,
    overlays: OverlayStage[],
    opts?: {resultAlpha?: 'premultiplied' | 'straight'},
): GatherEffect {
    return {
        kind: 'gather',
        resultAlpha: opts?.resultAlpha,
        build: (params): Expr => overlays.reduce((color, stage) => stage(color, params), sample(params)),
    }
}

/**
 * A geometry that several stages of one effect share: a warped coordinate, a surface normal, a
 * coverage mask.
 *
 * Make one with a frame word (`sphereFrame`, `glitchFrame`, `fluteFrame`, `peelFrame`), keep it
 * in a module constant, and hand that same frame to every stage. It is computed once per pixel
 * however many stages read it.
 * @see sphereFrame, glitchFrame, fluteFrame, peelFrame
 */
// Implementation: one Expr (usually an `asLocal` struct); the producer is memoized per composition
// params, so every stage sees the same local.
export type GatherFrame = (params: RttFilterParams) => Expr

/** Memoize a frame producer per composition params. */
function frameOf(make: (params: RttFilterParams) => Expr): GatherFrame {
    const cache = new WeakMap<RttFilterParams, Expr>()
    return (params) => {
        const hit = cache.get(params)
        if (hit) return hit
        const made = make(params)
        cache.set(params, made)
        return made
    }
}

// ── Sampling-stage parts ────────────────────────────────────────────────────────────────────

/**
 * The child with its red and blue pulled a little to either side, like a misaligned CRT.
 *
 * `amount` is the shift; 1 moves red and blue each by 0.002 of the canvas width, 0–10 is the
 * usual range. Returns a three-channel color, so finish the stack with `opaque`.
 *
 * @example
 * ```ts
 * effect: gatherStack(rgbSplit({amount: p('colorShift')}), [opaque()])
 * ```
 * @see glitchRgbSplit, scanlines, phosphorMask, opaque
 */
export function rgbSplit(opts: {amount: ArgSpec}): SampleStage {
    return (params) => {
        const amount = resolveArg(opts.amount, params)
        const redUV = call(motionBlurKit.crtSampleUV, 'crtSampleUV', [params.ctx.uv, amount, floatE(1)])
        const blueUV = call(motionBlurKit.crtSampleUV, 'crtSampleUV', [params.ctx.uv, amount, floatE(-1)])
        return call(motionBlurKit.rgbSplitCombine, 'rgbSplitCombine', [
            params.texture.sample(redUV), params.texture.sample(params.ctx.uv), params.texture.sample(blueUV),
        ])
    }
}

/**
 * The wobbling, jittering coordinates of a worn VHS tape, to feed into `chromaSmearTaps`.
 *
 * `wobble` (0–5) is the tape damage: waves, creases and head-switching noise that burst on and
 * off over time. `jitter` (0–1) is fine per-scanline noise. `speed` scales the global clock,
 * so the definition needs no `animatedTime`.
 *
 * @example
 * ```ts
 * chromaSmearTaps({warp: tapeWarp({wobble: p('wobble'), jitter: p('scanlineNoise'), speed: p('speed')}), smear: p('smear')})
 * ```
 * @see chromaSmearTaps, beatPulse
 */
// Returns vec4(lumaUV.xy, chromaUV.zw); animated by ctx.time × speed, not a per-node clock.
export function tapeWarp(opts: {wobble: ArgSpec; jitter: ArgSpec; speed: ArgSpec}): (params: RttFilterParams) => Expr {
    return (params) =>
        call(motionBlurKit.vhsSampleUVs, 'vhsSampleUVs', [
            params.ctx.uv, params.ctx.time, resolveArg(opts.speed, params), resolveArg(opts.wobble, params), resolveArg(opts.jitter, params),
        ])
}

/**
 * The child sampled like a VHS deck: brightness stays sharp while color bleeds sideways.
 *
 * `warp` is the coordinate set from `tapeWarp`. `smear` sets how far color trails; positive
 * trails it to the right, negative to the left, −2 to 2 is the usual range. Alpha comes from the
 * sharp tap.
 *
 * @example
 * ```ts
 * effect: gatherStack(chromaSmearTaps({warp: tapeWarp({wobble: p('wobble'), jitter: p('noise'), speed: p('speed')}), smear: p('smear')}), [beatPulse({wobble: p('wobble'), speed: p('speed')})])
 * ```
 * @see tapeWarp, beatPulse, rgbSplit
 */
// One sharp luma tap + 5 chroma taps trailing by `smear·0.0075` each, recombined through YIQ
// (sharp Y, smeared I/Q).
export function chromaSmearTaps(opts: {warp: (params: RttFilterParams) => Expr; smear: ArgSpec}): SampleStage {
    return (params) => {
        const uvs = asLocal(opts.warp(params), 'tapeUVs')
        const chromaUV = uvs.member('zw')
        const lumaSample = params.texture.sample(uvs.member('xy'))
        const smearScale = resolveArg(opts.smear, params).mul(0.0075)
        // 5 chroma taps i=1..5 (the i=0 tap has weight 0 → omitted).
        const tap = (i: number): Expr =>
            params.texture.sample(call(motionBlurKit.vhsChromaTapUV, 'vhsChromaTapUV', [chromaUV, smearScale, floatE(i)]))
        const rgb = call(motionBlurKit.yiqRecombine, 'yiqRecombine', [lumaSample, tap(1), tap(2), tap(3), tap(4), tap(5)])
        return vec4(rgb, lumaSample.member('a'))
    }
}

/**
 * The brightest color in a small ring around a point: the layer grown outward, so a spot
 * just outside a bright area still reads as bright.
 *
 * `read` samples the layer at a uv and returns straight color (pass `params.sampleStraight`).
 * `center` is the uv to look around; `radius` is the ring's radius per axis in uv, as a
 * `vec2` (divide a pixel radius by the viewport size). `taps` points sit evenly round the
 * ring starting at `angle` degrees, plus the centre itself. Returns the brightest of them as
 * `color` (straight) and its brightness as `luma` (Rec.709 luminance times alpha).
 *
 * @example
 * ```ts
 * const {color, luma} = brightestNear(params.sampleStraight, pointUV, {radius: div(p, viewport), taps: 4, angle: 45})
 * ```
 * @tip Thin bright details smaller than the gap between taps can slip through; more taps
 * close the gaps at one sample each.
 * @see gatherStack
 */
export function brightestNear(read: (uv: Expr) => Expr, center: Expr, opts: {radius: Expr; taps: number; angle?: number}): {
    color: Expr
    luma: Expr
} {
    // Max-by-luminance over centre + ring; the ring offsets are compile-time directions.
    const brightness = (c: Expr): Expr => math.mul(math.dot(c.member('rgb'), math.vec3(0.2126, 0.7152, 0.0722)), c.member('a'))
    let color = math.local(read(center), 'nearC')
    let luma = math.local(brightness(color), 'nearL')
    for (let i = 0; i < opts.taps; i++) {
        const a = ((opts.angle ?? 0) + (360 * i) / opts.taps) * (Math.PI / 180)
        const tap = math.local(read(math.add(center, math.mul(math.vec2(Math.cos(a), Math.sin(a)), opts.radius))), 'nearT')
        const tapLuma = math.local(brightness(tap), 'nearTL')
        color = math.local(math.select(math.gt(tapLuma, luma), tap, color), 'nearC')
        luma = math.local(math.max(tapLuma, luma), 'nearL')
    }
    return {color, luma}
}

// ── Overlay parts ───────────────────────────────────────────────────────────────────────────

/**
 * Brightness and contrast, pivoting around mid-grey.
 *
 * 1 leaves both unchanged; 0.5–2 is the usual range. Works on the three-channel color of a
 * stack begun with `rgbSplit`.
 *
 * @example
 * ```ts
 * adjust({brightness: p('brightness'), contrast: p('contrast')})
 * ```
 * @see rgbSplit, scanlines, opaque
 */
export function adjust(opts: {brightness: ArgSpec; contrast: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.adjustShade, 'adjustShade', [color, resolveArg(opts.contrast, params), resolveArg(opts.brightness, params)])
}

/**
 * Horizontal scanlines darkening the child in a smooth wave.
 *
 * `frequency` is the number of lines from top to bottom (100–800). `intensity` (0–1) is how
 * dark the troughs get. Works on the three-channel color of a stack begun with `rgbSplit`.
 *
 * @example
 * ```ts
 * scanlines({frequency: p('scanlineFrequency'), intensity: p('scanlineIntensity')})
 * ```
 * @see distortedScanlines, phosphorMask, vignetteOverlay
 */
export function scanlines(opts: {frequency: ArgSpec; intensity: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.scanlineShade, 'scanlineShade', [
            color, params.ctx.uv.member('y'), resolveArg(opts.frequency, params), resolveArg(opts.intensity, params),
        ])
}

/**
 * A faint red, green and blue phosphor stripe pattern across the child.
 *
 * `pitch` sets how many stripes fit across the canvas, about half of `pitch`; 8–128 is the usual
 * range. Works on the three-channel color of a stack begun with `rgbSplit`.
 *
 * @example
 * ```ts
 * phosphorMask({pitch: p('pixelSize')})
 * ```
 * @see scanlines, rgbSplit
 */
// The kit fn takes fract(uv × pitch × 0.5): a HIGHER pitch gives finer stripes, about pitch/2
// of them across the canvas.
export function phosphorMask(opts: {pitch: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.phosphorShade, 'phosphorShade', [color, params.ctx.uv, resolveArg(opts.pitch, params)])
}

/**
 * Darkens the child toward its corners.
 *
 * `radius` (0–1) is how far the darkening reaches inward, 0 for the edges only and 1 for all the
 * way to the centre. `intensity` (0–1) blends it in. Round on any canvas shape. Works on the
 * three-channel color of a stack begun with `rgbSplit`.
 *
 * @example
 * ```ts
 * vignetteOverlay({radius: p('vignetteRadius'), intensity: p('vignetteIntensity')})
 * ```
 * @see scanlines, opaque
 */
export function vignetteOverlay(opts: {radius: ArgSpec; intensity: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.vignetteShade, 'vignetteShade', [
            color, params.ctx.uv, params.ctx.aspect, resolveArg(opts.radius, params), resolveArg(opts.intensity, params),
        ])
}

/**
 * Closes a three-channel stack as a solid color with alpha 1.
 *
 * Put it last after `rgbSplit`, `adjust`, `scanlines`, `phosphorMask` or `vignetteOverlay`,
 * which work on color only and drop the child's alpha.
 *
 * @example
 * ```ts
 * effect: gatherStack(rgbSplit({amount: 1}), [scanlines({frequency: 200, intensity: 0.3}), opaque()])
 * ```
 * @see rgbSplit, gatherStack
 */
export function opaque(): OverlayStage {
    return (color) => vec4(color, floatE(1))
}

/**
 * A faint brightness throb, like the mains hum of an old television.
 *
 * `wobble` scales the throb (0 turns it off). `speed` scales the global clock. Works on the
 * four-channel color of a stack begun with `chromaSmearTaps`.
 *
 * @example
 * ```ts
 * beatPulse({wobble: p('wobble'), speed: p('speed')})
 * ```
 * @see chromaSmearTaps, tapeWarp
 */
// A very subtle pulse (±1.5% × wobble) from ctx.time × speed, clamped to [0,1].
export function beatPulse(opts: {wobble: ArgSpec; speed: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.beatShade, 'beatShade', [
            color,
            call(motionBlurKit.vhsAcBeat, 'vhsAcBeat', [
                params.ctx.uv, params.ctx.time, resolveArg(opts.speed, params), resolveArg(opts.wobble, params),
            ]),
        ])
}

/**
 * The path a motion blur smears along: its kind plus the prop that anchors it.
 *
 * Make one with `blurPath.linear`, `blurPath.orbit` or `blurPath.zoom` and pass it to `motionBlur`.
 * @see blurPath, motionBlur
 */
export interface BlurPathSpec {
    readonly kind: motionBlurKit.MotionBlurPathKind
    readonly focus: PropRef
}

/**
 * The three paths `motionBlur` can smear along.
 *
 * @example
 * ```ts
 * effect: motionBlur({path: blurPath.zoom(p('center')), amount: p('intensity')})
 * ```
 * @see motionBlur
 */
export const blurPath = {
    /**
     * A straight streak along an angle prop, in degrees (0 points right, 90 points down).
     * @example
     * ```ts
     * effect: motionBlur({path: blurPath.linear(p('angle')), amount: p('intensity')})
     * ```
     * @see motionBlur
     */
    linear: (angle: PropRef): BlurPathSpec => ({kind: 'linear', focus: angle}),
    /**
     * An arc streak rotating around a center prop (a position in uv).
     * @example
     * ```ts
     * effect: motionBlur({path: blurPath.orbit(p('center')), amount: p('intensity')})
     * ```
     * @see motionBlur
     */
    orbit: (center: PropRef): BlurPathSpec => ({kind: 'orbit', focus: center}),
    /**
     * A radial streak rushing outward from a center prop (a position in uv).
     * @example
     * ```ts
     * effect: motionBlur({path: blurPath.zoom(p('center')), amount: p('intensity')})
     * ```
     * @see motionBlur
     */
    zoom: (center: PropRef): BlurPathSpec => ({kind: 'zoom', focus: center}),
}

/**
 * Smears the child along a path: a straight streak, an arc around a point, or a zoom out of a point.
 *
 * `amount` is a 0–100 intensity. A linear streak spans about twice `amount` in pixels; at 100 an
 * orbit sweeps about 29 degrees and a zoom pulls in from twice the distance to its center.
 *
 * @example
 * ```ts
 * effect: motionBlur({path: blurPath.linear(p('angle')), amount: p('intensity')})
 * ```
 * @tip Samples past the canvas edge repeat the border color. Wrap the layer in a larger group if the smear must fade out instead.
 * @see blurPath, gaussianBlur, scatter
 */
// 32 Gaussian-weighted taps of the child RTT along the chosen tap trajectory; the kit owns the
// weights, the unroll and the per-path coordinate fns.
export function motionBlur(opts: {path: BlurPathSpec; amount: ArgSpec}): GatherEffect {
    const {path, amount} = opts
    return {
        kind: 'gather',
        build: (params): Expr =>
            motionBlurKit.motionBlurGather(
                path.kind,
                {
                    focus: resolveArg(path.focus, params),
                    amount: resolveArg(amount, params),
                    uv: params.ctx.uv,
                    aspect: params.ctx.aspect,
                    viewportSize: params.ctx.viewportSize,
                },
                (coord) => params.texture.sample(coord),
            ),
    }
}

/**
 * Scatters every pixel of the child by a random offset: a grainy diffusion, not a smooth blur.
 *
 * `amount` is the largest offset in pixels. `edges` is a prop with `transform: transformEdges`
 * and `compileTime: true` ('stretch' | 'transparent' | 'mirror' | 'wrap') that says what to read
 * where an offset lands outside the canvas.
 *
 * @example
 * ```ts
 * effect: scatter({amount: p('intensity'), edges: p('edges')})
 * ```
 * @see gaussianBlur, motionBlur
 */
export function scatter(opts: {amount: ArgSpec; edges: PropRef}): GatherEffect {
    const {amount, edges} = opts
    return {
        kind: 'gather',
        build: (params): Expr =>
            motionBlurKit.scatterGather({
                uv: params.ctx.uv,
                amount: resolveArg(amount, params),
                viewportSize: params.ctx.viewportSize,
                edgeMode: (params.propValues[edges.name] as number) ?? 0,
                sample: (coord) => params.texture.sample(coord),
            }),
    }
}

/**
 * Sharpens the child by boosting the contrast between each pixel and its neighbours.
 *
 * `amount` 0 leaves the child unchanged; 0–5 is the usual range. Alpha is untouched.
 *
 * @example
 * ```ts
 * effect: sharpen(p('sharpness'))
 * ```
 * @tip Declare `identityWhen: isZero('sharpness')` and `recompile: crosses(0)` on the prop so 0 costs nothing.
 * @see gaussianBlur, scatter
 */
// A 5-tap unsharp-mask convolution: centre × (1 + 4·amount) minus the four orthogonal one-pixel
// neighbours × amount. At `amount` 0 the kernel is the identity.
export function sharpen(amount: ArgSpec): GatherEffect {
    return {
        kind: 'gather',
        build: (params): Expr =>
            motionBlurKit.sharpenGather({
                uv: params.ctx.uv,
                viewportSize: params.ctx.viewportSize,
                amount: resolveArg(amount, params),
                sample: (coord) => params.texture.sample(coord),
            }),
    }
}

/**
 * Turns the child into a grid of flat cells, each showing one color.
 *
 * `scale` is the number of cells along the canvas's longer edge (1–200; more cells means smaller
 * pixels). `gap` (0–1) is the transparent space between cells as a fraction of a cell.
 * `roundness` (0–1) rounds each cell from a square to a circle.
 *
 * @example
 * ```ts
 * effect: pixelate({scale: p('scale'), gap: p('gap'), roundness: p('roundness')})
 * ```
 * @see scatter, sharpen
 */
// The gap/roundness cut is an alpha mask over one straight-alpha tap per cell, not a coordinate
// bend — so the effect always takes the RTT path and returns straight alpha.
export function pixelate(opts: {scale: ArgSpec; gap: ArgSpec; roundness: ArgSpec}): GatherEffect {
    const {scale, gap, roundness} = opts
    return {
        kind: 'gather',
        resultAlpha: 'straight',
        build: (params): Expr =>
            motionBlurKit.pixelateGather({
                uv: params.ctx.uv,
                aspect: params.ctx.aspect,
                scale: resolveArg(scale, params),
                gap: resolveArg(gap, params),
                roundness: resolveArg(roundness, params),
                sampleStraight: params.sampleStraight,
            }),
    }
}

/**
 * The geometry of a sphere bulging out of the canvas: where each pixel reads the child from,
 * where the sphere ends, and which way its surface faces.
 *
 * `center` is a position in uv. `radius` 1 is half the canvas height. `depth` is how far it bulges
 * toward the viewer, 0 for flat and up to about 3. Feed the frame to `crispTap` and `rimLit`.
 *
 * @example
 * ```ts
 * const sphere = sphereFrame({center: p('center'), radius: p('radius'), depth: p('depth')})
 * ```
 * @see crispTap, rimLit, GatherFrame
 */
// Exposes `.uv` (bulged coordinate), `.coverage` (inside the sphere) and `.normal`.
export function sphereFrame(opts: {center: ArgSpec; radius: ArgSpec; depth: ArgSpec}): GatherFrame {
    return frameOf((params) =>
        asLocal(call(warpMaps.sphereBulge, 'sphereBulge', [
            params.ctx.uv, params.ctx.viewportSize,
            resolveArg(opts.center, params), resolveArg(opts.radius, params), resolveArg(opts.depth, params),
        ]), 'sphere'))
}

/**
 * One clean sample of the child at a frame's warped coordinate.
 *
 * Use it under a warp that magnifies: hard edges in the child (text, a logo) stay smooth instead
 * of breaking into facets. Returns straight alpha, so close the stack with
 * `{resultAlpha: 'straight'}`.
 *
 * @example
 * ```ts
 * effect: gatherStack(crispTap(sphere), [rimLit(sphere, {position: p('light'), intensity: 0.5, softness: 0.5, color: p('lightColor')})], {resultAlpha: 'straight'})
 * ```
 * @see sphereFrame, rimLit, childTap
 */
// Catmull-Rom, not bilinear: a magnifying warp blows up its centre, and one bilinear tap turns any
// hard edge underneath into visible facets. Unpremultiplied here.
export function crispTap(frame: GatherFrame): SampleStage {
    return (params) =>
        call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [
            sampling.sampleCatmullRomExpr(params.texture, frame(params).member('uv')),
        ])
}

/**
 * A tinted rim of light around the edge of a bulged surface, added on top of the child's color.
 *
 * `position` is where the light sits, in uv. `intensity` (0–1) is its strength, 0 for off.
 * `softness` (0–1) runs from a hard edge to a soft glow. `color` is a color prop. Outside the
 * frame's surface the result is transparent.
 *
 * @example
 * ```ts
 * rimLit(sphere, {position: p('lightPosition'), intensity: p('lightIntensity'), softness: p('lightSoftness'), color: p('lightColor')})
 * ```
 * @see sphereFrame, crispTap, fluteHighlight
 */
// Directional fresnel rim over the frame's normal; the tinted rim is ADDED and the alpha gated by
// the frame's coverage. Added color is what a uv+mask analytic fold can't carry, so a recipe with
// this stage stays on the RTT fragment path.
export function rimLit(frame: GatherFrame, opts: {
    position: ArgSpec
    intensity: ArgSpec
    softness: ArgSpec
    color: PropRef
}): OverlayStage {
    return (color, params) => {
        const rim = call(warpMaps.rimLight, 'rimLight', [
            frame(params).member('normal'), resolveArg(opts.position, params),
            resolveArg(opts.intensity, params), resolveArg(opts.softness, params),
        ])
        return call(warpMaps.rimComposite, 'rimComposite', [
            color, resolveArg(opts.color, params).member('rgb'), rim, frame(params).member('coverage'),
        ])
    }
}

/**
 * The shared geometry of a digital glitch: when bursts fire, which horizontal bands jitter, and
 * which blocks shift or flip.
 *
 * `intensity` (0–1) sets both the strength and how often bursts happen. `speed` scales the global
 * clock, so no `animatedTime` is needed. `blockDensity` is the base number of bands (2–50).
 * `mirrorAmount` (0–1) is the chance a block shows flipped content. Keep the frame in a module
 * constant and pass it to `glitchRgbSplit`, `colorBarFills` and `distortedScanlines`.
 *
 * @example
 * ```ts
 * const glitch = glitchFrame({intensity: p('intensity'), speed: p('speed'), blockDensity: p('blockDensity'), mirrorAmount: p('mirrorAmount')})
 * ```
 * @see glitchRgbSplit, colorBarFills, distortedScanlines, GatherFrame
 */
// Its burst/band hashes feed the split, fills, and scanlines. Animated by the GLOBAL clock scaled
// by `speed` (not a per-node animated time).
export function glitchFrame(opts: {
    intensity: ArgSpec
    speed: ArgSpec
    blockDensity: ArgSpec
    mirrorAmount: ArgSpec
}): GatherFrame {
    return frameOf((params) =>
        asLocal(call(motionBlurKit.glitchGeom, 'glitchGeom', [
            params.ctx.uv, params.ctx.time,
            resolveArg(opts.intensity, params), resolveArg(opts.speed, params),
            resolveArg(opts.blockDensity, params), resolveArg(opts.mirrorAmount, params),
        ]), 'glitch'))
}

/**
 * The child sampled through a glitch frame, with red and blue pulled apart inside the active bands.
 *
 * `shift` is the split distance, 0–20. A transparent child stays transparent through the glitch.
 *
 * @example
 * ```ts
 * effect: gatherStack(glitchRgbSplit(glitch, {shift: p('rgbShift')}), [colorBarFills(glitch, {intensity: 0.2})])
 * ```
 * @see glitchFrame, colorBarFills, rgbSplit
 */
// Alpha comes from the centred green tap.
export function glitchRgbSplit(frame: GatherFrame, opts: {shift: ArgSpec}): SampleStage {
    return (params) => {
        const geom = frame(params)
        const splitUVs = asLocal(call(motionBlurKit.glitchSplitUVs, 'glitchSplitUVs', [
            geom.member('mirroredUV'), resolveArg(opts.shift, params), geom.member('spread'),
        ]), 'splitUVs')
        const greenSample = params.texture.sample(geom.member('mirroredUV'))
        return vec4(
            params.texture.sample(splitUVs.member('xy')).member('r'),
            greenSample.member('g'),
            params.texture.sample(splitUVs.member('zw')).member('b'),
            greenSample.member('a'),
        )
    }
}

/**
 * Neon color bars filling the blocks a glitch frame currently has active.
 *
 * `intensity` (0–1) is how strongly the bars show. They respect the child's alpha.
 *
 * @example
 * ```ts
 * colorBarFills(glitch, {intensity: p('colorBarIntensity')})
 * ```
 * @see glitchFrame, glitchRgbSplit, distortedScanlines
 */
export function colorBarFills(frame: GatherFrame, opts: {intensity: ArgSpec}): OverlayStage {
    return (color, params) => {
        const geom = frame(params)
        return call(motionBlurKit.fillBarsShade, 'fillBarsShade', [
            color, params.ctx.uv, geom.member('bandY1'), geom.member('slowFrame'),
            resolveArg(opts.intensity, params), geom.member('strength'),
            geom.member('bandGate'), geom.member('blockGate'),
        ])
    }
}

/**
 * CRT-style scanlines that appear only where a glitch frame is currently distorting.
 *
 * `intensity` (0–1) is how visible they are.
 *
 * @example
 * ```ts
 * distortedScanlines(glitch, {intensity: p('scanlineIntensity')})
 * ```
 * @see glitchFrame, scanlines
 */
export function distortedScanlines(frame: GatherFrame, opts: {intensity: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.distortScanShade, 'distortScanShade', [
            color, params.ctx.uv.member('y'), params.ctx.viewportSize.member('y'),
            resolveArg(opts.intensity, params), frame(params).member('distortion'),
        ])
}

// ── Compute-backed blurs ────────────────────────────────────────────────────────────────────
//
// These ride the kit's compute lifecycle wrappers (`withFixedBlurCompute` / `withVariableBlurCompute`
// / `withBloomCompute`) rather than the gather scaffold, so each noun returns BOTH halves for a
// custom-tier definition — spread it into the definition: `...gaussianBlur({intensity: p('intensity')})`.

/**
 * The two halves a heavy blur adds to a definition: a pass that blurs the child ahead of time,
 * and the fragment that reads it.
 *
 * Spread the whole object into a definition that declares `species: 'custom'`,
 * `requiresRTT: true` and `requiresChild: true`.
 *
 * @example
 * ```ts
 * ...gaussianBlur({intensity: p('intensity')})
 * ```
 * @see gaussianBlur, channelBlur, progressiveBlur, tiltShift, bloom
 */
export interface ComputeBackedEffect {
    compute: GpuComputeNode
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/**
 * A soft, even blur of the child.
 *
 * `intensity` is a prop from 0 to about 200; 100 is a blur radius of about 36 pixels. Bind the
 * prop to a map to vary the blur across the canvas. Only color blurs; the child's alpha stays
 * sharp, so soft edges need a group or mask around the layer.
 *
 * @example
 * ```ts
 * ...gaussianBlur({intensity: p('intensity')})
 * ```
 * @tip The blur runs at a fixed working resolution, so its cost does not grow with the canvas.
 * @see channelBlur, progressiveBlur, tiltShift, bloom, motionBlur
 */
// A 2-pass separable Gaussian at fixed compute resolution (decoupled from canvas). A
// static/mouse/auto intensity runs the fixed-kernel path with a live per-frame radius read; an
// intensity bound to a map fills a per-pixel radius map from the map source and runs the variable
// Gaussian. Color comes from the blurred buffer, alpha from the sharp child.
export function gaussianBlur(opts: {intensity: PropRef}): ComputeBackedEffect {
    const prop = opts.intensity.name
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params
            const mapInfo = getMapInfo(prop)
            if (mapInfo) {
                return blurKit.withVariableBlurCompute(params, {
                    source: mapInfo,
                    buildFill: (cw, ch) => blurKit.buildFillBlurMapGraph(cw, ch, mapInfo.channel as blurKit.BlurMapChannel),
                    // inputWidth/Height map compute px → map-source canvas px; the window is the live
                    // post-dynamic-bound remap, matching the fragment path's `_map_<prop>_*` uniforms.
                    fillValues: (dims, window) => ({
                        inputWidth: dims.width,
                        inputHeight: dims.height,
                        ...blurKit.remapWindowValues(window!),
                    }),
                })
            }
            // Static intensity: fixed-kernel Gaussian (uniform radius, live per-frame CPU read so a
            // mouse / auto driver pulls through without a recompose).
            return blurKit.withFixedBlurCompute(params, {
                radius: () => {
                    const raw = getCpuValue(prop)
                    return typeof raw === 'number' ? blurKit.intensityToRadius(raw) : 4
                },
            })
        },
        gpu: {
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) => vec4(blurred.member('rgb'), sharp.member('a'))),
        },
    }
}

/**
 * Blurs red, green and blue by separate amounts, for a soft chromatic fringe.
 *
 * Each of `red`, `green` and `blue` is a prop from 0 to 100; 100 is a radius of about 10 pixels.
 * A channel at 0 stays exactly sharp.
 *
 * @example
 * ```ts
 * ...channelBlur({red: p('redIntensity'), green: p('greenIntensity'), blue: p('blueIntensity')})
 * ```
 * @see gaussianBlur, rgbSplit
 */
// One fixed Gaussian at the MAX per-channel radius, with each channel mixed between the sharp
// source and the blurred buffer by `channelRadius / maxRadius` — the max-intensity channel takes
// the full blur, a zero channel stays exactly sharp.
export function channelBlur(opts: {red: PropRef; green: PropRef; blue: PropRef}): ComputeBackedEffect {
    const {red, green, blue} = opts
    return {
        compute: (params) =>
            blurKit.withFixedBlurCompute(params, {
                // Live per-frame per-channel radii → the fixed Gaussian runs at their max.
                radius: () => {
                    const r = ((params.getCpuValue(red.name) as number) ?? 0) * blurKit.CHANNEL_INTENSITY_TO_RADIUS
                    const g = ((params.getCpuValue(green.name) as number) ?? 0) * blurKit.CHANNEL_INTENSITY_TO_RADIUS
                    const b = ((params.getCpuValue(blue.name) as number) ?? 0) * blurKit.CHANNEL_INTENSITY_TO_RADIUS
                    return Math.max(r, g, b, 0.01)
                },
            }),
        gpu: {
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) =>
                    call(blurKit.channelBlurCompose, 'channelBlurCompose', [
                        sharp, blurred, uniformOf(red, params), uniformOf(green, params), uniformOf(blue, params),
                    ])),
        },
    }
}

/**
 * A blur that ramps from sharp to full strength in one direction across the child.
 *
 * The ramp starts at `center` (a position in uv) and runs along `angle` (degrees, 0 points
 * right), reaching full blur `falloff` (0–1, in uv) past the center. `intensity` (0–100) is the
 * blur at the far end; 100 is a radius of about 36 pixels. Bind `intensity` to a map to vary the
 * ceiling per pixel. Alpha stays sharp.
 *
 * @example
 * ```ts
 * ...progressiveBlur({intensity: p('intensity'), angle: p('angle'), center: p('center'), falloff: p('falloff')})
 * ```
 * @see tiltShift, gaussianBlur
 */
// A variable-radius Gaussian whose per-pixel radius ramps directionally from `center` along
// `angle` over `falloff`, up to the intensity's max radius. A map-driven intensity samples the
// per-pixel max radius from the map source instead.
export function progressiveBlur(opts: {
    intensity: PropRef
    angle: PropRef
    center: PropRef
    falloff: PropRef
}): ComputeBackedEffect {
    const {intensity, angle, center, falloff} = opts
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params

            // Live per-frame geometry → fill-map params. `getCpuValue(center)` is the POST-transform
            // VectorFieldView (its `.y` is `1 - authoredY`; the kernel recovers y). angle is degrees.
            const readGeometry = (dims: {width: number; height: number}) => {
                const centerValue = getCpuValue(center.name) as {x?: number; y?: number} | undefined
                return {
                    angle: (getCpuValue(angle.name) as number) ?? 0,
                    centerX: typeof centerValue?.x === 'number' ? centerValue.x : 0,
                    centerY: typeof centerValue?.y === 'number' ? centerValue.y : 0.5,
                    falloff: (getCpuValue(falloff.name) as number) ?? 1,
                    aspect: dims.width / dims.height,
                }
            }

            // Intensity bound to a map → per-pixel max radius sampled from the map source.
            const mapInfo = getMapInfo(intensity.name)
            if (mapInfo) {
                return blurKit.withVariableBlurCompute(params, {
                    source: mapInfo,
                    buildFill: (cw, ch) => blurKit.buildProgressiveBlurFillMapGraph(cw, ch, mapInfo.channel as blurKit.BlurMapChannel),
                    fillValues: (dims, window) => ({
                        ...readGeometry(dims),
                        inputWidth: dims.width,
                        inputHeight: dims.height,
                        ...blurKit.remapWindowValues(window!),
                    }),
                })
            }

            // Static / mouse / auto intensity: single uniform max radius (per-frame CPU value).
            return blurKit.withVariableBlurCompute(params, {
                buildFill: blurKit.buildProgressiveBlurFillGraph,
                fillValues: (dims) => ({
                    ...readGeometry(dims),
                    maxRadius: blurKit.intensityToRadius((getCpuValue(intensity.name) as number) ?? 0),
                }),
            })
        },
        gpu: {
            // The variable blur already applied the per-pixel ramp → color from the blurred buffer,
            // alpha from the sharp child (blur can spread alpha at edges).
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) => vec4(blurred.member('rgb'), sharp.member('a'))),
        },
    }
}

/**
 * Keeps a band of the child in focus and blurs everything beyond it, like a tilt-shift lens.
 *
 * The band runs through `center` (a position in uv) at `angle` (degrees, 0 is horizontal) and
 * is `width` wide (0–1, in uv). Beyond it the blur ramps to full strength over `falloff` (0–1).
 * `intensity` (0–100) is the blur at the far edges. Pixels inside the band stay pixel-sharp.
 *
 * @example
 * ```ts
 * ...tiltShift({intensity: p('intensity'), width: p('width'), falloff: p('falloff'), angle: p('angle'), center: p('center')})
 * ```
 * @see progressiveBlur, gaussianBlur, bokehDefocus
 */
// A variable-radius Gaussian whose per-pixel radius ramps with perpendicular distance from a focus
// line (halfKernel 14 — the ~36px max radius keeps tap spacing under the banding threshold with
// ~40% fewer taps). The fragment re-derives the same blur amount to mix the canvas-res sharp
// source against the compute-res blurred buffer, so in-focus pixels stay crisp.
export function tiltShift(opts: {
    intensity: PropRef
    width: PropRef
    falloff: PropRef
    angle: PropRef
    center: PropRef
}): ComputeBackedEffect {
    const {intensity, width, falloff, angle, center} = opts
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params

            // Live per-frame geometry → fill-map params (same POST-transform center convention as
            // progressiveBlur).
            const readGeometry = (dims: {width: number; height: number}) => {
                const centerValue = getCpuValue(center.name) as {x?: number; y?: number} | undefined
                return {
                    angle: (getCpuValue(angle.name) as number) ?? 0,
                    centerX: typeof centerValue?.x === 'number' ? centerValue.x : 0.5,
                    centerY: typeof centerValue?.y === 'number' ? centerValue.y : 0.5,
                    width: (getCpuValue(width.name) as number) ?? 0.3,
                    falloff: (getCpuValue(falloff.name) as number) ?? 0.3,
                    aspect: dims.width / dims.height,
                }
            }

            const mapInfo = getMapInfo(intensity.name)
            if (mapInfo) {
                return blurKit.withVariableBlurCompute(params, {
                    halfKernel: 14,
                    source: mapInfo,
                    buildFill: (cw, ch) => blurKit.buildTiltShiftFillMapGraph(cw, ch, mapInfo.channel as blurKit.BlurMapChannel),
                    fillValues: (dims, window) => ({
                        ...readGeometry(dims),
                        inputWidth: dims.width,
                        inputHeight: dims.height,
                        ...blurKit.remapWindowValues(window!),
                    }),
                })
            }

            return blurKit.withVariableBlurCompute(params, {
                halfKernel: 14,
                buildFill: blurKit.buildTiltShiftFillGraph,
                fillValues: (dims) => ({
                    ...readGeometry(dims),
                    maxRadius: blurKit.intensityToRadius((getCpuValue(intensity.name) as number) ?? 0),
                }),
            })
        },
        gpu: {
            // Recompute the focus-line blur amount per fragment (cheap), then mix sharp ↔ blurred.
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) => {
                    const blurAmount = call(blurKit.tiltShiftBlurAmount, 'tiltShiftBlurAmount', [
                        uniformOf(angle, params), uniformOf(center, params), uniformOf(width, params),
                        uniformOf(falloff, params), params.ctx.uv, params.ctx.aspect,
                    ])
                    return mixExpr(sharp, blurred, blurAmount)
                }),
        },
    }
}

/**
 * A glow: the child's bright parts spread into a soft halo that adds back on top.
 *
 * Pixels brighter than `threshold` (0–1) glow. `size` is the halo's spread in pixels (clean up
 * to about 72). `intensity` (0–50) is how bright the halo is. The halo reaches past the child's
 * alpha, so a glowing shape lights up transparent space around it. At size 0 nothing runs; give
 * the size prop `recompile: crosses(0)`.
 *
 * @example
 * ```ts
 * ...bloom({intensity: p('intensity'), threshold: p('threshold'), size: p('size')})
 * ```
 * @see screenedBloom, gaussianBlur, bokehDefocus
 */
// Bright-extract the child above `threshold` at aspect-aware compute resolution, blur the extract
// by `size` (per-pixel from the map source when size carries a map driver), and composite
// `original + bloom × intensity`. A scalar size of 0 skips compute entirely; the shared fragment
// tail then falls back to a sharp passthrough.
export function bloom(opts: {intensity: PropRef; threshold: PropRef; size: PropRef}): ComputeBackedEffect {
    const {intensity, threshold, size} = opts
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params
            const mapInfo = getMapInfo(size.name)
            if (!mapInfo && ((getCpuValue(size.name) as number) ?? 0) === 0) return null // size=0 bypass (scalar path only).

            return blurKit.withBloomCompute(params, {
                mapInfo,
                buildExtract: blurKit.buildGlowPrepassGraph,
                buildExtractMap: blurKit.buildGlowPrepassMapGraph,
                threshold: () => (getCpuValue(threshold.name) as number) ?? 0.5,
                radius: () => (getCpuValue(size.name) as number) ?? 25,
            })
        },
        gpu: {
            // Composite at canvas resolution: original from the canvas-res child RTT (sharp); bloom
            // from the compute-res buffer (intentionally soft).
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (bloomSample, original) =>
                    call(blurKit.glowCompose, 'glowCompose', [original, bloomSample, uniformOf(intensity, params)])),
        },
    }
}

/**
 * The child's own color at this pixel, unchanged.
 *
 * The sampling stage for an effect that only adds something around the child, like a shadow.
 *
 * @example
 * ```ts
 * effect: gatherStack(childTap(), [shadowComposite({coverage, color: p('color'), intensity: p('intensity'), cutout: p('cutout')})])
 * ```
 * @see shadowComposite, crispTap
 */
// Premultiplied, like every plain tap of the child RTT.
export function childTap(): SampleStage {
    return (params) => params.texture.sample(params.ctx.uv)
}

/**
 * The coordinate a shadow reads the child's silhouette from: `distance` away in the compass
 * direction `angle`.
 *
 * `angle` is in degrees, 0 up, 90 right, 180 down. `distance` (0–1) is a fraction of the canvas
 * height and covers the same number of pixels in either axis. Pass the result as the `at` of
 * `silhouetteCoverage`.
 *
 * @example
 * ```ts
 * silhouetteCoverage({at: shadowOffset({angle: p('angle'), distance: p('distance')}), blur: p('blur')})
 * ```
 * @see silhouetteCoverage, shadowComposite
 */
export function shadowOffset(opts: {angle: ArgSpec; distance: ArgSpec}): (params: RttFilterParams) => Expr {
    return (params) =>
        call(motionBlurKit.dropShadowUV, 'dropShadowUV', [
            params.ctx.uv, params.ctx.viewportSize, resolveArg(opts.angle, params), resolveArg(opts.distance, params),
        ])
}

/**
 * A soft copy of the child's alpha silhouette: a number per pixel from 0 to 1.
 *
 * `at` is where to read the silhouette from, usually `shadowOffset`. `blur` is the softness as a
 * radius in pixels (0–20). Feed it to `shadowComposite` as the `coverage`.
 *
 * @example
 * ```ts
 * silhouetteCoverage({at: shadowOffset({angle: 135, distance: 0.1}), blur: p('blur')})
 * ```
 * @see shadowOffset, shadowComposite
 */
// Two-pass separable Gaussian over the child's alpha, running inline in the fragment with an
// intermediate RTT between the passes.
export function silhouetteCoverage(opts: {at: (params: RttFilterParams) => Expr; blur: ArgSpec}): (params: RttFilterParams) => Expr {
    return (params) =>
        motionBlurKit.silhouetteBlur({
            at: opts.at(params),
            uv: params.ctx.uv,
            viewportSize: params.ctx.viewportSize,
            blurRadius: resolveArg(opts.blur, params),
            sample: (coord) => params.texture.sample(coord),
            convertToTexture: params.convertToTexture,
        })
}

/**
 * A tinted shadow behind the child.
 *
 * `coverage` is the soft silhouette from `silhouetteCoverage`, tinted with `color` (a color
 * prop) at `intensity` (0–1) and placed under the child. `cutout` is a boolean prop with
 * `compileTime: true`; when true only the shadow shows, with the child's shape punched out of it.
 *
 * @example
 * ```ts
 * effect: gatherStack(childTap(), [shadowComposite({coverage: silhouetteCoverage({at: shadowOffset({angle: p('angle'), distance: p('distance')}), blur: p('blur')}), color: p('color'), intensity: p('intensity'), cutout: p('cutout')})])
 * ```
 * @see childTap, shadowOffset, silhouetteCoverage
 */
export function shadowComposite(opts: {
    coverage: (params: RttFilterParams) => Expr
    color: PropRef
    intensity: ArgSpec
    cutout: PropRef
}): OverlayStage {
    return (original, params) => {
        const shadowAlpha = opts.coverage(params).mul(resolveArg(opts.intensity, params))
        return call(motionBlurKit.dropShadowComposite, 'dropShadowComposite', [
            original, resolveArg(opts.color, params), shadowAlpha,
            floatE((params.propValues[opts.cutout.name] as number) === 1 ? 1 : 0),
        ])
    }
}

// ── Warp-shading recipes ────────────────────────────────────────────────────────────────────

/** Corner names → screenUV corner + opposite corner (y=0 top). Baked as compile-time literals. */
const PEEL_CORNERS: Record<string, {corner: [number, number]; opposite: [number, number]}> = {
    'top-left': {corner: [0, 0], opposite: [1, 1]},
    'top-right': {corner: [1, 0], opposite: [0, 1]},
    'bottom-left': {corner: [0, 1], opposite: [1, 0]},
    'bottom-right': {corner: [1, 1], opposite: [0, 0]},
}

/**
 * The geometry of fluted (reeded) glass over the child: where each pixel reads through the
 * glass, how far its colors split, and the slope of the flute at that point.
 *
 * `shape` is a `compileTime` prop whose transform maps 'bars', 'rounded' and 'waves' to 0, 1
 * and 2. `angle` is in degrees, 0 for vertical flutes. `frequency` is the number of flutes
 * across the longer edge (1–20). `softness` (0–1) eases each flute from flat-with-sharp-seams to
 * a gentle curve. `waveAmplitude` and `waveFrequency` shape the waves variant only.
 * `refraction` (0–4) is how hard each flute bends the child. `aberration` (0–1) is the color
 * split at the seams. The pattern drifts on the layer's clock, so declare
 * `animatedTime: {speed: 'speed'}`. Keep the frame in a module constant and pass it to
 * `refractedTaps` and `fluteHighlight`.
 *
 * @example
 * ```ts
 * const flute = fluteFrame({shape: p('shape'), angle: p('angle'), frequency: p('frequency'), softness: p('softness'), waveAmplitude: p('waveAmplitude'), waveFrequency: p('waveFrequency'), refraction: p('refraction'), aberration: p('aberration')})
 * ```
 * @see refractedTaps, fluteHighlight, GatherFrame
 */
// `shape` bakes the slope-exponent endpoints + waves flag as compile-time literals. The clock is
// negated so positive speed drifts visually right at angle 0.
export function fluteFrame(opts: {
    shape: PropRef
    angle: ArgSpec
    frequency: ArgSpec
    softness: ArgSpec
    waveAmplitude: ArgSpec
    waveFrequency: ArgSpec
    refraction: ArgSpec
    aberration: ArgSpec
}): GatherFrame {
    return frameOf((params) => {
        const shapeId = (params.propValues[opts.shape.name] as number) ?? 0
        const expHi = shapeId === 0 ? 16 : 8 // bars vs rounded/waves
        const expLo = shapeId === 0 ? 4 : 3
        const wavesFlag = shapeId === 2 ? 1 : 0
        // Speed negated so positive speed drifts +u (visually right at angle=0).
        const t = animatedTime(params).mul(-1)
        return asLocal(call(warpMaps.flutedGlassGeom, 'flutedGlassGeom', [
            params.ctx.uv, params.ctx.aspect, t,
            resolveArg(opts.angle, params), resolveArg(opts.frequency, params), resolveArg(opts.softness, params),
            resolveArg(opts.waveAmplitude, params), resolveArg(opts.waveFrequency, params),
            resolveArg(opts.refraction, params), resolveArg(opts.aberration, params),
            expr(`vec3f(${formatFloat(expHi)}, ${formatFloat(expLo)}, ${formatFloat(wavesFlag)})`),
        ]), 'flute')
    })
}

/**
 * The child seen through a flute frame, with edge handling and a color split when `aberration`
 * is above 0.
 *
 * `edges` is a prop with `transform: transformEdges` and `compileTime: true`. `aberration` is
 * the same prop the frame uses; give it a recompile rule that fires when it crosses 0. Returns
 * straight alpha, so close the stack with `{resultAlpha: 'straight'}`.
 *
 * @example
 * ```ts
 * effect: gatherStack(refractedTaps(flute, {aberration: p('aberration'), edges: p('edges')}), [fluteHighlight(flute, {lightAngle: 30, highlight: 0.2, softness: 0.3, color: p('highlightColor')})], {resultAlpha: 'straight'})
 * ```
 * @see fluteFrame, fluteHighlight, crispTap
 */
// `edges` binds the compile-time sampling branch; `aberration`'s on/off crossing is structural —
// one tap vs a 3-tap chromatic split along the frame's offset.
export function refractedTaps(frame: GatherFrame, opts: {aberration: PropRef; edges: PropRef}): SampleStage {
    return (params) => {
        const edgeMode = (params.propValues[opts.edges.name] as number) ?? 2
        const aberrationEnabled = ((params.propValues[opts.aberration.name] as number) ?? 0.2) > 0
        const geom = frame(params)
        const sampleAt = (uvE: Expr): Expr =>
            warpMaps.edgeClipSample((uv) => params.texture.sample(uv), uvE, edgeMode)
        const refractedUV = geom.member('refractedUV')
        const sampled = aberrationEnabled
            ? warpMaps.rgbSplitTaps(sampleAt, refractedUV, geom.member('chrOff'))
            : sampleAt(refractedUV)
        // RTT children come back premultiplied; the shading stages run on straight color.
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sampled])
    }
}

/**
 * A specular highlight running along each flute, added on top of the child's color.
 *
 * `lightAngle` is the light's direction in degrees, 0 head-on and 90 grazing (−90 to 90).
 * `highlight` (0–2) is its strength. `softness` (0–1) spreads the peak from pin-tight to a
 * broad sheen. `color` is a color prop. Transparent parts of the child pick up no highlight.
 *
 * @example
 * ```ts
 * fluteHighlight(flute, {lightAngle: p('lightAngle'), highlight: p('highlight'), softness: p('highlightSoftness'), color: p('highlightColor')})
 * ```
 * @see fluteFrame, refractedTaps, rimLit
 */
// Blinn specular over the frame's flute slope, weighted by alpha.
export function fluteHighlight(frame: GatherFrame, opts: {
    lightAngle: ArgSpec
    highlight: ArgSpec
    softness: ArgSpec
    color: PropRef
}): OverlayStage {
    return (color, params) => {
        const spec = call(warpMaps.blinnHighlight, 'blinnHighlight', [
            frame(params).member('slope'),
            resolveArg(opts.lightAngle, params), resolveArg(opts.highlight, params), resolveArg(opts.softness, params),
        ])
        const specA = spec.mul(color.member('a'))
        const litRgb = color.member('rgb').add(resolveArg(opts.color, params).member('rgb').mul(specA))
        return vec4(litRgb, color.member('a'))
    }
}

/**
 * One shading term of a page peel: a number per pixel read against the peel frame.
 *
 * `curlShading`, `curlSheen`, `overhangShadow` and `revealShadow` each make one; `peelCompose`
 * takes all four.
 * @see peelCompose, peelFrame
 */
export type PeelShade = (params: RttFilterParams) => Expr

/**
 * The geometry of a page curling up from one corner: where the curl reads the child from, how
 * far it has turned, and how far each pixel is from the crease.
 *
 * `corner` is a `compileTime` string prop ('top-left' | 'top-right' | 'bottom-left' |
 * 'bottom-right') with no transform. `amount` (0–1) is how far the peel has progressed, 0 flat
 * and 1 fully peeled. `radius` is the tightness of the curl as a fraction of the page diagonal
 * (0.02–0.4). Keep the frame in a module constant and pass it to the peel shading words and
 * `peelCompose`.
 *
 * @example
 * ```ts
 * const peel = peelFrame({corner: p('corner'), amount: p('amount'), radius: p('radius')})
 * ```
 * @tip Leave the `corner` prop without a `transform`: it must stay a string for the frame to read.
 * @see peelCompose, curlShading, curlSheen, overhangShadow, revealShadow
 */
// The corner is a cpu-only compile-time STRING prop (an inline transform would make the bridge
// write the string into an f32 field); its geometry lands as literal args.
export function peelFrame(opts: {corner: PropRef; amount: ArgSpec; radius: ArgSpec}): GatherFrame {
    return frameOf((params) => {
        const cornerGeom = PEEL_CORNERS[(params.propValues[opts.corner.name] as string) ?? 'bottom-right'] ?? PEEL_CORNERS['bottom-right']
        return asLocal(call(warpMaps.pagePeelGeom, 'pagePeelGeom', [
            params.ctx.uv, params.ctx.aspect,
            floatE(cornerGeom.corner[0]), floatE(cornerGeom.corner[1]),
            floatE(cornerGeom.opposite[0]), floatE(cornerGeom.opposite[1]),
            resolveArg(opts.amount, params), resolveArg(opts.radius, params),
        ]), 'peel')
    })
}

/**
 * The darkening in the crook of a curl, easing to full brightness at the lip.
 *
 * `shading` (0–1) is how deep the darkening goes.
 *
 * @example
 * ```ts
 * curlShading(peel, {shading: p('shading')})
 * ```
 * @see peelFrame, peelCompose, curlSheen
 */
export function curlShading(frame: GatherFrame, opts: {shading: ArgSpec}): PeelShade {
    return (params) =>
        call(warpMaps.curlShade, 'curlShade', [frame(params).member('theta'), resolveArg(opts.shading, params)])
}

/**
 * The band of light running up a curl.
 *
 * `highlight` (0–1) is its strength. `softness` (0–1) runs from a tight gloss to a broad satin.
 *
 * @example
 * ```ts
 * curlSheen(peel, {highlight: p('highlight'), softness: p('highlightSoftness')})
 * ```
 * @see peelFrame, peelCompose, curlShading
 */
export function curlSheen(frame: GatherFrame, opts: {highlight: ArgSpec; softness: ArgSpec}): PeelShade {
    return (params) =>
        call(warpMaps.curlSheen, 'curlSheen', [
            frame(params).member('theta'), resolveArg(opts.highlight, params), resolveArg(opts.softness, params),
        ])
}

/**
 * The shadow a lifted curl casts on the flat page just past the crease.
 *
 * `amount` is the same peel amount the frame uses. `shadow` (0–1) is the shadow's strength.
 *
 * @example
 * ```ts
 * overhangShadow(peel, {amount: p('amount'), shadow: p('shadow')})
 * ```
 * @see peelFrame, peelCompose, revealShadow
 */
export function overhangShadow(frame: GatherFrame, opts: {amount: ArgSpec; shadow: ArgSpec}): PeelShade {
    return (params) => {
        const g = frame(params)
        return call(warpMaps.overhangShade, 'overhangShade', [
            g.member('distPastCrease'), g.member('flatReach'),
            resolveArg(opts.amount, params), resolveArg(opts.shadow, params),
        ])
    }
}

/**
 * The shadow on the surface a peel reveals, just inside the crease.
 *
 * `amount` is the same peel amount the frame uses. `shadow` (0–1) is the shadow's strength.
 *
 * @example
 * ```ts
 * revealShadow(peel, {amount: p('amount'), shadow: p('shadow')})
 * ```
 * @see peelFrame, peelCompose, overhangShadow
 */
export function revealShadow(frame: GatherFrame, opts: {amount: ArgSpec; shadow: ArgSpec}): PeelShade {
    return (params) => {
        const g = frame(params)
        return call(warpMaps.revealShadow, 'revealShadow', [
            g.member('distIntoPeel'), g.member('peelReach'),
            resolveArg(opts.amount, params), resolveArg(opts.shadow, params),
        ])
    }
}

/**
 * A finished page peel: the cast shadows, the flat page and the lit curl, layered back to front.
 *
 * The sampling stage of a peel. Pass the four shading terms built on the same frame. Wrap it in
 * `gatherStack` with no overlays.
 *
 * @example
 * ```ts
 * effect: gatherStack(peelCompose(peel, {shade: curlShading(peel, {shading: 0.55}), sheen: curlSheen(peel, {highlight: 0.4, softness: 0.2}), overhang: overhangShadow(peel, {amount: p('amount'), shadow: 1}), reveal: revealShadow(peel, {amount: p('amount'), shadow: 1})}), [])
 * ```
 * @see peelFrame, curlShading, curlSheen, overhangShadow, revealShadow
 */
// Samples the child at the frame's curl UV (the lip) and at the screen UV (the flat page).
// Premultiplied — the gather species appends the unpremultiply tail.
export function peelCompose(frame: GatherFrame, parts: {
    shade: PeelShade
    sheen: PeelShade
    overhang: PeelShade
    reveal: PeelShade
}): SampleStage {
    return (params) => {
        const g = frame(params)
        const curlSample = params.texture.sample(g.member('curlUV'))
        const flatSample = params.texture.sample(params.ctx.uv)
        return call(warpMaps.pagePeelCompose, 'pagePeelCompose', [
            curlSample, flatSample,
            parts.shade(params), parts.sheen(params), parts.overhang(params), parts.reveal(params),
            g.member('foldBlend'), g.member('showCurl'),
        ])
    }
}

// ── Planar reflection ───────────────────────────────────────────────────────────────────────
//
// The mirror-floor recipe: reflect the content across a horizontal line, blur the mirror image
// progressively by its depth below the line, and fade it out with distance. Three stage parts —
// the reflected coordinate, the depth-ramped variable blur (compute), and the over-composite.

/** @internal */
// Reflected sample coordinate: mirror across the horizontal line at `lineY`. Y is down (top=0),
// so a point at uv.y > lineY reflects to `2·lineY - uv.y`. Pure function; reached through
// `mirrorAcrossRow`.
export const mirrorRowUV = tgpu.fn([d.vec2f, d.f32], d.vec2f)((uv, lineY) => {
    'use gpu'
    return d.vec2f(uv.x, lineY * 2.0 - uv.y)
})

/** @internal */
// Planar-reflection composite (pre-unpremultiply). Above the line returns the original content
// untouched; below it, the (edge-handled) reflection sample composited "over" the original via
// premultiplied alpha, faded by `visibility` (full up to distance−fadeWidth, smoothly to 0 by
// `distance`; `falloff` is the fade-zone width as a fraction of distance). Pure function; reached
// through `planarReflection`.
export const planarReflectionCompose = tgpu.fn([d.vec4f, d.vec4f, d.f32, d.f32, d.f32, d.f32], d.vec4f)(
    (original, refl, uvY, lineY, distance, falloff) => {
        'use gpu'
        const distBelow = std.max(uvY - lineY, 0.0)
        const distanceMax = std.max(distance, 0.001)
        const fadeWidth = std.max(distanceMax * falloff, 0.0001)
        const fadeStart = distanceMax - fadeWidth
        const visibility = 1.0 - std.smoothstep(fadeStart, distanceMax, distBelow)

        const effReflA = refl.w * visibility
        const oneMinusReflA = 1.0 - effReflA
        const belowRgb = refl.xyz.mul(visibility).add(original.xyz.mul(oneMinusReflA))
        const belowA = effReflA + original.w * oneMinusReflA
        const belowLine = d.vec4f(belowRgb, belowA)

        const isBelowLine = std.step(lineY, uvY)
        return std.mix(original, belowLine, d.vec4f(isBelowLine))
    },
)

/** Depth-ramp blur-map units: blur slider 1.0 → this many source pixels of Gaussian radius. */
const DEPTH_RAMP_RADIUS_PX = 12

/** Depth-ramp fill kernel params: line height + blur amount + (clamped) blur ramp distance. */
function depthRampFillParams() {
    return d.struct({height: d.f32, blurAmount: d.f32, blurDistance: d.f32})
}

/** @internal */
// GPU-free construction of the depth-ramp blur-map fill graph. Per compute pixel, the desired
// blur radius ramps with the row's depth below the line:
// `blur × smoothstep(0, blurDistance, height − v) × DEPTH_RAMP_RADIUS_PX` (source pixels).
// Row-based (depends only on cy/computeHeight); 2D dispatch, STORAGE textureStore. Reached
// through `depthRampBlur`.
export function buildDepthRampFillGraph(computeWidth: number, computeHeight: number) {
    void computeWidth // row-based fill; width kept for buildFill API parity.
    const Params = depthRampFillParams()
    const layout = tgpu.bindGroupLayout({
        blurMap: {storageTexture: d.textureStorage2d(blurKit.BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        // Y-down v — same convention as screenUV + the height prop.
        const v = (d.f32(cy) + 0.5) / computeHeight
        const distBelowAtOutput = std.max(p.height - v, 0.0)
        const ramp = std.smoothstep(0.0, p.blurDistance, distBelowAtOutput)
        const radius = p.blurAmount * ramp * DEPTH_RAMP_RADIUS_PX
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('depthRampFillBlurMap')

    return {layout, kernel, Params}
}

/**
 * The coordinate mirrored across a horizontal line, for reading a reflection below it.
 *
 * `line` is a prop, 0–1 down the canvas in uv. Returns a function of the fragment params that
 * gives the mirrored uv. Above the line the coordinate is still mirrored; `planarReflection`
 * only shows the reflection below it.
 *
 * @example
 * ```ts
 * const reflectedUV = mirrorAcrossRow(p('height'))
 * ```
 * @see planarReflection, depthRampBlur
 */
export function mirrorAcrossRow(line: PropRef): (params: GpuFragmentParams) => Expr {
    return ({ctx, uniforms}) => call(mirrorRowUV, 'mirrorRowUV', [ctx.uv, uniforms[line.name]])
}

/**
 * The blur half of a mirror floor: blurs the child more the further a row sits below the line.
 *
 * `line` is the same 0–1 line prop as `mirrorAcrossRow`. `blur` is the blur far below the line,
 * 0–5 where 1 is a radius of about 12 pixels. `blurDistance` (in uv, 0.01–1) is how far below
 * the line the blur takes to reach full strength. `halfKernel` is the number of taps each side
 * per pass; 30 is a good value. Spread it into a definition with `species: 'custom'` and read
 * the result in the fragment as `computeOutputs.blurredTexture`.
 *
 * @example
 * ```ts
 * ...depthRampBlur({line: p('height'), blur: p('blur'), blurDistance: p('blurDistance'), halfKernel: 30})
 * ```
 * @see planarReflection, mirrorAcrossRow, progressiveBlur
 */
// RTT the child, fill a per-pixel radius map from the line geometry, run the kit's variable
// Gaussian. Map-driven props use their static scalar (compute maps don't drive these per-pixel —
// the Blur/ProgressiveBlur precedent).
export function depthRampBlur(opts: {line: PropRef; blur: PropRef; blurDistance: PropRef; halfKernel: number}): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => blurKit.withVariableBlurCompute(params, {
            halfKernel: opts.halfKernel,
            buildFill: buildDepthRampFillGraph,
            fillValues: () => ({
                height: (params.getCpuValue(opts.line.name) as number) ?? 0.5,
                blurAmount: (params.getCpuValue(opts.blur.name) as number) ?? 1,
                // blurDistance is floored at 0.01 (smoothstep is undefined at edge0 == edge1).
                blurDistance: Math.max(0.01, (params.getCpuValue(opts.blurDistance.name) as number) ?? 0.5),
            }),
        }),
    }
}

/**
 * A mirror floor: the child as it is above a line, and its reflection fading out below it.
 *
 * `line` (0–1 in uv), `distance` (how far below the line the reflection stays visible, in uv)
 * and `falloff` (the width of the fade as a fraction of `distance`) are props. `edges` is a prop
 * with `transform: transformEdges` and `compileTime: true`. `original` is the child's color at
 * this pixel, `reflection` samples the child (or its blurred copy) at a coordinate, and
 * `reflectedUV` is usually `mirrorAcrossRow(line)`. Returns a fragment function whose result is
 * premultiplied; unpremultiply it before returning from `gpu.fragment`.
 *
 * @example
 * ```ts
 * planarReflection({line: p('height'), distance: p('distance'), falloff: p('falloff'), edges: p('edges')}, source.sample(ctx.uv), (uv) => blurred.sample(uv), mirrorAcrossRow(p('height')))(params)
 * ```
 * @see mirrorAcrossRow, depthRampBlur
 */
export function planarReflection(
    opts: {line: PropRef; distance: PropRef; falloff: PropRef; edges: PropRef},
    original: Expr,
    reflection: (uv: Expr) => Expr,
    reflectedUV: (params: GpuFragmentParams) => Expr,
): (params: GpuFragmentParams) => Expr {
    return (params) => {
        const {ctx, uniforms, propValues} = params
        const edgeMode = (propValues[opts.edges.name] as number) ?? 0
        const refl = edges.applyEdgeHandlingExpr(reflectedUV(params), reflection, edgeMode)
        return call(planarReflectionCompose, 'planarReflectionCompose', [
            original, refl, ctx.uv.member('y'), uniforms[opts.line.name], uniforms[opts.distance.name], uniforms[opts.falloff.name],
        ])
    }
}

// ── Screened highlight bloom + frame sway ───────────────────────────────────────────────────

/** @internal */
// Screen a tinted glow (blurred bright highlights) over a base color. `tintStrength = (rgb tint,
// strength)`. The glow's coverage extends the alpha into transparent areas (Glow's aura
// precedent) so the halo isn't clipped to the child's α. Reached through `screenedBloom`.
export const tintedScreenGlow = tgpu.fn([d.vec4f, d.vec4f, d.vec4f], d.vec4f)(
    (base, glow, tintStrength) => {
        'use gpu'
        const strength = tintStrength.w
        const bloomC = std.clamp(
            glow.xyz.mul(tintStrength.xyz).mul(strength),
            d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0),
        )
        const one = d.vec3f(1.0, 1.0, 1.0)
        const screened = one.sub(one.sub(base.xyz).mul(one.sub(bloomC)))
        const a = base.w + glow.w * strength * (1.0 - base.w)
        return d.vec4f(screened.x, screened.y, screened.z, std.clamp(a, 0.0, 1.0))
    },
).$name('tintedScreenGlow')

/**
 * A tinted glow of the child's bright parts, screened over a color you have already graded:
 * film halation, phosphor bloom.
 *
 * `strength` is a prop; at 0 the glow pass does not run, so give it `recompile: crosses(0)`.
 * `radius` is a prop for the glow's spread in pixels. `tint` is an rgb triple (0–1) and
 * `threshold` (0–1) the brightness a pixel must pass to glow. `output` and `extractName` are
 * names unique to your definition. Returns `compute`, to set as the definition's `compute:`, and
 * `screen(base, at, params)`, which screens the glow at coordinate `at` over `base` in the
 * fragment. The glow reaches past the child's alpha.
 *
 * @example
 * ```ts
 * const halation = screenedBloom({strength: p('halation'), radius: p('halationRadius'), tint: [1, 0.38, 0.16], threshold: 0.62, output: 'halationTexture', extractName: 'myHalationExtract'})
 * ```
 * @tip Build it once at module level so the `compute:` field and the fragment read the same part.
 * @see bloom, frameSway
 */
// The compute half is the kit's highlight-bloom mechanism (`withBloomCompute`: bright-extract +
// variable-Gaussian blur at capped compute resolution); the screen half tints the blurred
// highlights and screens them back over the graded color (no-op when the bloom buffer is absent —
// GPU-free resolve, or `strength` = 0, which returns null from `compute`).
export function screenedBloom(slots: {
    strength: PropRef
    radius: PropRef
    tint: [number, number, number]
    threshold: number
    output: string
    extractName: string
}): {
    compute: GpuComputeNode
    screen: (base: Expr, at: Expr, params: GpuFragmentParams) => Expr
} {
    return {
        compute: (params) => {
            const {getCpuValue} = params
            if (((getCpuValue(slots.strength.name) as number) ?? 0) === 0) return null // bloom off → no compute.

            return blurKit.withBloomCompute(params, {
                outputKey: slots.output,
                buildExtract: (w, h) => blurKit.buildBloomExtractGraph(w, h, slots.extractName),
                threshold: () => slots.threshold,
                radius: () => (getCpuValue(slots.radius.name) as number) ?? 28,
            })
        },
        screen: (base, at, params) => {
            const glowTex = params.computeOutputs?.[slots.output] as KitTexture | undefined
            const glowSample = glowTex ? glowTex.sample(at) : expr('vec4f(0.0, 0.0, 0.0, 0.0)')
            const tintStrength = vec4(slots.tint[0], slots.tint[1], slots.tint[2], params.uniforms[slots.strength.name])
            return call(tintedScreenGlow, 'tintedScreenGlow', [base, glowSample, tintStrength])
        },
    }
}

/** @internal */
// Frame sway — a tiny animated translation + rotation about the frame centre, like an unsteady
// projector gate. `amount` scales both amplitude and (via the gated clock) presence; `t` is the
// per-frame sway-time accumulator. Returns the UV to sample the content at (a clamp sampler
// handles the sliver pushed off-frame). amount=0 → returns `uv` unchanged. Reached through
// `frameSway`.
export const swayUV = tgpu.fn([d.vec2f, d.f32, d.f32], d.vec2f)(
    (uv, amount, t) => {
        'use gpu'
        const amp = amount * 0.006
        const ox = (std.sin(t * 1.7) * 0.5 + std.sin(t * 0.9 + 1.3) * 0.5) * amp
        const oy = (std.sin(t * 1.3 + 0.7) * 0.5 + std.sin(t * 2.3) * 0.5) * amp
        const ang = std.sin(t * 0.6) * amount * 0.004
        const c = std.cos(ang)
        const s = std.sin(ang)
        const p = uv.sub(d.vec2f(0.5, 0.5))
        const rx = p.x * c - p.y * s
        const ry = p.x * s + p.y * c
        return d.vec2f(rx + 0.5 + ox, ry + 0.5 + oy)
    },
).$name('swayUV')

/**
 * A slight drifting shift and rotation of where the child is read from, like film in an
 * unsteady projector gate.
 *
 * `amount` is a prop from 0 to 1. `field` names an entry you declare in the definition's
 * `extraFields` as `{schema: schema.f32, initial: 0}`; the sway's clock lives there. Returns a
 * function of the fragment params that gives the uv to sample the child at. When `amount` drops
 * to 0 the motion freezes in place rather than snapping back.
 *
 * @example
 * ```ts
 * const weave = frameSway({amount: p('weave'), field: 'weaveTime'})
 * ```
 * @tip Build it once at module level and call it inside `gpu.fragment`; it registers its own per-frame clock.
 * @see screenedBloom
 */
// A gated clock (the `field` extraField — advances only while `amount` > 0) drives the sway.
export function frameSway(slots: {amount: PropRef; field: string}): (params: GpuFragmentParams) => Expr {
    return (params) => {
        let acc = 0
        params.onBeforeRender(({deltaTime}) => {
            const w = (params.getCpuValue(slots.amount.name) as number) ?? 0
            acc += deltaTime * (w > 0 ? 1 : 0)
            params.setExtraField(slots.field, acc)
        })
        return call(swayUV, 'swayUV', [params.ctx.uv, params.uniforms[slots.amount.name], params.uniforms[slots.field]])
    }
}

// ── Bokeh defocus ───────────────────────────────────────────────────────────────────────────
//
// The photographic lens-blur noun over the kit's aperture-table gather: RTT the composed child,
// run ONE scatter-as-gather pass over the CPU-precomputed aperture tap table at aspect-aware
// compute resolution, and bilinear-sample the defocused buffer at canvas resolution (true
// defocus — color AND coverage spread, so silhouettes soften like a real lens). The gather is
// in input-pixel space so discs stay round regardless of compute resolution.

const BOKEH_DEG_TO_RAD = constants.DEG_TO_RAD
const BOKEH_GATHER_FORMAT = 'rgba16float' as const

/** BakedTable part — the aperture tap table: regenerate + rewrite the uniform only when the
 *  (shape, blades) key changes. */
function apertureTable(
    opts: {shape: PropRef; blades: PropRef},
    getCpuValue: GpuFragmentParams['getCpuValue'],
    tapsUniform: {write: (value: d.v4f[]) => void},
): () => void {
    let key = ''
    return () => {
        const shape = (getCpuValue(opts.shape.name) as string) ?? 'blades'
        const count = (getCpuValue(opts.blades.name) as number) ?? 6
        const next = `${shape}|${count}`
        if (next === key) return
        key = next
        tapsUniform.write(blurKit.generateBokehTaps(shape, count, blurKit.BOKEH_TAP_COUNT).map((tap) => d.vec4f(tap.x, tap.y, tap.rim, 0)))
    }
}

/**
 * A photographic lens blur where bright highlights bloom into aperture-shaped discs.
 *
 * `radius` (0–100) is the defocus; 100 is a disc about 80 pixels across. `gain` (0–10) is how
 * strongly highlights bloom and `threshold` (0–1) the brightness they must pass. `shape` is a
 * string prop: 'blades' | 'circle' | 'star' | 'heart' | 'flower' | 'cross' | 'ring'. `blades`
 * (0–9) counts blades or points for the blades, star and flower shapes; 0–2 is a round iris.
 * `rotation` is in degrees and `fringe` (0–1) adds color fringing at the disc edges. Both color
 * and coverage blur, so silhouettes soften like a real lens. Changing the aperture never
 * recompiles. Spread it into a definition with `species: 'custom'`, `requiresRTT: true` and
 * `requiresChild: true`.
 *
 * @example
 * ```ts
 * ...bokehDefocus({radius: p('radius'), gain: p('highlightGain'), threshold: p('highlightThreshold'), shape: p('bladeShape'), blades: p('bladeCount'), rotation: p('bladeRotation'), fringe: p('chromaticFringe')})
 * ```
 * @see gaussianBlur, tiltShift, bloom
 */
// The aperture-table gather compute (uniform-radius, or the map-driven fork when `radius` binds a
// spatial map — mouse/auto radius stays a per-frame scalar) + the defocused-buffer sampling
// fragment. The aperture (shape + blades) is fully RUNTIME: changing it regenerates the CPU tap
// table and rewrites one small uniform — no recompile. Compute unavailable (GPU-free resolve / no
// device) → sharp unpremultiplied passthrough.
export function bokehDefocus(opts: {
    radius: PropRef
    gain: PropRef
    threshold: PropRef
    shape: PropRef
    blades: PropRef
    rotation: PropRef
    fringe: PropRef
}): {compute: GpuComputeNode; gpu: {fragment: (params: GpuFragmentParams) => Expr}} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {childNode, gpu, convertToTexture, registerComputeTexture, getCpuValue, getMapInfo, onCleanup, onResize, dimensions} = params
            if (!childNode) return null
            const root = gpu?.root
            if (!root) return null // GPU-free resolve/tests: fragment falls back to sharp passthrough.

            const childTexture = convertToTexture(childNode)
            // `dimensions` is already the device-pixel backing size (the renderer owns DPR).
            let curWidth = Math.max(1, Math.round(dimensions.width))
            let curHeight = Math.max(1, Math.round(dimensions.height))

            // Aspect-aware compute resolution: cap the LONGER edge at the default long edge and derive the
            // other from the canvas aspect (a fixed 1024×640 would squash the discs on non-16:10 canvases).
            const LONG_EDGE = Math.max(blurKit.DEFAULT_COMPUTE_WIDTH, blurKit.DEFAULT_COMPUTE_HEIGHT)
            const aspect = curHeight > 0 ? curWidth / curHeight : 1
            const computeWidth = Math.max(8, aspect >= 1 ? LONG_EDGE : Math.round(LONG_EDGE * aspect))
            const computeHeight = Math.max(8, aspect >= 1 ? Math.round(LONG_EDGE / aspect) : LONG_EDGE)

            // Gathered-output buffer at compute res: written by the gather (storage), sampled by the fragment.
            const outputTex = root.createTexture({size: [computeWidth, computeHeight], format: BOKEH_GATHER_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => outputTex.destroy())
            const blurredTexture = registerComputeTexture(outputTex)

            onResize(({width, height}) => {
                curWidth = Math.max(1, Math.round(width))
                curHeight = Math.max(1, Math.round(height))
            })

            // Look controls (per-frame CPU values; rotation degrees → a cos/sin pair for the kernel).
            const readLook = () => {
                const rot = ((getCpuValue(opts.rotation.name) as number) ?? 0) * BOKEH_DEG_TO_RAD
                return {
                    highlightGain: (getCpuValue(opts.gain.name) as number) ?? 4,
                    highlightThreshold: (getCpuValue(opts.threshold.name) as number) ?? 0.6,
                    rotCos: Math.cos(rot),
                    rotSin: Math.sin(rot),
                    chromaticFringe: (getCpuValue(opts.fringe.name) as number) ?? 0.2,
                }
            }

            // radius bound to a map → per-pixel radius sampled from the map source.
            const mapInfo = getMapInfo(opts.radius.name)
            if (mapInfo) {
                const graph = blurKit.buildBokehMapGraph(computeWidth, computeHeight, blurKit.BOKEH_TAP_COUNT, mapInfo.channel as blurKit.BokehMapChannel)
                const kernelParams = root.createUniform(graph.Params)
                const tapsUniform = root.createUniform(graph.TapArray)
                const ensureTaps = apertureTable({shape: opts.shape, blades: opts.blades}, getCpuValue, tapsUniform)
                let bokehPass = createGuardedCompute(root, (cx: number, cy: number) => {
                    'use gpu'
                    graph.kernel(cx, cy)
                }, {size: [computeWidth, computeHeight]})

                return {
                    outputs: {blurredTexture},
                    bindInputs: (resolve) => {
                        const src = resolve(childTexture.key)
                        const mapSrc = resolve(mapInfo.sourceTexture.key)
                        if (!src || !mapSrc) return
                        const bindGroup = root.createBindGroup(graph.layout, {
                            input: src.texture as never,
                            source: mapSrc.texture as never,
                            output: outputTex,
                            params: kernelParams.buffer,
                            taps: tapsUniform.buffer,
                        })
                        bokehPass = bokehPass.with(bindGroup)
                    },
                    getComputeNodes: () => {
                        ensureTaps()
                        const w = mapInfo.window()
                        kernelParams.write({
                            ...readLook(),
                            inputWidth: curWidth,
                            inputHeight: curHeight,
                            inputMin: w.inputMin,
                            inputMax: w.inputMax,
                            outputMin: w.outputMin,
                            outputMax: w.outputMax,
                            curve: w.curve,
                        })
                        return [bokehPass]
                    },
                }
            }

            // ── Static / mouse / auto radius: single uniform gather radius (per-frame CPU value). ──
            const graph = blurKit.buildBokehGraph(computeWidth, computeHeight, blurKit.BOKEH_TAP_COUNT)
            const kernelParams = root.createUniform(graph.Params)
            const tapsUniform = root.createUniform(graph.TapArray)
            const ensureTaps = apertureTable({shape: opts.shape, blades: opts.blades}, getCpuValue, tapsUniform)
            let bokehPass = createGuardedCompute(root, (cx: number, cy: number) => {
                'use gpu'
                graph.kernel(cx, cy)
            }, {size: [computeWidth, computeHeight]})

            return {
                outputs: {blurredTexture},
                // The child RTT is allocated after composition → build the gather bind group once it exists.
                bindInputs: (resolve) => {
                    const src = resolve(childTexture.key)
                    if (!src) return
                    const bindGroup = root.createBindGroup(graph.layout, {
                        input: src.texture as never,
                        output: outputTex,
                        params: kernelParams.buffer,
                        taps: tapsUniform.buffer,
                    })
                    bokehPass = bokehPass.with(bindGroup)
                },
                getComputeNodes: () => {
                    ensureTaps()
                    const radius = (getCpuValue(opts.radius.name) as number) ?? 50
                    kernelParams.write({
                        radius: blurKit.bokehRadiusToPixels(radius),
                        ...readLook(),
                        inputWidth: curWidth,
                        inputHeight: curHeight,
                    })
                    return [bokehPass]
                },
            }
        },

        gpu: {
            fragment: ({childNode, computeOutputs, ctx, convertToTexture}: GpuFragmentParams): Expr => {
                if (!childNode) return ZERO

                const blurred = computeOutputs?.blurredTexture as KitTexture | undefined
                if (!blurred) {
                    // Compute unavailable (GPU-free resolve / no device): sharp passthrough, unpremultiplied.
                    const tex = convertToTexture(childNode)
                    return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [tex.sample(ctx.uv)])
                }

                // The gather produced a full defocus in premultiplied space (color + coverage). Sample it at
                // canvas resolution and unpremultiply back into the straight-alpha blend pipeline.
                return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [blurred.sample(ctx.uv)])
            },
        },
    }
}
