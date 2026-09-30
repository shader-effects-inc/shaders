/**
 * Words for drawing light: glows, rays, beams, flares, halos, and the frames they are drawn
 * in. A frame is the local coordinate system a light evaluates in. Parts are shaped over a
 * frame, `withFrame` evaluates a part in one, `modulate` multiplies parts, `additive` sums
 * finished lights, and the heat, coverage and screen words turn a light field into color.
 */
// Maintainer notes.
// Part constructors are slot-taking — they bind `p()` prop refs (or scalar graphs / literals)
// and lower them against the composition params — so a shader file states its recipe as
// visible composition: a frame first, parts over it, one composite at the end.
// The GPU bodies live in the kit (`kit/lightfields`, `kit/gradientPaints` for the godrays and
// light-leak stages, `kit/shapePaints` for the lens-flare stack).
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE, vec4, mixExpr, asLocal} from '../../gpu/composer'
import {add, div, max, mul, normalize, pow, reflect, sin, smoothstep, sub, vec3} from '../math'
import {animatedTime} from '../../gpu/porters'
import {lightfields, gradientPaints, shapePaints} from '../../gpu/kit/index'
import type {FilterParams} from '../../gpu/scaffolds/pointwiseFilter'
import type {PointwiseEffect} from '../types'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {uniformOf, resolveScalar, type ArgSpec} from '../invoke'

/**
 * A light or a frame waiting for the shader's params. Call it with `params` inside a
 * `paint` or `effect` function to get its value at this pixel.
 */
export type LightValue = (params: GpuFragmentParams) => Expr

// ── Concept words (Expr-level — no slots, compose anywhere) ──────────────────────────────

/**
 * Bright bands that travel along a coordinate, for aurora curtains or shimmering rays.
 *
 * Returns a 0–1 field. `along` is the coordinate the bands cross (a uv axis, or a
 * noise-warped one for organic rays), `frequency` the band count per unit of it, and
 * `time` × `speed` how fast they travel. `sharpness` above 1 narrows the bright bands.
 *
 * @example
 * ```ts
 * const rays = rayBands({along: params.ctx.uv.member('x'), frequency: 12, time: animatedTime(params), speed: 0.3, sharpness: 2})
 * ```
 * @tip Add a noise field to `along` before passing it and the straight bands turn into curtains.
 * @see rayLobes, streakBand
 */
export function rayBands(opts: {
    along: Expr
    frequency: Expr | number
    time: Expr | number
    speed: Expr | number
    phase?: Expr | number
    sharpness: Expr | number
}): Expr {
    // sin(along·frequency + time·speed + phase) lifted to [0, 1], then pow(sharpness).
    let arg: Expr = add(mul(opts.along, opts.frequency), mul(opts.time, opts.speed))
    if (opts.phase !== undefined) arg = add(arg, opts.phase)
    return pow(add(mul(sin(arg), 0.5), 0.5), opts.sharpness)
}

/**
 * How light spilling from an emitter dims with distance, with a long realistic tail.
 *
 * Returns 1 at the emitter, softened over `radius`, then fading like real light. `d` and
 * `radius` share a unit (usually aspect-corrected uv). `power` 1 fades like a line of light,
 * `power` 2 like a point. Use it where `glowPoint` looks too much like a soft blob.
 *
 * @example
 * ```ts
 * const spill = emitterFalloff(frame.member('x'), {radius: p('radius'), power: 2})
 * ```
 * @see glowPoint, glowSpot
 */
export function emitterFalloff(d: Expr, opts: {radius: Expr | number; power: Expr | number}): Expr {
    // (radius / (radius + d))^power — 1/d tail at power 1 (line emitter), 1/d² at power 2 (point).
    return pow(div(opts.radius, add(opts.radius, max(d, 0))), opts.power)
}

/**
 * A soft round glow at a position on the canvas.
 *
 * Returns a 0–1 field, 1 at the position. `x` and `y` are in uv (0–1, y down), `aspect` is
 * the canvas width over height so the glow stays round, and `sizeSq` is the glow radius
 * squared. Pass several with different positions and sum them with `additive`.
 *
 * @example
 * ```ts
 * const glow = glowAt({uv: params.ctx.uv, aspect: params.ctx.aspect, x: 0.5, y: 0.4, sizeSq: 0.02})
 * ```
 * @tip The radius goes in squared: a glow 0.15 across the canvas is `sizeSq: 0.0225`.
 * @see glowSpot, glowPoint, radialFrame
 */
export function glowAt(opts: {uv: Expr; aspect: Expr | number; x: Expr | number; y: Expr | number; sizeSq: Expr | number}): Expr {
    // exp(−0.5·dist²/sizeSq). Takes the SQUARED radius because callers driving several glows
    // from CPU-computed data (StudioBackground's orbits) square once per frame, not per pixel.
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    return call(lightfields.radialGaussianGlow, 'glowAt', [opts.uv, e(opts.aspect), e(opts.x), e(opts.y), e(opts.sizeSq)])
}

/**
 * The surface normals of a flat disc treated as a puffy dome, ready to light with `shine`.
 *
 * `delta` is the offset from the disc center, `dist` its length and `radius` the disc size,
 * all in the same unit. `bulge.tilt` (default 0.2) sets how far the normals lean outward at
 * the rim and `bulge.drop` (default 0.1) how much the dome flattens toward its edge.
 *
 * @example
 * ```ts
 * const highlight = shine({normal: domeNormal({delta, dist, radius}), light: vec3(0.3, -0.5, 0.8), gloss: 32})
 * ```
 * @see shine
 */
export function domeNormal(opts: {
    delta: Expr
    dist: Expr
    radius: Expr | number
    bulge?: {tilt?: number; drop?: number}
}): Expr {
    // Geometry only — the Bevel & Emboss family's opening move ("make it look puffy").
    // Defaults are the house dome.
    const tilt = opts.bulge?.tilt ?? 0.2
    const drop = opts.bulge?.drop ?? 0.1
    const slope = asLocal(smoothstep(0, opts.radius, opts.dist), 'domeSlope')
    const leaned = add(slope, tilt)
    return normalize(vec3(
        mul(opts.delta.member('x'), leaned),
        mul(opts.delta.member('y'), leaned),
        sub(1, mul(slope, drop)),
    ))
}

/**
 * A specular highlight on any surface that gives you a normal.
 *
 * Returns the raw 0–1 highlight, viewed straight on. Multiply it by your intensity and the
 * shape's coverage yourself. `light` is a unit direction toward the light and `gloss` sets
 * the highlight's tightness (8 is broad, 32 is a small hot spot). Works with `domeNormal`,
 * a bevel, or a heightfield slope.
 *
 * @example
 * ```ts
 * const highlight = mul(shine({normal, light: vec3(0.3, -0.5, 0.8), gloss: p('gloss')}), mask)
 * ```
 * @see domeNormal
 */
export function shine(opts: {normal: Expr; light: Expr; gloss: Expr | number}): Expr {
    // The specular lobe of a unit light direction against `normal`, viewed along +z (the 2.5D
    // worldview): pow(max(0, reflect(−light, normal).z), gloss).
    return pow(max(0, reflect(mul(opts.light, -1), opts.normal).member('z')), opts.gloss)
}

/**
 * A light shaped over a frame's coordinates. Parts for `radialFrame` read distance and
 * angle (`vec2(dist, angle)`); parts for `beamFrame` read distance along and across the
 * beam (`vec2(u, v)`). Evaluate one with `withFrame`, combine several with `modulate`.
 */
export type LightPart = (frame: Expr, params: GpuFragmentParams) => Expr

// ── Shared resolution helpers ───────────────────────────────────────────────────────────

/** Resolve a scalar-ish slot (prop, scalar graph, ctx token, or number literal) to an Expr. */
function arg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

/** The composed UV + effective viewport every generator paint evaluates against. */
function paintFrame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

// ── Frames ──────────────────────────────────────────────────────────────────────────────

/**
 * The frame for a burst, halo or sweep around a center prop.
 *
 * Gives each pixel its distance and angle from the center as `vec2(dist, angle)`. Distance
 * is in aspect-corrected uv (0.5 is half the canvas height), the angle in radians. `center`
 * is a position prop. Evaluate parts in it with `withFrame`.
 *
 * @example
 * ```ts
 * coverage: withFrame(radialFrame(p('center')), modulate(rayLobes({count: p('rays'), spin: -1, sharpness: 4}), featherMask({radius: p('radius'), feather: 0.5})), 'burst')
 * ```
 * @see withFrame, seamlessRadialFrame, beamFrame
 */
export function radialFrame(center: PropRef): LightValue {
    // `center` is a transformPosition prop (stores 1 − y); the kit body recovers the authored y.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(lightfields.radialBurstFrame, 'radialFrame', [uv, viewport, uniformOf(center, params)])
    }
}

/**
 * The frame `noiseRays` needs: a radial frame around a center prop with no visible seam.
 *
 * Patterns that repeat around an angle show a line where the angle wraps. This frame carries
 * what a part needs to hide it. Only `noiseRays` reads it. `center` is a position prop.
 *
 * @example
 * ```ts
 * coverage: withFrame(seamlessRadialFrame(p('center')), noiseRays({density: p('density'), intensity: p('intensity'), spotty: p('spotty')}))
 * ```
 * @tip The other radial parts (`glowPoint`, `rayLobes`, `featherMask`) read `radialFrame`, not this one. Its layout is different.
 * @see noiseRays, radialFrame
 */
export function seamlessRadialFrame(center: PropRef): LightValue {
    // Returns vec4(angle, angleWrapped, blend, dist): the seamlessAngularField triple + radius.
    // Parts evaluate their pattern on both atan2 branches and cross-fade across the seam.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(gradientPaints.godraysFrame, 'godraysFrame', [uv, viewport, uniformOf(center, params)])
    }
}

/**
 * The frame of a beam that starts at an anchor prop and points at the canvas center.
 *
 * Gives each pixel its distance along the beam and its sideways offset from it as
 * `vec2(u, v)`, in aspect-corrected uv. Drag the anchor anywhere and the beam still points
 * inward. Built for filters over a child (a light leak inside `screenGlow`), where it reads
 * the child's uv.
 *
 * @example
 * ```ts
 * const frame = local(beamFrame(p('position'))(params), 'leakFrame')
 * ```
 * @see beamBloom, beamStreaks, screenGlow
 */
export function beamFrame(anchor: PropRef): LightValue {
    // Reads the raw filter context (params.ctx.uv / ctx.aspect — a pointwise effect's frame),
    // not paintFrame(): this is the one frame authored for `effect:` recipes.
    return (params) => call(lightfields.beamLocalFrame, 'beamFrame', [params.ctx.uv, params.ctx.aspect, uniformOf(anchor, params)])
}

/**
 * Evaluate a light part in a frame.
 *
 * This is where a framed light starts: a frame, then the part (or `modulate` of several)
 * drawn in it. Pass a name as the third argument when the part reads the frame more than
 * once, so the frame is computed once per pixel. `modulate` always reads it more than once.
 *
 * @example
 * ```ts
 * coverage: withFrame(radialFrame(p('center')), modulate(softRayLobes({count: p('rayCount'), softness: p('softness'), spin: -1}), featherMask({radius: p('radius'), feather: p('feather')})), 'burstFrame')
 * ```
 * @tip Without the name, a `modulate` of two parts computes the frame twice.
 * @see radialFrame, beamFrame, modulate
 */
export function withFrame(frame: LightValue, part: LightPart, hint?: string): LightValue {
    // The composition root of a framed light recipe. `hint` binds the frame as a WGSL local.
    return (params) => {
        const f = hint ? asLocal(frame(params), hint) : frame(params)
        return part(f, params)
    }
}

/**
 * Multiply light parts drawn in the same frame, such as a mask over a pattern.
 *
 * @example
 * ```ts
 * modulate(rayLobes({count: 12, spin: -1, sharpness: 4}), glowPoint({sharpness: 6}), featherMask({radius: p('radius'), feather: 0.4}))
 * ```
 * @see withFrame, additive
 */
export function modulate(first: LightPart, ...rest: LightPart[]): LightPart {
    return (frame, params) => rest.reduce((acc, part) => acc.mul(part(frame, params)), first(frame, params))
}

// ── Radial-frame parts ──────────────────────────────────────────────────────────────────

/**
 * Soft rays fanning out from the center of a radial frame, turning with the layer's clock.
 *
 * Returns a 0–1 field with `count` rays. `spin` is a plain number that scales the clock:
 * 1 turns counter-clockwise, -1 clockwise, 0 holds still. `sharpness` above 1 narrows the
 * bright rays, below 1 widens them.
 *
 * @example
 * ```ts
 * coverage: withFrame(radialFrame(p('center')), rayLobes({count: p('rays'), spin: -1, sharpness: 4}))
 * ```
 * @see softRayLobes, raySpikes, radialFrame
 */
export function rayLobes(slots: {count: ArgSpec; spin: number; sharpness: ArgSpec}): LightPart {
    // Reads the frame angle (.y). pow(sin(angle·count + clock·spin)·0.5 + 0.5, sharpness).
    return (frame, params) => call(lightfields.angularSineLobes, 'rayLobes', [
        frame.member('y'), arg(slots.count, params), animatedTime(params).mul(slots.spin), arg(slots.sharpness, params),
    ])
}

/**
 * `rayLobes` with a softness slider instead of a sharpness.
 *
 * `softness` 0 gives crisp rays and 1 gives wide soft ones. The rays widen without moving.
 * `spin` is a plain number scaling the layer's clock: negative turns clockwise.
 *
 * @example
 * ```ts
 * coverage: withFrame(radialFrame(p('center')), softRayLobes({count: p('rayCount'), softness: p('softness'), spin: -1}))
 * ```
 * @see rayLobes, featherMask
 */
export function softRayLobes(slots: {count: ArgSpec; softness: ArgSpec; spin: number}): LightPart {
    // Softness inverts into the lobe exponent in the kit (0.3 / (softness + 0.05)).
    return (frame, params) => call(lightfields.softRayLobes, 'softRayLobes', [
        frame.member('y'), arg(slots.count, params), animatedTime(params).mul(slots.spin), arg(slots.softness, params),
    ])
}

/**
 * Thin star spikes with dark gaps between them, like light through a camera aperture.
 *
 * Returns a 0–1 field with twice `halfCount` spikes, so pass half the spike count you want.
 * `power` from 3 to 8 makes them thinner. The spikes do not turn.
 *
 * @example
 * ```ts
 * coverage: withFrame(radialFrame(p('center')), modulate(raySpikes({halfCount: 3, power: 5}), glowPoint({sharpness: 12})))
 * ```
 * @see rayLobes, flareStarburst
 */
export function raySpikes(slots: {halfCount: ArgSpec; power: ArgSpec}): LightPart {
    // Reads the frame angle (.y). pow(abs(cos(angle·halfCount)), power) — abs gives the nulls.
    return (frame, params) => call(lightfields.angularCosineSpikes, 'raySpikes', [
        frame.member('y'), arg(slots.halfCount, params), arg(slots.power, params),
    ])
}

/**
 * Three thin colored rings around the center, the rainbow edge of a lens or a halo.
 *
 * Returns an RGB, not a field. The red ring sits at `center` + `spread`, green at `center`,
 * blue at `center` - `spread`, each `width` wide, all in the frame's distance unit. A
 * positive `spread` puts red outside, as a real lens does.
 *
 * @example
 * ```ts
 * const rings = withFrame(radialFrame(p('center')), ringBand({center: p('radius'), spread: 0.01, width: 0.02}))(params)
 * ```
 * @see flareHalo, glowPoint
 */
export function ringBand(slots: {center: ArgSpec; spread: ArgSpec; width: ArgSpec}): LightPart {
    // Reads the frame distance (.x). Width is floored at 1e-4 in the kit.
    return (frame, params) => call(lightfields.chromaticRingBand, 'ringBand', [
        frame.member('x'), arg(slots.center, params), arg(slots.spread, params), arg(slots.width, params),
    ])
}

/**
 * A soft glow at the center of a radial frame with no hard edge.
 *
 * Returns a 0–1 field, 1 at the center. `sharpness` is one over the radius squared, so a
 * larger value gives a tighter glow: for a glow that reaches radius `r`, pass `1 / (r * r)`.
 *
 * @example
 * ```ts
 * coverage: withFrame(radialFrame(p('center')), glowPoint({sharpness: 8}))
 * ```
 * @tip Sharpness is not a radius. Turning it up makes the glow smaller.
 * @see featherMask, emitterFalloff, glowAt
 */
export function glowPoint(slots: {sharpness: ArgSpec}): LightPart {
    // Reads the frame distance (.x). exp(−(dist²·sharpness)).
    return (frame, params) => call(lightfields.radialGaussianFalloff, 'glowPoint', [frame.member('x'), arg(slots.sharpness, params)])
}

/**
 * A round mask that is solid inside a radius and fades out at its edge.
 *
 * Returns 1 inside, fading to 0 at `radius` (in the frame's distance unit). `feather` is the
 * fraction of the radius spent fading: 0 is a hard edge, 1 fades all the way from the
 * center. Multiply it with a pattern in `modulate` to give a light an outer edge.
 *
 * @example
 * ```ts
 * modulate(rayLobes({count: 12, spin: -1, sharpness: 4}), featherMask({radius: p('radius'), feather: p('feather')}))
 * ```
 * @see glowPoint, modulate
 */
export function featherMask(slots: {radius: ArgSpec; feather: ArgSpec}): LightPart {
    // Reads the frame distance (.x). 1 − smoothstep(radius·(1 − feather), radius, dist).
    return (frame, params) => call(lightfields.radialFeatherMask, 'featherMask', [
        frame.member('x'), arg(slots.radius, params), arg(slots.feather, params),
    ])
}

/**
 * Volumetric god rays: shafts of light drifting outward from the center, moving on the
 * layer's clock.
 *
 * Returns a 0–1 coverage. `density` sets how many shafts, `intensity` (0–1) how visible they
 * are, and `spotty` breaks each shaft into drifting spots along its length. Draw it in
 * `seamlessRadialFrame`, not `radialFrame`.
 *
 * @example
 * ```ts
 * coverage: withFrame(seamlessRadialFrame(p('center')), noiseRays({density: p('density'), intensity: p('intensity'), spotty: p('spotty')}))
 * ```
 * @see seamlessRadialFrame, coverageOver
 */
export function noiseRays(slots: {density: ArgSpec; intensity: ArgSpec; spotty: ArgSpec}): LightPart {
    // Two seam-free noise layers accumulated over the whole vec4 frame (angle, wrapped angle,
    // blend, radius). The schedule constants live in the kit and ARE the look.
    return (frame, params) => call(gradientPaints.godraysRayStack, 'godraysRayStack', [
        frame, animatedTime(params),
        arg(slots.density, params), arg(slots.intensity, params), arg(slots.spotty, params),
    ])
}

// ── Beam-glow parts (a light leak's stages) ─────────────────────────────────────────────

/**
 * The slow breathing and drifting signals that keep a light leak alive.
 *
 * Call the result with a clock and the params. It returns four numbers: `.x` is a
 * brightness gain to multiply your light by (1 with `flicker` at 0, breathing between 0.25
 * and 1.75 at 1), and the rest are the slow drifts `beamBloom` and `beamStreaks` read. Pass
 * the whole value to them as `drift`. `seed` re-rolls the pattern.
 *
 * @example
 * ```ts
 * const drift = local(slowDrift({seed: p('seed'), flicker: p('flicker')})(animatedTime(params), params), 'leakDrift')
 * ```
 * @see beamBloom, beamStreaks, beamFrame
 */
export function slowDrift(slots: {seed: ArgSpec; flicker: ArgSpec}): (clock: Expr, params: GpuFragmentParams) => Expr {
    // vec4(flickerMod, streakDrift, elongationDrift, widthDrift) from layered value noise.
    return (clock, params) => call(gradientPaints.leakDrift, 'leakDrift', [
        clock, arg(slots.seed, params), arg(slots.flicker, params),
    ])
}

/**
 * The soft pool of light at the start of a beam, with a wider faint shoulder around it.
 *
 * Call the result with a `beamFrame` value, a `slowDrift` value and the params. It returns
 * `vec2(main, shoulder)`, two 0–1 fields you usually add together. `spread` is the pool's
 * size in the frame's unit (aspect-corrected uv), at least 0.05.
 *
 * @example
 * ```ts
 * const bloom = beamBloom({spread: p('spread')})(frame, drift, params)
 * const heat = add(mul(bloom.member('x'), 1.25), bloom.member('y'))
 * ```
 * @see beamFrame, slowDrift, beamStreaks
 */
export function beamBloom(slots: {spread: ArgSpec}): (frame: Expr, drift: Expr, params: GpuFragmentParams) => Expr {
    // Anisotropic gaussian at the anchor (main, sharpness 1.9) + shoulder (0.45 × 0.22),
    // elongated laterally by the drift's elongation channel (.z).
    return (frame, drift, params) => call(gradientPaints.leakBloom, 'leakBloom', [
        frame.member('x'), frame.member('y'), arg(slots.spread, params), drift,
    ])
}

/**
 * Two bright bands crossing a beam further along it, slowly drifting, like light bouncing
 * inside a film camera.
 *
 * Call the result with a `beamFrame` value, a `slowDrift` value and the params. It returns
 * a field scaled by `strength` (0 turns the bands off). `spread` is the same value you gave
 * `beamBloom`, so the bands sit past the pool.
 *
 * @example
 * ```ts
 * const streaks = beamStreaks({spread: p('spread'), strength: p('streaks')})(frame, drift, params)
 * ```
 * @see beamBloom, slowDrift, streakBand
 */
export function beamStreaks(slots: {spread: ArgSpec; strength: ArgSpec}): (frame: Expr, drift: Expr, params: GpuFragmentParams) => Expr {
    // Two streakBand calls at 0.42 and 0.8 of spread along u, drifting via .y/.w, faded across v.
    return (frame, drift, params) => call(gradientPaints.leakStreaks, 'leakStreaks', [
        frame.member('x'), frame.member('y'), arg(slots.spread, params), arg(slots.strength, params), drift,
    ])
}

/**
 * Adds a trace of noise to a heat field so long soft ramps do not show bands.
 *
 * Pass the summed heat, the pixel's uv and a clock. Use it right before `heatRamp`. This is
 * for a heat field, not a color: use `compose.dithered` for a finished color.
 *
 * @example
 * ```ts
 * return heatRamp({hot: p('colorHot'), mid: p('colorMid'), fringe: p('colorFringe')}).taps(dithered(heat, params.ctx.uv, animatedTime(params)), params)
 * ```
 * @see heatRamp
 */
export function dithered(heat: Expr, uv: Expr, clock: Expr): Expr {
    // hash12 over uv·1024 + clock, ±0.006 on the heat, floored at 0.
    return call(gradientPaints.heatDither, 'heatDither', [heat, uv, clock])
}

// ── Expr-level primitives (for bespoke recipes over hand-built coordinates) ─────────────

/**
 * A soft oval glow over an offset you have already scaled, the core of a bloom or a glare.
 *
 * Returns a 0–1 field. Divide each axis of the offset by its own radius first: `du` and
 * `dv` at different radii make the glow oval. `sharpness` larger gives a tighter glow.
 *
 * @example
 * ```ts
 * const pool = glowSpot(div(dx, p('width')), div(dy, p('height')), 2)
 * ```
 * @see glowAt, glowPoint, beamBloom
 */
export function glowSpot(du: Expr, dv: Expr, sharpness: Expr): Expr {
    // exp(−(du² + dv²)·sharpness).
    return call(lightfields.anisotropicGaussianSpot, 'glowSpot', [du, dv, sharpness])
}

/**
 * One soft band across a single coordinate, peaking at `center`.
 *
 * Returns a 0–1 field, 1 at `center`, fading over `width` on each side. `u`, `center` and
 * `width` share a unit. Multiply or add several for a stack of parallel bands.
 *
 * @example
 * ```ts
 * const bar = streakBand(params.ctx.uv.member('y'), 0.5, p('width'))
 * ```
 * @see rayBands, beamStreaks
 */
export function streakBand(u: Expr, center: Expr, width: Expr): Expr {
    // exp(−(u − center)² / width²).
    return call(lightfields.streakBand, 'streakBand', [u, center, width])
}

/**
 * Where a pixel falls along a line from `a` to `b`, and how far it is from that line.
 *
 * Returns `vec2(t, distance)`. `t` runs 0 at `a` to 1 at `b` and is clamped, so the ends
 * are rounded. All three points share a unit. Feed both values to `lightBeam`.
 *
 * @example
 * ```ts
 * const seg = segmentProject(uv, p('start'), p('end'))
 * ```
 * @see lightBeam
 */
export function segmentProject(p: Expr, a: Expr, b: Expr): Expr {
    // The zero-length guard (max(lengthSq, 1e-4)) keeps coincident endpoints finite.
    return call(lightfields.pointToSegment, 'segmentProject', [p, a, b])
}

/**
 * The glow of a beam that can be thicker or softer at one end than the other.
 *
 * Give it the distance from the line and the position along it from `segmentProject`.
 * Thickness and softness blend from `start` to `end` along the beam, in the distance unit.
 * Returns `vec2(colorT, alpha)`: `alpha` is 1 inside the beam and fades to 0 outside, and
 * `colorT` runs 0 at the core to 1 at the edge so you can shade the two differently.
 *
 * @example
 * ```ts
 * const beam = lightBeam(seg.member('y'), seg.member('x'), {start: p('startThickness'), end: p('endThickness')}, {start: p('startSoftness'), end: p('endSoftness')})
 * ```
 * @see segmentProject
 */
export function lightBeam(
    dist: Expr, t: Expr,
    thickness: {start: Expr; end: Expr},
    softness: {start: Expr; end: Expr},
): Expr {
    // Alpha is a smoothstep shoulder raised to 1 + softness·1.5 (soft beams fade non-linearly);
    // softness is floored at 1e-4 in the kit so a zero slider stays defined.
    return call(lightfields.taperedSegmentGlow, 'lightBeam', [
        dist, t, thickness.start, thickness.end, softness.start, softness.end,
    ])
}

// ── Lens-flare stack parts ──────────────────────────────────────────────────────────────
//
// Each part reads the shared flare frame — a struct local with members `aspect`, `lightPos`,
// `flareAxis`, `lightAngle`, `lightDist`, `masterFade` — bound once by the shader file.

/**
 * The shared frame of a camera lens flare, which every `flare*` part reads.
 *
 * `position` is the light's position prop. `intensity` scales the whole flare and
 * `edgeFade` (0–1) dims it as the light nears the canvas edge. Bind it once with a name and
 * hand the same value to each part, then close the stack with `flareComposite`.
 *
 * @example
 * ```ts
 * const flare = local(flareFrame({position: p('lightPosition'), intensity: p('intensity'), edgeFade: p('edgeFade')})(params), 'flare')
 * ```
 * @see flareCore, flareGlare, flareHalo, flareStarburst, flareStreak, lensGhost, flareComposite
 */
export function flareFrame(slots: {position: PropRef; intensity: ArgSpec; edgeFade: ArgSpec}): LightValue {
    // The struct carries light position (unflipped), flare axis (toward the frame centre),
    // master fade (intensity × edge fade × shimmer) and the pixel's polar coords about the light.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(shapePaints.flareFrame, 'flareFrame', [
            uv, viewport, animatedTime(params), uniformOf(slots.position, params),
            arg(slots.intensity, params), arg(slots.edgeFade, params),
        ])
    }
}

/**
 * One ghost disc of a lens flare, the faint circles that line up across from the light.
 *
 * Call the result once per ghost with the flare frame, a `disc` and a `tint`, and sum the
 * calls with `additive`. `disc` is `[offset, size, brightness, hollow]`: how far along the
 * axis from the light (scaled by `spread`), radius in aspect-corrected uv, brightness, and
 * 0–1 for a solid disc to a ring. `tint` is `[r, g, b, phase]`, the color and a shimmer
 * offset. `chroma` (0–1) adds a rainbow rim. Returns an RGB.
 *
 * @example
 * ```ts
 * const ghost = lensGhost({intensity: p('ghostIntensity'), spread: p('ghostSpread'), chroma: p('ghostChroma')})
 * const ghosts = additive(ghost(flare, [0.15, 0.20, 0.30, 0.10], [1.0, 0.95, 0.90, 0.0], params), ghost(flare, [0.30, 0.12, 0.25, 0.65], [1.0, 0.85, 0.55, 2.17], params))
 * ```
 * @see flareFrame, additive, flareHalo
 */
export function lensGhost(slots: {intensity: ArgSpec; spread: ArgSpec; chroma: ArgSpec}) {
    // `disc`/`tint` are baked per call so the ghost table unrolls at the call site (a JS array
    // can't index inside a GPU body). Ghost position = lightPos + flareAxis·offset·spread·2.
    return (flare: Expr, disc: [number, number, number, number], tint: [number, number, number, number], params: GpuFragmentParams): Expr => {
        const {uv} = paintFrame(params)
        return call(shapePaints.lensFlareGhost, 'lensFlareGhost', [
            uv, flare.member('aspect'), flare.member('lightPos'), flare.member('flareAxis'), animatedTime(params),
            arg(slots.spread, params), arg(slots.chroma, params), arg(slots.intensity, params),
            vec4(...disc), vec4(...tint),
        ])
    }
}

/**
 * The rainbow halo ring of a lens flare, drawn across the canvas from the light.
 *
 * Call the result with the flare frame and the params. The ring centers on the point
 * opposite the light through the canvas center. `radius` is in aspect-corrected uv,
 * `chroma` (0–1) splits it into colors and `softness` widens it. Returns an RGB.
 *
 * @example
 * ```ts
 * flareHalo({intensity: p('haloIntensity'), radius: p('haloRadius'), chroma: p('haloChroma'), softness: p('haloSoftness')})(flare, params)
 * ```
 * @see flareFrame, ringBand, lensGhost
 */
export function flareHalo(slots: {intensity: ArgSpec; radius: ArgSpec; chroma: ArgSpec; softness: ArgSpec}) {
    // chromaticRingBand about the anti-light point with a slow angular brightness variation.
    return (flare: Expr, params: GpuFragmentParams): Expr => {
        const {uv} = paintFrame(params)
        return call(shapePaints.flareHalo, 'flareHalo', [
            uv, flare.member('aspect'), flare.member('lightPos'), animatedTime(params),
            arg(slots.intensity, params), arg(slots.radius, params),
            arg(slots.chroma, params), arg(slots.softness, params),
        ])
    }
}

/**
 * The star of thin spikes around the light in a lens flare.
 *
 * Call the result with the flare frame and the params. `points` is the number of spikes.
 * They turn slowly with the layer's clock and carry a faint rainbow tint. Returns an RGB.
 *
 * @example
 * ```ts
 * flareStarburst({intensity: p('starburstIntensity'), points: p('starburstPoints')})(flare, params)
 * ```
 * @see flareFrame, raySpikes
 */
export function flareStarburst(slots: {intensity: ArgSpec; points: ArgSpec}) {
    // Two angularCosineSpikes sets (powers 3.5 and 7, the second offset 0.25 rad) — the pairing
    // is what reads as glass rather than a star primitive.
    return (flare: Expr, params: GpuFragmentParams): Expr =>
        call(shapePaints.flareStarburst, 'flareStarburst', [
            flare.member('lightAngle'), flare.member('flareAxis'), flare.member('aspect'), animatedTime(params),
            flare.member('lightDist'), arg(slots.points, params), arg(slots.intensity, params),
        ])
}

/**
 * The horizontal widescreen streak through the light in a lens flare.
 *
 * Call the result with the flare frame and the params. `length` is the streak's reach in
 * uv. Blue spreads furthest, giving the streak its cyan edges. Returns an RGB.
 *
 * @example
 * ```ts
 * flareStreak({intensity: p('streakIntensity'), length: p('streakLength')})(flare, params)
 * ```
 * @see flareFrame, streakBand
 */
export function flareStreak(slots: {intensity: ArgSpec; length: ArgSpec}) {
    // Per-channel extents (×0.7 / ×1 / ×1.4 of lenSq), vertical falloff exp(−800·dy²).
    return (flare: Expr, params: GpuFragmentParams): Expr => {
        const {uv} = paintFrame(params)
        return call(shapePaints.flareStreak, 'flareStreak', [
            uv, flare.member('lightPos'), arg(slots.length, params), arg(slots.intensity, params),
        ])
    }
}

/**
 * The broad warm-white wash around the light that lifts contrast near it.
 *
 * Call the result with the flare frame and the params. `size` is the wash radius in
 * aspect-corrected uv. Returns an RGB.
 *
 * @example
 * ```ts
 * flareGlare({intensity: p('glareIntensity'), size: p('glareSize')})(flare, params)
 * ```
 * @see flareFrame, flareCore
 */
export function flareGlare(slots: {intensity: ArgSpec; size: ArgSpec}) {
    // radialGaussianFalloff with sharpness 1 / (size² + 0.01), tinted (1, 0.98, 0.95).
    return (flare: Expr, params: GpuFragmentParams): Expr =>
        call(shapePaints.flareGlare, 'flareGlare', [
            flare.member('lightDist'), arg(slots.size, params), arg(slots.intensity, params),
        ])
}

/**
 * The bright point at the light itself: a hard pinpoint inside a small soft glow.
 *
 * Call the result with the flare frame and the params. Returns an RGB.
 *
 * @example
 * ```ts
 * flareCore({intensity: p('intensity')})(flare, params)
 * ```
 * @see flareFrame, flareGlare
 */
export function flareCore(slots: {intensity: ArgSpec}) {
    return (flare: Expr, params: GpuFragmentParams): Expr =>
        call(shapePaints.flareCore, 'flareCore', [flare.member('lightDist'), arg(slots.intensity, params)])
}

/**
 * Closes a lens flare: turns the summed flare parts into the color to return from `paint`.
 *
 * Pass the `additive` sum of the parts and the flare frame. The frame's intensity, edge fade
 * and shimmer are applied here, and alpha comes from how bright the flare is.
 *
 * @example
 * ```ts
 * return flareComposite(additive(ghosts, halo, starburst, streak, glare, core), flare)
 * ```
 * @see additive, flareFrame
 */
export function flareComposite(total: Expr, flare: Expr): Expr {
    // total × masterFade, alpha = clamp(luma601 × 2).
    return call(shapePaints.flareComposite, 'flareComposite', [total, flare.member('masterFade')])
}

// ── Masks, ramps, composites ────────────────────────────────────────────────────────────

/**
 * Darkens the corners of the canvas.
 *
 * Returns a field that is 1 at the center and falls toward 0 in the corners, kept round on
 * any canvas shape. Multiply your color by it. `strength` 0 leaves the image alone and
 * about 2 takes the corners to black. `scale` multiplies `strength`, for mapping a 0–100
 * slider onto that range.
 *
 * @example
 * ```ts
 * const vig = vignetteMask({strength: p('vignette'), scale: 0.025})(params)
 * return vec4(mul(rgb, vig), 1)
 * ```
 * @see featherMask
 */
export function vignetteMask(slots: {strength: ArgSpec; scale?: number}): LightValue {
    // clamp(1 − dist²·strength, 0, 1) about the frame centre, aspect from the effective viewport.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        let strength = arg(slots.strength, params)
        if (slots.scale !== undefined) strength = strength.mul(slots.scale)
        return call(lightfields.vignetteMask, 'vignetteMask', [uv, viewport, strength])
    }
}

/**
 * Adds lights together, since light from several sources simply sums.
 *
 * @example
 * ```ts
 * const total = additive(glare, halo, core)
 * ```
 * @see modulate, flareComposite
 */
export function additive(first: Expr, ...rest: Expr[]): Expr {
    return rest.reduce((acc, part) => acc.add(part), first)
}

/**
 * The colors of overexposed light, from a dim fringe through a mid tone to a hot core.
 *
 * Pass three color props. Call `.taps(heat, params)` on the result to turn a heat field into
 * RGB: heat 0 is black, the fringe color shows first, the mid color by 0.5, the hot color
 * above 1, and the light keeps getting brighter up to a heat of 1.6. Each channel reads a
 * slightly different heat, which gives bright edges their color fringing.
 *
 * @example
 * ```ts
 * return heatRamp({hot: p('colorHot'), mid: p('colorMid'), fringe: p('colorFringe')}).taps(heat, params)
 * ```
 * @tip Run the heat through `dithered` first on long soft ramps.
 * @see dithered, screenGlow
 */
export function heatRamp(colors: {hot: PropRef; mid: PropRef; fringe: PropRef}) {
    // fringe → mid over heat 0–0.5, mid → hot over 0.55–1.15, energy × clamp(heat, 0, 1.6);
    // channels tap at ×1.08 / ×1 / ×0.88 for the chromatic fringe.
    return {
        kind: 'heatRamp' as const,
        /** The RGB of a heat field through this ramp. */
        taps: (heat: Expr, params: {uniforms: Record<string, Expr>}): Expr =>
            call(lightfields.chromaticHeatTaps, 'chromaticHeatTaps', [
                heat,
                uniformOf(colors.hot, params).member('rgb'),
                uniformOf(colors.mid, params).member('rgb'),
                uniformOf(colors.fringe, params).member('rgb'),
            ]),
    }
}
/** What `heatRamp` returns: call its `taps(heat, params)` to color a heat field. */
export type HeatRamp = ReturnType<typeof heatRamp>

/**
 * Blends from a background color to a light color by a coverage field, as a finished paint.
 *
 * `color` and `background` are color props, `coverage` a 0–1 light such as a `withFrame`
 * result. Both the color and the alpha blend, so a transparent background gives a light that
 * fades to nothing. Pass a `hint` name when the coverage is expensive. Use `coverageOver`
 * when you want real layer compositing instead of a plain blend.
 *
 * @example
 * ```ts
 * paint: coverageMix({color: p('color'), background: p('background'), coverage: withFrame(radialFrame(p('center')), glowPoint({sharpness: 6}), 'glow'), hint: 'glowAlpha'})
 * ```
 * @see coverageOver, withFrame
 */
export function coverageMix(slots: {color: PropRef; background: PropRef; coverage: LightValue; hint?: string}): LightValue {
    // Component-wise mix of rgb AND alpha; `hint` binds the coverage as a WGSL local.
    return (params) => {
        const alpha = slots.hint ? asLocal(slots.coverage(params), slots.hint) : slots.coverage(params)
        const col = uniformOf(slots.color, params)
        const bg = uniformOf(slots.background, params)
        const rgb = mixExpr(bg.member('rgb'), col.member('rgb'), alpha)
        const a = mixExpr(bg.member('a'), col.member('a'), alpha)
        return vec4(rgb, a)
    }
}

/**
 * Lays a light color over a background color by a coverage field, like a layer with an
 * opacity mask, as a finished paint.
 *
 * `color` and `background` are color props, `coverage` a 0–1 light. The color's own alpha
 * counts as opacity, so a half-transparent ray color shows the background through it.
 *
 * @example
 * ```ts
 * paint: coverageOver({color: p('rayColor'), background: p('backgroundColor'), coverage: withFrame(seamlessRadialFrame(p('center')), noiseRays({density: p('density'), intensity: p('intensity'), spotty: p('spotty')}))})
 * ```
 * @see coverageMix, noiseRays
 */
export function coverageOver(slots: {color: PropRef; background: PropRef; coverage: LightValue}): LightValue {
    // Straight-alpha over in linear RGB, un-premultiplied with a 0.001 alpha floor.
    return (params) => call(gradientPaints.coverageOverComposite, 'coverageOver', [
        uniformOf(slots.color, params), uniformOf(slots.background, params), slots.coverage(params),
    ])
}

/**
 * An `effect` that shines emitted light onto the layer inside it, the way light hits film.
 *
 * Pass a function that builds the light's RGB from the params. The light is screened over
 * the child, pushed toward overexposure where it is bright, and shows even over transparent
 * areas. Build the light with `beamFrame`, `beamBloom`, `beamStreaks` and `heatRamp`.
 *
 * @example
 * ```ts
 * effect: screenGlow((params) => heatRamp({hot: p('colorHot'), mid: p('colorMid'), fringe: p('colorFringe')}).taps(heat, params))
 * ```
 * @see heatRamp, beamFrame
 */
export function screenGlow(glow: (params: FilterParams) => Expr): PointwiseEffect {
    // Screen blend + ×0.16 additive push + alpha raised by the light's Rec.601 luma (×0.85).
    // The pointwise composition root of a light-bleed filter.
    return {
        kind: 'pointwise',
        body: {fn: lightfields.screenGlowComposite, hint: 'screenGlowComposite'},
        args: (params) => [glow(params)],
    }
}
