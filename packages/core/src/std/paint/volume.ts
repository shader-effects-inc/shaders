/**
 * std/paint/volume — the inside of a shape.
 *
 * Words for filling a shape with something you look through: smoke, gas, murky glass, a
 * starfield behind a window. You work inside a shape-effect material (the `surface:` of
 * `shapedSurface`). `interiorRay` gives the line of sight through the shape, `turbulentMedium`
 * says how dense the cloud is at any point inside, and `volumeMarch` walks the ray front to
 * back, adding up what each sample glows and how much it dims what lies behind. The result
 * is a color plus a transmittance you multiply anything behind the medium by.
 */
// Maintainer notes. The medium itself (what glows, what absorbs, at each sample) is the look:
// it stays algebra at the call site. The integrator owns the compositing so self-shadowing and
// depth occlusion are structural, not re-derived per effect. Chord entry/length ride the
// volumetric field taps (`s0.w` depth, the shared `opticalThickness` reading).
import type {Expr} from '../../gpu/contract'
import {add, exp, local, mul, neg, splat3, vec3} from '../math'
import {volumeNoiseAt} from './materials'
import type {SurfaceFrame, SurfaceField} from './materials'
import {opticalThickness} from './materials'

/** The line of sight through a shape's interior, from `interiorRay`. */
export interface InteriorRay {
    /** The depth at which the ray enters the shape. 0 for flat shapes. */
    entry: Expr
    /** How far the ray travels inside the shape, roughly 0–1. */
    length: Expr
    /** The position inside the shape at depth `z` along the ray. */
    at(z: Expr): {x: Expr; y: Expr}
}

/**
 * The line of sight through a shape: where it enters, how far it travels inside, and where it is at any depth.
 *
 * Pass the material's `field` and `frame`, the point the ray enters at (`origin`, usually the
 * shape's centred placement coordinates) and optionally the `view` ray. `shear` (default 1)
 * sets how far deeper samples slide sideways along the view, which is what makes an interior
 * read as deep rather than flat. `lengthScale` stretches or shortens the travel. On a 3D shape
 * `length` is the real chord through it. On a flat shape it stands in for depth, growing
 * toward the middle of the shape.
 *
 * @example
 * ```ts
 * const ray = interiorRay(field, frame, {origin: interior, view, shear: 0.55})
 * ```
 * @tip Refract `origin` through the shell first (`shellRefract`) and the interior bends at the glass walls.
 * @see volumeMarch, parallaxPlane, turbulentMedium
 */
export function interiorRay(
    field: SurfaceField,
    frame: SurfaceFrame,
    opts: {origin: {x: Expr; y: Expr}; view?: Expr; shear?: number; lengthScale?: number},
): InteriorRay {
    // Volumetric taps carry the chord entry depth in `.w`; flat shapes enter at 0 (`volFlag`).
    const entry = local(mul(field.s0.member('w'), frame.volFlag), 'rayEntry')
    // The shared optical-thickness reading (chord on marched shapes, rim proxy on flat),
    // scaled by the material's own factor.
    const length = local(
        opts.lengthScale === undefined
            ? opticalThickness(field, frame)
            : mul(opticalThickness(field, frame), opts.lengthScale),
        'rayLen',
    )
    // Origin sheared along the view ray's in-plane direction: the perspective parallax of
    // looking INTO a solid.
    const shearX = opts.view ? local(mul(opts.view.member('x'), opts.shear ?? 1), 'rayShearX') : undefined
    const shearY = opts.view ? local(mul(opts.view.member('y'), opts.shear ?? 1), 'rayShearY') : undefined
    return {
        entry,
        length,
        at(z: Expr) {
            return {
                x: shearX ? add(opts.origin.x, mul(shearX, z)) : opts.origin.x,
                y: shearY ? add(opts.origin.y, mul(shearY, z)) : opts.origin.y,
            }
        },
    }
}

/**
 * A flat layer at a fixed depth inside the shape, like scenery seen through a window.
 *
 * `depth` is a fraction of the ray's length: 0 at the entry, 1 at the far side, beyond 1 for a
 * backdrop behind the shape. Give it a `pan` (the `rotationSensor` of the frame) and the layer
 * slides as the shape rotates, `panRate` times as far. Deeper layers with higher rates give
 * the parallax of real depth. Returns the position to sample content at, plus its depth `z`.
 *
 * @example
 * ```ts
 * const far = parallaxPlane(ray, {depth: 1.1, pan: rotationSensor(frame), panRate: 1.3, hint: 'far'})
 * ```
 * @tip Two planes at different depths and rates are enough to sell a starfield.
 * @see interiorRay, volumeMarch
 */
export function parallaxPlane(
    ray: InteriorRay,
    opts: {depth: number; pan?: {x: Expr; y: Expr}; panRate?: number; hint?: string},
): {x: Expr; y: Expr; z: Expr} {
    // Counter-panned by the rotation sensor so rotating the shape pans the plane.
    const hint = opts.hint ?? 'plane'
    const z = local(add(ray.entry, mul(ray.length, opts.depth)), `${hint}Z`)
    const base = ray.at(z)
    if (!opts.pan) return {x: base.x, y: base.y, z}
    const rate = opts.panRate ?? 1
    return {
        x: add(base.x, mul(opts.pan.x, rate)),
        y: add(base.y, mul(opts.pan.y, rate)),
        z,
    }
}

/**
 * A billowing cloud density at a point inside the shape.
 *
 * `frequency` sets how many cloud cells fit across the shape. `drift` is a time value that
 * moves the clouds. `seed` moves to a different part of the cloud so two instances differ.
 * `billow` (0–1) folds the clouds over themselves for a cumulus look. `density` is signed,
 * roughly −1 to 1: threshold it with `smoothstep` to make clouds with clear gaps. `mid` and
 * `fine` are two finer layers of the same cloud, free to use for dust bands or wisps.
 *
 * @example
 * ```ts
 * const gas = turbulentMedium({x: pos.x, y: pos.y, z}, {frequency: math.mul(u.gasScale, 3), drift: t, billow: math.float(0.4)})
 * ```
 * @tip Inside a `volumeMarch`, pass the step number as `hint` so each step gets its own values.
 * @see volumeMarch, interiorRay
 */
export function turbulentMedium(
    p: {x: Expr; y: Expr; z: Expr},
    opts: {frequency: Expr; drift?: Expr; seed?: Expr; billow?: Expr; hint?: string},
): {density: Expr; mid: Expr; fine: Expr} {
    // The house turbulence: one-octave domain warp (the folds) over a 3-octave fbm, drifting
    // on `drift` and decorrelated per axis by `seed`. The raw `mid`/`fine` octaves double as
    // free detail fields at zero extra noise cost.
    const h = opts.hint ?? ''
    const drift = opts.drift
    const seed = opts.seed
    const axis = (v: Expr, freqScale: number | undefined, driftScale: number, seedScale: number): Expr => {
        let e: Expr = freqScale === undefined ? mul(v, opts.frequency) : mul(v, mul(opts.frequency, freqScale))
        if (drift) e = add(e, driftScale === 1 ? drift : mul(drift, driftScale))
        if (seed) e = add(e, seedScale === 1 ? seed : mul(seed, seedScale))
        return e
    }
    const q = local(vec3(
        axis(p.x, undefined, 1, 1),
        axis(p.y, undefined, -0.7, 0.53),
        axis(p.z, 1.25, 1.3, 0.79),
    ), `q${h}`)
    const warp = local(
        volumeNoiseAt(drift ? add(mul(q, 0.5), vec3(mul(drift, 1.4), 0, mul(drift, -0.9))) : mul(q, 0.5)),
        `warp${h}`,
    )
    const qw = opts.billow
        ? local(add(q, mul(vec3(warp, mul(warp, -0.73), mul(warp, 0.41)), opts.billow)), `qw${h}`)
        : q
    const mid = local(volumeNoiseAt(add(mul(qw, 2.13), vec3(5.2, 1.3, 8.4))), `mid${h}`)
    const fine = local(volumeNoiseAt(add(mul(qw, 4.31), vec3(9.1, 3.7, 1.2))), `fine${h}`)
    const density = local(mul(add(add(
        volumeNoiseAt(qw),
        mul(mid, 0.5)),
        mul(fine, 0.27)),
        0.57), `fbm${h}`)
    return {density, mid, fine}
}

/** One sample along a `volumeMarch`, handed to its callback. */
export interface VolumeMarchStep {
    /** The step number, 0 first. A plain number, handy for naming per-step values. */
    i: number
    /** How far along the ray this step sits, 0–1. A plain number. */
    t: number
    /** The depth of this sample along the ray, to pass to `ray.at`. */
    z: Expr
}

/**
 * Walk a ray through the medium front to back, adding up what glows and what dims.
 *
 * `steps` is how many samples to take (4–8 is typical; it is fixed when the shader compiles).
 * `entry` and `length` come from `interiorRay`. For each step your callback returns `emit`, the
 * rgb light this sample gives off, and `absorb`, the rgb amount it blocks. Absorption is per
 * channel, so a medium can dim blue more than red. Returns the accumulated `color` and the
 * `transmittance` left at the far side: multiply anything behind the medium by it. `jitter`
 * is a small per-pixel offset along the ray, as a fraction of its length (a hash times
 * `1 / steps` works), that trades banding for fine grain. `emissionGain` scales every
 * sample's glow; use `k / steps` so brightness does not change with the step count.
 *
 * @example
 * ```ts
 * const march = volumeMarch({steps: 6, entry: ray.entry, length: ray.length, jitter}, ({z, i}) => ({emit: math.mul(rgb, dens), absorb: math.splat3(math.mul(dens, u.density))}))
 * ```
 * @tip Fewer steps plus `jitter` beats more steps: the noise becomes grain instead of bands.
 * @see interiorRay, turbulentMedium, parallaxPlane
 */
export function volumeMarch(
    opts: {
        steps: number
        entry: Expr
        length: Expr
        /** A per-pixel offset added to each step's position along the ray, as a fraction of the ray. */
        jitter?: Expr
        /** A multiplier on every sample's `emit`. */
        emissionGain?: Expr
        hint?: string
    },
    sample: (step: VolumeMarchStep) => {emit: Expr; absorb: Expr},
): {color: Expr; transmittance: Expr} {
    // Front-to-back emission/absorption: each step contributes `emit · transmittance
    // (· emissionGain)` and attenuates the transmittance by `exp(−absorb)`. The unrolled loop
    // costs exactly what the sample callback costs; banding is bought off with `jitter`, not
    // more steps.
    let transmit: Expr = splat3(1)
    let color: Expr = vec3(0, 0, 0)
    for (let i = 0; i < opts.steps; i++) {
        const t = (i + 0.5) / opts.steps
        const z = local(
            add(opts.entry, mul(opts.length, opts.jitter ? add(t, opts.jitter) : t)),
            `z${i}`,
        )
        const {emit, absorb} = sample({i, t, z})
        const contribution = mul(emit, transmit)
        color = add(color, opts.emissionGain ? mul(contribution, splat3(opts.emissionGain)) : contribution)
        transmit = local(mul(transmit, exp(neg(absorb))), `transmit${i}`)
    }
    return {color: local(color, opts.hint ?? 'marchAcc'), transmittance: transmit}
}
