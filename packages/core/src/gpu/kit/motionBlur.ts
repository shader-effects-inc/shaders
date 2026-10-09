/**
 * GATHER-EFFECT PRIMITIVES — the tap machinery behind std/effects/blurs.
 *
 * The three motion blurs (linear, orbit, zoom) are ONE gather with an interchangeable tap
 * trajectory: RTT the child, sample it 32 times along some path, weight each tap by the same
 * Gaussian, left-fold the sum. Only the per-tap coordinate differs, so only the coordinate fns
 * are per-path; {@link motionBlurGather} owns the weights and the unroll.
 *
 * The scatter gather is the degenerate single-tap case (one hash-displaced sample), the
 * drop-shadow gather is the two-pass silhouette blur (H pass → intermediate RTT → V pass →
 * composite), the sharpen gather is a 5-tap unsharp-mask convolution, and the pixelate gather
 * is a quantised single tap with a cell-shape alpha mask. Taps run in PREMULTIPLIED space —
 * the RTT filter species appends the unpremultiply tail (pixelate unpremultiplies its one tap
 * itself and returns straight alpha).
 */
import {asLocal, call, floatE, vec4} from '../composer'
import {Expr, formatFloat, type GpuFragmentParams, type KitTexture} from '../contract'
import {tgpu, d, std} from './index'
import * as blend from './blend'
import * as blur from './blur'
import * as constants from './constants'
import * as edges from './edges'
import * as noise from './noise'
import * as sampling from './sampling'

/** Degrees → radians (module const; `Math.*` inside a `'use gpu'` body is unproven — kit convention). */
const DEG_TO_RAD = constants.DEG_TO_RAD

// ══════════════════════════════════════════════════════════════════════════════════════════════
// MOTION BLUR — one 32-tap Gaussian gather, three tap trajectories
// ══════════════════════════════════════════════════════════════════════════════════════════════

/** Tap count of the motion-blur gather (one `textureSample` each — see `blur.unrolledTapGather`). */
export const MOTION_BLUR_TAP_COUNT = 32

/** Normalized Gaussian tap weights, computed at JS time — no `exp()` per tap per pixel. */
export const MOTION_BLUR_WEIGHTS = blur.gaussianTapWeights(MOTION_BLUR_TAP_COUNT, 0.8)

/**
 * Sample coordinate for one linear-blur tap. `tapT ∈ [-0.5, 0.5]` is the centered tap position
 * ((i/31)-0.5). The blur vector is the (aspect-corrected) direction × intensity, normalised to the
 * viewport and doubled. `angleDeg` is degrees (transformAngle is identity-degrees), so
 * the body converts. All scalar infix (avoids vector `.div`).
 */
export const linearBlurTapCoord = tgpu.fn([d.vec2f, d.f32, d.f32, d.vec2f, d.f32], d.vec2f)(
    (uv, angleDeg, intensity, viewport, tapT) => {
        'use gpu'
        const angleRad = angleDeg * DEG_TO_RAD
        const aspect = viewport.x / viewport.y
        const dirX = std.cos(angleRad) / aspect
        const dirY = std.sin(angleRad)
        const bvX = (dirX * intensity) / viewport.x * 2.0
        const bvY = (dirY * intensity) / viewport.y * 2.0
        return d.vec2f(uv.x + bvX * tapT, uv.y + bvY * tapT)
    },
)

/**
 * Sample coordinate for one orbit-blur tap: rotate the (aspect-corrected) offset from `center` by
 * `(tapIndex − 15.5) × angleStep`, where `angleStep = intensity·0.005 / 31`. Centering the 32-tap
 * sweep on the source pixel (taps at ±0.5·step, ±1.5·step, …) keeps the blurred image aligned with
 * the sharp one — a one-sided sweep visibly counter-rotated content by half the blur angle
 * (pre-2026-08 behavior; deliberately corrected). `center` is the ALREADY-transformed
 * prop value (transformPosition stores `(x, 1 - y)`), so `1.0 - center.y` recovers the authored y.
 *
 * Each tap's rotation is computed DIRECTLY (`cos/sin((tapIndex − 15.5)·angleStep)`) rather than via
 * a stateful recurrence — no accumulated f32 drift and no loop-carried state.
 */
export const angularBlurTapCoord = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32, d.f32], d.vec2f)(
    (center, intensity, uv, aspect, tapIndex) => {
        'use gpu'
        const centerPos = d.vec2f(center.x, 1.0 - center.y)
        const angleStep = (intensity * 0.005) / 31.0
        const angle = (tapIndex - 15.5) * angleStep
        const cosA = std.cos(angle)
        const sinA = std.sin(angle)
        const initX = uv.x - centerPos.x
        const initY = uv.y - centerPos.y
        const acX = initX * aspect
        const acY = initY
        const rotX = acX * cosA - acY * sinA
        const rotY = acX * sinA + acY * cosA
        return d.vec2f(rotX / aspect + centerPos.x, rotY + centerPos.y)
    },
)

/**
 * Sample coordinate for one zoom-blur tap: scale the offset from `center` by
 * `scale = 1 + radius·(tapIndex/31)` (radius = intensity·0.01), pulling near-center samples out and
 * far samples in for the radial streak. `center` is the ALREADY-transformed prop value
 * (transformPosition stores `(x, 1 - y)`), so `1.0 - center.y` recovers the authored y.
 *
 * The x offset is aspect-corrected (×aspect) then un-corrected (÷aspect) at the end; the two cancel
 * algebraically but are kept in that exact op order so the f32 rounding matches. Pure.
 */
export const zoomBlurTapCoord = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32, d.f32], d.vec2f)(
    (center, intensity, uv, aspect, tapIndex) => {
        'use gpu'
        const centerPos = d.vec2f(center.x, 1.0 - center.y)
        const radius = intensity * 0.01
        const scale = 1.0 + radius * (tapIndex / 31.0)
        const deltaX = uv.x - centerPos.x
        const deltaY = uv.y - centerPos.y
        const acdX = deltaX * aspect
        const acdY = deltaY
        const scaledDeltaX = acdX / scale
        const scaledDeltaY = acdY / scale
        return d.vec2f(scaledDeltaX / aspect + centerPos.x, scaledDeltaY + centerPos.y)
    },
)

/** The three tap trajectories a motion blur can follow. */
export type MotionBlurPathKind = 'linear' | 'orbit' | 'zoom'

export interface MotionBlurGatherArgs {
    /** The path anchor: `linear` → the angle in degrees; `orbit`/`zoom` → the transformed center. */
    focus: Expr
    /** Blur intensity (the shaders' 0–100 range). */
    amount: Expr
    uv: Expr
    aspect: Expr
    viewportSize: Expr
}

/**
 * The 32-tap Gaussian motion-blur gather: `Σ sample(tapCoord(i)) · weight(i)`, left-folded in tap
 * order (float addition is not associative — the fold order is part of the result). One
 * `texture.sample()` per tap with no control flow, the only legal shape for textureSample in a
 * fragment; the linearClamp sampler clamps out-of-range taps.
 */
export function motionBlurGather(
    kind: MotionBlurPathKind,
    args: MotionBlurGatherArgs,
    sample: (coord: Expr) => Expr,
): Expr {
    const {focus, amount, uv, aspect, viewportSize} = args
    const tapCoord =
        kind === 'linear'
            ? (i: number) => call(linearBlurTapCoord, 'linearBlurTapCoord', [uv, focus, amount, viewportSize, floatE(i / 31 - 0.5)])
            : kind === 'orbit'
              ? (i: number) => call(angularBlurTapCoord, 'angularBlurTapCoord', [focus, amount, uv, aspect, floatE(i)])
              : (i: number) => call(zoomBlurTapCoord, 'zoomBlurTapCoord', [focus, amount, uv, aspect, floatE(i)])
    return blur.unrolledTapGather({weights: MOTION_BLUR_WEIGHTS, tapCoord, sample})
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// MOTION BLUR DETAIL — the lens/streak dials layered on the same 32-tap gather
//
// Every dial defaults to 0 and the plain gather above is emitted whenever all of them are 0
// (the std word gates on the compose-time prop values), so the default shader is byte-identical
// to the detail-less one. With any dial on, the gather switches to this path: the same tap
// trajectory fns, but each tap's position is reshaped (bias / jitter / focus) and each tap's
// weight becomes a per-channel vec4 (spectral tint × highlight boost), normalised at the end.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Hue (0–1 around the wheel) → the smoothed RGB rainbow the spectral lens uses. CPU mirror of
 * lensParts' `hueColor` so the per-tap tints can be baked as WGSL literals.
 */
export function spectrumRgb(hue: number): [number, number, number] {
    const channel = (offset: number): number => {
        const raw = hue * 6 + offset
        const wrapped = raw - 6 * Math.floor(raw / 6)
        const clamped = Math.min(1, Math.max(0, Math.abs(wrapped - 3) - 1))
        return clamped * clamped * (3 - 2 * clamped)
    }
    return [channel(0), channel(4), channel(2)]
}

/** The slice of the hue wheel the dispersion streak walks: red (0) through violet (0.75). */
export const MOTION_BLUR_SPECTRUM_SPAN = 0.75

/**
 * Per-tap spectral tint minus one (`tint − 1`, so `1 + k·table[i]` is the tint at strength
 * `k`), tap 0 at the red end and tap 31 at the violet end. `MOTION_BLUR_SPECTRUM_REVERSED` is
 * the same table walked the other way, for a negative dispersion.
 */
export const MOTION_BLUR_SPECTRUM = Array.from({length: MOTION_BLUR_TAP_COUNT}, (_, i) => {
    const [r, g, b] = spectrumRgb((i / (MOTION_BLUR_TAP_COUNT - 1)) * MOTION_BLUR_SPECTRUM_SPAN)
    return [r - 1, g - 1, b - 1] as [number, number, number]
})
export const MOTION_BLUR_SPECTRUM_REVERSED = [...MOTION_BLUR_SPECTRUM].reverse()

/** How hard `highlights` 1 boosts a tap of luminance 1 (`weight × (1 + gain·lum²)`). */
export const MOTION_BLUR_HIGHLIGHT_GAIN = 12
/** Twist of the zoom spiral, in radians per unit tap-scale at intensity 100 and `spiral` 1. */
export const MOTION_BLUR_SPIRAL_TWIST = 1.2
/** Soft edge of the focus zone as a fraction of its radius (`focus·0.7` → `focus·1.3`). */
const FOCUS_SOFTNESS = 0.3

/**
 * Sample coordinate for one zoom-blur tap with a spiral twist: the zoom scale of
 * {@link zoomBlurTapCoord} plus a rotation of the (aspect-corrected) offset by
 * `spiral · radius · (tapIndex/31) · twist`, so the streak corkscrews out of the centre. At
 * `spiral` 0 the rotation is zero and the coordinate matches the plain zoom tap. Pure.
 */
export const zoomBlurSpiralTapCoord = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.vec2f)(
    (center, intensity, uv, aspect, tapIndex, spiral) => {
        'use gpu'
        const centerPos = d.vec2f(center.x, 1.0 - center.y)
        const radius = intensity * 0.01
        const along = tapIndex / 31.0
        const scale = 1.0 + radius * along
        const angle = spiral * radius * along * MOTION_BLUR_SPIRAL_TWIST
        const cosA = std.cos(angle)
        const sinA = std.sin(angle)
        const acdX = (uv.x - centerPos.x) * aspect
        const acdY = uv.y - centerPos.y
        const rotX = acdX * cosA - acdY * sinA
        const rotY = acdX * sinA + acdY * cosA
        return d.vec2f(rotX / scale / aspect + centerPos.x, rotY / scale + centerPos.y)
    },
)

/**
 * The per-pixel detail field, computed once per fragment:
 *   x = positive dispersion strength, y = negative dispersion strength (each already ramped by
 *       `falloff` over the distance from `center`: uniform at falloff 0, `d²` at falloff 1),
 *   z = focus mask (0 inside the clean zone of radius `focus`, 1 outside; exactly 1 at focus 0),
 *   w = jitter phase in tap-index units (interleaved gradient noise on device pixels, ±0.5·jitter).
 * `center` is the TRANSFORMED position (y stored as `1 − y`), like the tap coordinate fns. Distances
 * are normalised by the farthest canvas corner so the dials read the same on any frame size.
 */
export const motionBlurDetailField = tgpu.fn(
    [d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.vec2f],
    d.vec4f,
)((uv, center, aspect, dispersion, falloff, focus, jitter, viewport) => {
    'use gpu'
    const centerPos = d.vec2f(center.x, 1.0 - center.y)
    const dx = (uv.x - centerPos.x) * aspect
    const dy = uv.y - centerPos.y
    const dist = std.sqrt(dx * dx + dy * dy)
    const farX = std.max(centerPos.x, 1.0 - centerPos.x) * aspect
    const farY = std.max(centerPos.y, 1.0 - centerPos.y)
    const dMax = std.max(std.sqrt(farX * farX + farY * farY), 0.0001)
    const dNorm = std.clamp(dist / dMax, 0.0, 1.0)
    const ramp = std.mix(d.f32(1), dNorm * dNorm, falloff)
    const strength = dispersion * ramp
    const dPos = std.max(strength, 0.0)
    const dNeg = std.max(strength * -1.0, 0.0)
    const focusEdge = std.smoothstep(focus * (1.0 - FOCUS_SOFTNESS), focus * (1.0 + FOCUS_SOFTNESS) + 0.0001, dNorm)
    const focusMask = std.mix(d.f32(1), focusEdge, std.step(0.0001, focus))
    const px = d.vec2f(uv.x * viewport.x, uv.y * viewport.y)
    const ign = std.fract(52.9829189 * std.fract(px.x * 0.06711056 + px.y * 0.00583715))
    const jitterPhase = (ign - 0.5) * jitter
    return d.vec4f(dPos, dNeg, focusMask, jitterPhase)
})

/**
 * One tap's per-channel weight: the Gaussian `weight` × a highlight boost (`1 + gain·lum²`, so
 * bright taps streak further) × the spectral tint `1 + dPos·tintF + dNeg·tintR`. Returns
 * `vec4(rgb weights, alpha weight)`; the alpha weight carries the boost but no tint.
 */
export const motionBlurDetailTapWeight = tgpu.fn([d.vec4f, d.f32, d.vec3f, d.vec3f, d.vec4f, d.f32], d.vec4f)(
    (sample, weight, tintForward, tintReversed, field, highlightGain) => {
        'use gpu'
        const lum = std.dot(sample.xyz, d.vec3f(0.2126, 0.7152, 0.0722))
        const boost = weight * (1.0 + highlightGain * lum * lum)
        const tint = d.vec3f(1.0, 1.0, 1.0).add(tintForward.mul(field.x)).add(tintReversed.mul(field.y))
        return d.vec4f(tint.x * boost, tint.y * boost, tint.z * boost, boost)
    },
)

/** `colorSum / weightSum` per channel, guarded against an empty channel. */
export const motionBlurDetailResolve = tgpu.fn([d.vec4f, d.vec4f], d.vec4f)((colorSum, weightSum) => {
    'use gpu'
    return colorSum.div(std.max(weightSum, d.vec4f(0.0001, 0.0001, 0.0001, 0.0001)))
})

export interface MotionBlurDetailArgs {
    /** −1 to 1: spectral tint strength along the streak; the sign picks which end is red. */
    dispersion: Expr
    /** 0–1: how much the dispersion concentrates away from the centre. */
    falloff: Expr
    /** 0–1: radius of the clean zone around the centre (fraction of the farthest corner). */
    focus: Expr
    /** −1 to 1: trail the streak to one side of the pixel (zoom: shift the window). */
    bias: Expr
    /** 0–1: bright taps weigh more, so highlights streak further. */
    highlights: Expr
    /** 0–1: per-pixel tap phase jitter that dithers the 32-tap comb. */
    jitter: Expr
    /** Zoom only, −1 to 1: corkscrew the radial streak. */
    spiral?: Expr
    /**
     * The point the falloff and focus measure from: the path's centre for orbit/zoom; the linear
     * path has none, so pass the canvas centre.
     */
    center: Expr
}

/**
 * Literal `vec3f(…)` for a baked tint table entry (plain WGSL, no external).
 */
function vec3Literal([x, y, z]: [number, number, number]): Expr {
    return new Expr(() => `vec3f(${formatFloat(x)}, ${formatFloat(y)}, ${formatFloat(z)})`)
}

/**
 * The detail gather: the same 32 taps along the path, with each tap's position reshaped and
 * each tap's weight a per-channel vec4, normalised per channel at the end.
 *
 * Tap positions: linear/orbit taps sit at `c + |c|·bias + jitter` (c centred on the pixel, so
 * bias trails the streak off one side while the pixel stays the weight peak); zoom taps at
 * `i − 15.5·bias + jitter` (the inward window slides toward a symmetric one). The amount is
 * scaled by the focus mask so the clean zone blurs nothing. Only `focus`'s mask touches the
 * amount; everything else leaves the trajectory fns untouched.
 */
export function motionBlurDetailGather(
    kind: MotionBlurPathKind,
    args: MotionBlurGatherArgs,
    detail: MotionBlurDetailArgs,
    sample: (coord: Expr) => Expr,
): Expr {
    const {focus: anchor, uv, aspect, viewportSize} = args
    const field = asLocal(
        call(motionBlurDetailField, 'motionBlurDetailField', [
            uv, detail.center, aspect, detail.dispersion, detail.falloff, detail.focus, detail.jitter, viewportSize,
        ]),
        'mbField',
    )
    const amount = asLocal(args.amount.mul(field.member('z')), 'mbAmount')
    const bias = asLocal(detail.bias, 'mbBias')
    const jitterPhase = field.member('w')
    const highlightGain = asLocal(detail.highlights.mul(MOTION_BLUR_HIGHLIGHT_GAIN), 'mbHighlight')
    const last = MOTION_BLUR_TAP_COUNT - 1
    const mid = last / 2

    const tapCoord = (i: number): Expr => {
        if (kind === 'linear') {
            const c = i / last - 0.5
            const tapT = floatE(c).add(bias.mul(Math.abs(c))).add(jitterPhase.mul(1 / last))
            return call(linearBlurTapCoord, 'linearBlurTapCoord', [uv, anchor, amount, viewportSize, tapT])
        }
        if (kind === 'orbit') {
            const index = floatE(i).add(bias.mul(Math.abs(i - mid))).add(jitterPhase)
            return call(angularBlurTapCoord, 'angularBlurTapCoord', [anchor, amount, uv, aspect, index])
        }
        const index = floatE(i).add(bias.mul(-mid)).add(jitterPhase)
        if (detail.spiral) {
            return call(zoomBlurSpiralTapCoord, 'zoomBlurSpiralTapCoord', [anchor, amount, uv, aspect, index, detail.spiral])
        }
        return call(zoomBlurTapCoord, 'zoomBlurTapCoord', [anchor, amount, uv, aspect, index])
    }

    let colorSum: Expr | undefined
    let weightSum: Expr | undefined
    for (let i = 0; i < MOTION_BLUR_TAP_COUNT; i++) {
        const tap = asLocal(sample(tapCoord(i)), 'mbTap')
        const weight = asLocal(
            call(motionBlurDetailTapWeight, 'motionBlurDetailTapWeight', [
                tap, floatE(MOTION_BLUR_WEIGHTS[i]), vec3Literal(MOTION_BLUR_SPECTRUM[i]),
                vec3Literal(MOTION_BLUR_SPECTRUM_REVERSED[i]), field, highlightGain,
            ]),
            'mbWeight',
        )
        const weighted = tap.mul(weight)
        colorSum = colorSum ? colorSum.add(weighted) : weighted
        weightSum = weightSum ? weightSum.add(weight) : weight
    }
    return call(motionBlurDetailResolve, 'motionBlurDetailResolve', [colorSum as Expr, weightSum as Expr])
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// SCATTER — one hash-displaced sample (grain-like scatter, not a Gaussian kernel)
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * One decorrelated hash per pixel → a random offset in the [-1,1]² unit square, scaled to pixels by
 * `intensity` and normalised by the viewport (so the grain size is DPR-stable). Returns the
 * displaced sample coordinate. `viewport` is `_sys.viewportSize` (device pixels). The seed is
 * ~thousands, where the old sin-fract hash streaks on iOS Metal (low-precision large-argument sin)
 * — hence the integer bitcast hash. Vector ops fluent, scalar infix.
 */
export const diffuseBlurUV = tgpu.fn([d.vec2f, d.f32, d.vec2f], d.vec2f)((uv, intensity, viewport) => {
    'use gpu'
    const seed = uv.mul(1000.0)
    const randVec = noise.hash22(seed)
    // Map [0,1] → [-1,1] and scale by intensity in pixels, normalised by the viewport.
    const offset = randVec.mul(2.0).sub(d.vec2f(1.0, 1.0)).mul(intensity).div(viewport)
    return uv.add(offset)
})

/**
 * `diffuseBlurUV` for one colour channel: the channel's own hash (seeded by `salt`) mixed into the
 * shared one by `chromatic`, so at 0 every channel lands on the same displaced coordinate and at
 * 1 each scatters independently (a sparkly colour grain).
 */
export const diffuseBlurChannelUV = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32, d.vec2f], d.vec2f)(
    (uv, intensity, viewport, chromatic, salt) => {
        'use gpu'
        const seed = uv.mul(1000.0)
        const shared = noise.hash22(seed)
        const own = noise.hash22(seed.add(salt))
        const randVec = shared.add(own.sub(shared).mul(chromatic))
        const offset = randVec.mul(2.0).sub(d.vec2f(1.0, 1.0)).mul(intensity).div(viewport)
        return uv.add(offset)
    },
)

/** Hash salts that decorrelate the red and blue scatter from the shared (green) one. */
const SCATTER_SALT_RED: [number, number] = [17.3, 91.7]
const SCATTER_SALT_BLUE: [number, number] = [53.1, 29.9]

export interface ScatterGatherArgs {
    uv: Expr
    /** Displacement radius in pixels. */
    amount: Expr
    viewportSize: Expr
    /** Compile-time edge mode (transformEdges value); `stretch` relies on the clamping sampler. */
    edgeMode: number
    sample: (coord: Expr) => Expr
    /** 0–1: scatter red and blue on their own hashes (three taps instead of one). Omit for the single-tap gather. */
    chromatic?: Expr
}

/**
 * The scatter gather: sample the child once at the hash-displaced coordinate, with edge handling.
 * With `chromatic` set, red and blue take their own displaced taps (`diffuseBlurChannelUV`) and the
 * shared tap supplies green and alpha.
 */
export function scatterGather(args: ScatterGatherArgs): Expr {
    const displaced = call(diffuseBlurUV, 'diffuseBlurUV', [args.uv, args.amount, args.viewportSize])
    const chromatic = args.chromatic
    if (!chromatic) return edges.applyEdgeHandlingExpr(displaced, args.sample, args.edgeMode)
    // Hoisted: the shared tap feeds two channels, and an Expr re-emits its subtree per use.
    const shared = asLocal(edges.applyEdgeHandlingExpr(displaced, args.sample, args.edgeMode), 'scatterShared')
    const channelTap = (salt: [number, number]): Expr => {
        const saltLiteral = new Expr(() => `vec2f(${formatFloat(salt[0])}, ${formatFloat(salt[1])})`)
        const coord = call(diffuseBlurChannelUV, 'diffuseBlurChannelUV', [args.uv, args.amount, args.viewportSize, chromatic, saltLiteral])
        return edges.applyEdgeHandlingExpr(coord, args.sample, args.edgeMode)
    }
    const red = channelTap(SCATTER_SALT_RED)
    const blue = channelTap(SCATTER_SALT_BLUE)
    return vec4(red.member('r'), shared.member('g'), blue.member('b'), shared.member('a'))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// DROP SHADOW — two-pass separable blur of the alpha silhouette + composite
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Gaussian weights for a 13-tap kernel (sigma≈2.5), and the sum they are normalized by. Hand-tabulated
 * rather than `blur.gaussianTapWeights`: the divisor is 6.214 where the tabulated values actually sum
 * to 6.210, so regenerating them would change the shadow's density. Left as-is deliberately.
 */
const WEIGHTS = [0.056, 0.135, 0.278, 0.487, 0.726, 0.923, 1.0, 0.923, 0.726, 0.487, 0.278, 0.135, 0.056]
const WEIGHT_SUM = 6.214
export const DROP_SHADOW_TAP_WEIGHTS = WEIGHTS.map((w) => w / WEIGHT_SUM)
const TAP_HALF = 6

/** The shadow's screen UV: offset the sample by the compass-angle direction × distance (X divided by
 *  aspect so the offset covers equal pixel distance in both axes). Pure. */
export const dropShadowUV = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32], d.vec2f)((uv, viewport, angle, distance) => {
    'use gpu'
    const angleRad = angle * DEG_TO_RAD
    const dist = distance * 0.2
    const aspect = viewport.x / viewport.y
    const ox = std.sin(angleRad) * dist / aspect
    const oy = std.cos(angleRad) * -1.0 * dist
    return d.vec2f(uv.x - ox, uv.y - oy)
})

/** `uv + (ox, oy) · blurRadius / viewport` — one separable-blur tap offset in UV space. Pure. */
export const dropShadowTapUV = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32, d.f32], d.vec2f)((uv, viewport, blurRadius, ox, oy) => {
    'use gpu'
    return d.vec2f(uv.x + ox * blurRadius / viewport.x, uv.y + oy * blurRadius / viewport.y)
})

/**
 * The final drop-shadow composite in PREMULTIPLIED space (the child RTT is premultiplied) — the
 * gather's caller unpremultiplies the return. `cutout` is a runtime 0/1 flag selected via
 * `std.select`. Normal: original content OVER the shadow. Cutout: shadow only, with the original
 * silhouette punched out. `shadowColor` is the vec4 rgba tint. Pure.
 */
export const dropShadowComposite = tgpu.fn([d.vec4f, d.vec4f, d.f32, d.f32], d.vec4f)((originalColor, shadowColor, shadowAlpha, cutout) => {
    'use gpu'
    const shadowRgb = d.vec3f(shadowColor.x, shadowColor.y, shadowColor.z)
    const shadowA = shadowColor.w * shadowAlpha

    const cutoutA = std.clamp(shadowA - originalColor.w, 0.0, 1.0)
    const cutoutRgb = shadowRgb.mul(shadowColor.w).mul(cutoutA)

    const shadowPremulRgb = shadowRgb.mul(shadowColor.w).mul(shadowAlpha)
    const oneMinusOrigA = 1.0 - originalColor.w
    const normalRgb = d.vec3f(originalColor.x, originalColor.y, originalColor.z).add(shadowPremulRgb.mul(oneMinusOrigA))
    const normalA = std.max(originalColor.w + shadowA * oneMinusOrigA, originalColor.w)

    const useCut = cutout > 0.5
    const rgb = std.select(normalRgb, cutoutRgb, useCut)
    const a = std.select(normalA, cutoutA, useCut)
    return d.vec4f(rgb.x, rgb.y, rgb.z, a)
})

export interface SilhouetteBlurArgs {
    /** Where the silhouette is read from (e.g. the shadow-offset UV). */
    at: Expr
    /** The screen UV the vertical pass reads back at. */
    uv: Expr
    viewportSize: Expr
    /** Blur radius in pixels. */
    blurRadius: Expr
    /** Sample the child RTT (premultiplied). */
    sample: (coord: Expr) => Expr
    convertToTexture: GpuFragmentParams['convertToTexture']
}

/**
 * Two-pass separable Gaussian blur of the child's ALPHA silhouette (the frozen 13-tap table),
 * running INLINE in the fragment: horizontal pass at `at` → an intermediate RTT
 * (`convertToTexture` of the pass's Expr) → vertical pass at `uv`. Returns the blurred coverage
 * scalar.
 */
export function silhouetteBlur(args: SilhouetteBlurArgs): Expr {
    const {at, uv, viewportSize, blurRadius, sample, convertToTexture} = args
    const tapUV = (base: Expr, ox: number, oy: number): Expr =>
        call(dropShadowTapUV, 'dropShadowTapUV', [base, viewportSize, blurRadius, floatE(ox), floatE(oy)])

    // Pass 1: horizontal Gaussian blur of the child alpha at the offset → an RTT texture.
    const horizontal = blur.unrolledTapGather({
        weights: DROP_SHADOW_TAP_WEIGHTS,
        tapCoord: (i) => tapUV(at, i - TAP_HALF, 0),
        sample,
        component: 'a',
    })
    const horizontalTexture = convertToTexture(vec4(horizontal, floatE(0), floatE(0), floatE(1)))

    // Pass 2: vertical Gaussian blur of the horizontal pass's .r channel.
    return blur.unrolledTapGather({
        weights: DROP_SHADOW_TAP_WEIGHTS,
        tapCoord: (i) => tapUV(uv, 0, i - TAP_HALF),
        sample: (coord) => horizontalTexture.sample(coord),
        component: 'r',
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// SHARPEN — 5-tap unsharp-mask convolution
// ══════════════════════════════════════════════════════════════════════════════════════════════

/** One neighbour sample UV: uv + (dirX, dirY) × one device pixel (pixelSize = 1 / viewportSize).
 *  Pure. */
export const sharpnessTapUV = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32], d.vec2f)((uv, viewport, dirX, dirY) => {
    'use gpu'
    return d.vec2f(uv.x + dirX / viewport.x, uv.y + dirY / viewport.y)
})

/** Sharpening kernel: center × (1 + 4·amount) − neighbours × amount, clamped to [0,1]. Operates on
 *  the premultiplied samples; alpha is taken from the centre tap (neighbour alpha is unused — RGB
 *  only). The caller unpremultiplies the result. */
export const sharpnessCompose = tgpu.fn([d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec4f)(
    (center, top, bottom, left, right, amount) => {
        'use gpu'
        const centerWeight = 1.0 + amount * 4.0
        const neighborWeight = amount * -1.0
        const rgb = center.xyz.mul(centerWeight)
            .add(top.xyz.mul(neighborWeight))
            .add(bottom.xyz.mul(neighborWeight))
            .add(left.xyz.mul(neighborWeight))
            .add(right.xyz.mul(neighborWeight))
        const clamped = std.clamp(rgb, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0))
        return d.vec4f(clamped.x, clamped.y, clamped.z, center.w)
    })

export interface SharpenGatherArgs {
    uv: Expr
    viewportSize: Expr
    /** Kernel strength; 0 collapses to identity (centerWeight 1, neighbourWeight 0). */
    amount: Expr
    /** Sample the child RTT (premultiplied). */
    sample: (coord: Expr) => Expr
}

/** The sharpen gather: sample the centre + 4 orthogonal one-pixel neighbours and run the
 *  unsharp-mask kernel over them, in premultiplied space. */
export function sharpenGather(args: SharpenGatherArgs): Expr {
    const {uv, viewportSize, amount, sample} = args
    const center = sample(uv)
    const top = sample(call(sharpnessTapUV, 'sharpnessTapUV', [uv, viewportSize, floatE(0), floatE(1)]))
    const bottom = sample(call(sharpnessTapUV, 'sharpnessTapUV', [uv, viewportSize, floatE(0), floatE(-1)]))
    const left = sample(call(sharpnessTapUV, 'sharpnessTapUV', [uv, viewportSize, floatE(-1), floatE(0)]))
    const right = sample(call(sharpnessTapUV, 'sharpnessTapUV', [uv, viewportSize, floatE(1), floatE(0)]))
    return call(sharpnessCompose, 'sharpnessCompose', [center, top, bottom, left, right, amount])
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// PIXELATE — quantised single tap + cell-shape alpha mask
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Quantise the UV to a pixel grid + build the gap/roundness cell mask.
 *
 * Returns vec3(sampleUV.x, sampleUV.y, mask): the cell-center-ish sample coordinate (this samples
 * the floored cell origin, not the center) packed with the rounded-rect cell coverage mask.
 * `scale` counts pixels along the LONGEST side; the shorter side scales by aspect (clamped ≥ 1).
 * Floor-quantise for the sample UV, fract-local coords for the SDF. Pure.
 */
export const pixelateSample = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32], d.vec3f)(
    (uv, aspect, scale, gap, roundness) => {
        'use gpu'
        const isWide = aspect > 1.0
        const pixelCountX = std.max(std.select(scale * aspect, scale, isWide), 1.0)
        const pixelCountY = std.max(std.select(scale, scale / aspect, isWide), 1.0)

        // Quantise X/Y separately with the aspect-adjusted counts.
        const sampleUV = d.vec2f(
            std.floor(uv.x * pixelCountX) / pixelCountX,
            std.floor(uv.y * pixelCountY) / pixelCountY,
        )

        // Local position within each cell, centred at 0 (range -0.5 … 0.5).
        const localUV = d.vec2f(std.fract(uv.x * pixelCountX) - 0.5, std.fract(uv.y * pixelCountY) - 0.5)

        // Rounded-rectangle SDF: length(max(abs(p) - halfSize + r, 0)) - r.
        const halfSize = 0.5 - gap * 0.5
        const cornerRadius = roundness * halfSize
        const qx = std.max(std.abs(localUV.x) - halfSize + cornerRadius, 0.0)
        const qy = std.max(std.abs(localUV.y) - halfSize + cornerRadius, 0.0)
        const sdf = std.length(d.vec2f(qx, qy)) - cornerRadius
        const mask = std.select(d.f32(0.0), d.f32(1.0), sdf <= 0.0)

        return d.vec3f(sampleUV.x, sampleUV.y, mask)
    })

export interface PixelateGatherArgs {
    uv: Expr
    aspect: Expr
    /** Pixel count along the longest edge (higher = smaller pixels). */
    scale: Expr
    /** Space between cells as a fraction of cell size. */
    gap: Expr
    /** Corner roundness of each cell (0 = square, 1 = circle). */
    roundness: Expr
    /** Sample the child RTT and unpremultiply — the scaffold's straight-alpha sampler. */
    sampleStraight: (coord: Expr) => Expr
}

/** The pixelate gather: one straight-alpha sample at the quantised cell coordinate, with the
 *  rounded-rect cell mask applied to the alpha. Returns STRAIGHT alpha. */
export function pixelateGather(args: PixelateGatherArgs): Expr {
    const packed = call(pixelateSample, 'pixelateSample', [args.uv, args.aspect, args.scale, args.gap, args.roundness])
    const sampled = args.sampleStraight(packed.member('xy'))
    return vec4(sampled.member('rgb'), sampled.member('a').mul(packed.member('z')))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// GLASS TILES — tiled refraction UV bend (fragment + analytic remap pair)
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * The distorted sample coordinate for a screen UV. `tileCount` applies to the LONGEST side; the
 * shorter side scales by aspect. The grid is rotated in aspect-corrected square space; the
 * refraction offset pushes each fragment away from its tile centre (roundness fades that off
 * radially). Vector ops fluent, scalar ops infix. Pure.
 */
export const glassTilesUV = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (uv, aspect, intensity, baseTileCount, rotationDegrees, roundnessAmount) => {
        'use gpu'
        const isWide = aspect > 1.0
        const tileCount = d.vec2f(
            std.select(baseTileCount * aspect, baseTileCount, isWide),
            std.select(baseTileCount, baseTileCount / aspect, isWide),
        )

        // Rotate the grid in aspect-corrected square space around the centre.
        const aspectCorrectedUV = d.vec2f(uv.x * aspect, uv.y)
        const rotationRadians = rotationDegrees * DEG_TO_RAD
        const cosA = std.cos(rotationRadians)
        const sinA = std.sin(rotationRadians)
        const centered = aspectCorrectedUV.sub(d.vec2f(0.5 * aspect, 0.5))
        const rotatedX = centered.x * cosA - centered.y * sinA
        const rotatedY = centered.x * sinA + centered.y * cosA
        const rotatedUV = d.vec2f(rotatedX, rotatedY).add(d.vec2f(0.5 * aspect, 0.5))
        const gridUV = d.vec2f(rotatedUV.x / aspect, rotatedUV.y)

        // Tile grid: local UV within the current cell, normalised to [0,1] across the cell.
        const cellCoord = std.floor(gridUV.mul(tileCount)).div(tileCount)
        const localUV = gridUV.sub(cellCoord)
        const oneOverTile = d.vec2f(1.0, 1.0).div(tileCount)
        const normalizedLocalUV = localUV.div(oneOverTile)

        const fromCenter = normalizedLocalUV.sub(d.vec2f(0.5, 0.5))
        const distFromCenter = fromCenter.x * fromCenter.x + fromCenter.y * fromCenter.y
        const roundnessFactor = 1.0 - distFromCenter * (roundnessAmount * 4.0)
        const clampedRoundness = std.max(roundnessFactor, 0.0)

        const distortionFactor = intensity * 0.025
        const scaleFactor = distortionFactor * clampedRoundness
        // Aspect-correct the X offset so refraction is visually symmetric; apply to ORIGINAL uv.
        const distortion = d.vec2f((fromCenter.x * scaleFactor) / aspect, fromCenter.y * scaleFactor)
        return uv.add(distortion)
    })

export interface GlassTilesUVArgs {
    uv: Expr
    aspect: Expr
    intensity: Expr
    tileCount: Expr
    rotation: Expr
    roundness: Expr
}

/** The tile-distorted sample coordinate (shared by the fragment and the analytic remap). */
export function glassTilesDistortUV(args: GlassTilesUVArgs): Expr {
    return call(glassTilesUV, 'glassTilesUV', [args.uv, args.aspect, args.intensity, args.tileCount, args.rotation, args.roundness])
}

/**
 * The glass-tiles fragment: sample the child RTT at the tile-distorted coordinate with the default
 * clamp-to-edge sampler (no edge modes), then unpremultiply. Catmull-Rom reconstruction: each tile
 * magnifies its slice of the content, so a single bilinear tap would facet any hard edge underneath.
 * Returns STRAIGHT alpha.
 */
export function glassTilesFragment(args: GlassTilesUVArgs & {texture: KitTexture}): Expr {
    const distortedUV = glassTilesDistortUV(args)
    return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sampling.sampleCatmullRomExpr(args.texture, distortedUV)])
}

/** The analytic remap coordinate: the same tiled-refraction bend, clamped to [0,1] to match the
 *  fragment path's clamp sampler (edgeClampUV clamps to `vec2(0)`..`vec2(1)`). */
export function glassTilesRemapUV(args: GlassTilesUVArgs): Expr {
    return call(edges.edgeClampUV, 'edgeClampUV', [glassTilesDistortUV(args)])
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// CRT SCREEN — chromatic aberration + scanlines + phosphor mask + vignette
// ══════════════════════════════════════════════════════════════════════════════════════════════

/** One chromatic-aberration sample UV: shift X by ±(colorShift·0.002). `dir` is +1 (red) / -1
 *  (blue); green samples the un-shifted uv directly. */
export const crtSampleUV = tgpu.fn([d.vec2f, d.f32, d.f32], d.vec2f)((uv, colorShift, dir) => {
    'use gpu'
    const scaledColorShift = colorShift * 0.002
    return d.vec2f(uv.x + scaledColorShift * dir, uv.y)
})

/** Combine the three split-tap samples into one RGB: red from the +tap, green from the centre,
 *  blue from the −tap. */
export const rgbSplitCombine = tgpu.fn([d.vec4f, d.vec4f, d.vec4f], d.vec3f)((redSample, greenSample, blueSample) => {
    'use gpu'
    return d.vec3f(redSample.x, greenSample.y, blueSample.z)
})

/** Brightness + contrast around 0.5. */
export const adjustShade = tgpu.fn([d.vec3f, d.f32, d.f32], d.vec3f)((color, contrast, brightness) => {
    'use gpu'
    return color.sub(d.vec3f(0.5)).mul(contrast).add(d.vec3f(0.5)).mul(brightness)
})

/** Sinusoidal scanlines: darken by a sin wave over `uvY × frequency` (period constant 3.14159·2). */
export const scanlineShade = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32], d.vec3f)((color, uvY, scanlineFrequency, scanlineIntensity) => {
    'use gpu'
    const scanlinePosition = uvY * scanlineFrequency
    const scanlineWave = std.sin(scanlinePosition * (3.14159 * 2.0))
    const scanlineEffect = 1.0 - scanlineIntensity * (scanlineWave * 0.5 + 0.5)
    return color.mul(scanlineEffect)
})

/** Subtle RGB phosphor pattern (per-channel sin masks with phase offsets 2.09/4.18). */
export const phosphorShade = tgpu.fn([d.vec3f, d.vec2f, d.f32], d.vec3f)((color, uv, pixelSize) => {
    'use gpu'
    const phosphorScale = pixelSize * 0.5
    const pixelPhase = std.fract(uv.mul(phosphorScale))
    const redMask = std.sin(pixelPhase.x * 6.28318) * 0.1 + 0.95
    const greenMask = std.sin(pixelPhase.x * 6.28318 + 2.09) * 0.1 + 0.95
    const blueMask = std.sin(pixelPhase.x * 6.28318 + 4.18) * 0.1 + 0.95
    return d.vec3f(color.x * redMask, color.y * greenMask, color.z * blueMask)
})

/** Aspect-corrected circular vignette (X in height-units). */
export const vignetteShade = tgpu.fn([d.vec3f, d.vec2f, d.f32, d.f32, d.f32], d.vec3f)((color, uv, aspect, vignetteRadius, vignetteIntensity) => {
    'use gpu'
    const centeredUV = uv.sub(d.vec2f(0.5, 0.5))
    const aspectCorrectedUV = d.vec2f(centeredUV.x * aspect, centeredUV.y)
    const vignetteDistance = std.length(aspectCorrectedUV)
    const innerEdge = 1.0 - vignetteRadius * 0.9
    const outerEdge = innerEdge + 0.5
    const vignetteValue = 1.0 - std.smoothstep(innerEdge, outerEdge, vignetteDistance)
    const vignette = std.mix(1.0, vignetteValue, vignetteIntensity)
    return color.mul(vignette)
})

// ══════════════════════════════════════════════════════════════════════════════════════════════
// VHS — analog tape damage, per-scanline noise, chroma smear (YIQ recombine)
// ══════════════════════════════════════════════════════════════════════════════════════════════

/** Approximate NTSC scanline count. Used to quantise the fine per-row noise that gives the effect
 *  its characteristic horizontal streaking. */
const FIELD_LINES = 487.0

// Value-noise primitives (hash2D / smoothNoise2D). Hash-based → GPU-only.
const hash2D = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    // Integer bitcast hash — the noise lattices include the unbounded clock `t`, where the classic
    // sin-fract hash degrades on iOS Metal (low-precision large-argument sin range reduction).
    return noise.hash12(p)
})
const smoothNoise2D = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const i = std.floor(p)
    const f = std.fract(p)
    // Perlin-style smoothstep easing f²(3−2f).
    const u = f.mul(f).mul(d.vec2f(3.0, 3.0).sub(f.mul(2.0)))
    const a = hash2D(i)
    const b = hash2D(i.add(d.vec2f(1.0, 0.0)))
    const c = hash2D(i.add(d.vec2f(0.0, 1.0)))
    const dd = hash2D(i.add(d.vec2f(1.0, 1.0)))
    return std.mix(std.mix(a, b, u.x), std.mix(c, dd, u.x), u.y)
})

/** Burst gate: slow scrolling noise, hard-edged — turns constant wobble into intermittent bursts. */
export const burstGate = tgpu.fn([d.f32], d.f32)((t) => {
    'use gpu'
    return std.smoothstep(0.45, 0.8, smoothNoise2D(d.vec2f(t * 0.4, 0.0)))
})

/**
 * Per-scanline fine noise: quantise UV.y into fields, hash each field. Luma/chroma get different
 * time-seeded hashes (additive seed, non-degenerate at t≈0). The burst gate scales the jitter
 * (`0.25 + burst·0.75`). Returns vec2(lumaRowOffset, chromaRowOffset).
 */
export const rowJitter = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.vec2f)((uvY, t, scanlineNoise, burst) => {
    'use gpu'
    const rowInt = std.floor(uvY * FIELD_LINES)
    const h0 = (noise.hash12(d.vec2f(rowInt, t)) - 0.5) * 2.0
    const h1 = (noise.hash12(d.vec2f(rowInt, t + 69.42)) - 0.5) * 2.0
    const fineGate = 0.25 + burst * 0.75
    const chromaRowOffset = h0 * scanlineNoise * 0.008 * fineGate
    const lumaRowOffset = h1 * scanlineNoise * 0.004 * fineGate
    return d.vec2f(lumaRowOffset, chromaRowOffset)
})

/** Tape waves (two-octave low-frequency wobble, always subtly present). */
export const tapeWave = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((uvY, t, wobble) => {
    'use gpu'
    const wave1 = smoothNoise2D(d.vec2f(uvY * 3.0, t * 0.8)) - 0.5
    const wave2 = smoothNoise2D(d.vec2f(uvY * 30.0, t * 6.0)) - 0.5
    return (wave1 * 0.008 + wave2 * 0.002) * wobble
})

/** Tape crease: narrow horizontal band scrolling vertically, gated intermittently. */
export const creaseShift = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((uvY, t, wobble) => {
    'use gpu'
    const creasePhase = std.smoothstep(0.92, 0.99, std.sin(uvY * 8.0 - t * 3.77))
    const creaseNoise = std.smoothstep(0.3, 1.0, smoothNoise2D(d.vec2f(uvY * 4.77, t)))
    return creasePhase * creaseNoise * wobble * -0.018
})

/**
 * Head-switching noise: strong, localised to the top ~5% of the frame. A `smoothstep(0.06, 0, uvY)`
 * band would have reversed edges (edge0 > edge1); written as `1 − smoothstep(0, 0.06, uvY)`
 * (provably identical, satisfies WGSL's edge0<edge1). Returns vec2(switchX, switchY).
 */
export const headSwitch = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.vec2f)((uvY, t, wobble, burst) => {
    'use gpu'
    const switchPhase = 1.0 - std.smoothstep(0.0, 0.06, uvY)
    const switchNoise = smoothNoise2D(d.vec2f(uvY * 60.0, t * 14.0)) - 0.5
    const switchX = switchPhase * switchNoise * wobble * 0.09
    const switchY = switchPhase * wobble * burst * 0.02
    return d.vec2f(switchX, switchY)
})

/**
 * All per-fragment tape geometry → the luma sample UV (.xy) + chroma base UV (.zw). t = time·speed.
 * Composes the tape parts: one burst gate feeds both the row jitter and the head switch (a shared
 * intermediate — computed once here), and the wave/crease/switch offsets fold into one global X.
 */
export const vhsSampleUVs = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32], d.vec4f)(
    (uv, time, speed, wobble, scanlineNoise) => {
        'use gpu'
        const t = time * speed
        const burst = burstGate(t)
        const jitter = rowJitter(uv.y, t, scanlineNoise, burst)
        const sw = headSwitch(uv.y, t, wobble, burst)
        const globalX = tapeWave(uv.y, t, wobble) + creaseShift(uv.y, t, wobble) + sw.x
        const sy = uv.y + sw.y
        return d.vec4f(uv.x + globalX + jitter.x, sy, uv.x + globalX + jitter.y, sy)
    })

/** AC beat: very subtle brightness pulse. mod(t, 2π) is safe with std.mod (truncated) because
 *  t = time·speed ≥ 0 (speed ≥ 0.1), so floored ≡ truncated. */
export const vhsAcBeat = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32], d.f32)((uv, time, speed, wobble) => {
    'use gpu'
    const t = time * speed
    return 1.0 + std.cos(std.mod(t, 6.2831853) * 2.0 + uv.y * 0.5) * 0.015 * wobble
})

/** One chroma smear tap UV: chromaUV shifted left by i·smearScale (smearScale = smear·0.0075).
 *  Pure. Sign of smear flips the trailing side. */
export const vhsChromaTapUV = tgpu.fn([d.vec2f, d.f32, d.f32], d.vec2f)((chromaUV, smearScale, i) => {
    'use gpu'
    return d.vec2f(chromaUV.x + i * -1.0 * smearScale, chromaUV.y)
})

/**
 * YIQ recombine: sharp luma Y from the luma tap + smeared I/Q from the 6-tap chroma blur. The i=0
 * tap has weight (0/5)·(2/6)=0, so it contributes nothing and is omitted — c1..c5 carry weights
 * i/15 (from `(i/(N-1))·(2/N)`, N=6), which sum to 1. Returns the recombined RGB (unclamped).
 * Deterministic given the samples.
 */
export const yiqRecombine = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f], d.vec3f)(
    (luma, c1, c2, c3, c4, c5) => {
        'use gpu'
        const sharpY = std.dot(luma.xyz, d.vec3f(0.299, 0.587, 0.114))
        const iAxis = d.vec3f(0.596, -0.274, -0.322)
        const qAxis = d.vec3f(0.211, -0.523, 0.312)
        const accI =
            std.dot(c1.xyz, iAxis) * 0.06666666666666667 +
            std.dot(c2.xyz, iAxis) * 0.13333333333333333 +
            std.dot(c3.xyz, iAxis) * 0.2 +
            std.dot(c4.xyz, iAxis) * 0.26666666666666666 +
            std.dot(c5.xyz, iAxis) * 0.3333333333333333
        const accQ =
            std.dot(c1.xyz, qAxis) * 0.06666666666666667 +
            std.dot(c2.xyz, qAxis) * 0.13333333333333333 +
            std.dot(c3.xyz, qAxis) * 0.2 +
            std.dot(c4.xyz, qAxis) * 0.26666666666666666 +
            std.dot(c5.xyz, qAxis) * 0.3333333333333333
        return d.vec3f(
            sharpY + accI * 0.956 + accQ * 0.621,
            sharpY - accI * 0.272 - accQ * 0.647,
            sharpY - accI * 1.106 + accQ * 1.703,
        )
    })

/** Pulse the RGB by the beat factor and clamp to [0,1], preserving alpha. */
export const beatShade = tgpu.fn([d.vec4f, d.f32], d.vec4f)((color, acBeat) => {
    'use gpu'
    const clamped = std.clamp(color.xyz.mul(acBeat), d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0))
    return d.vec4f(clamped.x, clamped.y, clamped.z, color.w)
})

// ══════════════════════════════════════════════════════════════════════════════════════════════
// GLITCH — burst-gated band displacement, mirror blocks, RGB split, color bars, scanlines
// ══════════════════════════════════════════════════════════════════════════════════════════════

// Float → float pseudo-random [0,1]. Integer bitcast hash: the seeds are frame counters that grow
// with the clock, where the classic sin-fract hash degrades on iOS Metal (low-precision
// large-argument sin range reduction). GPU-only.
const hash11 = tgpu.fn([d.f32], d.f32)((p) => {
    'use gpu'
    return noise.hash11(p)
})

/**
 * Temporal pulse — bursts of glitch followed by calm. A cubed slow hash gates the major bursts, a
 * squared fast hash the minor flickers; both threshold against `1 − intensity`.
 */
export const burstPulse = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((slowFrame, fastFrame, intensity) => {
    'use gpu'
    const slowHash = hash11(slowFrame)
    const slowPulse = slowHash * slowHash * slowHash
    const fastHash = hash11(fastFrame + 73.0)
    const fastPulse = fastHash * fastHash
    const threshold = 1.0 - intensity
    const majorBurst = std.step(threshold, slowPulse)
    const minorFlicker = std.step(threshold + 0.15, fastPulse)
    return std.clamp(majorBurst * 0.8 + minorFlicker * 0.4, 0.0, 1.0) * intensity
})

/**
 * One horizontal jitter band scale: hash the band row against a frame counter, gate it by the
 * glitch strength, and displace active bands. Returns vec3(displace, active, seed) — the seed is
 * exposed so the caller can derive correlated values (vertical jitter) for the same band.
 */
export const bandJitter = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec3f)(
    (bandY, frame, seedScale, frameScale, gateFactor, hashOffset, amplitude, glitchStrength) => {
        'use gpu'
        const seed = bandY * seedScale + frame * frameScale
        const bandHash = hash11(seed)
        const active = std.step(1.0 - glitchStrength * gateFactor, bandHash)
        const displace = (hash11(seed + hashOffset) - 0.5) * amplitude * active
        return d.vec3f(displace, active, seed)
    })

/** Coarse block displacement on a 4-column grid. Returns vec2(displace, active). */
export const blockShift = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.vec2f)((colX, bandY1, slowFrame, glitchStrength) => {
    'use gpu'
    const blockSeed = colX * 0.41 + bandY1 * 0.67 + slowFrame * 2.31
    const blockHash = hash11(blockSeed)
    const blockActive = std.step(1.0 - glitchStrength * 0.35, blockHash)
    const displace = (hash11(blockSeed + 77.0) - 0.5) * 0.18 * blockActive
    return d.vec2f(displace, blockActive)
})

/** Mirror distortion — flip content horizontally/vertically in some active blocks. */
export const mirrorFlips = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (displacedUV, colX, bandY1, slowFrame, bandActive1, mirrorChance, glitchStrength) => {
        'use gpu'
        const mirrorSeed = colX * 0.83 + bandY1 * 1.17 + slowFrame * 4.57
        const hMirrorHash = hash11(mirrorSeed)
        const vMirrorHash = hash11(mirrorSeed + 500.0)
        const hMirrorActive = std.step(1.0 - mirrorChance * glitchStrength, hMirrorHash) * bandActive1
        const vMirrorActive = std.step(1.0 - mirrorChance * glitchStrength * 0.4, vMirrorHash) * bandActive1
        return d.vec2f(
            std.mix(displacedUV.x, 1.0 - displacedUV.x, hMirrorActive),
            std.mix(displacedUV.y, 1.0 - displacedUV.y, vMirrorActive),
        )
    })

/**
 * The shared glitch warp geometry: the burst pulse and band hashes feed the displacement, the
 * mirror flips, the RGB-split spread, AND the fill/scanline gates together, so they are computed
 * once here and exported as struct fields for the downstream parts. (Member reads re-emit the
 * call; Metal CSEs the identical `glitchGeom(...)` across them.)
 */
export const GlitchGeom = d.struct({
    /** The displaced + mirror-flipped sample coordinate. */
    mirroredUV: d.vec2f,
    /** `glitchStrength + |totalDisplaceX|·2` — scales the RGB-split offset. */
    spread: d.f32,
    /** The burst-gated glitch strength. */
    strength: d.f32,
    bandY1: d.f32,
    slowFrame: d.f32,
    /** bandActive1 — gates fills to active primary bands. */
    bandGate: d.f32,
    /** blockActive — gates fills to active blocks. */
    blockGate: d.f32,
    /** Any band active at any scale — confines the scanlines to distorted regions. */
    distortion: d.f32,
})

export const glitchGeom = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], GlitchGeom)(
    (uv, time, intensity, speed, blockDensity, mirrorChance) => {
        'use gpu'
        const t = time * speed
        const slowFrame = std.floor(t * 2.5)
        const fastFrame = std.floor(t * 10.0)
        const glitchStrength = burstPulse(slowFrame, fastFrame, intensity)

        // Multi-scale horizontal band displacement.
        const bandY1 = std.floor(uv.y * blockDensity)
        const band1 = bandJitter(bandY1, slowFrame, 0.37, 7.13, 0.5, 42.0, 0.25, glitchStrength)
        const bandY2 = std.floor(uv.y * (blockDensity * 2.0))
        const band2 = bandJitter(bandY2, fastFrame, 0.53, 3.37, 0.7, 99.0, 0.12, glitchStrength)
        const bandY3 = std.floor(uv.y * (blockDensity * 4.0))
        const band3 = bandJitter(bandY3, fastFrame, 0.71, 11.71, 0.9, 173.0, 0.05, glitchStrength)

        const colX = std.floor(uv.x * 4.0)
        const block = blockShift(colX, bandY1, slowFrame, glitchStrength)

        const totalDisplaceX = band1.x + band2.x + band3.x + block.x
        const vertJitter = band1.y * (hash11(band1.z + 200.0) - 0.5) * 0.012

        const displacedUV = d.vec2f(
            std.clamp(uv.x + totalDisplaceX * glitchStrength, 0.0, 1.0),
            std.clamp(uv.y + vertJitter * glitchStrength, 0.0, 1.0),
        )

        const mirroredUV = mirrorFlips(displacedUV, colX, bandY1, slowFrame, band1.y, mirrorChance, glitchStrength)

        const spread = glitchStrength + std.abs(totalDisplaceX) * 2.0
        const anyDistortion = std.clamp(band1.y + band2.y + band3.y, 0.0, 1.0)

        return GlitchGeom({
            mirroredUV,
            spread,
            strength: glitchStrength,
            bandY1,
            slowFrame,
            bandGate: band1.y,
            blockGate: block.y,
            distortion: anyDistortion,
        })
    })

/** The two RGB-split tap UVs around the mirrored coordinate, clamped to [0,1]:
 *  vec4(redUV.xy, blueUV.xy). The offset is `rgbShift·0.003·spread`. */
export const glitchSplitUVs = tgpu.fn([d.vec2f, d.f32, d.f32], d.vec4f)((mirroredUV, rgbShift, spread) => {
    'use gpu'
    const rgbOffset = rgbShift * 0.003 * spread
    const redX = std.clamp(mirroredUV.x + rgbOffset, 0.0, 1.0)
    const blueX = std.clamp(mirroredUV.x - rgbOffset, 0.0, 1.0)
    return d.vec4f(redX, mirroredUV.y, blueX, mirroredUV.y)
})

/**
 * color fills — SMPTE bars or a solid neon hue per block, mixed into the color in active
 * blocks. fillColor is a full-strength (unpremultiplied) neon color — scaled by `color.w`
 * before mixing into premultiplied-space content, so the bar respects a semi-transparent
 * child's alpha instead of reading as fully opaque and then getting over-brightened by the
 * trailing unpremultiplyAlpha.
 */
export const fillBarsShade = tgpu.fn([d.vec4f, d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec4f)(
    (color, uv, bandY1, slowFrame, colorBarMix, strength, bandGate, blockGate) => {
        'use gpu'
        const colX = std.floor(uv.x * 4.0)
        const blockSeed = colX * 0.41 + bandY1 * 0.67 + slowFrame * 2.31
        const fillStyleHash = hash11(blockSeed + 555.0)
        const barOffset = hash11(bandY1 + slowFrame * 3.33 + 333.0)
        const barX = std.fract(uv.x * 3.0 + barOffset * 7.0)
        const barIndex = std.floor(barX * 7.0)
        const barR = (1.0 - std.step(2.0, barIndex)) + std.step(4.0, barIndex) * (1.0 - std.step(6.0, barIndex))
        const barG = 1.0 - std.step(4.0, barIndex)
        const barB = 1.0 - std.step(0.5, std.fract(barIndex * 0.5))
        const smpteColor = d.vec3f(barR, barG, barB)
        const solidHue = std.fract(hash11(bandY1 + 13.0) * 0.8 + slowFrame * 0.07)
        const sh6 = solidHue * 6.0
        const solidColor = d.vec3f(
            std.clamp(std.abs(sh6 - 3.0) - 1.0, 0.0, 1.0),
            std.clamp(2.0 - std.abs(sh6 - 2.0), 0.0, 1.0),
            std.clamp(2.0 - std.abs(sh6 - 4.0), 0.0, 1.0),
        ).mul(1.3)
        const fillColor = std.mix(solidColor, smpteColor, d.vec3f(std.step(0.5, fillStyleHash)))
        const colorBarStrength = bandGate * blockGate * colorBarMix * strength
        const withBars = std.mix(color.xyz, fillColor.mul(color.w), d.vec3f(colorBarStrength))
        return d.vec4f(withBars.x, withBars.y, withBars.z, color.w)
    })

/** Scanlines confined to actively distorted regions (pixel-space sin wave, ×0.3 ceiling). */
export const distortScanShade = tgpu.fn([d.vec4f, d.f32, d.f32, d.f32, d.f32], d.vec4f)(
    (color, uvY, viewportY, scanlines, distortion) => {
        'use gpu'
        const pixelY = uvY * viewportY
        const scanlineWave = std.sin(pixelY * 3.14159265) * 0.5 + 0.5
        const scanlineEffect = 1.0 - scanlines * scanlineWave * 0.3 * distortion
        const shaded = color.xyz.mul(scanlineEffect)
        return d.vec4f(shaded.x, shaded.y, shaded.z, color.w)
    })
