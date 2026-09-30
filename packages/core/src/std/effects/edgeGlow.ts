/**
 * A band of light hugging the edge of any shape, with colored spots racing around it. The
 * words read the shape's distance field inside a `materials.shapedSurface` surface function
 * and return light: `heartbeatPulse` is the shared beat, `edgeGlowBand` the band plus the
 * angle around the shape, `orbitSpotsAccum` one color's orbiting spots, `edgeGlowAccumZero`
 * and `edgeGlowAccumColorFor` fold the colors together, and `edgeGlowCompose` turns the fold
 * into the final color.
 *
 * These are GPU functions. Invoke one with `call(fn, 'name', [args])`. The result is straight
 * alpha over a transparent background, so it composites over whatever sits behind the shape.
 */
// Maintainer: adapted from paper-design/shaders "pulsing-border" (MIT), generalized over ANY
// distance field (2D analytic, custom SVG, raymarched 3D silhouette) instead of a fixed rounded
// rectangle. The running blend/add accumulation composites any number of orbit colors (hues
// mixed in a compile-time color space, additive pile-up for bloom).
import {tgpu, d, std, noise, colorMixing, constants} from '../../gpu/kit/index'
import {MAX_COLOR_STOPS} from '../../utilities/colorStops'

const TWO_PI = constants.TWO_PI

// Border thickness reference: thickness 1 ≈ a band as wide as the default shape (radius 0.35 —
// mirrors the paper's `min(halfSize)` reference).
const BAND_THICKNESS_REF = 0.35

/**
 * A double heartbeat, 0 to 1, driven by the layer's clock.
 *
 * Compute it once per pixel and hand the same value to `edgeGlowBand` and every
 * `orbitSpotsAccum` call, so the band's smoke and the spots beat together.
 *
 * @example
 * ```ts
 * const beat = asLocal(call(heartbeatPulse, 'heartbeatPulse', [animatedTime(params)]), 'beat')
 * ```
 * @see edgeGlowBand, orbitSpotsAccum
 */
// Maintainer: two staggered sine spikes (the paper's beat()), period 1/0.18 clock seconds.
export const heartbeatPulse = tgpu.fn([d.f32], d.f32)((animTime) => {
    'use gpu'
    const time = 0.18 * animTime
    const first = std.pow(std.abs(std.sin(time * TWO_PI)), 10.0)
    const second = std.pow(std.abs(std.sin((time - 0.15) * TWO_PI)), 10.0)
    return std.clamp(first + 0.6 * second, 0.0, 1.0)
})

/**
 * The glowing band around the shape's edge, with drifting smoke, plus the angle around the shape.
 *
 * Takes the shape's signed distance (`frame.surf0.member('r')`), the shape-space uv
 * (`frame.sdfUV`), the viewport size, the `scale` prop, `thickness` (0.01 to 1; 1 is about as
 * wide as the default shape), `softness` (0 is a crisp ring, 1 a wide gradient), `smokeSize`
 * and `smokeAmt` (0 to 1), the layer's clock, `pulse` (0 to 1) and the shared `beat`. Returns
 * `.x` the band (0 to 1) and `.y` the angle around the shape in turns (0 to 1), which
 * `orbitSpotsAccum` needs.
 *
 * @example
 * ```ts
 * const band = call(edgeGlowBand, 'edgeGlowBand', [frame.surf0!.member('r'), frame.sdfUV!, params.ctx.viewportSize, uniforms.scale, uniforms.thickness, uniforms.softness, uniforms.smokeSize, uniforms.smoke, t, uniforms.pulse, beat])
 * ```
 * @see heartbeatPulse, orbitSpotsAccum
 */
// Maintainer: soft borders bleed outward, so the field is inset by mix(th, 0, softness) to keep
// the band hugging the silhouette; border = 1 − smoothstep over |f| with an aa term from the
// viewport, raised to 1 + softness. Smoke is two counter-drifting value-noise fields banded around
// the silhouette, squared and scaled by smokeAmt², heartbeat-modulated by `pulse`. Returns
// vec2(border + smoke clamped to [0,1], angle01).
export const edgeGlowBand = tgpu.fn(
    [d.f32, d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec2f,
)((fieldRaw, sdfUV, viewport, scale, thickness, softness, smokeSize, smokeAmt, animTime, pulse, beat) => {
    'use gpu'
    const t = animTime * 1.2 + 131.0
    const th = std.max(0.5 * thickness * BAND_THICKNESS_REF, 0.002)

    // Soft borders bleed outward — inset the field so the band stays hugging the silhouette.
    const f = fieldRaw / std.max(scale, 0.001) + std.mix(th, d.f32(0), softness)
    const bd = std.abs(f)

    // Band mask: hard ring at softness 0, wide gradient at softness 1.
    const borderTh = std.mix(th, 3.0 * th, softness)
    const aa = 2.0 / (viewport.y * std.max(scale, 0.001))
    const e0 = std.min(std.mix(borderTh, borderTh * -1.0, softness), borderTh + aa)
    const e1 = std.max(std.mix(borderTh, borderTh * -1.0, softness), borderTh + aa)
    let border = 1.0 - std.smoothstep(e0, e1, bd)
    border = std.pow(std.clamp(border, 0.0, 1.0), 1.0 + softness)

    // Smoke: two counter-drifting value-noise fields, banded around the silhouette, squared and
    // scaled, heartbeat-modulated by `pulse`.
    const suv = d.vec2f(sdfUV.x - 0.5, sdfUV.y - 0.5).mul(0.5 + 2.5 * smokeSize)
    const drift = 0.5 * t
    const n1 = noise.value12(d.vec2f(suv.x * 2.7 + drift, suv.y * 2.7 + drift))
    const n2 = noise.value12(d.vec2f(suv.x * 3.4 - drift, suv.y * 3.4 - drift))
    let smoke = std.clamp(3.0 * n1, 0.0, 1.0) - n2
    const smokeTh = std.clamp(th + 0.2, 0.1, 0.4)
    const smokeBand = std.pow(std.clamp(1.0 - std.smoothstep(smokeTh * -1.0, smokeTh + aa, bd), 0.0, 1.0), 2.0)
    smoke = smoke * smokeBand
    smoke = 30.0 * smoke * smoke * (0.5 * smokeAmt * smokeAmt)
    smoke = smoke * std.mix(d.f32(1), beat, pulse)
    border = std.clamp(border + std.clamp(smoke, 0.0, 1.0), 0.0, 1.0)

    const angle01 = std.atan2(sdfUV.y - 0.5, sdfUV.x - 0.5) / TWO_PI
    return d.vec2f(border, angle01)
})

/**
 * The spots of one color orbiting the band, as coverage at this pixel.
 *
 * Call it once per color. Takes the angle and band from `edgeGlowBand`, the layer's clock, the
 * shared `beat`, `spots` (1 to 5 per color), `spotSize` (0 to 1), `pulse` (0 to 1),
 * `intensity` (0 to 1), `softness`, `seed` and the color's index. Returns a pair for
 * `edgeGlowAccumColorFor`: `.x` the layered coverage and `.y` the additive sum used for bloom.
 *
 * @example
 * ```ts
 * const sectors = call(orbitSpotsAccum, 'orbitSpotsAccum', [band.member('y'), band.member('x'), t, beat, uniforms.spots, uniforms.spotSize, uniforms.pulse, uniforms.intensity, uniforms.softness, uniforms.seed, floatE(colorIndex)])
 * ```
 * @see edgeGlowBand, edgeGlowAccumColorFor
 */
// Maintainer: up to 5 spots orbit with hash-decorrelated speeds, directions, phases and sizes;
// each spot is an angular window × breathing mask × border × intensity, runtime-gated by
// `spotsF`. Returns vec2(overSector, addSector): the Porter-Duff-accumulated coverage (blend
// path) and the raw additive sum (bloom path). `beat` is evaluated ONCE per pixel by the caller
// and shared across every per-color unrolled call.
export const orbitSpotsAccum = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec2f,
)((angle01, border, animTime, beat, spotsF, spotSize, pulse, intensity, softness, seed, colorIdxF) => {
    'use gpu'
    const t = animTime * 1.2 + 131.0
    const intensityEff = 1.0 + (1.0 + 4.0 * softness) * intensity
    let overAcc = d.f32(0)
    let addAcc = d.f32(0)
    for (let s = 0; s < 5; s++) {
        const sF = d.f32(s)
        const gate = std.select(d.f32(0), d.f32(1), sF < spotsF)
        const randVal = noise.hash22(d.vec2f(sF * 10.0 + 2.0 + seed * 7.31, 40.0 + colorIdxF + seed * 3.17))

        // Per-spot orbit speed + direction flip.
        const speedFactor = 0.1 + 0.15 * std.abs(std.sin(sF * (2.0 + colorIdxF)) * std.cos(sF * (2.0 + 2.5 * colorIdxF)))
        const dir = std.select(d.f32(1), d.f32(-1), randVal.y < 0.5)
        const sTime = (speedFactor * t + randVal.x * 3.0) * dir

        // Luminance breathing at full t, sin/cos family by color parity.
        const evenCond = std.abs(colorIdxF - 2.0 * std.floor(colorIdxF * 0.5)) < 0.5
        const sinVal = std.sin(t + sF * (5.0 - 1.5 * colorIdxF))
        const cosVal = std.cos(t + sF * (3.0 + 1.3 * colorIdxF))
        let mask = 0.5 + 0.5 * std.select(cosVal, sinVal, evenCond)

        const pWeight = std.clamp(2.0 * pulse - randVal.x, 0.0, 1.0)
        mask = std.mix(mask, beat, pWeight)

        const atg = std.fract(angle01 + sTime)
        let ss = 0.05 + 0.6 * spotSize * spotSize + 0.05 * randVal.x
        ss = std.mix(ss, 0.1, pWeight)
        let sector = std.smoothstep(0.5 - ss, 0.5, atg) * (1.0 - std.smoothstep(0.5, 0.5 + ss, atg))

        sector = std.clamp(sector * mask * border * intensityEff, 0.0, 1.0) * gate
        overAcc = overAcc + (1.0 - overAcc) * sector
        addAcc = addAcc + sector
    }
    return d.vec2f(overAcc, addAcc)
})

/** @internal The running fold state threaded through the per-color loop. */
// Running accumulation threaded through the consumer's per-color unroll:
//   rgb/w — the coverage-weighted BLENDED color (hues mixed in the compile-time color space)
//   ba    — Porter-Duff coverage (the blend alpha)
//   ac/aa — raw additive pile-up (the add path, for the bloom mix; light adds linearly)
export const EdgeGlowAccum = d.struct({rgb: d.vec3f, w: d.f32, ba: d.f32, ac: d.vec3f, aa: d.f32})

/**
 * The empty state to start folding colors into.
 *
 * @example
 * ```ts
 * let state = call(edgeGlowAccumZero, 'edgeGlowAccumZero', [])
 * ```
 * @see edgeGlowAccumColorFor, edgeGlowCompose
 */
export const edgeGlowAccumZero = tgpu.fn([], EdgeGlowAccum)(() => {
    'use gpu'
    return EdgeGlowAccum({rgb: d.vec3f(0.0, 0.0, 0.0), w: d.f32(0), ba: d.f32(0), ac: d.vec3f(0.0, 0.0, 0.0), aa: d.f32(0)})
})

// Per-colorSpace accumulation fn factory (`mode` is a JS compile-time value selecting the
// mixColors variant, so only that variant's math is emitted). Folds one color's sector coverage
// (`sectors` = vec2(overSector, addSector)) into the state: where colors overlap, the running
// color is a coverage-weighted mix in the chosen space (OKLCh keeps overlapping lights vivid
// instead of muddying in linear RGB).
const accumFnByMode = new Map<number, ReturnType<typeof makeAccumFn>>()
function makeAccumFn(mode: number) {
    const variant = colorMixing.mixColorsVariants[mode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
    return tgpu.fn([EdgeGlowAccum, d.vec4f, d.vec2f], EdgeGlowAccum)((st, color, sectors) => {
        'use gpu'
        const wNew = color.w * sectors.x
        const t = wNew / std.max(st.w + wNew, 0.00001)
        const mixed = variant(d.vec4f(st.rgb, 1.0), d.vec4f(color.xyz, 1.0), t)
        const ba = st.ba + (1.0 - st.ba) * wNew
        const ac = st.ac.add(color.xyz.mul(color.w * sectors.y))
        const aa = st.aa + color.w * sectors.y
        return EdgeGlowAccum({rgb: mixed.xyz, w: st.w + wNew, ba, ac, aa})
    }).$name(`edgeGlowAccumColor_${mode}`)
}
/**
 * The fold step for one color space: adds one color's spots to the running state.
 *
 * `mode` is the compile-time value of a `colorSpace` prop (see `transformColorSpace`); where
 * colors overlap they mix in that space. The returned GPU function takes the state, a color
 * and the pair from `orbitSpotsAccum`, and returns the new state. Call it once per color, then
 * close with `edgeGlowCompose`.
 *
 * @example
 * ```ts
 * const accumColor = edgeGlowAccumColorFor(propValues.colorSpace as number)
 * state = call(accumColor, 'edgeGlowAccumColor', [state, uniforms.colorA, sectors])
 * ```
 * @tip OKLCh keeps overlapping lights vivid; linear RGB muddies them.
 * @see edgeGlowAccumZero, orbitSpotsAccum, edgeGlowCompose
 */
export function edgeGlowAccumColorFor(mode: number) {
    let fn = accumFnByMode.get(mode)
    if (!fn) {
        fn = makeAccumFn(mode)
        accumFnByMode.set(mode, fn)
    }
    return fn
}

/**
 * One color from a `stops` prop, by index.
 *
 * Reads the packed stops array (`uniforms.colorsArray` when the definition uses
 * `colorStopsPropConfig`). Stop positions are ignored; the stops are a list of orbit colors.
 *
 * @example
 * ```ts
 * const color = call(colorAtIndex, 'colorAtIndex', [uniforms.colorsArray, floatE(i)])
 * ```
 * @see edgeGlowAccumColorFor
 */
// Maintainer: copied component-wise — TypeGPU forbids returning references to fn arguments.
export const colorAtIndex = tgpu.fn([d.arrayOf(d.vec4f, MAX_COLOR_STOPS), d.f32], d.vec4f)((colors, idx) => {
    'use gpu'
    const c = colors[d.i32(idx)]
    return d.vec4f(c.x, c.y, c.z, c.w)
})

/**
 * The final color from the folded state, over a transparent background.
 *
 * `bloom` (0 to 1) sets how additively overlapping lights pile up; high values overdrive to
 * white. Returns straight alpha, ready to return from the surface function.
 *
 * @example
 * ```ts
 * return call(edgeGlowCompose, 'edgeGlowCompose', [state, uniforms.bloom])
 * ```
 * @see edgeGlowAccumColorFor
 */
// Maintainer: the `k = 4·bloom` blend/add extrapolation. The blend path premultiplies the
// space-blended color by its coverage, mixes toward the additive pile-up by k, then divides out
// the clamped alpha → STRAIGHT rgba.
export const edgeGlowCompose = tgpu.fn([EdgeGlowAccum, d.f32], d.vec4f)((st, bloom) => {
    'use gpu'
    const k = bloom * 4.0
    const blendPremult = st.rgb.mul(st.ba)
    const accumC = std.mix(blendPremult, st.ac, d.vec3f(k))
    const accumA = std.clamp(std.mix(st.ba, st.aa, k), 0.0, 1.0)
    const straight = std.clamp(accumC.div(std.max(accumA, 0.0001)), d.vec3f(0.0), d.vec3f(1.0))
    return d.vec4f(straight, accumA)
})
