/**
 * std/paint/noise — noise textures: the framings, bases, tone words and finished recipes
 * behind every grainy, cloudy, cellular or streaked surface.
 *
 * Read a noise recipe from the inside out. A basis is the raw pattern (a number per point).
 * A framing decides where on the canvas it is sampled and how big it is. A tone word turns
 * the raw number into a 0–1 ramp position. A palette gives it color. The tail of almost
 * every recipe is `rampOver(tone(framing(basis)), palette)`.
 */
// Maintainer notes (not read by the docs extractor).
//
// Every noise generator is a visible composition over the field pipeline (`./fields`): a basis
// (`noiseField` from the shared registry, or one of the parameter-taking producers below)
// sampled through a domain part (`seededPlane`, `pixelGrid`, `evolving`), shaped by a tone part
// (`noiseTone`, `signedTone`, `gainTone`), and ramped by a palette (`stops`, `pair`,
// `linearPair`). The fractal and worley recipes live in the nouns at the bottom; the voronoi
// cell parts compose in their shader definitions (Marble's veining is pure algebra in its file).
//
// Animated parts read the definition's per-node accumulated time themselves; the
// `animatedTime:` declaration stays on the definition.
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE, vec4, asLocal} from '../../gpu/composer'
import {mixExpr, animatedTime} from '../../gpu/porters'
import {colorMixing, colorStops as colorStopsKit, fields as fieldsKit, noise as noiseKit, noiseColor, noisePaints, noiseStylize, tone as toneKit} from '../../gpu/kit/index'
import {colorSpaceModeOf, rampOver, stops, type Field, type Paint, type Palette} from './fields'
import type {PropRef} from '../values'
import {uniformOf, paintFrame} from '../invoke'
import {add, mul, sin} from '../math'

// ── Waves — the organic sine-interference engine ─────────────────────────────────────────

/**
 * One traveling sine wave, written as plain data.
 *
 * `x`, `y` and `t` are the wave's frequency along that axis: larger is tighter, and `t` is
 * its drift speed along the layer's clock. Leave an axis out to ignore it. A negative value
 * runs the wave the other way. `phase` shifts the wave, in radians.
 */
export interface Wave {
    x?: number
    y?: number
    t?: number
    /** Radians, or `hashPhase(...)` for a seeded random phase. */
    phase?: Expr | number
}
// A wave is `{x: 3.2, t: 0.8}`: "a wave across x at frequency 3.2, drifting at speed 0.8".
// Coefficient `1` emits the bare axis (no `× 1`); every field is serializable data.

/**
 * One term of a wave schedule: a single wave, or several multiplied together.
 *
 * Multiplying an x-wave by a y-wave gives an interference pattern instead of stripes.
 * `weight` scales the term, default 1.
 */
export interface WaveTerm {
    waves: Wave[]
    weight?: number
}
// `weight` is emitted only when ≠ 1.

/**
 * Overlapping traveling sine waves from a schedule written as data: the organic wobble
 * behind blob edges, water lines and drifting fills.
 *
 * Pass the coordinates to wave over (`x`, optionally `y`, and `t` for the layer's clock)
 * and a list of terms. Each term multiplies its waves and the terms add up. The result is
 * signed: one wave runs −1 to 1, a schedule runs within about ±(sum of weights). It returns
 * an expression, not a field, so it slots into `paint: (params) => …` recipes and math.
 *
 * @example
 * ```ts
 * const wobble = waves({x: delta.member('x'), y: delta.member('y'), t: time}, [
 *   {waves: [{x: 3, t: 0.8}, {y: 2.2, t: -0.5}]},
 *   {waves: [{x: 5.1, y: -3.3, t: 0.6}], weight: 0.5},
 * ])
 * ```
 * @tip Keep the schedule as constants next to the definition. It is the look, so keep it where you can tune it.
 * @see wavyLine, organicWaves, hashPhase
 */
export function waves(axes: {x: Expr | number; y?: Expr | number; t?: Expr | number}, schedule: WaveTerm[]): Expr {
    // Evaluates `Σ weightᵢ · Π sin(x·kx + y·ky + t·kt + phase)` over the named axes, SIGNED
    // output (tone words shape it; `warped` coords warp it — both stay outside, deliberately).
    // Emission is deterministic — terms sum in schedule order, waves multiply in wave order,
    // axes emit x → y → t → phase — so a ported schedule reproduces its original arithmetic.
    // Schedules are the caller's look; `organicWaves` generates one from intent when nobody
    // has hand-tuned constants to preserve.
    const axisTerm = (axis: Expr | number | undefined, k: number | undefined, name: string): Expr | number | undefined => {
        if (k === undefined) return undefined
        if (axis === undefined) throw new Error(`std: waves() schedule uses axis '${name}' but the axes object doesn't provide it`)
        return k === 1 ? axis : mul(axis, k)
    }
    const waveExpr = (w: Wave): Expr => {
        const parts = [
            axisTerm(axes.x, w.x, 'x'),
            axisTerm(axes.y, w.y, 'y'),
            axisTerm(axes.t, w.t, 't'),
            w.phase,
        ].filter((part): part is Expr | number => part !== undefined)
        if (parts.length === 0) throw new Error('std: a wave needs at least one axis coefficient or phase')
        return sin(parts.reduce((a, b) => add(a, b)))
    }
    const terms = schedule.map((term) => {
        if (term.waves.length === 0) throw new Error('std: a waves() term needs at least one wave')
        const product = term.waves.map(waveExpr).reduce((a, b) => mul(a, b))
        return term.weight === undefined || term.weight === 1 ? product : mul(product, term.weight)
    })
    if (terms.length === 0) throw new Error('std: waves() needs at least one term')
    return terms.reduce((a, b) => add(a, b))
}

/**
 * The height of an animated wavy line at a point along it: a stack of traveling sines.
 *
 * `along` is the coordinate along the line and `time` the layer's clock. Each wave has a
 * `freq` (tightness), a `speed` (drift along the clock), an `amount` (height) and an
 * optional `phase` in radians. Signed, within ±(sum of amounts). Add it to a baseline to get
 * the curve. Give each line of a family its own `hashPhase` so they do not move in step.
 *
 * @example
 * ```ts
 * const arc = wavyLine({along: x, time: t, waves: [{freq: 1.4, speed: 0.3, amount: 0.5}, {freq: 3.1, speed: -0.2, amount: 0.2}]})
 * ```
 * @see waves, hashPhase
 */
export function wavyLine(opts: {
    along: Expr | number
    time: Expr | number
    waves: {freq: number; speed: number; amount: number; phase?: Expr | number}[]
}): Expr {
    // Sugar over `waves` on one axis. Curtain paths, water horizons, wavy baselines, cloth edges.
    return waves({x: opts.along, t: opts.time}, opts.waves.map((w) => ({
        waves: [{x: w.freq, t: w.speed, phase: w.phase}],
        weight: w.amount,
    })))
}

/**
 * A ready-made `waves` schedule from three knobs, for when you have no hand-tuned constants.
 *
 * `frequency` sets the size of the largest waves. `detail` is how many finer layers stack on
 * top (default 3, each about 1.8× finer and half as strong). `drift` is the animation speed
 * (default 0.5). The same `seed` always gives the same schedule. Weights are normalized so
 * the summed wave stays within about −1 to 1. Treat the result as a starting point to tune.
 *
 * @example
 * ```ts
 * const wobble = waves({x: delta.member('x'), y: delta.member('y'), t: time}, organicWaves({frequency: 3, detail: 3, drift: 0.4}))
 * ```
 * @tip The schedule uses x, y and t, so the axes you pass to `waves` must provide all three.
 * @see waves
 */
export function organicWaves(opts: {frequency: number; detail?: number; drift?: number; seed?: number}): WaveTerm[] {
    // Three knobs + seed, deliberately (knobs grow under pressure from real use, never
    // speculatively). Deterministic per seed. TUNING CANDIDATE — constants pending an eye pass;
    // treat the output as a starting point, not a spec.
    const detail = Math.max(1, Math.round(opts.detail ?? 3))
    const drift = opts.drift ?? 0.5
    let state = ((opts.seed ?? 1) >>> 0) || 1
    const rand = () => {
        state = (state * 1664525 + 1013904223) >>> 0
        return state / 2 ** 32
    }
    const jitter = () => 0.8 + rand() * 0.4
    const terms: WaveTerm[] = []
    let weight = 1
    let totalWeight = 0
    for (let i = 0; i < detail; i++) {
        const f = opts.frequency * 1.8 ** i
        terms.push({
            waves: [
                {x: f * jitter(), t: drift * jitter() * (rand() < 0.5 ? -1 : 1)},
                {y: f * jitter(), t: drift * jitter()},
            ],
            weight,
        })
        // A diagonal single every other layer breaks up the grid feel of pure x×y products.
        if (i % 2 === 0) {
            terms.push({
                waves: [{x: f * 1.5 * jitter(), y: -f * 1.1 * jitter(), t: -drift * jitter()}],
                weight: weight * 0.8,
            })
            totalWeight += weight * 0.8
        }
        totalWeight += weight
        weight *= 0.5
    }
    // Normalize so the summed weight is 1 — amplitude is the caller's knob.
    return terms.map((t) => ({...t, weight: (t.weight ?? 1) / totalWeight}))
}

// ── Domain parts ────────────────────────────────────────────────────────────────────────

/**
 * Frames a field on a zoomable plane: how big the pattern is and where it starts.
 *
 * Takes the canvas position, corrects it for aspect ratio, zooms it by `scale` and slides it
 * by `seed`, then hands a 2D pattern position to the field inside. `scale` is exponential:
 * each +1 makes the pattern about 2.7× finer, so a slider from −2 to 5 runs from coarse to
 * fine. A different `seed` shows a different part of the pattern. The framing every noise
 * texture opens with.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(unitized(gaborGrains({frequency: p('frequency')})), {scale: p('scale'), seed: p('seed')})), stops(p('colorSpace')))
 * ```
 * @tip `noiseField` bases take a 3D position. Put `evolving` between `seededPlane` and the basis, even at `rate: 0`, to supply the third axis.
 * @see evolving, pixelGrid, scaledPlane
 */
export function seededPlane(field: Field, slots: {scale: PropRef; seed: PropRef}): Field {
    // The exponential-scale noise framing: aspect-correct the UV (guarded divide), scale by
    // `exp(scale)`, offset by `seed`. Hands the inner field a vec2 pattern position.
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        return field(params, call(fieldsKit.aspectScaledDomain, 'aspectScaledDomain', [
            uv, viewport, uniformOf(slots.scale, params), uniformOf(slots.seed, params),
        ]))
    }
}

/**
 * @internal Scratches' historical framing: `seededPlane` with an unguarded aspect divide,
 * kept so that shader's exact arithmetic is preserved. Authors use `seededPlane`.
 */
export function rawPlane(field: Field, slots: {scale: PropRef; seed: PropRef}): Field {
    // The RAW-divide exponential-scale framing (Scratches' historical domain — the unguarded
    // aspect divide is part of its exact arithmetic).
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        return field(params, call(noisePaints.rawAspectScaledDomain, 'rawAspectScaledDomain', [
            uv, viewport, uniformOf(slots.scale, params), uniformOf(slots.seed, params),
        ]))
    }
}

/**
 * Frames a field on the pixel grid: one value per block of `grain` device pixels.
 *
 * For patterns that live on pixels rather than on the canvas (blue noise, dither). `grain`
 * is the block size in pixels (1 = every pixel), `seed` picks a different block of the
 * pattern. The field inside gets whole-number pixel coordinates.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(pixelGrid(blueSpeckle(), {grain: p('grain'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see blueSpeckle, seededPlane
 */
export function pixelGrid(field: Field, slots: {grain: PropRef; seed: PropRef}): Field {
    // The per-pixel framing: floor the device-pixel position into `grain`-sized cells, offset
    // by `seed` — for patterns defined on the pixel grid (blue noise) rather than in UV space.
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        return field(params, call(fieldsKit.pixelGridDomain, 'pixelGridDomain', [
            uv, viewport, uniformOf(slots.grain, params), uniformOf(slots.seed, params),
        ]))
    }
}

/**
 * Frames a field on the aspect-corrected canvas plane, zoomed by a plain multiplier.
 *
 * Unlike `seededPlane`, `scale` is linear: roughly the number of pattern cells across the
 * canvas height. There is no seed. The position is computed once, so a recipe whose parts
 * read it several times pays for it once.
 *
 * @example
 * ```ts
 * paint: rampOver(unitized(scaledPlane(waveletBands({detail: p('detail')}), {scale: p('scale')})), stops(p('colorSpace')))
 * ```
 * @see seededPlane, cellDistances
 */
export function scaledPlane(field: Field, slots: {scale: PropRef}): Field {
    // The linear-scale noise framing: aspect-correct the UV and multiply by `scale` directly
    // (no exponential remap), bound as a local — for recipes whose parts read the same plane
    // position several times.
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        const pos = asLocal(
            call(noisePaints.aspectPlane, 'aspectPlane', [uv, viewport]).mul(uniformOf(slots.scale, params)),
            'plane',
        )
        return field(params, pos)
    }
}

/**
 * Makes a 3D noise basis morph in place over time instead of sliding across the canvas.
 *
 * Wraps a basis that takes a 3D position (`noiseField`): the 2D pattern position from the
 * framing outside becomes `(x, y, time × rate)`. `rate` multiplies the layer's clock, on top
 * of the definition's `speed` prop. It sits between the framing and the basis.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(evolving(noiseField('perlin'), {rate: 0.3}), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @tip The clock is the definition's `animatedTime: {speed: 'speed'}`. Without it nothing moves.
 * @see seededPlane, curlSpeed
 */
export function evolving(field: Field, opts: {rate: number}): Field {
    // Lift a vec2 pattern position into 3D by walking the third axis with time — the
    // difference between noise that MORPHS in place (this) and noise that slides. `rate`
    // scales the evolution independently of the node's `speed` prop.
    return (params, coord) => field(params, call(fieldsKit.timeAxisDomain, 'timeAxisDomain', [
        coord, animatedTime(params), floatE(opts.rate),
    ]))
}

// ── Basis producers the NOISE_BASES registry shape doesn't fit ──────────────────────────
// (Registry bases are `(vec3) → f32`; these take a phase, an extra shape parameter, or a
// vec2 pixel-grid coordinate. The basis math itself stays atomic in kit/noise.ts.)

/**
 * Oriented sine grains: short striped patches, each facing a random way, like brushed metal
 * or wood grain.
 *
 * `frequency` is how many stripes fit in one grain (1 to 24, default 8). The stripes slide
 * along their grain on the layer's clock. Signed, −1 to 1: wrap it in `unitized` before a
 * tone word. Takes a 2D position, so frame it with `seededPlane`.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(unitized(gaborGrains({frequency: p('frequency')})), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see waveletBands, unitized
 */
export function gaborGrains(slots: {frequency: PropRef}): Field {
    return (params, coord) => call(noiseKit.gabor12, 'gabor12', [
        coord, uniformOf(slots.frequency, params), animatedTime(params),
    ])
}

/**
 * Rotating bands of wavelets: layered ripples that turn against each other, a watery
 * interference look.
 *
 * `detail` is the frequency ratio between the four layers (1.05 to 2, default 1.24; higher
 * adds finer ripples). The bands slide on the layer's clock. Signed, about −1 to 1: wrap it
 * in `unitized` before a tone word. Takes a 2D position.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(unitized(waveletBands({detail: p('detail')})), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see gaborGrains, unitized
 */
export function waveletBands(slots: {detail: PropRef}): Field {
    return (params, coord) => call(noiseKit.wavelet12, 'wavelet12', [
        coord, animatedTime(params), uniformOf(slots.detail, params),
    ])
}

/**
 * Fine, even speckle with no clumps: blue noise, one random value per pixel block.
 *
 * Still (no animation). Unit range, about 0 to 1, so it feeds `noiseTone` directly. Only
 * makes sense over `pixelGrid`, which gives it the pixel coordinates it expects.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(pixelGrid(blueSpeckle(), {grain: p('grain'), seed: p('seed')})), stops(p('colorSpace')))
 * ```
 * @tip A good dither or film-grain layer over another shader at low opacity.
 * @see pixelGrid
 */
export function blueSpeckle(): Field {
    // Spatial-high-pass blue-noise speckle over a pixel-grid coordinate.
    return (_params, coord) => call(noiseKit.blue12, 'blue12', [coord])
}

/**
 * Terrain eroded by water: sharp branching ridges with gullies running off them.
 *
 * Still (no animation). Signed, centred on 0: wrap it in `unitized` before a tone word.
 * Takes a 2D position, so frame it with `seededPlane`.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(unitized(erosionRidges()), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see unitized, seededPlane
 */
export function erosionRidges(): Field {
    // Branching hydraulic-erosion ridge height (the `.x` of the erosion field).
    return (_params, coord) => call(noiseKit.erosion12, 'erosion12', [coord]).member('x')
}

/**
 * How fast a swirling flow moves at each point: soft eddies and streams that morph in place.
 *
 * `rate` multiplies the layer's clock. Never negative and mostly 0 to 1, so it feeds
 * `noiseTone` directly. Takes a 2D position, so frame it with `seededPlane`.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(curlSpeed({rate: 0.2}), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see evolving, noiseTone
 */
export function curlSpeed(opts: {rate: number}): Field {
    // Swirling curl-flow speed (√2-normalized magnitude), morphing in place at `rate` × the clock.
    return (params, coord) => call(noisePaints.curlMagnitude, 'curlMagnitude', [
        coord, animatedTime(params).mul(opts.rate),
    ])
}

/**
 * Fine hairline scratches, like a scuffed surface, flickering on the layer's clock.
 *
 * `thickness` widens the bright core of each streak (0.2 to 5, default 1). The value is 0
 * off a streak and rises well past 1 on one, so send it through `noiseTone`, which clamps.
 * Takes a 2D position. Works in a `paint:` only, not in a compute pass.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(scratchStreaks({thickness: p('thickness')}), {scale: p('scale'), seed: p('seed')})), linearPair(p('colorA'), p('colorB')))
 * ```
 * @see noiseTone, linearPair
 */
export function scratchStreaks(slots: {thickness: PropRef}): Field {
    // Fragment-only (fwidth).
    return (params, coord) => call(noiseKit.scratches12, 'scratches12', [
        coord, animatedTime(params), uniformOf(slots.thickness, params),
    ])
}

// ── Tone parts ──────────────────────────────────────────────────────────────────────────

/**
 * Turns a signed field (−1 to 1) into a unit field (0 to 1).
 *
 * Put it directly around a signed basis (`gaborGrains`, `waveletBands`, `erosionRidges`,
 * `noiseField('mx3signed')`) so `noiseTone` and the palette see 0 to 1. Or skip it and use
 * `signedTone`, which takes the signed value itself.
 *
 * @example
 * ```ts
 * seededPlane(unitized(gaborGrains({frequency: p('frequency')})), {scale: p('scale'), seed: p('seed')})
 * ```
 * @see signedTone, noiseTone
 */
export function unitized(field: Field): Field {
    // `v · 0.5 + 0.5`.
    return (params, coord) => call(toneKit.signedToUnit, 'signedToUnit', [field(params, coord)])
}

/**
 * The standard contrast and balance controls of a noise texture, turning a 0–1 field into
 * the ramp position.
 *
 * `contrast` steepens the field around mid-grey (0 = unchanged, positive = sharper, negative
 * = flatter). `balance` shifts the whole range (positive = more of the first color). Leave
 * the slots out for a texture without those controls. The result is flipped: where the
 * field is high the ramp shows its first color. Expects a unit field, so put `unitized`
 * around a signed basis first.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(evolving(noiseField('perlin'), {rate: 0.3}), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @tip Contrast here is additive: 0 is the identity, not 1. Default the prop to 0.
 * @see signedTone, gainTone, unitized
 */
export function noiseTone(field: Field, slots?: {contrast?: PropRef; balance?: PropRef}): Field {
    // The shared noise tone tail: additive contrast about mid-grey + balance shift, inverted so
    // a ramp's colorA reads as the LOW end of the ramp parameter. `contrast`/`balance` omitted
    // = 0 (identity) for textures without tone controls.
    return (params, coord) => call(noiseColor.noiseToneKColor, 'noiseToneKColor', [
        field(params, coord),
        slots?.contrast ? uniformOf(slots.contrast, params) : floatE(0),
        slots?.balance ? uniformOf(slots.balance, params) : floatE(0),
    ])
}

/**
 * Contrast and balance for a signed field (−1 to 1), applied before it is squashed to 0–1.
 *
 * The same controls as `noiseTone` (`contrast` 0 = unchanged, `balance` shifts), but working
 * on the raw signed value lets high contrast push further before it clips. Flipped like
 * `noiseTone`: a high field shows the first color. Use it in place of `unitized` plus
 * `noiseTone`.
 *
 * @example
 * ```ts
 * paint: rampOver(signedTone(seededPlane(evolving(noiseField('mx3signed'), {rate: 0.5}), {scale: p('scale'), seed: p('seed')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see noiseTone, unitized
 */
export function signedTone(field: Field, slots: {contrast: PropRef; balance: PropRef}): Field {
    // The signed tone tail: contrast/balance applied to the RAW [-1,1] field BEFORE the squash
    // to [0,1] (then inverted), so high contrast rides further before clipping.
    return (params, coord) => call(toneKit.toneSignedInverted, 'noiseToneSigned', [
        field(params, coord), uniformOf(slots.contrast, params), uniformOf(slots.balance, params),
    ])
}

/**
 * Contrast as a multiplier and balance as a shift over a 0–1 field, not flipped.
 *
 * `contrast` 1 leaves the field alone (a slider from 0.25 to 4 works well) and `balance`
 * shifts it. Low field values read the first color. The tone inside `worleyNoise`. Reach for
 * it when a "1 = normal" contrast slider fits better than `noiseTone`'s "0 = normal".
 *
 * @example
 * ```ts
 * paint: rampOver(gainTone(cellFill(cellDistances({scale: p('scale'), seed: p('seed')}), {edgeIntensity: p('edgeIntensity')}), {contrast: p('contrast'), balance: p('balance')}), stops(p('colorSpace')))
 * ```
 * @see noiseTone, worleyNoise
 */
export function gainTone(field: Field, slots: {contrast: PropRef; balance: PropRef}): Field {
    // The multiplicative tone tail (contrast is a gain, 1 = identity), non-inverted — Worley's shape.
    return (params, coord) => call(noisePaints.worleyTone, 'worleyTone', [
        field(params, coord), uniformOf(slots.contrast, params), uniformOf(slots.balance, params),
    ])
}

// ── Palettes ────────────────────────────────────────────────────────────────────────────

/**
 * A two-color palette mixed in linear RGB, for a definition with no color-space prop.
 *
 * Reads two color props. Use `pair` or `stops` from paint when the definition has a
 * `colorSpace` prop.
 *
 * @example
 * ```ts
 * paint: rampOver(noiseTone(seededPlane(scratchStreaks({thickness: p('thickness')}), {scale: p('scale'), seed: p('seed')})), linearPair(p('colorA'), p('colorB')))
 * ```
 * @see pair, stops
 */
export function linearPair(a: PropRef, b: PropRef): Palette {
    return (t, params) => call(colorMixing.mixColorsLinear, 'mixColors', [
        uniformOf(a, params), uniformOf(b, params), t,
    ])
}

// ── Structured-texture recipes ──────────────────────────────────────────────────────────

/**
 * A complete fractal-noise paint: layered simplex noise that morphs over time, read through
 * the color ramp.
 *
 * `angle` rotates the pattern (degrees). `octaves` is how many layers stack (1 to 8, changed
 * live without a recompile), `detail` how much finer each layer is (1 to 4) and `contrast`
 * how much weaker (0.1 to 1). `seed` re-mixes the layers. `space` is the color-space prop.
 * Reads the definition's `colorA`, `colorB` and, when present, `stops` props.
 *
 * @example
 * ```ts
 * paint: fractalNoise({angle: p('angle'), detail: p('detail'), contrast: p('contrast'), octaves: p('octaves'), seed: p('seed'), space: p('colorSpace')})
 * ```
 * @tip Declare `animatedTime: {speed: 'speed'}` on the definition. The morphing runs on the layer's clock.
 * @see worleyNoise, noiseField
 */
export function fractalNoise(slots: {
    angle: PropRef
    detail: PropRef
    contrast: PropRef
    octaves: PropRef
    seed: PropRef
    space: PropRef
}): Paint {
    // Multi-octave fractal Brownian motion: rotated centred plane → gated 8-octave golden-angle
    // fBm over MaterialX simplex → weight-normalized unit field → stop ramp. `octaves` gates
    // the fixed loop at runtime, so the slider does not recompile.
    const ridges: Field = (params, uv) => {
        const {viewport} = paintFrame(params)
        const pos = call(noisePaints.rotatedCenteredDomain, 'rotatedCenteredDomain', [
            uv, viewport, uniformOf(slots.angle, params),
        ])
        const sum = call(noisePaints.fractalSum, 'fractalSum', [
            pos, uniformOf(slots.detail, params), uniformOf(slots.contrast, params),
            animatedTime(params), uniformOf(slots.seed, params), uniformOf(slots.octaves, params),
        ])
        return call(noisePaints.fbmNormalize, 'fbmNormalize', [sum])
    }
    return rampOver(ridges, stops(slots.space))
}

/**
 * A complete cellular-noise paint: distances to drifting cell points, read through the
 * color ramp.
 *
 * `scale` is the number of cells across (1 to 30) and `jitter` how far the points wander
 * from a grid (0 = rigid grid, 1 = random). `mode` picks the distance field: `'f1'`
 * (cells), `'f2'`, `'f2MinusF1'` (edges), `'f1PlusF2'`, `'f1TimesF2'`. `distance` picks the
 * cell shape: `'euclidean'` (round), `'manhattan'` (diamond), `'chebyshev'` (square).
 * `octaves` (1 to 4) stacks finer layers, with `lacunarity` and `persistence` setting how
 * much finer and weaker. `contrast` is a multiplier (1 = unchanged) and `balance` a shift.
 * Low values read the first color. `mode`, `distance` and `octaves` are compile-time props.
 * Reads the definition's `colorA`, `colorB` and, when present, `stops` props.
 *
 * @example
 * ```ts
 * paint: worleyNoise({scale: p('scale'), jitter: p('jitter'), lacunarity: p('lacunarity'), persistence: p('persistence'), contrast: p('contrast'), balance: p('balance'), seed: p('seed'), mode: p('mode'), distance: p('distance'), octaves: p('octaves'), space: p('colorSpace')})
 * ```
 * @tip Give the `mode` and `distance` props `transformWorleyMode` and `transformWorleyDistance` as their `transform`.
 * @see cellDistances, transformWorleyMode, transformWorleyDistance, fractalNoise
 */
export function worleyNoise(slots: {
    scale: PropRef
    jitter: PropRef
    lacunarity: PropRef
    persistence: PropRef
    contrast: PropRef
    balance: PropRef
    seed: PropRef
    /** Compile-time prop: `'f1' | 'f2' | 'f2MinusF1' | 'f1PlusF2' | 'f1TimesF2'`. */
    mode: PropRef
    /** Compile-time prop: `'euclidean' | 'manhattan' | 'chebyshev'`. */
    distance: PropRef
    /** Compile-time prop: 1 to 4. */
    octaves: PropRef
    space: PropRef
}): Paint {
    // Cellular (Worley) noise: aspect plane → gated 4-octave cellular sum (nearest-2 distances
    // under a selectable metric, reduced per `mode`) → per-mode normalization → gain tone →
    // ramp. `mode`/`distance`/`octaves` are compile-time props read as CPU values and folded
    // to literals.
    const cells: Field = (params, uv) => {
        const {viewport} = paintFrame(params)
        const modeNum = noisePaints.worleyModeNum(params.propValues[slots.mode.name])
        const distNum = noisePaints.worleyDistNum(params.propValues[slots.distance.name])
        const octavesNum = Math.max(1, Math.min(4, Math.round((params.propValues[slots.octaves.name] as number) ?? 1)))
        const pos = call(noisePaints.aspectPlane, 'aspectPlane', [uv, viewport])
        const sum = call(noisePaints.worleyCells, 'worleyCells', [
            pos, uniformOf(slots.scale, params), uniformOf(slots.seed, params),
            uniformOf(slots.jitter, params), uniformOf(slots.lacunarity, params),
            uniformOf(slots.persistence, params), animatedTime(params),
            floatE(octavesNum), floatE(distNum), floatE(modeNum),
        ])
        return call(noisePaints.worleyNormalize, 'worleyNormalize', [
            sum, floatE(noisePaints.worleyModeScale(modeNum, distNum)),
        ])
    }
    // No `stops` prop on the definition → the ramp always takes its two-color branch; the
    // palette is here for the compile-time colorSpace dispatch.
    return rampOver(gainTone(cells, {contrast: slots.contrast, balance: slots.balance}), stops(slots.space))
}

// ── Voronoi cell parts ──────────────────────────────────────────────────────────────────

/**
 * Distances from each pixel to its nearest two cell points, as `(d1, d2)`: the raw material
 * for Voronoi fills and borders.
 *
 * `scale` is the number of cells across and `seed` picks a different set of points. The
 * points drift on the layer's clock. Feed it to `cellFill` or `cellBorders`, and wrap it in
 * `share` when both read it.
 *
 * @example
 * ```ts
 * const cells = share(cellDistances({scale: p('scale'), seed: p('seed')}))
 * ```
 * @see cellFill, cellBorders, share
 */
export function cellDistances(slots: {scale: PropRef; seed: PropRef}): Field {
    // Nearest-two cellular distances `vec2(d1, d2)` over the aspect plane at `scale`, the cell
    // points drifting on the node's clock.
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        const pos = call(noisePaints.aspectPlane, 'aspectPlane', [uv, viewport]).mul(uniformOf(slots.scale, params))
        return call(noisePaints.voronoiCells, 'voronoiNearest2', [pos, animatedTime(params), uniformOf(slots.seed, params)])
    }
}

/**
 * A 0–1 fill for each cell: 0 at the cell's point, 1 at its boundary, whatever the cell size.
 *
 * `edgeIntensity` (0 to 1) is how far the boundary color reaches into the cell. Takes the
 * `(d1, d2)` field from `cellDistances`. Ramp it directly: the first color lands at the cell
 * centres.
 *
 * @example
 * ```ts
 * paint: rampOver(cellFill(cells, {edgeIntensity: p('edgeIntensity')}), stops(p('colorSpace')))
 * ```
 * @see cellDistances, cellBorders
 */
export function cellFill(cells: Field, slots: {edgeIntensity: PropRef}): Field {
    // The F1/F2 fill gradient.
    return (params, coord) => call(noisePaints.cellFill, 'cellFill', [
        cells(params, coord), uniformOf(slots.edgeIntensity, params),
    ])
}

/**
 * A mask of the lines between cells: 0 on a boundary, 1 inside a cell.
 *
 * `softness` is the line width (0 to about 0.4). Pass the same `scale` as `cellDistances`
 * so the lines keep their width when the cell count changes. Takes the `(d1, d2)` field.
 * Use it as the `mask` of `borderOverlay`.
 *
 * @example
 * ```ts
 * cellBorders(cells, {softness: p('edgeSoftness'), scale: p('scale')})
 * ```
 * @see cellDistances, borderOverlay
 */
export function cellBorders(cells: Field, slots: {softness: PropRef; scale: PropRef}): Field {
    // `scale` compensates the line width for the cell count.
    return (params, coord) => call(noisePaints.cellBorders, 'cellBorders', [
        cells(params, coord), uniformOf(slots.softness, params), uniformOf(slots.scale, params),
    ])
}

/**
 * Draws a color over a paint wherever a mask falls to 0, keeping the paint's alpha.
 *
 * `mask` is 1 where the paint shows through and 0 where `color` covers it. The Voronoi
 * border move: fill the cells, then overlay the border lines.
 *
 * @example
 * ```ts
 * paint: borderOverlay(rampOver(cellFill(cells, {edgeIntensity: p('edgeIntensity')}), stops(p('colorSpace'))), cellBorders(cells, {softness: p('edgeSoftness'), scale: p('scale')}), {color: p('colorBorder')})
 * ```
 * @see cellBorders, cellFill
 */
export function borderOverlay(paint: Paint, mask: Field, slots: {color: PropRef}): Paint {
    // The base binds as a local: both the overlay mix and the alpha read it.
    return (params) => {
        const {uv} = paintFrame(params)
        const base = asLocal(paint(params), 'base')
        const rgb = mixExpr(uniformOf(slots.color, params).member('rgb'), base.member('rgb'), mask(params, uv))
        return vec4(rgb, base.member('a'))
    }
}

// ── Strands parts ───────────────────────────────────────────────────────────────────────

/** A color-only step for `overRgb`: takes a paint's rgb and returns new rgb. Alpha is not touched. */
export type RgbStage = (rgb: Expr, params: GpuFragmentParams) => Expr

/**
 * Runs color-only steps over a paint, in order, leaving its alpha alone.
 *
 * This is how `ribbons` is closed: the ribbons come out in the working color space,
 * `backToP3` brings them to display color and `tonePow` lifts them.
 *
 * @example
 * ```ts
 * const strands = ribbons({from: p('start'), to: p('end'), count: p('lineCount'), width: p('lineWidth'), amplitude: p('amplitude'), frequency: p('frequency'), softness: p('softness'), spread: p('spread'), pinEdges: p('pinEdges'), colorScale: p('colorScale'), colorVariance: p('colorVariance')})
 * paint: overRgb(strands, backToP3({space: p('colorSpace')}), tonePow())
 * ```
 * @see ribbons, backToP3, tonePow
 */
export function overRgb(paint: Paint, ...stages: RgbStage[]): Paint {
    return (params) => {
        const base = paint(params)
        let rgb = base.member('rgb')
        for (const stage of stages) rgb = stage(rgb, params)
        return vec4(rgb, base.member('a'))
    }
}

/**
 * Brings `ribbons` color from its working color space back to display color.
 *
 * `space` is the definition's compile-time color-space prop; leave it out for linear. Put it
 * first in `overRgb`, right after the ribbons.
 *
 * @example
 * ```ts
 * paint: overRgb(strands, backToP3({space: p('colorSpace')}), tonePow())
 * ```
 * @see overRgb, ribbons
 */
export function backToP3(slots: {space?: PropRef}): RgbStage {
    // The closing working-space → P3-linear back-conversion at the compile-time `space` mode.
    return (rgb, params) => {
        const mode = slots.space ? colorSpaceModeOf(params.propValues[slots.space.name]) : 0
        return colorStopsKit.backConvertToP3(rgb, mode)
    }
}

/**
 * A gentle lift of the mid-tones (rgb to the power 0.85), as an `overRgb` step.
 *
 * @example
 * ```ts
 * paint: overRgb(strands, backToP3({space: p('colorSpace')}), tonePow())
 * ```
 * @see overRgb
 */
export function tonePow(): RgbStage {
    return (rgb) => call(noisePaints.toneLift, 'toneLift', [rgb])
}

/**
 * Flowing gradient-colored ribbons between two points: the Strands look.
 *
 * `from` and `to` are position props. `count` is how many ribbons, `width` and `softness`
 * their thickness and edge (0 to 1), `spread` how far apart they sit (0 to 1), `amplitude`
 * and `frequency` the size and tightness of the wave (0 to 5). `pinEdges` is a boolean prop
 * that gathers the ribbons at both ends. `colorScale` (1 to 6) is how many times the
 * gradient repeats along a ribbon and `colorVariance` (0 to 1) how far each ribbon is offset
 * in it. The gradient comes from the definition's `stops` prop. The color comes out in the
 * working color space, so close it with `overRgb(ribbons(…), backToP3({space}), tonePow())`.
 * The wave runs on `animatedTime: {speed: 'speed'}` and the colors scroll on a second clock,
 * `extraAnimatedTimes: {color: 'colorSpeed'}`. Declare both on the definition.
 *
 * @example
 * ```ts
 * paint: overRgb(
 *   ribbons({from: p('start'), to: p('end'), count: p('lineCount'), width: p('lineWidth'), amplitude: p('amplitude'), frequency: p('frequency'), softness: p('softness'), spread: p('spread'), pinEdges: p('pinEdges'), colorScale: p('colorScale'), colorVariance: p('colorVariance')}),
 *   backToP3({space: p('colorSpace')}),
 *   tonePow(),
 * )
 * ```
 * @tip Add `stops: colorStopsPropConfig()` to the props. The ribbons read the stop list, not `colorA` / `colorB`.
 * @see overRgb, backToP3, tonePow
 */
export function ribbons(slots: {
    from: PropRef
    to: PropRef
    count: PropRef
    width: PropRef
    amplitude: PropRef
    frequency: PropRef
    softness: PropRef
    spread: PropRef
    pinEdges: PropRef
    colorScale: PropRef
    colorVariance: PropRef
}): Paint {
    // The ribbons genre primitive: an ATOMIC runtime-count reduce — each ribbon's working-space
    // gradient lookup lives inside the loop — returning working-space rgba. Close it with
    // `backToP3` (and `tonePow`) via `overRgb`. Reads BOTH of the definition's clocks: the main
    // one drives the wave motion, the `color` extra clock (`extraAnimatedTimes: {color: ...}`
    // stays declared on the definition) scrolls the colors.
    return (params) => {
        const {uniforms} = params
        const {uv, viewport} = paintFrame(params)
        const waveT = animatedTime(params)
        const colorT = animatedTime(params, undefined, '_animTime_color')

        // Pack scalars into vec4 params (ribbonsField is over the 15-arg tgpu.fn cap otherwise).
        const packed1 = vec4(
            uniformOf(slots.amplitude, params), uniformOf(slots.frequency, params),
            uniformOf(slots.count, params), uniformOf(slots.width, params),
        )
        const packed2 = vec4(
            uniformOf(slots.softness, params), uniformOf(slots.spread, params),
            uniformOf(slots.colorScale, params), uniformOf(slots.colorVariance, params),
        )
        const packed3 = vec4(uniformOf(slots.pinEdges, params), waveT, colorT, uniforms.stopCount)

        return call(noisePaints.ribbonsField, 'ribbonsField', [
            uv, viewport, uniformOf(slots.from, params), uniformOf(slots.to, params),
            packed1, packed2, packed3,
            uniforms.colorsArray, uniforms.positionsArray, uniforms.convertedColorsArray,
        ])
    }
}

// ── Relief + paper filters ──────────────────────────────────────────────────────────────

/** A height field `noiseRelief` can emboss a layer with. Pick one from `reliefBases`. */
export interface ReliefBasis {
    /** The height function: a 2D position in, a height out. */
    readonly fn: unknown
    /** Its name in the compiled shader. */
    readonly hint: string
}

/**
 * The height fields `noiseRelief` accepts: `stone` (marbled stone) and `wool` (interwoven
 * fibres).
 *
 * @example
 * ```ts
 * gpu: {fragment: noiseRelief(reliefBases.wool)}
 * ```
 * @see noiseRelief
 */
export const reliefBases = {
    /** Marbled stone height field. */
    stone: {fn: noiseKit.stone12, hint: 'stone12'},
    /** Interwoven fibrous fabric height field. */
    wool: {fn: noiseKit.wool12, hint: 'wool12'},
} as const satisfies Record<string, ReliefBasis>

/**
 * Embosses the layer inside with a textured surface: the child is slightly distorted, then
 * lit by a height field so it reads as carved stone or woven cloth.
 *
 * A filter, not a paint. Set it as the definition's fragment with `requiresRTT` and
 * `requiresChild`, and give the definition props named `intensity`, `scale`, `contrast`,
 * `distortion` and `seed`. `basis` is an entry of `reliefBases`.
 *
 * @example
 * ```ts
 * export const Stone = defineShader({
 *   name: 'Stone',
 *   requiresRTT: true,
 *   requiresChild: true,
 *   props: {intensity: {default: 0.5}, scale: {default: 1}, contrast: {default: 0}, distortion: {default: 0.15}, seed: {default: 0}},
 *   gpu: {fragment: noiseRelief(reliefBases.stone)},
 * })
 * ```
 * @tip The library shaders build those five props with `reliefStylizeProps` from the utilities, which also sets their sliders.
 * @see reliefBases
 */
export function noiseRelief(basis: ReliefBasis): (params: GpuFragmentParams) => Expr {
    // Relief filter over child content: sample the child texture through a Perlin-gradient
    // surface distortion, then modulate its brightness by the `basis` height field (an RTT
    // filter — the definition keeps `requiresRTT`/`requiresChild` and the shared
    // `reliefStylizeProps` block).
    return (params) => noiseStylize.applyNoiseReliefExpr(params, basis.fn, basis.hint)
}

/**
 * The `transform` for a `worleyNoise` `mode` prop: maps `'f1'`, `'f2'`, `'f2MinusF1'`,
 * `'f1PlusF2'` or `'f1TimesF2'` to the number the paint reads.
 *
 * @example
 * ```ts
 * mode: {default: 'f1', transform: transformWorleyMode, compileTime: true}
 * ```
 * @see worleyNoise, transformWorleyDistance
 */
export const transformWorleyMode = noisePaints.transformWorleyMode
/**
 * The `transform` for a `worleyNoise` `distance` prop: maps `'euclidean'`, `'manhattan'` or
 * `'chebyshev'` to the number the paint reads.
 *
 * @example
 * ```ts
 * distance: {default: 'euclidean', transform: transformWorleyDistance, compileTime: true}
 * ```
 * @see worleyNoise, transformWorleyMode
 */
export const transformWorleyDistance = noisePaints.transformWorleyDistance
/** @internal The raw cell hash under WorleyNoise's historical name, re-exported for an external importer. */
export const worleyHash = noisePaints.worleyHash
