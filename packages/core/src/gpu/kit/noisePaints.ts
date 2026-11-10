/**
 * Noise-texture paint parts behind std/paint/noise — the mid-level pieces the noise GENERATORS
 * compose: domain framings, tone tails, fractal sums, cellular fills, marble veining. Each part
 * is one nameable visual stage; the std sugar wires them into visible recipes
 * (`rampOver(noiseTone(seededPlane(...)), stops(...))`), so no finished effect lives fused here.
 *
 * What stays atomic (deliberately):
 *  - The raw noise/cellular bases themselves (`kit/noise.ts`, `kit/cellular.ts`).
 *  - {@link ribbonsField} — a runtime-count accumulation loop whose per-strand gradient lookup
 *    cannot leave the loop; its separable tails (tone-pow, back-conversion) already live outside.
 *  - {@link paperSurface} — the grain UV feeds both the displacement and the brightness, so
 *    splitting would re-derive it.
 */
import {tgpu, d, std} from './index'
import {fbmGated} from './fields'
import {aspectOf} from './geom'
import {curl22, curl22z, paper12, mxNoiseFloat2} from './noise'
import {cellHash2, nearest2Cells, cellReduceSelectable, cellRatio, cellEdgeMask, CELL_MODES, CELL_METRICS} from './cellular'
import {gradientStopsInSpace, gradientStopsInSpaceVariants, MAX_COLOR_STOPS} from './colorStops'
import {toneRemap} from './tone'
import {DEG_TO_RAD, PI, TAU} from './constants'

// ─── Domain framings ──────────────────────────────────────────────────────────────────────────

/** Aspect-corrected screen plane: `vec2(uv.x · aspect, uv.y)` (guarded divide). */
export const aspectPlane = tgpu.fn([d.vec2f, d.vec2f], d.vec2f)((uv, viewport) => {
    'use gpu'
    const aspect = aspectOf(viewport)
    return d.vec2f(uv.x * aspect, uv.y)
})

/**
 * The RAW-divide exponential-scale framing (Scratches' historical domain): like
 * `fields.aspectScaledDomain` but with an unguarded `viewport.x / viewport.y`, preserving the
 * shader's exact arithmetic.
 */
export const rawAspectScaledDomain = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32], d.vec2f)(
    (uv, viewport, scale, seed) => {
        'use gpu'
        const aspect = viewport.x / viewport.y
        const aspectUV = d.vec2f(uv.x * aspect, uv.y)
        return aspectUV.mul(std.exp(scale)).add(seed)
    })

/**
 * The rotated fractal framing: aspect-correct, centre on the canvas midpoint, rotate by
 * `angle` (degrees).
 */
export const rotatedCenteredDomain = tgpu.fn([d.vec2f, d.vec2f, d.f32], d.vec2f)(
    (uv, viewport, angle) => {
        'use gpu'
        const aspect = aspectOf(viewport)
        const angleRad = angle * DEG_TO_RAD
        const cosA = std.cos(angleRad)
        const sinA = std.sin(angleRad)
        const au = uv.x * aspect - aspect * 0.5
        const av = uv.y - 0.5
        const ru = au * cosA + av * sinA
        const rv = au * -sinA + av * cosA
        return d.vec2f(ru, rv)
    })

// ─── Curl flow ────────────────────────────────────────────────────────────────────────────────

/**
 * Flow-speed magnitude of the divergence-free 2D curl field, √2-normalized
 * (`length(curl22z(pos, z)) / 1.414`) so it reads as a [0,1]-ish swirling intensity. `z` is the
 * evolution phase (a scaled time). Hash-based → GPU-only.
 */
export const curlMagnitude = tgpu.fn([d.vec2f, d.f32], d.f32)((pos, z) => {
    'use gpu'
    return std.length(curl22z(pos, z)) / 1.414
})

// ─── Fractal Brownian motion ──────────────────────────────────────────────────────────────────

// The gated 8-octave fractal sum. Each octave drifts in its own golden-angle direction so the
// octaves interfere rather than translating together, which is what makes this MORPH rather than
// scroll. The loop is fixed at 8 with a runtime gate: unrolling exactly the active count would
// recompile the pipeline on every slider step.
const fractalOctaveSum = fbmGated(mxNoiseFloat2, {
    maxOctaves: 8,
    drift: 'goldenAngle',
    name: 'fractalOctaveSum',
})

/**
 * The fBm accumulation stage: run the gated golden-angle octave sum over MaterialX simplex at
 * base frequency 2, returning `vec2(accumulated, totalWeight)`. `detail` is the lacunarity and
 * `contrast` the persistence; the seed offsets the two axes differently (`seed`, `seed · 0.7`)
 * so re-rolling re-mixes rather than translating. GPU-only via the hash.
 */
export const fractalSum = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (pos, detail, contrast, animTime, seed, octaveCount) => {
        'use gpu'
        return fractalOctaveSum(pos, 2.0, detail, contrast, animTime, d.vec2f(seed, seed * 0.7), octaveCount)
    })

/**
 * Normalize an fBm sum to the unit range: divide by the accumulated weight, remap the [-1,1]
 * result to [0,1], clamp.
 */
export const fbmNormalize = tgpu.fn([d.vec2f], d.f32)((sum) => {
    'use gpu'
    return std.clamp(sum.x / sum.y * 0.5 + 0.5, 0.0, 1.0)
})

// ─── Worley (cellular) ────────────────────────────────────────────────────────────────────────

/** The shared legacy sin-fract cell hash, under WorleyNoise's historical name. */
export const worleyHash = cellHash2

// The 9-neighbour cellular fold, per octave. `distNum` / `modeNum` are compile-time props, but a
// module-scope `tgpu.fn` cannot be specialised per node, so both arrive as f32 args and every
// metric / reduction is computed and `std.select`-ed. See `kit/cellular.ts` for why that is
// cheaper than recompiling.
const worleyNearest2 = nearest2Cells({
    distance: 'selectableSquared',
    jitter: 'mix',
    name: 'worleyNearest2',
})

/** One Worley octave: the nearest-2 cellular fold + the selectable mode reduction. */
export const worleyEvalOctave = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (oUV, animT, seedOff, jitter, distNum, modeNum) => {
        'use gpu'
        const dd = worleyNearest2(oUV, animT, seedOff, jitter, distNum)
        return cellReduceSelectable(dd.x, dd.y, distNum, modeNum)
    })

// The fractal-octave sum: up to 4 octaves at progressively finer scales, each drifting on its own
// time/seed offset (the `timeSeed` drift shape). `octaves` gates the sum at runtime — octaves at
// or beyond it contribute 0 — so the octave slider does not recompile the pipeline. The
// per-octave parameters (jitter, metric, mode) ride the builder's `extra` vec3.
const worleyOctaveSum = fbmGated(worleyEvalOctave, {
    maxOctaves: 4,
    drift: 'timeSeed',
    name: 'worleyOctaveSum',
})

/**
 * The Worley accumulation stage: the gated cellular octave sum over an aspect-plane position,
 * returning `vec2(accumulated, totalWeight)`. `scale` is the base cell frequency; jitter, the
 * distance metric, and the field reduction ride the per-octave `extra` bundle.
 */
export const worleyCells = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec2f,
)((pos, scale, seed, jitter, lacunarity, persistence, animTime, octaves, distNum, modeNum) => {
    'use gpu'
    return worleyOctaveSum(
        pos, scale, lacunarity, persistence, animTime, seed,
        d.vec3f(jitter, distNum, modeNum), octaves,
    )
})

/** Normalize a Worley sum: guarded weight divide, scaled by the per-mode normalization. */
export const worleyNormalize = tgpu.fn([d.vec2f, d.f32], d.f32)((sum, modeScale) => {
    'use gpu'
    return sum.x / std.max(sum.y, 0.0001) * modeScale
})

/**
 * Worley's tone tail: balance (shift) + contrast (scale about 0.5), remapped to [0,1] — the
 * multiplicative, non-inverted tone variant (see `kit/tone.ts`; `contrast` is a gain here,
 * 1 = identity). Pure float (CPU golden-testable).
 */
export const worleyTone = toneRemap({domain: 'unit', contrastMode: 'multiplicative', invert: false})

/** Enum transform for the Worley `mode` prop (applied by the uniform bridge). */
export const transformWorleyMode = (value: string): number => CELL_MODES[value] ?? 0

/** Enum transform for the Worley `distance` prop (applied by the uniform bridge). */
export const transformWorleyDistance = (value: string): number => CELL_METRICS[value] ?? 0

/** `mode` as its mapped number, robust to a raw string arriving untransformed. */
export function worleyModeNum(raw: unknown): number {
    return typeof raw === 'number' ? raw : transformWorleyMode(raw as string)
}

/** `distance` as its mapped number, robust to a raw string arriving untransformed. */
export function worleyDistNum(raw: unknown): number {
    return typeof raw === 'number' ? raw : transformWorleyDistance(raw as string)
}

/**
 * Per-mode normalization so default contrast=1 lands the result in roughly [0,1]. The distance
 * metric also affects scale: Manhattan distances run larger than Euclidean, Chebyshev smaller.
 */
export function worleyModeScale(modeNum: number, distNum: number): number {
    let modeScale = 1.5  // f1
    if (modeNum === 1) modeScale = 1.0       // f2
    else if (modeNum === 2) modeScale = 2.5  // f2-f1
    else if (modeNum === 3) modeScale = 0.6  // f1+f2
    else if (modeNum === 4) modeScale = 2.0  // f1*f2
    if (distNum === 1) modeScale *= 0.7      // Manhattan distances run larger
    else if (distNum === 2) modeScale *= 1.4 // Chebyshev runs smaller
    return modeScale
}

// ─── Voronoi cells ────────────────────────────────────────────────────────────────────────────

/**
 * The 3×3 nearest-two-features fold, returning `vec2(d1, d2)`. Plain Euclidean distance per tap
 * (a ratio reduction needs real distances, not squared ones) and no jitter control — the cells
 * are always fully random. The drifting cell points hash per-neighbour → GPU-only.
 */
export const voronoiCells = nearest2Cells({distance: 'length', jitter: 'none', name: 'voronoiNearest2'})

/**
 * Fill gradient from the F1/F2 ratio, so cell size does not change the tonal range.
 * `edgeIntensity` inverts into the ratio's exponent: high intensity = low power = the edge
 * color reaches further into each cell.
 */
export const cellFill = tgpu.fn([d.vec2f, d.f32], d.f32)((dd, edgeIntensity) => {
    'use gpu'
    const gradientPower = 4.0 - edgeIntensity * 3.0
    return cellRatio(dd.x, dd.y, gradientPower)
})

/**
 * Boundary-line mask between cells, with `softness` compensated for the cell count so the line
 * width reads constant as `scale` changes. 0 on a boundary, 1 inside a cell.
 */
export const cellBorders = tgpu.fn([d.vec2f, d.f32, d.f32], d.f32)((dd, softness, scale) => {
    'use gpu'
    const scaledEdgeSoftness = softness * scale / 6.0
    return cellEdgeMask(dd.x, dd.y, scaledEdgeSoftness)
})

// ─── Ribbons ──────────────────────────────────────────────────────────────────────────────────

/**
 * The gentle `pow(rgb, 0.85)` tone lift on a final P3-linear rgb. GPU-only.
 *
 * Negative channels clamp to 0 first: WGSL `pow` is undefined below zero (NaN on real GPUs), and
 * a colour mixed in a cylindrical space can land outside the gamut with a negative channel (an
 * OKLCh blue → yellow mix takes the short arc through a cyan more saturated than P3 can show, so
 * its red goes negative). Unclamped, that NaN took over the pixel regardless of the strand's
 * coverage: flat, hard-edged patches running out to the strand's soft edge.
 */
export const toneLift = tgpu.fn([d.vec3f], d.vec3f)((rgb) => {
    'use gpu'
    return std.pow(std.max(rgb, d.vec3f(0.0, 0.0, 0.0)), d.vec3f(0.85, 0.85, 0.85))
})

// Literal copy of colorStops.MAX_COLOR_STOPS: this feeds module-scope d.arrayOf schemas, and
// module-scope reads through the './index' barrel are a load-order hazard — so the literal is
// guarded against drifting from the real constant instead.
const MAX = 8
if (typeof MAX_COLOR_STOPS === 'number' && MAX_COLOR_STOPS !== MAX) {
    throw new Error(`noisePaints: MAX (${MAX}) out of sync with MAX_COLOR_STOPS (${MAX_COLOR_STOPS})`)
}

/**
 * The strands path frame — everything about the start→end path that is loop-invariant across
 * strands: `vec4(progress, perpDist, pinch, edgeFade)`. Progress runs 0→1 start→end; perpDist is
 * the signed distance from the path; pinch/edgeFade converge and fade the strands at the endpoints
 * when pinning is on (`pinFlag` is the transformBoolean scalar: on = > 0). Start/end are
 * aspect-corrected and Y-flipped here (transformPosition stores (x, 1-y) → `1 - .y` = authored).
 */
export const ribbonPathFrame = tgpu.fn([d.vec2f, d.vec2f, d.vec2f, d.vec2f, d.f32], d.vec4f)(
    (uv, viewport, start, end, pinFlag) => {
        'use gpu'
        const aspectRatio = viewport.x / std.max(viewport.y, 1e-6)
        const startPos = d.vec2f(start.x * aspectRatio, 1.0 - start.y)
        const endPos = d.vec2f(end.x * aspectRatio, 1.0 - end.y)
        const aspectUV = d.vec2f(uv.x * aspectRatio, uv.y)

        const direction = d.vec2f(endPos.x - startPos.x, endPos.y - startPos.y)
        const len = std.max(std.length(direction), 1e-6)
        const normalizedDir = d.vec2f(direction.x / len, direction.y / len)
        const perpDir = d.vec2f(-normalizedDir.y, normalizedDir.x)
        const relativePos = d.vec2f(aspectUV.x - startPos.x, aspectUV.y - startPos.y)
        const progress = std.dot(relativePos, normalizedDir) / len
        const perpDist = std.dot(relativePos, perpDir)

        const pinOn = pinFlag > 0.0
        const clampedProg = std.clamp(progress, 0.0, 1.0)
        const pinch = std.select(d.f32(1.0), std.sin(clampedProg * PI), pinOn)
        const fade = std.smoothstep(0.0, 0.06, progress) * (1.0 - std.smoothstep(0.94, 1.0, progress))
        const edgeFade = std.select(d.f32(1.0), fade, pinOn)
        return d.vec4f(progress, perpDist, pinch, edgeFade)
    })

/**
 * The flowing ribbons field (the ribbons genre primitive). A runtime loop over the ACTIVE ribbon count accumulates each strand's
 * WORKING-SPACE gradient color (via {@link gradientStopsInSpace} — a preconverted, alpha-weighted
 * per-segment mix) weighted by coverage. Returns `vec4(workingColorRGB, alpha)`; the caller does
 * the single back-conversion + tone-pow at the compile-time colorSpace mode. Scalars are packed
 * into vec4 params (tgpu.fn 15-arg cap). Loop is hash-free but nested → GPU-only. `packed3`
 * carries pinEdges (transformBoolean: on = > 0), the two animated clocks, and the runtime stop
 * count.
 *
 * ATOMIC (deliberately): the per-strand gradient lookup and coverage weighting live inside a
 * runtime-count reduce — splitting them would hoist a loop-carried dependency. The separable
 * pieces already live outside: the loop-invariant path geometry in {@link ribbonPathFrame}, the
 * tails (tone-pow, back-conversion) in the noun.
 */
export const ribbonsField = tgpu.fn(
    [d.vec2f, d.vec2f, d.vec2f, d.vec2f, d.vec4f, d.vec4f, d.vec4f, d.arrayOf(d.vec4f, MAX), d.arrayOf(d.vec4f, MAX / 4), d.arrayOf(d.vec3f, MAX)],
    d.vec4f,
)((uv, viewport, start, end, packed1, packed2, packed3, colors, positions, converted) => {
    'use gpu'
    const amplitude = packed1.x
    const frequency = packed1.y
    const lineCount = packed1.z
    const lineWidth = packed1.w
    const softnessP = packed2.x
    const spread = packed2.y
    const colorScale = packed2.z
    const colorVariance = packed2.w
    const waveT = packed3.y
    const colorT = packed3.z
    const stopCount = packed3.w

    // Loop-invariant path geometry (progress/perpDist/pinch/edgeFade) — the shared frame part.
    const frame = ribbonPathFrame(uv, viewport, start, end, packed3.x)
    const progress = frame.x
    const perpDist = frame.y
    const pinch = frame.z
    const edgeFade = frame.w

    const thickness = lineWidth * 0.07
    const softness = softnessP * 0.08 + 0.0008
    const denom = std.max(lineCount - 1.0, 1.0)

    let accumColor = d.vec3f(0.0, 0.0, 0.0)
    let accumWeight = d.f32(0.0)

    // Strand-invariant bases, hoisted out of the loop.
    const waveBase = progress * TAU * frequency
    const shimmerBase = progress * 8.0 + waveT * 2.0
    const colorBase = progress * colorScale + colorT

    // Runtime loop over the ACTIVE strand count (round() guards a transient fractional value).
    const n = d.i32(std.round(lineCount))
    for (let idx = 0; idx < n; idx++) {
        const fi = d.f32(idx)
        const norm = fi / denom
        const phase = fi * 0.9
        const strandT = waveT * (norm * 0.6 + 0.7)

        // Coherent, flowing waves (two octaves).
        const wave = std.sin(waveBase + strandT + phase) + std.sin(waveBase * 2.1 - strandT * 0.7 + phase) * 0.5
        const displacement = wave * amplitude * 0.06
        const baseline = (norm - 0.5) * spread
        const centre = (baseline + displacement) * pinch

        const dd = std.abs(perpDist - centre)
        const line = 1.0 - std.smoothstep(thickness, thickness + softness, dd)
        const shimmer = std.sin(shimmerBase + fi) * 0.15 + 0.85

        // Seamless ping-pong of the color coordinate: 1 - |fract(x/2)*2 - 1|.
        const cc = colorBase + norm * colorVariance
        const colorCoord = 1.0 - std.abs(std.fract(cc * 0.5) * 2.0 - 1.0)
        const stepRes = gradientStopsInSpace(colorCoord, colors, positions, converted, stopCount)
        const strandColor = stepRes.conv

        const weight = line * shimmer
        accumColor = accumColor.add(strandColor.mul(weight))
        accumWeight = accumWeight + weight
    }

    // Weighted-average working-space color; coverage → alpha.
    const workingColor = accumColor.div(std.max(accumWeight, 1e-4))
    const alpha = std.clamp(accumWeight * edgeFade, 0.0, 1.0)
    return d.vec4f(workingColor, alpha)
})

// ─── Ribbons: cylindrical colour spaces ──────────────────────────────────────────────────────
// The strand reduce above averages the working colour channel-wise, which is right for linear /
// OKLAB but not for a hue angle. The per-mode variants below average hue as a CIRCULAR mean
// (unit hue vectors weighted by coverage × chroma, so achromatic strands carry no hue and two
// hues meet along the shortest arc); lightness stays a plain weighted average.
//
// Chroma (saturation for HSL/HSV) is scaled by how much the hues AGREE: the length of the summed
// hue vectors over the summed chroma (1 when every strand shares a hue, toward 0 as opposing
// hues cancel). Without it, overlapping strands of roughly opposite hue kept their full average
// chroma while their mean hue, the angle of a near-zero vector, jumped from pixel to pixel:
// vivid, wrong-hued, hard-edged patches clipped out of gamut. With it, opposites meet in grey,
// as they do in OKLAB (for OKLCh the result IS the OKLAB a/b average).

/** Coverage-free hue direction × chroma, for (L, C, h) working colours (OKLCh / LCh). */
const hueDirLch = tgpu.fn([d.vec3f], d.vec2f)((c) => {
    'use gpu'
    return d.vec2f(std.cos(c.z), std.sin(c.z)).mul(c.y)
}).$name('ribbons_hueDirLch')
/** Hue direction × RGB chroma s·(1−|2l−1|), for HSL working colours. */
const hueDirHsl = tgpu.fn([d.vec3f], d.vec2f)((c) => {
    'use gpu'
    return d.vec2f(std.cos(c.x), std.sin(c.x)).mul(c.y * (1.0 - std.abs(c.z * 2.0 - 1.0)))
}).$name('ribbons_hueDirHsl')
/** Hue direction × RGB chroma s·v, for HSV working colours. */
const hueDirHsv = tgpu.fn([d.vec3f], d.vec2f)((c) => {
    'use gpu'
    return d.vec2f(std.cos(c.x), std.sin(c.x)).mul(c.y * c.z)
}).$name('ribbons_hueDirHsv')
/**
 * An averaged (L, C, h) colour with the circular-mean hue in .z and its chroma scaled by
 * `agreement` (|Σ hue vectors| / Σ chroma, 0..1).
 */
const meanHueLch = tgpu.fn([d.vec3f, d.vec2f, d.f32], d.vec3f)((c, dir, agreement) => {
    'use gpu'
    return d.vec3f(c.x, c.y * agreement, std.atan2(dir.y, dir.x))
}).$name('ribbons_meanHueLch')
/**
 * An averaged (h, s, l|v) colour with the circular-mean hue in .x and its saturation scaled by
 * `agreement` (with l / v unchanged, that scales the RGB chroma the hue vectors were weighted by).
 */
const meanHueHs = tgpu.fn([d.vec3f, d.vec2f, d.f32], d.vec3f)((c, dir, agreement) => {
    'use gpu'
    return d.vec3f(std.atan2(dir.y, dir.x), c.y * agreement, c.z)
}).$name('ribbons_meanHueHs')

type RibbonStops = typeof gradientStopsInSpace
type RibbonHueDir = typeof hueDirLch
type RibbonMeanHue = typeof meanHueLch

/** {@link ribbonsField} with a pluggable stops loop and a circular-mean hue (see section note). */
function makeRibbonsFieldHue(stopsFn: RibbonStops, hueDir: RibbonHueDir, withMeanHue: RibbonMeanHue) {
    return tgpu.fn(
        [d.vec2f, d.vec2f, d.vec2f, d.vec2f, d.vec4f, d.vec4f, d.vec4f, d.arrayOf(d.vec4f, MAX), d.arrayOf(d.vec4f, MAX / 4), d.arrayOf(d.vec3f, MAX)],
        d.vec4f,
    )((uv, viewport, start, end, packed1, packed2, packed3, colors, positions, converted) => {
        'use gpu'
        const amplitude = packed1.x
        const frequency = packed1.y
        const lineCount = packed1.z
        const lineWidth = packed1.w
        const softnessP = packed2.x
        const spread = packed2.y
        const colorScale = packed2.z
        const colorVariance = packed2.w
        const waveT = packed3.y
        const colorT = packed3.z
        const stopCount = packed3.w

        // Loop-invariant path geometry (progress/perpDist/pinch/edgeFade) — the shared frame part.
        const frame = ribbonPathFrame(uv, viewport, start, end, packed3.x)
        const progress = frame.x
        const perpDist = frame.y
        const pinch = frame.z
        const edgeFade = frame.w

        const thickness = lineWidth * 0.07
        const softness = softnessP * 0.08 + 0.0008
        const denom = std.max(lineCount - 1.0, 1.0)

        let accumColor = d.vec3f(0.0, 0.0, 0.0)
        let accumWeight = d.f32(0.0)
        let accumHue = d.vec2f(0.0, 0.0)
        let accumChroma = d.f32(0.0)

        // Strand-invariant bases, hoisted out of the loop.
        const waveBase = progress * TAU * frequency
        const shimmerBase = progress * 8.0 + waveT * 2.0
        const colorBase = progress * colorScale + colorT

        // Runtime loop over the ACTIVE strand count (round() guards a transient fractional value).
        const n = d.i32(std.round(lineCount))
        for (let idx = 0; idx < n; idx++) {
            const fi = d.f32(idx)
            const norm = fi / denom
            const phase = fi * 0.9
            const strandT = waveT * (norm * 0.6 + 0.7)

            // Coherent, flowing waves (two octaves).
            const wave = std.sin(waveBase + strandT + phase) + std.sin(waveBase * 2.1 - strandT * 0.7 + phase) * 0.5
            const displacement = wave * amplitude * 0.06
            const baseline = (norm - 0.5) * spread
            const centre = (baseline + displacement) * pinch

            const dd = std.abs(perpDist - centre)
            const line = 1.0 - std.smoothstep(thickness, thickness + softness, dd)
            const shimmer = std.sin(shimmerBase + fi) * 0.15 + 0.85

            // Seamless ping-pong of the color coordinate: 1 - |fract(x/2)*2 - 1|.
            const cc = colorBase + norm * colorVariance
            const colorCoord = 1.0 - std.abs(std.fract(cc * 0.5) * 2.0 - 1.0)
            const stepRes = stopsFn(colorCoord, colors, positions, converted, stopCount)
            const strandColor = stepRes.conv

            const weight = line * shimmer
            accumColor = accumColor.add(strandColor.mul(weight))
            accumWeight = accumWeight + weight
            const dir = hueDir(strandColor)
            accumHue = accumHue.add(dir.mul(weight))
            accumChroma = accumChroma + std.length(dir) * weight
        }

        // Weighted-average working-space color; coverage → alpha. Chroma follows how much the
        // strands' hues agree (see section note).
        const agreement = std.min(std.length(accumHue) / std.max(accumChroma, 1e-6), 1.0)
        const workingColor = withMeanHue(accumColor.div(std.max(accumWeight, 1e-4)), accumHue, agreement)
        const alpha = std.clamp(accumWeight * edgeFade, 0.0, 1.0)
        return d.vec4f(workingColor, alpha)
    })
}

/**
 * {@link ribbonsField} for a compile-time colorSpace mode (0 linear · 1 OKLCh · 2 OKLAB · 3 HSL ·
 * 4 HSV · 5 LCH). Same signature; the cylindrical modes average strand hue on the circle. Built
 * lazily and cached: colorStops ↔ kit/index form an import cycle, so its variants table is not
 * initialised yet when this module first evaluates.
 */
const ribbonsFieldCache = new Map<number, typeof ribbonsField>()
export function ribbonsFieldFor(mode: number): typeof ribbonsField {
    if (mode !== 1 && mode !== 3 && mode !== 4 && mode !== 5) return ribbonsField
    let fn = ribbonsFieldCache.get(mode)
    if (!fn) {
        const stopsFn = gradientStopsInSpaceVariants[mode]
        fn = mode === 1 ? makeRibbonsFieldHue(stopsFn, hueDirLch, meanHueLch).$name('ribbonsFieldOklch')
            : mode === 5 ? makeRibbonsFieldHue(stopsFn, hueDirLch, meanHueLch).$name('ribbonsFieldLch')
            : mode === 3 ? makeRibbonsFieldHue(stopsFn, hueDirHsl, meanHueHs).$name('ribbonsFieldHsl')
            : makeRibbonsFieldHue(stopsFn, hueDirHsv, meanHueHs).$name('ribbonsFieldHsv')
        ribbonsFieldCache.set(mode, fn)
    }
    return fn
}

// ─── Paper ────────────────────────────────────────────────────────────────────────────────────

/**
 * Paper surface (displaced sample UV + grain brightness). Packs the two per-pixel derivations
 * Paper needs into one `vec3f` so the grain UV is computed once: `.xy` is the child sample UV
 * shifted along the divergence-free curl flow (fiber-direction micro-roughness), `.z` is the
 * fibrous-grain brightness. `curl22`/`paper12` are hash-based → GPU-only (the u32-hash rule); the
 * math: grain freq `grainScale*6`, disp `curl22 * displacement * 0.0025`, brightness centred on
 * paper12's ~0.7 mid.
 *
 * ATOMIC (deliberately): the shared grain UV feeds BOTH outputs — splitting into a displacement
 * part and a brightness part would re-derive it.
 */
export const paperSurface = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], d.vec3f)(
    (uv, aspect, grainScale, seed, displacement, roughness) => {
        'use gpu'
        const aspectUV = d.vec2f(uv.x * aspect, uv.y)
        const grainFreq = grainScale * 6.0
        const grainUV = aspectUV.mul(grainFreq).add(seed)
        // Fiber-direction micro-displacement: shift the sampled child UVs along the curl flow.
        const disp = curl22(grainUV).mul(displacement).mul(0.0025)
        const displacedUV = uv.add(disp)
        // Fibrous grain (~[0.4, 1]); centring on its ~0.7 mid turns it into signed brightness.
        const grain = paper12(grainUV, d.i32(6))
        const brightness = std.clamp(1.0 + (grain - 0.7) * (roughness * 1.2), 0.0, 1.5)
        return d.vec3f(displacedUV.x, displacedUV.y, brightness)
    })
