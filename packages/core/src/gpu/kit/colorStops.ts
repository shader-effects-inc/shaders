/**
 * Multi-stop color gradient primitive.
 *
 * Split by stage:
 *   - Plain TS (dependency-free): MAX_COLOR_STOPS, ColorStop, colorStopsTransform,
 *     sortStops, resolveStops, colorStopsPropConfig.
 *   - Pack math (plain TS): `packStops` / `packConvertedStops` are pure array functions,
 *     decoupled from the uniform store, that do the write/sort/pack logic and return flat number
 *     arrays for `createColorStopsUniformDefinition` to write into its `ArrayFieldHandle`s. The
 *     CSS-color → P3-rgba conversion is INJECTED as `toRGBA` — it belongs to transformations.ts,
 *     not here.
 *   - GPU unroll (`mixColorStops`): a BUILDER that unrolls the sorted stops into a sequential
 *     per-segment mix. `stopCount` is the CPU unroll bound (a change recompiles — already in the
 *     pipeline hash via `compileTimeWhen`). It reuses colorMixing's `mixPreconvertedInSpace` (the
 *     per-segment alpha-weighted mix) + `convertMixSpaceToP3` (the single closing back-conversion).
 *
 * Dependency: colorMixing.ts (sibling kit file). Consumer of the pack + builder: the uniform
 * store and the LinearGradient/Radial/Conic/Diamond shaders.
 */
import {tgpu, d, std} from './index'
import {
    MixStep,
    mixPreconvertedInSpace,
    mixPreconvertedInSpaceVariants,
    convertMixSpaceToP3,
    convertP3ToMixSpaceCPU,
    // Per-mode back-conversion fns (working space → P3-linear) for the runtime-loop builder.
    sRGBToP3,
    oklabToRgb,
    oklchToOklab,
    hslToRgb,
    hsvToRgb,
    labToRgb,
    lchToLab,
} from './colorMixing'
// KitExpr factories for the COMPOSITION-time (Expr-level) `mixColorStopsRuntime` builder below.
// `call` / `vec4` are hoisted `function` declarations in the composer, so this composer↔kit import
// resolves safely (same pattern kit/edges.ts uses; the composer never imports this file at init).
import {call, vec4, asLocal} from '../composer'
import type {Expr} from '../contract'
import type {PropConfig} from '../../types'

export const MAX_COLOR_STOPS = 8

export interface ColorStop {
    /** CSS color string (hex, rgb, named, etc.) */
    color: string
    /** Position along the gradient axis, 0–1 */
    position: number
}

/**
 * Sentinel marker used as the `stops` prop's `transform`. Never invoked to produce a node —
 * the color-stops branches in the uniform store intercept the prop by this identity before
 * the generic transform path runs (mirrors the old `updateTransformMap` identity pattern).
 */
export const colorStopsTransform = (value: unknown): unknown => value

/**
 * Stable sort stops by position (CPU). Keeps `{color, position}` pairs together — sorting
 * positions independently of colors would scramble the gradient.
 */
export function sortStops(stops: ColorStop[]): ColorStop[] {
    return stops
        .map((stop, index) => ({stop, index}))
        .sort((a, b) => (a.stop.position - b.stop.position) || (a.index - b.index))
        .map(({stop}) => stop)
}

/**
 * The effective sorted stop list for a gradient's props: explicit `stops` (sorted) when
 * present and non-empty, otherwise the legacy two-color [colorA@0, colorB@1] pair. Shared by
 * core and the editor so they always agree.
 */
export function resolveStops(props: {stops?: ColorStop[] | null; colorA?: string; colorB?: string}): ColorStop[] {
    if (Array.isArray(props.stops) && props.stops.length > 0) {
        return sortStops(props.stops)
    }
    return [
        {color: props.colorA ?? '#1aff00', position: 0},
        {color: props.colorB ?? '#0000ff', position: 1},
    ]
}

/** A stop color parser: CSS string → P3-linear rgba. Supplied by transformations.ts. */
export type ToRGBA = (css: string) => [number, number, number, number]

/** Packed stop arrays in the fixed-MAX uniform-array layout. */
export interface PackedStops {
    /** rgba per stop, flat: MAX_COLOR_STOPS * 4 floats. */
    colors: number[]
    /** position per stop: MAX_COLOR_STOPS floats. */
    positions: number[]
    /** number of active stops (the CPU unroll bound). */
    stopCount: number
}

/**
 * Pack (already-resolvable) stops into fixed-MAX flat arrays and compute `stopCount`. Inactive
 * (null/empty) → stopCount 0 (the shader takes its legacy two-color path). Sort/pack decoupled
 * from the uniform arrays. `toRGBA` converts each CSS color.
 */
export function packStops(stops: ColorStop[] | null | undefined, toRGBA: ToRGBA): PackedStops {
    const colors = new Array<number>(MAX_COLOR_STOPS * 4).fill(0)
    const positions = new Array<number>(MAX_COLOR_STOPS).fill(0)
    const active = Array.isArray(stops) && stops.length > 0
    const sorted = active ? sortStops(stops as ColorStop[]) : []
    const n = Math.min(sorted.length, MAX_COLOR_STOPS)
    for (let i = 0; i < n; i++) {
        const c = toRGBA(sorted[i].color)
        colors[i * 4] = c[0]
        colors[i * 4 + 1] = c[1]
        colors[i * 4 + 2] = c[2]
        colors[i * 4 + 3] = c[3]
        positions[i] = sorted[i].position
    }
    return {colors, positions, stopCount: n}
}

/**
 * Refill the CPU-preconverted working-space array from packed stop colors, for `colorSpaceMode`
 * (uses colorMixing's `convertP3ToMixSpaceCPU`). Returns MAX_COLOR_STOPS * 3 floats; only the
 * first `stopCount` entries are meaningful.
 */
export function packConvertedStops(colors: number[], stopCount: number, colorSpaceMode: number): number[] {
    const out = new Array<number>(MAX_COLOR_STOPS * 3).fill(0)
    for (let i = 0; i < stopCount; i++) {
        const c = convertP3ToMixSpaceCPU(colors[i * 4], colors[i * 4 + 1], colors[i * 4 + 2], colorSpaceMode)
        out[i * 3] = c[0]
        out[i * 3 + 1] = c[1]
        out[i * 3 + 2] = c[2]
    }
    return out
}

// ─── GPU unroll helpers (durable, resolve-tested tgpu.fns) ───────────────────

/** Initialize the running mix state from the first stop's working-space color + alpha. */
const initStep = tgpu.fn([d.vec3f, d.f32], MixStep)((conv, alpha) => {
    'use gpu'
    return MixStep({conv, alpha})
})

/**
 * Advance the running mix across one gradient segment [posA, posB]. `seg` clamps to 0/1 for
 * segments before/after the active one so the running color passes through exactly; only the
 * active segment interpolates (per-segment linstep). The mix is colorMixing's alpha-weighted
 * `mixPreconvertedInSpace` (transparent stops contribute no RGB).
 */
type InSpaceStep = typeof mixPreconvertedInSpace

const gradientSegmentStep = tgpu.fn(
    [MixStep, d.vec3f, d.f32, d.f32, d.f32, d.f32],
    MixStep,
)((running, convI, alphaI, tClamped, posA, posB) => {
    'use gpu'
    const seg = std.clamp((tClamped - posA) / std.max(posB - posA, 1e-6), 0.0, 1.0)
    return mixPreconvertedInSpace(running.conv, running.alpha, convI, alphaI, seg)
})

/** Same step with a pluggable in-space mix (the hue-aware cylindrical variants). */
function makeGradientSegmentStep(step: InSpaceStep) {
    return tgpu.fn(
        [MixStep, d.vec3f, d.f32, d.f32, d.f32, d.f32],
        MixStep,
    )((running, convI, alphaI, tClamped, posA, posB) => {
        'use gpu'
        const seg = std.clamp((tClamped - posA) / std.max(posB - posA, 1e-6), 0.0, 1.0)
        return step(running.conv, running.alpha, convI, alphaI, seg)
    })
}

/** Segment steps keyed by colorSpaceMode — cylindrical modes take the hue-aware in-space step. */
const gradientSegmentStepVariants = {
    0: gradientSegmentStep,
    1: makeGradientSegmentStep(mixPreconvertedInSpaceVariants[1]).$name('gradientSegmentStepOklch'),
    3: makeGradientSegmentStep(mixPreconvertedInSpaceVariants[3]).$name('gradientSegmentStepHsl'),
    4: makeGradientSegmentStep(mixPreconvertedInSpaceVariants[4]).$name('gradientSegmentStepHsv'),
    5: makeGradientSegmentStep(mixPreconvertedInSpaceVariants[5]).$name('gradientSegmentStepLch'),
} as const

/** Pick the segment step for a compile-time colorSpace mode (2 OKLAB shares the plain step). */
function gradientSegmentStepFor(colorSpaceMode: number) {
    const key = colorSpaceMode === 2 ? 0 : colorSpaceMode
    return gradientSegmentStepVariants[key as keyof typeof gradientSegmentStepVariants] ?? gradientSegmentStepVariants[0]
}

/** Read the running working-space color out of the mix state (for builder-level back-convert). */
const stepConv = tgpu.fn([MixStep], d.vec3f)((s) => {
    'use gpu'
    return s.conv
})

/** Read the accumulated alpha out of the mix state. */
const stepAlpha = tgpu.fn([MixStep], d.f32)((s) => {
    'use gpu'
    return s.alpha
})

/** `vec4(rgb, alpha)`. */
const packRGBA = tgpu.fn([d.vec3f, d.f32], d.vec4f)((rgb, a) => {
    'use gpu'
    return d.vec4f(rgb, a)
})

/**
 * Working-space color of stop `i`: either the CPU-preconverted color (non-linear modes — zero
 * forward conversions on the GPU) or the raw P3 rgb (mode 0, where the mix space IS P3). The
 * caller supplies both accessors; `mixColorStops` selects between them by mode, exactly as the
 * original's `useConverted` branch did.
 */
export interface StopAccessors {
    /** Working-space (or, mode 0, raw P3) color of stop `i`. */
    convAt: (i: number) => d.v3f
    /** Alpha of stop `i` (scalar GPU expr — typed `number`, matching colorMixing's convention). */
    alphaAt: (i: number) => number
    /** Position of stop `i` along the gradient axis. */
    positionAt: (i: number) => number
}

/**
 * Build-time multi-stop color evaluation at scalar `t` (0–1). When there are >1 active stops,
 * unrolls the sorted stops into a sequential per-segment mix chain (in the working color space)
 * and returns the resulting P3-linear vec4; otherwise returns `null` so the caller runs its
 * original two-color path (`mixColorStops(...) ?? mixColors(...)`).
 *
 * BUILDER (composer-consumed): `stopCount` and `colorSpaceMode` are CPU compile-time values, so
 * this JS-unrolls and only ever calls tgpu.fns (consistent with colorMixing's builder-dispatchers
 * — never raw infix/construction at builder level). The chain runs IN the working space: each
 * stop is forward-converted once (via the `convAt` accessor), segments blend with the
 * alpha-weighted step, and a single back-conversion closes the chain.
 */
export function mixColorStops(
    stopCount: number,
    colorSpaceMode: number,
    t: number,
    stops: StopAccessors,
): d.v4f | null {
    if (stopCount <= 1) return null
    const tClamped = std.clamp(t, 0, 1)
    const segmentStep = gradientSegmentStepFor(colorSpaceMode)
    let step = initStep(stops.convAt(0), stops.alphaAt(0))
    for (let i = 1; i < stopCount; i++) {
        step = segmentStep(
            step,
            stops.convAt(i),
            stops.alphaAt(i),
            tClamped,
            stops.positionAt(i - 1),
            stops.positionAt(i),
        )
    }
    const rgb = convertMixSpaceToP3(stepConv(step), colorSpaceMode)
    return packRGBA(rgb, stepAlpha(step))
}

// ─── Runtime-loop multi-stop (Expr-level builder) ────────────────────────────
//
// The build-time `mixColorStops` unroll above is UNUSABLE from a shader body — its StopAccessors
// are arrow fns TGSL rejects, and it returns a d.v4f not an Expr. This section is the runtime-loop
// pattern EVERY multi-stop gradient (LinearGradient/Radial/Conic/Diamond/…) shares:
//   - `gradientStopsInSpace` — a WGSL runtime loop (bound by the `stopCount` uniform) that
//     accumulates the per-segment alpha-weighted mix IN the working color space, over the packed
//     `convertedColorsArray` (CPU-preconverted) + `colorsArray[i].w` (alpha) + `positionsArray`.
//     Mode-independent: the single closing back-conversion is applied by the builder.
//   - `mixColorStopsRuntime` — the composition-time builder: calls the loop, back-converts the
//     accumulated working-space color to P3-linear per the compile-time colorSpace mode, and packs
//     `vec4(rgb, alpha)`.

/**
 * Working-space multi-stop accumulation at scalar `t` (runtime loop, bound by `stopCount`). See
 * the section note. `converted` holds each stop's color ALREADY in the working space (mode 0 =
 * raw P3); `colors[i].w` is the alpha; `positions` packs the 8 stop positions 4-per-vec4. Returns
 * the accumulated working-space color + alpha (`MixStep`) — the builder does the closing
 * back-conversion. Exported for CPU golden-value tests (DualFn).
 */
export const gradientStopsInSpace = tgpu.fn(
    [d.f32, d.arrayOf(d.vec4f, MAX_COLOR_STOPS), d.arrayOf(d.vec4f, MAX_COLOR_STOPS / 4), d.arrayOf(d.vec3f, MAX_COLOR_STOPS), d.f32],
    MixStep,
)((t, colors, positions, converted, stopCount) => {
    'use gpu'
    const clampedT = std.clamp(t, 0.0, 1.0)
    const n = d.i32(stopCount)
    const c0 = converted[0]
    let conv = d.vec3f(c0.x, c0.y, c0.z)
    let alpha = colors[0].w
    for (let i = 1; i < n; i++) {
        const va = positions[(i - 1) >> 2]
        const vb = positions[i >> 2]
        const posA = va[(i - 1) & 3]
        const posB = vb[i & 3]
        // Segments before/after the active one clamp to 0/1 so the running color passes through
        // exactly; only the active segment interpolates (per-segment linstep).
        const seg = std.clamp((clampedT - posA) / std.max(posB - posA, 1e-6), 0.0, 1.0)
        const step = mixPreconvertedInSpace(conv, alpha, converted[i], colors[i].w, seg)
        conv = d.vec3f(step.conv.x, step.conv.y, step.conv.z)
        alpha = step.alpha
    }
    return MixStep({conv, alpha})
})

/** Same loop with a pluggable in-space step (the hue-aware cylindrical variants below). */
function makeGradientStopsInSpace(step: InSpaceStep) {
    return tgpu.fn(
        [d.f32, d.arrayOf(d.vec4f, MAX_COLOR_STOPS), d.arrayOf(d.vec4f, MAX_COLOR_STOPS / 4), d.arrayOf(d.vec3f, MAX_COLOR_STOPS), d.f32],
        MixStep,
    )((t, colors, positions, converted, stopCount) => {
        'use gpu'
        const clampedT = std.clamp(t, 0.0, 1.0)
        const n = d.i32(stopCount)
        const c0 = converted[0]
        let conv = d.vec3f(c0.x, c0.y, c0.z)
        let alpha = colors[0].w
        for (let i = 1; i < n; i++) {
            const va = positions[(i - 1) >> 2]
            const vb = positions[i >> 2]
            const posA = va[(i - 1) & 3]
            const posB = vb[i & 3]
            // Segments before/after the active one clamp to 0/1 so the running color passes through
            // exactly; only the active segment interpolates (per-segment linstep).
            const seg = std.clamp((clampedT - posA) / std.max(posB - posA, 1e-6), 0.0, 1.0)
            const res = step(conv, alpha, converted[i], colors[i].w, seg)
            conv = d.vec3f(res.conv.x, res.conv.y, res.conv.z)
            alpha = res.alpha
        }
        return MixStep({conv, alpha})
    })
}

/**
 * {@link gradientStopsInSpace} per compile-time colorSpace mode: the cylindrical modes (OKLCh,
 * HSL, HSV, LCh) chain the hue-aware in-space step (shortest-arc hue, achromatic stops borrow
 * the neighbour's hue). Linear/OKLAB share the plain loop.
 */
const gradientStopsInSpaceOklch = makeGradientStopsInSpace(mixPreconvertedInSpaceVariants[1]).$name('gradientStopsInSpaceOklch')
const gradientStopsInSpaceHsl = makeGradientStopsInSpace(mixPreconvertedInSpaceVariants[3]).$name('gradientStopsInSpaceHsl')
const gradientStopsInSpaceHsv = makeGradientStopsInSpace(mixPreconvertedInSpaceVariants[4]).$name('gradientStopsInSpaceHsv')
const gradientStopsInSpaceLch = makeGradientStopsInSpace(mixPreconvertedInSpaceVariants[5]).$name('gradientStopsInSpaceLch')
export const gradientStopsInSpaceVariants = {
    0: gradientStopsInSpace,
    1: gradientStopsInSpaceOklch,
    2: gradientStopsInSpace,
    3: gradientStopsInSpaceHsl,
    4: gradientStopsInSpaceHsv,
    5: gradientStopsInSpaceLch,
} as const

/**
 * Builder-level back-conversion: working space → P3-linear, one `call` chain per colorSpace mode.
 * Mirrors colorMixing.convertMixSpaceToP3 at the Expr level (that JS dispatcher emits DualFn
 * VALUES, not Exprs). Mode 0 (linear) is already P3.
 */
export function backConvertToP3(conv: Expr, colorSpaceMode: number): Expr {
    switch (colorSpaceMode) {
        case 1: return call(sRGBToP3, 'sRGBToP3', [call(oklabToRgb, 'oklabToRgb', [call(oklchToOklab, 'oklchToOklab', [conv])])])
        case 2: return call(sRGBToP3, 'sRGBToP3', [call(oklabToRgb, 'oklabToRgb', [conv])])
        case 3: return call(sRGBToP3, 'sRGBToP3', [call(hslToRgb, 'hslToRgb', [conv])])
        case 4: return call(sRGBToP3, 'sRGBToP3', [call(hsvToRgb, 'hsvToRgb', [conv])])
        case 5: return call(sRGBToP3, 'sRGBToP3', [call(labToRgb, 'labToRgb', [call(lchToLab, 'lchToLab', [conv])])])
        default: return conv
    }
}

/** The four packed color-stop uniform accessors a runtime multi-stop gradient consumes. */
export interface ColorStopRuntimeUniforms {
    /** `d.arrayOf(vec4f, 8)` — rgba per stop (rgb = xyz, alpha = w). */
    colorsArray: Expr
    /** `d.arrayOf(vec4f, 2)` — 8 positions packed 4-per-vec4. */
    positionsArray: Expr
    /** `d.arrayOf(vec3f, 8)` — CPU-preconverted working-space rgb per stop (at the active mode). */
    convertedColorsArray: Expr
    /** `d.f32` — the active stop count (the runtime loop bound). */
    stopCount: Expr
}

/**
 * Composition-time multi-stop color for a gradient parameter `t` (0–1). Runs the working-space
 * accumulation loop over the packed stop uniforms, then back-converts to P3-linear at the
 * compile-time `colorSpaceMode`, returning `vec4(rgb, alpha)`. The caller gates this behind
 * `stopCount > 1` and runs the literal two-color `mixColors` path otherwise. `t` may be
 * pre-clamped by the caller; the loop clamps internally regardless.
 */
export function mixColorStopsRuntime(t: Expr, uniforms: ColorStopRuntimeUniforms, colorSpaceMode: number): Expr {
    // Hoisted: `.conv` and `.alpha` are both read — without the local the stop loop runs twice.
    const stopsFn = gradientStopsInSpaceVariants[colorSpaceMode as keyof typeof gradientStopsInSpaceVariants] ?? gradientStopsInSpace
    const acc = asLocal(call(stopsFn, 'gradientStopsInSpace', [
        t,
        uniforms.colorsArray,
        uniforms.positionsArray,
        uniforms.convertedColorsArray,
        uniforms.stopCount,
    ]), 'stopMix')
    const rgb = backConvertToP3(acc.member('conv'), colorSpaceMode)
    return vec4(rgb, acc.member('alpha'))
}

/**
 * The standard `stops` PropConfig shared by every multi-stop gradient. Default `null` → the
 * shader runs its literal two-color (colorA/colorB) path, so legacy presets and npm consumers
 * stay byte-identical. Recompiles only when the unrolled structure changes: presence toggling
 * (null ↔ multi) or the active stop count changing; same-count edits update in place.
 */
export function colorStopsPropConfig(): PropConfig<ColorStop[] | null> {
    return {
        default: null,
        transform: colorStopsTransform,
        compileTimeWhen: (previousValue: ColorStop[] | null, newValue: ColorStop[] | null) => {
            const effective = (v: ColorStop[] | null) => {
                const n = Array.isArray(v) ? v.length : 0
                return n > 1 ? n : 0
            }
            return effective(previousValue) !== effective(newValue)
        },
        description: 'Multi-stop gradient colors (overrides Color A / Color B when set)',
        ui: {type: 'gradient-stops', label: 'Colors', group: 'Colors'},
    }
}
