import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, uniformOf} from "@coreroot/std"
import {surfaceOf, segmentFrame} from "@coreroot/std/frames"
import {waves, type WaveTerm} from "@coreroot/std/paint/noise"
import {stops} from "@coreroot/std/paint/fields"
import {crispFill, crispStroke} from "@coreroot/std/mask"
import {circle, roundedRect} from "@coreroot/std/shape"
import {abs, add, clamp, div, floor, fract, gaussBell, local, max, mix, mul, neg, smoothstep, sqrt, step, sub, vec4} from "@coreroot/std/math"
import {transformColor, transformColorSpace, transformPosition, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

// ── The simulated signal (this data IS the look) ──────────────────────────────────────────
// A stack of traveling sines standing in for an audio spectrum: broad slow swells carry the
// fast detail so the whole thing breathes instead of fizzing. Weights sum to 1, so the raw
// signal sits in [-1, 1] and `amplitude` is the only gain.
const SPECTRUM: WaveTerm[] = [
    {waves: [{x: 5.1, t: 1.3}], weight: 0.28},
    {waves: [{x: 9.7, t: -1.9}, {x: 1.7, t: 0.6}], weight: 0.24},
    {waves: [{x: 17.3, t: 2.7}], weight: 0.18},
    {waves: [{x: 31.9, t: -3.4}, {x: 3.3, t: -0.9}], weight: 0.14},
    {waves: [{x: 67.7, t: 4.1}], weight: 0.09},
    {waves: [{x: 131.3, t: -5.3}, {x: 7.9, t: 1.1}], weight: 0.07},
]
// Loudness falls off toward the edges like a real spectrum (mids carry the energy); the edge
// bands keep this much of the centre's level so the ends never go dead.
const EDGE_LEVEL = 0.4
const ENVELOPE_FALLOFF = 4
// Level bias: the signed signal lands on `LEVEL_FLOOR ± LEVEL_SWING` before the amplitude gain.
const LEVEL_FLOOR = 0.45
const LEVEL_SWING = 0.6
// Finite-difference step (uv-x) for the line style's slope-corrected stroke distance.
const SLOPE_EPS = 0.002

type Style = 'bars' | 'wave' | 'line' | 'dots'
/** Which side of the start→end baseline the waveform grows on. */
type Align = 'mirrored' | 'top' | 'bottom'

const STYLE_MODES: Record<Style, number> = {bars: 0, wave: 1, line: 2, dots: 3}
const ALIGN_MODES: Record<Align, number> = {mirrored: 0, top: 1, bottom: 2}
// Compile-time enum readers — robust to a raw string (a preset loaded before the prop
// transform ran) as well as the bridge-mapped number.
const styleOf = (raw: unknown): Style => {
    if (typeof raw === 'number') return (Object.keys(STYLE_MODES) as Style[])[raw] ?? 'bars'
    return raw === 'wave' || raw === 'line' || raw === 'dots' ? raw : 'bars'
}
const alignOf = (raw: unknown): Align => {
    if (typeof raw === 'number') return (Object.keys(ALIGN_MODES) as Align[])[raw] ?? 'mirrored'
    return raw === 'top' || raw === 'bottom' ? raw : 'mirrored'
}

export interface ComponentProps {
    style: Style
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    colorSpace: string
    from: Parameters<typeof transformPosition>[0]
    to: Parameters<typeof transformPosition>[0]
    amplitude: number
    frequency: number
    height: number
    align: Align
    count: number
    barWidth: number
    rounding: number
    dotSize: number
    lineWidth: number
    softness: number
    speed: number
    seed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Waveform",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Audio-visualizer waveform — equalizer bars, a filled wave, an oscilloscope line, or a dot matrix — driven by a simulated signal whose amplitude you can map to your own audio level",
    acceptsUVContext: true,
    // Per-node accumulated time driven by `speed`; the signal reads it (offset by `seed`).
    animatedTime: { speed: 'speed' },
    props: {
        // First so the editor shows it above the props whose visibility depends on it.
        style: {
            default: 'bars',
            transform: (value: Style) => STYLE_MODES[value] ?? 0,
            compileTime: true,
            description: "Visualizer style: equalizer bars, a filled wave envelope, an oscilloscope line, or a dot matrix",
            ui: {
                type: 'select',
                options: [
                    {label: 'Bars', value: 'bars'},
                    {label: 'Wave', value: 'wave'},
                    {label: 'Line', value: 'line'},
                    {label: 'Dots', value: 'dots'},
                ],
                label: 'Style',
                group: 'Style'
            }
        },
        colorA: {
            default: "#3b5bff",
            transform: transformColor,
            description: "Color at the start",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ff3bd4",
            transform: transformColor,
            description: "Color at the end",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Multi-stop colors along the start→end baseline. Default null → the two-color
        // colorA/colorB ramp; when set with 2+ stops they take over. See utilities/colorStops.ts.
        stops: colorStopsPropConfig(),
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for the start→end gradient',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        },
        from: {
            default: { x: 0, y: 0.5 },
            transform: transformPosition,
            description: "Where the waveform starts (Color A end)",
            ui: { type: 'position', label: 'Start', group: 'Position', units: ['%', 'px'] }
        },
        to: {
            default: { x: 1, y: 0.5 },
            transform: transformPosition,
            description: "Where the waveform ends (Color B end)",
            ui: { type: 'position', label: 'End', group: 'Position', units: ['%', 'px'] }
        },
        amplitude: {
            default: 1,
            description: "Overall signal level (0 = silence). Map this to your own audio level to make the waveform react to real sound.",
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Amplitude', group: 'Signal' }
        },
        frequency: {
            default: 1,
            description: "How busy the signal is across the width (higher = more peaks)",
            ui: { type: ['range', 'map'], min: 0.2, max: 3, step: 0.01, label: 'Frequency', group: 'Signal' }
        },
        height: {
            default: 0.6,
            description: "Maximum vertical extent of the waveform as a fraction of the canvas height",
            ui: { type: ['range', 'map'], min: 0.05, max: 1, step: 0.01, label: 'Height', group: 'Effect' }
        },
        align: {
            default: 'mirrored',
            transform: (value: Align) => ALIGN_MODES[value] ?? 0,
            compileTime: true,
            description: "Which side of the start→end baseline the waveform grows on: mirrored around it, above it (top), or below it (bottom)",
            ui: {
                type: 'select',
                options: [
                    {label: 'Mirrored', value: 'mirrored'},
                    {label: 'Top', value: 'top'},
                    {label: 'Bottom', value: 'bottom'},
                ],
                label: 'Align',
                group: 'Effect'
            }
        },
        count: {
            default: 32,
            description: "Number of bars (or dot columns) across the width",
            ui: { type: ['range', 'map'], min: 4, max: 128, step: 1, label: 'Count', group: 'Effect', condition: {style: ['bars', 'dots']} }
        },
        barWidth: {
            default: 0.6,
            description: "Bar thickness as a fraction of its column (1 = bars touch)",
            ui: { type: ['range', 'map'], min: 0.1, max: 1, step: 0.01, label: 'Bar Width', group: 'Effect', condition: {style: 'bars'} }
        },
        rounding: {
            default: 1,
            description: "Rounds the bar ends (0 = square, 1 = fully rounded caps)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Rounding', group: 'Effect', condition: {style: 'bars'} }
        },
        dotSize: {
            default: 0.7,
            description: "Dot diameter as a fraction of its cell",
            ui: { type: ['range', 'map'], min: 0.2, max: 1, step: 0.01, label: 'Dot Size', group: 'Effect', condition: {style: 'dots'} }
        },
        lineWidth: {
            default: 0.006,
            description: "Stroke thickness of the line. A value of one (1) matches the canvas height.",
            ui: { type: ['range', 'map'], min: 0.001, max: 0.05, step: 0.001, label: 'Line Width', group: 'Effect', dimensional: 'canvas-height', condition: {style: 'line'} }
        },
        softness: {
            default: 0,
            description: "Feathers the wave's edge into a glow",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect', condition: {style: 'wave'} }
        },
        speed: {
            default: 1,
            description: "Animation speed of the simulated signal",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 1,
            description: "Adjusts the starting state, useful for variation",
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation' }
        },
    },

    // The recipe: one simulated signal, rendered four ways. The signal is the SPECTRUM stack
    // (signed) shaped by a loudness envelope and the amplitude gain; each style turns the
    // level at a column or pixel into a distance field and fills/strokes it crisply.
    paint: (params) => {
        const style = styleOf(params.propValues.style)
        const align = alignOf(params.propValues.align)
        const {viewport} = surfaceOf(params)
        const u = (name: keyof ComponentProps) => uniformOf(p(name), params)

        // The waveform lives in the frame of the start→end segment: `along` runs the baseline,
        // `across` is the signed distance from it (positive below, in aspect-corrected uv units).
        const {along, across, length} = segmentFrame({from: p('from'), to: p('to')})(params, surfaceOf(params))
        // Position along the baseline as 0→1 — the signal's coordinate and the gradient's.
        const x = local(div(along, length), 'wfX')
        // One device pixel in uv-y units — the anti-aliasing footprint.
        const px = local(div(1, viewport.member('y')), 'wfPx')
        const t = local(animatedTime(params, 'seed'), 'wfTime')
        const amplitude = u('amplitude')
        const frequency = u('frequency')
        const height = u('height')

        // The signal at a position along the baseline: the raw signed spectrum, the loudness
        // envelope, the signed curve (oscilloscope) and the [0, 1] level (everything else).
        const rawAt = (xs: Expr) => waves({x: mul(xs, frequency), t}, SPECTRUM)
        const envelopeAt = (xs: Expr) => mix(EDGE_LEVEL, 1, gaussBell(sub(xs, 0.5), ENVELOPE_FALLOFF))
        const signalAt = (xs: Expr) => clamp(mul(mul(rawAt(xs), envelopeAt(xs)), amplitude), -1, 1)
        const levelAt = (xs: Expr) => clamp(mul(mul(add(LEVEL_FLOOR, mul(rawAt(xs), LEVEL_SWING)), envelopeAt(xs)), amplitude), 0, 1)

        // Alignment: `dy` is the distance from the baseline on the growing side (signed when
        // mirrored), `extentOf` turns a level into the distance `dy` is tested against.
        const mirrored = align === 'mirrored'
        const dy = local(align === 'top' ? neg(across) : across, 'wfDy')
        const rise = mirrored ? abs(dy) : dy
        const extentOf = (level: Expr) => mul(mul(level, height), mirrored ? 0.5 : 1)
        // Signed distance to the band between the baseline and `extent` on the growing side
        // (both sides when mirrored) — one-sided fills must stop at the baseline too.
        const bandDistance = (extent: Expr) => mirrored ? sub(rise, extent) : max(sub(rise, extent), neg(rise))

        // Column frame (bars/dots): square cells along the baseline — the column's centre
        // position and the pixel's offset from it.
        const count = u('count')
        const cell = local(div(length, count), 'wfCell')
        const xCells = local(mul(x, count), 'wfXCells')
        const columnX = local(div(add(floor(xCells), 0.5), count), 'wfColX')
        const acrossColumn = local(mul(sub(fract(xCells), 0.5), cell), 'wfColDx')

        let coverage: Expr
        if (style === 'bars') {
            // A rounded rect per column; never shorter than it is wide so a silent bar is a dot.
            const halfW = local(mul(mul(u('barWidth'), cell), 0.5), 'wfHalfW')
            const extent = local(max(extentOf(levelAt(columnX)), mirrored ? halfW : mul(halfW, 2)), 'wfExtent')
            const halfH = mirrored ? extent : mul(extent, 0.5)
            const barY = mirrored ? dy : sub(dy, halfH)
            coverage = crispFill({
                distance: roundedRect(acrossColumn, barY, halfW, halfH, mul(u('rounding'), halfW)),
                footprint: px,
            })
        } else if (style === 'dots') {
            // Cells stacked from the baseline; a dot lights as the level reaches its row.
            const rows = local(div(rise, cell), 'wfRows')
            const rowCentre = local(mul(add(floor(rows), 0.5), cell), 'wfRowY')
            const extent = local(extentOf(levelAt(columnX)), 'wfExtent')
            // Rows below the baseline (negative, one-sided alignments only) never light.
            const lit = mul(
                sub(1, smoothstep(sub(extent, mul(cell, 0.5)), add(extent, mul(cell, 0.5)), rowCentre)),
                mirrored ? 1 : step(0, rise),
            )
            const radius = mul(mul(u('dotSize'), cell), 0.5)
            const dot = crispFill({
                distance: circle(acrossColumn, mul(sub(fract(rows), 0.5), cell), radius),
                footprint: px,
            })
            coverage = mul(dot, lit)
        } else if (style === 'line') {
            // Oscilloscope: stroke the curve (signed around the baseline when mirrored, the level
            // on the growing side otherwise), distance corrected by the local slope so steep
            // runs stay as thick as flat ones.
            const curveAt = (xs: Expr) => mirrored ? extentOf(signalAt(xs)) : extentOf(levelAt(xs))
            const curve = local(curveAt(x), 'wfCurve')
            const slope = local(div(
                sub(curveAt(add(x, SLOPE_EPS)), curveAt(sub(x, SLOPE_EPS))),
                mul(2 * SLOPE_EPS, length),
            ), 'wfSlope')
            const distance = div(sub(dy, curve), sqrt(add(1, mul(slope, slope))))
            coverage = crispStroke({distance, width: u('lineWidth'), footprint: px})
        } else {
            // Wave: the filled envelope of the level, feathered by softness.
            const extent = local(extentOf(levelAt(x)), 'wfExtent')
            coverage = crispFill({
                distance: bandDistance(extent),
                footprint: add(px, mul(u('softness'), 0.2)),
            })
        }

        // Clip to the start→end span (crisp ends), then show the start→end gradient in the
        // compile-time color space through the coverage.
        const inSpan = crispFill({distance: max(neg(along), sub(along, length)), footprint: px})
        const color = local(stops(p('colorSpace'))(x, params), 'wfColor')
        return vec4(color.member('rgb'), mul(mul(color.member('a'), coverage), inSpan))
    }
})

export default componentDefinition
