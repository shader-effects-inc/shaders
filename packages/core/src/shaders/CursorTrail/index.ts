import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {call, vec4, ZERO} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {createPointerVelocityTracker} from "@coreroot/gpu/kit/host/pointer"
import {agents, colorMixing, colorStops as kitColorStops, toHalfFloat} from "@coreroot/gpu/kit"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    stops: ColorStop[] | null,
    radius: number,
    length: number,
    shrink: number,
    softness: number,
    colorSpace: string
}

// ── Architecture: analytic capsule-chain (the Shatter data-texture pattern) ─────────────────────
//
// The trail IS a polyline: the cursor path recorded as up to MAX_POINTS points (x, y, age, link),
// maintained on the CPU in onBeforeRender and packed into a MAX_POINTS×1 rgba16float data texture
// each frame. The fragment walks the chain once per pixel and evaluates a round-cone (capsule with
// age-shrunk per-end radii) SDF per segment — so edges are analytic and resolution-independent,
// and where the path crosses itself the NEWEST covering segment wins (real paint layering), not a
// blurry shared-field accumulation. This replaced the old 128×128 stamp/decay compute grid, whose
// additive intensity field produced soft blob edges and muddled ages at crossings.
const MAX_POINTS = 64
const HALF_ONE = toHalfFloat(1)

// CPU path recorder: committed anchors + one live head. The head tracks the cursor every frame
// (so the trail tip never lags); once it travels ANCHOR_SPACING from the last anchor it is
// committed and a new head starts. Front-culling only (ages are monotonic oldest-first), so texel
// i's predecessor is always texel i−1 and the GPU pairing stays valid.
//
// `width` is the pen-pressure analogue, driven by pointer SPEED at the moment the point was laid
// down: a slow nudge paints a thin stroke, a flick paints at full radius. sqrt easing gives the
// sublinear response a pressure pen has; the floor keeps tiny movements visible, just small.
interface TrailPoint {x: number; y: number; age: number; link: number; width: number}
const WIDTH_MIN = 0.2 // width scale at (near-)zero speed
const SPEED_FULL = 1.5 // smoothed pointer speed (UV/s) that reaches full radius

// Sub-capsule count per joint quadratic. 3 keeps the Bézier's sagitta error sub-pixel at these
// anchor spacings while costing only ALU (no extra texture loads).
const BEZIER_SUBDIVISIONS = 3

/** Per-pixel scan of the trail as a SMOOTH CURVE: the stroke-polyline capsule-chain scan
 *  (midpoint quadratic-Bézier spine, min-union SDF, depth-weighted age blend) over this
 *  trail's point budget → vec2(age, coverage). */
export const cursorTrailScan = agents.makeStrokePolylineScan({
    maxPoints: MAX_POINTS,
    subdivisions: BEZIER_SUBDIVISIONS,
    name: 'cursorTrailScan',
})

// ── HOST PART: the polyline recorder ────────────────────────────────────────────────────────────
// The CPU half of the simulation: track the pointer, maintain the anchor/head polyline, age +
// front-cull it, and pack it into the MAX_POINTS×1 data texture the scan walks. Runs in
// onBeforeRender every frame; returns the registered points texture. This is deliberately a HOST
// recipe part (no compute pass at all): 64 points × ~60 Hz is nothing for the CPU, and keeping the
// recorder here gives the stroke logic (anchor spacing, Chaikin relax, pen-pressure width) plain
// JS semantics instead of a kernel round-trip.
function polylineRecorder(params: GpuFragmentParams) {
    const {createDataTexture, registerMediaTexture, onBeforeRender, onCleanup, onResize, getCpuValue, dimensions} = params

    const pts: TrailPoint[] = []
    const texData = new Uint16Array(MAX_POINTS * 4)
    for (let i = 0; i < MAX_POINTS; i++) texData[i * 4 + 2] = HALF_ONE // all slots dead
    const dataTex = createDataTexture({width: MAX_POINTS, height: 1, format: 'rgba16float', data: texData, label: 'cursortrail-points'})
    onCleanup(() => dataTex.destroy())
    const pointsKit = registerMediaTexture(() => dataTex.texture)

    // Teleport guard DISABLED on purpose: a trail's whole job is drawing fast strokes, and any
    // jump (including canvas re-entry) should paint a straight stroke — the pre-refactor
    // intended look (reviewed visually, 2026-08-15).
    const tracker = createPointerVelocityTracker({minDrag: 0.001, teleportGuard: Number.POSITIVE_INFINITY})
    let curW = Math.max(1, Math.round(dimensions.width))
    let curH = Math.max(1, Math.round(dimensions.height))
    onResize(({width, height}) => {
        curW = Math.max(1, Math.round(width))
        curH = Math.max(1, Math.round(height))
    })
    let headDist = 0
    let wasActive = false

    onBeforeRender((fp: unknown) => {
        const {pointer, deltaTime} = fp as {pointer: {x: number; y: number}; deltaTime: number}
        // Two dt bounds: the tracker (velocity → pen width) gets frame-scale dt with a spike cap
        // that still preserves genuine low-frame-rate intervals (30Hz = 33ms must NOT be capped,
        // or velocity reads high); ageing gets a higher bound so trail persistence tracks
        // wall-clock time on slow frames, while a single pathological spike (tab return, debug
        // pause) can't erase the whole trail in one tick.
        const rawDt = Math.max(0, deltaTime ?? 0)
        const dt = Math.min(rawDt, 0.1)
        const length = (getCpuValue('length') as number) ?? 0.5
        const drawRadius = ((getCpuValue('radius') as number) ?? 0.5) * 0.1
        const aspect = curW / curH
        const move = tracker.update(pointer, dt)

        // Age + front-cull (ages are oldest-first, so dead points only ever leave the front —
        // which keeps every survivor adjacent to its true predecessor in the packed texture).
        const ageRate = Math.min(rawDt, 0.25) / Math.max(0.1, length)
        for (const p of pts) p.age = Math.min(1, p.age + ageRate)
        while (pts.length > 0 && pts[0].age >= 1) pts.shift()

        if (move.dragDist > 0.001) {
            const stepDist = Math.sqrt(move.dx * aspect * (move.dx * aspect) + move.dy * move.dy)
            // Pen-pressure width from smoothed pointer speed (sqrt easing, floored so tiny
            // movements stay visible — just thin instead of a full-radius blob).
            const widthNow = Math.min(1, Math.max(WIDTH_MIN, Math.sqrt(move.smoothSpeed / SPEED_FULL)))
            if (pts.length === 0) {
                pts.push({x: move.prevX, y: move.prevY, age: 0, link: 0, width: widthNow}) // stroke-start anchor
                pts.push({x: move.x, y: move.y, age: 0, link: 1, width: widthNow}) // live head
                headDist = stepDist
            } else {
                const head = pts[pts.length - 1]
                head.x = move.x
                head.y = move.y
                head.age = 0
                head.width = widthNow
                headDist += stepDist
                // Commit the head as an anchor once it has travelled far enough; the fresh head
                // starts at the same spot (a zero-length segment until the next move).
                if (headDist >= Math.max(0.006, drawRadius * 0.5)) {
                    // Chaikin-style relax of the joint one back (its neighbours both exist now):
                    // averages out perpendicular pointer jitter so the spine reads as a smooth
                    // curve instead of a chain of frame-quantized chords.
                    const n = pts.length
                    if (n >= 3 && pts[n - 2].link === 1) {
                        pts[n - 2].x = 0.25 * pts[n - 3].x + 0.5 * pts[n - 2].x + 0.25 * pts[n - 1].x
                        pts[n - 2].y = 0.25 * pts[n - 3].y + 0.5 * pts[n - 2].y + 0.25 * pts[n - 1].y
                    }
                    pts.push({x: move.x, y: move.y, age: 0, link: 1, width: widthNow})
                    headDist = 0
                }
            }
            while (pts.length > MAX_POINTS - 1) pts.shift() // reserve one slot for the head duplicate
        }

        // Upload only while there is something alive (plus one final all-dead frame to clear).
        // The head is duplicated as one extra texel so the midpoint-Bézier chain's final
        // quadratic collapses onto the head — the curve reaches the very tip of the stroke.
        const active = pts.length > 0
        if (active || wasActive) {
            const count = pts.length > 0 ? pts.length + 1 : 0
            for (let i = 0; i < MAX_POINTS; i++) {
                const base = i * 4
                if (i < count) {
                    const p = pts[Math.min(i, pts.length - 1)]
                    texData[base] = toHalfFloat(p.x)
                    texData[base + 1] = toHalfFloat(p.y)
                    texData[base + 2] = toHalfFloat(p.age)
                    // w channel = signed width: |w| is the speed width scale, sign is the link
                    // flag (negative = stroke start). The head duplicate is always linked.
                    texData[base + 3] = toHalfFloat((i === pts.length || p.link === 1) ? p.width : -p.width)
                } else {
                    texData[base] = 0
                    texData[base + 1] = 0
                    texData[base + 2] = HALF_ONE
                    texData[base + 3] = 0
                }
            }
            dataTex.write(texData)
        }
        wasActive = active
    })

    return pointsKit
}

// ── PART: age → color ramp ─────────────────────────────────────────────────────────────────────
// Multi-stop when >1 active stops (working-space accumulate + back-convert), else the literal
// two-color path: `mixColorStops(...) ?? mixColors(colorA, colorB, ...)`.
function ageColorRamp(params: GpuFragmentParams, trailAge: Expr): Expr {
    const {propValues, uniforms} = params
    const colorSpaceMode = (propValues.colorSpace as number) ?? 0
    const stopCount = (propValues.stopCount as number) ?? 0
    if (stopCount > 1) {
        return kitColorStops.mixColorStopsRuntime(
            trailAge,
            {
                colorsArray: uniforms.colorsArray,
                positionsArray: uniforms.positionsArray,
                convertedColorsArray: uniforms.convertedColorsArray,
                stopCount: uniforms.stopCount,
            },
            colorSpaceMode,
        )
    }
    const cm = colorMixing
    const variant = cm.mixColorsVariants[colorSpaceMode as keyof typeof cm.mixColorsVariants] ?? cm.mixColorsLinear
    return call(variant, 'mixColors', [uniforms.colorA, uniforms.colorB, trailAge])
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "CursorTrail",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Animated trail effect that tracks cursor movement",
    usesPointer: true,
    props: {
        colorA: {
            default: "#00aaff",
            transform: transformColor,
            description: "Color of fresh trails",
            ui: { type: 'color', label: 'Start Color', group: 'Colors' }
        },
        colorB: {
            default: "#ff00aa",
            transform: transformColor,
            description: "Color trails transition to as they fade",
            ui: { type: 'color', label: 'End Color', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
        radius: {
            default: 0.5,
            description: 'Base radius of the trail stroke',
            ui: { type: 'range', min: 0.5, max: 2, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        length: {
            default: 0.5,
            description: 'How long the trail persists (in seconds)',
            ui: { type: 'range', min: 0.1, max: 2, step: 0.1, label: 'Trail Length', group: 'Animation' }
        },
        shrink: {
            default: 1,
            description: 'How much the stroke tapers as it fades out (0 = no taper, 1 = full taper)',
            ui: { type: 'range', min: 0, max: 1, step: 0.1, label: 'Shrink Amount', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: 'Edge softness of the trail (0 = crisp paint edge, 1 = very soft)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        }
    },

    // The pipeline: HOST polyline recorder → GPU capsule-chain scan (`cursorTrailScan`) →
    // age-driven color ramp × coverage. No compute pass — the trail state is a 64×1 data texture
    // the CPU packs each frame and the fragment walks analytically.
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {ctx, uniforms, gpu} = params
        if (!gpu?.root) return ZERO // GPU-free resolve / no device → transparent.

        const pointsKit = polylineRecorder(params)
        const scan = call(cursorTrailScan, 'cursorTrailScan', [
            pointsKit.accessor(), ctx.uv, ctx.aspect, uniforms.radius, uniforms.shrink, uniforms.softness,
            ctx.viewportSize.member('y'),
        ])
        const trailAge = scan.member('x')
        const coverage = scan.member('y')

        const trailColor = ageColorRamp(params, trailAge)
        return vec4(trailColor.member('rgb'), trailColor.member('a').mul(coverage))
    }}
})

export default componentDefinition
