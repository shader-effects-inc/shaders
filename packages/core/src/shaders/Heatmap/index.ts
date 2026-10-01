import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {call, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface, hashNoise, valueNoise} from "@coreroot/std/paint/materials"
import {directionFrame} from "@coreroot/std/frames"
import {
    abs, add, clamp, div, dot, exp, float, fract, local, max, min, mix, mul, pow, smoothstep,
    sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {colorMixing, colorStops as kitColorStops} from "@coreroot/gpu/kit"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

// Adapted from paper-design/shaders "heatmap" (MIT), rebuilt as an SDF shape effect. The paper
// original needs a pre-processed image (contour / inner-blur / outer-blur baked into RGB
// channels); the signed-distance field gives all three analytically — |field| hugs the contour,
// exp(−|f|) inside is the inner blur, exp(−f) outside is the outer glow — so it works with any
// 2D shape, custom SVG, or raymarched 3D silhouette, no preprocessing. The heat model follows the
// paper: an edge-bright inner fill crossed by three staggered travelling "cold front" shadows,
// contour heat, an outer glow swept by a travelling band, sensor noise, and the heat → gradient
// color walk (our colorStops system, defaulting to a thermal "iron" ramp).

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// The default thermal-camera "iron" ramp (cold → hot).
const DEFAULT_STOPS: ColorStop[] = [
    {color: '#02010f', position: 0},
    {color: '#2a0a8a', position: 0.2},
    {color: '#a41c9b', position: 0.45},
    {color: '#e8632b', position: 0.7},
    {color: '#f9e25f', position: 0.9},
    {color: '#ffffff', position: 1},
]

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    colorSpace: string
    innerGlow: number
    outerGlow: number
    contour: number
    angle: number
    speed: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

// ── The heat model as algebra ──────────────────────────────────────────────────────────────────

// One travelling "cold front" (the paper shadowShape, generalized from its hand-drawn apple-logo
// animation): a soft band sweeping across the shape along the heat direction, its centre-line
// wobbled by value noise so the front reads organic. `a` = coordinate along the travel direction,
// `b` = lateral, `tk` = this copy's loop phase [0,1); `phase` decorrelates the three copies.
const heatmapShadow = (a: Expr, b: Expr, tk: Expr, phase: number): Expr => {
    const posS = mix(-0.6, 0.6, tk)
    // phase stays a runtime f32 factor (as in the tuned original) — folding phase·13.7 in JS
    // would land one ulp off the f32 product for phase 3.
    const wob = sub(valueNoise(add(mul(b, 4), mul(float(phase), 13.7)), add(mul(tk, 3), mul(float(phase), 7.1))), 0.5)
    const aEff = local(add(sub(a, posS), mul(wob, 0.12)), 'aEff')
    const band = exp(div(mul(mul(aEff, aEff), -1), 0.0256))
    // Ease the front in and out of its loop so it never pops at the wrap point.
    const gate = mul(smoothstep(0, 0.15, tk), sub(1, smoothstep(0.85, 1, tk)))
    return clamp(mul(band, gate), 0, 1)
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Heatmap",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Thermal-camera heat flowing through any 2D, SVG, or 3D shape. Powered by Paper Shaders.",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine, centre-tap only: only `.r` (field/chord) is read, so patternMode
    // 'none', no grad sampler and no neighbour taps. The heat model (the paper's main() over the
    // signed field) is algebra over that tap: an edge-bright inner fill crossed by three staggered
    // travelling cold fronts, contour heat, an outer glow swept by a travelling band, the
    // field-frame falloff, sensor dither, and the heat → gradient color walk.
    ...shapedSurface({
        stencil: 'centre',
        surface: (frame, params) => {
            const {uniforms, propValues} = params
            const t = animatedTime(params)

            const suv = local(frame.sdfUV!, 'suv')
            const scl = max(uniforms.scale, 0.001)
            const f = local(div(frame.surf0!.member('r'), scl), 'f')
            const inside = local(sub(1, smoothstep(-0.003, 0.003, f)), 'inside')

            // Direction frame for the travelling animation (paper's rotated animationUV).
            const travel = directionFrame(uniforms.angle, 'travel')
            const rel = local(sub(suv.member('xy'), vec2(0.5, 0.5)), 'rel')
            const a = local(dot(rel, travel.tangent), 'a')
            const b = local(mul(dot(rel, travel.perp), -1), 'b')

            // Paper t: three staggered copies of the loop.
            const tBase = local(sub(mul(0.1, t), 0.3), 'tBase')

            // Inner fill (paper: .8 + .8·innerBlur): brightest hugging the edge, dimming into the
            // deep interior — exp(−|f|) is the analytic inner blur.
            const innerBlur = exp(div(mul(max(mul(f, -1), 0), -1), 0.12))
            let inner: Expr = add(0.8, mul(0.8, innerBlur))

            // Cold fronts sweep the heat away (paper: inner = mix(inner, 0, shadow) × 3 copies).
            inner = local(mix(inner, 0, heatmapShadow(a, b, fract(tBase), 1)), 'inner1')
            inner = local(mix(inner, 0, heatmapShadow(a, b, fract(add(tBase, 1 / 3)), 2)), 'inner2')
            inner = local(mix(inner, 0, heatmapShadow(a, b, fract(add(tBase, 2 / 3)), 3)), 'inner3')

            inner = mul(inner, mul(2, uniforms.innerGlow))

            // Contour heat hugging the boundary (paper: + u_contour·2·contour).
            inner = add(inner, mul(mul(uniforms.contour, 2), exp(div(mul(abs(f), -1), 0.03))))
            inner = mul(min(inner, 1), inside)
            inner = local(pow(max(inner, 0), 1.2), 'inner')

            // Outer glow with the travelling band (paper: t·3 loop, mask .5 + band).
            const outerBlur = exp(div(mul(max(f, 0), -1), add(0.05, mul(0.15, uniforms.outerGlow))))
            const t3 = fract(sub(mul(tBase, 3), 0.1))
            const yy = local(fract(sub(mul(a, 1.2), t3)), 'yy')
            const animatedMask = add(0.5, mul(smoothstep(0.3, 0.65, yy), sub(1, smoothstep(0.65, 1, yy))))
            const outer = local(mul(mul(mul(mul(mul(mul(0.9, pow(outerBlur, 0.8)), animatedMask), 5), uniforms.outerGlow), uniforms.outerGlow), sub(1, inside)), 'outer')

            let heat: Expr = clamp(add(inner, outer), 0, 1)

            // Soft field-frame falloff (the paper's getImgFrame). The flat custom-SVG sampler
            // CLAMPS its texture beyond the 0..1 field, so the distance plateaus down those
            // columns instead of growing — without this fade the travelling band tiles forever
            // over the constant glow (and an SVG touching the field border smears a hot line to
            // infinity). Analytic/3D fields keep growing, so for them this only trims glow that
            // exp(−f) had already killed.
            const frameFade = local(mul(mul(mul(smoothstep(-0.06, 0.1, suv.member('x')), sub(1, smoothstep(0.9, 1.06, suv.member('x')))), smoothstep(-0.06, 0.1, suv.member('y'))), sub(1, smoothstep(0.9, 1.06, suv.member('y')))), 'frameFade')
            heat = mul(heat, frameFade)

            // Minimal fixed dither — breaks banding on the long soft ramps (invisible otherwise).
            heat = add(heat, mul(0.005, sub(hashNoise(mul(suv.member('x'), 991.3), mul(suv.member('y'), 787.7)), 0.5)))

            // ramp position (color) + shape coverage (alpha): the shape interior is fully OPAQUE —
            // the cold-front shadows only darken the COLOR (via `inner`), never the silhouette
            // (`inside`) — plus the outer glow halo, faded by the field frame. Transparency comes
            // only from a stop's own alpha.
            heat = local(clamp(heat, 0, 1), 'heat')
            const coverage = mul(clamp(add(inside, outer), 0, 1), frameFade)

            // Heat → color: the multi-stop ramp when stops are active (the default thermal palette),
            // else the two-color colorA/colorB path — the LinearGradient recipe.
            const colorSpaceMode = (propValues.colorSpace as number) ?? 0
            const stopCount = (propValues.stopCount as number) ?? 0
            let ramp: Expr
            if (stopCount > 1) {
                ramp = kitColorStops.mixColorStopsRuntime(heat, {
                    colorsArray: uniforms.colorsArray,
                    positionsArray: uniforms.positionsArray,
                    convertedColorsArray: uniforms.convertedColorsArray,
                    stopCount: uniforms.stopCount,
                }, colorSpaceMode)
            } else {
                const variant = colorMixing.mixColorsVariants[colorSpaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
                ramp = call(variant, 'mixColors', [uniforms.colorA, uniforms.colorB, heat])
            }

            // SOLID over a TRANSPARENT background: alpha = shape COVERAGE × the stop's own alpha
            // (`ramp.w`). Returns STRAIGHT rgba.
            const rampL = local(ramp, 'ramp')
            return vec4(clamp(rampL.member('xyz'), vec3(0, 0, 0), vec3(1, 1, 1)), mul(rampL.member('w'), coverage))
        },
    }),
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the heatmap shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the heatmap shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the heatmap shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        colorA: {
            default: '#02010f',
            transform: transformColor,
            description: 'Cold color (used when no gradient stops are set)',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#f9e25f',
            transform: transformColor,
            description: 'Hot color (used when no gradient stops are set)',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Multi-stop heat ramp, defaulting to the thermal "iron" palette.
        stops: {...colorStopsPropConfig(), default: DEFAULT_STOPS},
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for the heat ramp interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        },
        innerGlow: {
            default: 0.4,
            description: 'Heat filling the inside of the shape',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Inner Glow', group: 'Heat' }
        },
        outerGlow: {
            default: 0.2,
            description: 'Heat radiating beyond the silhouette',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Outer Glow', group: 'Heat' }
        },
        contour: {
            default: 0.5,
            description: 'Heat concentrated along the shape boundary',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Contour', group: 'Heat' }
        },
        angle: {
            default: 90,
            description: 'Direction the heat waves travel, in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Angle', group: 'Heat' }
        },
        speed: {
            default: 1,
            description: 'Speed of the flowing heat. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Heat' }
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: { type: 'shape', label: 'Shape', group: 'Shape' }
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type'
        }
    },

})

export default componentDefinition
