import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, schema, resolveScalar, uniformOf} from "@coreroot/std"
import {rawSurfaceOf, centredFrame} from "@coreroot/std/frames"
import {stops as stopsPalette} from "@coreroot/std/paint/fields"
import {waves, type WaveTerm} from "@coreroot/std/paint/noise"
import {domeNormal, shine} from "@coreroot/std/paint/light"
import {softDisc} from "@coreroot/std/mask"
import {driveUnitDirection} from "@coreroot/std/signal"
import {add, local, mul, smoothstep, vec3, vec4} from "@coreroot/std/math"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"
import type {BoundingBoxOrigin} from "@coreroot/types"

export type {ColorStop}

// ── The Blob look (this data IS the look) ─────────────────────────────────────────────────
// Four wobble layers deform the disc's radius; two drift layers pick the fill color.
const WOBBLE: {schedule: WaveTerm[]; amount: number}[] = [
    {amount: 0.15, schedule: [
        {waves: [{x: 3.2, t: 0.8}, {y: 2.8, t: 0.6}]},
        {waves: [{x: 4.8, y: -3.6, t: -0.4}]},
    ]},
    {amount: 0.12, schedule: [{waves: [{x: 5.6, t: -0.5}, {y: 4.4, t: 0.7}]}]},
    {amount: 0.1, schedule: [
        {waves: [{x: 7.2, y: 6.4, t: 0.3}]},
        {waves: [{x: 2.4, t: -0.9}]},
    ]},
    {amount: 0.04, schedule: [{waves: [{x: 8.8, t: 0.2}, {y: 7.6, t: -0.8}]}]},
]
const FILL_DRIFT_A: WaveTerm[] = [
    {waves: [{x: 1, t: 0.4}, {y: 1, t: 0.3}]},
    {waves: [{x: 0.7, y: 0.8, t: 0.2}]},
]
const FILL_DRIFT_B: WaveTerm[] = [
    {waves: [{x: 1.3, t: -0.5}, {y: 1.1, t: 0.6}]},
    {waves: [{x: 0.5, y: -0.6, t: 0.3}]},
]

export interface ComponentProps {
    origin: BoundingBoxOrigin
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    stops: ColorStop[] | null,
    size: number,
    deformation: number,
    softness: number,
    highlightIntensity: number,
    highlightX: number,
    highlightY: number,
    highlightZ: number,
    highlightColor: Parameters<typeof transformColor>[0],
    speed: number,
    seed: number,
    colorSpace: string,
    center: Parameters<typeof transformPosition>[0]
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Blob",
    role: 'generator',
    category: "Textures",
    description: "Organic animated blob with 3D lighting and gradients",
    acceptsUVContext: true,
    // Per-node animated time with a seed offset: the renderer advances `_animTime` by
    // `deltaTime * speed`; the blob noun adds the seed uniform on the GPU.
    animatedTime: { speed: 'speed' },
    // Normalized light direction, precomputed on the CPU each frame (avoids a per-fragment
    // normalize()). Declared as extraFields and written each frame by the blob noun's
    // onBeforeRender driver.
    extraFields: {
        normLx: { schema: schema.f32, initial: 0 },
        normLy: { schema: schema.f32, initial: 0 },
        normLz: { schema: schema.f32, initial: 1 }
    },
    boundingBoxDeclaration: {
        propBindings: {
            x: { prop: 'center', as: 'position-x' },
            y: { prop: 'center', as: 'position-y' }
        },

        // Bounding box matches the core size prop only.
        // Deformation and softness are shown separately via the softness ring handle,
        // so they don't contribute to the box size.
        computeBounds(props, cw, ch) {
            const size = (props.size as number) ?? 0.5
            // Plain-uv baseline only — px-unit DimensionalValue axes fall back to 0.5 (the
            // Text/Trapezoid convention), so consumers deriving the visual-box shift as
            // `centerPx − plainCenter` get a pure offset instead of NaN.
            const cx   = typeof props.center?.x === 'number' ? props.center.x : 0.5
            const cy   = typeof props.center?.y === 'number' ? props.center.y : 0.5
            const diameter = size * 2 * ch
            return { centerXPx: cx * cw, centerYPx: cy * ch, widthPx: diameter, heightPx: diameter, rotationDeg: 0 }
        },

        writeBounds(bounds, currentProps, cw, ch) {
            const newSize = Math.max(0, bounds.widthPx / (2 * ch))
            return {
                center: { ...(currentProps.center ?? { x: 0.5, y: 0.5 }), x: bounds.centerXPx / cw, y: bounds.centerYPx / ch },
                size: newSize
            }
        },

        // Softness ring: Blob's edgeWidth = softness × 0.3 in aspect-corrected UV space,
        // which equals softness × 0.3 × canvasHeight pixels (same conversion as canvas-height).
        softnessBinding: {
            prop: 'softness',
            toPx: (v, _, ch) => v * 0.3 * ch,
            fromPx: (px, _, ch) => Math.max(0, Math.min(1, px / (0.3 * ch)))
        }
    },
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: {
                type: 'origin',
                label: 'Origin',
                group: 'Position'
            }
        },
        colorA: {
            default: "#ff6b35",
            transform: transformColor,
            description: "Primary color of the blob",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#e91e63",
            transform: transformColor,
            description: "Secondary color of the blob",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
        size: {
            default: 0.5,
            description: 'Size of the blob',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Size', group: 'Effect' }
        },
        deformation: {
            default: 0.5,
            description: 'How organic and blobby the shape is (0 = circle, 1 = very blobby)',
            ui: { type: 'range', min: 0, max: 1, step: 0.1, label: 'Deformation', group: 'Effect' }
        },
        softness: {
            default: 0.5,
            description: 'Softness of the blob edges (combines edge width and transition curve)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Softness', group: 'Effect' }
        },
        highlightIntensity: {
            default: 0.5,
            description: 'Intensity of specular highlight effect',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Highlight Intensity', group: 'Highlight' }
        },
        highlightX: {
            default: 0.3,
            description: 'Light direction X component',
            ui: { type: 'range', min: -1, max: 1, step: 0.1, label: 'Highlight X', group: 'Highlight' }
        },
        highlightY: {
            default: -0.3,
            description: 'Light direction Y component',
            ui: { type: 'range', min: -1, max: 1, step: 0.1, label: 'Highlight Y', group: 'Highlight' }
        },
        highlightZ: {
            default: 0.4,
            description: 'Light direction Z component',
            ui: { type: 'range', min: -1, max: 1, step: 0.1, label: 'Highlight Z', group: 'Highlight' }
        },
        highlightColor: {
            default: "#ffe11a",
            transform: transformColor,
            description: "Color of the specular highlight",
            ui: { type: 'color', label: 'Highlight Color', group: 'Highlight' }
        },
        speed: {
            default: 0.5,
            description: 'Animation speed',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 1,
            description: 'Adjusts the starting state, useful for variation',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: "The center point of the blob",
            ui: {
                type: 'position',
                label: 'Center Position',
                group: 'Position',
                units: ['%', 'px']
            }
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
    // The Blob look: a disc whose radius wobbles organically, thresholded into a soft disc,
    // lit as a puffy dome, filled with a drifting gradient. The schedules at the top of this
    // file are the look; the words are the language.
    paint: (params) => {
        const {uniforms} = params
        const scalar = (ref: ReturnType<typeof p>) => resolveScalar(ref, params as never)

        // Normalize the highlight direction on the CPU each frame into the normL* extraFields.
        driveUnitDirection(params,
            {x: 'highlightX', y: 'highlightY', z: 'highlightZ'},
            {x: 'normLx', y: 'normLy', z: 'normLz'})

        // Frame: RAW canvas viewport (the disc anchors to the canvas, not a resize-fit box),
        // centred on the position prop. Per-node clock offset by the seed.
        const t = local(animatedTime(params, 'seed'), 'blobTime')
        const {delta, dist} = centredFrame({center: p('center')})(params, rawSurfaceOf(params))
        const axes = {x: delta.member('x'), y: delta.member('y'), t}

        // Radius: base size + the wobble stack, scaled by deformation.
        const deform = scalar(p('deformation'))
        const radius = local(add(scalar(p('size')),
            WOBBLE
                .map(({schedule, amount}) => mul(mul(waves(axes, schedule), amount), deform))
                .reduce((a, b) => add(a, b))), 'blobRadius')

        // Coverage: softness drives both the band width (× 0.3, the softness-ring
        // conversion) and the edge curve (× 2 + 0.5).
        const softness = local(scalar(p('softness')), 'blobSoftness')
        const mask = local(softDisc({dist, radius, softness: {
            width: mul(softness, 0.3),
            curve: add(mul(softness, 2), 0.5),
        }}), 'blobMask')

        // Highlight: shine a light on the puffy dome, lifted toward the rim by curvature.
        const specular = shine({
            normal: domeNormal({delta, dist, radius}),
            light: vec3(uniforms.normLx, uniforms.normLy, uniforms.normLz),
            gloss: 32,
        })
        const curvature = add(mul(smoothstep(0, radius, dist), 0.5), 0.5)
        const highlight = mul(mul(mul(specular, curvature), scalar(p('highlightIntensity'))), mask)

        // Fill: two drifting wave layers (over a 3× finer domain) pick along the gradient
        // (two-color or multi-stop, mixed in the compile-time colorSpace).
        const fillAxes = {x: mul(axes.x, 3), y: mul(axes.y, 3), t}
        const drift = (schedule: WaveTerm[]) => add(mul(waves(fillAxes, schedule), 0.5), 0.5)
        const fill = stopsPalette(p('colorSpace'))(
            smoothstep(0.1, 0.9, add(mul(drift(FILL_DRIFT_A), 0.6), mul(drift(FILL_DRIFT_B), 0.4))), params)

        // Compose: additive highlight over the fill; alpha = mean endpoint alpha × coverage.
        const highlightColor = uniformOf(p('highlightColor'), params)
        const avgAlpha = mul(add(
            uniforms.colorA.member('a'), uniforms.colorB.member('a')), 0.5)
        return vec4(
            add(fill.member('rgb'), mul(highlightColor.member('rgb'), highlight)),
            mul(avgAlpha, mask))
    }
})

export default componentDefinition
