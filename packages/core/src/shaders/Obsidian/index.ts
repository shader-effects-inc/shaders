import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd, p} from "@coreroot/std"
import {direction, directionFrame} from "@coreroot/std/frames"
import {
    shapedSurface, surfaceField, geometricNormal, surfacePattern, viewRay, grazingOf, reflect,
    studioSoftboxes, schlickFresnel, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {stops, foldRamp} from "@coreroot/std/paint/fields"
import {add, local, mix, mul, smoothstep, splat3, sub, vec4} from "@coreroot/std/math"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// Default iridescence — the vivid magenta → ultramarine → cyan → mint palette dark tinted glass
// throws off its side walls. The palette is folded end-to-end as it flows, so it must read as a
// rich multi-hue ramp, not two colors.
const DEFAULT_STOPS: ColorStop[] = [
    {color: '#ff2f7a', position: 0},
    {color: '#2a2cff', position: 0.3},
    {color: '#5fe6ff', position: 0.55},
    {color: '#a6ffd8', position: 0.75},
    {color: '#ffb3e6', position: 1},
]

// How far the palette slides per unit of view angle — turning a wall away from the viewer walks
// its color through the ramp, the way a thin-film hue shifts with incidence.
const ANGLE_SHIFT = 0.9
// Half-cycle length of the palette along the surface, in pattern units at flowScale 1.
const FLOW_CYCLE = 0.55
// Studio sky gain for the dark-glass reflection (glass reads a lower sky than metal).
const GLASS_SKY_GAIN = 0.25

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    bodyColor: Parameters<typeof transformColor>[0]
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    colorSpace: string
    iridescence: number
    rimWidth: number
    rimSoftness: number
    flowAngle: number
    flowScale: number
    speed: number
    gloss: number
    envRotation: number
    bevelWidth: number
    bevelShape: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Obsidian",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Dark tinted glass whose faces stay near-black while every surface turning away from the viewer ignites with a flowing iridescent gradient — vivid color living only on the oblique walls and bevels",
    animatedTime: {speed: 'speed'},
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the glass shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the glass shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the glass shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        bodyColor: {
            default: '#06071a',
            transform: transformColor,
            description: 'The dark body of the glass — what the faces looking straight at you show. Keep it deep for the iridescence to pop.',
            ui: { type: 'color', label: 'Body Color', group: 'Colors' }
        },
        colorA: {
            default: '#ff2f7a',
            transform: transformColor,
            description: 'First iridescent color (two-color fallback when stops are cleared)',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#5fe6ff',
            transform: transformColor,
            description: 'Second iridescent color (two-color fallback when stops are cleared)',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Defaults to a full multi-stop palette (the MeshGradient precedent) — the folded flow
        // needs a multi-hue ramp to read as iridescence rather than a two-tone sheen.
        stops: {...colorStopsPropConfig(), default: DEFAULT_STOPS},
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        },
        iridescence: {
            default: 1,
            description: 'Strength of the color on the oblique surfaces — 0 = plain dark glass, above 1 = saturated neon walls',
            ui: { type: ['range', 'map'], min: 0, max: 1.5, step: 0.01, label: 'Iridescence', group: 'Iridescence' }
        },
        rimWidth: {
            default: 0.55,
            description: 'How far in from edge-on the color reaches — small keeps it to the steepest walls, large lets it creep across gently turning faces',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Rim Width', group: 'Iridescence' }
        },
        rimSoftness: {
            default: 0.25,
            description: 'Softness of the transition from dark face to colored wall',
            ui: { type: ['range', 'map'], min: 0.02, max: 0.6, step: 0.01, label: 'Rim Softness', group: 'Iridescence' }
        },
        flowAngle: {
            default: 30,
            description: 'Direction the gradient flows across the surface, in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Flow Angle', group: 'Iridescence' }
        },
        flowScale: {
            default: 1,
            description: 'How many times the palette cycles across the shape — higher = tighter color bands',
            ui: { type: ['range', 'map'], min: 0.2, max: 4, step: 0.01, label: 'Flow Scale', group: 'Iridescence' }
        },
        speed: {
            default: 0.5,
            description: 'Speed the gradient flows along the walls. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Iridescence' }
        },
        gloss: {
            default: 0.35,
            description: 'Strength of the studio softboxes reflected in the glass — the pale highlights riding the top edges',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Gloss', group: 'Surface' }
        },
        ...glassShellProps({
            envRotation: {ui: {group: 'Surface'}},
        }),
        bevelWidth: {
            default: 0.08,
            description: 'Width of the edge bevel on flat shapes — the band that turns away from the viewer and catches the color',
            ui: { type: ['range', 'map'], min: 0.005, max: 0.3, step: 0.001, label: 'Bevel Width', group: 'Surface' }
        },
        bevelShape: {
            default: 0,
            description: 'Bevel profile — 0 = one smooth round fillet, 1 = machined: a steep outer fillet, a chamfer plateau and an inner knee',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Bevel Profile', group: 'Surface' }
        },
        ...glassShellProps({
            edgeSoftness: {ui: {group: 'Surface'}},
        }),
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

    // Dark glass as a recipe: the geometric normal → how edge-on each point is to the view → a rim
    // mask over that grazing term → the palette folded along a flow frame across the surface,
    // slid by view angle and time → mixed over the dark body by the rim → the studio reflection,
    // Fresnel-weighted so it too lives on the walls. The constants here ARE the look.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)

            const field = surfaceField(frame, params, {scale: u.scale})
            const n = geometricNormal(frame, field, {bevelWidth: u.bevelWidth, bevelShape: u.bevelShape})
            const view = viewRay(params, field)
            const grazing = grazingOf(n, view)

            // ── The rim: where the surface turns away from the viewer ──────────────
            const rimEdge = local(sub(1, u.rimWidth), 'rimEdge')
            const rim = local(smoothstep(sub(rimEdge, u.rimSoftness), add(rimEdge, u.rimSoftness), grazing), 'rim')

            // ── The iridescence: the palette flowing along the walls ───────────────
            const flow = directionFrame(u.flowAngle, 'flow')
            const along = flow.coordsOf(surfacePattern(frame, field)).along
            const phase = add(add(mul(along, mul(u.flowScale, 1 / FLOW_CYCLE)), mul(grazing, ANGLE_SHIFT)), mul(t, 0.2))
            const palette = stops(p('colorSpace'))
            const iri = local(palette(foldRamp(phase), params).member('rgb'), 'iri')

            // ── The studio, seen in dark glass: Fresnel puts it on the walls too ───
            const R = local(reflect(view, n), 'R')
            const env = studioSoftboxes(R.member('x'), R.member('y'), {
                rotation: direction(u.envRotation, 'env'),
                keyRadius: 0.3,
                drift: 0,
                strength: u.gloss,
                skyGain: GLASS_SKY_GAIN,
            })
            const shine = local(mul(env, schlickFresnel(grazing, {r0: 0.04, gain: 0.96, power: 4})), 'shine')

            // ── Compose: dark body → colored walls → pale glass highlights ────────
            const body = u.bodyColor.member('rgb')
            const walls = mix(body, iri, splat3(mul(rim, u.iridescence)))
            const rgb = add(walls, mul(mix(iri, splat3(1), splat3(0.6)), splat3(mul(shine, 0.5))))

            return guarded(insideShape(field.sdf, field.pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), ZERO, 'obsidian')
        },
    }),
})

export default componentDefinition
