import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {call, floatE, animatedTime, asLocal} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface} from "@coreroot/std/paint/materials"
import {
    heartbeatPulse, edgeGlowBand, orbitSpotsAccum, edgeGlowAccumZero, edgeGlowAccumColorFor, colorAtIndex, edgeGlowCompose,
} from "@coreroot/std/effects/edgeGlow"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

// Adapted from paper-design/shaders "pulsing-border" (MIT), rebuilt as an SDF SHAPE EFFECT: the
// luminous color spots race around the boundary of ANY shape (2D analytic, custom SVG, or
// raymarched 3D silhouette) instead of a fixed rounded rectangle. The border band, spot orbits,
// heartbeat pulse, smoke, and bloom accumulation are transcribed closely from the original; the
// rounded-box distance is replaced by the shape's signed distance field and the corner-circle
// fade (specific to a box) is dropped. Value noise replaces the randomizer texture.

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// Default spot colors as gradient stops (positions are ignored — each stop is one orbit color).
const DEFAULT_STOPS: ColorStop[] = [
    {color: '#7b2ff7', position: 0},
    {color: '#00e0ff', position: 0.5},
    {color: '#ff3d81', position: 1},
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
    thickness: number
    softness: number
    intensity: number
    bloom: number
    spots: number
    spotSize: number
    pulse: number
    smoke: number
    smokeSize: number
    speed: number
    seed: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "LightEdge",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Glowing, pulsing light racing around the edge of any 2D, SVG, or 3D shape. Powered by Paper Shaders.",
    animatedTime: { speed: 'speed' },
    // The shape-effect spine, centre-tap only: only `.r` (the field) is read, so patternMode
    // 'none', no grad sampler and no neighbour taps; 'firstLobe' keeps the band hugging the
    // FRONT silhouette of 3D shapes. The recipe below stages border → per-color spot
    // accumulation → bloom compose.
    ...shapedSurface({
        chord: 'firstLobe',
        stencil: 'centre',
        surface: (frame, params) => {
            const {uniforms, propValues} = params
            const t = animatedTime(params)

            // Heartbeat pulse evaluated ONCE per pixel (hoisted to a WGSL local) and shared by the
            // band's smoke and every per-color orbitSpotsAccum call below.
            const beat = asLocal(call(heartbeatPulse, 'heartbeatPulse', [t]), 'beat')

            const bAndA = call(edgeGlowBand, 'edgeGlowBand', [
                frame.surf0!.member('r'), frame.sdfUV!, params.ctx.viewportSize, uniforms.scale, uniforms.thickness, uniforms.softness,
                uniforms.smokeSize, uniforms.smoke, t, uniforms.pulse, beat,
            ])
            const border = bAndA.member('x')
            const angle01 = bAndA.member('y')

            // Color list: the gradient stops when active (compile-time stopCount — colorStopsPropConfig
            // recomposes on count changes), else the colorA/colorB pair. Each color gets its own spot
            // orbit accumulation, folded through the running state in paper order; overlaps blend in
            // the compile-time colorSpace.
            const stopCount = (propValues.stopCount as number) ?? 0
            const colorCount = stopCount > 1 ? Math.min(stopCount, 8) : 2
            const colorSpaceMode = (propValues.colorSpace as number) ?? 0
            const accumColorFn = edgeGlowAccumColorFor(colorSpaceMode)
            const accumFor = (idx: number): Expr => call(orbitSpotsAccum, 'orbitSpotsAccum', [
                angle01, border, t, beat, uniforms.spots, uniforms.spotSize, uniforms.pulse,
                uniforms.intensity, uniforms.softness, uniforms.seed, floatE(idx),
            ])
            const colorFor = (idx: number): Expr => stopCount > 1
                ? call(colorAtIndex, 'colorAtIndex', [uniforms.colorsArray, floatE(idx)])
                : (idx === 0 ? uniforms.colorA : uniforms.colorB)

            let state = call(edgeGlowAccumZero, 'edgeGlowAccumZero', [])
            for (let i = 0; i < colorCount; i++) {
                state = call(accumColorFn, 'edgeGlowAccumColor', [state, colorFor(i), accumFor(i)])
            }
            return call(edgeGlowCompose, 'edgeGlowCompose', [state, uniforms.bloom])
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
            description: 'Center position of the shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        colorA: {
            default: '#7b2ff7',
            transform: transformColor,
            description: 'First border light color (used when no gradient stops are set)',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#00e0ff',
            transform: transformColor,
            description: 'Second border light color (used when no gradient stops are set)',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Spot colors as gradient stops — each stop is one orbiting color (positions ignored).
        stops: {...colorStopsPropConfig(), default: DEFAULT_STOPS},
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space overlapping lights blend in',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        },
        thickness: {
            default: 0.25,
            description: 'Width of the border band',
            ui: { type: ['range', 'map'], min: 0.01, max: 1, step: 0.01, label: 'Thickness', group: 'Border' }
        },
        softness: {
            default: 0.5,
            description: 'Edge softness — 0 = crisp band, 1 = wide smooth gradient',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Border' }
        },
        intensity: {
            default: 0.6,
            description: 'Brightness of the individual light spots',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Light' }
        },
        bloom: {
            default: 0.3,
            description: 'How additively overlapping lights pile up — high values overdrive to white',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Bloom', group: 'Light' }
        },
        spots: {
            default: 3,
            description: 'Number of light spots orbiting per color',
            ui: { type: 'range', min: 1, max: 5, step: 1, label: 'Spots', group: 'Light' }
        },
        spotSize: {
            default: 0.5,
            description: 'Angular size of each light spot',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Spot Size', group: 'Light' }
        },
        pulse: {
            default: 0,
            description: 'Heartbeat pulsing — synchronizes the lights to a double-beat rhythm',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Pulse', group: 'Animation' }
        },
        smoke: {
            default: 0.3,
            description: 'Noisy luminous smoke drifting along the border',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Smoke', group: 'Smoke' }
        },
        smokeSize: {
            default: 0.5,
            description: 'Scale of the smoke billows',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Smoke Size', group: 'Smoke' }
        },
        speed: {
            default: 1,
            description: 'Animation speed. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: 'Random seed — re-rolls spot speeds, directions and phases',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation' }
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
