import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface} from "@coreroot/std/paint/materials"
import {effects} from "@coreroot/gpu/kit"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {applyThinFilmEffect} = effects.thinFilm

// Default shape configuration (matches Glass)
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

/** Resolve the active analytic shape type (compile-time `shapeType`, else the shape JSON's type). */
// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    intensity: number
    rimWidth: number
    edgeSoftness: number
    thickness: number
    dispersion: number
    saturation: number
    hueShift: number
    lightAngle: number
    speed: number
    mode: string
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorC: Parameters<typeof transformColor>[0]
    colorSpace: string
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "ThinFilm",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Iridescent thin-film edge",
    // The rim spectrum rotation reads the CPU-accumulated per-node animated-time field; the kit
    // builder reads `params.props.member('_animTime')`.
    animatedTime: {speed: 'speed'},
    // The shape-effect spine (stencil 'none' — the rim builder samples for itself).
    // `bakedGradients: 'volumetric'` — on the compute path the grad sampler carries the analytic
    // (ring-free B-spline) gradient in `.g/.b` of the single centre tap, so the rim/hue stays
    // smooth across 3D creases with no finite-difference taps; the flat paths take their own.
    // The material is the kit's golden-tested thin-film rim builder.
    ...shapedSurface({
        chord: 'firstLobe',
        gradSampler: 'same',
        bakedGradients: 'volumetric',
        stencil: 'none',
        surface: (frame, params) =>
            applyThinFilmEffect(params, frame.sampler, {volumetric: frame.volumetric, bakedGradients: frame.bakedGradients}),
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
            ui: { type: ['range', 'map'], min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        intensity: {
            default: 1,
            description: 'Rim brightness — values above 1 push the edge past white for HDR bloom',
            ui: { type: ['range', 'map'], min: 0, max: 4, step: 0.01, label: 'Intensity', group: 'Film' }
        },
        rimWidth: {
            default: 1,
            description: 'How far inward from the edge the iridescence reaches',
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Rim Width', group: 'Film' }
        },
        edgeSoftness: {
            default: 0.3,
            description: 'Softness of the shape boundary',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Film' }
        },
        thickness: {
            default: 0.5,
            description: 'Film thickness — higher values pack more spectral bands across the rim',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Thickness', group: 'Film' }
        },
        dispersion: {
            default: 0.5,
            description: 'How strongly color separates around the perimeter (warm vs cool sides)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Dispersion', group: 'Film' }
        },
        saturation: {
            default: 1,
            description: 'Spectral saturation — 0 = white rim, 1 = full iridescence',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Saturation', group: 'Film' }
        },
        hueShift: {
            default: 0,
            description: 'Rotate the spectrum to reposition the colors',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Hue Shift', group: 'Film' }
        },
        lightAngle: {
            default: 300,
            description: 'Light angle in degrees — sets which side of the rim runs warm vs cool',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Film' }
        },
        mode: {
            default: 'rainbow',
            compileTime: true,
            description: 'Rainbow uses the full iridescent spectrum; Custom cycles through your three chosen colors',
            ui: {
                type: 'select',
                options: [
                    { label: 'Rainbow', value: 'rainbow' },
                    { label: 'Custom', value: 'custom' }
                ],
                label: 'Mode',
                group: 'Colors'
            }
        },
        colorA: {
            default: '#2b6fff',
            transform: transformColor,
            description: 'First color in the rim cycle',
            ui: { type: 'color', label: 'Color 1', group: 'Colors', condition: { mode: 'custom' } }
        },
        colorB: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Second color in the rim cycle',
            ui: { type: 'color', label: 'Color 2', group: 'Colors', condition: { mode: 'custom' } }
        },
        colorC: {
            default: '#ff7a21',
            transform: transformColor,
            description: 'Third color in the rim cycle',
            ui: { type: 'color', label: 'Color 3', group: 'Colors', condition: { mode: 'custom' } }
        },
        colorSpace: {
            default: 'oklch',
            transform: transformColorSpace,
            compileTime: true,
            description: 'color space used to blend between custom colors',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors', condition: { mode: 'custom' } }
        },
        speed: {
            default: 0.1,
            description: 'Speed at which the colors rotate around the rim (0 = static)',
            ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: { type: 'shape', label: 'Shape', group: 'Shape' }
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file — when non-empty, activates SVG mode and triggers a shader recompile'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type — triggers recompile when shape is switched. When empty, derived from shape JSON at mount time.'
        }
    },

})

export default componentDefinition
