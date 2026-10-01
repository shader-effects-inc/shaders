import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface} from "@coreroot/std/paint/materials"
import {effects} from "@coreroot/gpu/kit"
import {transformPosition} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {applyEmbossEffect} = effects.emboss

// Default shape configuration
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

/** Resolve the active analytic shape type (compile-time `shapeType`, else the shape JSON's type). */
// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    depth: number
    lightAngle: number
    lightIntensity: number
    shadowIntensity: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Emboss",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Embossed / debossed relief shading on top of child content, driven by a custom shape",
    requiresRTT: true,
    requiresChild: true,
    // The shape-effect spine (stencil 'none' — Emboss takes its own finite-difference neighbour +
    // trace taps, so it wants the CHEAP bilinear sampler for them on the volumetric path
    // (`gradSampler: 'fast'`, the default); on the flat paths gradSampler is the centre sampler
    // itself). The material is the kit's golden-tested emboss relief builder over the child RTT.
    ...shapedSurface({
        chord: 'firstLobe',
        stencil: 'none',
        child: 'required',
        surface: (frame, params) =>
            applyEmbossEffect(params, frame.sampler, frame.childTexture!, {volumetric: frame.volumetric, gradSampler: frame.gradSampler}),
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
            description: 'Center position of the embossed shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the embossed shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the embossed shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        depth: {
            default: -0.5,
            description: 'Relief depth — negative = inset (debossed), positive = raised (embossed)',
            ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Depth', group: 'Relief' }
        },
        lightAngle: {
            default: 260,
            description: 'Directional light angle in degrees — controls highlight and shadow direction',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Lighting' }
        },
        lightIntensity: {
            default: 0.6,
            description: 'Strength of the directional edge highlights and shadows',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Light Intensity', group: 'Lighting' }
        },
        shadowIntensity: {
            default: 0.3,
            description: 'Darkness of the relief shadow',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shadow Intensity', group: 'Relief' }
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
