import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface} from "@coreroot/std/paint/materials"
import {effects} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {applyNeonEffect} = effects.neon

// Default shape configuration
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

/** Resolve the active analytic shape type (compile-time `shapeType`, else the shape JSON's type). */
// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    color: Parameters<typeof transformColor>[0]
    secondaryColor: Parameters<typeof transformColor>[0]
    secondaryBlend: number
    glowColor: Parameters<typeof transformColor>[0]
    tubeThickness: number
    intensity: number
    hotCoreIntensity: number
    glowIntensity: number
    glowRadius: number
    lightAngle: number
    specularIntensity: number
    specularSize: number
    cornerSmoothing: number
    flickerSpeed: number
    flickerAmount: number
    flowSpeed: number
    flowAmount: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Neon",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Photorealistic neon tube / 3D pipe effect driven by a custom shape",
    // The shape-effect spine (stencil 'none' — Neon takes its OWN 5-point finite-difference
    // stencil, so the neighbour taps use the cheap bilinear sampler on the volumetric path and
    // the centre sampler on the flat ones). The material is the kit's golden-tested neon tube
    // builder. `tubeThickness` carries a `(v) => v*0.05` prop transform, applied by the uniform
    // bridge, so its uniform already arrives scaled — no in-body fold.
    ...shapedSurface({
        chord: 'firstLobe',
        stencil: 'none',
        surface: (frame, params) =>
            applyNeonEffect(params, frame.sampler, params.ctx.time, {volumetric: frame.volumetric, gradSampler: frame.gradSampler}),
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
            description: 'Center position of the neon shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the neon shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the neon shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        color: {
            default: '#00ddff',
            transform: transformColor,
            description: 'Primary neon tube color',
            ui: { type: 'color', label: 'Tube Color', group: 'Colors' }
        },
        secondaryColor: {
            default: '#ff00aa',
            transform: transformColor,
            description: 'Shadow-side color for a two-tone / dual-lit pipe look',
            ui: { type: 'color', label: 'Secondary Color', group: 'Colors' }
        },
        secondaryBlend: {
            default: 0.5,
            description: 'Blend between mono (0) and two-tone (1) tube coloring',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Blend', group: 'Colors' }
        },
        glowColor: {
            default: '#00ddff',
            transform: transformColor,
            description: 'Color of the outer glow / bloom',
            ui: { type: 'color', label: 'Glow Color', group: 'Colors' }
        },
        tubeThickness: {
            default: 0.2,
            transform: (v: number) => v * 0.05,
            description: 'How far inward from the boundary the tube extends. Low = thin neon outline, high = thick 3D pipe',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Tube Thickness', group: 'Tube' }
        },
        intensity: {
            default: 1.5,
            description: 'Overall brightness multiplier',
            ui: { type: 'range', min: 0.5, max: 4, step: 0.01, label: 'Intensity', group: 'Tube' }
        },
        hotCoreIntensity: {
            default: 0.6,
            description: 'Bright white-hot center line — the gas discharge glow inside the tube',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Hot Core', group: 'Tube' }
        },
        glowIntensity: {
            default: 0.6,
            description: 'Outer glow / bloom strength',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Glow Intensity', group: 'Glow' }
        },
        glowRadius: {
            default: 0.25,
            description: 'How far the glow extends beyond the tube',
            ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Glow Radius', group: 'Glow' }
        },
        lightAngle: {
            default: 300,
            description: 'Directional light angle in degrees — controls 3D shading on the tube',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Lighting' }
        },
        specularIntensity: {
            default: 0.5,
            description: 'Specular highlight brightness on the tube surface',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Specular', group: 'Lighting' }
        },
        specularSize: {
            default: 0.5,
            description: 'Specular highlight size — 0 = tight pinpoint, 1 = broad sheen',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Specular Size', group: 'Lighting' }
        },
        cornerSmoothing: {
            default: 0.15,
            description: 'Rounds sharp corners to mimic how real glass tubes curve at bends',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Corner Smoothing', group: 'Tube' }
        },
        flickerSpeed: {
            default: 0,
            description: 'Flicker animation speed — 0 = off, higher = faster sporadic on/off',
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Speed', group: 'Flicker' }
        },
        flickerAmount: {
            default: 0.2,
            description: 'How often the neon flickers off — 0 = always on, 1 = frequent outages',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Amount', group: 'Flicker' }
        },
        flowSpeed: {
            default: 0,
            description: 'Flow animation speed — 0 = off, light rotates through the tube',
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Speed', group: 'Flow' }
        },
        flowAmount: {
            default: 0.3,
            description: 'Strength of the flowing brightness variation — 0 = uniform, 1 = dramatic',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Amount', group: 'Flow' }
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
