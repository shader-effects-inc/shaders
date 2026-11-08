import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, rings, dist, stops} from "@coreroot/std/paint/fields"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    center: Parameters<typeof transformPosition>[0]
    size: number
    rotation: number
    repeat: number
    roundness: number
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "DiamondGradient",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Diamond-shaped gradient radiating from a center point using Manhattan distance",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` so the gradient evaluates against the distorted UV at full canvas resolution.
    // Gradients are continuous, so no AA changes are needed.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#4ffb4a",
            transform: transformColor,
            description: "Color at the center of the diamond",
            ui: { type: 'color', label: 'Color A (Center)', group: 'Colors' }
        },
        colorB: {
            default: "#4f1238",
            transform: transformColor,
            description: "Color at the outer edges of the diamond",
            ui: { type: 'color', label: 'Color B (Edge)', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Center point of the diamond",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        size: {
            default: 0.7,
            description: "Extent of the gradient — controls how far Color A reaches before transitioning to Color B",
            ui: { type: ['range', 'map'], min: 0.01, max: 2, step: 0.01, label: 'Size', group: 'Effect' }
        },
        rotation: {
            default: 0,
            description: "Rotation in degrees — tilts the diamond into a rhombus",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        repeat: {
            default: 1,
            description: "Number of times the gradient repeats outward. Values above 1 create concentric diamond or square bands.",
            ui: { type: ['range', 'map'], min: 1, max: 16, step: 0.5, label: 'Repeat', group: 'Effect' }
        },
        roundness: {
            default: 0,
            description: "Morphs from a sharp diamond (0) to a square (1)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Roundness', group: 'Effect' }
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
    paint: rampOver(
        rings(dist.diamond({center: p('center'), size: p('size'), rotation: p('rotation'), roundness: p('roundness')}), p('repeat')),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
