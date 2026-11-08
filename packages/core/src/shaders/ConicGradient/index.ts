import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, tiles, dist, stops} from "@coreroot/std/paint/fields"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    center: Parameters<typeof transformPosition>[0]
    rotation: number
    repeat: number
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ConicGradient",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Colors sweep in a full circle around a center point, like a color wheel",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` so the gradient evaluates against the distorted UV at full canvas resolution.
    // Gradients are continuous, so no AA changes are needed.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#FF0080",
            transform: transformColor,
            description: "Starting color of the sweep",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#00BFFF",
            transform: transformColor,
            description: "Ending color of the sweep (wraps back to Color A)",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Center point of the sweep",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: "Rotation offset in degrees — shifts where Color A begins",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        repeat: {
            default: 1,
            description: "Number of times the gradient repeats around the circle. Values above 1 create a starburst pattern.",
            ui: { type: ['range', 'map'], min: 1, max: 24, step: 1, label: 'Repeat', group: 'Effect' }
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
        tiles(dist.conic({center: p('center'), rotation: p('rotation')}), p('repeat')),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
