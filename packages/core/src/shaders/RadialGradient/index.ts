import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, rings, dist, stops} from "@coreroot/std/paint/fields"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    stops: ColorStop[] | null,
    center: Parameters<typeof transformPosition>[0],
    radius: number,
    repeat: number,
    aspect: number,
    skewAngle: number,
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "RadialGradient",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Radial gradient radiating from a center point",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` so the gradient evaluates against the distorted UV at full canvas resolution.
    // Gradients are continuous, so no AA changes are needed.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#ff0000",
            transform: transformColor,
            description: "The starting color at the center of the gradient",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#0000ff",
            transform: transformColor,
            description: "The ending color at the edge of the gradient",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: "The center point of the radial gradient",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        radius: {
            default: 1,
            description: "The radius of the gradient (normalized, 0.0-1.0)",
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        repeat: {
            default: 1,
            description: "Number of times the gradient repeats. Values above 1 create concentric rings.",
            ui: { type: ['range', 'map'], min: 1, max: 20, step: 0.5, label: 'Repeat', group: 'Effect' }
        },
        aspect: {
            default: 1,
            description: "Stretches the gradient into an ellipse. Values below 1 compress vertically, above 1 compress horizontally.",
            ui: { type: ['range', 'map'], min: 0.1, max: 4, step: 0.01, label: 'Aspect', group: 'Effect' }
        },
        skewAngle: {
            default: 0,
            description: "Rotates the ellipse axis in degrees. Only visible when Aspect is not 1.",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Skew Angle', group: 'Effect' }
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
        rings(dist.radial({center: p('center'), radius: p('radius'), aspect: p('aspect'), skew: p('skewAngle')}), p('repeat')),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
