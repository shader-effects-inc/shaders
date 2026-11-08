import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, dist, stops} from "@coreroot/std/paint/fields"
import {transformColor, transformPosition, transformAngle, transformEdges, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    stops: ColorStop[] | null,
    start: Parameters<typeof transformPosition>[0],
    end: Parameters<typeof transformPosition>[0],
    angle: Parameters<typeof transformAngle>[0],
    edges: string,
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "LinearGradient",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Create smooth linear color gradients",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` so the gradient evaluates against the distorted UV at full canvas resolution.
    // Gradients are continuous, so no AA changes are needed.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#1aff00",
            transform: transformColor,
            description: "The starting color of the gradient",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#0000ff",
            transform: transformColor,
            description: "The ending color of the gradient",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Multi-stop colors. Default null → the shader runs the literal original two-color
        // (colorA/colorB) path, so legacy presets and npm consumers keep the original result. When
        // present and non-empty, these stops override colorA/colorB. See utilities/colorStops.ts.
        stops: colorStopsPropConfig(),
        start: {
            default: {
                x: 0,
                y: 0.5
            },
            transform: transformPosition,
            description: "The starting point of the gradient",
            ui: { type: 'position', label: 'Start', group: 'Position' }
        },
        end: {
            default: {
                x: 1,
                y: 0.5
            },
            transform: transformPosition,
            description: "The ending point of the gradient",
            ui: { type: 'position', label: 'End', group: 'Position' }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: "Additional rotation angle of the gradient (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        edges: {
            default: 'stretch',
            transform: transformEdges,
            compileTime: true,
            description: 'How to handle areas beyond the gradient endpoints',
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Transparent', value: 'transparent'},
                    {label: 'Mirror', value: 'mirror'},
                    {label: 'Wrap', value: 'wrap'}
                ],
                label: 'Edges',
                group: 'Effect'
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
    paint: rampOver(
        dist.linear({from: p('start'), to: p('end'), angle: p('angle')}),
        stops(p('colorSpace')),
        {edges: p('edges')},
    )
})

export default componentDefinition
