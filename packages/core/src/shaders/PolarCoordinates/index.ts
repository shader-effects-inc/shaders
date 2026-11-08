import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {toPolar} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    wrap: number
    radius: number
    intensity: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "PolarCoordinates",
    role: 'warp',
    category: "Distortions",
    description: "Convert rectangular coordinates to polar space",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The center point for polar coordinate conversion"),
        wrap: {
            default: 1,
            description: "Controls how much of the angular range to use (1 = full 360°, 0.5 = 180°)",
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Wrap', group: 'Effect' }
        },
        radius: {
            default: 1,
            description: "Controls how much of the radius range to use (affects the radial mapping)",
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        intensity: {
            default: 1,
            description: "Blends between original UVs (0) and polar coordinates (1)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        edges: edgesPropConfig('transparent', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: toPolar({center: p('center'), wrap: p('wrap'), radius: p('radius'), intensity: p('intensity')}),
    missingChildMessage: 'You must pass a child component into the PolarCoordinates shader.',
})

export default componentDefinition
