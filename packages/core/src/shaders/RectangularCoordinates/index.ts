import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {fromPolar} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    scale: number
    intensity: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "RectangularCoordinates",
    role: 'warp',
    category: "Distortions",
    description: "Convert polar coordinates back to rectangular space",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The center point for rectangular coordinate conversion"),
        scale: {
            default: 1,
            description: "Scale factor for the rectangular output",
            ui: { type: ['range', 'map'], min: 0.1, max: 3, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        intensity: {
            default: 1,
            description: "Blends between original UVs (0) and rectangular coordinates (1)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        edges: edgesPropConfig('transparent', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: fromPolar({center: p('center'), scale: p('scale'), intensity: p('intensity')}),
    missingChildMessage: 'You must pass a child component into the RectangularCoordinates shader.',
})

export default componentDefinition
