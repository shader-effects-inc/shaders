import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {twirl} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    intensity: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Twirl",
    role: 'warp',
    category: "Distortions",
    description: "Rotate and twist content around a center point",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The center point of the twirl effect"),
        intensity: {
            default: 1.0,
            description: "The strength of the twirl effect",
            ui: { type: ['range', 'map'], min: -5, max: 5, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        edges: edgesPropConfig('stretch', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: twirl({center: p('center'), intensity: p('intensity')}),
    missingChildMessage: 'You must pass a child component into the Twirl shader.',
})

export default componentDefinition
