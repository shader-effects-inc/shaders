import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {stretch} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    strength: number
    angle: number
    falloff: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Stretch",
    role: 'warp',
    category: "Distortions",
    description: "Stretch content towards a direction from a center point",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The center point of the stretch effect"),
        strength: {
            default: 1,
            description: "The intensity of the stretch effect",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        angle: {
            default: 0,
            description: "The direction of the stretch in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        falloff: {
            default: 0,
            description: "Controls the sharpness of the transition (0 = sharp edge, 1 = gradual transition)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Falloff', group: 'Effect' }
        },
        edges: edgesPropConfig('stretch', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: stretch({center: p('center'), strength: p('strength'), angle: p('angle'), falloff: p('falloff')}),
    missingChildMessage: 'You must pass a child component into the Stretch shader.',
})

export default componentDefinition
