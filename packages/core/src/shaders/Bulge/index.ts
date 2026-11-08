import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {bulge} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    strength: number
    radius: number
    falloff: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Bulge",
    role: 'warp',
    category: "Distortions",
    description: "Magnify or pinch content around a center point",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The center point of the bulge effect"),
        strength: {
            default: 1,
            description: "The intensity of the bulge effect (positive = bulge out, negative = pinch in)",
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        radius: {
            default: 1,
            description: "The radius of the bulge effect area",
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        falloff: {
            default: 0.5,
            description: "Controls the smoothness of the transition (0 = hard edge, 1 = very smooth)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Falloff', group: 'Effect' }
        },
        edges: edgesPropConfig('stretch', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: bulge({center: p('center'), strength: p('strength'), radius: p('radius'), falloff: p('falloff')}),
    missingChildMessage: 'You must pass a child component into the Bulge shader.',
})

export default componentDefinition
