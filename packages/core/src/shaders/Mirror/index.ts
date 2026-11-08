import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {mirrorLine} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    angle: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Mirror",
    role: 'warp',
    category: "Distortions",
    description: "Mirror content across a line defined by center point and angle",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The point the mirror line passes through"),
        angle: {
            default: 0,
            description: "The angle of the mirror line in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        edges: edgesPropConfig('mirror', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: mirrorLine({center: p('center'), angle: p('angle')}),
    missingChildMessage: 'You must pass a child component into the Mirror shader.',
})

export default componentDefinition
