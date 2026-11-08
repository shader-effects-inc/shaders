import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {kaleidoscope} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    segments: number
    angle: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Kaleidoscope",
    role: 'warp',
    category: "Distortions",
    description: "Create a kaleidoscope effect with radial mirrored segments",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("The center point of the kaleidoscope effect"),
        segments: {
            default: 6,
            description: "Number of radial segments in the kaleidoscope",
            ui: { type: ['range', 'map'], min: 2, max: 24, step: 1, label: 'Segments', group: 'Effect' }
        },
        angle: {
            default: 0,
            description: "Rotation offset for the entire kaleidoscope pattern",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        edges: edgesPropConfig('mirror', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: kaleidoscope({center: p('center'), segments: p('segments'), angle: p('angle')}),
    missingChildMessage: 'You must pass a child component into the Kaleidoscope shader.',
})

export default componentDefinition
