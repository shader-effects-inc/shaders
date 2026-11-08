import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {posterize} from "@coreroot/std/effects/color"

export interface ComponentProps {
    intensity: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Posterize",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Reduce color depth to create a poster effect",
    props: {
        intensity: {
            default: 5,
            description: 'The intensity of the posterization effect (lower is more posterized)',
            ui: { type: ['range', 'map'], min: 2, max: 20, step: 1, label: 'Intensity', group: 'Effect' }
        }
    },
    effect: posterize(p('intensity')),
    // No identity: quantisation is visible at every step count in range (max 20 steps still bands).
    missingChildMessage: 'You must pass a child component into the Posterize shader.',
})

export default componentDefinition
