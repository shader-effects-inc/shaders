import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isValue} from "@coreroot/std"
import {saturate} from "@coreroot/std/effects/color"

export interface ComponentProps {
    intensity: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Saturation",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Adjust color saturation intensity",
    props: {
        intensity: {
            default: 1.0,
            description: 'The intensity of the saturation effect (1 being no change)',
            recompile: crosses(1),
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.1, label: 'Intensity', group: 'Adjustments' }
        }
    },
    effect: saturate(p('intensity')),
    // At intensity=1 the mix is identity — bypass the luminance/mix math.
    identityWhen: isValue('intensity', 1),
    missingChildMessage: 'You must pass a child component into the Saturation shader.',
})

export default componentDefinition
