import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isZero} from "@coreroot/std"
import {vibrance} from "@coreroot/std/effects/color"

export interface ComponentProps {
    intensity: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Vibrance",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Selective saturation adjustment protecting skin tones",
    props: {
        intensity: {
            default: 0,
            description: 'The intensity of the vibrance effect',
            // Recompile only when intensity crosses 0 (identity ↔ effect) — pairs with identityWhen.
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: -2, max: 2, step: 0.1, label: 'Intensity', group: 'Adjustments' }
        }
    },
    effect: vibrance(p('intensity')),
    // intensity=0 zeroes the mix amount, so the mix returns the original rgb. Note the identity is
    // 0, not 1 (this is an ADJUSTMENT around zero, unlike Saturation's multiplier) — and it is the
    // default, so a freshly-added Vibrance is free until it is adjusted.
    identityWhen: isZero('intensity'),
    missingChildMessage: 'You must pass a child component into the Vibrance shader.',
})

export default componentDefinition
