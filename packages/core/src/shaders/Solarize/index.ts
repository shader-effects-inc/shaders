import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isZero} from "@coreroot/std"
import {solarize} from "@coreroot/std/effects/color"

export interface ComponentProps {
    threshold: number
    strength: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Solarize",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Inverts tones above a luminance threshold — a classic darkroom and photo effect",
    props: {
        threshold: {
            default: 0.5,
            description: "Luminance level above which colors are inverted. Pixels brighter than this threshold get flipped.",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Threshold', group: 'Effect' }
        },
        strength: {
            default: 1,
            description: "Blend between original (0) and fully solarized (1)",
            // Recompile only when strength crosses 0 (identity ↔ effect) — drives the bypass below.
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        }
    },
    effect: solarize({threshold: p('threshold'), strength: p('strength')}),
    // strength=0 collapses the final mix to the original rgb, whatever the threshold.
    identityWhen: isZero('strength'),
    missingChildMessage: 'You must pass a child component into the Solarize shader.',
})

export default componentDefinition
