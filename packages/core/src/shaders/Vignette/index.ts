import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, tintToward, radialMask, isZero, crosses} from "@coreroot/std"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number
    falloff: number
    intensity: number
}

// std pointwise filter (species: no RTT, no premultiply — operates on the composed child inline).
// Tints the frame toward `color` by an aspect-corrected radial mask × intensity; alpha preserved.
// GPU bodies live in the kit behind the nouns: `fields.radialFalloffMask` (owns the aspect
// correction + the transformPosition center double-flip) and `colorMixing.mixToward`.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Vignette",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Darkens or tints the edges of the frame, drawing attention toward the center",
    props: {
        color: {
            default: "#000000",
            transform: transformColor,
            description: "Color of the vignette at the edges",
            ui: { type: 'color', label: 'Color', group: 'Colors' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Center of the clear area where the vignette begins",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        radius: {
            default: 0.5,
            description: "Distance from center where the vignette begins to fade in",
            ui: { type: ['range', 'map'], min: 0, max: 1.5, step: 0.01, label: 'Radius', group: 'Effect' }
        },
        falloff: {
            default: 0.5,
            description: "Width of the transition zone from clear to full vignette",
            ui: { type: ['range', 'map'], min: 0.01, max: 1.5, step: 0.01, label: 'Falloff', group: 'Effect' }
        },
        intensity: {
            default: 1,
            description: "Strength of the vignette effect",
            // Recompile only when intensity crosses 0 (identity ↔ effect) — pairs with identityWhen.
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Effect' }
        }
    },
    effect: tintToward(p('color'), {
        amount: radialMask({ center: p('center'), radius: p('radius'), falloff: p('falloff') })
            .times(p('intensity')),
    }),
    // intensity=0 zeroes the mix amount whatever the mask is — skips the whole radial computation.
    identityWhen: isZero('intensity'),
    missingChildMessage: 'You must pass a child component into the Vignette shader.',
})

export default componentDefinition
