import {defineShader, p} from "@coreroot/std"
import {pixelate} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    scale: number
    gap: number
    roundness: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "Pixelate",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Pixelation effect with adjustable cell size",
    props: {
        scale: {
            default: 50,
            description: "Number of pixels along the longest edge (higher = smaller pixels)",
            ui: { type: ['range', 'map'], min: 1, max: 200, step: 1, label: 'Scale', group: 'Effect' }
        },
        gap: {
            default: 0,
            description: "Space between pixels as a fraction of cell size (0 = no gap, 1 = fully invisible)",
            ui: { type: ['range', 'map'], min: 0, max: 0.95, step: 0.01, label: 'Gap', group: 'Effect' }
        },
        roundness: {
            default: 0,
            description: "Roundness of each pixel's corners (0 = square, 1 = circle)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Roundness', group: 'Effect' }
        }
    },

    // The gap/roundness mask is an alpha cut, not just a coordinate bend, so the composer always
    // takes the RTT path (no analytic uvRemap).
    effect: pixelate({scale: p('scale'), gap: p('gap'), roundness: p('roundness')}),
})

export default componentDefinition
