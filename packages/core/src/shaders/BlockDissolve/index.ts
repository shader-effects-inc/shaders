import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    blockSize: number
    softness: number
    invert: boolean
}

// Shuffled cells: each block gets a stable random threshold (no time dependence, so the ordering is
// frozen frame-to-frame and only `progress` moves it) and blocks with a lower threshold dissolve
// first. `softness` cross-fades blocks (0 = hard on/off squares).
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "BlockDissolve",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Dissolve the content away as a grid of blocks vanishing in random order",
    props: {
        progress: {
            default: 0.5,
            description: "How far the dissolve has progressed (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        blockSize: {
            default: 0.08,
            description: "Size of each block as a fraction of the frame width",
            ui: { type: ['range', 'map'], min: 0.01, max: 0.5, step: 0.01, label: 'Block Size', group: 'Transition' }
        },
        softness: {
            default: 0.15,
            description: "How softly each block fades out (0 = hard-edged blocks)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Reverse the order blocks dissolve in",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.cells(p('blockSize')).shuffled(),
        progress: p('progress'),
        feather: p('softness'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Block Dissolve shader.',
})

export default componentDefinition
