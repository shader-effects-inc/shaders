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

// The classic staggered checkerboard: even-parity cells sweep away corner-to-corner over the first
// half of progress, odd-parity cells over the second. `softness` cross-fades cells (0 = hard
// on/off squares).
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "CheckerWipe",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Wipe the content away as a checkerboard of fading squares",
    props: {
        progress: {
            default: 0.5,
            description: "How far the wipe has progressed (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        blockSize: {
            default: 0.1,
            description: "Size of each square as a fraction of the frame width",
            ui: { type: ['range', 'map'], min: 0.02, max: 0.5, step: 0.01, label: 'Block Size', group: 'Transition' }
        },
        softness: {
            default: 0.15,
            description: "How softly each square fades out (0 = hard-edged squares)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Reverse the order the squares wipe in",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.cells(p('blockSize')).checker(),
        progress: p('progress'),
        feather: p('softness'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Checker Wipe shader.',
})

export default componentDefinition
