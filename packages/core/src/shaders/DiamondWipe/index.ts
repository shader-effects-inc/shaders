import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    size: number
    feather: number
    invert: boolean
}

// A diamond hole grows from every cell center in lockstep; at progress 1 the diamonds of
// neighboring cells meet at the corners and tile the plane exactly, clearing every pixel.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "DiamondWipe",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Wipe the content away through a lattice of growing diamonds",
    props: {
        progress: {
            default: 0.5,
            description: "How far the diamonds have grown (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        size: {
            default: 0.15,
            description: "Size of each diamond cell as a fraction of the frame width",
            ui: { type: ['range', 'map'], min: 0.02, max: 0.5, step: 0.01, label: 'Size', group: 'Transition' }
        },
        feather: {
            default: 0.1,
            description: "Softness of each diamond edge",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Wipe from the cell corners inward instead",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.cells(p('size')).diamond(),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Diamond Wipe shader.',
})

export default componentDefinition
