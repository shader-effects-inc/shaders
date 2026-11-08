import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    angle: number
    feather: number
    invert: boolean
}

// The plainest member of the wipe family: a directional coverage (0→1 along `angle`, aspect
// corrected) fed straight to the reveal tail.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "LinearWipe",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Wipe the content away along a straight edge with a soft feathered transition",
    props: {
        progress: {
            default: 0.5,
            description: "How far the wipe has travelled (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        angle: {
            default: 0,
            description: "Direction of the wipe in degrees (0 = left to right)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Transition' }
        },
        feather: {
            default: 0.1,
            description: "Softness of the wipe edge",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Reverse the direction the wipe travels",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.directional(p('angle')),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Linear Wipe shader.',
})

export default componentDefinition
