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

// LinearWipe's directional coverage folded about its midpoint: two wipe fronts open outward from
// the center line simultaneously — barn doors parting. `invert` folds the other way, so the doors
// close in from both edges.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "BarnDoors",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Split the content along a center line and wipe outward in both directions",
    props: {
        progress: {
            default: 0.5,
            description: "How far the doors have opened (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        angle: {
            default: 0,
            description: "Direction the doors open in degrees (0 = apart horizontally)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Transition' }
        },
        feather: {
            default: 0.1,
            description: "Softness of the wipe edges",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Close in from both edges instead of opening from the center",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.directional(p('angle')).folded(),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Barn Doors shader.',
})

export default componentDefinition
