import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    angle: number
    stripCount: number
    feather: number
    invert: boolean
}

// LinearWipe's directional coverage tiled into `stripCount` strips: every strip closes an identical
// soft band in lockstep as progress grows — parallel venetian blinds shutting.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "VenetianBlinds",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Wipe the content away behind a set of parallel closing strips",
    props: {
        progress: {
            default: 0.5,
            description: "How far the blinds have closed (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        angle: {
            default: 0,
            description: "Orientation of the strips in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Transition' }
        },
        stripCount: {
            default: 5,
            description: "Number of strips across the frame",
            ui: { type: ['range', 'map'], min: 2, max: 60, step: 1, label: 'Strips', group: 'Transition' }
        },
        feather: {
            default: 0.15,
            description: "Softness of each closing strip edge",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Close each strip from the opposite edge",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.directional(p('angle')).tiled(p('stripCount')),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Venetian Blinds shader.',
})

export default componentDefinition
