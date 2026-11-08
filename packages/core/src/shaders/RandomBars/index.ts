import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    angle: number
    barCount: number
    softness: number
    invert: boolean
}

// The directional coverage banded into `barCount` parallel bars, each with a stable random
// threshold (no time dependence — the shuffle is frozen frame-to-frame): bars with a lower
// threshold vanish first, the PowerPoint "Random Bars" wipe. `softness` cross-fades each bar out.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "RandomBars",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Wipe the content away as parallel bars vanishing in random order",
    props: {
        progress: {
            default: 0.5,
            description: "How far the wipe has progressed (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        angle: {
            default: 0,
            description: "Orientation of the bars in degrees (0 = vertical bars)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Transition' }
        },
        barCount: {
            default: 12,
            description: "Number of bars across the frame",
            ui: { type: ['range', 'map'], min: 2, max: 100, step: 1, label: 'Bars', group: 'Transition' }
        },
        softness: {
            default: 0.15,
            description: "How softly each bar fades out (0 = hard-edged bars)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Reverse the order the bars vanish in",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.directional(p('angle')).bands(p('barCount')).shuffled(),
        progress: p('progress'),
        feather: p('softness'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Random Bars shader.',
})

export default componentDefinition
