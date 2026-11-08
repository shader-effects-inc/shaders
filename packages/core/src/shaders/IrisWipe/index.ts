import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformPosition, transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    center: Parameters<typeof transformPosition>[0]
    feather: number
    invert: boolean
}

// An expanding circle grows from `center`, wiping the child to transparent inside it. The radial
// coverage is corner-normalized, so progress 1 clears every pixel however off-center the origin is.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "IrisWipe",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Reveal through an expanding circle growing from a center point",
    props: {
        progress: {
            default: 0.5,
            description: "How far the iris has expanded (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Point the iris expands from",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        feather: {
            default: 0.1,
            description: "Softness of the iris edge",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Wipe from the outside inward instead",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.radial(p('center')),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Iris Wipe shader.',
})

export default componentDefinition
