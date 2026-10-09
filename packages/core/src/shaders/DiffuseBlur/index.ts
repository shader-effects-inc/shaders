import {defineStd, p, crosses} from "@coreroot/std"
import {scatter} from "@coreroot/std/effects/blurs"
import {transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    intensity: number
    edges: string
    chromatic: number
}

export const componentDefinition = defineStd<ComponentProps>({
    name: "DiffuseBlur",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Blurs",
    description: "Grain-like pixel displacement at random",
    props: {
        intensity: {
            default: 30,
            description: 'Intensity of the diffuse blur effect',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Effect' }
        },
        edges: {
            default: 'stretch',
            description: 'How to handle edges when distortion pushes content out of bounds',
            transform: transformEdges,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Transparent', value: 'transparent'},
                    {label: 'Mirror', value: 'mirror'},
                    {label: 'Wrap', value: 'wrap'}
                ],
                label: 'Edges',
                group: 'Effect'
            }
        },
        chromatic: {
            default: 0,
            description: 'Scatters red and blue separately for a sparkly color grain',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Chromatic', group: 'Effect'}
        }
    },

    // chromatic is off at 0; the three-tap gather is only compiled while it is set.
    effect: scatter({amount: p('intensity'), edges: p('edges'), chromatic: p('chromatic')}),
})

export default componentDefinition
