import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {displaceByLayer} from "@coreroot/std/warps"
import {transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    source: string
    amount: number
    channelMode: string
    angle: number
    edges: string
}

// The prop uses an inline transform, so `propValues.channelMode` arrives as the mapped number.
const CHANNEL_MODES: Record<string, number> = {twoAxis: 0, directional: 1}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "DisplacementMap",
    role: 'filter',
    species: 'gather',
    category: "Distortions",
    description: "Distorts child content using another layer's pixels as a displacement map",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        source: {
            default: '',
            description: 'The layer whose pixels drive the displacement — its red/green (or luminance) push the content around',
            // compileTime → the selected layer id is part of the structural hash; switching layers
            // recomposes and re-resolves the source RTT boundary.
            compileTime: true,
            ui: { type: 'layer', label: 'Source Layer', group: 'Effect' }
        },
        amount: {
            default: 0.3,
            description: 'Overall displacement strength',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Amount', group: 'Effect' }
        },
        channelMode: {
            default: 'twoAxis',
            description: 'How the source drives the push — red→X / green→Y, or luminance along a fixed angle',
            compileTime: true,
            transform: (value: string) => CHANNEL_MODES[value] ?? 0,
            ui: {
                type: 'select',
                options: [
                    {label: 'Red / Green', value: 'twoAxis'},
                    {label: 'Luminance', value: 'directional'}
                ],
                label: 'Channels',
                group: 'Effect'
            }
        },
        angle: {
            default: 0,
            description: 'Push direction in degrees (used by the Luminance channel mode)',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect', condition: { channelMode: 'directional' } }
        },
        edges: {
            default: 'mirror',
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
        }
    },

    // Silent on a missing child, as this shader has always been. No source selected (or an
    // unresolvable id) → sharp passthrough. Animation comes for free: an animated source layer
    // displaces live.
    effect: displaceByLayer({
        source: p('source'),
        amount: p('amount'),
        channels: p('channelMode'),
        angle: p('angle'),
        edges: p('edges'),
    }),
})

export default componentDefinition
