import {defineShader, p} from "@coreroot/std"
import {channelBlur} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    redIntensity: number
    greenIntensity: number
    blueIntensity: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "ChannelBlur",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Blurs",
    description: "Independent blur for red, green, and blue channels",
    requiresRTT: true,
    requiresChild: true,
    props: {
        redIntensity: {
            default: 0,
            description: 'Blur intensity for red channel',
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Red Intensity', group: 'Effect'}
        },
        greenIntensity: {
            default: 20,
            description: 'Blur intensity for green channel',
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Green Intensity', group: 'Effect'}
        },
        blueIntensity: {
            default: 40,
            description: 'Blur intensity for blue channel',
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Blue Intensity', group: 'Effect'}
        }
    },

    // Compute-backed: one fixed Gaussian at the max per-channel radius; the fragment mixes each
    // channel toward it.
    ...channelBlur({red: p('redIntensity'), green: p('greenIntensity'), blue: p('blueIntensity')}),
})

export default componentDefinition
