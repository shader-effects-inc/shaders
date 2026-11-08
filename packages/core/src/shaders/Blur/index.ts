import {defineShader, p} from "@coreroot/std"
import {gaussianBlur} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    intensity: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "Blur",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Blurs",
    description: "A simple Gaussian blur effect",
    requiresRTT: true,
    requiresChild: true,
    props: {
        intensity: {
            default: 50,
            description: 'Intensity of the blur effect',
            ui: {type: ['range', 'map'], min: 0, max: 200, step: 1, label: 'Intensity', group: 'Effect'}
        }
    },

    // Compute-backed: fixed-kernel Gaussian for a static/mouse/auto intensity, per-pixel variable
    // Gaussian when intensity carries a map driver. The noun owns both compute paths and the
    // blurred-over-sharp fragment composite.
    ...gaussianBlur({intensity: p('intensity')}),
})

export default componentDefinition
