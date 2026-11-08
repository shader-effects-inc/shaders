import {defineShader, p, crosses} from "@coreroot/std"
import {bloom} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    intensity: number
    threshold: number
    size: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "Glow",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Stylize",
    description: "Soft glow effect with adjustable intensity",
    requiresRTT: true,
    requiresChild: true,
    props: {
        intensity: {
            default: 1.0,
            description: 'Glow intensity (brightness of the glow effect)',
            ui: {type: ['range', 'map'], min: 0, max: 50, step: 0.5, label: 'Intensity', group: 'Effect'}
        },
        threshold: {
            default: 0.5,
            description: 'Brightness threshold for glow extraction (lower = more glow)',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Threshold', group: 'Effect'}
        },
        size: {
            default: 25,
            description: 'Glow spread in pixels (clean up to ~72px, mild banding above)',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Glow Size', group: 'Effect'}
        }
    },

    // Compute-backed bloom: bright-extract + blur at aspect-aware compute resolution, composited
    // as `original + bloom × intensity`. size=0 skips compute (the crosses(0) rule recomposes on
    // crossing) and the fragment falls through to child passthrough.
    ...bloom({intensity: p('intensity'), threshold: p('threshold'), size: p('size')}),
})

export default componentDefinition
