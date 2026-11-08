import {defineShader, p, crosses, isZero} from "@coreroot/std"
import {sharpen} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    sharpness: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "Sharpness",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Adjust image sharpness using a convolution kernel",
    props: {
        sharpness: {
            default: 0,
            description: 'How sharp to make the underlying image',
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.1, label: 'Sharpness', group: 'Adjustments' }
        }
    },
    // No `missingChildMessage`: Sharpness fails silently (transparent) when childless.
    // At sharpness=0 the kernel collapses to identity (centerWeight=1, neighborWeight=0). The
    // species' bypass samples the centre and unpremultiplies — it CANNOT return the child raw
    // (see the rttFilter header).
    identityWhen: isZero('sharpness'),
    effect: sharpen(p('sharpness')),
})

export default componentDefinition
