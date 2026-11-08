import {defineShader} from "@coreroot/std"
import {noiseRelief, reliefBases} from "@coreroot/std/paint/noise"
import {reliefStylizeProps} from "@coreroot/utilities/noiseStylize"

export interface ComponentProps {
    intensity: number
    scale: number
    contrast: number
    distortion: number
    seed: number
}

// std custom-tier filter: an RTT relief filter over the child (the relief noun samples the child
// texture through the height-field displacement, so it is not a gather build).
export const componentDefinition = defineShader<ComponentProps>({
    name: "Wool",
    role: 'filter',
    species: 'custom',
    requiresRTT: true,
    requiresChild: true,
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Applies an interwoven fibrous fabric texture and distortion to child content",
    // High scale + low contrast keeps the weave fine and subtle, so it reads as a fabric texture.
    props: reliefStylizeProps({ intensity: 0.5, scale: 4, contrast: -0.5, distortion: 0.15 }),

    // RTT relief filter over the interwoven wool height field.
    gpu: {fragment: noiseRelief(reliefBases.wool)}
})

export default componentDefinition
