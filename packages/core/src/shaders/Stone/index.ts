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

export const componentDefinition = defineShader<ComponentProps>({
    name: "Stone",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Applies a marbled stone relief and surface distortion to child content",
    requiresRTT: true,
    requiresChild: true,
    props: reliefStylizeProps({ intensity: 0.5, scale: 1, contrast: 0, distortion: 0.15 }),

    // RTT relief filter over the marbled stone height field.
    gpu: {fragment: noiseRelief(reliefBases.stone)}
})

export default componentDefinition
