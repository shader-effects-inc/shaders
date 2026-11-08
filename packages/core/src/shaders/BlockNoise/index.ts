import {defineShader, p} from "@coreroot/std"
import {rampOver, stops, noiseField} from "@coreroot/std/paint/fields"
import {seededPlane, evolving, noiseTone} from "@coreroot/std/paint/noise"
import {noiseColorProps, noiseToneProps, noiseSpeedProp} from "@coreroot/utilities/noiseColor"
import type {ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: string
    colorB: string
    stops: ColorStop[] | null
    colorSpace: string
    scale: number
    contrast: number
    balance: number
    seed: number
    speed: number
}

// std generator: identical framing to PerlinNoise, but the soft-celled value basis (Quilez
// permutation) gives the blocky cells. Morphs in place by walking the z (time) axis.
export const componentDefinition = defineShader<ComponentProps>({
    name: "BlockNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Blocky value noise with soft cells that morph over time",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        ...noiseColorProps(),
        scale: {
            default: 2,
            description: 'Pattern scale (higher = larger patterns)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        ...noiseToneProps(),
        ...noiseSpeedProp()
    },
    paint: rampOver(
        noiseTone(
            seededPlane(evolving(noiseField('value'), {rate: 0.3}), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
