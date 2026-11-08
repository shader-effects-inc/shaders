import {defineShader, p} from "@coreroot/std"
import {rampOver, stops} from "@coreroot/std/paint/fields"
import {seededPlane, curlSpeed, noiseTone} from "@coreroot/std/paint/noise"
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

// std generator: the magnitude of a divergence-free 2D curl vector — a swirling flow field that
// morphs in place as time walks the curl's z argument.
export const componentDefinition = defineShader<ComponentProps>({
    name: "CurlNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Swirling divergence-free flow field that drifts over time",
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
            seededPlane(curlSpeed({rate: 0.2}), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
