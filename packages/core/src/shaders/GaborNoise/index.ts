import {defineStd, p} from "@coreroot/std"
import {rampOver, stops} from "@coreroot/std/paint/fields"
import {seededPlane, gaborGrains, unitized, noiseTone} from "@coreroot/std/paint/noise"
import {noiseColorProps, noiseToneProps, noiseSpeedProp} from "@coreroot/utilities/noiseColor"
import type {ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: string
    colorB: string
    stops: ColorStop[] | null
    colorSpace: string
    scale: number
    frequency: number
    contrast: number
    balance: number
    seed: number
    speed: number
}

// std generator: oriented sine grains summed at `frequency` waves per cell, their phase driven
// by the node's accumulated time so the ridges flow in place.
export const componentDefinition = defineStd<ComponentProps>({
    name: "GaborNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Oriented sine-grain noise with a fingerprint-like flow",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        ...noiseColorProps(),
        scale: {
            default: 1.5,
            description: 'Pattern scale (higher = finer, more detailed patterns; each +1 roughly doubles the frequency)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        frequency: {
            default: 8,
            description: 'Frequency of the oriented waves within each cell',
            ui: { type: 'range', min: 1, max: 24, step: 0.5, label: 'Frequency', group: 'Effect' }
        },
        ...noiseToneProps(),
        ...noiseSpeedProp()
    },
    paint: rampOver(
        noiseTone(
            seededPlane(unitized(gaborGrains({frequency: p('frequency')})), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
