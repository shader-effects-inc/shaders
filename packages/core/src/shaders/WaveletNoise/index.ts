import {defineShader, p} from "@coreroot/std"
import {rampOver, stops} from "@coreroot/std/paint/fields"
import {seededPlane, waveletBands, unitized, noiseTone} from "@coreroot/std/paint/noise"
import {noiseColorProps, noiseToneProps, noiseSpeedProp} from "@coreroot/utilities/noiseColor"
import type {ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: string
    colorB: string
    stops: ColorStop[] | null
    colorSpace: string
    scale: number
    detail: number
    contrast: number
    balance: number
    seed: number
    speed: number
}

// std generator: rotating banded wavelets whose per-octave frequency ratio is `detail`; the
// node's accumulated time drives the band phase so the ripples animate in place.
export const componentDefinition = defineShader<ComponentProps>({
    name: "WaveletNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Rotating banded wavelets that ripple as they animate",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        ...noiseColorProps(),
        scale: {
            default: 1.5,
            description: 'Pattern scale (higher = larger patterns)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        detail: {
            default: 1.24,
            description: 'Per-octave frequency ratio — higher adds finer wavelet detail',
            ui: { type: 'range', min: 1.05, max: 2, step: 0.01, label: 'Detail', group: 'Effect' }
        },
        ...noiseToneProps(),
        ...noiseSpeedProp()
    },
    paint: rampOver(
        noiseTone(
            seededPlane(unitized(waveletBands({detail: p('detail')})), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
