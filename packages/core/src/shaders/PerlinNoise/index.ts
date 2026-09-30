import {defineStd, p} from "@coreroot/std"
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

// std generator: 3D Perlin sampled over the aspect-corrected exp-scale domain, walking the z
// (time) axis so the pattern MORPHS in place rather than sliding.
export const componentDefinition = defineStd<ComponentProps>({
    name: "PerlinNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Smooth gradient noise that morphs over time",
    acceptsUVContext: true,
    // Per-node accumulated time driven by `speed` — the renderer registers + advances the
    // `_animTime` field; the paint reads it as the z-walk phase.
    animatedTime: { speed: 'speed' },
    props: {
        ...noiseColorProps(),
        scale: {
            default: 2,
            description: 'Pattern scale (higher = finer, more detailed patterns; each +1 roughly doubles the frequency)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        ...noiseToneProps(),
        ...noiseSpeedProp()
    },
    paint: rampOver(
        noiseTone(
            seededPlane(evolving(noiseField('perlin'), {rate: 0.3}), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
