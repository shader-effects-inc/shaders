import {defineShader, p} from "@coreroot/std"
import {rampOver, stops} from "@coreroot/std/paint/fields"
import {seededPlane, erosionRidges, unitized, noiseTone} from "@coreroot/std/paint/noise"
import {noiseColorProps, noiseToneProps} from "@coreroot/utilities/noiseColor"
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
}

// std generator: STATIC (no animatedTime) — erosion is built from 2D Perlin-with-derivatives + a
// 2D gully lattice, which has no tractable in-place (3D) evolution, so unlike the other noise
// generators this declares NO `animatedTime`.
export const componentDefinition = defineShader<ComponentProps>({
    name: "ErosionNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Branching, hydraulic-erosion ridges carved into noise",
    acceptsUVContext: true,
    props: {
        ...noiseColorProps(),
        scale: {
            default: 1.5,
            description: 'Pattern scale (higher = larger patterns)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        ...noiseToneProps()
    },
    paint: rampOver(
        noiseTone(
            seededPlane(unitized(erosionRidges()), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
