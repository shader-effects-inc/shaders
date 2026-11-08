import {defineShader, p} from "@coreroot/std"
import {rampOver, stops} from "@coreroot/std/paint/fields"
import {pixelGrid, blueSpeckle, noiseTone} from "@coreroot/std/paint/noise"
import {noiseColorProps, noiseToneProps} from "@coreroot/utilities/noiseColor"
import type {ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: string
    colorB: string
    stops: ColorStop[] | null
    colorSpace: string
    grain: number
    contrast: number
    balance: number
    seed: number
}

// std generator: blue noise is a per-pixel pattern, so it samples the pixel-grid domain —
// `grain`-sized device-pixel cells against the effective viewport — and is static.
export const componentDefinition = defineShader<ComponentProps>({
    name: "BlueNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "High-frequency blue noise — even, grainy speckle ideal for dithering",
    acceptsUVContext: true,
    props: {
        ...noiseColorProps(),
        grain: {
            default: 1,
            description: 'Grain size in pixels (1 = per-pixel, higher = chunkier speckle)',
            ui: { type: 'range', min: 1, max: 32, step: 1, label: 'Grain Size', group: 'Effect' }
        },
        ...noiseToneProps()
    },
    paint: rampOver(
        noiseTone(
            pixelGrid(blueSpeckle(), {grain: p('grain'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
