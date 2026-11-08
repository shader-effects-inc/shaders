import {defineShader, p} from "@coreroot/std"
import {rampOver, stops, noiseField} from "@coreroot/std/paint/fields"
import {seededPlane, evolving, signedTone} from "@coreroot/std/paint/noise"
import {noiseColorProps} from "@coreroot/utilities/noiseColor"
import {type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: string
    colorB: string
    stops: ColorStop[] | null
    colorSpace: string
    scale: number
    balance: number
    contrast: number
    seed: number
    speed: number
}

// std generator: 3D MaterialX simplex, z-walked in place. `tone: 'signed'` applies
// contrast/balance to the RAW [-1,1] noise BEFORE the squash to [0,1] (then inverts so the
// gradient runs colorA→colorB as the noise rises) — which is why this shader's contrast/balance
// sliders have different ranges than the shared `noiseToneProps()` and are spelled out below.
export const componentDefinition = defineShader<ComponentProps>({
    name: "SimplexNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Organic noise with animated movement",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        // The shared four-prop color block (colorA/colorB/stops/colorSpace). The tone props below
        // are NOT `noiseToneProps()`: this shader's contrast/balance sliders have different ranges
        // and sit in a different order, and both are visible in the settings panel and the uniform
        // struct layout.
        ...noiseColorProps(),
        scale: {
            default: 2,
            description: 'Pattern scale (higher = larger patterns)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        balance: {
            default: 0,
            description: 'Balance between colors (negative = more colorB, positive = more colorA)',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.1, label: 'Balance', group: 'Effect' }
        },
        contrast: {
            default: 0,
            description: 'Pattern contrast (higher = sharper transitions)',
            ui: { type: ['range', 'map'], min: -2, max: 5, step: 0.1, label: 'Contrast', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: 'Random seed for pattern variation',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        speed: {
            default: 1,
            description: 'Animation speed',
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Speed', group: 'Animation' }
        }
    },
    paint: rampOver(
        signedTone(
            seededPlane(evolving(noiseField('mx3signed'), {rate: 0.5}), {scale: p('scale'), seed: p('seed')}),
            {contrast: p('contrast'), balance: p('balance')},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
