import {defineShader, p} from "@coreroot/std"
import {rampOver} from "@coreroot/std/paint/fields"
import {rawPlane, scratchStreaks, noiseTone, linearPair} from "@coreroot/std/paint/noise"
import {noiseColorPropsAB, noiseSpeedProp} from "@coreroot/utilities/noiseColor"

export interface ComponentProps {
    colorA: string
    colorB: string
    scale: number
    thickness: number
    seed: number
    speed: number
}

// std generator: fine streaks at `thickness`, flickering on/off with the node's accumulated time
// (evolving in place — the streaks do NOT drift). Scratches carries no tone controls, so the
// shared tone tail just clamps the bright streaks. Fragment-only (the field uses `fwidth`).
export const componentDefinition = defineShader<ComponentProps>({
    name: "Scratches",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Fine hairline scratches, like a worn film or scratched surface",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        ...noiseColorPropsAB(),
        scale: {
            default: 2,
            description: 'Pattern scale (higher = more, finer scratches)',
            ui: { type: 'range', min: -2, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        thickness: {
            default: 1,
            description: 'Thickness of the scratches (higher = bolder streaks)',
            ui: { type: 'range', min: 0.2, max: 5, step: 0.01, label: 'Thickness', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: 'Random seed for pattern variation',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        ...noiseSpeedProp()
    },
    // No tone controls: noiseTone's identity clamp still saturates the bright streaks.
    paint: rampOver(
        noiseTone(rawPlane(scratchStreaks({thickness: p('thickness')}), {scale: p('scale'), seed: p('seed')})),
        linearPair(p('colorA'), p('colorB')),
    )
})

export default componentDefinition
