import {defineShader, p} from "@coreroot/std"
import {gatherStack, chromaSmearTaps, tapeWarp, beatPulse} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    wobble: number
    scanlineNoise: number
    smear: number
    speed: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "VHS",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Analog VHS tape with intermittent tape damage, chroma bleed, and per-scanline noise",
    props: {
        wobble: {
            default: 1,
            description: 'Overall amount of tape damage — waves, creases, and head-switching noise. Bursts on and off organically over time.',
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.01, label: 'Wobble', group: 'Effect' }
        },
        scanlineNoise: {
            default: 0.6,
            description: 'Per-scanline fine chroma/luma jitter. Adds the classic horizontal-streak detail.',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Scanline Noise', group: 'Effect' }
        },
        smear: {
            default: 0.2,
            description: 'Horizontal chroma smear (color bleed) amount. Positive trails color to the right (classic VHS), negative trails it to the left.',
            ui: { type: ['range', 'map'], min: -2, max: 2, step: 0.01, label: 'Smear', group: 'Effect' }
        },
        speed: {
            default: 1,
            description: 'Animation speed of the tape effects.',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.1, label: 'Speed', group: 'Effect' }
        }
    },

    // Analog tape damage — chroma-smeared taps over the tape-warp UVs (burst-gated wobble,
    // per-scanline jitter, crease, head switching), then the AC brightness beat. Driven by the
    // GLOBAL clock (ctx.time × speed — NOT createAnimatedTime).
    effect: gatherStack(
        chromaSmearTaps({
            warp: tapeWarp({wobble: p('wobble'), jitter: p('scanlineNoise'), speed: p('speed')}),
            smear: p('smear'),
        }),
        [beatPulse({wobble: p('wobble'), speed: p('speed')})],
    ),
})

export default componentDefinition
