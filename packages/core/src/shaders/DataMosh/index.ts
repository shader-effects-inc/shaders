import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {call, mixExpr, ZERO, buildMacroblockAdvectKernel} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {feedbackSim, type FeedbackSimConfig} from "@coreroot/std/sim/feedback"
import {blend} from "@coreroot/gpu/kit"

/**
 * DataMosh — corrupted-codec motion smearing. Not a filter: a per-frame FEEDBACK SIMULATION.
 *
 * A fixed-resolution state texture holds "the last decoded frame". Each frame, every macroblock
 * either REFRESHES from the live content (an intact I-frame block) or is HELD: it re-samples its
 * own previous contents displaced by a per-block motion vector — what a video decoder does when
 * P-frame motion vectors keep arriving after the I-frame was dropped. Held blocks drag their
 * pixels across the canvas in liquid smears.
 *
 * There is deliberately NO global clock: every block runs its OWN re-roll timer (a hash-phased
 * epoch), so blocks lock, drift, and recover at independent moments — some regions of the screen
 * clear up while others keep melting, with no synchronized "heartbeat". `churn` sets how often
 * those per-block timers re-roll; a slow spatial noise wave biases which regions are corrupted.
 * Held pixels quantize + decay slightly per generation (fixed, subtle — codec generation loss).
 *
 * Compute: two rgba16float ping-pong state textures + a display copy the fragment samples. The
 * child RTT is late-bound (bindInputs — it only exists after the pass manager allocates
 * boundaries). State is premultiplied, like the RTT it derives from.
 */

const STATE_RES = 768
const STATE_FORMAT = 'rgba16float' as const

export interface ComponentProps {
    intensity: number
    blockSize: number
    drift: number
    churn: number
    blend: number
    speed: number
    seed: number
}

// ── The decode rule: the kit's macroblock hold/refresh advection, at this look's settings. ─────
// The kernel takes the child's size from textureDimensions(src) — never from CPU-side dimensions,
// which can disagree with the actual RTT allocation (editor zoom, DPR, resize timing). Generation
// loss (quantize levels / blend / energy decay) is fixed and subtle — codec generation loss.
const decoder = buildMacroblockAdvectKernel({
    res: STATE_RES,
    format: STATE_FORMAT,
    namePrefix: 'dataMosh',
    blockMix: {coarseScale: 2.6, coarsePick: 0.35},
    generationLoss: {levels: 18, amount: 0.12, decay: 0.9985},
})

/** The composed decode-step kernel (the shape the port gate resolves). */
export const moshKernel = decoder.kernel


export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "DataMosh",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Corrupted-codec motion smearing — macroblocks lock in place and drag their pixels across the frame in liquid trails, each recovering on its own clock, like a video stream that lost its keyframes",
    requiresRTT: true,
    requiresChild: true,
    props: {
        intensity: {
            default: 0.7,
            description: 'How much of the image locks and smears — the fraction of corrupted macroblocks',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Mosh' }
        },
        blockSize: {
            default: 48,
            description: 'Macroblock size — the granularity of the corruption',
            ui: { type: 'range', min: 30, max: 150, step: 1, label: 'Block Size', group: 'Mosh' }
        },
        drift: {
            default: 0.35,
            description: 'Motion vector strength — how fast corrupted areas drag their pixels away',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Drift', group: 'Mosh' }
        },
        churn: {
            default: 0.4,
            description: 'How often each block re-rolls its fate — low values let corrupted regions melt for a long time before recovering',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Churn', group: 'Mosh' }
        },
        blend: {
            default: 1,
            description: 'Blend between the live image and the moshed stream',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Blend', group: 'Mosh' }
        },
        speed: {
            default: 1,
            description: 'Simulation speed. 0 freezes the decay in place.',
            ui: { type: 'range', min: 0, max: 4, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: 'Random seed for the corruption layout',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation' }
        }
    },

    // WebGPU compute: the decoder feedback loop as the std feedback pipeline. Child RTT →
    // (held advection | fresh re-key) → ping-pong state + display copy. The child input binds LATE
    // (bindInputs) since the RTT is allocated after composition; both ping-pong orientations are
    // (re)built by the scaffold. The kernel takes the child's size from textureDimensions(src) —
    // never from CPU dims — and its clock is the sim's OWN speed-scaled `localTime` (no global
    // clock: every macroblock phase-offsets it, so re-rolls never synchronize).
    ...feedbackSim((params, root): FeedbackSimConfig<{
        time: number; dt: number; seed: number
        intensity: number; blockSize: number; drift: number; churn: number
    }, unknown> => {
        const {getCpuValue} = params
        return {
            size: STATE_RES,
            format: STATE_FORMAT,
            speedProp: 'speed',
            paramsSchema: decoder.Params,
            bindGroups: ({childTexture, display, paramsBuffer}, read, write) => root.createBindGroup(decoder.layout, {
                src: childTexture, prev: read.state, next: write.state, display, params: paramsBuffer,
            } as never),
            step: moshKernel,
            values: ({dt, localTime}) => ({
                time: localTime,
                dt,
                seed: (getCpuValue('seed') as number) ?? 0,
                intensity: (getCpuValue('intensity') as number) ?? 0.7,
                blockSize: (getCpuValue('blockSize') as number) ?? 48,
                drift: (getCpuValue('drift') as number) ?? 0.35,
                churn: (getCpuValue('churn') as number) ?? 0.4,
            }),
        }
    }),

    gpu: {
        fragment: ({childNode, ctx, computeOutputs, uniforms, convertToTexture}: GpuFragmentParams): Expr => {
            if (!childNode) return ZERO
            const display = computeOutputs?.display as KitTexture | undefined
            const childTex = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
            const live = childTex.sample(ctx.uv)
            // No compute (GPU-free resolve) → live passthrough.
            if (!display) return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [live])
            const moshed = display.sample(ctx.uv, 'linearClamp')
            const mixed = mixExpr(live, moshed, uniforms.blend)
            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [mixed])
        }
    }
})

export default componentDefinition
