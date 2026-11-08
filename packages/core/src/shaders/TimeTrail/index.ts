import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {call, floatE, ZERO, buildFeedbackTrailStep} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {div, dot, local, max, mul, vec3, vec4} from "@coreroot/std/math"
import {feedbackSim, type FeedbackSimConfig} from "@coreroot/std/sim/feedback"
import {blend} from "@coreroot/gpu/kit"
import {isMobileGpuViewport} from "@coreroot/utilities/device"
import {transformColor} from "@coreroot/utilities/transformations"

/**
 * TimeTrail — an Echo-style temporal trail. Not a filter: a per-frame FEEDBACK SIMULATION.
 *
 * A fixed-resolution history texture holds the smeared, decaying accumulation of past child
 * frames. Every frame the loop advects the old history (optional zoom-feedback + drift), softens
 * it with a small tent blur (`diffusion` — what makes the trail smear smoothly instead of
 * ghost-stamping crisp copies), fades it toward zero by a TIME-BASED persistence factor derived
 * from `trailLength`, then stamps the live child frame back on top. The result is a buttery,
 * long-exposure motion trail that follows moving content and converges to invisibility behind a
 * static frame.
 *
 * Decay is time-based (`persistence = exp(-dt / trailLength)`), so the trail's on-screen length
 * is framerate-independent. The current frame is kept perfectly crisp: the fragment composites the
 * full-resolution child RTT with the (blurred, decayed) trail sampled from the history — so the
 * history only ever carries the soft trail, never the sharp live edge the viewer reads as "now".
 *
 * TWO TRAIL SOURCES (compile-time `trailSource`):
 * - Motion (default): a frame-difference mask gates the stamp — only pixels that CHANGED since
 *   last frame enter the history, so the static background stays clean and the ghosts trail the
 *   mover. Works on fully opaque children (images, video), where coverage never changes. The
 *   fragment lays the trail OVER the live frame (the only visible order on opaque content); the
 *   display copy is taken BEFORE the stamp so the mover's leading edge stays crisp.
 * - Alpha: the original coverage-driven stamp (m = 1). Best for a clean shape over transparency —
 *   no threshold to tune, and it trails even under slow sub-threshold movement. The crisp live
 *   frame composites over the trail.
 *
 * Compute: two rgba16float ping-pong state textures + a display copy the fragment samples. The
 * child RTT is late-bound (bindInputs — it only exists after the pass manager allocates
 * boundaries). State is premultiplied, like the RTT it derives from.
 */

// History resolution. Canvas-independent (the trail is a fixed-grid simulation), tiered by device
// class — the kernel cost is STATE_RES² per frame, so a phone GPU runs a coarser grid.
const STATE_RES = isMobileGpuViewport() ? 384 : 640
const STATE_FORMAT = 'rgba16float' as const

// Diffusion: at diffusion=1 the blur reaches this many texels off-centre each frame. Small,
// because it compounds every frame — a large per-frame radius washes the trail out instantly.
// Halved into UV up front: the tent is four hardware-bilinear taps at ±half the offset.
const DIFFUSE_HALF_UV = 2.5 * 0.5 / STATE_RES
// Drift prop (−1..1) → UV per second before the dt scale. Kept modest so a full-tilt drift is a
// readable smear, not an off-screen launch.
const DRIFT_SCALE = 0.5
// Rainbow hue turns per second (the trail cycles hue as it ages — a per-frame rotation that
// accumulates along the trail into a smooth spectrum).
const RAINBOW_RATE = 0.35

export interface ComponentProps {
    trailSource: string
    motionThreshold: number
    trailLength: number
    diffusion: number
    driftX: number
    driftY: number
    zoom: number
    trailOpacity: number
    tintMode: string
    tint: Parameters<typeof transformColor>[0]
    trailBlend: string
    speed: number
}

const TINT_MODES: Record<string, number> = {none: 0, color: 1, rainbow: 2}
const SOURCE_MODES: Record<string, number> = {motion: 1, alpha: 0}

/** The step kernel at this trail's look, per compile-time variant (source × hue-cycle). */
const buildTrailKernel = (source: 'motion' | 'alpha', hueCycle: boolean) => buildFeedbackTrailStep({
    res: STATE_RES,
    format: STATE_FORMAT,
    namePrefix: 'timeTrail',
    source,
    hueCycle,
    diffuseHalfUv: DIFFUSE_HALF_UV,
    driftScale: DRIFT_SCALE,
})

/** The default variant (Motion source, untinted) — the shape the port gate resolves. */
export const timeTrailKernel = buildTrailKernel('motion', false).kernel

/** trailBlend value → kit/blend registry key ('add' is our friendlier alias for linearDodge). */
const resolveBlendKey = (v: unknown): blend.BlendMode => {
    const key = v === 'add' ? 'linearDodge' : String(v ?? 'normal')
    return (key in blend.blendModes ? key : 'normal') as blend.BlendMode
}
/** The kernel's stamp behaviour: screen/add accumulate light INSIDE the history (light-painting —
 *  crossing trails brighten); every other mode stamps plain source-over and the blend is applied
 *  once at the composite (multiply etc. must not compound every frame of the trail's life). */
const stampModeFor = (key: blend.BlendMode): number => (key === 'screen' ? 1 : key === 'linearDodge' ? 2 : 0)

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "TimeTrail",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "An Echo-style temporal trail — whatever moves sheds a smooth, decaying long-exposure ghost trail (frame-difference Motion mode works on opaque images and video; Alpha mode trails a shape's coverage), with optional zoom-feedback tunnels and rainbow light-painting",
    requiresRTT: true,
    requiresChild: true,
    props: {
        trailSource: {
            default: 'motion',
            description: "What sheds the trail — Motion trails whatever changes frame-to-frame (works on opaque images and video), Alpha trails the child's transparency coverage (best for a clean shape over empty canvas)",
            compileTime: true,
            transform: (v: string) => SOURCE_MODES[v] ?? 1,
            ui: {
                type: 'select',
                options: [
                    {label: 'Motion', value: 'motion'},
                    {label: 'Alpha', value: 'alpha'}
                ],
                label: 'Source',
                group: 'Trail'
            }
        },
        motionThreshold: {
            default: 0.06,
            description: 'How much a pixel must change between frames to shed a trail (Motion source only) — lower catches subtle movement, higher only fast motion',
            ui: { type: ['range', 'map'], min: 0.01, max: 0.5, step: 0.01, label: 'Motion Threshold', group: 'Trail' }
        },
        trailLength: {
            default: 0.6,
            description: 'How long the trail lingers, in seconds — the time constant of the fade (the star control)',
            ui: { type: ['range', 'map'], min: 0.05, max: 4, step: 0.01, label: 'Trail Length', group: 'Trail' }
        },
        diffusion: {
            default: 1,
            description: 'How much the trail smears and softens as it fades — 0 keeps crisp ghost frames, higher melts them together',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Diffusion', group: 'Trail' }
        },
        trailOpacity: {
            default: 1,
            description: 'Overall strength of the trail behind the live frame',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Trail Opacity', group: 'Trail' }
        },
        trailBlend: {
            default: 'normal',
            description: 'How the trail combines with the frame — Add / Screen build glowing light-painting trails, Multiply lays down dark ink ghosts, plus the full standard blend set',
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Normal', value: 'normal'},
                    {label: 'Add', value: 'add'},
                    {label: 'Screen', value: 'screen'},
                    {label: 'Multiply', value: 'multiply'},
                    {label: 'Overlay', value: 'overlay'},
                    {label: 'Soft Light', value: 'softLight'},
                    {label: 'Hard Light', value: 'hardLight'},
                    {label: 'Darken', value: 'darken'},
                    {label: 'Lighten', value: 'lighten'},
                    {label: 'Color Dodge', value: 'colorDodge'},
                    {label: 'Color Burn', value: 'colorBurn'},
                    {label: 'Linear Burn', value: 'linearBurn'},
                    {label: 'Difference', value: 'difference'},
                    {label: 'Exclusion', value: 'exclusion'},
                    {label: 'Hue', value: 'hue'},
                    {label: 'Saturation', value: 'saturation'},
                    {label: 'Color', value: 'color'},
                    {label: 'Luminosity', value: 'luminosity'}
                ],
                label: 'Blend',
                group: 'Trail'
            }
        },
        driftX: {
            default: 0,
            description: 'Sideways drift of the trail per second — smears the trail in a direction',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Drift X', group: 'Feedback' }
        },
        driftY: {
            default: 0,
            description: 'Vertical drift of the trail per second',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Drift Y', group: 'Feedback' }
        },
        zoom: {
            default: 1,
            description: 'Per-frame zoom of the trail around the centre — above 1 pulls the trail into an outward tunnel, below 1 sucks it inward',
            ui: { type: ['range', 'map'], min: 0.9, max: 1.1, step: 0.001, label: 'Zoom', group: 'Feedback' }
        },
        tintMode: {
            default: 'none',
            description: 'color the trail — Color tints it a single hue, Rainbow cycles hue as the trail ages',
            compileTime: true,
            transform: (v: string) => TINT_MODES[v] ?? 0,
            ui: {
                type: 'select',
                options: [
                    {label: 'None', value: 'none'},
                    {label: 'Color', value: 'color'},
                    {label: 'Rainbow', value: 'rainbow'}
                ],
                label: 'Tint Mode',
                group: 'Color'
            }
        },
        tint: {
            default: '#33aaff',
            transform: transformColor,
            description: 'Trail tint color (used in Color mode)',
            ui: { type: 'color', label: 'Tint', group: 'Color' }
        },
        speed: {
            default: 1,
            description: 'Simulation speed. 0 freezes the trail in place.',
            ui: { type: 'range', min: 0, max: 4, step: 0.1, label: 'Speed', group: 'Animation' }
        }
    },

    // WebGPU compute: the echo feedback loop as the std feedback pipeline. Child RTT → the kit's
    // feedback-trail step (advect → tent diffuse + decay → frame-diff gate → stamp) → ping-pong
    // state + display copy. The child input binds LATE (bindInputs) since the RTT is allocated
    // after composition; both ping-pong orientations are (re)built by the scaffold.
    ...feedbackSim((params, root): FeedbackSimConfig<{
        persistence: number; diffusion: number; driftX: number; driftY: number; dt: number
        zoom: number; hueRate: number; blendMode: number; motionThreshold: number
    }, unknown> => {
        const {getCpuValue, propValues} = params

        // tintMode / trailBlend / trailSource are compileTime — captured here; a change recomposes → re-runs this hook.
        const tintMode = Number(propValues.tintMode) || 0
        const blendMode = stampModeFor(resolveBlendKey(propValues.trailBlend))
        const sourceMode = propValues.trailSource == null ? 1 : Number(propValues.trailSource)
        const isMotion = sourceMode > 0.5

        const trailSampler = root.createSampler({
            magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
        })

        // Compile-time kernel specialization: the Alpha source drops the prev-live load/store, and
        // an untinted trail drops the HSL round-trip — neither is a runtime branch.
        const trail = buildTrailKernel(isMotion ? 'motion' : 'alpha', tintMode === 2)

        return {
            size: STATE_RES,
            format: STATE_FORMAT,
            // The prev-live pair rides along as an extra lockstep slot: the kernel reads last
            // frame's copy while writing this frame's. Zero-initialised, so the first frame reads
            // as one full-frame stamp that decays away over trailLength. The Alpha kernel never
            // touches them, so they get 1×1 stubs (the bind group still needs a valid resource).
            extraSlots: {prevLive: isMotion ? STATE_RES : [1, 1]},
            speedProp: 'speed',
            paramsSchema: trail.Params,
            bindGroups: ({childTexture, display, paramsBuffer}, read, write) => root.createBindGroup(trail.layout, {
                src: childTexture, prev: read.state, next: write.state, display,
                prevLive: read.extra.prevLive, prevLiveOut: write.extra.prevLive,
                samp: trailSampler, params: paramsBuffer,
            } as never),
            step: trail.kernel,
            // Idle skip, and ONLY this one: at trailOpacity 0 the fragment's blend runs at
            // opacity 0, so the composite returns the live frame whatever the state holds.
            // `speed === 0` is NOT a skip condition — dt 0 freezes decay/drift/hue, but the
            // kernel still resamples the LIVE child every dispatch (and the tent blur is not
            // dt-scaled), and the CPU cannot know whether the child RTT changed. Skipping
            // would freeze the displayed trail over animating content (video, live child).
            skip: () => ((getCpuValue('trailOpacity') as number) ?? 1) === 0,
            values: ({dt}) => {
                const zoom = (getCpuValue('zoom') as number) ?? 1
                const trailLength = Math.max((getCpuValue('trailLength') as number) ?? 0.6, 0.001)
                return {
                    // Time-based fade: the trail loses a fixed FRACTION per unit time, so its length is
                    // framerate-independent. dt=0 (paused) → persistence 1 → frozen.
                    persistence: Math.exp(-dt / trailLength),
                    diffusion: (getCpuValue('diffusion') as number) ?? 0.3,
                    driftX: (getCpuValue('driftX') as number) ?? 0,
                    driftY: (getCpuValue('driftY') as number) ?? 0,
                    dt,
                    // `zoom` is a PER-FRAME divide in the kernel, so it must be raised to a dt-scaled
                    // power here (60 fps is the reference cadence) — otherwise the tunnel speed rides
                    // on the framerate and a frozen trail keeps zooming.
                    zoom: Math.pow(zoom, dt * 60),
                    hueRate: tintMode === 2 ? RAINBOW_RATE * dt : 0,
                    blendMode,
                    motionThreshold: (getCpuValue('motionThreshold') as number) ?? 0.06,
                }
            },
        }
    }),

    gpu: {fragment: ({childNode, ctx, computeOutputs, uniforms, propValues, convertToTexture}: GpuFragmentParams): Expr => {
        if (!childNode) return ZERO
        const display = computeOutputs?.display as KitTexture | undefined
        const childTex = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
        const live = childTex.sample(ctx.uv)
        // No compute (GPU-free resolve) → live passthrough. The RTT sample is premultiplied; the
        // blend pipeline expects straight alpha, so unpremultiply (a no-op at alpha 1).
        if (!display) return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [live])
        const blendKey = resolveBlendKey(propValues.trailBlend)
        const blendFn = blend.blendModes[blendKey]
        const tintMode = Number(propValues.tintMode) || 0
        const sourceMode = propValues.trailSource == null ? 1 : Number(propValues.trailSource)
        let trail = local(display.sample(ctx.uv, 'linearClamp'), 'trail')
        // Color mode (compile-time branch): recolor the whole (premultiplied) trail toward the
        // tint hue, weighted by the trail's luminance — a clean, monochrome light-painting look.
        if (tintMode === 1) {
            const a = max(trail.member('a'), 0.0001)
            const lum = dot(div(trail.member('rgb'), a), vec3(0.299, 0.587, 0.114))
            const colored = mul(uniforms.tint.member('rgb'), mul(lum, 1.5))
            trail = local(vec4(mul(colored, trail.member('a')), trail.member('a')), 'tinted')
        }

        // Composite via the kit blend fns (compile-time selected — trailBlend recomposes). They take
        // STRAIGHT-alpha base/overlay and return premultiplied, so: unpremultiply both inputs, blend,
        // unpremultiply the result for the pipeline. Layer order depends on the trail source: Motion
        // lays the ghost trail OVER the (typically opaque) live frame — the only visible order on
        // opaque content, with trailOpacity as the blend's own opacity; Alpha keeps the crisp live
        // frame on top, with trailOpacity pre-scaled into the trail (the base has no opacity arg).
        const liveStraight = call(blend.unpremultiplyAlpha, 'unpremultiplyLive', [live])
        if (sourceMode === 1) {
            const trailStraight = call(blend.unpremultiplyAlpha, 'unpremultiplyTrail', [trail])
            const blended = call(blendFn, `trailBlend_${blendKey}`, [liveStraight, trailStraight, uniforms.trailOpacity])
            return call(blend.unpremultiplyAlpha, 'unpremultiplyOut', [blended])
        }
        // Premultiplied scale — trailOpacity applied to the trail before the Alpha-order composite.
        const trailScaled = mul(trail, uniforms.trailOpacity)
        const trailStraight = call(blend.unpremultiplyAlpha, 'unpremultiplyTrail', [trailScaled])
        const blended = call(blendFn, `trailBlend_${blendKey}`, [trailStraight, liveStraight, floatE(1)])
        return call(blend.unpremultiplyAlpha, 'unpremultiplyOut', [blended])
    }}
})

export default componentDefinition
