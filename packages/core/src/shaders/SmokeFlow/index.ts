import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {
    call, vec4, ZERO, buildStableFluidsKernels, buildFluidOutputKernel, buildEmitterSplatKernel,
} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {fluidSim, cursorRibbon, type FluidSimConfig} from "@coreroot/std/sim/fluids"
import {tgpu, d, colorMixing, colorStops as kitColorStops} from "@coreroot/gpu/kit"
import {decayFadeSeconds} from "@coreroot/gpu/kit/host/pointer"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    intensity: number
    emitRadius: number
    momentum: number
    dissipation: number
    detail: number
    gravity: number
    colorDecay: number
    colorSpace: string
}

const N = 256
const COUNT = N * N
const JACOBI_ITERS = 10
const VELOCITY_DISSIPATION = 0.2
// Bounds the per-frame splat ribbon so a full-canvas flick cannot blow out the dispatch count
// (InkFlow / ParticleFlow use the same cap).
const MAX_STEPS = 16
const STATE_FORMAT = 'rgba16float' as const
const cm = colorMixing

// Cursor-driven Stable-Fluids params: the emitter IS the cursor, so the emit* members of the
// splat family's emitter ABI carry the cursor position/velocity, and emitGate carries the
// smoothed-speed emission gate. dt / curlStrength / velFade / dyeFade / colorDecay are the
// solver's contract (scaffolds/fluids).
const FlowParams = d.struct({
    dt: d.f32, emitX: d.f32, emitY: d.f32, emitVelX: d.f32, emitVelY: d.f32, emitGate: d.f32,
    emitRad: d.f32, emitIntensity: d.f32, dyeFade: d.f32, velFade: d.f32, curlStrength: d.f32, gravity: d.f32, colorDecay: d.f32,
})
const fluidLayout = tgpu.bindGroupLayout({
    velA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    velB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    pressure: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    divergence: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    params: {uniform: FlowParams},
})
const outputLayout = tgpu.bindGroupLayout({
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Splat: the general emitter splat in its gated-puff shape — the cursor is the emitter, and the
//    emission gate (smoothed cursor speed) scales the whole puff. ──────────────────────────────
export const splatKernel = buildEmitterSplatKernel(fluidLayout, {
    n: N, namePrefix: 'smokeFlow', cone: false, cursorPush: false, gated: true,
    densityGain: 10, velocityBlendGain: 4,
})

// ── The solver: free-field (clamped, free-slip) Stable Fluids over a density+age dye. ──
const solverKernels = buildStableFluidsKernels(fluidLayout, {
    n: N, namePrefix: 'smokeFlow', boundary: 'clamped', dye: 'densityAge',
    dyeDissipation: true, ageAdvance: true,
})
export const {
    curl: curlKernel, vorticity: vorticityKernel, divergence: divergenceKernel, jacobi: jacobiKernel,
    gradSubtract: gradSubtractKernel, advectVel: advectVelKernel, copyVel: copyVelKernel,
    advectDye: advectDensKernel, copyDye: copyDensKernel,
} = solverKernels
export const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'smokeFlow', dye: 'densityAge'})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "SmokeFlow",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Cursor-driven smoke that lingers, swirls, and dissipates with fluid dynamics",
    usesPointer: true,
    props: {
        colorA: {
            default: "#e29c8b",
            transform: transformColor,
            description: "Color of fresh smoke",
            ui: {type: 'color', label: 'Fresh Color', group: 'Colors'}
        },
        colorB: {
            default: "#d517f9",
            transform: transformColor,
            description: "Color smoke transitions to as it ages",
            ui: {type: 'color', label: 'Aged Color', group: 'Colors'}
        },
        stops: colorStopsPropConfig(),
        intensity: {
            default: 1,
            description: "How much smoke is emitted as you move the cursor",
            ui: {type: 'range', min: 0.1, max: 2.0, step: 0.05, label: 'Intensity', group: 'Emission'}
        },
        emitRadius: {
            default: 0.07,
            description: "Size of smoke puff emitted at cursor",
            ui: {type: 'range', min: 0.01, max: 0.3, step: 0.01, label: 'Puff Size', group: 'Emission'}
        },
        momentum: {
            default: 20,
            description: "How much cursor velocity is transferred into the smoke (higher = more directed flow)",
            ui: {type: 'range', min: 0, max: 50.0, step: 1, label: 'Momentum', group: 'Emission'}
        },
        dissipation: {
            default: 0.5,
            description: "How fast smoke fades over time",
            ui: {type: 'range', min: 0.05, max: 3.0, step: 0.05, label: 'Dissipation', group: 'Effect'}
        },
        detail: {
            default: 10,
            description: "Fine-scale swirling detail driven by vorticity confinement",
            ui: {type: 'range', min: 0, max: 60, step: 1, label: 'Swirl Detail', group: 'Effect'}
        },
        gravity: {
            default: 2,
            description: "Vertical drift — negative floats up, positive sinks down",
            ui: {type: 'range', min: -5, max: 5, step: 0.05, label: 'Gravity', group: 'Effect'}
        },
        colorDecay: {
            default: 0.5,
            description: "How quickly smoke shifts from fresh to aged color",
            ui: {type: 'range', min: 0, max: 3, step: 0.1, label: 'Color Decay', group: 'Colors'}
        },
        colorSpace: {
            default: 'OKLAB',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        }
    },

    // WebGPU compute: cursor-driven Stable Fluids, as the std fluid pipeline. Ordered
    // multi-dispatch: splat ribbon (one splat per interpolated path stamp) → curl → vorticity →
    // divergence → 10× jacobi → gradient-subtract → advect vel/dens (+copies) → output. All params
    // written once per frame; each ribbon stamp rewrites only the cursor position along the path.
    // GENERATOR (no child, no bindInputs). Idle skip once the density field has decayed away.
    ...fluidSim((): FluidSimConfig<d.Infer<typeof FlowParams>> => {
        // The emission gate is the SMOOTHED speed squared, so a slow graze emits a wisp and a
        // flick emits a full puff.
        const speedGateOf = (smoothSpeed: number) => Math.min(smoothSpeed * smoothSpeed * 60, 1.0)

        // Emit a RIBBON of puffs interpolated along this frame's drag path rather than one puff
        // at the current cursor — a fast stroke covers most of the canvas in a single frame, and
        // a single splat leaves dotted gaps. Each stamp carries the full puff strength, so a
        // fast pass deposits the same smoke per unit of path as a slow one.
        const strokes = cursorRibbon<d.Infer<typeof FlowParams>>({
            // Teleport guard DISABLED: emitting smoke along fast strokes is this shader's whole job,
            // and the default 0.25-UV guard classifies a quick flick as a teleport and zeroes it, so
            // the flick emits nothing at all. The pre-refactor behavior (any delta emits, including
            // canvas re-entry) is the intended look (reviewed visually, 2026-08-15).
            tracker: {teleportGuard: Number.POSITIVE_INFINITY},
            activeWhen: (ptr) => speedGateOf(ptr.smoothSpeed) > 0.005,
            fadeSeconds: (f) => decayFadeSeconds(255, f.num('dissipation', 0.4)),
            pass: {kernel: splatKernel},
            ribbon: (f, _ptr, values) => ({
                stepSize: Math.max(0.004, f.num('emitRadius', 0.07) * 0.5),
                maxSteps: MAX_STEPS,
                write: (emitX, emitY) => f.writeParams({...values, emitX, emitY}),
            }),
        })

        return {
            resolution: N,
            layout: fluidLayout, outputLayout, paramsSchema: FlowParams,
            solver: {kernels: solverKernels, jacobiIters: JACOBI_ITERS},
            output: {kernel: outputKernel, key: 'smokeTexture'},
            inject: [strokes],
            // One params struct for the whole frame: the solver passes read dt / the fades / curl
            // from it, and each splat rewrites it with only the cursor position moved along the path.
            values: (f) => {
                const ptr = strokes.ptr()!
                const momentumScale = f.num('momentum', 1.2)
                const speedGate = speedGateOf(ptr.smoothSpeed)
                const isActive = strokes.active()
                const emitRadiusUv = f.num('emitRadius', 0.07)
                return {
                    dt: f.dt,
                    emitX: ptr.x * N,
                    emitY: ptr.y * N,
                    emitGate: isActive ? speedGate : 0.0,
                    emitVelX: ptr.smoothVelX * N * momentumScale * 0.12,
                    emitVelY: ptr.smoothVelY * N * momentumScale * 0.12,
                    emitRad: emitRadiusUv * N,
                    emitIntensity: f.num('intensity', 1.2),
                    dyeFade: f.num('dissipation', 0.4),
                    velFade: VELOCITY_DISSIPATION,
                    curlStrength: f.num('detail', 28.0),
                    gravity: f.num('gravity', -0.3),
                    colorDecay: f.num('colorDecay', 0.5),
                }
            },
        }
    }),

    gpu: {
        fragment: ({ctx, computeOutputs, propValues, uniforms}: GpuFragmentParams): Expr => {
            const smokeField = computeOutputs?.smokeTexture as KitTexture | undefined
            if (!smokeField) return ZERO
            const s = smokeField.sample(ctx.uv, 'linearClamp')
            const smokeDensity = s.member('x')
            const smokeAge = s.member('y')
            const colorSpaceMode = (propValues.colorSpace as number) ?? 0
            const stopCount = (propValues.stopCount as number) ?? 0
            let smokeColor: Expr
            if (stopCount > 1) {
                smokeColor = kitColorStops.mixColorStopsRuntime(
                    smokeAge,
                    {colorsArray: uniforms.colorsArray, positionsArray: uniforms.positionsArray, convertedColorsArray: uniforms.convertedColorsArray, stopCount: uniforms.stopCount},
                    colorSpaceMode,
                )
            } else {
                const variant = cm.mixColorsVariants[colorSpaceMode as keyof typeof cm.mixColorsVariants] ?? cm.mixColorsLinear
                smokeColor = call(variant, 'mixColors', [uniforms.colorA, uniforms.colorB, smokeAge])
            }
            return vec4(smokeColor.member('rgb'), smokeColor.member('a').mul(smokeDensity))
        }
    }
})

export default componentDefinition
