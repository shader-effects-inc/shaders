import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {
    call, vec4, ZERO, buildStableFluidsKernels, buildFluidOutputKernel,
    buildNoiseFieldInitKernel, buildTrigTurbulenceKernel, buildNoiseRestoreKernel,
} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {fluidSim, seededFieldInit, ambientForce, restoreToward, type FluidFrame, type FluidSimConfig} from "@coreroot/std/sim/fluids"
import {tgpu, d, colorMixing} from "@coreroot/gpu/kit"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    seed: number
    speed: number
    turbulence: number
    detail: number
    blending: number
    mouseInfluence: number
    mouseRadius: number
    colorSpace: string
}

const N = 256
const COUNT = N * N
const JACOBI_ITERS = 10
const WARM_JACOBI_ITERS = 6
const WARM_STEPS = 50
const VELOCITY_DISSIPATION = 0.15 // gentler than Smoke — fog persists longer
const INTERNAL_MAX_VEL = N * 0.6
const NOISE_FREQ = 3 // fBm lattice cells across the field — the cloud pattern's base scale
const STATE_FORMAT = 'rgba16float' as const
const cm = colorMixing

// dt / curlStrength / velFade are the solver's contract (scaffolds/fluids); the rest drive Fog's own
// init / force / colorRestore passes.
const FogParams = d.struct({
    dt: d.f32, time: d.f32, seed: d.f32, turbulence: d.f32, curlStrength: d.f32, velFade: d.f32, blending: d.f32,
    cursorX: d.f32, cursorY: d.f32, cursorVelX: d.f32, cursorVelY: d.f32, mouseActive: d.f32, mouseRadSq: d.f32,
})
const fluidLayout = tgpu.bindGroupLayout({
    velA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    velB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    pressure: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    divergence: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    params: {uniform: FogParams},
})
const outputLayout = tgpu.bindGroupLayout({
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Passes (2D-dispatched; toroidal): the general seeded-noise-field kernels. The init seeds
//    density + color-variation from the fBm cloud pattern; the force is the analytic ambient wind
//    plus the cursor shove. ──────────────────────────────────────
export const initKernel = buildNoiseFieldInitKernel(fluidLayout, {n: N, namePrefix: 'fog', frequency: NOISE_FREQ})
export const forceKernel = buildTrigTurbulenceKernel(fluidLayout, {n: N, namePrefix: 'fog'})

// ── The solver: TOROIDAL (seamless, no walls) Stable Fluids over a permanent density+color-variation
//    field — no dye dissipation and no ageing, so the fog neither fades nor drifts to one color.
//    The velocity cap is what keeps a never-dissipating field from accumulating enough energy over
//    minutes of runtime to advect more than a cell per step. ──
const solverKernels = buildStableFluidsKernels(fluidLayout, {
    n: N, namePrefix: 'fog', boundary: 'toroidal', dye: 'densityAge', velocityCap: INTERNAL_MAX_VEL,
})
export const {
    curl: curlKernel, vorticity: vorticityKernel, divergence: divergenceKernel, jacobi: jacobiKernel,
    gradSubtract: gradSubtractKernel, advectVel: advectVelKernel, copyVel: copyVelKernel,
    advectDye: advectDensKernel, copyDye: copyDensKernel,
} = solverKernels

// Counteract numerical diffusion: gently blend the color-variation channel back toward the seed
// pattern (blending controls oil-vs-water: low = distinct colors, high = fully mixed).
export const colorRestoreKernel = buildNoiseRestoreKernel(fluidLayout, {n: N, namePrefix: 'fog', frequency: NOISE_FREQ})

export const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'fog', dye: 'densityAge'})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Fog",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Fog that fills the screen and interacts with the mouse",
    usesPointer: true,
    props: {
        colorA: {
            default: "#e0e0e0",
            transform: transformColor,
            description: "Primary fog color",
            ui: {type: 'color', label: 'Color A', group: 'Colors'}
        },
        colorB: {
            default: "#888888",
            transform: transformColor,
            description: "Secondary fog color — creates variation across the field",
            ui: {type: 'color', label: 'Color B', group: 'Colors'}
        },
        seed: {
            default: 0,
            description: "Deterministic starting pattern — different seeds produce different fog configurations",
            ui: {type: 'range', min: 0, max: 999, step: 1, label: 'Seed', group: 'Field'}
        },
        speed: {
            default: 1.0,
            description: "Simulation speed multiplier",
            ui: {type: 'range', min: 0.1, max: 3.0, step: 0.1, label: 'Speed', group: 'Field'}
        },
        turbulence: {
            default: 1,
            description: "Ambient motion strength",
            ui: {type: 'range', min: 0, max: 3.0, step: 0.01, label: 'Turbulence', group: 'Field'}
        },
        detail: {
            default: 15,
            description: "Fine-scale swirling structure — higher values produce more intricate wisps and vortices",
            ui: {type: 'range', min: 0, max: 50, step: 1, label: 'Detail', group: 'Field'}
        },
        blending: {
            default: 0.3,
            description: "How much the two colors blend together — 0 behaves like oil & water (colors stay distinct with sharp boundaries), 1 behaves like food coloring (colors fully mix)",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Blending', group: 'Field'}
        },
        mouseInfluence: {
            default: 0.1,
            description: "Strength of cursor influence — move the cursor to push fog",
            ui: {type: 'range', min: 0, max: 2, step: 0.01, label: 'Cursor Influence', group: 'Interaction'}
        },
        mouseRadius: {
            default: 0.1,
            description: "Radius of cursor influence area",
            ui: {type: 'range', min: 0.02, max: 0.5, step: 0.01, label: 'Cursor Radius', group: 'Interaction'}
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'}
        }
    },

    // WebGPU compute: toroidal Stable Fluids (permanent fog), as the std fluid pipeline. On the
    // first frame (or seed change), `seededFieldInit` seeds the field from fBm noise and runs a
    // 50-step silent warm-up (thunks advance the sim time between dispatches). Every solve wraps
    // the shared chain in Fog's own stages: ambient turbulence force before, color restoration
    // after (numerical diffusion would otherwise wash the two colors into one).
    // Per frame: force → curl → vorticity → divergence → 10× jacobi → gradient-subtract → advect(+copy)×2
    // → colorRestore → output. GENERATOR (no child, no bindInputs).
    ...fluidSim((): FluidSimConfig<d.Infer<typeof FogParams>> => {
        // The warm-up and the per-frame write share this params shape; only dt/time/cursor differ.
        const fieldValues = (f: FluidFrame) => ({
            seed: f.num('seed', 0),
            turbulence: f.num('turbulence', 1.0),
            curlStrength: f.num('detail', 15.0),
            velFade: VELOCITY_DISSIPATION,
            blending: f.num('blending', 0.3),
        })
        return {
            resolution: N,
            layout: fluidLayout, outputLayout, paramsSchema: FogParams,
            solver: {kernels: solverKernels, jacobiIters: JACOBI_ITERS},
            output: {kernel: outputKernel, key: 'fogTexture'},
            write: 'thunk',
            pointer: {minDrag: 0.0005},
            init: seededFieldInit({
                kernel: initKernel,
                seed: (f) => f.num('seed', 0),
                warm: {
                    steps: WARM_STEPS, jacobiIters: WARM_JACOBI_ITERS, dt: 0.1,
                    startTime: () => Math.random() * 60,
                    values: (t, f) => ({
                        dt: 0.1, time: t, ...fieldValues(f),
                        cursorX: 0, cursorY: 0, cursorVelX: 0, cursorVelY: 0, mouseActive: 0, mouseRadSq: 1,
                    }),
                },
            }),
            solve: [ambientForce(forceKernel), restoreToward(colorRestoreKernel)],
            values: (f) => {
                const mouseInf = f.num('mouseInfluence', 0.1)
                const mouseRadGrid = f.num('mouseRadius', 0.1) * N
                const ptr = f.ptr!
                const active = ptr.moving && mouseInf > 0
                return {
                    dt: f.dt * f.num('speed', 1.0), time: f.elapsed, ...fieldValues(f),
                    cursorX: active ? ptr.x * N : 0,
                    cursorY: active ? ptr.y * N : 0,
                    cursorVelX: active ? ptr.dx * N * 15 * mouseInf : 0,
                    cursorVelY: active ? ptr.dy * N * 15 * mouseInf : 0,
                    mouseActive: active ? 1.0 : 0.0,
                    mouseRadSq: mouseRadGrid * mouseRadGrid,
                }
            },
        }
    }),

    gpu: {fragment: ({ctx, computeOutputs, propValues, uniforms}: GpuFragmentParams): Expr => {
        const fogField = computeOutputs?.fogTexture as KitTexture | undefined
        if (!fogField) return ZERO // GPU-free / no device → transparent.
        const s = fogField.sample(ctx.uv, 'linearClamp')
        const fogDensity = s.member('x')
        const colorVar = s.member('y')
        const colorSpaceMode = (propValues.colorSpace as number) ?? 0
        const variant = cm.mixColorsVariants[colorSpaceMode as keyof typeof cm.mixColorsVariants] ?? cm.mixColorsLinear
        const fogColor = call(variant, 'mixColors', [uniforms.colorA, uniforms.colorB, colorVar])
        return vec4(fogColor.member('rgb'), fogColor.member('a').mul(fogDensity))
    }}
})

export default componentDefinition
