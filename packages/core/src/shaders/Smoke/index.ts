import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {
    call, vec4, ZERO, buildStableFluidsKernels, buildFluidOutputKernel, buildEmitterSplatKernel,
} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {fluidSim, splat, type FluidSimConfig} from "@coreroot/std/sim/fluids"
import {tgpu, d, colorMixing, colorStops as kitColorStops} from "@coreroot/gpu/kit"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    emitFrom: Parameters<typeof transformPosition>[0]
    direction: number
    speed: number
    spread: number
    emitRadius: number
    intensity: number
    dissipation: number
    detail: number
    gravity: number
    colorDecay: number
    mouseInfluence: number
    mouseRadius: number
    colorSpace: string
}

const N = 256
const COUNT = N * N
// 10 iterations/frame: the solve is warm-started (pressure decays instead of resetting), so
// convergence accumulates across frames for these quasi-steady flows.
const JACOBI_ITERS = 10
const VELOCITY_DISSIPATION = 0.2
const STATE_FORMAT = 'rgba16float' as const
const cm = colorMixing

// All per-frame CPU-derived inputs. Compute kernels can't read node uniforms, so the renderer
// writes these each frame (Stable-Fluids Stam/Dobryakov architecture). dt / curlStrength / velFade /
// dyeFade / colorDecay are the solver's contract (scaffolds/fluids); the rest drive the splat.
const FluidParams = d.struct({
    dt: d.f32, emitX: d.f32, emitY: d.f32, emitVelX: d.f32, emitVelY: d.f32,
    perpDirX: d.f32, perpDirY: d.f32, spreadFactor: d.f32, emitRad: d.f32, emitIntensity: d.f32,
    dyeFade: d.f32, velFade: d.f32, curlStrength: d.f32, gravity: d.f32, cursorX: d.f32, cursorY: d.f32,
    cursorVelX: d.f32, cursorVelY: d.f32, mouseActive: d.f32, mouseRadSq: d.f32, colorDecay: d.f32,
})
const fluidLayout = tgpu.bindGroupLayout({
    velA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    velB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    pressure: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    divergence: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    params: {uniform: FluidParams},
})
const outputLayout = tgpu.bindGroupLayout({
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── The one pass that is Smoke's own: the general cone-emitter splat — density at a fixed source,
//    the injected velocity fanned across the emission cone, gravity, and the cursor shove. ──
export const splatKernel = buildEmitterSplatKernel(fluidLayout, {
    n: N, namePrefix: 'smoke', cone: true, cursorPush: true, densityGain: 8, velocityBlendGain: 3,
})

// ── Everything else is the shared solver: free-field (clamped, free-slip) Stable Fluids over a
//    density+age dye that dissipates and ages. ──
const solverKernels = buildStableFluidsKernels(fluidLayout, {
    n: N, namePrefix: 'smoke', boundary: 'clamped', dye: 'densityAge',
    dyeDissipation: true, ageAdvance: true,
})
export const {
    curl: curlKernel, vorticity: vorticityKernel, divergence: divergenceKernel, jacobi: jacobiKernel,
    gradSubtract: gradSubtractKernel, advectVel: advectVelKernel, copyVel: copyVelKernel,
    advectDye: advectDensKernel, copyDye: copyDensKernel,
} = solverKernels
export const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'smoke', dye: 'densityAge'})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Smoke",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Realistic fluid smoke simulation with vorticity dynamics",
    usesPointer: true,
    props: {
        colorA: {
            default: "#fc83f9",
            transform: transformColor,
            description: "Color of fresh smoke",
            ui: {type: 'color', label: 'Start Color', group: 'Colors'}
        },
        colorB: {
            default: "#c21c79",
            transform: transformColor,
            description: "Color smoke transitions to as it ages",
            ui: {type: 'color', label: 'End Color', group: 'Colors'}
        },
        stops: colorStopsPropConfig(),
        emitFrom: {
            default: {x: 0.5, y: 1},
            transform: transformPosition,
            description: "The emission source point",
            ui: {type: 'position', label: 'Emit From', group: 'Emission'}
        },
        direction: {
            default: 0,
            description: "Emission direction (0 = up, 90 = right, 180 = down, 270 = left)",
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Direction', group: 'Emission'}
        },
        speed: {
            default: 20,
            description: "Emission velocity strength",
            ui: {type: 'range', min: 0.1, max: 50, step: 0.1, label: 'Speed', group: 'Emission'}
        },
        spread: {
            default: 60,
            description: "Emission cone angle in degrees",
            ui: {type: 'range', min: 0, max: 180, step: 1, label: 'Spread', group: 'Emission'}
        },
        emitRadius: {
            default: 0.08,
            description: "Size of the emission area",
            ui: {type: 'range', min: 0.01, max: 0.3, step: 0.01, label: 'Emit Size', group: 'Emission'}
        },
        intensity: {
            default: 1,
            description: "Smoke emission density",
            ui: {type: 'range', min: 0.1, max: 1.0, step: 0.01, label: 'Intensity', group: 'Effect'}
        },
        dissipation: {
            default: 0.2,
            description: "How fast smoke fades over time",
            ui: {type: 'range', min: 0.1, max: 3.0, step: 0.1, label: 'Dissipation', group: 'Effect'}
        },
        detail: {
            default: 25,
            description: "Fine-scale swirling detail",
            ui: {type: 'range', min: 0, max: 50, step: 1, label: 'Detail', group: 'Effect'}
        },
        gravity: {
            default: 0.5,
            description: "Downward gravitational pull on smoke",
            ui: {type: 'range', min: -2, max: 2, step: 0.1, label: 'Gravity', group: 'Effect'}
        },
        colorDecay: {
            default: 0.4,
            description: "How quickly smoke shifts from Color A to Color B",
            ui: {type: 'range', min: 0, max: 3, step: 0.1, label: 'Color Decay', group: 'Colors'}
        },
        mouseInfluence: {
            default: 0.1,
            description: "Strength of cursor influence",
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
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        }
    },

    // WebGPU compute: Stable Fluids (Stam/Dobryakov), as the std fluid pipeline. The whole solve is
    // an ORDERED multi-dispatch: splat → curl → vorticity → divergence → 10× jacobi →
    // gradient-subtract → advect vel (+copy) → advect dens (+copy) → output. All params written ONCE
    // per frame (no per-dispatch writes → no thunks). GENERATOR (no child, no bindInputs).
    ...fluidSim((): FluidSimConfig<d.Infer<typeof FluidParams>> => ({
        resolution: N,
        layout: fluidLayout, outputLayout, paramsSchema: FluidParams,
        solver: {kernels: solverKernels, jacobiIters: JACOBI_ITERS},
        output: {kernel: outputKernel, key: 'smokeTexture'},
        pointer: {minDrag: 0.0005},
        // The one pass that is Smoke's own: the fixed-source emission cone (+ gravity + cursor shove).
        inject: [splat({kernel: splatKernel})],
        values: (f) => {
            const speed = f.num('speed', 3.0)
            const dirRad = (f.num('direction', 0) * Math.PI) / 180
            const spreadHalf = Math.min((89 * Math.PI) / 180, (f.num('spread', 45) * Math.PI) / 360)
            const emitPos = f.getCpuValue('emitFrom') as {x: number; y: number} | undefined
            const epx = emitPos?.x ?? 0.5
            const epy = emitPos?.y ?? 0.0 // post-transform y (= 1 - authored y)
            const velMag = speed * N * 0.15

            const mouseInf = f.num('mouseInfluence', 0.5)
            const mouseRad = f.num('mouseRadius', 0.12)
            const ptr = f.ptr!
            const mouseRadGrid = mouseRad * N
            const active = ptr.moving && mouseInf > 0

            return {
                dt: f.dt,
                emitX: epx * N,
                emitY: (1 - epy) * N,
                emitVelX: Math.sin(dirRad) * velMag,
                emitVelY: -Math.cos(dirRad) * velMag,
                perpDirX: Math.cos(dirRad),
                perpDirY: Math.sin(dirRad),
                spreadFactor: Math.tan(spreadHalf),
                emitRad: f.num('emitRadius', 0.06) * N,
                emitIntensity: f.num('intensity', 0.7),
                dyeFade: f.num('dissipation', 1.0),
                velFade: VELOCITY_DISSIPATION,
                curlStrength: f.num('detail', 30.0),
                gravity: f.num('gravity', 0.5),
                colorDecay: f.num('colorDecay', 1.0),
                cursorX: active ? ptr.x * N : 0,
                cursorY: active ? ptr.y * N : 0,
                cursorVelX: active ? ptr.dx * N * 15 * mouseInf : 0,
                cursorVelY: active ? ptr.dy * N * 15 * mouseInf : 0,
                mouseActive: active ? 1.0 : 0.0,
                mouseRadSq: mouseRadGrid * mouseRadGrid,
            }
        },
    })),

    gpu: {fragment: ({ctx, computeOutputs, propValues, uniforms}: GpuFragmentParams): Expr => {
        const smokeField = computeOutputs?.smokeTexture as KitTexture | undefined
        if (!smokeField) return ZERO // GPU-free (no device) → transparent.
        // The fp16 storage texture is hardware-bilinear filtered — one tap. x=density, y=age.
        const s = smokeField.sample(ctx.uv, 'linearClamp')
        const smokeDensity = s.member('x')
        const smokeAge = s.member('y')

        const colorSpaceMode = (propValues.colorSpace as number) ?? 0
        const stopCount = (propValues.stopCount as number) ?? 0
        let smokeColor: Expr
        if (stopCount > 1) {
            smokeColor = kitColorStops.mixColorStopsRuntime(
                smokeAge,
                {
                    colorsArray: uniforms.colorsArray,
                    positionsArray: uniforms.positionsArray,
                    convertedColorsArray: uniforms.convertedColorsArray,
                    stopCount: uniforms.stopCount,
                },
                colorSpaceMode,
            )
        } else {
            const variant = cm.mixColorsVariants[colorSpaceMode as keyof typeof cm.mixColorsVariants] ?? cm.mixColorsLinear
            smokeColor = call(variant, 'mixColors', [uniforms.colorA, uniforms.colorB, smokeAge])
        }
        return vec4(smokeColor.member('rgb'), smokeColor.member('a').mul(smokeDensity))
    }}
})

export default componentDefinition
