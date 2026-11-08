import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {
    vec4, ZERO, buildStableFluidsKernels, buildFluidOutputKernel, buildBrushSplatKernel,
} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {fluidSim, cursorRibbon, type FluidSimConfig} from "@coreroot/std/sim/fluids"
import {clamp, local, max, mul} from "@coreroot/std/math"
import {tgpu, d, colorMixing} from "@coreroot/gpu/kit"
import {decayFadeSeconds} from "@coreroot/gpu/kit/host/pointer"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorMode: string
    colorSpeed: number
    color: Parameters<typeof transformColor>[0]
    color1: Parameters<typeof transformColor>[0]
    color2: Parameters<typeof transformColor>[0]
    color3: Parameters<typeof transformColor>[0]
    colorSpace: string
    radius: number
    force: number
    curl: number
    decay: number
    momentum: number
}

// ── Simulation grid ────────────────────────────────────────────────────────────────────────────
// A fixed square grid runs a Stable-Fluids Navier-Stokes solve (Jos Stam / Pavel Dobryakov's
// WebGL-Fluid-Simulation). Velocity (vec4f: xy = flow, z = curl scratch) and RGB dye (vec4f: rgb)
// ping-pong through the classic pass chain: splat → curl → vorticity confinement → divergence →
// pressure Jacobi → gradient subtract → advect velocity → advect dye. 256² matches SmokeFlow's
// proven cost budget for a full 10-iteration pressure solve; the dye is luminous and colorful, so
// it reads crisply even at this resolution once the fragment's linear sampler + bloom curve run.
const N = 256
const COUNT = N * N
const JACOBI_ITERS = 10
const STATE_FORMAT = 'rgba16float' as const

// Splat / feel tuning. IMPULSE_K converts a cursor's UV-per-second speed into a grid-space velocity
// impulse — this velocity-proportional kick is the heart of the fluid feel (a fast flick throws a
// hard jet, a gentle graze barely stirs). MAX_STEPS bounds the per-drag path interpolation so a fast
// stroke lays a continuous ribbon of splats instead of dotted gaps (CursorTrail's technique).
const IMPULSE_K = 0.16
const MAX_STEPS = 16
const CYCLE_PER_SPLAT = 0.006    // color-cycle advance per emitted splat (the trailing rainbow/gradient)
const CYCLE_TIME_RATE = 0.06     // color-cycle drift per second — both scaled by the Color Speed prop
// Motion damping mapping: momentum 0 → velocity fade 4 (dies fast), momentum 1 → 0.05 (flows on).
const VEL_FADE_MAX = 4
const VEL_FADE_MIN = 0.05
// The ink's brightness drives its coverage over the background so thin wisps fade out cleanly.
// Deliberately NO baked bloom — layer a Glow shader on top for luminous looks.
const GAIN = 1.35

// Solve params — written ONCE per frame (dt + the three character knobs the fluid passes read).
const FlowParams = d.struct({
    dt: d.f32, curlStrength: d.f32, velFade: d.f32, dyeFade: d.f32,
})
// Splat params — written PER STAMP inside a thunk (CursorTrail pattern): the pass manager runs the
// thunk then the dispatch in device.queue order, so each stamp sees its own position/velocity/color.
const SplatParams = d.struct({
    posX: d.f32, posY: d.f32, velX: d.f32, velY: d.f32,
    colR: d.f32, colG: d.f32, colB: d.f32,
    col2R: d.f32, col2G: d.f32, col2B: d.f32, mixT: d.f32,
    radius: d.f32, strength: d.f32,
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
// Splat pass: velocity + dye are both mutable here (a splat kicks velocity AND paints dye). Kept in
// its own bind group so it can carry the per-stamp SplatParams uniform independent of the solve's.
const splatLayout = tgpu.bindGroupLayout({
    velA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    sparams: {uniform: SplatParams},
})
const outputLayout = tgpu.bindGroupLayout({
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Splat: the general dye BRUSH splat (paint-over mix — newest color wins — plus an ADDED
//    velocity impulse), with the two brush endpoints blended in the compile-time working color
//    space. Rainbow/Single set mixT 0 (the blend degenerates to endpoint A). One pre-built kernel
//    per space, module-level (the colorMixing "pre-transpiled variants" idiom). ─────────────────────
const makeSplatKernel = (colorSpaceMode: number) =>
    buildBrushSplatKernel(splatLayout, {n: N, namePrefix: 'inkFlow', colorSpace: colorSpaceMode})
const splatKernelVariants = {
    0: makeSplatKernel(0),
    1: makeSplatKernel(1),
    2: makeSplatKernel(2),
    3: makeSplatKernel(3),
    4: makeSplatKernel(4),
    5: makeSplatKernel(5),
} as const
export const splatKernel = splatKernelVariants[2] // oklab default — exported for the resolve gate

// ── Everything else is the shared solver: free-field (clamped, free-slip) Stable Fluids over an RGB
//    dye. Vorticity confinement is what keeps the swirls alive after a stroke ends. ──
const solverKernels = buildStableFluidsKernels(fluidLayout, {
    n: N, namePrefix: 'inkFlow', boundary: 'clamped', dye: 'rgb', dyeDissipation: true,
})
export const {
    curl: curlKernel, vorticity: vorticityKernel, divergence: divergenceKernel, jacobi: jacobiKernel,
    gradSubtract: gradSubtractKernel, advectVel: advectVelKernel, copyVel: copyVelKernel,
    advectDye: advectDyeKernel, copyDye: copyDyeKernel,
} = solverKernels
export const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'inkFlow', dye: 'rgb'})

// ── CPU color helpers (brush color is resolved per splat on the CPU and passed as a uniform, so no
//    HSV maths runs in TGSL). Fragment colors are LINEAR, so linearise the rainbow's sRGB hue. ──────
function srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}
function hueToLinearRgb(hue: number): [number, number, number] {
    const h = ((hue % 1) + 1) % 1
    const i = Math.floor(h * 6)
    const f = h * 6 - i
    const q = 1 - f
    let r = 0
    let g = 0
    let b = 0
    if (i === 0) { r = 1; g = f; b = 0 }
    else if (i === 1) { r = q; g = 1; b = 0 }
    else if (i === 2) { r = 0; g = 1; b = f }
    else if (i === 3) { r = 0; g = q; b = 1 }
    else if (i === 4) { r = f; g = 0; b = 1 }
    else { r = 1; g = 0; b = q }
    return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)]
}
type Rgb3 = [number, number, number]

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "InkFlow",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Interactive",
    description: "Drag to paint swirling ribbons of ink through a real fluid field — eddies keep evolving after you let go",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    props: {
        colorMode: {
            default: 'rainbow',
            description: "How the brush color is chosen: an ever-cycling rainbow, a single fixed color, or a custom three-color cycle",
            ui: {
                type: 'select',
                options: [
                    {label: 'Rainbow', value: 'rainbow'},
                    {label: 'Single Color', value: 'single'},
                    {label: 'Custom Colors', value: 'custom'},
                ],
                label: 'Color Mode',
                group: 'Color'
            }
        },
        colorSpeed: {
            default: 1,
            description: "How quickly the brush cycles through the rainbow or the custom colors as you paint",
            ui: {type: 'range', min: 0, max: 4, step: 0.05, label: 'Color Speed', group: 'Color', condition: {colorMode: ['rainbow', 'custom']}}
        },
        color: {
            default: "#ff2d7e",
            transform: transformColor,
            description: "Brush color used when Color Mode is Single",
            ui: {type: 'color', label: 'Brush Color', group: 'Color', condition: {colorMode: 'single'}}
        },
        color1: {
            default: "#4338ff",
            transform: transformColor,
            description: "First color of the custom brush cycle",
            ui: {type: 'color', label: 'Color 1', group: 'Color', condition: {colorMode: 'custom'}}
        },
        color2: {
            default: "#ff2d7e",
            transform: transformColor,
            description: "Second color of the custom brush cycle",
            ui: {type: 'color', label: 'Color 2', group: 'Color', condition: {colorMode: 'custom'}}
        },
        color3: {
            default: "#19e3ff",
            transform: transformColor,
            description: "Third color of the custom brush cycle",
            ui: {type: 'color', label: 'Color 3', group: 'Color', condition: {colorMode: 'custom'}}
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: "color space the custom colors blend through as the brush cycles between them",
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Color', condition: {colorMode: 'custom'}}
        },
        radius: {
            default: 0.3,
            description: "Size of the ink splat painted under the cursor",
            ui: {type: 'range', min: 0.1, max: 5, step: 0.05, label: 'Radius', group: 'Brush'}
        },
        force: {
            default: 1,
            description: "How hard the cursor's motion pushes the fluid — higher throws faster, more violent jets and curls",
            ui: {type: 'range', min: 0, max: 3, step: 0.05, label: 'Force', group: 'Brush'}
        },
        curl: {
            default: 0,
            description: "Vorticity confinement — swirl energy that spins the ink into persistent eddies and curls",
            ui: {type: 'range', min: 0, max: 60, step: 1, label: 'Swirl', group: 'Dynamics'}
        },
        decay: {
            default: 0.5,
            description: "How fast the ink fades away — low values leave long, lingering trails",
            ui: {type: 'range', min: 0.05, max: 4, step: 0.05, label: 'Decay', group: 'Dynamics'}
        },
        momentum: {
            default: 0.6,
            description: "How long the fluid keeps flowing after a stroke — high momentum lets the motion carry on and on",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Momentum', group: 'Dynamics'}
        }
    },

    // WebGPU compute: Pavel-Dobryakov-style Stable Fluids over an RGB dye field, as the std fluid
    // pipeline. Per frame: emit splats (the `cursorRibbon` part — each stamp a thunk that writes the
    // per-stamp SplatParams then a dispatch, so a fast drag is interpolated into a ribbon of stamps
    // along its path), then the ordered solve: curl → vorticity confinement → divergence →
    // 10× jacobi → gradient-subtract → advect velocity (+copy) → advect dye (+copy) → output.
    // GENERATOR (no child). Idle-skips once the ink has decayed away after the last stroke.
    ...fluidSim((params, root): FluidSimConfig<d.Infer<typeof FlowParams>> => {
        const {getCpuValue, propValues} = params
        // colorSpace is compileTime — the composition picks the matching pre-built splat kernel.
        const spaceMode = (propValues.colorSpace as number) ?? 2
        const activeSplatKernel = splatKernelVariants[spaceMode as keyof typeof splatKernelVariants] ?? splatKernelVariants[2]
        // Splat pass carries its own per-stamp uniform (see SplatParams) on its own bind group.
        const splatU = root.createUniform(SplatParams)

        let hueCursor = 0
        const toRgb = (v: unknown, fallback: Rgb3): Rgb3 => {
            const c = v as {r?: number; g?: number; b?: number} | undefined
            return c && typeof c.r === 'number' ? [c.r, c.g ?? 0, c.b ?? 0] : fallback
        }
        // All endpoints are CPU-preconverted into the compile-time working color space —
        // the splat kernel mixes there and back-converts (canonical gradient-mixing path).
        const conv = (c: Rgb3): Rgb3 => colorMixing.convertP3ToMixSpaceCPU(c[0], c[1], c[2], spaceMode) as Rgb3

        // User splats: a ribbon of stamps interpolated along the drag path (no dotted gaps on a fast
        // flick), each carrying the velocity-proportional impulse that gives the feel. The tracker's
        // default teleport guard is what keeps a pointer jump (entering the canvas, a tab switch)
        // from laying a splat ribbon clean across the field.
        const strokes = cursorRibbon<d.Infer<typeof FlowParams>, {a: Rgb3; b: Rgb3; t: number}>({
            activeWhen: (ptr) => ptr.moving,
            // Idle-skip once the ink has decayed after the last stroke — until then keep evolving
            // (the fluid should keep curling for seconds after you let go).
            fadeSeconds: (f) => decayFadeSeconds(64, f.num('decay', 0.5)),
            // The color cycle drifts with time even between strokes (and on later-skipped frames).
            onFrame: (f) => {
                hueCursor += f.dt * CYCLE_TIME_RATE * f.num('colorSpeed', 1)
            },
            pass: {
                setup: (ctx) => {
                    const splatBg = ctx.root.createBindGroup(splatLayout, {velA: ctx.buffers.velA, dyeA: ctx.buffers.dyeA, sparams: splatU.buffer} as never)
                    return ctx.pass(activeSplatKernel, splatBg)
                },
            },
            ribbon: (f, ptr) => {
                const force = f.num('force', 1)
                // radius is a 10× UI-friendly scale: radius 1 ≡ 10% of the field width.
                const radiusGrid = Math.max(f.num('radius', 0.3) * 0.1 * N, 1)
                const colorSpeed = f.num('colorSpeed', 1)
                const mode = (getCpuValue('colorMode') as string) || 'rainbow'
                const customStops: Rgb3[] | null = mode === 'custom'
                    ? [
                        conv(toRgb(getCpuValue('color1'), [0.26, 0.22, 1])),
                        conv(toRgb(getCpuValue('color2'), [1, 0.18, 0.49])),
                        conv(toRgb(getCpuValue('color3'), [0.1, 0.89, 1])),
                    ]
                    : null

                // Each splat advances the color cycle and resolves two working-space endpoints plus
                // a mix position. The custom cycle runs the closed loop c1→c2→c3→c1 so it never jumps;
                // rainbow/single degenerate to a single endpoint with mixT 0.
                const brushSplat = (): {a: Rgb3; b: Rgb3; t: number} => {
                    hueCursor += CYCLE_PER_SPLAT * colorSpeed
                    if (customStops) {
                        const h = ((hueCursor % 1) + 1) % 1
                        const seg = Math.min(2, Math.floor(h * 3))
                        return {a: customStops[seg], b: customStops[(seg + 1) % 3], t: h * 3 - seg}
                    }
                    if (mode === 'single') {
                        const c = conv(toRgb(getCpuValue('color'), [1, 0.18, 0.49]))
                        return {a: c, b: c, t: 0}
                    }
                    const c = conv(hueToLinearRgb(hueCursor))
                    return {a: c, b: c, t: 0}
                }

                const velX = ptr.velX * N * force * IMPULSE_K
                const velY = ptr.velY * N * force * IMPULSE_K
                return {
                    stepSize: Math.max(0.004, (radiusGrid / N) * 0.6),
                    maxSteps: MAX_STEPS,
                    // The color cycle advances once per stamp at BUILD time, so the ribbon's hues
                    // are laid down in stroke order whether or not the frame is later dispatched.
                    prepare: brushSplat,
                    write: (posX, posY, _t, bc) => splatU.write({
                        posX, posY, velX, velY,
                        colR: bc.a[0], colG: bc.a[1], colB: bc.a[2],
                        col2R: bc.b[0], col2G: bc.b[1], col2B: bc.b[2], mixT: bc.t,
                        radius: radiusGrid, strength: 1,
                    }),
                }
            },
        })

        return {
            resolution: N,
            layout: fluidLayout, outputLayout, paramsSchema: FlowParams,
            solver: {kernels: solverKernels, jacobiIters: JACOBI_ITERS},
            output: {kernel: outputKernel, key: 'dyeTexture'},
            inject: [strokes],
            values: (f) => {
                const momentum = Math.min(Math.max(f.num('momentum', 0.6), 0), 1)
                return {
                    dt: f.dt,
                    curlStrength: f.num('curl', 0),
                    // Momentum is the inverse of velocity fade, mapped exponentially across the range.
                    velFade: VEL_FADE_MAX * Math.pow(VEL_FADE_MIN / VEL_FADE_MAX, momentum),
                    dyeFade: f.num('decay', 0.5),
                }
            },
        }
    }),

    // Fragment: the ink over a TRANSPARENT layer — the dye's brightness (its max channel) drives its
    // alpha coverage, so thin wisps fade out cleanly and whatever sits behind the layer shows
    // through. No baked bloom — layer a Glow shader for luminous looks; stack over a SolidColor for
    // an opaque backdrop.
    gpu: {fragment: ({ctx, computeOutputs}: GpuFragmentParams): Expr => {
        const dyeField = computeOutputs?.dyeTexture as KitTexture | undefined
        if (!dyeField) return ZERO // GPU-free (no device) → transparent.
        const dye = local(dyeField.sample(ctx.uv, 'linearClamp').member('rgb'), 'inkDye')
        const lum = max(dye.member('x'), max(dye.member('y'), dye.member('z')))
        return vec4(dye, clamp(mul(lum, GAIN), 0, 1))
    }}
})

export default componentDefinition
