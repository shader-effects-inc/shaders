import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    call, vec4, ZERO, buildStableFluidsKernels, buildFluidOutputKernel,
    buildEmitterSplatKernel, makeShapeMaskSet,
} from "@coreroot/gpu/porters"
import {fluidSim, splat, type FluidSimConfig, type FluidSetupCtx, type FluidContainerParts} from "@coreroot/std/sim/fluids"
import {clamp, mul} from "@coreroot/std/math"
import {tgpu, d, colorMixing, effects, sdf, sdf3d} from "@coreroot/gpu/kit"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration, resolveShapeType} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {sdfSpaceUV} = effects.glass
const {createSvgSdfSampler, createSdfDataTexture, createAnalyticSdfSampler, driveAnalyticSubProps, ANALYTIC_SDF_EXTRA_FIELDS} = sdf
const {createVolumetricFieldComputeNode, buildVolumetricFieldSampler, VOLUMETRIC_FIELD_EXTRA_FIELDS} = sdf3d

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

export interface ComponentProps {
    origin: BoundingBoxOrigin
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
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
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

const N = 256
const COUNT = N * N
const JACOBI_ITERS = 10
const VELOCITY_DISSIPATION = 0.2
const STATE_FORMAT = 'rgba16float' as const
const cm = colorMixing

// Fluid + shape params. dt / curlStrength / velFade / dyeFade / colorDecay are the solver's contract
// (scaffolds/fluids). Beyond those: the emitter/cursor fields + the shape-space transform (center/scale/
// rotation[rad]/aspect) + the 8 analytic SDF sub-props (for the mask kernel's buildAnalyticSdfFn call)
// + the volumetric field's aspect-fit domain (for the 3D mask kernel's field-texture sampling).
const FillParams = d.struct({
    dt: d.f32, emitX: d.f32, emitY: d.f32, emitVelX: d.f32, emitVelY: d.f32,
    perpDirX: d.f32, perpDirY: d.f32, spreadFactor: d.f32, emitRad: d.f32, emitIntensity: d.f32,
    dyeFade: d.f32, velFade: d.f32, curlStrength: d.f32, gravity: d.f32, colorDecay: d.f32,
    centerX: d.f32, centerY: d.f32, scale: d.f32, rotation: d.f32, aspect: d.f32,
    cursorX: d.f32, cursorY: d.f32, cursorVelX: d.f32, cursorVelY: d.f32, mouseActive: d.f32, mouseRadSq: d.f32,
    saRadius: d.f32, saSides: d.f32, saRounding: d.f32, saInnerRatio: d.f32, saRotation: d.f32, saHeight: d.f32, saOffset: d.f32, saAperture: d.f32,
    vfOriginX: d.f32, vfOriginY: d.f32, vfSpanX: d.f32, vfSpanY: d.f32, vfActiveRes: d.f32,
})
const fluidLayout = tgpu.bindGroupLayout({
    velA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    velB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    dyeB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
    pressure: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    divergence: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
    maskBuf: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
    params: {uniform: FillParams},
})
const outputLayout = tgpu.bindGroupLayout({
    dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Mask kernels: the general shape-container mask set — per grid cell, evaluate the shape-space
//    transform and one of the three standard distance sources (analytic SDF fn / uploaded SVG SDF
//    texture / pre-marched volumetric field texture), writing (inside, dist) into maskBuf. ─
const maskSet = makeShapeMaskSet(FillParams, {n: N, namePrefix: 'smokeFill'})
export const makeMaskKernel = maskSet.analytic
export const svgMaskKernel = maskSet.svgKernel
export const vfMaskKernel = maskSet.fieldKernel

// ── Splat: the general cone-emitter splat, mask-gated so smoke only emits (and weighs) inside the
//    shape. ──────────────────────
export const splatKernel = buildEmitterSplatKernel(fluidLayout, {
    n: N, namePrefix: 'smokeFill', cone: true, cursorPush: true, masked: true,
    densityGain: 8, velocityBlendGain: 3,
})

// ── The solver: Stable Fluids CONFINED to the shape. `solidMask` makes every cell outside maskBuf a
//    solid wall in the divergence pass and zeroes advected velocity/dye there, so smoke piles against
//    the shape's inner edge instead of leaking past it. ──
const solverKernels = buildStableFluidsKernels(fluidLayout, {
    n: N, namePrefix: 'smokeFill', boundary: 'clamped', dye: 'densityAge',
    dyeDissipation: true, ageAdvance: true, solidMask: true,
})
export const {
    curl: curlKernel, vorticity: vorticityKernel, divergence: divergenceKernel, jacobi: jacobiKernel,
    gradSubtract: gradSubtractKernel, advectVel: advectVelKernel, copyVel: copyVelKernel,
    advectDye: advectDensKernel, copyDye: copyDensKernel,
} = solverKernels
export const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'smokeFill', dye: 'densityAge'})

// ── Container part: confine the fluid to the shape. Owns the triple mask-kernel routing — 3D shapes
//    load the pre-marched volumetric field texture (the pre-march runs ahead of the fluid each frame
//    via `preFrame`, and a sub-millisecond frame still returns it — never drop a pending march);
//    flat SVG shapes (shapeSdfUrl) sample the uploaded SDF DATA texture (bound into the mask bind
//    group, createSvg3dSdfSetup's precedent, async-loaded — a filled circle fallback until the SVG
//    lands); analytic shapes bake the SDF fn. ──────────────────────────────────────────────────────
function shapeContainerField(params: GpuFragmentParams) {
    const {getCpuValue} = params
    const shapeSdfUrl = (getCpuValue('shapeSdfUrl') as string) || ''
    const shapeType = resolveShapeType((getCpuValue('shapeType') as string) || '', getCpuValue('shape'))
    // Volumetric shapes: pre-march the 3D field (null for flat shapes → SVG/analytic mask below).
    const vf = createVolumetricFieldComputeNode(params, () => getCpuValue('shape'), 'none')
    return {
        setup(ctx: FluidSetupCtx): FluidContainerParts {
            const {root} = ctx
            const maskBuf = ctx.buffers.maskBuf as never
            const paramsBuffer = ctx.paramsBuffer as never
            let maskPass
            if (vf) {
                const maskVfBg = root.createBindGroup(maskSet.fieldLayout, {maskBuf, params: paramsBuffer, fieldTex: vf.fieldTexture as never})
                maskPass = ctx.pass(vfMaskKernel, maskVfBg)
            } else if (shapeSdfUrl) {
                const sdfTex = createSdfDataTexture(ctx.params, shapeSdfUrl)
                const maskSvgBg = root.createBindGroup(maskSet.svgLayout, {maskBuf, params: paramsBuffer, sdfSource: sdfTex.texture.texture as never})
                maskPass = ctx.pass(svgMaskKernel, maskSvgBg)
            } else {
                const maskBg = root.createBindGroup(maskSet.analyticLayout, {maskBuf, params: paramsBuffer})
                maskPass = ctx.pass(makeMaskKernel(shapeType), maskBg)
            }
            return {
                maskPass,
                outputs: vf ? vf.outputs : {},
                preFrame: (fp) => vf?.getComputeNodes(fp as never) ?? null,
            }
        },
        /** The field's aspect-fit domain — valid once this frame's pre-march has been collected. */
        domain: () => vf?.getFieldDomain(),
    }
}

/** CPU sub-prop resolver (mirror of driveAnalyticSubProps' JSON→sub-prop mapping) for the mask. */
function makeSubPropResolver(getCpuValue: (key: string) => unknown) {
    const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)
    return () => {
        const raw = getCpuValue('shape')
        let c: Record<string, unknown> = {}
        if (raw && typeof raw === 'object') c = raw as Record<string, unknown>
        else if (typeof raw === 'string') { try { c = JSON.parse(raw) } catch { c = {} } }
        return {
            saRadius: num(c.radius, num(c.width, num(c.bottomWidth, 0.35))),
            saSides: num(c.sides, 6),
            saRounding: num(c.rounding, 0),
            saInnerRatio: num(c.innerRatio, num(c.thickness, num(c.spread, num(c.topWidth, num(c.topRatio, 0.4))))),
            saRotation: num(c.rotation, 0),
            saHeight: num(c.height, 0.25),
            saOffset: num(c.offset, num(c.skew, 0.2)),
            saAperture: num(c.aperture, 270),
        }
    }
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "SmokeFill",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Fill a shape with swirling fluid smoke that interacts with the shape boundary",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    // Analytic SDF sub-props + volumetric field SampleParams for the FRAGMENT's crisp shape mask
    // (driven each frame from the shape JSON / the field pre-march).
    extraFields: {...VOLUMETRIC_FIELD_EXTRA_FIELDS, ...ANALYTIC_SDF_EXTRA_FIELDS},
    props: {
        colorA: {
            default: "#8cf3ff",
            transform: transformColor,
            description: "Color of fresh smoke",
            ui: {type: 'color', label: 'Start Color', group: 'Colors'}
        },
        colorB: {
            default: "#04a0d6",
            transform: transformColor,
            description: "Color smoke transitions to as it ages",
            ui: {type: 'color', label: 'End Color', group: 'Colors'}
        },
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: {type: 'origin', label: 'Origin', group: 'Position'}
        },
        center: {
            default: {x: 0.5, y: 0.5},
            transform: transformPosition,
            description: "Center position of the shape",
            ui: {type: 'position', label: 'Center', group: 'Position', units: ['%', 'px']}
        },
        scale: {
            default: 1,
            description: "Scale of the shape (1 = default size)",
            ui: {type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position'}
        },
        rotation: {
            default: 0,
            description: "Rotation of the shape in degrees",
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position'}
        },
        emitFrom: {
            default: {x: 0.5, y: 0.5},
            transform: transformPosition,
            description: "Emission source point within the shape",
            ui: {type: 'position', label: 'Emit From', group: 'Emission'}
        },
        direction: {
            default: 0,
            description: "Emission direction (0 = up, 90 = right, 180 = down, 270 = left)",
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Direction', group: 'Emission'}
        },
        speed: {
            default: 10,
            description: "Emission velocity strength",
            ui: {type: 'range', min: 0.1, max: 30, step: 0.1, label: 'Speed', group: 'Emission'}
        },
        spread: {
            default: 60,
            description: "Emission cone angle in degrees",
            ui: {type: 'range', min: 0, max: 180, step: 1, label: 'Spread', group: 'Emission'}
        },
        emitRadius: {
            default: 0.03,
            description: "Size of the emission area",
            ui: {type: 'range', min: 0.01, max: 0.3, step: 0.01, label: 'Emit Size', group: 'Emission'}
        },
        intensity: {
            default: 1,
            description: "Smoke emission density",
            ui: {type: 'range', min: 0.1, max: 1.0, step: 0.01, label: 'Intensity', group: 'Effect'}
        },
        dissipation: {
            default: 0.3,
            description: "How fast smoke fades over time",
            ui: {type: 'range', min: 0.1, max: 5.0, step: 0.1, label: 'Dissipation', group: 'Effect'}
        },
        detail: {
            default: 25,
            description: "Fine-scale swirling detail",
            ui: {type: 'range', min: 0, max: 50, step: 1, label: 'Detail', group: 'Effect'}
        },
        gravity: {
            default: 0.5,
            description: "Downward gravitational pull on smoke — 0 = weightless, negative values = smoke rises",
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
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'}
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: {type: 'shape', label: 'Shape', group: 'Shape'}
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type'
        }
    },

    // WebGPU compute: Stable Fluids CONFINED to a shape, as the std fluid pipeline. The
    // `shapeContainerField` part fills a per-cell mask buffer (SDF inside/dist) each frame, then the
    // fluid passes gate emission/advection/pressure by the mask. Ordered multi-dispatch: [volumetric
    // pre-march] → mask → splat → curl → vorticity → divergence → 10× jacobi → gradSub →
    // advect(+copy)×2 → output. GENERATOR (no child).
    ...fluidSim((params): FluidSimConfig<d.Infer<typeof FillParams>> => {
        const {getCpuValue} = params
        const container = shapeContainerField(params)
        const resolveSub = makeSubPropResolver(getCpuValue)

        return {
            resolution: N,
            layout: fluidLayout, outputLayout, paramsSchema: FillParams,
            solver: {kernels: solverKernels, jacobiIters: JACOBI_ITERS},
            output: {kernel: outputKernel, key: 'smokeTexture'},
            pointer: {minDrag: 0.0005},
            container,
            // Emit inside the shape only (the splat kernel gates on the mask).
            inject: [splat({kernel: splatKernel})],
            values: (f) => {
                const speed = f.num('speed', 2.0)
                const dirRad = (f.num('direction', 0) * Math.PI) / 180
                const spreadHalf = Math.min((89 * Math.PI) / 180, (f.num('spread', 60) * Math.PI) / 360)
                const centerView = getCpuValue('center') as {x: number; y: number} | undefined
                const emitView = getCpuValue('emitFrom') as {x: number; y: number} | undefined
                const h = f.frameParams.dimensions?.height ?? 0
                const aspect = h > 0 ? (f.frameParams.dimensions?.width ?? 0) / h : 1
                const velMag = speed * N * 0.15
                const sub = resolveSub()

                const mouseInf = f.num('mouseInfluence', 0.5)
                const mouseRad = f.num('mouseRadius', 0.12)
                const ptr = f.ptr!
                const active = ptr.moving && mouseInf > 0
                const mouseRadGrid = mouseRad * N
                // Read AFTER the container's pre-march (the noun runs `preFrame` before `values`) —
                // the domain then matches the field's contents.
                const dom = container.domain()

                return {
                    dt: f.dt,
                    emitX: (emitView?.x ?? 0.5) * N,
                    emitY: (1 - (emitView?.y ?? 0.5)) * N,
                    emitVelX: Math.sin(dirRad) * velMag,
                    emitVelY: -Math.cos(dirRad) * velMag,
                    perpDirX: Math.cos(dirRad),
                    perpDirY: Math.sin(dirRad),
                    spreadFactor: Math.tan(spreadHalf),
                    emitRad: f.num('emitRadius', 0.03) * N,
                    emitIntensity: f.num('intensity', 0.8),
                    dyeFade: f.num('dissipation', 0.5),
                    velFade: VELOCITY_DISSIPATION,
                    curlStrength: f.num('detail', 30.0),
                    gravity: f.num('gravity', 0.5),
                    colorDecay: f.num('colorDecay', 1.0),
                    centerX: centerView?.x ?? 0.5,
                    centerY: 1 - (centerView?.y ?? 0.5),
                    scale: f.num('scale', 1.0),
                    rotation: (f.num('rotation', 0) * Math.PI) / 180,
                    aspect,
                    cursorX: active ? ptr.x * N : 0,
                    cursorY: active ? ptr.y * N : 0,
                    cursorVelX: active ? ptr.dx * N * 15 * mouseInf : 0,
                    cursorVelY: active ? ptr.dy * N * 15 * mouseInf : 0,
                    mouseActive: active ? 1.0 : 0.0,
                    mouseRadSq: mouseRadGrid * mouseRadGrid,
                    ...sub,
                    vfOriginX: dom?.originX ?? 0,
                    vfOriginY: dom?.originY ?? 0,
                    vfSpanX: dom?.spanX ?? 1,
                    vfSpanY: dom?.spanY ?? 1,
                    vfActiveRes: dom?.activeRes ?? 1,
                }
            },
        }
    }),

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {ctx, computeOutputs, propValues, uniforms, getCpuValue} = params
        const smokeField = computeOutputs?.smokeTexture as KitTexture | undefined
        if (!smokeField) return ZERO // GPU-free (no device) → transparent.

        // Fragment-level crisp shape mask — the Glass/Heatmap sampler routing (volumetric → SVG → analytic).
        let sampler: (uv: Expr) => Expr
        const fieldTex = computeOutputs?.volumetricFieldTexture as KitTexture | undefined
        if (fieldTex) {
            const s = buildVolumetricFieldSampler(fieldTex, {
                originX: uniforms._vfOriginX, originY: uniforms._vfOriginY,
                spanX: uniforms._vfSpanX, spanY: uniforms._vfSpanY,
                activeResF: uniforms._vfActiveRes, rBound: uniforms._vfRBound,
            })
            sampler = s.volumetricFieldSampler
        } else {
            const shapeSdfUrl = (propValues.shapeSdfUrl as string) || ''
            if (shapeSdfUrl) {
                sampler = createSvgSdfSampler(params, shapeSdfUrl)
            } else {
                const st = resolveShapeType((propValues.shapeType as string) || '', getCpuValue('shape'))
                sampler = createAnalyticSdfSampler(st, driveAnalyticSubProps(params, () => getCpuValue('shape')))
            }
        }
        const sdfUV = call(sdfSpaceUV, 'sdfSpaceUV', [uniforms.center, uniforms.scale, uniforms.rotation, ctx.uv, ctx.aspect])
        // Crisp shape edge from the SDF distance (~2px anti-aliased transition).
        const shapeMask = clamp(mul(sampler(sdfUV).member('x'), -(N * 4)), 0, 1)

        const s = smokeField.sample(ctx.uv, 'linearClamp')
        const smokeDensity = s.member('x')
        const smokeAge = s.member('y')
        const colorSpaceMode = (propValues.colorSpace as number) ?? 0
        const variant = cm.mixColorsVariants[colorSpaceMode as keyof typeof cm.mixColorsVariants] ?? cm.mixColorsLinear
        const smokeColor = call(variant, 'mixColors', [uniforms.colorA, uniforms.colorB, smokeAge])
        return vec4(smokeColor.member('rgb'), smokeColor.member('a').mul(smokeDensity).mul(shapeMask))
    }}
})

export default componentDefinition
