import type {GpuShaderDefinition, GpuFragmentParams} from "@coreroot/gpu/porters"
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {agentSim, renderAgents, force, integrator, agentFrame, shapeField, type ShapeField3} from "@coreroot/std/sim/agents"
import {tgpu, d, sdf, sdf3d, agents, constants} from "@coreroot/gpu/kit"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration, resolveShapeType} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {buildAnalyticSdfFn, createSdfDataTexture, SVG_SDF_SIZE} = sdf
const {createAnalytic3dSdfSetup, createSvg3dSdfSetup, is3dShapeType, isSvg3dShapeType, resolveShapeSubProp, SHAPE3D_DEFAULTS, SVG3D_DEFAULTS, VOLUMETRIC_FIELD_FORMAT} = sdf3d
type ShapeSubPropFrame = sdf3d.ShapeSubPropFrame
type VolumetricFieldSetup = sdf3d.VolumetricFieldSetup

const DEG_TO_RAD = constants.DEG_TO_RAD
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

function parseShapeConfig(raw: unknown): Record<string, unknown> {
    if (raw && typeof raw === 'object') return raw as Record<string, unknown>
    if (typeof raw === 'string') { try { return JSON.parse(raw) as Record<string, unknown> } catch { return {} } }
    return {}
}

// ── Simulation constants ─────────────────────────────────────────────────────────────────────
// A TRUE 3D particle simulation: every particle carries a 3D position + velocity in shape-local
// space (y up, z toward the viewer). Even filling is EMERGENT: particles splat a 3D density grid
// each frame and then descend its gradient (the `force.pressure` part), while a containment spring
// pulls strays back through the shape surface (the `force.containment` bundle) — so at rest they
// spread evenly through the volume, and any disturbance (cursor, shape rotation, animated
// sub-props) scatters them before they settle home again.
const MAX_PARTICLES = 16000
const DGRID = 32 // 3D density grid linear resolution
const DCELLS = DGRID * DGRID * DGRID
const DOMAIN = 1.0 // density grid covers shape-local [-DOMAIN, DOMAIN]³
// Additive render target resolution. The two full-target passes (clear + resolve) are the
// system's dominant FIXED cost, so a coarse-pointer phone/tablet GPU drops to 768 (~45% fewer
// threads + bandwidth) at the cost of slightly softer particles at phone pixel densities.
// Build-time constant (the VOLUMETRIC_FIELD_RES_MOBILE pattern); SSR/tests resolve desktop. Splat
// size is authored in texels of the harness's FIXED SIZE_REF_RES then rescaled to whatever OUT_RES
// actually is — otherwise shrinking OUT_RES on mobile would inflate apparent particle size.
const OUT_RES = resolveRenderRes({desktop: 1024, mobile: 768})
const SIZE_SCALE = OUT_RES / SIZE_REF_RES
// Splat radial reject: a texel contributes while q = dist²/size² < EXTQ, i.e. within √EXTQ·size
// texels of the center. The window is sized per particle inside the render part, so a small
// particle iterates a few texels instead of a fixed block.
const EXTQ = 2.6
const MAX_SPLAT_SIZE = 6 // matches the `size` prop's max — bounds the window at 21×21
const MAX_SPEED = 3
const STATE_FORMAT = 'rgba16float' as const

// All per-frame CPU-derived inputs. Kernels can't read node uniforms (those live in the
// composer's bind groups), so the renderer writes these each frame from getCpuValue.
const SimParams = d.struct({
    colA: d.vec4f, colB: d.vec4f,
    dt: d.f32, time: d.f32, spread: d.f32, agitation: d.f32,
    dragMul: d.f32, gravX: d.f32, gravY: d.f32, halfDepth: d.f32,
    cursorX: d.f32, cursorY: d.f32, cursorForce: d.f32, cursorRadSq: d.f32,
    saRadius: d.f32, saSides: d.f32, saRounding: d.f32, saInnerRatio: d.f32,
    saRotation: d.f32, saHeight: d.f32, saOffset: d.f32, saAperture: d.f32,
    centerX: d.f32, centerYv: d.f32, scale: d.f32, rotC: d.f32, rotS: d.f32, aspect: d.f32,
    size: d.f32, exposure: d.f32, softness: d.f32, speedColorK: d.f32,
    omegaX: d.f32, omegaY: d.f32, omegaZ: d.f32, entrain: d.f32,
    gridOffX: d.f32, gridOffY: d.f32, gridOffZ: d.f32,
})

// One layout for the whole pipeline (every kernel binds the same group; the 3D/SVG shape paths
// chain a second group for their own SDF resources).
const simLayout = tgpu.bindGroupLayout({
    pos: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
    vel: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
    dens: {storage: d.arrayOf(d.atomic(d.u32), DCELLS), access: 'mutable'},
    accumE: {storage: d.arrayOf(d.atomic(d.u32), OUT_RES * OUT_RES), access: 'mutable'},
    accumS: {storage: d.arrayOf(d.atomic(d.u32), OUT_RES * OUT_RES), access: 'mutable'},
    params: {uniform: SimParams},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})
// SVG shape path: the uploaded 2D SDF DATA texture, sampled per particle and extruded in z.
const svgSdfLayout = tgpu.bindGroupLayout({
    sdfSource: {texture: d.texture2d(d.f32)},
})

// ── Spawn (module scope — the seeded ball cloud IS the look) ─────────────────────────────────────────────────────────────────────

/** Spawn: a loose ball cloud around the origin — the first frames converge it into the shape. */
export const particleInitKernel = agents.makeBallCloudInit(simLayout, 'particlesInit')

// ── The composed physics (std/sim force parts) ───────────────────────────────────────────────

/**
 * Gas pressure over the auxiliary 3D density grid — the part owns the grid entirely (per-frame
 * clear + trilinear splat kernels, the trilinear sampler the force differentiates, and the R3
 * sub-cell dither contract on `gridOffX/Y/Z`).
 */
const pressure = force.pressure(simLayout, {
    gridDim: DGRID,
    domain: DOMAIN,
    names: {clear: 'particlesClearDensity', splat: 'particlesDensity'},
})
export const particleClearDensityKernel = pressure.clearKernel
export const particleDensityKernel = pressure.splatKernel

/**
 * Update-kernel factory. `evalSdf` is the baked shape family — 2D analytic extruded in z, an SVG
 * SDF texture extruded in z, or the full analytic 3D setup (rotation + animated sub-props applied
 * inside) — so ONE integrator serves every shape kind. The declared force stack, folded in this
 * order into one kernel by the volume-family integrator (shared semi-implicit Euler + drag +
 * speed clamp): density-gradient pressure (even fill), the containment bundle (surface spring
 * wall / far-field recall / gradient-free homing / rotation entrainment — the four forces that
 * share one SDF evaluation), the cursor magnet (xy cylinder, signed), gravity (screen-down,
 * pre-rotated on the CPU), and the hash-turbulence kick.
 */
export function makeParticleUpdateKernel(evalSdf: ShapeField3) {
    return integrator.forces3d(simLayout, {
        forces: [
            pressure.force,
            force.containment(simLayout, {
                field: evalSdf,
                gradEps: 0.02,
                // A sharp feather packs the rest state into a thin coherent shell at the rim,
                // which then stands in moiré with the density grid; the wide band keeps it diffuse.
                wall: {featherFrom: -0.10, featherTo: 0.01, base: 1.1, springK: 22.0},
                recall: {from: DOMAIN * 0.85, to: DOMAIN * 1.1, k: 9.0},
                homing: {from: 0.12, to: 0.45, k: 5.0},
                entrainment: {featherHalf: 0.05},
            }),
            force.cursorXY(simLayout),
            force.gravity(simLayout),
            force.turbulence(simLayout, {posScale: 193.0, timeX: 31.7, timeY: 27.3, seedScale: 997.0, gain: 5.0}),
        ],
        maxSpeed: MAX_SPEED,
        // The position clamp is a distant last-resort safety net — recall turns particles around
        // long before it.
        posClamp: 1.9,
        name: 'particlesUpdate',
    })
}

/** The render subsystem: z-perspective projected softness-blended profiles, accumulated with a
 *  speed-weighted copy, resolved through the rest→excited ramp with the exposure knob. */
function particlesRender(shape: string, colorSpaceMode: number) {
    return renderAgents.volume(simLayout, {
        shape,
        outRes: OUT_RES,
        maxSplatSize: MAX_SPLAT_SIZE,
        extQ: EXTQ,
        ramp: {colorSpace: colorSpaceMode, exposure: true, trails: 'none'},
        names: {splat: 'particlesSplat', glow: 'particlesGlowProfile', profile: 'particlesProfile', resolve: 'particlesResolve'},
        extraOnSplat: true,
    })
}

export const makeParticleSplatKernel = (shape: string) => particlesRender(shape, 0).splat.kernel as (i: number) => void
export const makeParticleResolveKernel = (mode: number) => particlesRender('dot', mode).resolve.kernel as (x: number, y: number) => void
export const particleSplatKernel = makeParticleSplatKernel('dot')

// ── The shape-family field recipe (composition-time) ─────────────────────────────────────────

interface ShapeFieldSetup {
    evalSdf: ShapeField3
    extraBindGroup: unknown
    setup3d: VolumetricFieldSetup | null
}

/**
 * Bake the shape family's `evalSdf` + any extra bind group it needs: SVG lifted into 3D, a flat
 * SVG outline extruded in z, a true analytic 3D volume, or a flat analytic shape extruded in z
 * (sub-props resolved on the CPU each frame).
 */
function resolveShapeField(params: GpuFragmentParams, shapeSdfUrl: string, shapeType: string): ShapeFieldSetup {
    const {gpu, getCpuValue, onCleanup} = params
    const root = gpu!.root!

    if (shapeSdfUrl && isSvg3dShapeType(shapeType)) {
        // SVG lifted into 3D (extrude): reuse the SVG-3D setup — its sdfFn samples the 2D SDF
        // texture and bevel-extrudes it, and its per-frame `update` drives rotX/rotY/rotZ +
        // depth/bevel through MarchParams, so the extruded solid rotates under the swarm
        // exactly like the analytic 3D shapes.
        const setup3d = createSvg3dSdfSetup(params, shapeSdfUrl, () => getCpuValue('shape'))
        const dummyField = root.createTexture({size: [1, 1], format: VOLUMETRIC_FIELD_FORMAT}).$usage('storage')
        onCleanup(() => dummyField.destroy())
        // The setup's layout is the SVG variant (field + params + sdfSource) behind the shared
        // VolumetricFieldLayout type — the entries object is cast for the extra sdfSource key.
        const extraBindGroup = root.createBindGroup(setup3d.layout, {
            field: dummyField, params: setup3d.marchParamsBuffer.buffer, sdfSource: setup3d.sdfSourceTexture,
        } as never)
        // SVG-lifted field: rotated by MarchParams + footprint-box extended (beyond the 512²
        // footprint the clamped field is flat — zero gradient — or can even read inside under
        // a border-touching feature).
        return {evalSdf: shapeField.svgLifted3d(setup3d.layout, setup3d.sdfFn as ShapeField3), extraBindGroup, setup3d}
    }
    if (shapeSdfUrl) {
        // Flat SVG outline extruded in z (nearest texel read — plenty for containment forces),
        // with the same footprint-box extension as the lifted variant.
        const sdfTex = createSdfDataTexture(params, shapeSdfUrl)
        const extraBindGroup = root.createBindGroup(svgSdfLayout, {sdfSource: sdfTex.texture.texture as never})
        return {evalSdf: shapeField.svgExtruded(simLayout, svgSdfLayout, {size: SVG_SDF_SIZE}), extraBindGroup, setup3d: null}
    }
    if (is3dShapeType(shapeType)) {
        // True 3D volume: reuse the analytic 3D setup (rotation + animated sub-props live in
        // its MarchParams uniform; its `update` runs in our per-frame thunk). The layout's
        // `field` texture entry is satisfied with a dummy — our kernels never march.
        const setup3d = createAnalytic3dSdfSetup(root, shapeType, parseShapeConfig(getCpuValue('shape')), () => getCpuValue('shape'))
        const dummyField = root.createTexture({size: [1, 1], format: VOLUMETRIC_FIELD_FORMAT}).$usage('storage')
        onCleanup(() => dummyField.destroy())
        const extraBindGroup = root.createBindGroup(setup3d.layout, {field: dummyField, params: setup3d.marchParamsBuffer.buffer as never})
        return {evalSdf: shapeField.analytic3d(setup3d.layout, setup3d.sdfFn as ShapeField3), extraBindGroup, setup3d}
    }
    // Flat analytic shape extruded in z (sub-props resolved on the CPU each frame).
    const sdf2d = buildAnalyticSdfFn(shapeType)
    return {evalSdf: shapeField.analyticExtruded(simLayout, sdf2d), extraBindGroup: null, setup3d: null}
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    count: number
    size: number
    particleShape: string
    spread: number
    agitation: number
    damping: number
    gravity: number
    mouseInfluence: number
    mouseRadius: number
    exposure: number
    softness: number
    depth: number
    colorSpace: string
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Particles",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "A swarm of simulated particles that settles into the shape, filling it evenly — flat, SVG or true 3D volumes — scattering from the cursor (or chasing it), tumbling when the shape moves, and always drifting back home",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    props: {
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
        colorA: {
            default: "#8ec5ff",
            transform: transformColor,
            description: "Color of particles at rest",
            ui: {type: 'color', label: 'Rest Color', group: 'Colors'}
        },
        colorB: {
            default: "#ff7ad9",
            transform: transformColor,
            description: "Color particles shift toward when agitated — scattering from the cursor or catching up to a moving shape",
            ui: {type: 'color', label: 'Excited Color', group: 'Colors'}
        },
        count: {
            default: 4000,
            description: "Number of simulated particles",
            ui: {type: 'range', min: 1000, max: MAX_PARTICLES, step: 1000, label: 'Count', group: 'Particles'}
        },
        size: {
            default: 1.5,
            description: "Base particle size (each particle also varies slightly, and grows as it drifts toward the viewer)",
            ui: {type: 'range', min: 0.6, max: 6, step: 0.1, label: 'Size', group: 'Particles'}
        },
        particleShape: {
            default: 'dot',
            compileTime: true,
            description: "What each particle is drawn as — a crisp dot or square (softness still feathers them), or a soft glow puff",
            ui: {type: 'select', options: agents.pointShapeOptions, label: 'Shape', group: 'Particles'}
        },
        spread: {
            default: 1,
            description: "The pressure pushing particles apart — how strongly they insist on even spacing inside the shape",
            ui: {type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Spread', group: 'Physics'}
        },
        agitation: {
            default: 0.12,
            description: "Idle thermal motion of the swarm — 0 freezes it crystal-still once settled",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Agitation', group: 'Physics'}
        },
        damping: {
            default: 0.4,
            description: "How quickly disturbed particles calm back down and settle into place",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Damping', group: 'Physics'}
        },
        gravity: {
            default: 0,
            description: "Downward pull on the swarm — negative floats the particles upward",
            ui: {type: 'range', min: -2, max: 2, step: 0.01, label: 'Gravity', group: 'Physics'}
        },
        mouseInfluence: {
            default: 2,
            description: "Strength of the cursor field. Positive scatters particles away; negative pulls the swarm toward the cursor",
            ui: {type: 'range', min: -5, max: 5, step: 0.01, label: 'Cursor Force', group: 'Interaction'}
        },
        mouseRadius: {
            default: 0.2,
            description: "Reach of the cursor field, in shape-local units",
            ui: {type: 'range', min: 0.05, max: 0.8, step: 0.01, label: 'Cursor Radius', group: 'Interaction'}
        },
        exposure: {
            default: 1,
            description: "Brightness of the additive particle glow",
            ui: {type: 'range', min: 0.2, max: 3, step: 0.01, label: 'Exposure', group: 'Particles'}
        },
        softness: {
            default: 0.1,
            description: "Edge softness of each particle — 0 is a crisp disc, 1 a soft glow puff",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Particles'}
        },
        depth: {
            default: 0.18,
            description: "Thickness of the particle volume behind flat shapes (3D shapes use their own true depth)",
            ui: {type: 'range', min: 0.02, max: 0.6, step: 0.01, label: 'Depth', group: 'Particles'}
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for the rest→excited color ramp',
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

    // The whole simulation as a composed agentSim. Per frame — clear density → trilinear density
    // splat (both owned by `force.pressure`) → integrate (the declared force stack) → additive
    // render splat → ramp resolve (which also clears the accumulators). The shape family (2D
    // analytic / SVG / 3D analytic) is a compile-time branch baking `evalSdf` for the ONE
    // integrator kernel; 3D shapes reuse createAnalytic3dSdfSetup so rotation and CPU-animated
    // sub-props drive the field — and the swarm visibly chases the moving surface. GENERATOR.
    ...agentSim<typeof SimParams>({
        layout: simLayout,
        params: SimParams,
        maxAgents: MAX_PARTICLES,
        // Per-particle cost here is high (the integrator does 7 SDF evaluations and 56 atomic
        // density loads), so a coarse-pointer GPU caps the EFFECTIVE count well below the max.
        countCap: {desktop: MAX_PARTICLES, mobile: 6000},
        output: {key: 'outTex', name: 'particleTexture', size: [OUT_RES, OUT_RES], format: STATE_FORMAT},
        bake: (params) => {
            const {getCpuValue} = params
            const shapeSdfUrl = (getCpuValue('shapeSdfUrl') as string) || ''
            const shapeType = resolveShapeType((getCpuValue('shapeType') as string) || '', getCpuValue('shape'))
            const shapeField = resolveShapeField(params, shapeSdfUrl, shapeType)

            const render = particlesRender(
                (getCpuValue('particleShape') as string) || 'dot',
                (getCpuValue('colorSpace') as number) ?? 0,
            )

            // The four per-particle steps chain the shape family's own SDF bind group (`extra`);
            // the two full-target passes (density clear, resolve) never touch the shape field.
            return {
                pipelines: {
                    init: {kernel: particleInitKernel, threads: 'max', extra: true},
                    clearDensity: {kernel: pressure.clearKernel, threads: 'fixed', size: [pressure.cells]},
                    density: {kernel: pressure.splatKernel, threads: 'agents', extra: true},
                    update: {kernel: makeParticleUpdateKernel(shapeField.evalSdf), threads: 'agents', extra: true},
                    splat: render.splat,
                    resolve: render.resolve,
                },
                initStep: 'init',
                program: ['clearDensity', 'density', 'update', 'splat', 'resolve'],
                extraBindGroup: shapeField.extraBindGroup,
                setup: {shapeField, shapeSdfUrl, shapeType},
            }
        },
        frame: (sys, {getCpuValue}, setupRaw) => {
            const {shapeField, shapeSdfUrl, shapeType} = setupRaw as {shapeField: ShapeFieldSetup; shapeSdfUrl: string; shapeType: string}
            const setup3d = shapeField.setup3d

            // CPU sub-prop resolver for flat shapes (auto-animate / mouse sub-props supported).
            let elapsed = 0
            const springs = new Map<string, {current: number; velocity: number}>()
            const num = (v: unknown, f: number, frame: ShapeSubPropFrame, k: string): number => resolveShapeSubProp(v, f, frame, k)

            let time = 0
            let frameIdx = 0
            let prevRot: {x: number; y: number; z: number} | null = null
            const CELL = (2 * DOMAIN) / DGRID
            const g = makeCpuValueGetter(getCpuValue)

            return (frameParams) => {
                // Square shape-local domain, so an unknown canvas size means aspect 1, not 16/9.
                const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams, 1)
                elapsed += dt
                time += dt
                const frame: ShapeSubPropFrame = {
                    elapsed, deltaTime: dt,
                    pointerX, pointerY,
                    springs,
                }
                const centerView = getCpuValue('center') as {x: number; y: number} | undefined
                const centerX = centerView?.x ?? 0.5
                const centerYv = 1 - (centerView?.y ?? 0.5)
                const scale = Math.max(g('scale', 1), 0.01)
                const rotRad = g('rotation', 0) * DEG_TO_RAD
                const rotC = Math.cos(rotRad)
                const rotS = Math.sin(rotRad)

                // Pointer (screen UV) → shape-local xy (y up) — the shared placement-inversion recipe.
                const cursor = agentFrame.pointerToShapeLocal({
                    pointerX, pointerY, centerX, centerYv, scale, rotC, rotS, aspect,
                })

                // Gravity is screen-down; rotate it into shape space (y up).
                const gravMag = g('gravity', 0) * 1.2
                const gravX = rotS * gravMag
                const gravY = -(rotC * gravMag)

                // Flat-shape sub-props (aliases mirror driveAnalyticSubProps).
                const cfg = parseShapeConfig(getCpuValue('shape'))
                const isSvg3d = isSvg3dShapeType(shapeType)
                const halfDepth = shapeSdfUrl && isSvg3d
                    ? num(cfg.depth, (SVG3D_DEFAULTS.depth ?? 0.25), frame, 'depth') * 0.5
                    : Math.max(g('depth', 0.18), 0.01) * 0.5
                const saRotationV = num(cfg.rotation, 0, frame, 'shapeRotation')

                // Shape-rotation rate → the entrainment omega (particle-space): the euler rates
                // conjugated by Rᵀ (the shared omega recipe — features move at −(Rᵀω)×r when the
                // SDF samples at R·p); flat shapes rotate about z only. Gated to zero when idle.
                let om = {x: 0, y: 0, z: 0}
                if (setup3d) {
                    const d3 = isSvg3dShapeType(shapeType)
                        ? SVG3D_DEFAULTS
                        : SHAPE3D_DEFAULTS[shapeType as keyof typeof SHAPE3D_DEFAULTS] ?? {}
                    const next = {
                        x: num(cfg.rotX, d3.rotX ?? 0, frame, 'rotX') * DEG_TO_RAD,
                        y: num(cfg.rotY, d3.rotY ?? 0, frame, 'rotY') * DEG_TO_RAD,
                        z: num(cfg.rotZ, d3.rotZ ?? 0, frame, 'rotZ') * DEG_TO_RAD,
                    }
                    om = agentFrame.omegaFromRotationDeltas(prevRot, next, dt) ?? om
                    prevRot = next
                } else if (!shapeSdfUrl) {
                    const next = {x: 0, y: 0, z: saRotationV * DEG_TO_RAD}
                    om = agentFrame.omegaFromRotationDeltas(prevRot, next, dt) ?? om
                    prevRot = next
                }
                const {omegaX, omegaY, omegaZ, entrain} = agentFrame.entrainmentFromOmega(om, {cap: 10, gainRate: 2.5, gainMax: 5.0})

                const damping = g('damping', 0.4)
                const mouseRadius = g('mouseRadius', 0.22)
                const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
                const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined

                // R3 low-discrepancy dither — evenly covers the sub-cell cube over frames.
                const gridOff = agentFrame.r3SubCellOffset(frameIdx, CELL)

                sys.writeParams({
                    colA: d.vec4f(colA?.x ?? 1, colA?.y ?? 1, colA?.z ?? 1, colA?.w ?? 1),
                    colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 1, colB?.z ?? 1, colB?.w ?? 1),
                    dt, time,
                    spread: g('spread', 1),
                    agitation: g('agitation', 0.12),
                    dragMul: agentFrame.expDecay(2.2 + damping * 6, dt),
                    gravX, gravY, halfDepth,
                    cursorX: cursor.x, cursorY: cursor.y,
                    cursorForce: g('mouseInfluence', 1.2) * 3.0,
                    cursorRadSq: mouseRadius * mouseRadius,
                    saRadius: num(cfg.radius ?? cfg.width ?? cfg.bottomWidth, 0.35, frame, 'radius'),
                    saSides: num(cfg.sides, 6, frame, 'sides'),
                    saRounding: num(cfg.rounding, 0, frame, 'rounding'),
                    saInnerRatio: num(cfg.innerRatio ?? cfg.thickness ?? cfg.spread ?? cfg.topWidth ?? cfg.topRatio, 0.4, frame, 'innerRatio'),
                    saRotation: saRotationV,
                    saHeight: num(cfg.height, 0.25, frame, 'height'),
                    saOffset: num(cfg.offset ?? cfg.skew, 0.2, frame, 'offset'),
                    saAperture: num(cfg.aperture, 270, frame, 'aperture'),
                    centerX, centerYv, scale, rotC, rotS, aspect,
                    size: g('size', 2.2) * SIZE_SCALE,
                    exposure: g('exposure', 1),
                    softness: Math.min(Math.max(g('softness', 0.5), 0), 1),
                    speedColorK: 2.0 / MAX_SPEED,
                    omegaX, omegaY, omegaZ,
                    entrain,
                    gridOffX: gridOff.x,
                    gridOffY: gridOff.y,
                    gridOffZ: gridOff.z,
                })
                frameIdx++
                setup3d?.update({deltaTime: dt, pointer: {x: pointerX, y: pointerY}})

                return sys.frame({count: sys.resolveCount(g('count', 4000))})
            }
        },
        // Sample the resolved additive field over the full canvas (rgba16f is filterable — a
        // single bilinear tap upsamples cleanly). GPU-free (no device) → transparent.
        fragment: {output: 'particleTexture', fallback: 'transparent'},
    }),
})

export default componentDefinition
