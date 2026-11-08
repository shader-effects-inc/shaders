import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {
    createGuardedCompute, createStateBuffer, type ComputeStep, buildStableFluidsKernels,
    createStableFluidsPasses, buildVelocityImpulseKernel, createFluidKernelPass, resolveRenderRes, SIZE_REF_RES,
} from "@coreroot/gpu/porters"
import {defineShader, recompileWhen} from "@coreroot/std"
import {agentSim, renderAgents, integrator} from "@coreroot/std/sim/agents"
import {tgpu, d, agents} from "@coreroot/gpu/kit"
import {createPointerVelocityTracker, createIdleGate, pathStampRibbon} from "@coreroot/gpu/kit/host/pointer"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

// ── "Dust in the wind" ───────────────────────────────────────────────────────────────────────
// Thousands of particles advected by a real incompressible fluid velocity field that the cursor
// stirs. Two decoupled sim layers, connected by a single velocity texture:
//
//   1. VELOCITY FIELD — InkFlow's Stable-Fluids solver minus the dye (the shared solver kernel
//      set). A 256² grid ping-pongs through the classic chain (cursor-drag force splats → curl →
//      vorticity confinement → divergence → 10× Jacobi pressure → gradient subtract → advect
//      velocity). Because the field is projected divergence-free, an initially-uniform particle
//      cloud STAYS roughly uniform — no respawn hacks. An ambient curl-noise breeze is folded
//      into the vorticity pass so the scene keeps breathing even with no cursor input (the noise
//      is itself divergence-free, so the projection preserves it). The solved velocity is written
//      to `velTex` in the copyVel pass — no extra full pass.
//   2. PARTICLES + RENDER — the advection-family integrator + the oriented-world render variant:
//      N particles in screen-proportional world space bilinear-sample `velTex`, ease toward the
//      field (inertia → massy drift), get a weak R2 home spring (even coverage self-restores —
//      no wrap), and are splatted as velocity-oriented comet shapes colored rest→excited by
//      LOCAL SPEED, blended into a persistent trail canvas.
//
// The two layers can't share one kernel (the integrate step needs both the field and the particle
// buffer), so they live in two bind-group layouts bridged by `velTex` — every kernel references
// exactly ONE layout, and neither layout approaches the storage-buffer limit.

const N = 256                    // fluid grid resolution (matches InkFlow's proven cost budget)
const FLUID_COUNT = N * N
// Pressure projection is the fluid solve's dominant cost (one full-grid pass per iteration); mobile
// trades a little incompressibility for the frame time. The grid itself stays 256² — halving it
// would change the flow's character, the iteration count only softens the projection.
const JACOBI_ITERS = resolveRenderRes({desktop: 10, mobile: 6})
const VEL_FORMAT = 'rgba16float' as const

// Square render/accumulator resolution; the fragment bilinear-upsamples it over the canvas. Mobile
// halves it, which quarters the resolve pass, the two accumulators and the trail canvas. Dust
// particles are sized against the harness's FIXED SIZE_REF_RES, not this live value.
const RES = resolveRenderRes({desktop: 1024, mobile: 512})
const STATE_FORMAT = 'rgba16float' as const
const MAX_PARTICLES = 16384      // static state-buffer bound; runtime `count` is the dispatch size
const COUNT_MAX_MOBILE = 6000    // effective cap on mobile GPUs (the prop range stays the same)

// Sanity cap on a particle's body radius in render texels — the splat window itself is DYNAMIC
// (sized to the shape's real accept radius per axis), so this only guards against an absurd size.
const SPLAT_R = 14

// Cursor force feel. IMPULSE_K converts a cursor's UV-per-second speed into a grid-space velocity
// impulse (the heart of the fluid feel — a fast flick throws a hard jet). MAX_STEPS bounds the
// per-drag ribbon so a fast stroke lays a continuous wake instead of dotted gaps.
const IMPULSE_K = 0.16
const MAX_STEPS = 16
const FORCE_RADIUS_UV = 0.16                  // cursor plow width as a fraction of the field
const FORCE_RADIUS_GRID = FORCE_RADIUS_UV * N

// Momentum → velocity fade (inverse, exponential across the range): momentum 0 → fade 4 (eddies die
// in ~1s), momentum 1 → 0.05 (the flow carries on and on). Same mapping as InkFlow.
const VEL_FADE_MAX = 4
const VEL_FADE_MIN = 0.05

// Ambient breeze: a slow, in-place-morphing curl-noise force added to the field each frame. AMBIENT_K
// scales the `ambient` prop (0–1) to a grid-velocity kick; AMBIENT_FREQ sets how many eddies span the
// field; AMBIENT_DRIFT is how fast the pattern evolves (the third noise axis).
const AMBIENT_K = 26.0
const AMBIENT_FREQ = 2.4
const AMBIENT_DRIFT = 0.09

// Particle advection feel. INERTIA_RATE is the rate (1/s) a particle's velocity eases toward the
// field's — lower feels heavier/laggier. SPEED_NORM is the world-speed (units/s) mapped to fully
// "excited" color; local speed genuinely varies here (unlike Boids' clamped cruise), so this is the
// real driver of the rest→excited ramp.
const INERTIA_RATE = 6.0
const SPEED_NORM = 0.35
// Weak spring toward each particle's R2 home (the harness's low-discrepancy sequence — quasi-
// uniform coverage for ANY count prefix, derived from the index alone): restores even coverage
// after every gust. The fluid box does NOT wrap — a toroidal wrap teleported pushed particles
// into dead zones and piled the population up along the viewport edges.
const HOME_RATE = 1.8 // home-spring acceleration per unit displacement (1/s²) — weak vs the flow

// Idle handling: once the field has settled and nothing is driving it, freeze (return null) — the
// last rendered texture persists on screen, so the still dust stays visible with zero GPU cost.
// WARMUP_FRAMES guarantees at least the init + a settled render happen before the first freeze.
const WARMUP_FRAMES = 3

// ── Uniforms (written fully each frame) ────────────────────────────────────────────────────────
// Fluid solve knobs — read by the grid kernels.
const FlowParams = d.struct({
    dt: d.f32, curlStrength: d.f32, velFade: d.f32,
    ambient: d.f32, ambientTime: d.f32, ambientFreq: d.f32,
})
// Per-stamp cursor force — written inside a thunk per ribbon stamp (CursorTrail/InkFlow pattern).
const SplatParams = d.struct({
    posX: d.f32, posY: d.f32, velX: d.f32, velY: d.f32, radius: d.f32,
})
// Particle + render knobs — read by the particle kernels.
const SimParams = d.struct({
    colA: d.vec4f, colB: d.vec4f,
    dt: d.f32, aspect: d.f32, trails: d.f32,
    bodyR: d.f32, speedNorm: d.f32, advect: d.f32,
})

// ── Layouts ──────────────────────────────────────────────────────────────────────────────────
// FLUID: velA/velB = velocity (xy = flow, z = curl scratch) ping-pong; pressure/divergence f32
// scratch; both param uniforms live here so the force-splat kernel (velA + sparams) stays single-
// layout; velOutTex is the handoff — copyVel writes the solved velocity here for the particles.
const fluidLayout = tgpu.bindGroupLayout({
    velA: {storage: d.arrayOf(d.vec4f, FLUID_COUNT), access: 'mutable'},
    velB: {storage: d.arrayOf(d.vec4f, FLUID_COUNT), access: 'mutable'},
    pressure: {storage: d.arrayOf(d.f32, FLUID_COUNT), access: 'mutable'},
    divergence: {storage: d.arrayOf(d.f32, FLUID_COUNT), access: 'mutable'},
    params: {uniform: FlowParams},
    sparams: {uniform: SplatParams},
    velOutTex: {storageTexture: d.textureStorage2d(VEL_FORMAT, 'write-only')},
})
// PARTICLES: per-particle (pos.xy, vel.xy) state; accumE additive energy + accumS agitation-weighted
// copy (their ratio → per-texel average excitement); trailBuf the persistent decayed canvas; velTex
// the field sampled for advection; outTex the resolved render.
const pfLayout = tgpu.bindGroupLayout({
    agents: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
    accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    accumS: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    trailBuf: {storage: d.arrayOf(d.vec4f, RES * RES), access: 'mutable'},
    velTex: {texture: d.texture2d(d.f32)},
    params: {uniform: SimParams},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Fluid kernels: the shared free-field Stable-Fluids solver, dye-less. The force splat is the
// general velocity-impulse stamp, and the ambient breeze rides the solver's ambientWind option. ──

/** Momentum stamp: the general dye-less Gaussian velocity impulse over this fluid's layout. */
export const particleFlowForceKernel = buildVelocityImpulseKernel(fluidLayout, {n: N, namePrefix: 'particleFlow'})

// The rest of the chain is the shared solver: free-field (clamped, free-slip) Stable Fluids with NO
// dye, publishing the solved velocity to `velOutTex` from the copy pass so the particle layer can
// sample it without a dedicated output dispatch. Only the vorticity pass stays local (below) —
// ParticleFlow folds its ambient breeze into it.
const solverKernels = buildStableFluidsKernels(fluidLayout, {
    n: N, namePrefix: 'particleFlow', boundary: 'clamped', dye: 'none', publishVelocityTexture: true,
    // Divergence-free curl-noise breeze folded into the vorticity pass (uniform-branch gated —
    // the "only moves when stirred" default pays nothing).
    ambientWind: {gain: AMBIENT_K},
})
export const {
    curl: particleFlowCurlKernel, divergence: particleFlowDivergenceKernel,
    jacobi: particleFlowJacobiKernel, gradSubtract: particleFlowGradSubtractKernel,
    advectVel: particleFlowAdvectVelKernel, copyVel: particleFlowCopyVelKernel,
} = solverKernels

export const particleFlowVorticityKernel = solverKernels.vorticity

// ── Particle kernels (std/sim vocabulary) ──────────────────────────────────────────────────────

/** Uniform-scatter spawn (the general agents part): even coverage, at rest. */
export const particleFlowInitKernel = agents.makeUniformScatterInit(pfLayout, 'particleFlowInit')

/**
 * Advect one particle: the advection-family integrator — bilinear-sample the fluid velocity at
 * the particle's screen-uv, convert grid velocity to world units, ease toward it (inertia →
 * massy drift), add the weak R2 home-spring, integrate, and clamp to the domain (the field is
 * wall-tangential, so the clamp almost never engages — the spring does the real edge recovery).
 */
export const particleFlowIntegrateKernel = integrator.advect2d(pfLayout, {
    gridN: N,
    inertiaRate: INERTIA_RATE,
    homeRate: HOME_RATE,
    name: 'particleFlowIntegrate',
})

/** The render subsystem: velocity-oriented comet shapes splatted in world units, excitement from
 *  LOCAL SPEED (dust genuinely varies from still to fast wake), resolved through the rest→excited
 *  ramp into the trail canvas — with the trail READ baked out at trails = 0 (the default; the
 *  16 MB/frame read goes away, the WRITE stays so scrubbing up never ghosts). */
function particleFlowRender(shape: string, colorSpaceMode: number, trailsOn: boolean) {
    return renderAgents.orientedWorld(pfLayout, {
        shape,
        res: RES,
        splatRCap: SPLAT_R,
        heading: 'velocity',
        agitation: 'speed',
        comet: true,
        ramp: {colorSpace: colorSpaceMode, trails: {baked: trailsOn}},
        names: {splat: 'particleFlowSplat', resolve: 'particleFlowResolve'},
    })
}

export const makeParticleFlowSplatKernel = (shape: string) =>
    particleFlowRender(shape, 2, false).splat.kernel as (i: number) => void
export const makeParticleFlowResolveKernel = (mode: number, trailsOn: boolean) =>
    particleFlowRender('streak', mode, trailsOn).resolve.kernel as (x: number, y: number) => void
export const particleFlowSplatKernel = makeParticleFlowSplatKernel('streak')
export const particleFlowResolveKernel = makeParticleFlowResolveKernel(2, false)

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    count: number
    force: number
    swirl: number
    momentum: number
    ambient: number
    speed: number
    shape: string
    size: number
    trails: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ParticleFlow",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Interactive",
    description: "Thousands of drifting dust particles carried by a real incompressible fluid field the cursor stirs — drag to plow a wake and the particles ride the eddies and vortices it leaves behind, while a gentle ambient breeze keeps the whole scene breathing on its own",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    props: {
        colorA: {
            default: "#8ec5ff",
            transform: transformColor,
            description: "Color of particles drifting slowly with the flow",
            ui: {type: 'color', label: 'Rest Color', group: 'Colors'}
        },
        colorB: {
            default: "#ffd98e",
            transform: transformColor,
            description: "Color particles flash toward as they speed up in the fast jets and wakes",
            ui: {type: 'color', label: 'Excited Color', group: 'Colors'}
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for the rest→excited color ramp',
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'}
        },
        shape: {
            default: 'streak',
            compileTime: true,
            description: "What each particle is drawn as — a streak or arrow that stretches along the flow, a plain dot, a square, or a soft glowing comet",
            ui: {type: 'select', options: agents.orientedShapeOptions, label: 'Shape', group: 'Particles'}
        },
        count: {
            default: 6000,
            description: "Number of drifting particles",
            ui: {type: 'range', min: 1000, max: 16000, step: 500, label: 'Count', group: 'Particles'}
        },
        size: {
            default: 1.2,
            description: "Size of each particle",
            ui: {type: 'range', min: 0.3, max: 3, step: 0.05, label: 'Size', group: 'Particles'}
        },
        trails: {
            default: 0,
            // The resolve kernel has a variant that skips the trail-buffer read entirely at 0 (the
            // default). This is a continuously-scrubbed slider, so recompile only when the value
            // CROSSES zero — not on every tick.
            recompile: recompileWhen((previousValue, newValue) => ((previousValue as number) > 0) !== ((newValue as number) > 0)),
            description: "How long each particle's motion trail persists — 0 draws crisp dust, 1 leaves long flowing ribbons",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Trails', group: 'Particles'}
        },
        speed: {
            default: 1,
            description: "How strongly the fluid carries the particles — higher makes the dust ride the flow faster",
            ui: {type: 'range', min: 0.2, max: 3, step: 0.05, label: 'Flow Speed', group: 'Particles'}
        },
        force: {
            default: 1,
            description: "How hard the cursor's motion pushes the fluid — higher plows faster, more violent wakes and jets",
            ui: {type: 'range', min: 0, max: 3, step: 0.05, label: 'Cursor Force', group: 'Dynamics'}
        },
        swirl: {
            default: 25,
            description: "Vorticity confinement — swirl energy that spins the flow into persistent eddies and vortices the dust orbits",
            ui: {type: 'range', min: 0, max: 60, step: 1, label: 'Swirl', group: 'Dynamics'}
        },
        momentum: {
            default: 0.6,
            description: "How long the fluid keeps flowing after you stop dragging — high momentum lets the eddies carry the dust on and on",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Momentum', group: 'Dynamics'}
        },
        ambient: {
            default: 0.15,
            description: "A gentle, ever-evolving breeze that keeps the dust drifting even without the cursor — set to 0 for a field that only moves when you stir it",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Ambient', group: 'Dynamics'}
        }
    },

    // The particle + render half rides agentSim (the harness allocates the particle buffers,
    // accumulators, trail canvas and output texture off pfLayout); the FLUID half is composed in
    // `bake` from the shared Stable-Fluids solver, bridged by `velTex` (created in bake, bound
    // immediately onto the layout's external key). Per frame: cursor force ribbon → fluid solve
    // (curl → vorticity+ambient → divergence → JACOBI_ITERS× jacobi → gradSub → advect+copy, the
    // copy publishing velTex) → advect particles on the fresh field → oriented shape splat →
    // ramp resolve into the trail canvas / output texture. `shape` and `colorSpace` are
    // compile-time (baked kernel variants), `trails` recompiles across 0; `count` is runtime.
    // Idle-freezes once the field has settled and nothing is driving it. GENERATOR.
    ...agentSim<typeof SimParams>({
        layout: pfLayout,
        params: SimParams,
        maxAgents: MAX_PARTICLES,
        countCap: {desktop: MAX_PARTICLES, mobile: COUNT_MAX_MOBILE},
        output: {key: 'outTex', name: 'flowTexture', size: [RES, RES], format: STATE_FORMAT},
        externalKeys: ['velTex'],
        bake: (params) => {
            const {gpu, getCpuValue, onCleanup} = params
            const root = gpu!.root!

            // The fluid layer: its own uniforms + bind group over harness-free state buffers,
            // the local force/vorticity kernels, and the shared solver passes.
            const velTex = root.createTexture({size: [N, N], format: VEL_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => velTex.destroy())
            const paramsU = root.createUniform(FlowParams)
            const splatU = root.createUniform(SplatParams)
            const fluidBg = root.createBindGroup(fluidLayout, {
                velA: createStateBuffer(root, d.vec4f, FLUID_COUNT),
                velB: createStateBuffer(root, d.vec4f, FLUID_COUNT),
                pressure: createStateBuffer(root, d.f32, FLUID_COUNT),
                divergence: createStateBuffer(root, d.f32, FLUID_COUNT),
                params: paramsU.buffer, sparams: splatU.buffer, velOutTex: velTex,
            })
            const force = createFluidKernelPass(root, particleFlowForceKernel, {n: N, bindGroup: fluidBg})
            const vorticity = createFluidKernelPass(root, particleFlowVorticityKernel, {n: N, bindGroup: fluidBg})
            const solver = createStableFluidsPasses(root, solverKernels, {n: N, bindGroup: fluidBg})

            // shape + colorSpace are compile-time, and `trails` recompiles only across the 0
            // boundary — pick the matching baked kernels at composition.
            const trailsCpu = getCpuValue('trails')
            const render = particleFlowRender(
                (getCpuValue('shape') as string) || 'streak',
                (getCpuValue('colorSpace') as number) ?? 2,
                typeof trailsCpu === 'number' && trailsCpu > 0,
            )

            return {
                pipelines: {
                    // Dispatched manually on the first live frame (after the solver steps have a
                    // sane dt), not via the harness init latch — see `frame`.
                    init: {kernel: particleFlowInitKernel, threads: 'max'},
                    integrate: {kernel: particleFlowIntegrateKernel, threads: 'agents'},
                    splat: render.splat,
                    resolve: render.resolve,
                },
                program: ['integrate', 'splat', 'resolve'],
                bindNow: {velTex},
                setup: {force, vorticity, solver, paramsU, splatU},
            }
        },
        frame: (sys, {getCpuValue}, setupRaw) => {
            const {force, vorticity, solver, paramsU, splatU} = setupRaw as {
                force: ReturnType<typeof createGuardedCompute>
                vorticity: ReturnType<typeof createGuardedCompute>
                solver: ReturnType<typeof createStableFluidsPasses>
                paramsU: {write: (v: never) => void}
                splatU: {write: (v: never) => void}
            }

            let lastTime = Date.now()
            let ambientTime = 0
            let initialized = false
            const pointer = createPointerVelocityTracker()
            // The init dispatch plus a few settling frames must land before the gate may freeze.
            const idle = createIdleGate({warmupFrames: WARMUP_FRAMES + 1})

            const num = (key: string, fallback: number): number => {
                const v = getCpuValue(key)
                return typeof v === 'number' ? v : fallback
            }

            return (frameParams) => {
                const fp = frameParams as {pointer: {x: number; y: number}; deltaTime: number; dimensions?: {width: number; height: number}}
                const now = Date.now()
                const dt = Math.min(fp.deltaTime ?? (now - lastTime) / 1000, 0.033)
                lastTime = now
                if (dt < 0.001) return null

                const h = fp.dimensions?.height ?? 0
                const aspect = h > 0 ? (fp.dimensions?.width ?? 0) / h : 16 / 9
                const domainX = Math.max(aspect, 0.01)

                const forceVal = num('force', 1)
                const swirl = num('swirl', 25)
                const ambient = Math.min(Math.max(num('ambient', 0.15), 0), 1)
                const momentum = Math.min(Math.max(num('momentum', 0.6), 0), 1)
                const advect = num('speed', 1)
                const count = sys.resolveCount(num('count', 6000), 100)
                const bodyR = Math.min(Math.max(num('size', 1.2), 0.3), 3) * 1.3 / SIZE_REF_RES

                // The tracker's teleport guard is what keeps a pointer jump (entering the canvas,
                // a tab switch) from plowing a wake across the whole field.
                const ptr = pointer.update(fp.pointer, dt)
                const active = ptr.moving || ambient > 0.001
                if (active) idle.markActive()

                ambientTime += dt * AMBIENT_DRIFT

                // Idle-freeze: once the field has settled and nothing is driving it, stop
                // dispatching — the last rendered texture persists, so the still dust stays
                // on screen for free.
                if (initialized && idle.shouldSkip(2 + momentum * 13, active)) return null
                idle.tickFrame(dt)

                const velFade = VEL_FADE_MAX * Math.pow(VEL_FADE_MIN / VEL_FADE_MAX, momentum)
                paramsU.write({dt, curlStrength: swirl, velFade, ambient, ambientTime, ambientFreq: AMBIENT_FREQ} as never)

                const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
                const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined
                sys.writeParams({
                    colA: d.vec4f(colA?.x ?? 0.56, colA?.y ?? 0.77, colA?.z ?? 1, colA?.w ?? 1),
                    colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 0.85, colB?.z ?? 0.56, colB?.w ?? 1),
                    dt, aspect: domainX, trails: Math.min(Math.max(num('trails', 0), 0), 1) * 0.92,
                    bodyR, speedNorm: SPEED_NORM, advect,
                })

                const nodes: ComputeStep[] = []
                if (!initialized) {
                    initialized = true
                    nodes.push(() => sys.pipelines.init.dispatchThreads(MAX_PARTICLES))
                }

                // Cursor force: interpolate a ribbon of stamps along the drag path (no dotted gaps
                // on a fast flick), each a velocity-proportional impulse into the field.
                if (ptr.moving) {
                    const velX = ptr.velX * N * forceVal * IMPULSE_K
                    const velY = ptr.velY * N * forceVal * IMPULSE_K
                    pathStampRibbon(nodes, {
                        fromX: ptr.prevX, fromY: ptr.prevY, dx: ptr.dx, dy: ptr.dy,
                        dragDist: ptr.dragDist, stepSize: Math.max(0.004, (FORCE_RADIUS_GRID / N) * 0.6),
                        maxSteps: MAX_STEPS, scale: N,
                        write: (posX, posY) => splatU.write({posX, posY, velX, velY, radius: FORCE_RADIUS_GRID} as never),
                        pass: force,
                    })
                }

                // Fluid solve (its own vorticity pass, which carries the ambient breeze), then the
                // particles ride the freshly-solved field.
                nodes.push(...solver.solveSteps({jacobiIters: JACOBI_ITERS, vorticity}))
                nodes.push(...(sys.frame({count}) ?? []))
                return nodes
            }
        },
        // Bilinear-sample the resolved field over the full canvas (rgba16f is filterable).
        // GPU-free (no device) → transparent, so it composites cleanly as a background layer.
        fragment: {output: 'flowTexture', fallback: 'transparent'},
    }),
})

export default componentDefinition
