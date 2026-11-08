import type {GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {call, makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {agentSim, renderAgents, force, drift, integrator, agentFrame, pointSlot} from "@coreroot/std/sim/agents"
import {tgpu, d, agents, constants} from "@coreroot/gpu/kit"
import {transformColor} from "@coreroot/utilities/transformations"

// ── Simulation constants ─────────────────────────────────────────────────────────────────────
// FloatingParticles v2 — ported from the old procedural Voronoi field onto the shared agent
// simulation: N REAL particles in "screen-proportional" world space (x ∈ [0, aspect], y ∈ [0, 1],
// 1 unit == the canvas HEIGHT, isotropic on any aspect). The signature look is kept — soft drifting
// motes with per-particle speed/heading variance, a gentle orbital wander (`randomness`, the old
// Voronoi points rotating around their cells) and a hash-phased twinkle — but each mote is now a
// real body: hashed size/alpha variance gives the old multi-layer parallax, and the cursor stirs
// the field with a decaying gust (new — impossible in the procedural version). Motion is pure
// uniform drift, so the toroidal wrap preserves even coverage exactly.
const MAX_MOTES = 8192
// Effective count cap on mobile GPUs — the prop's declared range is unchanged, so a
// desktop-authored preset still loads, it just simulates fewer motes on a phone.
const COUNT_MAX_MOBILE = 3000
// Square render/accumulator resolution; the fragment bilinear-upsamples to screen. Build-time
// constant (SSR/tests resolve desktop) because the bind-group layout is module-level. Motes are
// sized against the harness's FIXED SIZE_REF_RES, not this live value — otherwise shrinking RES on
// mobile would shrink the world-space texel and inflate apparent mote size.
const RES = resolveRenderRes({desktop: 1024, mobile: 512})
const STATE_FORMAT = 'rgba16float' as const
const DEG_TO_RAD = constants.DEG_TO_RAD

// Base drift in world-units/second at `speed` = 1 — leisurely, matching the old default feel (a
// mote takes ~8s to cross the height at speed 1).
const BASE_DRIFT = 0.12
// Orbital wander: radius (world units) and spin rate (rad/s) at `randomness` = 1, both varied
// per particle so the field never phase-locks.
const ORBIT_RAD = 0.035
const ORBIT_RATE = 1.4
const TWINKLE_FREQ = 2.0 // matches the old twinkle clock

// Cursor gust: a kit magnet impulse on the stored velocity, decaying over ~1s. The strength
// slider is 0–1 and deliberately gentle — these are ambient motes, not a fluid to shove around.
const CURSOR_FORCE_K = 3.5
const CURSOR_DRAG = 1.1

// The drift heading is flipped 180° internally: the classic (procedural) FloatingParticles moved
// the SAMPLE WINDOW along the angle, so the particles visually drifted the opposite way — the
// default angle 90 floated them UPWARD. The flip keeps every legacy preset's visual direction.
const ANGLE_FLIP = constants.PI

// All per-frame CPU-derived inputs (kernels can't read node uniforms — the renderer writes these
// each frame from getCpuValue).
const SimParams = d.struct({
    color: d.vec4f,
    count: d.f32, dt: d.f32, time: d.f32, aspect: d.f32, domainX: d.f32,
    driftBase: d.f32, angleRad: d.f32, speedVar: d.f32, angleVarRad: d.f32,
    randomness: d.f32, twinkle: d.f32, softness: d.f32,
    bodyR: d.f32,
    cursorX: d.f32, cursorY: d.f32, cursorRadSq: d.f32, cursorForce: d.f32, dragMul: d.f32,
})

// One layout for the whole pipeline. `agents` xyzw = (posX, posY, gustVX, gustVY) — the base drift
// is recomputed deterministically each frame (per-mote speed/heading hashed from the index), only
// the decaying cursor gust is stored. `accumE` is the additive splat energy (no excitement
// ramp here, so there is no second accumulator).
const simLayout = tgpu.bindGroupLayout({
    agents: {storage: d.arrayOf(d.vec4f, MAX_MOTES), access: 'mutable'},
    accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    params: {uniform: SimParams},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Spawn + composed physics (std/sim vocabulary) ────────────────────────────────────────────

/** Spawn: scatter motes across the WHOLE canvas (no empty-screen start), at rest. */
export const floatingParticlesInitKernel = agents.makeUniformScatterInit(simLayout, 'floatingParticlesInit')

/**
 * The integrator: deterministic per-mote drift (the varied-heading part — density-preserving
 * uniform translation, same ± ranges as the old per-layer variance) plus the decaying
 * cursor-gust impulse, wrapped toroidally by the drift-family integrator — correct here
 * because the motion is pure drift, so coverage stays even (unlike a fluid box, nothing clumps).
 */
export const floatingParticlesUpdateKernel = integrator.drift2d(simLayout, {
    drift: drift.variedHeading(simLayout),
    gust: force.cursorGust(simLayout),
    name: 'floatingParticlesUpdate',
})

/** The render subsystem: un-oriented point motes with live softness feathering, resolved as a
 *  single tint's alpha. The slots carry the FloatingParticles character — the orbital wander
 *  (the old Voronoi points spinning around their cells), the hash-phased twinkle, and the
 *  presence cascade (the old layer size/alpha parallax) scaling size and brightness together. */
function floatingParticlesRender(shape: string) {
    return renderAgents.pointWorld(simLayout, {
        shape,
        res: RES,
        place: pointSlot.orbitalPlace(simLayout, {radius: ORBIT_RAD, rate: ORBIT_RATE}),
        brightness: pointSlot.twinkleBrightness(simLayout, {freq: TWINKLE_FREQ}),
        bodyRadius: pointSlot.presenceRadius(simLayout),
        names: {splat: 'floatingParticlesSplat', resolve: 'floatingParticlesResolve'},
    })
}

export const makeFloatingParticlesSplatKernel = (shape: string) => floatingParticlesRender(shape).splat.kernel as (i: number) => void
export const floatingParticlesSplatKernel = makeFloatingParticlesSplatKernel('dot')
export const floatingParticlesResolveKernel = floatingParticlesRender('dot').resolve.kernel as (x: number, y: number) => void

/** Composite the particles OVER a child color (the kit straight-alpha over) — kept from v1 so
 *  the shader still works as a wrapping layer without an RTT. */
export const particleOverChild = agents.straightAlphaOver

// ─────────────────────────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    count: number
    speed: number
    angle: number
    speedVariance: number
    angleVariance: number
    randomness: number
    twinkle: number
    particleSize: number
    softness: number
    shape: string
    particleColor: Parameters<typeof transformColor>[0]
    cursorStrength: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "FloatingParticles",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Textures",
    description: "Drifting, twinkling motes — thousands of real simulated particles floating in a shared heading with per-particle wander and variance, and a cursor that stirs the field with a gust that settles back into the drift",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    props: {
        particleColor: {
            default: "#ffffff",
            transform: transformColor,
            description: "Color of the particles",
            ui: {type: 'color', label: 'Particle Color', group: 'Colors'}
        },
        shape: {
            default: 'dot',
            compileTime: true,
            description: "What each particle is drawn as — a crisp dot or square (softness still feathers them), or a soft glow puff",
            ui: {type: 'select', options: agents.pointShapeOptions, label: 'Shape', group: 'Effect'}
        },
        count: {
            default: 1200,
            description: "Number of particles",
            ui: {type: 'range', min: 100, max: MAX_MOTES, step: 100, label: 'Count', group: 'Effect'}
        },
        particleSize: {
            default: 1.2,
            description: "Size of the particles (each also varies slightly for depth)",
            ui: {type: ['range', 'map'], min: 0.3, max: 4, step: 0.05, label: 'Particle Size', group: 'Effect'}
        },
        softness: {
            default: 0.1,
            description: "Edge softness of each particle — 0 is a crisp shape, 1 a soft glow puff",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect'}
        },
        speed: {
            default: 0.25,
            description: "Speed of the shared drift",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Speed', group: 'Animation'}
        },
        angle: {
            default: 90,
            description: "Drift heading in degrees (0 = left, 90 = up, 180 = right, 270 = down — the classic default floats upward)",
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Angle', group: 'Animation'}
        },
        speedVariance: {
            default: 0.3,
            description: "Per-particle speed variance around the shared drift (0 = everything moves in lockstep)",
            ui: {type: 'range', min: 0, max: 1, step: 0.05, label: 'Speed Variance', group: 'Animation'}
        },
        angleVariance: {
            default: 30,
            description: "Per-particle heading variance in degrees (0 = one shared direction, 180 = every which way)",
            ui: {type: 'range', min: 0, max: 180, step: 1, label: 'Angle Variance', group: 'Animation'}
        },
        randomness: {
            default: 0.25,
            description: "Orbital wander — each particle circles lazily around its drift path instead of gliding straight",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.05, label: 'Randomness', group: 'Animation'}
        },
        twinkle: {
            default: 0.5,
            description: "Intensity of the twinkle (0 = steady, 1 = full shimmer)",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.05, label: 'Twinkle', group: 'Animation'}
        },
        cursorStrength: {
            default: 0,
            description: "How strongly the cursor stirs the field — 0 (default) leaves the drift undisturbed",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Cursor Gust', group: 'Interaction'}
        }
    },

    // The whole drift sim as a composed agentSim: per frame — integrate (deterministic drift +
    // decaying cursor gust, toroidal wrap) → point shape splat (orbit + twinkle applied at
    // render) → tint resolve (which also clears the accumulator). `shape` is compile-time (baked
    // kernel variant); `count` is runtime (dispatchThreads). GENERATOR: no child required, no RTT.
    ...agentSim<typeof SimParams>({
        layout: simLayout,
        params: SimParams,
        maxAgents: MAX_MOTES,
        countCap: {desktop: MAX_MOTES, mobile: COUNT_MAX_MOBILE},
        output: {key: 'outTex', name: 'particleTexture', size: [RES, RES], format: STATE_FORMAT},
        bake: ({getCpuValue}) => {
            const render = floatingParticlesRender((getCpuValue('shape') as string) || 'dot')
            return {
                pipelines: {
                    init: {kernel: floatingParticlesInitKernel, threads: 'max'},
                    update: {kernel: floatingParticlesUpdateKernel, threads: 'agents'},
                    splat: render.splat,
                    resolve: render.resolve,
                },
                initStep: 'init',
                program: ['update', 'splat', 'resolve'],
            }
        },
        frame: (sys, {getCpuValue}) => {
            const g = makeCpuValueGetter(getCpuValue)
            let localTime = 0
            return (frameParams) => {
                const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams)
                localTime += dt
                const domainX = Math.max(aspect, 0.01)

                const count = sys.resolveCount(g('count', 1200))
                const cursorX = pointerX * aspect
                const cursorY = pointerY
                const cursorRadius = 0.22

                const col = getCpuValue('particleColor') as {x: number; y: number; z: number; w: number} | undefined
                const bodyR = Math.min(Math.max(g('particleSize', 1.2), 0.3), 4) * 2 / SIZE_REF_RES

                sys.writeParams({
                    color: d.vec4f(col?.x ?? 1, col?.y ?? 1, col?.z ?? 1, col?.w ?? 1),
                    count, dt, time: localTime, aspect, domainX,
                    driftBase: Math.min(Math.max(g('speed', 0.25), 0), 1.5) * BASE_DRIFT / 0.25,
                    angleRad: g('angle', 90) * DEG_TO_RAD + ANGLE_FLIP,
                    speedVar: Math.min(Math.max(g('speedVariance', 0.3), 0), 1),
                    angleVarRad: Math.min(Math.max(g('angleVariance', 30), 0), 180) * DEG_TO_RAD,
                    randomness: Math.min(Math.max(g('randomness', 0.25), 0), 1),
                    twinkle: Math.min(Math.max(g('twinkle', 0.5), 0), 1),
                    softness: Math.min(Math.max(g('softness', 0.1), 0), 1),
                    bodyR,
                    cursorX, cursorY,
                    cursorRadSq: cursorRadius * cursorRadius,
                    cursorForce: Math.min(Math.max(g('cursorStrength', 0), 0), 1) * CURSOR_FORCE_K,
                    dragMul: agentFrame.expDecay(CURSOR_DRAG, dt),
                })

                return sys.frame({count})
            }
        },
        // Bilinear-sample the resolved field over the full canvas; composite OVER the child when
        // one is nested (kept from v1 — no RTT needed for a plain over). GPU-free → transparent
        // (or the child untouched).
        fragment: {
            output: 'particleTexture',
            fallback: 'child',
            compose: (field: Expr, {childNode}: GpuFragmentParams): Expr =>
                childNode ? call(particleOverChild, 'particleOverChild', [field, childNode]) : field,
        },
    }),
})

export default componentDefinition
