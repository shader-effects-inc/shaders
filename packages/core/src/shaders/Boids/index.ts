import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {agentSim, renderAgents, force, integrator} from "@coreroot/std/sim/agents"
import {tgpu, d, agents} from "@coreroot/gpu/kit"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

// ── Simulation constants ─────────────────────────────────────────────────────────────────────
// Classic Reynolds flocking (separation / alignment / cohesion) as a living background layer. Every
// agent carries a 2D position + velocity in "screen-proportional" world space — x ∈ [0, aspect],
// y ∈ [0, 1], so 1 unit == the canvas HEIGHT and every distance is isotropic (perception is a true
// circle, murmurations don't skew on wide canvases). Neighbour search is brute-force O(N²) — the
// WebGPU "Compute Boids" approach, engine policy inside the flocking force part (static MAX bound,
// runtime-count break — the slider never recompiles); it stays smooth into the low thousands.
const MAX_AGENTS = 4096
const COUNT_MAX_MOBILE = 1200 // effective cap on mobile GPUs (the prop range stays the same)
// Square render/accumulator resolution; the fragment bilinear-upsamples to screen. Halved area on
// mobile GPUs — the resolve is one thread per texel, so this is the dominant per-frame cost there.
// `size` is calibrated against the harness's fixed SIZE_REF_RES rather than this live value —
// otherwise the lower mobile RES would render the same flock 1.6× larger.
const RES = resolveRenderRes({desktop: 1024, mobile: 640})
const STATE_FORMAT = 'rgba16float' as const
// Sanity clamp on an agent's body half-width, in texels. The splat WINDOW itself is dynamic (sized
// per agent from its actual accept radius, per axis — renderAgents), so this only exists so a
// future shape/size combination can't produce a runaway window.
const SPLAT_R = 16

// Base cruise speed in world-units/second at `speed` = 1 (a boid crosses the height in ~7.5s). The
// steering-force cap is a multiple of it, sized so a full about-face takes ~0.3s → smooth arcs, not
// gas-molecule jitter. Both are load-bearing for the murmuration read.
const BASE_SPEED = 0.13
const FORCE_RATIO = 3.5

// Agitation envelope: steering effort (|accel| relative to maxForce) above REST_ACCEL charges a
// per-agent excitement value that peaks instantly and cools over ~1.5s — scattering from the
// predator cursor or a hard murmuration turn flashes toward the excited color, cruising stays at
// rest. Replaces the old speed-based tint (speed is clamped to cruise, so it never varied).
const AGIT_REST = 1.5 // steering effort (in maxForce units) where excitement starts
const AGIT_GAIN = 0.4545 // 1 / 2.2 — effort span from rest to fully excited
const AGIT_COOL = 1.3 // exponential cooling rate (1/s)

// All per-frame CPU-derived inputs. Kernels can't read node uniforms (those live in the composer's
// bind groups), so the renderer writes these each frame from getCpuValue.
const SimParams = d.struct({
    colA: d.vec4f, colB: d.vec4f,
    count: d.f32, trails: d.f32,
    dt: d.f32, aspect: d.f32, domainX: d.f32,
    maxSpeed: d.f32, maxForce: d.f32,
    perceptionSq: d.f32, sepRadiusSq: d.f32,
    sepW: d.f32, aliW: d.f32, cohW: d.f32,
    cursorX: d.f32, cursorY: d.f32, cursorMode: d.f32, cursorRadius: d.f32, cursorRadiusSq: d.f32, cursorForce: d.f32,
    bodyR: d.f32,
    margin: d.f32, turnForce: d.f32,
    seed: d.f32,
})

// One layout for the whole pipeline. `agents` xyzw = (posX, posY, velX, velY); updated IN PLACE
// (a Gauss-Seidel relaxation — cross-agent reads may see this frame's or last frame's neighbour,
// which is invisible in a smooth flocking field and saves a ping-pong buffer). `agit` is the
// per-agent excitement envelope. `accumE` is additive splat energy, `accumS` the agitation-weighted
// copy (their ratio → per-texel average excitement for the rest→excited color ramp). `trailBuf` is
// the persistent decayed canvas that produces motion trails.
const simLayout = tgpu.bindGroupLayout({
    agents: {storage: d.arrayOf(d.vec4f, MAX_AGENTS), access: 'mutable'},
    agit: {storage: d.arrayOf(d.f32, MAX_AGENTS), access: 'mutable'},
    accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    accumS: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    trailBuf: {storage: d.arrayOf(d.vec4f, RES * RES), access: 'mutable'},
    params: {uniform: SimParams},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Spawn (module scope — the seeded formation IS the look) ─────────────────────────────────

/**
 * Spawn: seed the flock already IN FORMATION — K elongated clusters (one per ~250 agents), each
 * with a shared heading and cruise velocity, jittered per agent. Frame one reads as a murmuration
 * instead of a uniform scatter that takes seconds to self-organise; actually pre-running the sim
 * would cost O(N²) per warm-up step, so the formation is seeded, not simulated. `seed` re-rolls the
 * layout (the harness re-dispatches this one-shot when it changes — runtime, no recompile).
 */
export const boidsInitKernel = agents.makeClusterFormationInit(simLayout, 'boidsInit')

// ── The composed physics + render (std/sim vocabulary) ──────────────────────────────────────

/**
 * The integrator: the Reynolds triple over one neighbour scan, the cursor attract/predator
 * field, and the soft edge turn — folded in that order into one kernel by the steering-family
 * integrator, which owns the cruise variation, the agitation envelope, the min-cruise speed
 * clamp and the reflective wall safety net.
 */
export const boidsUpdateKernel = integrator.steering2d(simLayout, {
    forces: [
        force.flocking(simLayout, {maxAgents: MAX_AGENTS}),
        force.cursorSteer(simLayout),
        force.wallTurn(simLayout),
    ],
    agitation: {rest: AGIT_REST, gain: AGIT_GAIN, cool: AGIT_COOL},
    cruiseFloor: 0.35,
    wallRestitution: 0.6,
    name: 'boidsUpdate',
})

/** The render subsystem: velocity-oriented comet shapes splatted in world units, resolved
 *  through the rest→excited ramp into the persistent trail canvas. */
function boidsRender(shape: string, colorSpaceMode: number) {
    return renderAgents.orientedWorld(simLayout, {
        shape,
        res: RES,
        splatRCap: SPLAT_R,
        heading: 'velocity',
        agitation: 'buffer',
        comet: true,
        ramp: {colorSpace: colorSpaceMode, trails: 'on'},
        names: {splat: 'boidsSplat', resolve: 'boidsResolve'},
    })
}

export const makeBoidsSplatKernel = (shape: string) => boidsRender(shape, 2).splat.kernel as (i: number) => void
export const makeBoidsResolveKernel = (mode: number) => boidsRender('arrow', mode).resolve.kernel as (x: number, y: number) => void
export const boidsSplatKernel = makeBoidsSplatKernel('arrow')
export const boidsResolveKernel = makeBoidsResolveKernel(2)

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    count: number
    seed: number
    speed: number
    separation: number
    alignment: number
    cohesion: number
    perception: number
    cursorMode: string
    cursorStrength: number
    agentShape: string
    size: number
    trails: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Boids",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "A living murmuration of hundreds of flocking agents drawn as crisp arrows, streaks, dots or glowing comets — separation, alignment and cohesion drive fluid, splitting-and-merging ribbons, agents flash toward an excited color when scattered, and the cursor acts as an attractor or a predator",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    props: {
        colorA: {
            default: "#8ec5ff",
            transform: transformColor,
            description: "Color of agents cruising calmly with the flock",
            ui: { type: 'color', label: 'Rest Color', group: 'Colors' }
        },
        colorB: {
            default: "#ff7ad9",
            transform: transformColor,
            description: "Color agents flash toward when agitated — scattering from the predator cursor or banking through a hard turn",
            ui: { type: 'color', label: 'Excited Color', group: 'Colors' }
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for the rest→excited color ramp',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        },
        agentShape: {
            default: 'arrow',
            compileTime: true,
            description: "What each agent is drawn as — a hard-edged arrow, streak or square pointing along its heading, a plain dot, or a soft glowing comet",
            ui: { type: 'select', options: agents.orientedShapeOptions, label: 'Shape', group: 'Flock' }
        },
        count: {
            default: 2000,
            description: "Number of flocking agents",
            ui: { type: 'range', min: 100, max: 4000, step: 100, label: 'Count', group: 'Flock' }
        },
        speed: {
            default: 2,
            description: "Overall cruising speed of the flock",
            ui: { type: 'range', min: 0.2, max: 5, step: 0.1, label: 'Speed', group: 'Flock' }
        },
        seed: {
            default: 0,
            description: "Re-rolls the flock's starting formation — each value spawns the murmuration in a different arrangement",
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Flock' }
        },
        size: {
            default: 1.5,
            description: "Size of each agent",
            ui: { type: 'range', min: 0.5, max: 3, step: 0.05, label: 'Size', group: 'Flock' }
        },
        trails: {
            default: 0,
            description: "How long each agent's motion trail persists — 0 draws crisp agents, 1 leaves long ribbons",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Trails', group: 'Flock' }
        },
        separation: {
            default: 1.7,
            description: "How strongly agents steer away from crowding their neighbours",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Separation', group: 'Flocking' }
        },
        alignment: {
            default: 1.5,
            description: "How strongly agents match their neighbours' heading — the driver of coherent murmuration ribbons",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Alignment', group: 'Flocking' }
        },
        cohesion: {
            default: 1,
            description: "How strongly agents steer toward the local center of the flock",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Cohesion', group: 'Flocking' }
        },
        perception: {
            default: 0.12,
            description: "How far each agent can sense its neighbours (in screen heights) — larger sees bigger, smoother flocks",
            ui: { type: 'range', min: 0.05, max: 0.35, step: 0.01, label: 'Perception', group: 'Flocking' }
        },
        cursorMode: {
            default: 'repel',
            description: "How the flock reacts to the cursor",
            ui: { type: 'select', options: [
                { label: 'None', value: 'none' },
                { label: 'Attract', value: 'attract' },
                { label: 'Predator', value: 'repel' },
            ], label: 'Cursor', group: 'Interaction' }
        },
        cursorStrength: {
            default: 1.5,
            description: "Strength of the cursor's pull or push on the flock",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Cursor Strength', group: 'Interaction' }
        }
    },

    // The whole flocking sim as a composed agentSim: per frame — integrate (flocking + cursor +
    // edges via the declared force parts) → oriented shape splat → ramp resolve into the trail
    // canvas (the resolve also zeroes the accumulators, so there is no separate clear pass).
    // `agentShape` and `colorSpace` are compile-time (baked kernel variants); `count` is runtime
    // (static loop bound + break). GENERATOR: no child, no RTT.
    ...agentSim<typeof SimParams>({
        layout: simLayout,
        params: SimParams,
        maxAgents: MAX_AGENTS,
        countCap: {desktop: MAX_AGENTS, mobile: COUNT_MAX_MOBILE},
        output: {key: 'outTex', name: 'boidsTexture', size: [RES, RES], format: STATE_FORMAT},
        bake: ({getCpuValue}) => {
            const render = boidsRender(
                (getCpuValue('agentShape') as string) || 'arrow',
                (getCpuValue('colorSpace') as number) ?? 2,
            )
            return {
                pipelines: {
                    init: {kernel: boidsInitKernel, threads: 'max'},
                    update: {kernel: boidsUpdateKernel, threads: 'agents'},
                    splat: render.splat,
                    resolve: render.resolve,
                },
                // `seed` re-dispatches the one-shot spawn (runtime — no recompile).
                initStep: 'init',
                program: ['update', 'splat', 'resolve'],
            }
        },
        frame: (sys, {getCpuValue}) => {
            const g = makeCpuValueGetter(getCpuValue)
            return (frameParams) => {
                const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams)
                const domainX = Math.max(aspect, 0.01)

                const count = sys.resolveCount(g('count', 2000))
                const seed = g('seed', 0)
                const maxSpeed = BASE_SPEED * g('speed', 2)
                const maxForce = maxSpeed * FORCE_RATIO
                const perception = g('perception', 0.12)
                const cursorModeStr = (getCpuValue('cursorMode') as string) ?? 'repel'
                const cursorMode = cursorModeStr === 'attract' ? 1 : cursorModeStr === 'repel' ? 2 : 0

                // Pointer arrives in screen UV [0,1] (y down). Lift x into world (× aspect).
                const cursorX = pointerX * aspect
                const cursorY = pointerY
                const cursorRadius = 0.22

                const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
                const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined

                // Body half-width in world units: `size` maps to 2 texels per unit (a size-1.5 arrow
                // spans ~14 texels tip-to-tail).
                const bodyR = Math.min(Math.max(g('size', 1.5), 0.5), 3) * 2 / SIZE_REF_RES

                sys.writeParams({
                    colA: d.vec4f(colA?.x ?? 0.56, colA?.y ?? 0.77, colA?.z ?? 1, colA?.w ?? 1),
                    colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 0.48, colB?.z ?? 0.85, colB?.w ?? 1),
                    count, trails: Math.min(Math.max(g('trails', 0.35), 0), 1) * 0.92,
                    dt, aspect, domainX,
                    maxSpeed, maxForce,
                    perceptionSq: perception * perception,
                    sepRadiusSq: (perception * 0.5) * (perception * 0.5),
                    sepW: g('separation', 1.7),
                    aliW: g('alignment', 1.5),
                    cohW: g('cohesion', 1),
                    cursorX, cursorY, cursorMode,
                    cursorRadius,
                    cursorRadiusSq: cursorRadius * cursorRadius,
                    cursorForce: g('cursorStrength', 1.5) * maxSpeed * 12,
                    bodyR,
                    margin: 0.07, turnForce: maxSpeed * 2.2,
                    seed,
                })

                return sys.frame({count, reseed: seed})
            }
        },
        // Bilinear-sample the resolved field over the full canvas (rgba16f is filterable).
        // GPU-free (no device) → transparent, so it composites cleanly as a background layer.
        fragment: {output: 'boidsTexture', fallback: 'transparent'},
    }),
})

export default componentDefinition
