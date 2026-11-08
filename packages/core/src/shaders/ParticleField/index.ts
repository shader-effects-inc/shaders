import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {agentSim, renderAgents, force, channel, integrator, agentFrame} from "@coreroot/std/sim/agents"
import {tgpu, d, agents, constants} from "@coreroot/gpu/kit"

/**
 * ParticleField — a STYLIZE effect that explodes the child content into a breathing 3D relief
 * of particles. Not a filter: a per-frame FEEDBACK SIMULATION whose targets are read from the
 * live child.
 *
 * Each particle has a fixed home on a dense grid and keeps the child's color at that UV. It is a
 * TRUE 3D point: x/y home + a simulated Z driven by a channel of the child (`depthSource`:
 * luminance / inverse / r/g/b / saturation / alpha) — bright areas float toward you, dark areas
 * recede. The whole cloud can be orbited with the camera rotation props (the relief seen from any
 * angle, extrusion and all) because every particle is projected individually.
 *
 * PHYSICS: the relief-family integrator — a spring toward home (xy) / the child-driven depth
 * target (z), plus `force.cursorInView` (the shared kit magnet applied in VIEW space, so the
 * cursor interacts with what is actually under the pointer at any camera angle — particles get
 * thrown, coast, and spring home over an underdamped settle wobble).
 *
 * RENDER: the `renderAgents.relief` variant — the camera + colorFrom-layer splat (depth-weighted
 * RGBW color accumulation: near particles dominate the weighted average, a stylized pseudo-
 * occlusion needing no z-sort) and the weighted-color resolve. The gaps stay transparent, so
 * the field layers over whatever is behind it. The child RTT is late-bound (bindInputs — it only
 * exists after the pass manager allocates boundaries).
 */

// ── Simulation constants ──────────────────────────────────────────────────────────────────────
// The grid is refit EVERY FRAME to the runtime `count` and the frame aspect (GW/GH ≈ aspect) so
// particle spacing stays isotropic — sim/splat dispatch (GW, GH) threads at runtime, so the count
// slider never recompiles. Positions are in "screen-proportional" world space (x ∈ ±aspect/2,
// y ∈ ±1/2, z toward the viewer) so distances are isotropic on any canvas.
const GRID_DIM_MAX = 256
const MAX_PARTICLES = GRID_DIM_MAX * GRID_DIM_MAX // 65536 — headroom for extreme aspects
const COUNT_MAX = 40000
const COUNT_MAX_MOBILE = 12000 // effective cap on mobile GPUs (the prop range stays the same)

// Additive render target. Square MAX buffers (indexed by a per-frame outW/outH ≤ MAX), so the
// module-scope layout stays fixed while the texture matches the frame aspect (square texels →
// round dots, 1:1 uv sampling). Mobile drops to 768 to cut the two full-target passes (clear +
// resolve), the system's dominant fixed cost — build-time constant (SSR/tests resolve desktop).
const OUT_MAX = resolveRenderRes({desktop: 1024, mobile: 768})
const OUT_CELLS = OUT_MAX * OUT_MAX
const SPLAT_R = 8 // cap on a particle's body radius in texels (the window itself is dynamic)
const AA_W = 0.75 // hard-edge anti-alias half-width, in render texels
const MIN_ALPHA = 0.002 // child alpha below which a particle is skipped entirely
const STATE_FORMAT = 'rgba16float' as const
const FP = 2048.0 // fixed-point scale for the atomic accumulators
const WOBBLE_FREQ = 1.15 // idle-wobble base frequency (× localTime)
const ALPHA_K = 2.6 // accumulated-weight → coverage alpha curve
const PERSP_K = 1.1 // projection strength — depth alone controls how much the relief pops

// Spring/drag tuning (the "gold standard" feel): underdamped return (~0.4 damping ratio) so a
// thrown particle overshoots home once and settles, cursor force sized so the default strength
// carves a clear wake and a fast flick genuinely throws particles across the frame.
const STIFF = 26 // spring stiffness toward home (1/s²)
const DRAG = 4.2 // exponential velocity drag (1/s)
const MAX_SPEED = 4 // world-units/s safety clamp
const CURSOR_FORCE_K = 10 // xy force at strength 1
const CURSOR_ZFORCE_K = 4 // z bulge force at strength 1

// All per-frame CPU-derived inputs (kernels can't read node uniforms — those live in the composer's
// bind groups — so the renderer writes these each frame from getCpuValue). rowX/rowY/rowZ are the
// camera rotation matrix rows (built on the CPU each frame, w unused).
const SimParams = d.struct({
    rowX: d.vec4f, rowY: d.vec4f, rowZ: d.vec4f,
    dt: d.f32, time: d.f32, snap: d.f32,
    gridW: d.f32, gridH: d.f32, outW: d.f32, outH: d.f32,
    depth: d.f32, dragMul: d.f32, wobbleAmp: d.f32,
    depthShading: d.f32, spacing: d.f32, particleSize: d.f32, aspect: d.f32,
    zoom: d.f32, transX: d.f32, transY: d.f32,
    pointerX: d.f32, pointerY: d.f32, cursorForce: d.f32, cursorZForce: d.f32, cursorRadSq: d.f32,
})

// One layout for the whole pipeline (every kernel binds the same group; the sim additionally reads
// the late-bound child RTT). `src` is the child; `outTex` is the resolved field.
const fieldLayout = tgpu.bindGroupLayout({
    pos: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
    vel: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
    col: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
    accumR: {storage: d.arrayOf(d.atomic(d.u32), OUT_CELLS), access: 'mutable'},
    accumG: {storage: d.arrayOf(d.atomic(d.u32), OUT_CELLS), access: 'mutable'},
    accumB: {storage: d.arrayOf(d.atomic(d.u32), OUT_CELLS), access: 'mutable'},
    accumW: {storage: d.arrayOf(d.atomic(d.u32), OUT_CELLS), access: 'mutable'},
    params: {uniform: SimParams},
    src: {texture: d.texture2d(d.f32)},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// The `depthSource` prop → the std channel-extractor menu's canonical mode numbering; the
// relief integrator calls the picked extractor as its colorFrom slot.
const DEPTH_SOURCE_MODES: Record<string, number> = {
    luminance: 0, luminanceInverted: 1, red: 2, green: 3, blue: 4, saturation: 5, alpha: 6,
}

// ── The composed physics + render (std/sim vocabulary) ──────────────────────────────────────

/**
 * Sim factory — bakes the depth-channel extractor into the relief-family integrator: bilinear-
 * sample the child at the home UV, derive the target Z from the chosen channel, then integrate
 * the FORCE-BASED physics (spring toward home/target + the view-space cursor magnet) through the
 * shared semi-implicit Euler. A particle SNAPS to its target until it has actually seen child
 * content (the vel.w latch) — no start lurch, no fly-in on load.
 */
export function makeParticleFieldSimKernel(channelMode: number) {
    return integrator.relief(fieldLayout, {
        channel: channel.byMode[channelMode] ?? channel.luminance,
        cursor: force.cursorInView(fieldLayout, {perspK: PERSP_K}),
        stiffness: STIFF,
        maxSpeed: MAX_SPEED,
        minAlpha: MIN_ALPHA,
        name: 'particleFieldSim',
    })
}

/** The render subsystem: the camera + colorFrom-layer variant — depth-weighted RGBW color
 *  splat (near particles dominate → pseudo-occlusion) + the weighted-average resolve. */
function particleFieldRender(shape: string, out: [number, number]) {
    return renderAgents.relief(fieldLayout, {
        shape,
        splatRCap: SPLAT_R,
        aaW: AA_W,
        minAlpha: MIN_ALPHA,
        fp: FP,
        wobbleFreq: WOBBLE_FREQ,
        perspK: PERSP_K,
        out,
        weightedColor: {alphaK: ALPHA_K},
        names: {splat: 'particleFieldSplat', resolve: 'particleFieldResolve'},
    })
}

export const makeParticleFieldSplatKernel = (shape: string) =>
    particleFieldRender(shape, [OUT_MAX, OUT_MAX]).splat.kernel as (x: number, y: number) => void
export const particleFieldSplatKernel = makeParticleFieldSplatKernel('dot')
export const particleFieldResolveKernel =
    particleFieldRender('dot', [OUT_MAX, OUT_MAX]).resolve.kernel as (x: number, y: number) => void

// ─────────────────────────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    count: number
    depthSource: string
    depth: number
    particleShape: string
    wobble: number
    particleSize: number
    depthShading: number
    zoom: number
    rotationX: number
    rotationY: number
    rotationZ: number
    offsetX: number
    offsetY: number
    cursorMode: string
    cursorStrength: number
    cursorRadius: number
}

const depthSourceOptions = [
    {value: 'luminance', label: 'Luminance'},
    {value: 'luminanceInverted', label: 'Luminance (Inverted)'},
    {value: 'red', label: 'Red'},
    {value: 'green', label: 'Green'},
    {value: 'blue', label: 'Blue'},
    {value: 'saturation', label: 'Saturation'},
    {value: 'alpha', label: 'Alpha'},
]
const cursorModeOptions = [
    {value: 'none', label: 'None'},
    {value: 'push', label: 'Push'},
    {value: 'pull', label: 'Pull'},
]

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ParticleField",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Stylize",
    description: "Explodes the child content into a breathing 3D field of particles — bright areas float toward you and dark areas recede, the whole relief can be orbited with the camera rotations, and the cursor physically throws particles that spring back home",
    requiresRTT: true,
    requiresChild: true,
    usesPointer: true,
    props: {
        count: {
            default: 15000,
            description: "Number of particles — how finely the child is sampled into the field",
            ui: {type: 'range', min: 1000, max: COUNT_MAX, step: 500, label: 'Count', group: 'Field'}
        },
        depthSource: {
            default: 'luminance',
            compileTime: true,
            description: "Which channel of the child drives each particle's depth toward or away from the camera",
            ui: {type: 'select', options: depthSourceOptions, label: 'Depth Source', group: 'Field'}
        },
        depth: {
            default: 0.7,
            description: "Depth of the relief — how far bright and dark areas rise toward and sink away from the camera",
            ui: {type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Depth', group: 'Field'}
        },
        particleShape: {
            default: 'dot',
            compileTime: true,
            description: "What each particle is drawn as — a crisp dot or square, or a soft glow puff",
            ui: {type: 'select', options: agents.pointShapeOptions, label: 'Shape', group: 'Look'}
        },
        particleSize: {
            default: 0.75,
            description: "Size of each particle relative to the grid spacing",
            ui: {type: ['range', 'map'], min: 0.3, max: 3, step: 0.01, label: 'Particle Size', group: 'Look'}
        },
        depthShading: {
            default: 0.6,
            description: "How much far particles dim into the distance — the main cue that sells the 3D relief",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Depth Shading', group: 'Look'}
        },
        wobble: {
            default: 0.35,
            description: "Gentle idle drift that keeps the field breathing even on a still image",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Wobble', group: 'Motion'}
        },
        zoom: {
            default: 1,
            description: "Camera distance — dolly the whole field closer or further away",
            ui: {type: ['range', 'map'], min: 0.3, max: 2.5, step: 0.01, label: 'Zoom', group: 'Camera'}
        },
        rotationX: {
            default: 0,
            description: "Camera orbit around the horizontal axis — tilt the relief toward or away from you",
            ui: {type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Rotate X', group: 'Camera'}
        },
        rotationY: {
            default: 0,
            description: "Camera orbit around the vertical axis — view the particle extrusion from the side",
            ui: {type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Rotate Y', group: 'Camera'}
        },
        rotationZ: {
            default: 0,
            description: "Camera roll around the view axis",
            ui: {type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Rotate Z', group: 'Camera'}
        },
        offsetX: {
            default: 0,
            description: "Shift the whole field horizontally across the frame (in screen widths)",
            ui: {type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Offset X', group: 'Camera'}
        },
        offsetY: {
            default: 0,
            description: "Shift the whole field vertically across the frame (in screen heights) — tilt it flat, then sink it toward the bottom like a ground plane",
            ui: {type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Offset Y', group: 'Camera'}
        },
        cursorMode: {
            default: 'push',
            description: "How the cursor disturbs the field — push throws particles away and toward you, pull gathers them in",
            ui: {type: 'select', options: cursorModeOptions, label: 'Cursor Mode', group: 'Interaction'}
        },
        cursorStrength: {
            default: 0.8,
            description: "Strength of the cursor force — how hard particles get thrown before springing back",
            ui: {type: 'range', min: 0, max: 2, step: 0.01, label: 'Cursor Strength', group: 'Interaction'}
        },
        cursorRadius: {
            default: 0.2,
            description: "Reach of the cursor field, in screen units",
            ui: {type: 'range', min: 0.05, max: 1, step: 0.01, label: 'Cursor Radius', group: 'Interaction'}
        }
    },

    // The force-based simulation + additive render as a composed agentSim. Per frame — sim
    // (sample child → target Z → forces + integrate) → relief splat (rotate + project +
    // accumulate) → weighted-color resolve (which also re-zeroes the accumulators). The child
    // input binds LATE (the noun's childTexture wiring) since its RTT is allocated after
    // composition. `depthSource` and `particleShape` are compile-time (baked extractor + shape);
    // `count` refits the grid and dispatches at runtime; everything else is a runtime uniform.
    ...agentSim<typeof SimParams>({
        layout: fieldLayout,
        params: SimParams,
        maxAgents: MAX_PARTICLES,
        // Phone pixel density hides the difference between 12k and 40k particles far better than
        // it hides the frame time, so cap the count rather than letting a desktop-authored preset
        // drag a phone down. Build-time constant (SSR/tests resolve desktop), like OUT_MAX.
        countCap: {desktop: COUNT_MAX, mobile: COUNT_MAX_MOBILE},
        // The child RTT only exists after the pass manager allocates boundaries.
        externalKeys: ['src'],
        childTexture: {externalKey: 'src'},
        // Square MAX fallback — bake overrides the extent to match the frame aspect.
        output: {key: 'outTex', name: 'field', size: [OUT_MAX, OUT_MAX], format: STATE_FORMAT},
        bake: (params) => {
            const {getCpuValue, dimensions} = params

            // Output texture matches the frame aspect (square texels) within OUT_MAX — decided
            // here per composition and handed to the noun via `outputSize`.
            const aspect0 = dimensions.height > 0 ? dimensions.width / dimensions.height : 1
            let outW = OUT_MAX
            let outH = OUT_MAX
            if (aspect0 >= 1) outH = Math.max(Math.round(OUT_MAX / aspect0), 1)
            else outW = Math.max(Math.round(OUT_MAX * aspect0), 1)

            const channelMode = DEPTH_SOURCE_MODES[(getCpuValue('depthSource') as string) || 'luminance'] ?? 0
            const render = particleFieldRender((getCpuValue('particleShape') as string) || 'dot', [outW, outH])

            return {
                pipelines: {
                    // 2D grid dispatch so the grid coordinates arrive as clean u32 (see the
                    // relief integrator — a 1D `idx / gw` reconstruction is unsafe in TGSL).
                    sim: {kernel: makeParticleFieldSimKernel(channelMode), threads: 'grid'},
                    splat: render.splat,
                    resolve: render.resolve,
                },
                program: ['sim', 'splat', 'resolve'],
                outputSize: [outW, outH],
                setup: {outW, outH, aspect0},
            }
        },
        frame: (sys, {getCpuValue}, setupRaw) => {
            const {outW, outH, aspect0} = setupRaw as {outW: number; outH: number; aspect0: number}
            const g = makeCpuValueGetter(getCpuValue)
            let localTime = 0
            let frameIdx = 0
            const DEG_TO_RAD = constants.DEG_TO_RAD

            return (frameParams) => {
                // Falls back to the COMPOSITION aspect, not 16/9: the grid is refit to the aspect
                // every frame, so it must be sane before the first frame reports a size.
                const frame = readAgentFrame(frameParams, aspect0)
                const dt = frame.dt
                localTime += dt
                const aspect = frame.aspect

                // Refit the grid to the runtime count + aspect (the shared iso-grid recipe);
                // dispatch below uses these live dimensions, so the slider never recompiles.
                const count = sys.resolveCount(g('count', 15000), 500)
                const {w: GW, h: GH} = agentFrame.fitIsoGrid(count, aspect, 8, GRID_DIM_MAX)

                const cursorMode = (getCpuValue('cursorMode') as string) || 'push'
                const cursorSign = cursorMode === 'pull' ? -1 : 1
                const cursorStrength = cursorMode === 'none' ? 0 : g('cursorStrength', 0.5)
                const cursorRadius = g('cursorRadius', 0.22)

                // Camera rotation rows (Rz·Ry·Rx, y-down conjugation) — the shared camera recipe.
                const rows = agentFrame.cameraRowsYDown(
                    g('rotationX', 0) * DEG_TO_RAD,
                    g('rotationY', 0) * DEG_TO_RAD,
                    g('rotationZ', 0) * DEG_TO_RAD,
                )

                sys.writeParams({
                    rowX: d.vec4f(rows.rowX[0], rows.rowX[1], rows.rowX[2], 0),
                    rowY: d.vec4f(rows.rowY[0], rows.rowY[1], rows.rowY[2], 0),
                    rowZ: d.vec4f(rows.rowZ[0], rows.rowZ[1], rows.rowZ[2], 0),
                    dt, time: localTime, snap: frameIdx === 0 ? 1 : 0,
                    gridW: GW, gridH: GH, outW, outH,
                    depth: g('depth', 0.7),
                    dragMul: agentFrame.expDecay(DRAG, dt),
                    wobbleAmp: g('wobble', 0.35),
                    depthShading: Math.min(Math.max(g('depthShading', 0.6), 0), 1),
                    // The true isotropic pitch: GW and GH clamp to [8, 256] independently, so at
                    // extreme aspects the vertical pitch can be the tighter of the two — sizing
                    // dots off the horizontal one alone would overlap them vertically.
                    spacing: Math.min(outW / GW, outH / GH),
                    particleSize: g('particleSize', 1.3),
                    aspect,
                    zoom: Math.min(Math.max(g('zoom', 1), 0.05), 10),
                    transX: g('offsetX', 0),
                    transY: g('offsetY', 0),
                    pointerX: frame.pointerX,
                    pointerY: frame.pointerY,
                    cursorForce: cursorSign * cursorStrength * CURSOR_FORCE_K,
                    cursorZForce: cursorSign * cursorStrength * CURSOR_ZFORCE_K,
                    cursorRadSq: cursorRadius * cursorRadius,
                })
                frameIdx++

                return sys.frame({count, grid: [GW, GH]})
            }
        },
        // Sample the resolved field over the full canvas (rgba16f is filterable). GPU-free
        // (no device / no compute) → child passthrough so SSR/tests still show content.
        fragment: {output: 'field', fallback: 'child'},
    }),
})

export default componentDefinition
