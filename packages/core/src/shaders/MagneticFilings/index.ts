import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {agentSim, renderAgents, torque, field, integrator, agentFrame} from "@coreroot/std/sim/agents"
import {tgpu, d, agents} from "@coreroot/gpu/kit"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

// ── Simulation overview ──────────────────────────────────────────────────────────────────────
// Iron filings on paper: thousands of tiny oriented slivers anchored on a jittered grid that swing
// to ALIGN with a magnetic field emanating from the cursor, revealing the field lines. The physical
// read lives entirely in the ROTATION dynamics — moving the magnet sends a wave of filings flipping
// and settling, not sliding around. Every filing carries a screen-proportional world position (x ∈
// [0, aspect], y ∈ [0, 1], so 1 unit == the canvas HEIGHT and every distance is isotropic — the
// field is a true circle, no skew on wide canvases), an angle θ and an angular velocity ω. The home
// (rest) position is derived each frame from the filing's index on a jittered grid (no persisted
// home needed); only the small offset-from-home, θ and ω are stored.
//
// Runtime `count` uses the WebGPU-boids trick (engine policy on the harness): a static MAX_FILINGS
// dispatch bound with the active count as a uniform, so the slider never recompiles. `fieldType`
// and `restOrientation` are runtime uniforms too (a select in the field/torque parts). Only `shape`
// and `colorSpace` bake kernel variants (compile-time).
const MAX_FILINGS = 12288 // ≥ the 12000 count ceiling, dispatch/loop bound
// Square render/accumulator resolution; the fragment bilinear-upsamples to screen. Filings are
// sized against the harness's FIXED SIZE_REF_RES, not this live (device-tiered) value — otherwise
// shrinking RES on mobile would shrink the world-space texel and inflate apparent size.
const RES = resolveRenderRes({desktop: 1024, mobile: 768})
const STATE_FORMAT = 'rgba16float' as const
// Sanity clamp on a filing's body half-width, in texels — NOT a splat-window bound. The window is
// computed per-filing from its actual footprint, so it can never slice a large shape into a square.
const SPLAT_R = 14

// ── Torque physics (the core feel) ─────────────────────────────────────────────────────────────
// A filing is a NEMATIC director: it aligns with the field LINE, so θ and θ+π are equivalent and it
// must always take the SHORT way (never spin a full turn). The π-periodic sin(2Δ) restoring torque
// (kit `nematicTorque`) gives that analytically — stable equilibria at the two line-aligned angles,
// shortest-arc push, no explicit wrap. Torque ∝ ALIGN_BASE·fieldStrength·sin(2Δ).
const ALIGN_BASE = 45.0 // field alignment stiffness at full field strength
const REST_K = 6.0 // weak restoring torque toward each filing's rest orientation (dominant only where the field is faint)
// `response` maps to angular drag: higher = snappier, quicker to settle; lower = sluggish and floppy.
// Left underdamped across the whole range so a fast magnet pass leaves a visible settling wobble.
const DAMP_MIN = 3.0
const DAMP_SPAN = 9.0
// Angular-activity envelope: |ω| above a floor charges a per-filing excitement value that peaks
// instantly and cools over ~1s — a magnet sweep paints a glowing wave through the filings.
const OMEGA_REF = 6.0 // |ω| (rad/s) mapping to fully excited
const AGIT_COOL = 1.0 // exponential cooling rate (1/s) → ~1s decay

// ── Position dynamics (gentle — orientation is the show) ────────────────────────────────────────
// Filings stay anchored near home: a weak spring pulls the offset back to zero, and a subtle pull
// toward the magnet (scaled by field strength) lets them crowd a touch toward the pole like the real
// thing. Both deliberately soft so nothing drifts far.
const HOME_K = 2.5 // spring-home decay rate (1/s)
const PULL_K = 0.25 // magnet in-pull speed (world units/s) at full field strength
const MAX_OFFSET = 0.12 // hard clamp on how far a filing may wander from home (world units)
const OVERSCAN = 0.14 // home-grid border beyond the viewport — backfills edges when the magnet pulls filings inward (≥ MAX_OFFSET + splat extent)

// All per-frame CPU-derived inputs. Kernels can't read node uniforms (those live in the composer's
// bind groups), so the renderer writes these each frame from getCpuValue. Written in full every frame
// (strict uniform layout — no partial patch of a plain struct).
const SimParams = d.struct({
    colA: d.vec4f, colB: d.vec4f,
    count: d.f32,
    dt: d.f32, aspect: d.f32, domainX: d.f32,
    cursorX: d.f32, cursorY: d.f32, axisX: d.f32, axisY: d.f32,
    fieldType: d.f32, restMode: d.f32,
    strength: d.f32, reachSq: d.f32,
    alignK: d.f32, damping: d.f32, restK: d.f32, pullK: d.f32, homeK: d.f32,
    omegaRef: d.f32, agitCool: d.f32,
    gridCols: d.f32, cellW: d.f32, cellH: d.f32, jitter: d.f32,
    bodyR: d.f32,
})

// One layout for the whole pipeline. `agents` xyzw = (offX, offY, θ, ω); updated IN PLACE (a
// Gauss-Seidel relaxation — no ping-pong buffer, and each filing only reads its own slot so there is
// no cross-agent hazard here at all). `agit` is the per-filing angular-activity envelope. `accumE` is
// additive splat energy, `accumS` the agitation-weighted copy (their ratio → per-texel average
// excitement for the rest→excited color ramp).
const simLayout = tgpu.bindGroupLayout({
    agents: {storage: d.arrayOf(d.vec4f, MAX_FILINGS), access: 'mutable'},
    agit: {storage: d.arrayOf(d.f32, MAX_FILINGS), access: 'mutable'},
    accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    accumS: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
    params: {uniform: SimParams},
    outTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

// ── Spawn (module scope — the seeded rest state IS the look) ─────────────────────────────────────────────────────────────────────

/** Spawn: park every filing at its home (zero offset), scattered starting angle, zero spin. */
export const filingsInitKernel = agents.makeRestingDirectorInit(simLayout, 'filingsInit')

// ── The composed physics + render (std/sim vocabulary) ──────────────────────────────────────

/** The jittered-grid home every filing anchors to (overscanned so pulled-in edges backfill). */
const filingsHome = field.jitteredGridHome(simLayout, {overscan: OVERSCAN})

/** The cursor's magnetic field: dipole lobes along the CPU-smoothed motion axis, or radial
 *  spokes — `fieldType` selects at runtime. */
const filingsField = field.dipoleOrRadial(simLayout)

/**
 * The integrator: the nematic align-to-field torque plus the weak rest-orientation torque,
 * folded in that order, with the orientation-family integrator owning the angular drag, the
 * |ω| activity envelope, and the gentle position dynamics (field pull + spring home + clamp).
 */
export const filingsUpdateKernel = integrator.orientation2d(simLayout, {
    home: filingsHome,
    fieldAt: filingsField,
    torques: [torque.alignToField(simLayout), torque.rest(simLayout)],
    maxOffset: MAX_OFFSET,
    name: 'filingsUpdate',
})

/** The render subsystem: angle-oriented slivers splatted in world units (no comet — the heading
 *  is a compass needle, not a velocity), resolved through the rest→excited ramp. */
function filingsRender(shape: string, colorSpaceMode: number) {
    return renderAgents.orientedWorld(simLayout, {
        shape,
        res: RES,
        splatRCap: SPLAT_R,
        heading: 'angle',
        agitation: 'buffer',
        comet: false,
        home: filingsHome,
        ramp: {colorSpace: colorSpaceMode, trails: 'none'},
        names: {splat: 'filingsSplat', resolve: 'filingsResolve'},
    })
}

export const makeFilingsSplatKernel = (shape: string) => filingsRender(shape, 2).splat.kernel as (i: number) => void
export const makeFilingsResolveKernel = (mode: number) => filingsRender('streak', mode).resolve.kernel as (x: number, y: number) => void
export const filingsSplatKernel = makeFilingsSplatKernel('streak')
export const filingsResolveKernel = makeFilingsResolveKernel(2)

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    count: number
    fieldType: string
    strength: number
    response: number
    reach: number
    restOrientation: string
    shape: string
    size: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "MagneticFilings",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Thousands of tiny iron filings scattered on paper that swing to align with a magnetic field around the cursor, tracing out the field lines — a dipole whose axis follows the cursor's motion draws the classic two-lobed swirl, and every sweep of the magnet sends a glowing wave of filings flipping and settling",
    requiresRTT: false,
    requiresChild: false,
    usesPointer: true,
    props: {
        colorA: {
            default: "#be1ef9",
            transform: transformColor,
            description: "Color of filings resting quietly, aligned and still",
            ui: { type: 'color', label: 'Rest Color', group: 'Colors' }
        },
        colorB: {
            default: "#6326ff",
            transform: transformColor,
            description: "Color filings flash toward as they swing and settle — a wave of it follows the moving magnet",
            ui: { type: 'color', label: 'Excited Color', group: 'Colors' }
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for the rest→excited color ramp',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        },
        shape: {
            default: 'dot',
            compileTime: true,
            description: "How each filing is drawn — a slim streak or arrow reads as a compass needle, a box as a chunky filing, a dot or soft glow for a finer grain",
            ui: { type: 'select', options: agents.orientedShapeOptions, label: 'Shape', group: 'Filings' }
        },
        count: {
            default: 8000,
            description: "Number of iron filings",
            ui: { type: 'range', min: 500, max: 12000, step: 500, label: 'Count', group: 'Filings' }
        },
        size: {
            default: 1,
            description: "Size of each filing",
            ui: { type: 'range', min: 0.5, max: 3, step: 0.05, label: 'Size', group: 'Filings' }
        },
        restOrientation: {
            default: 'random',
            description: "Which way filings point where the field can't reach them — randomly, all horizontal, or all vertical",
            ui: { type: 'select', options: [
                { label: 'Random', value: 'random' },
                { label: 'Horizontal', value: 'horizontal' },
                { label: 'Vertical', value: 'vertical' },
            ], label: 'Rest Orientation', group: 'Filings' }
        },
        fieldType: {
            default: 'dipole',
            description: "The magnetic field the filings reveal — a dipole (bar-magnet lobes that follow the cursor's motion) or a radial monopole (spokes out from the cursor)",
            ui: { type: 'select', options: [
                { label: 'Dipole', value: 'dipole' },
                { label: 'Radial', value: 'radial' },
            ], label: 'Field Type', group: 'Field' }
        },
        strength: {
            default: 1.7,
            description: "How strongly the magnet grips the filings — higher snaps more of them into sharp alignment",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Strength', group: 'Field' }
        },
        reach: {
            default: 0.6,
            description: "How far the magnet's influence spreads (in screen heights) — larger reveals more of the field at once",
            ui: { type: 'range', min: 0.1, max: 1, step: 0.01, label: 'Reach', group: 'Field' }
        },
        response: {
            default: 0.5,
            description: "How briskly filings swing and settle — low is loose and floppy with a long wobble, high snaps into place quickly",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Response', group: 'Field' }
        }
    },

    // The whole filings sim as a composed agentSim: per frame — integrate (field sample + the
    // declared torques + gentle position spring) → oriented shape splat → ramp resolve (which
    // also clears the accumulators as it reads them). `shape` and `colorSpace` are compile-time
    // (baked kernel variants); `count`, `fieldType`, `restOrientation` are runtime. GENERATOR.
    ...agentSim<typeof SimParams>({
        layout: simLayout,
        params: SimParams,
        maxAgents: MAX_FILINGS,
        countCap: {desktop: MAX_FILINGS, mobile: 6000}, // runtime cap only — the slider range is unchanged
        output: {key: 'outTex', name: 'filingsTexture', size: [RES, RES], format: STATE_FORMAT},
        bake: ({getCpuValue}) => {
            const render = filingsRender(
                (getCpuValue('shape') as string) || 'streak',
                (getCpuValue('colorSpace') as number) ?? 2,
            )
            return {
                pipelines: {
                    init: {kernel: filingsInitKernel, threads: 'max'},
                    update: {kernel: filingsUpdateKernel, threads: 'agents'},
                    splat: render.splat,
                    resolve: render.resolve,
                },
                initStep: 'init',
                program: ['update', 'splat', 'resolve'],
            }
        },
        frame: (sys, {getCpuValue}) => {
            const g = makeCpuValueGetter(getCpuValue)
            // The dipole axis follows the cursor's recent travel, CPU-smoothed with a teleport
            // guard (a jump wider than a quarter screen-height is the pointer re-entering the
            // canvas, not a stroke) — the shared motion-axis recipe.
            const motionAxis = agentFrame.createMotionAxis({smoothing: 0.1, teleport: 0.25})
            return (frameParams) => {
                const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams)
                const domainX = Math.max(aspect, 0.01)

                const count = sys.resolveCount(g('count', 5000))

                // Jittered-grid layout sized to the active count so coverage stays even at any
                // count — the shared grid-fit recipe over the OVERSCANNED domain (off-screen
                // filings backfill pulled-in edges).
                const {cols, cellW, cellH} = agentFrame.fitJitteredGrid(count, domainX, OVERSCAN)

                // Pointer arrives in screen UV [0,1] (y down). Lift x into world (× aspect).
                const cursorX = pointerX * aspect
                const cursorY = pointerY
                const axis = motionAxis.update(cursorX, cursorY)

                const fieldTypeStr = (getCpuValue('fieldType') as string) ?? 'dipole'
                const fieldType = fieldTypeStr === 'radial' ? 1 : 0
                const restStr = (getCpuValue('restOrientation') as string) ?? 'random'
                const restMode = restStr === 'horizontal' ? 1 : restStr === 'vertical' ? 2 : 0

                const strength = g('strength', 1)
                const reach = g('reach', 0.35)
                const response = Math.min(Math.max(g('response', 0.5), 0), 1)

                const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
                const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined

                // Body half-width in world units (size → texels): a size-1 streak spans ~a few
                // texels (tiny slivers). The splat window follows this, so bigger sizes are
                // never clipped.
                const bodyR = Math.min(Math.max(g('size', 1), 0.5), 3) * 1.7 / SIZE_REF_RES

                sys.writeParams({
                    colA: d.vec4f(colA?.x ?? 0.522, colA?.y ?? 0.573, colA?.z ?? 0.639, colA?.w ?? 1),
                    colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 0.616, colB?.z ?? 0.361, colB?.w ?? 1),
                    count,
                    dt, aspect, domainX,
                    cursorX, cursorY, axisX: axis.x, axisY: axis.y,
                    fieldType, restMode,
                    strength, reachSq: reach * reach,
                    alignK: ALIGN_BASE, damping: DAMP_MIN + response * DAMP_SPAN,
                    restK: REST_K, pullK: PULL_K, homeK: HOME_K,
                    omegaRef: OMEGA_REF, agitCool: AGIT_COOL,
                    gridCols: cols, cellW, cellH, jitter: 0.85,
                    bodyR,
                })

                return sys.frame({count})
            }
        },
        // Bilinear-sample the resolved field over the full canvas (rgba16f is filterable).
        // GPU-free (no device) → transparent, so it composites cleanly as a background layer.
        fragment: {output: 'filingsTexture', fallback: 'transparent'},
    }),
})

export default componentDefinition
