/**
 * The Stable-Fluids solver (Jos Stam / Pavel Dobryakov), as one parameterized kernel set.
 *
 * Fog, Smoke, SmokeFlow, InkFlow, ParticleFlow and SmokeFill each carried a line-for-line copy of
 * the same 8–11 compute kernels. This module owns them once; the shaders keep only what is genuinely
 * theirs — the emission/force splat, the CPU parameter derivation, and the fragment color mapping.
 *
 * ── What the solver is ────────────────────────────────────────────────────────────────────────
 * A fixed N×N grid of velocity + optional dye, advanced each frame by an ordered chain of
 * single-purpose compute passes:
 *
 *   curl → vorticity confinement → divergence → J× Jacobi pressure → gradient subtract
 *        → advect velocity (+copy) → advect dye (+copy) → output
 *
 * Velocity is `vec4f`: `xy` is the flow, `z` is the curl scratch written by the curl pass and read
 * by the vorticity pass. Pressure is warm-started — the divergence pass decays the previous frame's
 * pressure instead of clearing it, so Jacobi convergence accumulates across frames and 10 iterations
 * per frame suffice for these quasi-steady flows.
 *
 * ── What the consuming shader owns ────────────────────────────────────────────────────────────
 * Per C4, the solver does not allocate: the shader declares the bind-group layouts (so its uniform
 * and buffer surface stays visible in one place), allocates the state buffers, and passes the layouts
 * in. The solver bakes kernels against them.
 *
 * The layout MUST use these entry names, because the kernel bodies reference them directly:
 *
 *   velA, velB       mutable `arrayOf(vec4f, N*N)`  — velocity ping-pong
 *   dyeA, dyeB     mutable `arrayOf(vec4f, N*N)`  — dye ping-pong (omit when `dye: 'none'`)
 *   pressure         mutable `arrayOf(f32, N*N)`
 *   divergence       mutable `arrayOf(f32, N*N)`
 *   maskBuf          readonly `arrayOf(vec4f, N*N)` — `.x` = 1 inside the fluid (only `solidMask`)
 *   velOutTex        write-only storage texture     — only `publishVelocityTexture`
 *   params           uniform struct, see below
 *
 * And the params struct MUST expose these members (its own extra members are free — every consumer
 * carries emitter/cursor fields the solver never reads):
 *
 *   dt            f32  seconds this step
 *   curlStrength  f32  vorticity confinement gain
 *   velFade       f32  velocity dissipation rate (was a per-shader module constant; it is a uniform
 *                      now so one kernel serves every consumer — write your constant each frame)
 *   dyeFade       f32  dye dissipation rate       (only when `dyeDissipation`)
 *   colorDecay    f32  age-channel advance rate   (only when `ageAdvance`)
 *
 * ── Compile-time vs runtime (C5) ──────────────────────────────────────────────────────────────
 * EVERYTHING in `StableFluidsOptions` is structural: read at composition time, baked into the kernel
 * bodies, part of the pipeline hash. The five params members above are the only runtime values.
 */
import {tgpu, d, std, colorMixing, constants, sdf} from '../kit'
import * as noise from '../kit/noise'
import {gaussianBrushSq, gaussianBrushFnSq, type GaussianBrushOptions} from './simShared'
import {createGuardedCompute, type ComputeStep, type KitComputePipeline} from '../compute'
import type {TgpuRoot} from 'typegpu'

/**
 * The storage-texture form every fluid consumer publishes through: rgba16float, write-only. Half
 * floats are enough for a velocity or dye field and they give the fragment hardware-bilinear
 * filtering on the upsample, which a 32-bit format would not on most Android GPUs.
 */
export type FluidStorageTexture = d.textureStorage2d<'rgba16float', 'write-only'>

// ── Layout contracts ───────────────────────────────────────────────────────────────────────────
// Hand-written structural views of the shader's bind-group layout. A real `tgpu.bindGroupLayout` is
// assignable to these, which is what lets the kernel bodies below index the shader's buffers.

/** The solve uniforms every kernel set reads. */
export interface FluidSolveParams {
    readonly dt: number
    readonly curlStrength: number
    readonly velFade: number
}

/** The additional uniforms the dye passes read. */
export interface FluidDyeParams extends FluidSolveParams {
    readonly dyeFade: number
    readonly colorDecay: number
    // ambientWind ABI (present only when the option is enabled).
    readonly ambient: number
    readonly ambientFreq: number
    readonly ambientTime: number
}

/**
 * The union of every layout entry the solver can touch.
 *
 * Which entries are actually required depends on the options, and TypeScript cannot express
 * "`dyeA` is required iff `dye !== 'none'`" across one signature — so the builder takes the core
 * shape and casts once internally. A layout missing an entry its options need fails at
 * `tgpu.resolve` time, which every consumer's resolve gate exercises.
 */
export interface FluidSolverLayout {
    readonly $: {
        readonly velA: d.v4f[]
        readonly velB: d.v4f[]
        readonly pressure: number[]
        readonly divergence: number[]
        readonly params: FluidSolveParams
    }
}

/** The standard splat-stamp ABI (`sparams` uniform) the impulse family reads. */
export interface FluidSplatParams {
    readonly posX: number
    readonly posY: number
    readonly velX: number
    readonly velY: number
    readonly radius: number
}

/** A solver layout that also carries the splat-stamp uniform. */
export interface FluidImpulseLayout extends FluidSolverLayout {
    readonly $: FluidSolverLayout['$'] & {readonly sparams: FluidSplatParams}
}

interface FullFluidLayout {
    readonly $: {
        readonly velA: d.v4f[]
        readonly velB: d.v4f[]
        readonly dyeA: d.v4f[]
        readonly dyeB: d.v4f[]
        readonly pressure: number[]
        readonly divergence: number[]
        readonly maskBuf: d.v4f[]
        readonly velOutTex: FluidStorageTexture
        readonly params: FluidDyeParams
    }
}

/** The output pass's layout: the dye buffer read-only, plus the storage texture it publishes to. */
export interface FluidOutputLayout {
    readonly $: {
        readonly dyeA: d.v4f[]
        readonly outTex: FluidStorageTexture
    }
}

// ── Options ────────────────────────────────────────────────────────────────────────────────────

/**
 * How the grid edges behave.
 *
 * - `clamped` — neighbour lookups clamp to the grid, and the divergence pass reflects the wall-normal
 *   velocity at the border (free-slip): the fluid is in a box.
 * - `toroidal` — neighbour lookups and advection backtraces wrap modulo N: the field is seamless and
 *   has no walls at all. Fog uses this so its cloud has no visible frame.
 */
export type FluidBoundary = 'clamped' | 'toroidal'

/**
 * What the dye field carries.
 *
 * - `none` — velocity only. The consumer reads the field for something else (ParticleFlow advects
 *   particles through it), so there is no dye to advect, copy or publish.
 * - `densityAge` — `x` = density, `y` = age in [0, 1]. The fragment colors density by age, which is
 *   what gives smoke its fresh→aged ramp.
 * - `rgb` — `xyz` = color. The dye IS the picture (InkFlow).
 */
export type FluidDyeMode = 'none' | 'densityAge' | 'rgb'

export interface StableFluidsOptions {
    /** Grid resolution per axis. The kernels are dispatched 2D over N×N. */
    n: number
    /**
     * Prefix for every kernel's `$name`. Kernel names appear in the emitted WGSL, so this is what
     * keeps each consumer's snapshots stable and keeps two fluid shaders in one tree from colliding
     * on WGSL identifiers (C3).
     */
    namePrefix: string
    boundary: FluidBoundary
    dye: FluidDyeMode
    /** Divide advected dye by `1 + dyeFade·dt` and floor it at zero. Fog's fog is permanent. */
    dyeDissipation?: boolean
    /** Advance the age channel by `dt · 0.4 · colorDecay` (capped at 1). `densityAge` only. */
    ageAdvance?: boolean
    /**
     * Gate the fluid on `maskBuf.x`: cells outside the mask are solid walls in the divergence pass,
     * and advected velocity/dye is zeroed there. SmokeFill uses it to confine smoke to a shape.
     * `clamped` only — a toroidal field with interior walls is not a configuration we ship.
     */
    solidMask?: boolean
    /**
     * Clamp the post-projection velocity magnitude to this many grid cells per second. Fog needs it
     * because a permanent, never-dissipating field can otherwise accumulate enough energy over
     * minutes to advect further than one cell per step and go unstable.
     */
    velocityCap?: number
    /** Also `textureStore` the solved velocity into `velOutTex` from the copy pass. */
    publishVelocityTexture?: boolean
    /**
     * Fold an ambient divergence-free breeze into the vorticity pass: an in-place-evolving
     * curl-noise force added to velocity BEFORE the pressure projection (curl noise is itself
     * divergence-free, so the projection preserves it). Extends the params ABI with
     * `ambient`, `ambientFreq`, `ambientTime` f32 members; `gain` scales the force and is baked.
     * The noise taps hide behind a uniform-valued `ambient > 0` branch, so the mode that never
     * uses the breeze pays nothing.
     */
    ambientWind?: {gain: number}
    /** Per-frame pressure retention in the divergence pass (the warm start). Default 0.8. */
    pressureDecay?: number
}

/**
 * A velocity-only Gaussian impulse splat: kick momentum into the field around the stamp
 * centre (ADDED, so overlapping stamps build momentum). The dye-less member of the splat
 * family — pair with `dye: 'none'` solvers. Reads the standard `sparams` splat ABI
 * (posX/posY/velX/velY/radius).
 */
export function buildVelocityImpulseKernel(
    layout: FluidImpulseLayout,
    opts: {n: number; namePrefix: string},
) {
    const {n, namePrefix} = opts
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.sparams
        const idx = cy * d.u32(n) + cx
        const cellX = d.f32(cx) + 0.5
        const cellY = d.f32(cy) + 0.5
        const dx = cellX - p.posX
        const dy = cellY - p.posY
        const distSq = dx * dx + dy * dy
        const radSq = std.max(p.radius * p.radius, d.f32(1.0))
        const inf = gaussianBrushSq(distSq, radSq)
        const vel = layout.$.velA[idx]
        layout.$.velA[idx] = d.vec4f(vel.x + p.velX * inf, vel.y + p.velY * inf, vel.z, 0.0)
    }).$name(`${namePrefix}Force`)
}

// ── The dye-splat family ───────────────────────────────────────────────────────────────────────
// Every dye-carrying fluid injects material the same way: a Gaussian influence around a stamp
// centre, a density/color write into the dye field, and a velocity change. What differs is data:
// whether the dye write is a paint-over mix (brush) or an additive emission with age rejuvenation
// (emitter), whether the injected velocity fans across a cone, whether a cursor shove and a solid
// mask participate. Those are compile-time options here (C5) — the per-frame values ride the ABIs.

/** The brush-splat `sparams` ABI: the impulse ABI plus two working-space color endpoints. */
export interface FluidBrushSplatParams extends FluidSplatParams {
    readonly colR: number
    readonly colG: number
    readonly colB: number
    readonly col2R: number
    readonly col2G: number
    readonly col2B: number
    /** Mix position between the endpoints (0 degenerates to endpoint A). */
    readonly mixT: number
    /** Paint-over coverage multiplier on the Gaussian influence. */
    readonly strength: number
}

/** A brush splat's layout: velocity + dye mutable, plus the per-stamp `sparams` uniform. */
export interface FluidBrushSplatLayout {
    readonly $: {
        readonly velA: d.v4f[]
        readonly dyeA: d.v4f[]
        readonly sparams: FluidBrushSplatParams
    }
}

/**
 * A dye BRUSH splat: paint the brush color over the dye field (a MIX — the newest color wins, so
 * hues stay crisp) and kick a velocity impulse, both Gaussian-weighted around the stamp centre.
 *
 * The brush color arrives as TWO endpoints CPU-preconverted into the compile-time working color
 * space (`convertP3ToMixSpaceCPU`) plus a mix position; the kernel blends them with the canonical
 * `mixPreconverted*` variant (which back-converts to P3-linear). One kernel per color space —
 * build the variants at module scope (the colorMixing "pre-transpiled variants" idiom).
 */
export function buildBrushSplatKernel(
    layout: FluidBrushSplatLayout,
    opts: {n: number; namePrefix: string; colorSpace: number; brush?: GaussianBrushOptions},
) {
    const {n, namePrefix, colorSpace} = opts
    const mixFn = colorMixing.mixPreconvertedVariants[colorSpace as keyof typeof colorMixing.mixPreconvertedVariants]
        ?? colorMixing.mixPreconvertedLinear
    const brushWeight = gaussianBrushFnSq(opts.brush ?? {edgeShift: false})
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.sparams
        const idx = cy * d.u32(n) + cx
        const cellX = d.f32(cx) + 0.5
        const cellY = d.f32(cy) + 0.5
        const dx = cellX - p.posX
        const dy = cellY - p.posY
        const distSq = dx * dx + dy * dy
        const radSq = std.max(p.radius * p.radius, d.f32(1.0))
        const inf = brushWeight(distSq, radSq)
        const w = std.clamp(inf * p.strength, d.f32(0.0), d.f32(1.0))
        const brush = mixFn(
            d.vec3f(p.colR, p.colG, p.colB),
            d.vec3f(p.col2R, p.col2G, p.col2B),
            1.0, 1.0, p.mixT,
        )
        const cell = layout.$.dyeA[idx]
        layout.$.dyeA[idx] = d.vec4f(
            std.mix(cell.x, brush.x, w),
            std.mix(cell.y, brush.y, w),
            std.mix(cell.z, brush.z, w),
            0.0,
        )
        const vel = layout.$.velA[idx]
        layout.$.velA[idx] = d.vec4f(vel.x + p.velX * inf, vel.y + p.velY * inf, vel.z, 0.0)
    }).$name(`${namePrefix}Splat${colorSpace}`)
}

/** The cursor-shove params ABI (the force/emitter kernels that let the mouse push the field). */
export interface FluidCursorForceParams {
    readonly cursorX: number
    readonly cursorY: number
    readonly cursorVelX: number
    readonly cursorVelY: number
    /** 1 while the cursor is actively pushing, 0 otherwise (multiplies the shove). */
    readonly mouseActive: number
    /** Squared cursor-influence radius in grid cells. */
    readonly mouseRadSq: number
}

/** The emitter-splat params ABI: a fixed/moving source injecting density+age dye and momentum. */
export interface FluidEmitterParams {
    readonly dt: number
    readonly emitX: number
    readonly emitY: number
    readonly emitVelX: number
    readonly emitVelY: number
    /** Emission radius in grid cells (the Gaussian's σ). */
    readonly emitRad: number
    readonly emitIntensity: number
    /** Buoyancy/weight: a y-velocity force proportional to local density. */
    readonly gravity: number
}

/** An emitter splat's layout: velocity + dye mutable, the per-frame `params` uniform. */
export interface FluidEmitterLayout {
    readonly $: {
        readonly velA: d.v4f[]
        readonly dyeA: d.v4f[]
        readonly params: FluidEmitterParams
    }
}

interface FullEmitterLayout {
    readonly $: {
        readonly velA: d.v4f[]
        readonly dyeA: d.v4f[]
        readonly maskBuf: d.v4f[]
        readonly params: FluidEmitterParams & FluidCursorForceParams & {
            readonly perpDirX: number
            readonly perpDirY: number
            readonly spreadFactor: number
            readonly emitGate: number
        }
    }
}

export interface EmitterSplatOptions {
    n: number
    namePrefix: string
    /**
     * Fan the injected velocity across an emission cone: cells offset along the perpendicular get
     * a proportional sideways component (`perpDirX/perpDirY/spreadFactor` ABI members).
     */
    cone: boolean
    /** Add a Gaussian cursor shove to the injected velocity (the `FluidCursorForceParams` ABI). */
    cursorPush: boolean
    /**
     * Multiply the whole emission by `emitGate` (a runtime 0–1 gate — SmokeFlow's cursor-speed
     * gate, so a slow graze emits a wisp and a flick a full puff). `cone`/`cursorPush` excluded.
     */
    gated?: boolean
    /** Gate emission, gravity and the cursor shove by `maskBuf.x` (pair with `solidMask` solvers). */
    masked?: boolean
    /** Density injected per second at full influence × intensity. */
    densityGain: number
    /** Rate (1/s) the local velocity blends toward the injected velocity at full influence. */
    velocityBlendGain: number
}

/**
 * A dye EMITTER splat: ADD density around the source (capped at 1), rejuvenate the age channel in
 * proportion to the fresh material, blend velocity toward the injected (optionally cone-fanned)
 * velocity, and apply the density-proportional gravity force — the `densityAge` member of the
 * splat family. Two shapes ship: the cone emitter (fixed source, optional cursor shove, optional
 * solid mask) and the gated puff (the source IS the cursor; emission scaled by `emitGate`).
 */
export function buildEmitterSplatKernel(
    layoutIn: FluidEmitterLayout,
    opts: EmitterSplatOptions,
) {
    const layout = layoutIn as unknown as FullEmitterLayout
    const {n, namePrefix, densityGain, velocityBlendGain} = opts
    const masked = opts.masked ?? false
    const gated = opts.gated ?? false
    // The y-force per unit density·dt·gravity — the shared "smoke has weight" constant.
    const gravityGain = n * 0.08
    if (gated && (opts.cone || opts.cursorPush || masked)) {
        throw new Error(
            `buildEmitterSplatKernel: 'gated' is the cursor-puff shape — it does not combine with ` +
            `cone/cursorPush/masked (got cone=${opts.cone}, cursorPush=${opts.cursorPush}, masked=${masked})`,
        )
    }
    if (!gated && (!opts.cone || !opts.cursorPush)) {
        throw new Error(
            `buildEmitterSplatKernel: the fixed-source shape ships as cone+cursorPush — pass both ` +
            `(got cone=${opts.cone}, cursorPush=${opts.cursorPush})`,
        )
    }

    if (gated) {
        return tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const cellX = d.f32(cx) + 0.5
            const cellY = d.f32(cy) + 0.5
            const dx = cellX - p.emitX
            const dy = cellY - p.emitY
            const distSq = dx * dx + dy * dy
            const radSq = std.max(p.emitRad * p.emitRad, d.f32(1.0))
            const influence = gaussianBrushSq(distSq, radSq) * p.emitGate
            const cell = layout.$.dyeA[idx]
            const oldDens = cell.x
            const addDens = p.emitIntensity * influence * p.dt * densityGain
            const newDens = std.min(oldDens + addDens, 1.0)
            const rejuvenate = addDens / std.max(newDens, d.f32(0.001))
            layout.$.dyeA[idx] = d.vec4f(newDens, cell.y * (1.0 - std.min(rejuvenate, 1.0)), 0.0, 0.0)
            const vel = layout.$.velA[idx]
            const br = std.min(influence * p.dt * velocityBlendGain, 1.0)
            const gravForce = p.gravity * oldDens * p.dt * gravityGain
            layout.$.velA[idx] = d.vec4f(std.mix(vel.x, p.emitVelX, br), std.mix(vel.y, p.emitVelY, br) + gravForce, vel.z, 0.0)
        }).$name(`${namePrefix}Splat`)
    }

    if (masked) {
        return tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const mask = layout.$.maskBuf[idx].x
            const cellX = d.f32(cx) + 0.5
            const cellY = d.f32(cy) + 0.5
            const dx = cellX - p.emitX
            const dy = cellY - p.emitY
            const distSq = dx * dx + dy * dy
            const radSq = std.max(p.emitRad * p.emitRad, d.f32(1.0))
            const influence = gaussianBrushSq(distSq, radSq) * mask
            const cell = layout.$.dyeA[idx]
            const oldDens = cell.x
            const addDens = p.emitIntensity * influence * p.dt * densityGain
            const newDens = std.min(oldDens + addDens, 1.0)
            const rejuvenate = addDens / std.max(newDens, d.f32(0.001))
            layout.$.dyeA[idx] = d.vec4f(newDens, cell.y * (1.0 - std.min(rejuvenate, 1.0)), 0.0, 0.0)
            const vel = layout.$.velA[idx]
            const perpOff = (dx * p.perpDirX + dy * p.perpDirY) / std.max(p.emitRad, d.f32(1.0))
            const vmSq = std.max(p.emitVelX * p.emitVelX + p.emitVelY * p.emitVelY, d.f32(1.0))
            const tvx = p.emitVelX + p.perpDirX * perpOff * p.spreadFactor * std.sqrt(vmSq)
            const tvy = p.emitVelY + p.perpDirY * perpOff * p.spreadFactor * std.sqrt(vmSq)
            const br = std.min(influence * p.dt * velocityBlendGain, 1.0)
            const gravForce = p.gravity * oldDens * p.dt * gravityGain * mask
            const cursorDX = cellX - p.cursorX
            const cursorDY = cellY - p.cursorY
            const cursorDistSq = cursorDX * cursorDX + cursorDY * cursorDY
            const cursorInf = gaussianBrushSq(cursorDistSq, std.max(p.mouseRadSq, d.f32(1.0))) * p.mouseActive * mask
            layout.$.velA[idx] = d.vec4f(
                std.mix(vel.x, tvx, br) + p.cursorVelX * cursorInf,
                std.mix(vel.y, tvy, br) + gravForce + p.cursorVelY * cursorInf,
                vel.z, 0.0,
            )
        }).$name(`${namePrefix}Splat`)
    }

    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const idx = cy * d.u32(n) + cx
        const cellX = d.f32(cx) + 0.5
        const cellY = d.f32(cy) + 0.5
        const dx = cellX - p.emitX
        const dy = cellY - p.emitY
        const distSq = dx * dx + dy * dy
        const radSq = std.max(p.emitRad * p.emitRad, d.f32(1.0))
        const influence = gaussianBrushSq(distSq, radSq)
        const cell = layout.$.dyeA[idx]
        const oldDens = cell.x
        const addDens = p.emitIntensity * influence * p.dt * densityGain
        const newDens = std.min(oldDens + addDens, 1.0)
        const rejuvenate = addDens / std.max(newDens, d.f32(0.001))
        layout.$.dyeA[idx] = d.vec4f(newDens, cell.y * (1.0 - std.min(rejuvenate, 1.0)), 0.0, 0.0)
        const vel = layout.$.velA[idx]
        const perpOff = (dx * p.perpDirX + dy * p.perpDirY) / std.max(p.emitRad, d.f32(1.0))
        const vmSq = std.max(p.emitVelX * p.emitVelX + p.emitVelY * p.emitVelY, d.f32(1.0))
        const tvx = p.emitVelX + p.perpDirX * perpOff * p.spreadFactor * std.sqrt(vmSq)
        const tvy = p.emitVelY + p.perpDirY * perpOff * p.spreadFactor * std.sqrt(vmSq)
        const br = std.min(influence * p.dt * velocityBlendGain, 1.0)
        const gravForce = p.gravity * oldDens * p.dt * gravityGain
        const cursorDX = cellX - p.cursorX
        const cursorDY = cellY - p.cursorY
        const cursorDistSq = cursorDX * cursorDX + cursorDY * cursorDY
        const cursorInf = gaussianBrushSq(cursorDistSq, std.max(p.mouseRadSq, d.f32(1.0))) * p.mouseActive
        layout.$.velA[idx] = d.vec4f(
            std.mix(vel.x, tvx, br) + p.cursorVelX * cursorInf,
            std.mix(vel.y, tvy, br) + gravForce + p.cursorVelY * cursorInf,
            vel.z, 0.0,
        )
    }).$name(`${namePrefix}Splat`)
}

// ── Neighbour indexing ─────────────────────────────────────────────────────────────────────────
// Memoized per (n, boundary): every kernel in one consumer must share ONE instance or the resolved
// WGSL grows `nidx`, `nidx_1`, `nidx_2`… duplicates.
//
// The names are deliberately the ones the original hand-written copies used (`nidx` clamped, `nw`
// toroidal) so the migration's emitted WGSL stays byte-identical.

const neighbourCache = new Map<string, ReturnType<typeof buildNeighbourIndex>>()

function buildNeighbourIndex(n: number, boundary: FluidBoundary) {
    if (boundary === 'toroidal') {
        return tgpu.fn([d.i32, d.i32, d.i32, d.i32], d.u32)((ii, ji, di, dj) => {
            'use gpu'
            const row = (ii + di + n) % n
            const col = (ji + dj + n) % n
            return d.u32(row * n + col)
        }).$name('nw')
    }
    return tgpu.fn([d.i32, d.i32, d.i32, d.i32], d.u32)((ii, ji, di, dj) => {
        'use gpu'
        const row = std.clamp(ii + di, 0, n - 1)
        const col = std.clamp(ji + dj, 0, n - 1)
        return d.u32(row * n + col)
    }).$name('nidx')
}

/**
 * The neighbour flat index for a (row, col) plus a (di, dj) offset, under the given boundary mode.
 * Exported for consumers whose own kernels need the same lookup (a mask or obstacle pass).
 */
export function neighbourIndex(n: number, boundary: FluidBoundary) {
    const key = `${n}:${boundary}`
    const hit = neighbourCache.get(key)
    if (hit) return hit
    const built = buildNeighbourIndex(n, boundary)
    neighbourCache.set(key, built)
    return built
}

// ── Kernels ────────────────────────────────────────────────────────────────────────────────────

/** The kernel set for one configuration. Absent members are the ones the options exclude. */
export interface StableFluidsKernels {
    curl: (cx: number, cy: number) => void
    vorticity: (cx: number, cy: number) => void
    divergence: (cx: number, cy: number) => void
    jacobi: (cx: number, cy: number) => void
    gradSubtract: (cx: number, cy: number) => void
    advectVel: (cx: number, cy: number) => void
    copyVel: (cx: number, cy: number) => void
    advectDye?: (cx: number, cy: number) => void
    copyDye?: (cx: number, cy: number) => void
}

/**
 * Bake the solver kernels against a consumer's layout.
 *
 * Call this at MODULE scope, next to the layout declaration — the kernels are `tgpu.fn`s and want to
 * exist once per module, not once per component instance.
 */
export function buildStableFluidsKernels(
    layoutIn: FluidSolverLayout,
    opts: StableFluidsOptions,
): StableFluidsKernels {
    // One cast: see the note on FluidSolverLayout for why the option-dependent entries cannot be
    // expressed in the parameter type.
    const layout = layoutIn as unknown as FullFluidLayout
    const {n, namePrefix, boundary, dye} = opts
    const nb = neighbourIndex(n, boundary)
    const pressureDecay = opts.pressureDecay ?? 0.8
    const toroidal = boundary === 'toroidal'
    const dyeSuffix = dye === 'rgb' ? 'Dye' : 'Dens'

    // Only the combinations that have a kernel branch are accepted. Both of these used to fall
    // through to a branch that quietly did something else, which is a wrong picture rather than an
    // error, so they are rejected at options-resolution time.
    if (dye === 'densityAge' && (opts.dyeDissipation ?? false) !== (opts.ageAdvance ?? false)) {
        throw new Error(
            `buildStableFluidsKernels: dye 'densityAge' supports dyeDissipation and ageAdvance only ` +
            `together or not at all (got dyeDissipation=${opts.dyeDissipation ?? false}, ` +
            `ageAdvance=${opts.ageAdvance ?? false})`,
        )
    }
    if (opts.solidMask && toroidal) {
        throw new Error(
            `buildStableFluidsKernels: solidMask requires boundary 'clamped' — a toroidal field with ` +
            `interior walls is not supported (the masked divergence pass assumes clamped edges)`,
        )
    }

    // ── Curl: the z-component of ∇×v, stashed in velocity.z for the vorticity pass. ──
    const curl = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const ii = d.i32(cy)
        const ji = d.i32(cx)
        const vL = layout.$.velA[nb(ii, ji, 0, -1)]
        const vR = layout.$.velA[nb(ii, ji, 0, 1)]
        const vU = layout.$.velA[nb(ii, ji, -1, 0)]
        const vD = layout.$.velA[nb(ii, ji, 1, 0)]
        const curlZ = (vR.y - vL.y - (vD.x - vU.x)) * 0.5
        const vel = layout.$.velA[idx]
        layout.$.velA[idx] = d.vec4f(vel.x, vel.y, curlZ, 0.0)
    }).$name(`${namePrefix}Curl`)

    // ── Vorticity confinement: push velocity along the gradient of |curl|, which re-injects the
    //    small-scale rotation that first-order advection numerically damps out. Without it the
    //    swirls die within a second and the flow reads as a smear. ──
    const ambientGain = opts.ambientWind?.gain ?? 0
    const vorticity = opts.ambientWind
        ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const ii = d.i32(cy)
            const ji = d.i32(cx)
            const cL = std.abs(layout.$.velA[nb(ii, ji, 0, -1)].z)
            const cR = std.abs(layout.$.velA[nb(ii, ji, 0, 1)].z)
            const cU = std.abs(layout.$.velA[nb(ii, ji, -1, 0)].z)
            const cD = std.abs(layout.$.velA[nb(ii, ji, 1, 0)].z)
            const cc = layout.$.velA[idx].z
            const gx = (cR - cL) * 0.5
            const gy = (cD - cU) * 0.5
            const gl = std.max(std.sqrt(gx * gx + gy * gy), d.f32(1e-5))
            const fx = gy / gl * cc * p.curlStrength * p.dt
            const fy = (gx / gl) * -1.0 * cc * p.curlStrength * p.dt
            // Ambient divergence-free breeze from an in-place-evolving curl-noise field.
            let ax = d.f32(0)
            let ay = d.f32(0)
            if (p.ambient > 0.0) {
                const uvx = (d.f32(cx) + 0.5) / d.f32(n)
                const uvy = (d.f32(cy) + 0.5) / d.f32(n)
                const amb = noise.curl22z(d.vec2f(uvx * p.ambientFreq, uvy * p.ambientFreq), p.ambientTime)
                const af = p.ambient * ambientGain * p.dt
                ax = amb.x * af
                ay = amb.y * af
            }
            const vel = layout.$.velA[idx]
            layout.$.velA[idx] = d.vec4f(vel.x + fx + ax, vel.y + fy + ay, vel.z, 0.0)
        }).$name(`${namePrefix}Vorticity`)
        : tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const ii = d.i32(cy)
            const ji = d.i32(cx)
            const cL = std.abs(layout.$.velA[nb(ii, ji, 0, -1)].z)
            const cR = std.abs(layout.$.velA[nb(ii, ji, 0, 1)].z)
            const cU = std.abs(layout.$.velA[nb(ii, ji, -1, 0)].z)
            const cD = std.abs(layout.$.velA[nb(ii, ji, 1, 0)].z)
            const cc = layout.$.velA[idx].z
            const gx = (cR - cL) * 0.5
            const gy = (cD - cU) * 0.5
            const gl = std.max(std.sqrt(gx * gx + gy * gy), d.f32(1e-5))
            const fx = gy / gl * cc * p.curlStrength * p.dt
            const fy = (gx / gl) * -1.0 * cc * p.curlStrength * p.dt
            const vel = layout.$.velA[idx]
            layout.$.velA[idx] = d.vec4f(vel.x + fx, vel.y + fy, vel.z, 0.0)
        }).$name(`${namePrefix}Vorticity`)

    // ── Divergence + pressure warm start. Three boundary treatments; each writes ∇·v for the Jacobi
    //    solve and decays the standing pressure field rather than clearing it. ──
    const divergence = opts.solidMask
        ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
            // Walls are "outside the grid OR outside the shape mask" — reflecting the wall-normal
            // velocity there is what makes smoke pile against a shape's inner edge.
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const ii = d.i32(cy)
            const ji = d.i32(cx)
            const jf = d.f32(cx)
            const iF = d.f32(cy)
            const selfMask = layout.$.maskBuf[idx].x
            const selfVel = layout.$.velA[idx]
            const isWallL = jf < 0.5 || layout.$.maskBuf[nb(ii, ji, 0, -1)].x < 0.5
            const isWallR = jf > d.f32(n - 1.5) || layout.$.maskBuf[nb(ii, ji, 0, 1)].x < 0.5
            const isWallT = iF < 0.5 || layout.$.maskBuf[nb(ii, ji, -1, 0)].x < 0.5
            const isWallB = iF > d.f32(n - 1.5) || layout.$.maskBuf[nb(ii, ji, 1, 0)].x < 0.5
            const L = std.select(layout.$.velA[nb(ii, ji, 0, -1)].x, selfVel.x * -1.0, isWallL)
            const R = std.select(layout.$.velA[nb(ii, ji, 0, 1)].x, selfVel.x * -1.0, isWallR)
            const T = std.select(layout.$.velA[nb(ii, ji, -1, 0)].y, selfVel.y * -1.0, isWallT)
            const B = std.select(layout.$.velA[nb(ii, ji, 1, 0)].y, selfVel.y * -1.0, isWallB)
            const div = (R - L + (B - T)) * 0.5 * selfMask
            layout.$.pressure[idx] = layout.$.pressure[idx] * pressureDecay * selfMask
            layout.$.divergence[idx] = div
        }).$name(`${namePrefix}Divergence`)
        : toroidal
            ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
                // No walls: the wrapped neighbours are the only term.
                'use gpu'
                const idx = cy * d.u32(n) + cx
                const ii = d.i32(cy)
                const ji = d.i32(cx)
                const vL = layout.$.velA[nb(ii, ji, 0, -1)].x
                const vR = layout.$.velA[nb(ii, ji, 0, 1)].x
                const vU = layout.$.velA[nb(ii, ji, -1, 0)].y
                const vD = layout.$.velA[nb(ii, ji, 1, 0)].y
                const div = (vR - vL + (vD - vU)) * 0.5
                layout.$.pressure[idx] = layout.$.pressure[idx] * pressureDecay
                layout.$.divergence[idx] = div
            }).$name(`${namePrefix}Divergence`)
            : tgpu.fn([d.u32, d.u32])((cx, cy) => {
                // Free-slip box: at a border cell the outside neighbour is replaced by the reflected
                // self velocity, so the projection cannot push flow through the wall.
                'use gpu'
                const idx = cy * d.u32(n) + cx
                const ii = d.i32(cy)
                const ji = d.i32(cx)
                const jf = d.f32(cx)
                const iF = d.f32(cy)
                const selfVel = layout.$.velA[idx]
                const normalL = layout.$.velA[nb(ii, ji, 0, -1)].x
                const normalR = layout.$.velA[nb(ii, ji, 0, 1)].x
                const normalU = layout.$.velA[nb(ii, ji, -1, 0)].y
                const normalD = layout.$.velA[nb(ii, ji, 1, 0)].y
                const L = std.select(normalL, selfVel.x * -1.0, jf < 0.5)
                const R = std.select(normalR, selfVel.x * -1.0, jf > d.f32(n - 1.5))
                const T = std.select(normalU, selfVel.y * -1.0, iF < 0.5)
                const B = std.select(normalD, selfVel.y * -1.0, iF > d.f32(n - 1.5))
                const div = (R - L + (B - T)) * 0.5
                layout.$.pressure[idx] = layout.$.pressure[idx] * pressureDecay
                layout.$.divergence[idx] = div
            }).$name(`${namePrefix}Divergence`)

    // ── One Jacobi relaxation of ∇²p = ∇·v. Dispatched J times per frame. ──
    const jacobi = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const ii = d.i32(cy)
        const ji = d.i32(cx)
        const pL = layout.$.pressure[nb(ii, ji, 0, -1)]
        const pR = layout.$.pressure[nb(ii, ji, 0, 1)]
        const pU = layout.$.pressure[nb(ii, ji, -1, 0)]
        const pD = layout.$.pressure[nb(ii, ji, 1, 0)]
        const div = layout.$.divergence[idx]
        layout.$.pressure[idx] = (pL + pR + pU + pD - div) * 0.25
    }).$name(`${namePrefix}Jacobi`)

    // ── Project: v ← v − ∇p, which is what makes the field incompressible. ──
    const velocityCap = opts.velocityCap
    const gradSubtract = velocityCap === undefined
        ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const ii = d.i32(cy)
            const ji = d.i32(cx)
            const pL = layout.$.pressure[nb(ii, ji, 0, -1)]
            const pR = layout.$.pressure[nb(ii, ji, 0, 1)]
            const pU = layout.$.pressure[nb(ii, ji, -1, 0)]
            const pD = layout.$.pressure[nb(ii, ji, 1, 0)]
            const vel = layout.$.velA[idx]
            layout.$.velA[idx] = d.vec4f(vel.x - (pR - pL) * 0.5, vel.y - (pD - pU) * 0.5, vel.z, 0.0)
        }).$name(`${namePrefix}GradSubtract`)
        : tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const ii = d.i32(cy)
            const ji = d.i32(cx)
            const pL = layout.$.pressure[nb(ii, ji, 0, -1)]
            const pR = layout.$.pressure[nb(ii, ji, 0, 1)]
            const pU = layout.$.pressure[nb(ii, ji, -1, 0)]
            const pD = layout.$.pressure[nb(ii, ji, 1, 0)]
            const vel = layout.$.velA[idx]
            const vx = vel.x - (pR - pL) * 0.5
            const vy = vel.y - (pD - pU) * 0.5
            const maxVel = d.f32(velocityCap)
            const vLen = std.max(std.sqrt(vx * vx + vy * vy), d.f32(1e-6))
            // Scale down only when over the cap: max(vLen, maxVel) in the denominator makes the
            // factor exactly 1 below it, with no branch.
            const capScale = maxVel / std.max(vLen, maxVel)
            layout.$.velA[idx] = d.vec4f(vx * capScale, vy * capScale, vel.z, 0.0)
        }).$name(`${namePrefix}GradSubtract`)

    // ── Semi-Lagrangian advection: trace backwards along the velocity and bilinear-sample the
    //    source. Unconditionally stable (Stam's key insight), at the cost of numerical diffusion. ──
    const mask = opts.solidMask ?? false
    const advectVel = toroidal
        ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const vel = layout.$.velA[idx]
            const srcX = (d.f32(cx) - vel.x * p.dt + d.f32(n)) % d.f32(n)
            const srcY = (d.f32(cy) - vel.y * p.dt + d.f32(n)) % d.f32(n)
            const x0 = std.floor(srcX)
            const y0 = std.floor(srcY)
            const sx = srcX - x0
            const sy = srcY - y0
            const ix0 = d.i32(x0)
            const iy0 = d.i32(y0)
            const ix1 = (ix0 + 1) % n
            const iy1 = (iy0 + 1) % n
            const v00 = layout.$.velA[d.u32(iy0 * n + ix0)]
            const v10 = layout.$.velA[d.u32(iy0 * n + ix1)]
            const v01 = layout.$.velA[d.u32(iy1 * n + ix0)]
            const v11 = layout.$.velA[d.u32(iy1 * n + ix1)]
            const decay = 1.0 + p.velFade * p.dt
            layout.$.velB[idx] = d.vec4f(
                std.mix(std.mix(v00.x, v10.x, sx), std.mix(v01.x, v11.x, sx), sy) / decay,
                std.mix(std.mix(v00.y, v10.y, sx), std.mix(v01.y, v11.y, sx), sy) / decay,
                0.0, 0.0,
            )
        }).$name(`${namePrefix}AdvectVel`)
        : mask
            ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
                'use gpu'
                const p = layout.$.params
                const idx = cy * d.u32(n) + cx
                const vel = layout.$.velA[idx]
                const srcX = std.clamp(d.f32(cx) - vel.x * p.dt, 0.0, d.f32(n - 1))
                const srcY = std.clamp(d.f32(cy) - vel.y * p.dt, 0.0, d.f32(n - 1))
                const x0 = std.clamp(std.floor(srcX), 0.0, d.f32(n - 2))
                const y0 = std.clamp(std.floor(srcY), 0.0, d.f32(n - 2))
                const sx = srcX - x0
                const sy = srcY - y0
                const ix0 = d.i32(x0)
                const iy0 = d.i32(y0)
                const v00 = layout.$.velA[d.u32(iy0 * n + ix0)]
                const v10 = layout.$.velA[d.u32(iy0 * n + ix0 + 1)]
                const v01 = layout.$.velA[d.u32((iy0 + 1) * n + ix0)]
                const v11 = layout.$.velA[d.u32((iy0 + 1) * n + ix0 + 1)]
                const decay = 1.0 + p.velFade * p.dt
                const m = layout.$.maskBuf[idx].x
                layout.$.velB[idx] = d.vec4f(
                    std.mix(std.mix(v00.x, v10.x, sx), std.mix(v01.x, v11.x, sx), sy) / decay * m,
                    std.mix(std.mix(v00.y, v10.y, sx), std.mix(v01.y, v11.y, sx), sy) / decay * m,
                    0.0, 0.0,
                )
            }).$name(`${namePrefix}AdvectVel`)
            : tgpu.fn([d.u32, d.u32])((cx, cy) => {
                'use gpu'
                const p = layout.$.params
                const idx = cy * d.u32(n) + cx
                const vel = layout.$.velA[idx]
                const srcX = std.clamp(d.f32(cx) - vel.x * p.dt, 0.0, d.f32(n - 1))
                const srcY = std.clamp(d.f32(cy) - vel.y * p.dt, 0.0, d.f32(n - 1))
                const x0 = std.clamp(std.floor(srcX), 0.0, d.f32(n - 2))
                const y0 = std.clamp(std.floor(srcY), 0.0, d.f32(n - 2))
                const sx = srcX - x0
                const sy = srcY - y0
                const ix0 = d.i32(x0)
                const iy0 = d.i32(y0)
                const v00 = layout.$.velA[d.u32(iy0 * n + ix0)]
                const v10 = layout.$.velA[d.u32(iy0 * n + ix0 + 1)]
                const v01 = layout.$.velA[d.u32((iy0 + 1) * n + ix0)]
                const v11 = layout.$.velA[d.u32((iy0 + 1) * n + ix0 + 1)]
                const decay = 1.0 + p.velFade * p.dt
                layout.$.velB[idx] = d.vec4f(
                    std.mix(std.mix(v00.x, v10.x, sx), std.mix(v01.x, v11.x, sx), sy) / decay,
                    std.mix(std.mix(v00.y, v10.y, sx), std.mix(v01.y, v11.y, sx), sy) / decay,
                    0.0, 0.0,
                )
            }).$name(`${namePrefix}AdvectVel`)

    // ── Copy the advected velocity back into velA. The z (curl scratch) channel is dropped here,
    //    which is why the curl pass has to run before the vorticity pass every frame. ──
    const copyVel = opts.publishVelocityTexture
        ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
            // Publishing the field to a texture from this pass saves a dedicated output dispatch.
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const v = layout.$.velB[idx]
            layout.$.velA[idx] = d.vec4f(v.x, v.y, 0.0, 0.0)
            std.textureStore(layout.$.velOutTex, d.vec2u(cx, cy), d.vec4f(v.x, v.y, 0.0, 0.0))
        }).$name(`${namePrefix}CopyVel`)
        : tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const v = layout.$.velB[idx]
            layout.$.velA[idx] = d.vec4f(v.x, v.y, 0.0, 0.0)
        }).$name(`${namePrefix}CopyVel`)

    const kernels: StableFluidsKernels = {curl, vorticity, divergence, jacobi, gradSubtract, advectVel, copyVel}
    if (dye === 'none') return kernels

    // ── Dye advection. Same backtrace as the velocity pass, over the dye channels. ──
    const dyeDissipation = opts.dyeDissipation ?? false
    const ageAdvance = opts.ageAdvance ?? false
    if (dye === 'rgb') {
        kernels.advectDye = tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const vel = layout.$.velA[idx]
            const srcX = std.clamp(d.f32(cx) - vel.x * p.dt, 0.0, d.f32(n - 1))
            const srcY = std.clamp(d.f32(cy) - vel.y * p.dt, 0.0, d.f32(n - 1))
            const x0 = std.clamp(std.floor(srcX), 0.0, d.f32(n - 2))
            const y0 = std.clamp(std.floor(srcY), 0.0, d.f32(n - 2))
            const sx = srcX - x0
            const sy = srcY - y0
            const ix0 = d.i32(x0)
            const iy0 = d.i32(y0)
            const c00 = layout.$.dyeA[d.u32(iy0 * n + ix0)]
            const c10 = layout.$.dyeA[d.u32(iy0 * n + ix0 + 1)]
            const c01 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0)]
            const c11 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0 + 1)]
            const r = std.mix(std.mix(c00.x, c10.x, sx), std.mix(c01.x, c11.x, sx), sy)
            const g = std.mix(std.mix(c00.y, c10.y, sx), std.mix(c01.y, c11.y, sx), sy)
            const b = std.mix(std.mix(c00.z, c10.z, sx), std.mix(c01.z, c11.z, sx), sy)
            const decay = 1.0 + p.dyeFade * p.dt
            layout.$.dyeB[idx] = d.vec4f(
                std.max(r / decay, d.f32(0.0)),
                std.max(g / decay, d.f32(0.0)),
                std.max(b / decay, d.f32(0.0)),
                0.0,
            )
        }).$name(`${namePrefix}Advect${dyeSuffix}`)
        kernels.copyDye = tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const c = layout.$.dyeB[idx]
            layout.$.dyeA[idx] = d.vec4f(c.x, c.y, c.z, 0.0)
        }).$name(`${namePrefix}Copy${dyeSuffix}`)
        return kernels
    }

    // densityAge. Four combinations of (dissipation, age advance); each consumer wants exactly one.
    if (dyeDissipation && ageAdvance && mask) {
        kernels.advectDye = tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const vel = layout.$.velA[idx]
            const srcX = std.clamp(d.f32(cx) - vel.x * p.dt, 0.0, d.f32(n - 1))
            const srcY = std.clamp(d.f32(cy) - vel.y * p.dt, 0.0, d.f32(n - 1))
            const x0 = std.clamp(std.floor(srcX), 0.0, d.f32(n - 2))
            const y0 = std.clamp(std.floor(srcY), 0.0, d.f32(n - 2))
            const sx = srcX - x0
            const sy = srcY - y0
            const ix0 = d.i32(x0)
            const iy0 = d.i32(y0)
            const d00 = layout.$.dyeA[d.u32(iy0 * n + ix0)]
            const d10 = layout.$.dyeA[d.u32(iy0 * n + ix0 + 1)]
            const d01 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0)]
            const d11 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0 + 1)]
            const dens = std.mix(std.mix(d00.x, d10.x, sx), std.mix(d01.x, d11.x, sx), sy)
            const age = std.mix(std.mix(d00.y, d10.y, sx), std.mix(d01.y, d11.y, sx), sy)
            const decay = 1.0 + p.dyeFade * p.dt
            const m = layout.$.maskBuf[idx].x
            layout.$.dyeB[idx] = d.vec4f(std.max(dens / decay, d.f32(0.0)) * m, std.min(age + p.dt * 0.4 * p.colorDecay, 1.0), 0.0, 0.0)
        }).$name(`${namePrefix}Advect${dyeSuffix}`)
    } else if (dyeDissipation && ageAdvance) {
        kernels.advectDye = tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = layout.$.params
            const idx = cy * d.u32(n) + cx
            const vel = layout.$.velA[idx]
            const srcX = std.clamp(d.f32(cx) - vel.x * p.dt, 0.0, d.f32(n - 1))
            const srcY = std.clamp(d.f32(cy) - vel.y * p.dt, 0.0, d.f32(n - 1))
            const x0 = std.clamp(std.floor(srcX), 0.0, d.f32(n - 2))
            const y0 = std.clamp(std.floor(srcY), 0.0, d.f32(n - 2))
            const sx = srcX - x0
            const sy = srcY - y0
            const ix0 = d.i32(x0)
            const iy0 = d.i32(y0)
            const d00 = layout.$.dyeA[d.u32(iy0 * n + ix0)]
            const d10 = layout.$.dyeA[d.u32(iy0 * n + ix0 + 1)]
            const d01 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0)]
            const d11 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0 + 1)]
            const dens = std.mix(std.mix(d00.x, d10.x, sx), std.mix(d01.x, d11.x, sx), sy)
            const age = std.mix(std.mix(d00.y, d10.y, sx), std.mix(d01.y, d11.y, sx), sy)
            const decay = 1.0 + p.dyeFade * p.dt
            layout.$.dyeB[idx] = d.vec4f(std.max(dens / decay, d.f32(0.0)), std.min(age + p.dt * 0.4 * p.colorDecay, 1.0), 0.0, 0.0)
        }).$name(`${namePrefix}Advect${dyeSuffix}`)
    } else {
        // Neither: both channels are advected untouched (Fog's fog never fades and never ages).
        kernels.advectDye = toroidal
            ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
                'use gpu'
                const p = layout.$.params
                const idx = cy * d.u32(n) + cx
                const vel = layout.$.velA[idx]
                const srcX = (d.f32(cx) - vel.x * p.dt + d.f32(n)) % d.f32(n)
                const srcY = (d.f32(cy) - vel.y * p.dt + d.f32(n)) % d.f32(n)
                const x0 = std.floor(srcX)
                const y0 = std.floor(srcY)
                const sx = srcX - x0
                const sy = srcY - y0
                const ix0 = d.i32(x0)
                const iy0 = d.i32(y0)
                const ix1 = (ix0 + 1) % n
                const iy1 = (iy0 + 1) % n
                const d00 = layout.$.dyeA[d.u32(iy0 * n + ix0)]
                const d10 = layout.$.dyeA[d.u32(iy0 * n + ix1)]
                const d01 = layout.$.dyeA[d.u32(iy1 * n + ix0)]
                const d11 = layout.$.dyeA[d.u32(iy1 * n + ix1)]
                layout.$.dyeB[idx] = d.vec4f(
                    std.mix(std.mix(d00.x, d10.x, sx), std.mix(d01.x, d11.x, sx), sy),
                    std.mix(std.mix(d00.y, d10.y, sx), std.mix(d01.y, d11.y, sx), sy),
                    0.0, 0.0,
                )
            }).$name(`${namePrefix}Advect${dyeSuffix}`)
            : tgpu.fn([d.u32, d.u32])((cx, cy) => {
                'use gpu'
                const p = layout.$.params
                const idx = cy * d.u32(n) + cx
                const vel = layout.$.velA[idx]
                const srcX = std.clamp(d.f32(cx) - vel.x * p.dt, 0.0, d.f32(n - 1))
                const srcY = std.clamp(d.f32(cy) - vel.y * p.dt, 0.0, d.f32(n - 1))
                const x0 = std.clamp(std.floor(srcX), 0.0, d.f32(n - 2))
                const y0 = std.clamp(std.floor(srcY), 0.0, d.f32(n - 2))
                const sx = srcX - x0
                const sy = srcY - y0
                const ix0 = d.i32(x0)
                const iy0 = d.i32(y0)
                const d00 = layout.$.dyeA[d.u32(iy0 * n + ix0)]
                const d10 = layout.$.dyeA[d.u32(iy0 * n + ix0 + 1)]
                const d01 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0)]
                const d11 = layout.$.dyeA[d.u32((iy0 + 1) * n + ix0 + 1)]
                layout.$.dyeB[idx] = d.vec4f(
                    std.mix(std.mix(d00.x, d10.x, sx), std.mix(d01.x, d11.x, sx), sy),
                    std.mix(std.mix(d00.y, d10.y, sx), std.mix(d01.y, d11.y, sx), sy),
                    0.0, 0.0,
                )
            }).$name(`${namePrefix}Advect${dyeSuffix}`)
    }

    kernels.copyDye = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const dd = layout.$.dyeB[idx]
        layout.$.dyeA[idx] = d.vec4f(dd.x, dd.y, 0.0, 0.0)
    }).$name(`${namePrefix}Copy${dyeSuffix}`)

    return kernels
}

/**
 * The output pass: publish the dye buffer to a storage texture the fragment samples.
 *
 * Separate from `buildStableFluidsKernels` because it reads a DIFFERENT bind group — the dye buffer
 * bound read-only alongside a write-only storage texture — and a kernel may only reference one
 * layout.
 *
 * `densityAge` writes `(density, age, 0, 0)`; `rgb` writes `(r, g, b, 1)`. Both formats are
 * rgba16float, so the fragment gets hardware-bilinear filtering for free on the upsample.
 */
export function buildFluidOutputKernel(
    layout: FluidOutputLayout,
    opts: {n: number; namePrefix: string; dye: Exclude<FluidDyeMode, 'none'>},
) {
    const {n, namePrefix} = opts
    if (opts.dye === 'rgb') {
        return tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const idx = cy * d.u32(n) + cx
            const c = layout.$.dyeA[idx]
            std.textureStore(layout.$.outTex, d.vec2u(cx, cy), d.vec4f(c.x, c.y, c.z, 1.0))
        }).$name(`${namePrefix}Output`)
    }
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const dd = layout.$.dyeA[idx]
        std.textureStore(layout.$.outTex, d.vec2u(cx, cy), d.vec4f(dd.x, dd.y, 0.0, 0.0))
    }).$name(`${namePrefix}Output`)
}

// ── Seeded noise fields ────────────────────────────────────────────────────────────────────────
// A permanent fluid (Fog) is seeded from deterministic noise, stirred by an analytic ambient wind,
// and periodically restored toward its seed pattern (numerical diffusion would otherwise wash the
// dye channels flat). Shared basis: a scalar-lattice value-noise fBm, offset by a seed so every
// seed value is a different region of one infinite pattern.

// Integer bitcast hash — the seeded lattice coords reach the hundreds, where the classic sin-fract
// hash degrades on iOS Metal (low-precision large-argument sin range reduction).
const valueHash = tgpu.fn([d.f32, d.f32], d.f32)((px, py) => {
    'use gpu'
    return noise.hash12(d.vec2f(px, py))
})
const valueNoise2D = tgpu.fn([d.f32, d.f32], d.f32)((nx, ny) => {
    'use gpu'
    const ix = std.floor(nx)
    const iy = std.floor(ny)
    const sx = std.fract(nx)
    const sy = std.fract(ny)
    const ux = sx * sx * (3.0 - sx * 2.0)
    const uy = sy * sy * (3.0 - sy * 2.0)
    return std.mix(
        std.mix(valueHash(ix, iy), valueHash(ix + 1.0, iy), ux),
        std.mix(valueHash(ix, iy + 1.0), valueHash(ix + 1.0, iy + 1.0), ux),
        uy,
    )
})
// Three-octave scalar fBm (organic cloud patterns, no stripes).
const fbm = tgpu.fn([d.f32, d.f32], d.f32)((nx, ny) => {
    'use gpu'
    return valueNoise2D(nx, ny) * 0.5
        + valueNoise2D(nx * 2.0 + 5.37, ny * 2.0 + 9.13) * 0.3
        + valueNoise2D(nx * 4.0 + 1.79, ny * 4.0 + 3.51) * 0.2
})
// The seed → lattice mapping every seeded-field kernel shares: grid coords scaled to `frequency`
// lattice cells across the field, offset by a seed-hashed jump into the infinite pattern.
const seedLattice = tgpu.fn([d.f32, d.f32, d.f32, d.f32], d.vec2f)((x, y, seed, cellsPerTexel) => {
    'use gpu'
    const sdx = noise.hash11(seed * 37.31) * 97.0
    const sdy = noise.hash11(seed * 83.17) * 97.0
    return d.vec2f(x * cellsPerTexel + sdx, y * cellsPerTexel + sdy)
})

/** The seeded-field params ABI (the init / ambient-wind / restore kernels read these). */
export interface FluidNoiseFieldParams {
    readonly dt: number
    /** The ambient wind's clock (seconds — advanced by warm-up thunks during a silent warm-up). */
    readonly time: number
    readonly seed: number
    /** Ambient wind strength. */
    readonly turbulence: number
    /** Restore softness: 0 = distinct dye channels (oil & water), 1 = fully mixed. */
    readonly blending: number
}

interface FullNoiseFieldLayout {
    readonly $: {
        readonly velA: d.v4f[]
        readonly velB: d.v4f[]
        readonly dyeA: d.v4f[]
        readonly dyeB: d.v4f[]
        readonly pressure: number[]
        readonly divergence: number[]
        readonly params: FluidNoiseFieldParams & FluidCursorForceParams
    }
}

export interface NoiseFieldOptions {
    n: number
    namePrefix: string
    /** How many noise lattice cells span the field. Default 3. */
    frequency?: number
}

/**
 * Seed the whole state from the fBm pattern: dye `x` = density (biased into a mid range so the
 * field starts neither empty nor saturated), dye `y` = the raw pattern (the color-variation
 * channel the restore kernel later pulls back toward); velocity/pressure/divergence zeroed.
 * Reads `params.seed` — pair with `seededFieldInit` for the run-once warm-up.
 */
export function buildNoiseFieldInitKernel(
    layoutIn: FluidSolverLayout,
    opts: NoiseFieldOptions & {density?: {scale: number; bias: number; min: number; max: number}},
) {
    const layout = layoutIn as unknown as FullNoiseFieldLayout
    const {n, namePrefix} = opts
    const cellsPerTexel = (opts.frequency ?? 3) / n
    const dens = opts.density ?? {scale: 0.75, bias: 0.25, min: 0.15, max: 0.9}
    const dScale = dens.scale
    const dBias = dens.bias
    const dMin = dens.min
    const dMax = dens.max
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const idx = cy * d.u32(n) + cx
        const lat = seedLattice(d.f32(cx), d.f32(cy), p.seed, cellsPerTexel)
        const dens0 = std.clamp(fbm(lat.x + 31.7, lat.y + 53.3) * dScale + dBias, dMin, dMax)
        const colorVar = std.clamp(fbm(lat.x, lat.y), 0.0, 1.0)
        layout.$.dyeA[idx] = d.vec4f(dens0, colorVar, 0.0, 0.0)
        layout.$.dyeB[idx] = d.vec4f(dens0, colorVar, 0.0, 0.0)
        layout.$.velA[idx] = d.vec4f(0.0, 0.0, 0.0, 0.0)
        layout.$.velB[idx] = d.vec4f(0.0, 0.0, 0.0, 0.0)
        layout.$.pressure[idx] = 0.0
        layout.$.divergence[idx] = 0.0
    }).$name(`${namePrefix}Init`)
}

const TAU = constants.TAU

/**
 * An analytic ambient wind: a sum of slowly-drifting incommensurate sine/cosine eddies (a cheap
 * always-alive stirring force that never repeats visibly), scaled by `params.turbulence`, plus the
 * Gaussian cursor shove. Dispatched before the solve chain (`ambientForce`).
 */
export function buildTrigTurbulenceKernel(
    layoutIn: FluidSolverLayout,
    opts: {n: number; namePrefix: string; gain?: number},
) {
    const layout = layoutIn as unknown as FullNoiseFieldLayout
    const {n, namePrefix} = opts
    const gain = n * (opts.gain ?? 0.04)
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const idx = cy * d.u32(n) + cx
        const jf = d.f32(cx) + 0.5
        const iF = d.f32(cy) + 0.5
        const x = jf / d.f32(n) * TAU
        const y = iF / d.f32(n) * TAU
        const t = p.time
        const vel = layout.$.velA[idx]
        const fx = (std.sin(x * 2.0 + t * 0.31) * std.cos(y * 1.5 + t * 0.23)
            + std.sin(x * 0.7 + y * 1.3 + t * 0.17) * 0.4) * p.turbulence * gain * p.dt
        const fy = (std.cos(x * 1.5 + t * 0.27) * std.sin(y * 2.0 + t * 0.19)
            + std.cos(x * 1.1 + y * 0.9 + t * 0.21) * 0.4) * p.turbulence * gain * p.dt
        const cdx = jf - p.cursorX
        const cdy = iF - p.cursorY
        const cursorDistSq = cdx * cdx + cdy * cdy
        const cursorInf = gaussianBrushSq(cursorDistSq, std.max(p.mouseRadSq, d.f32(1.0))) * p.mouseActive
        layout.$.velA[idx] = d.vec4f(vel.x + fx + p.cursorVelX * cursorInf, vel.y + fy + p.cursorVelY * cursorInf, vel.z, 0.0)
    }).$name(`${namePrefix}Force`)
}

/**
 * Counteract numerical diffusion: gently blend the dye's variation channel (`y`) back toward the
 * seed pattern. `params.blending` sets the character — low boosts the reference's contrast and the
 * restore rate (distinct channels, sharp boundaries), high leaves the advected mix alone.
 * Dispatched after the solve chain (`restoreToward`).
 */
export function buildNoiseRestoreKernel(
    layoutIn: FluidSolverLayout,
    opts: NoiseFieldOptions,
) {
    const layout = layoutIn as unknown as FullNoiseFieldLayout
    const {n, namePrefix} = opts
    const cellsPerTexel = (opts.frequency ?? 3) / n
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const idx = cy * d.u32(n) + cx
        const lat = seedLattice(d.f32(cx), d.f32(cy), p.seed, cellsPerTexel)
        const refRaw = fbm(lat.x, lat.y)
        const invBlend = 1.0 - p.blending
        const contrastBoost = 1.0 + invBlend * 5.0
        const refBoosted = std.clamp((refRaw - 0.5) * contrastBoost + 0.5, 0.0, 1.0)
        const cell = layout.$.dyeA[idx]
        const rateMult = 1.0 + invBlend * 2.5
        const blendRate = std.min(p.dt * 0.5 * rateMult, 0.3)
        layout.$.dyeA[idx] = d.vec4f(cell.x, std.mix(cell.y, refBoosted, blendRate), 0.0, 0.0)
    }).$name(`${namePrefix}ColorRestore`)
}

// ── Shape-container masks ──────────────────────────────────────────────────────────────────────
// A `solidMask` fluid needs `maskBuf` filled every frame: per grid cell, is it inside the confining
// shape and how far from the boundary. The shape lives in the standard shape-effect space (a
// centred, scaled, rotated, aspect-corrected [0,1] SDF square); the signed distance comes from one
// of the three standard sources — an analytic SDF fn, an uploaded SVG SDF data texture, or the
// pre-marched volumetric (3D) field texture.


// Grid cell → shape-space SDF UV (the shape-effect transform, evaluated per cell).
const gridShapeUV = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.vec2f,
)((u, v, centerX, centerY, scale, rotation, aspect) => {
    'use gpu'
    const dxAc = (u - centerX) * aspect
    const dy = v - centerY
    const cosR = std.cos(rotation)
    const sinR = std.sin(rotation)
    const rdx = dxAc * cosR + dy * sinR
    const rdy = dy * cosR - dxAc * sinR
    return d.vec2f(rdx / scale + 0.5, rdy / scale + 0.5)
})

/** The shape-mask params ABI: the shape-space transform + the analytic SDF sub-props + the
 *  volumetric field's aspect-fit domain. A consumer's params struct carries all of them (only the
 *  members the active source reads are live). */
export interface FluidShapeMaskParams {
    readonly centerX: number
    readonly centerY: number
    readonly scale: number
    readonly rotation: number
    readonly aspect: number
    readonly saRadius: number
    readonly saSides: number
    readonly saRounding: number
    readonly saInnerRatio: number
    readonly saRotation: number
    readonly saHeight: number
    readonly saOffset: number
    readonly saAperture: number
    readonly vfOriginX: number
    readonly vfOriginY: number
    readonly vfSpanX: number
    readonly vfSpanY: number
    readonly vfActiveRes: number
}

// Typing schema for the mask layouts: the runtime schema is the consumer's own params struct (a
// superset), cast to this view so the kernel bodies see the ABI members.
const shapeMaskAbiSchema = d.struct({
    centerX: d.f32, centerY: d.f32, scale: d.f32, rotation: d.f32, aspect: d.f32,
    saRadius: d.f32, saSides: d.f32, saRounding: d.f32, saInnerRatio: d.f32,
    saRotation: d.f32, saHeight: d.f32, saOffset: d.f32, saAperture: d.f32,
    vfOriginX: d.f32, vfOriginY: d.f32, vfSpanX: d.f32, vfSpanY: d.f32, vfActiveRes: d.f32,
})

/**
 * The mask-kernel set for one shape-confined fluid: three layouts (analytic / SVG texture /
 * volumetric field texture — a kernel may reference exactly one) and their kernels, all writing
 * `(inside, signedDist, 0, 0)` per cell. `paramsSchema` is the consumer's OWN params struct (its
 * members must include the {@link FluidShapeMaskParams} ABI); the same params buffer binds to
 * whichever mask layout the shape routing picks. Build at module scope.
 */
export function makeShapeMaskSet(paramsSchema: unknown, opts: {n: number; namePrefix: string}) {
    const {n, namePrefix} = opts
    const schema = paramsSchema as typeof shapeMaskAbiSchema
    const maskEntry = {storage: d.arrayOf(d.vec4f, n * n), access: 'mutable'} as const

    const analyticLayout = tgpu.bindGroupLayout({
        maskBuf: maskEntry,
        params: {uniform: schema},
    })
    // SVG variant: + the uploaded SVG SDF DATA texture (r16float, `.r` = signed distance).
    const svgLayout = tgpu.bindGroupLayout({
        maskBuf: maskEntry,
        params: {uniform: schema},
        sdfSource: {texture: d.texture2d(d.f32)},
    })
    // Volumetric variant: + the pre-marched rgba32float field texture ('unfilterable-float' —
    // reads go through textureLoad over its aspect-fit domain).
    const fieldLayout = tgpu.bindGroupLayout({
        maskBuf: maskEntry,
        params: {uniform: schema},
        fieldTex: {texture: d.texture2d(d.f32), sampleType: 'unfilterable-float'},
    })

    /** Analytic source: the SDF fn for `shapeType` is baked and evaluated per cell. */
    const analytic = (shapeType: string) => {
        const sdfFn = sdf.buildAnalyticSdfFn(shapeType)
        return tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const p = analyticLayout.$.params
            const idx = cy * d.u32(n) + cx
            const u = (d.f32(cx) + 0.5) / d.f32(n)
            const v = (d.f32(cy) + 0.5) / d.f32(n)
            const sdfUV = gridShapeUV(u, v, p.centerX, p.centerY, p.scale, p.rotation, p.aspect)
            const res = sdfFn(sdfUV, p.saRadius, p.saSides, p.saRounding, p.saInnerRatio, p.saRotation, p.saHeight, p.saOffset, p.saAperture)
            const inside = std.select(d.f32(0.0), d.f32(1.0), res.x < 0.0)
            analyticLayout.$.maskBuf[idx] = d.vec4f(inside, res.x, 0.0, 0.0)
        }).$name(`${namePrefix}Mask`)
    }

    // SVG source: the signed distance is SAMPLED (nearest) from the SDF texture (`.r`). The field
    // is 0.5 (outside) beyond the [0,1] SDF footprint, so texel coords clamp to the edge.
    const svgKernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = svgLayout.$.params
        const idx = cy * d.u32(n) + cx
        const u = (d.f32(cx) + 0.5) / d.f32(n)
        const v = (d.f32(cy) + 0.5) / d.f32(n)
        const sdfUV = gridShapeUV(u, v, p.centerX, p.centerY, p.scale, p.rotation, p.aspect)
        // Generated fields are finer than the legacy 512² `.bin`: read the source's size.
        const sdfSize = d.i32(std.textureDimensions(svgLayout.$.sdfSource).x)
        const sizeM1 = sdfSize - 1
        const tx = std.clamp(d.i32(sdfUV.x * d.f32(sdfSize)), 0, sizeM1)
        const ty = std.clamp(d.i32(sdfUV.y * d.f32(sdfSize)), 0, sizeM1)
        const dist = std.textureLoad(svgLayout.$.sdfSource, d.vec2u(d.u32(tx), d.u32(ty)), 0).x
        const inside = std.select(d.f32(0.0), d.f32(1.0), dist < 0.0)
        svgLayout.$.maskBuf[idx] = d.vec4f(inside, dist, 0.0, 0.0)
    }).$name(`${namePrefix}SvgMask`)

    // Volumetric source: the signed field (.r — distance outside / −chord/2 inside) is loaded
    // (nearest) from the marched field texture over its aspect-fit domain (the buildFieldSampleGraph
    // mapping: clamp to the footprint, map to the active top-left block, extend outside by the
    // euclidean distance to the clamped point).
    const fieldKernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = fieldLayout.$.params
        const idx = cy * d.u32(n) + cx
        const u = (d.f32(cx) + 0.5) / d.f32(n)
        const v = (d.f32(cy) + 0.5) / d.f32(n)
        const sdfUV = gridShapeUV(u, v, p.centerX, p.centerY, p.scale, p.rotation, p.aspect)
        const uxc = std.clamp(sdfUV.x, p.vfOriginX, p.vfOriginX + p.vfSpanX)
        const uyc = std.clamp(sdfUV.y, p.vfOriginY, p.vfOriginY + p.vfSpanY)
        const outsideDist = std.length(d.vec2f(sdfUV.x - uxc, sdfUV.y - uyc))
        const maxT = p.vfActiveRes - 1.0
        const tx = std.clamp((uxc - p.vfOriginX) / p.vfSpanX * p.vfActiveRes - 0.5, d.f32(0), maxT)
        const ty = std.clamp((uyc - p.vfOriginY) / p.vfSpanY * p.vfActiveRes - 0.5, d.f32(0), maxT)
        const dist = std.textureLoad(fieldLayout.$.fieldTex, d.vec2u(d.u32(tx), d.u32(ty)), 0).x + outsideDist
        const inside = std.select(d.f32(0.0), d.f32(1.0), dist < 0.0)
        fieldLayout.$.maskBuf[idx] = d.vec4f(inside, dist, 0.0, 0.0)
    }).$name(`${namePrefix}VfMask`)

    return {analyticLayout, svgLayout, fieldLayout, analytic, svgKernel, fieldKernel}
}

// ── Pass wiring ────────────────────────────────────────────────────────────────────────────────

/** The guarded compute pipelines for one kernel set, plus the ordered chain that drives them. */
export interface StableFluidsPasses {
    curl: KitComputePipeline
    vorticity: KitComputePipeline
    divergence: KitComputePipeline
    jacobi: KitComputePipeline
    gradSubtract: KitComputePipeline
    advectVel: KitComputePipeline
    copyVel: KitComputePipeline
    advectDye?: KitComputePipeline
    copyDye?: KitComputePipeline
    /**
     * The core per-frame chain: curl → vorticity → divergence → J× jacobi → gradSubtract →
     * advectVel → copyVel → advectDye → copyDye.
     *
     * Emission splats go BEFORE it, the output pass and any per-shader restoration go after. Pass
     * `vorticity` to substitute a shader's own variant (ParticleFlow folds an ambient breeze into
     * that pass) without giving up the rest of the chain.
     */
    solveSteps(opts: {jacobiIters: number; vorticity?: ComputeStep}): ComputeStep[]
}

/**
 * One kernel as a guarded 2D compute pass over the N×N grid — the same wrapper every consumer's
 * local `mk` closure was. Exported for the std fluid vocabulary (`std/sim/fluids`), whose module
 * is not a `'use gpu'` transpilation site.
 */
export function createFluidKernelPass(
    root: TgpuRoot,
    kernel: (cx: number, cy: number) => void,
    opts: {n: number; bindGroup: unknown},
): KitComputePipeline {
    return createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; kernel(cx, cy)}, {size: [opts.n, opts.n], bindGroup: opts.bindGroup as never})
}

/**
 * Wrap a kernel set in guarded compute pipelines bound to the caller's bind group.
 *
 * All passes share one bind group and one 2D dispatch size, which is what makes the whole solve a
 * flat ordered list of dispatches with no per-pass rebinding.
 */
export function createStableFluidsPasses(
    root: TgpuRoot,
    kernels: StableFluidsKernels,
    opts: {n: number; bindGroup: unknown},
): StableFluidsPasses {
    const size: [number, number] = [opts.n, opts.n]
    const mk = (kernel: (cx: number, cy: number) => void) =>
        createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; kernel(cx, cy)}, {size, bindGroup: opts.bindGroup as never})

    const passes = {
        curl: mk(kernels.curl),
        vorticity: mk(kernels.vorticity),
        divergence: mk(kernels.divergence),
        jacobi: mk(kernels.jacobi),
        gradSubtract: mk(kernels.gradSubtract),
        advectVel: mk(kernels.advectVel),
        copyVel: mk(kernels.copyVel),
        advectDye: kernels.advectDye ? mk(kernels.advectDye) : undefined,
        copyDye: kernels.copyDye ? mk(kernels.copyDye) : undefined,
    }

    return {
        ...passes,
        solveSteps({jacobiIters, vorticity}) {
            const steps: ComputeStep[] = [passes.curl, vorticity ?? passes.vorticity, passes.divergence]
            for (let i = 0; i < jacobiIters; i++) steps.push(passes.jacobi)
            steps.push(passes.gradSubtract, passes.advectVel, passes.copyVel)
            if (passes.advectDye) steps.push(passes.advectDye)
            if (passes.copyDye) steps.push(passes.copyDye)
            return steps
        },
    }
}
