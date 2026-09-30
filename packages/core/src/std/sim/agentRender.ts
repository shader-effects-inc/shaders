/**
 * std/sim/agentRender — how agents become a picture.
 *
 * Every agent frame ends the same way: each agent stamps its shape into a shared canvas (the
 * **splat**), then one pass turns the canvas into the output picture and clears it for the
 * next frame (the **resolve**). `renderAgents` pairs a splat with its resolve for you. Reach
 * into this module directly when you need an unusual pairing, or for the `pointSlot` parts
 * that give round motes their placement, brightness and size.
 *
 * Four splats match the four kinds of agent: `orientedWorld` (heading-space shapes in canvas
 * units), `pointWorld` (round motes with live softness), `volume` (a 3D swarm in perspective),
 * `relief` (particles standing on an image, seen through a camera). Three resolves: `ramp`
 * (rest-to-excited color, optional trails), `tint` (one color's alpha), `weightedColor` (an
 * average of the colors that landed, nearest first).
 */
// Maintainer notes. Per-agent additive splat of an oriented/point shape profile into fixed-point
// atomic accumulators, then one full-target resolve that reads-and-zeroes them and writes the
// rgba16f output texture the fragment bilinear-upsamples. Splat windows, gains and the alpha
// curve stay on the harness (`scaffolds/agentSystem`); the shape menu and profile parts on
// `kit/agents`. Every resolve CLEARS the accumulators as it reads them (one thread owns each
// cell, nothing else is in flight — read-then-zero needs no exchange, and WebGPU zero-inits
// storage buffers so frame 0 is correct with no clear pass).
import {tgpu, d, std, agents, colorMixing, noise} from '../../gpu/kit/index'
import {
    worldSplatWindow, texelSplatWindow, energyCoverageAlpha, FIXED_POINT_GAINS,
} from '../../gpu/scaffolds/agentSystem'
import type {Vec4StateArray, AgentHome2} from './agentForces'

type AtomicU32 = d.atomicU32
type StorageTex = d.textureStorage2d<'rgba16float', 'write-only'>

// ── Layout views ──────────────────────────────────────────────────────────────────────────
// Public factory signatures take the CORE shape every consumer of that variant has; variant-
// gated entries (`agit`, `trailBuf`, `speedNorm`, `exposure`) live on private full views the
// factory casts to once internally — a layout missing an entry its variant needs fails at
// resolve time, which every consumer's resolve gate exercises (the fluids-view precedent).

/** What an oriented splat's layout must name: `agents`, the `accumE` and `accumS` canvases, and `params.aspect` and `params.bodyR`. */
export interface WorldSplatLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly accumE: AtomicU32[]
        readonly accumS: AtomicU32[]
        readonly params: {
            readonly aspect: number
            readonly bodyR: number
        }
    }
}

interface FullWorldSplatView {
    readonly $: WorldSplatLayout['$'] & {
        readonly agit: number[]
        readonly params: WorldSplatLayout['$']['params'] & {readonly speedNorm: number}
    }
}

/** What a ramp resolve's layout must name: the two canvases, the output texture `outTex`, and the two colors `params.colA` and `params.colB`. */
export interface RampResolveLayout {
    readonly $: {
        readonly accumE: AtomicU32[]
        readonly accumS: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {
            readonly colA: d.v4f
            readonly colB: d.v4f
        }
    }
}

interface FullRampResolveView {
    readonly $: RampResolveLayout['$'] & {
        readonly trailBuf: Vec4StateArray
        readonly params: RampResolveLayout['$']['params'] & {
            readonly trails: number
            readonly exposure: number
        }
    }
}

/** One agent's rendered pose: position, unit heading, speed (0 for angle-headed agents). */
const AgentPose = d.struct({pos: d.vec2f, dir: d.vec2f, spd: d.f32}).$name('AgentPose')

// ── Oriented world splat (Boids / MagneticFilings / ParticleFlow) ─────────────────────────

/** How agents with a heading are drawn. */
export interface OrientedWorldSplatConfig {
    /** The shape name: `arrow`, `streak`, `square`, `dot`, `glow`. */
    shape: string
    /** The canvas side in texels. */
    res: number
    /** Largest body half-width accepted, in texels. A safety cap, not a size. */
    // The window itself is dynamic.
    splatRCap: number
    /** Where the heading comes from: the agent's velocity, or a stored angle plus `home`. */
    heading: 'velocity' | 'angle'
    /** Where excitement comes from: a per-agent buffer, or speed over `params.speedNorm`. */
    agitation: 'buffer' | 'speed'
    /** Stretch a soft shape into a comet along the heading. */
    comet: boolean
    /** For `heading: 'angle'`: the resting position each stored offset is measured from. */
    // MagneticFilings.
    home?: AgentHome2
    name: string
}

/**
 * The splat for agents with a heading: the shape is drawn along the way the agent points, in
 * canvas units so it looks the same on any aspect. Hard shapes get a one-texel soft edge;
 * `glow` is a soft falloff, comet-stretched when `comet` is on.
 */
// Rasterize one oriented agent into the accumulators by evaluating the baked shape's coverage
// per texel in heading space (t along the heading, n perpendicular), in WORLD units. Hard
// shapes over-drive the fixed-point gain (interiors resolve fully opaque). The window is sized
// per agent from its ACTUAL accept radius per axis and clipped once (the shared harness
// helper), and additive fixed-point atomics are order-independent — no sort. Symmetric shapes
// (a plain dot, an un-stretched glow) skip the heading rotation entirely — it cannot change a
// radial profile.
function splatOrientedWorld(layout: WorldSplatLayout, cfg: OrientedWorldSplatConfig): (i: number) => void {
    const L = layout as FullWorldSplatView
    const profile = agents.orientedShapeProfile(cfg.shape, {comet: cfg.comet})
    const RES = cfg.res
    const INV_RES = 1.0 / RES
    const AA_W = 0.75 * INV_RES // hard-edge anti-alias half-width (¾ of a render texel, world units)
    const SPLAT_R = cfg.splatRCap
    const EXT = profile.ext
    const GAIN = profile.soft ? FIXED_POINT_GAINS.SOFT : FIXED_POINT_GAINS.HARD

    // Pose: where the agent is and which way it points, per the declared state family.
    const home = cfg.home
    const pose = cfg.heading === 'angle'
        ? tgpu.fn([d.u32], AgentPose)((i) => {
            'use gpu'
            const a = L.$.agents[i]
            const homeP = home!(d.f32(i))
            // Heading is the agent's own angle (compass-needle orientation), NOT a velocity.
            return AgentPose({
                pos: d.vec2f(homeP.x + a.x, homeP.y + a.y),
                dir: d.vec2f(std.cos(a.z), std.sin(a.z)),
                spd: d.f32(0),
            })
        }).$name(`${cfg.name}Pose`)
        : tgpu.fn([d.u32], AgentPose)((i) => {
            'use gpu'
            const a = L.$.agents[i]
            const vel = d.vec2f(a.z, a.w)
            const spd = std.length(vel)
            return AgentPose({
                pos: d.vec2f(a.x, a.y),
                dir: std.select(d.vec2f(1.0, 0.0), vel.div(std.max(spd, 1e-5)), spd > 1e-5),
                spd,
            })
        }).$name(`${cfg.name}Pose`)

    // Excitement for the rest→excited ramp: the per-agent envelope, or normalized local speed.
    const excitement = cfg.agitation === 'buffer'
        ? tgpu.fn([d.u32, d.f32], d.f32)((i, _spd) => {
            'use gpu'
            return std.clamp(L.$.agit[i], 0.0, 1.0)
        }).$name(`${cfg.name}Excitement`)
        : tgpu.fn([d.u32, d.f32], d.f32)((_i, spd) => {
            'use gpu'
            return std.clamp(spd / std.max(L.$.params.speedNorm, 1e-5), 0.0, 1.0)
        }).$name(`${cfg.name}Excitement`)

    // Per-texel coverage: the comet-stretched heading-space weight, the plain heading-space
    // weight, or — for symmetric shapes — the radial weight with the rotation baked out (it
    // cannot change a radial profile, so it was pure cost).
    const cometWeight = agents.makeOrientedCometWeightFn(cfg.shape)
    const plainWeight = agents.makeAgentTexelWeightFn(profile.name)
    const useComet = cfg.comet && !profile.symmetric
    const texelWeight = useComet
        ? tgpu.fn([d.f32, d.f32, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.f32)((_dSq, offX, offY, dir, bodyR, _bodyRSq, aaW) => {
            'use gpu'
            const t = offX * dir.x + offY * dir.y
            const n = offX * -dir.y + offY * dir.x
            return cometWeight(t, n, bodyR, aaW)
        }).$name(`${cfg.name}Weight`)
        : profile.symmetric
            ? tgpu.fn([d.f32, d.f32, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.f32)((dSq, offX, offY, _dir, bodyR, bodyRSq, aaW) => {
                'use gpu'
                return plainWeight(dSq / bodyRSq, offX, offY, bodyR, aaW)
            }).$name(`${cfg.name}Weight`)
            : tgpu.fn([d.f32, d.f32, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.f32)((dSq, offX, offY, dir, bodyR, bodyRSq, aaW) => {
                'use gpu'
                const t = offX * dir.x + offY * dir.y
                const n = offX * -dir.y + offY * dir.x
                return plainWeight(dSq / bodyRSq, t, n, bodyR, aaW)
            }).$name(`${cfg.name}Weight`)

    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = L.$.params
        const p = pose(i)
        const agitI = excitement(i, p.spd)
        const agitGain = agitI * d.f32(GAIN)

        const aspect = std.max(prm.aspect, 1e-5)
        const bodyR = std.clamp(prm.bodyR, 1e-6, d.f32(SPLAT_R) * INV_RES)
        const bodyRSq = bodyR * bodyR
        // Accept radius in WORLD units: the shape's radial extent plus the AA margin.
        const radW = bodyR * EXT + AA_W * 2.0
        const extSq = radW * radW

        // Per-axis window sized to the ACTUAL accept radius and clipped once (the shared harness
        // helper — the aspect asymmetry and the "no bounds test in the inner loop" trick live
        // there). An off-screen agent clips to an empty range.
        const win = worldSplatWindow(p.pos, radW, aspect, d.f32(RES))
        for (let py = win.y0; py <= win.y1; py++) {
            const offY = (d.f32(py) + 0.5) * INV_RES - p.pos.y
            const offYSq = offY * offY
            for (let px = win.x0; px <= win.x1; px++) {
                const offX = (d.f32(px) + 0.5) * INV_RES * aspect - p.pos.x
                const dSq = offX * offX + offYSq
                if (dSq < extSq) {
                    const w = texelWeight(dSq, offX, offY, p.dir, bodyR, bodyRSq, AA_W)
                    const e = d.u32(w * d.f32(GAIN))
                    if (e > d.u32(0)) {
                        const idx = d.u32(py * d.i32(RES) + px)
                        std.atomicAdd(L.$.accumE[idx], e)
                        // A calm agent's agitI is exactly 0 (the common case), and a zero add is
                        // still a full atomic round-trip — so skip its traffic entirely.
                        if (agitI > 0.0) {
                            std.atomicAdd(L.$.accumS[idx], d.u32(w * agitGain))
                        }
                    }
                }
            }
        }
    }).$name(cfg.name)
}

// ── Point world splat (FloatingParticles) ─────────────────────────────────────────────────

// The generic parts that feed the point splat's slots. Each is a factory over the consumer's
// layout (REQUIRED param names documented per part); the per-agent hash phases keep the field
// from ever phase-locking, and the presence cascade (kit `agents.presenceVariation`) couples
// size and brightness through ONE hash so near-reading agents are bigger AND brighter together.
/**
 * The three parts a round mote is drawn with: where it sits (`orbitalPlace`), how bright it is
 * (`twinkleBrightness`) and how big it is (`presenceRadius`). Each is built over your layout
 * and handed to `renderAgents.pointWorld`.
 *
 * Every part varies per mote by a hash of its index, so a field of motes never moves in step.
 *
 * @example
 * ```ts
 * renderAgents.pointWorld(simLayout, {shape, res: 1024, place: pointSlot.orbitalPlace(simLayout, {radius: 0.035, rate: 1.4}), brightness: pointSlot.twinkleBrightness(simLayout, {freq: 2}), bodyRadius: pointSlot.presenceRadius(simLayout), names: {splat: 'motesSplat', resolve: 'motesResolve'}})
 * ```
 * @see renderAgents, splat
 */
export const pointSlot = {
    /**
     * The mote's drawn position: its stored position plus a slow circle of `radius` (canvas
     * heights) at `rate` turns per second, both scaled by `params.randomness` and varied per
     * mote. Reads `params.time`.
     *
     * @example
     * ```ts
     * place: pointSlot.orbitalPlace(simLayout, {radius: 0.035, rate: 1.4})
     * ```
     * @see twinkleBrightness, presenceRadius
     */
    // Phase + rate hashed per agent. Needs `params.{time, randomness}`.
    orbitalPlace(
        layout: {readonly $: {readonly params: {readonly time: number; readonly randomness: number}}},
        cfg: {radius: number; rate: number},
    ): PointWorldSplatConfig['place'] {
        const RATE = cfg.rate
        const RAD = cfg.radius
        const TAU = 6.283185307179586
        return tgpu.fn([d.f32, d.vec2f], d.vec2f)((fi, stored) => {
            'use gpu'
            const prm = layout.$.params
            const h = noise.hash22(d.vec2f(fi * 2.41 + 0.37, fi * 0.577 + 8.19))
            const rate = (0.5 + h.x) * RATE * prm.randomness
            const ph = h.y * TAU
            const rad = RAD * prm.randomness * (0.5 + h.x * 0.5)
            const t = prm.time * rate + ph
            return stored.add(d.vec2f(std.cos(t), std.sin(t)).mul(rad))
        }).$name('pointSlotOrbitalPlace') as PointWorldSplatConfig['place']
    },

    /**
     * The mote's brightness: a twinkle at `freq` cycles per second, blended in by
     * `params.twinkle` (0 steady, 1 full shimmer), times a per-mote presence so some motes read
     * nearer than others. Reads `params.time`.
     *
     * @example
     * ```ts
     * brightness: pointSlot.twinkleBrightness(simLayout, {freq: 2})
     * ```
     * @see orbitalPlace, presenceRadius
     */
    // Hash-phased sinusoid × the presence cascade. Needs `params.{time, twinkle}`.
    twinkleBrightness(
        layout: {readonly $: {readonly params: {readonly time: number; readonly twinkle: number}}},
        cfg: {freq: number},
    ): PointWorldSplatConfig['brightness'] {
        const FREQ = cfg.freq
        const TAU = 6.283185307179586
        return tgpu.fn([d.f32], d.f32)((fi) => {
            'use gpu'
            const prm = layout.$.params
            const ph = noise.hash11(fi * 3.7 + 0.93) * TAU
            const twk = std.sin(prm.time * FREQ + ph) * 0.5 + 0.5
            return std.mix(d.f32(1), twk, prm.twinkle) * (0.4 + 0.6 * agents.presenceVariation(fi))
        }).$name('pointSlotTwinkleBrightness') as PointWorldSplatConfig['brightness']
    },

    /**
     * The mote's size: `params.bodyR` scaled by the same per-mote presence as the brightness,
     * so the motes that read nearer are bigger and brighter together.
     *
     * @example
     * ```ts
     * bodyRadius: pointSlot.presenceRadius(simLayout)
     * ```
     * @see twinkleBrightness
     */
    // Floored at 1e-6.
    presenceRadius(
        layout: {readonly $: {readonly params: {readonly bodyR: number}}},
    ): PointWorldSplatConfig['bodyRadius'] {
        return tgpu.fn([d.f32], d.f32)((fi) => {
            'use gpu'
            return std.max(layout.$.params.bodyR * (0.55 + 0.45 * agents.presenceVariation(fi)), 1e-6)
        }).$name('pointSlotPresenceRadius') as PointWorldSplatConfig['bodyRadius']
    },
} as const

/** How round motes are drawn: the shape, the canvas size, and the three `pointSlot` parts. */
export interface PointWorldSplatConfig {
    /** The shape name: `dot`, `glow`, `square`. */
    shape: string
    /** The canvas side in texels. */
    res: number
    /** Where the mote is drawn, from its stored position. */
    place: (fi: number, stored: d.v2f) => d.v2f
    /** How bright the mote is. */
    brightness: (fi: number) => number
    /** The mote's body radius in canvas heights, already floored. */
    bodyRadius: (fi: number) => number
    name: string
}

/** What a point splat's layout must name: `agents`, the `accumE` canvas, and `params.aspect` and `params.softness`. */
// Rasterize one un-oriented mote with a LIVE softness uniform (`params.softness` feathers the
// baked hard shapes toward the glow skirt at runtime; the glow shape is always fully soft —
// its mix folds away and the fast texel fn emits the Gaussian skirt only). Placement,
// brightness and per-agent size arrive as slot parts — they are the consumer's look.
export interface PointWorldSplatLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly accumE: AtomicU32[]
        readonly params: {
            readonly aspect: number
            readonly softness: number
        }
    }
}

function splatPointWorld(layout: PointWorldSplatLayout, cfg: PointWorldSplatConfig): (i: number) => void {
    const s = agents.AGENT_SHAPES[agents.resolveAgentShape(cfg.shape, 'dot')]
    const name = agents.resolveAgentShape(cfg.shape, 'dot')
    const FORCE_SOFT = s.soft
    const EXT = s.ext
    const RES = cfg.res
    const INV_RES = 1.0 / RES
    const AA_W = 0.75 * INV_RES
    const HARD_GAIN = FIXED_POINT_GAINS.HARD
    const SOFT_GAIN = FIXED_POINT_GAINS.SOFT
    const {place, brightness, bodyRadius} = cfg
    // `softness` is a live uniform for the hard shapes, so those keep the general weight fn that
    // evaluates and mixes both profiles. For `glow` the registry pins soft = 1, so the mix — and
    // the general SDF call — fold away: the fast texel fn emits the Gaussian skirt only.
    const weightFn = s.soft === 1
        ? (() => {
            const fast = agents.makeAgentTexelWeightFn(name)
            return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((q, t, n, bodyR, aaW, _soft) => {
                'use gpu'
                return fast(q, t, n, bodyR, aaW)
            }).$name('agentPointWeight')
        })()
        : (() => {
            const general = agents.makeAgentWeightFn(name)
            return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((_q, t, n, bodyR, aaW, soft) => {
                'use gpu'
                return general(t, n, bodyR, aaW, soft)
            }).$name('agentPointWeight')
        })()
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const s4 = layout.$.agents[i]
        const fi = d.f32(i)
        const pos = place(fi, d.vec2f(s4.x, s4.y))
        const bright = brightness(fi)

        const aspect = std.max(prm.aspect, 1e-5)
        const softV = std.max(prm.softness, d.f32(FORCE_SOFT))
        // Runtime softness widens the footprint toward the glow skirt → widen the reject with it.
        const extK = std.mix(d.f32(EXT), std.max(d.f32(EXT), 2.4), softV)
        const gain = std.mix(d.f32(HARD_GAIN), d.f32(SOFT_GAIN), softV)
        const bodyR = bodyRadius(fi)
        const invBodyR = 1.0 / bodyR
        // World-space accept radius of THIS mote — most motes iterate a handful of texels.
        const reach = bodyR * extK + AA_W * 2.0
        const reachQ = (reach * invBodyR) * (reach * invBodyR) // reject in normalized q, not world d²

        const win = worldSplatWindow(pos, reach, aspect, d.f32(RES))
        for (let py = win.y0; py <= win.y1; py++) {
            const offY = (d.f32(py) + 0.5) * INV_RES - pos.y
            for (let px = win.x0; px <= win.x1; px++) {
                const offX = (d.f32(px) + 0.5) * INV_RES * aspect - pos.x
                const q = (offX * offX + offY * offY) * invBodyR * invBodyR
                if (q < reachQ) {
                    const w = weightFn(q, offX, offY, bodyR, AA_W, softV) * bright
                    const e = d.u32(w * gain)
                    if (e > d.u32(0)) {
                        const idx = d.u32(py * RES + px)
                        std.atomicAdd(layout.$.accumE[idx], e)
                    }
                }
            }
        }
    }).$name(cfg.name)
}

// ── Volume splat (Particles) ──────────────────────────────────────────────────────────────

interface VolumeRenderLayout {
    readonly $: {
        readonly pos: Vec4StateArray
        readonly vel: Vec4StateArray
        readonly accumE: AtomicU32[]
        readonly accumS: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {
            readonly colA: d.v4f
            readonly colB: d.v4f
            readonly centerX: number
            readonly centerYv: number
            readonly scale: number
            readonly rotC: number
            readonly rotS: number
            readonly aspect: number
            readonly size: number
            readonly exposure: number
            readonly softness: number
            readonly speedColorK: number
        }
    }
}

/** The Gaussian puff — the whole profile for the `glow` shape (its forced soft pins the blend
 *  at 1, so the hard half was dead code), and the soft end of the blend for the crisp shapes. */
function makeSoftenedProfile(shape: agents.AgentShapeName, names: {glow: string; blend: string}) {
    const glowProfile = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((q, _ddx, _ddy, _size, _soft) => {
        'use gpu'
        return std.exp(q * -2.5)
    }).$name(names.glow)
    if (agents.AGENT_SHAPES[shape].soft === 1) return glowProfile
    const weightFn = agents.makeAgentTexelWeightFn(shape)
    return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((q, ddx, ddy, size, soft) => {
        'use gpu'
        const wHard = weightFn(q, ddx, ddy, size, 0.9)
        const wSoft = std.exp(q * -2.5)
        return std.mix(wHard, wSoft, soft)
    }).$name(names.blend)
}

/** How a 3D swarm is drawn: the shape, the output size, the largest particle, and the reach of each particle's profile. */
export interface VolumeSplatConfig {
    /** The shape name: `dot`, `glow`, `square`. */
    shape: string
    /** The output side in texels. */
    outRes: number
    /** Largest particle radius drawn, in texels. */
    maxSplatSize: number
    /** How far a particle's profile reaches, as distance² over size². 6 covers a soft glow. */
    // Radial reject: a texel contributes while q = dist²/size² < extQ.
    extQ: number
    names: {splat: string; glow: string; profile: string}
}

/**
 * The splat for a 3D swarm: each particle is placed on screen through the layer's center,
 * scale and rotation with a soft perspective from its depth, then drawn with a profile that
 * `params.softness` blends from crisp to a glow. Speed is recorded alongside so the ramp
 * resolve can color fast particles.
 */
// Project a 3D swarm particle to screen (2D placement rotation + a soft perspective from its
// z), then accumulate the particle's softness-blended profile into the energy buffer and a
// speed-weighted copy into the second buffer (the resolve derives per-texel average speed for
// the rest→excited ramp). Additive fixed-point atomics are order-independent — no z-sort.
function splatVolume(layout: VolumeRenderLayout, cfg: VolumeSplatConfig): (i: number) => void {
    const name = agents.resolveAgentShape(cfg.shape, 'dot')
    const profileFn = makeSoftenedProfile(name, {glow: cfg.names.glow, blend: cfg.names.profile})
    const OUT_RES = cfg.outRes
    const MAX_SPLAT_SIZE = cfg.maxSplatSize
    const EXTQ = cfg.extQ
    const EXT = Math.sqrt(cfg.extQ) // pre-folded — never Math.* in a 'use gpu' body
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const pos = layout.$.pos[i]
        const vel = layout.$.vel[i]

        // Soft perspective: z toward the viewer enlarges and brightens.
        const persp = 1.0 / std.clamp(1.0 - pos.z * 0.35, 0.45, 2.0)
        const xs = pos.x * persp
        const ys = pos.y * persp * -1.0 // shape-local y-up → screen y-down
        const rx = xs * prm.rotC - ys * prm.rotS
        const ry = ys * prm.rotC + xs * prm.rotS
        const u = prm.centerX + rx * prm.scale / prm.aspect
        const v = prm.centerYv + ry * prm.scale

        const outF = d.f32(OUT_RES)
        const tx = u * outF
        const ty = v * outF
        const size = std.clamp(prm.size * (0.65 + 0.7 * pos.w) * persp * prm.scale, 0.6, d.f32(MAX_SPLAT_SIZE))
        const bright = (0.7 + 0.6 * pos.w) * persp * persp
        const speedNorm = std.clamp(vel.w * prm.speedColorK, 0.0, 1.0)
        // The speed accumulator's quantity is w · bright · speedNorm · 256 with w ≤ 1, so once
        // the w-free part can't reach one fixed-point step the atomic could only ever add zero.
        const doSpeed = bright * speedNorm * 256.0 >= 1.0

        // Iterate only the texels the profile can actually reach, clipped to the target.
        const rad = d.i32(std.ceil(size * d.f32(EXT)))
        const radF = d.f32(rad)
        const onTarget = tx > radF * -1.0 && tx < outF + radF && ty > radF * -1.0 && ty < outF + radF
        if (onTarget) {
            // Isotropic window (square render texels), clipped to the target ONCE by the shared
            // harness helper — so the bounds test drops out of the inner loop entirely.
            const win = texelSplatWindow(d.vec2f(tx, ty), rad, d.vec2i(d.i32(OUT_RES - 1), d.i32(OUT_RES - 1)))
            const sizeSq = std.max(size * size, 0.01)
            for (let py = win.y0; py <= win.y1; py++) {
                const ddy = d.f32(py) + 0.5 - ty
                const ddySq = ddy * ddy
                for (let px = win.x0; px <= win.x1; px++) {
                    const ddx = d.f32(px) + 0.5 - tx
                    const q = (ddx * ddx + ddySq) / sizeSq
                    if (q < EXTQ) {
                        // Softness blends the particle profile: the crisp baked shape with a
                        // thin AA rim at 0, the classic Gaussian glow puff at 1.
                        const w = profileFn(q, ddx, ddy, size, prm.softness)
                        const e = d.u32(w * bright * 256.0)
                        if (e > d.u32(0)) {
                            const idx = d.u32(py * d.i32(OUT_RES) + px)
                            std.atomicAdd(layout.$.accumE[idx], e)
                            if (doSpeed) std.atomicAdd(layout.$.accumS[idx], d.u32(w * bright * speedNorm * 256.0))
                        }
                    }
                }
            }
        }
    }).$name(cfg.names.splat)
}

// ── Relief splat (ParticleField) ──────────────────────────────────────────────────────────

interface ReliefRenderLayout {
    readonly $: {
        readonly pos: Vec4StateArray
        readonly col: Vec4StateArray
        readonly accumR: AtomicU32[]
        readonly accumG: AtomicU32[]
        readonly accumB: AtomicU32[]
        readonly accumW: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {
            readonly rowX: d.v4f
            readonly rowY: d.v4f
            readonly rowZ: d.v4f
            readonly time: number
            readonly gridW: number
            readonly gridH: number
            readonly outW: number
            readonly outH: number
            readonly depth: number
            readonly wobbleAmp: number
            readonly depthShading: number
            readonly spacing: number
            readonly particleSize: number
            readonly aspect: number
            readonly zoom: number
            readonly transX: number
            readonly transY: number
        }
    }
}

/** How particles standing on an image are drawn. */
export interface ReliefSplatConfig {
    /** The shape name: `dot`, `glow`, `square`. */
    shape: string
    /** Largest body radius drawn, in texels. */
    // The window itself is dynamic.
    splatRCap: number
    /** Width of the soft edge on hard shapes, in texels. 0.75 is typical. */
    aaW: number
    /** Child alpha below which a particle is not drawn at all. */
    minAlpha: number
    /** Precision of the color accumulation. 1024 is typical. */
    // Fixed-point scale for the atomic accumulators.
    fp: number
    /** How fast particles breathe when idle, in cycles per second of `params.time`. */
    wobbleFreq: number
    /** How strongly depth changes size and spread. */
    perspK: number
    name: string
}

/**
 * The splat for particles standing on an image: each is placed through the camera rows with
 * perspective, then drawn in the child's color at its home. Nearer particles weigh more, so
 * they win the color where particles overlap, with no sorting.
 */
// Rotate the particle's 3D world position by the camera rows, project through a perspective
// from screen center (near ones enlarge, brighten and spread outward), then accumulate the
// shape's coverage as DEPTH-WEIGHTED color into the four fixed-point buffers. The weight
// (persp²·shade) makes near particles dominate the per-texel average → stylized
// pseudo-occlusion, no z-sort.
function splatRelief(layout: ReliefRenderLayout, cfg: ReliefSplatConfig): (x: number, y: number) => void {
    const name = agents.resolveAgentShape(cfg.shape, 'dot')
    const s = agents.AGENT_SHAPES[name]
    const weightFn = agents.makeAgentTexelWeightFn(name)
    // The registry's soft `ext` is sized for a gain of ~255; here the per-texel factor is
    // FP × depthWeight (peaking ≈ 8.2), so a near particle deposits nonzero quanta out to
    // √q ≈ 2.34 — widen the soft reject to cover that.
    const EXT_SHAPE = s.soft === 1 ? Math.max(s.ext, 2.4) : s.ext
    const SPLAT_R = cfg.splatRCap
    const AA_W = cfg.aaW
    const MIN_ALPHA = cfg.minAlpha
    const FP = cfg.fp
    const MIN_WEIGHT = 1.0 / cfg.fp
    const WOBBLE_FREQ = cfg.wobbleFreq
    const PERSP_K = cfg.perspK
    const TAU = 6.283185307179586
    return tgpu.fn([d.u32, d.u32])((gx, gy) => {
        'use gpu'
        const P = layout.$.params
        const i = gy * d.u32(P.gridW) + gx
        const p4 = layout.$.pos[i]
        const col = layout.$.col[i]
        const childA = col.w

        // Early-out 1: the particle sits over transparent child content, so every texel it
        // would touch scales to zero.
        if (childA > MIN_ALPHA) {
            const homeU = (d.f32(gx) + 0.5) / P.gridW
            const homeV = (d.f32(gy) + 0.5) / P.gridH

            // Idle wobble — a hash-phased per-particle breath in xy + z, keeps the field alive.
            const ph = p4.w * TAU
            const wob = P.wobbleAmp
            const wx = std.sin(P.time * WOBBLE_FREQ + ph) * wob * 0.012
            const wy = std.cos(P.time * WOBBLE_FREQ * 1.13 + ph * 1.7) * wob * 0.012
            const wz = std.sin(P.time * WOBBLE_FREQ * 0.87 + ph * 2.3) * wob * 0.06

            // World position → camera rotation (rows built on the CPU) → perspective projection.
            const pw = d.vec3f((homeU - 0.5) * P.aspect + p4.x + wx, homeV - 0.5 + p4.y + wy, p4.z + wz)
            const rv = agents.applyCameraRows(pw, P.rowX, P.rowY, P.rowZ)

            // Zoom scales the projected offset from center and the particle size (a camera
            // dolly), but NOT the render weight — persp0 keeps brightness/occlusion zoom-invariant.
            const persp0 = 1.0 / std.clamp(1.0 - rv.z * PERSP_K, 0.35, 2.6)
            const persp = persp0 * P.zoom
            const su = 0.5 + rv.x * persp / P.aspect + P.transX
            const sv = 0.5 + rv.y * persp + P.transY

            const tx = su * P.outW
            const ty = sv * P.outH
            const dotR = std.clamp(P.particleSize * P.spacing * 0.5 * persp, 0.5, d.f32(SPLAT_R))

            // Depth shading: near → full brightness, far → dimmer. The render weight leans on
            // near particles so they win the weighted-average color.
            const dr = std.max(P.depth, 1e-3)
            const near = std.smoothstep(dr * -1.0, dr, rv.z)
            const shade = std.mix(d.f32(1), 0.35 + 0.65 * near, P.depthShading)
            const depthWeight = persp0 * persp0 * shade
            // Everything in the per-texel weight that does NOT vary across the window.
            const cw = childA * depthWeight

            // The accept radius in TEXELS is `dotR · EXT_SHAPE + AA_W` — the shape extent scales
            // with the body, the anti-alias skirt does not. Divided back into dotR units it gives
            // the reject radius `extR`, so the window and the per-texel radial test agree.
            const extR = d.f32(EXT_SHAPE) + d.f32(AA_W) / dotR
            const rad = d.i32(std.ceil(dotR * extR))
            const radF = d.f32(rad)

            // Early-out 2: the projected particle can't reach the target at all.
            const onTarget = tx > radF * -1.0 && tx < P.outW + radF && ty > radF * -1.0 && ty < P.outH + radF
            if (cw > MIN_WEIGHT && onTarget) {
                const owU = d.u32(P.outW)
                const win = texelSplatWindow(d.vec2f(tx, ty), rad, d.vec2i(d.i32(P.outW) - 1, d.i32(P.outH) - 1))
                for (let py = win.y0; py <= win.y1; py++) {
                    const ddy = d.f32(py) + 0.5 - ty
                    for (let px = win.x0; px <= win.x1; px++) {
                        const ddx = d.f32(px) + 0.5 - tx
                        const q = (ddx * ddx + ddy * ddy) / std.max(dotR * dotR, 0.01)
                        if (q < extR * extR) {
                            const wc = weightFn(q, ddx, ddy, dotR, d.f32(AA_W)) * cw
                            if (wc > 0.0) {
                                const idx = d.u32(py) * owU + d.u32(px)
                                std.atomicAdd(layout.$.accumR[idx], d.u32(col.x * wc * FP))
                                std.atomicAdd(layout.$.accumG[idx], d.u32(col.y * wc * FP))
                                std.atomicAdd(layout.$.accumB[idx], d.u32(col.z * wc * FP))
                                std.atomicAdd(layout.$.accumW[idx], d.u32(wc * FP))
                            }
                        }
                    }
                }
            }
        }
    }).$name(cfg.name)
}

// ── Resolves ──────────────────────────────────────────────────────────────────────────────

/** How the rest-to-excited color resolve behaves. */
export interface RampResolveConfig {
    /** The color space the two colors mix in, as the `colorSpace` prop's number. */
    // Baked color-space mode for the rest→excited mix (the canonical mixColorsVariants path).
    colorSpace: number
    /** The canvas side in texels. */
    res: number
    /** The scale the splat accumulated at. `renderAgents` fills this in. */
    // 1/255 for the world splats, 1/256 for the volume splat.
    invGain: number
    /** Multiply brightness by `params.exposure`. */
    exposure?: boolean
    /** Motion trails: `'on'` always, `'none'` never, or `{baked}` for a layer whose trails prop can be 0. */
    // 'on' (always composited), {baked: boolean} (read baked out at trails = 0, write kept so
    // scrubbing up never ghosts), or 'none' (no trail buffer in the layout).
    trails: 'on' | 'none' | {baked: boolean}
    name: string
}

/**
 * The resolve that colors agents from a rest color to an excited color: where more excited
 * agents landed, the color leans toward `params.colB`. Empty texels are transparent. With
 * trails on, each frame fades into a persistent canvas by `params.trails`.
 */
// Accumulators → color: the rest→excited ramp mixes in the baked color space, driven by the
// per-texel average excitement (`accumS / accumE`). Most texels get no splat energy at all and
// the endpoints are uniforms, so zero-energy texels skip the mix — RGB must still be colA (not
// black) or the fragment's bilinear upsample would drag a dark fringe into every edge texel.
function resolveRamp(layout: RampResolveLayout, cfg: RampResolveConfig): (x: number, y: number) => void {
    const L = layout as FullRampResolveView
    const mixFn = colorMixing.mixColorsVariants[cfg.colorSpace as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
    const RES = cfg.res
    const INV_GAIN = cfg.invGain
    // Fixed-point energy, optionally scaled by the exposure knob (baked — the field only
    // exists on the consumers that declare it).
    const energyOf = cfg.exposure === true
        ? tgpu.fn([d.f32], d.f32)((eRaw) => {
            'use gpu'
            return eRaw * INV_GAIN * L.$.params.exposure
        }).$name(`${cfg.name}Energy`)
        : tgpu.fn([d.f32], d.f32)((eRaw) => {
            'use gpu'
            return eRaw * INV_GAIN
        }).$name(`${cfg.name}Energy`)

    if (cfg.trails === 'none') {
        // No trail canvas: write the ramped color straight to the output texture.
        return tgpu.fn([d.u32, d.u32])((x, y) => {
            'use gpu'
            const prm = L.$.params
            const idx = y * d.u32(RES) + x
            const eRaw = d.f32(std.atomicLoad(L.$.accumE[idx]))
            std.atomicStore(L.$.accumE[idx], d.u32(0))
            const sRaw = d.f32(std.atomicLoad(L.$.accumS[idx]))
            std.atomicStore(L.$.accumS[idx], d.u32(0))
            if (eRaw > 0.0) {
                const energy = energyOf(eRaw)
                const agitAvg = sRaw / std.max(eRaw, 1.0)
                const col = mixFn(prm.colA, prm.colB, std.clamp(agitAvg, 0.0, 1.0))
                const a = energyCoverageAlpha(energy, 1.6) * col.w
                std.textureStore(L.$.outTex, d.vec2u(x, y), d.vec4f(col.x, col.y, col.z, a))
            } else {
                std.textureStore(L.$.outTex, d.vec2u(x, y), d.vec4f(prm.colA.x, prm.colA.y, prm.colA.z, d.f32(0)))
            }
        }).$name(cfg.name)
    }

    // Trail-canvas variant: composite into the persistent decayed canvas (max-with-decay), or —
    // with the read baked out at trails = 0 — write the bare splat through the same path.
    const trailsOn = cfg.trails === 'on' || cfg.trails.baked
    const composite = trailsOn
        ? tgpu.fn([d.vec4f, d.f32, d.u32], d.vec4f)((col, curA, idx) => {
            'use gpu'
            return agents.trailMaxDecay(col, curA, L.$.trailBuf[idx], L.$.params.trails)
        }).$name(`${cfg.name}TrailComposite`)
        : tgpu.fn([d.vec4f, d.f32, d.u32], d.vec4f)((col, curA, _idx) => {
            'use gpu'
            return d.vec4f(col.x, col.y, col.z, curA)
        }).$name(`${cfg.name}TrailComposite`)
    return tgpu.fn([d.u32, d.u32])((x, y) => {
        'use gpu'
        const prm = L.$.params
        const idx = y * d.u32(RES) + x
        const eRaw = d.f32(std.atomicLoad(L.$.accumE[idx]))
        std.atomicStore(L.$.accumE[idx], d.u32(0))
        const sRaw = d.f32(std.atomicLoad(L.$.accumS[idx]))
        std.atomicStore(L.$.accumS[idx], d.u32(0))
        const energy = energyOf(eRaw)
        // The rest→excited mix is the expensive part (a full color-space round trip). At zero
        // energy it resolves to exactly colA and `curA` is 0 regardless, so empty texels skip it.
        let col = d.vec4f(prm.colA)
        if (eRaw > 0.0) {
            const agitAvg = sRaw / std.max(eRaw, 1.0)
            col = mixFn(prm.colA, prm.colB, std.clamp(agitAvg, 0.0, 1.0))
        }
        const curA = energyCoverageAlpha(energy, 1.6) * col.w

        const out = composite(col, curA, idx)
        // The trail WRITE stays even when the read is baked out — dropping it would leave stale
        // content that ghosts for a frame when the user scrubs trails back up off zero.
        L.$.trailBuf[idx] = d.vec4f(out)
        std.textureStore(L.$.outTex, d.vec2u(x, y), out)
    }).$name(cfg.name)
}

/** The resolve for one tint: `params.color` with an alpha from how much landed on each texel, transparent between agents. */
function resolveTint(layout: {
    readonly $: {
        readonly accumE: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {readonly color: d.v4f}
    }
}, cfg: {res: number; name: string}): (x: number, y: number) => void {
    const RES = cfg.res
    return tgpu.fn([d.u32, d.u32])((x, y) => {
        'use gpu'
        const prm = layout.$.params
        const idx = y * d.u32(RES) + x
        const energy = d.f32(std.atomicLoad(layout.$.accumE[idx])) * (1.0 / 255.0)
        std.atomicStore(layout.$.accumE[idx], d.u32(0))
        const a = energyCoverageAlpha(energy, 1.6) * prm.color.w
        std.textureStore(layout.$.outTex, d.vec2u(x, y), d.vec4f(prm.color.x, prm.color.y, prm.color.z, a))
    }).$name(cfg.name)
}

/**
 * The resolve for the image relief: the average of the colors that landed on each texel,
 * weighted so nearer particles win, with an alpha from the total coverage. `fp` must match
 * the splat's; `alphaK` sets how fast coverage turns opaque.
 */
// Weighted-average color (RGB/W — the fixed-point scale cancels) + a coverage alpha from W.
function resolveWeightedColor(layout: ReliefRenderLayout, cfg: {fp: number; alphaK: number; name: string}): (x: number, y: number) => void {
    const FP = cfg.fp
    const ALPHA_K = cfg.alphaK
    return tgpu.fn([d.u32, d.u32])((x, y) => {
        'use gpu'
        const P = layout.$.params
        const idx = y * d.u32(P.outW) + x
        const wRaw = d.f32(std.atomicLoad(layout.$.accumW[idx]))
        const rRaw = d.f32(std.atomicLoad(layout.$.accumR[idx]))
        const gRaw = d.f32(std.atomicLoad(layout.$.accumG[idx]))
        const bRaw = d.f32(std.atomicLoad(layout.$.accumB[idx]))
        std.atomicStore(layout.$.accumR[idx], d.u32(0))
        std.atomicStore(layout.$.accumG[idx], d.u32(0))
        std.atomicStore(layout.$.accumB[idx], d.u32(0))
        std.atomicStore(layout.$.accumW[idx], d.u32(0))
        const denom = std.max(wRaw, d.f32(1))
        const r = rRaw / denom
        const g = gRaw / denom
        const b = bRaw / denom
        const alpha = energyCoverageAlpha(wRaw * (1.0 / FP), ALPHA_K)
        std.textureStore(layout.$.outTex, d.vec2u(x, y), d.vec4f(r, g, b, alpha))
    }).$name(cfg.name)
}

// ── The renderAgents surface ──────────────────────────────────────────────────────────────

/**
 * The step that draws every agent into the shared canvas, one variant per kind of agent.
 * `renderAgents` pairs each with its resolve; reach for `splat` alone when you need another
 * pairing.
 *
 * @example
 * ```ts
 * splat: {kernel: splat.orientedWorld(simLayout, {shape: 'arrow', res: 1024, splatRCap: 16, heading: 'velocity', agitation: 'buffer', comet: true, name: 'boidsSplat'}), threads: 'agents'}
 * ```
 * @tip `relief` dispatches over a 2D grid (`threads: 'grid'`); the other three dispatch over `agents`.
 * @see resolve, renderAgents, pointSlot
 */
export const splat = {
    /**
     * Agents with a heading: arrows, streaks, squares, dots or comets drawn along the way they
     * point.
     *
     * @example
     * ```ts
     * splat: {kernel: splat.orientedWorld(simLayout, {shape: 'arrow', res: 1024, splatRCap: 16, heading: 'velocity', agitation: 'buffer', comet: true, name: 'boidsSplat'}), threads: 'agents'}
     * ```
     * @see resolve.ramp, renderAgents.orientedWorld
     */
    orientedWorld: splatOrientedWorld,
    /**
     * Round motes with a live softness, placed, brightened and sized by `pointSlot` parts.
     *
     * @example
     * ```ts
     * splat: {kernel: splat.pointWorld(simLayout, {shape: 'glow', res: 1024, place: pointSlot.orbitalPlace(simLayout, {radius: 0.035, rate: 1.4}), brightness: pointSlot.twinkleBrightness(simLayout, {freq: 2}), bodyRadius: pointSlot.presenceRadius(simLayout), name: 'motesSplat'}), threads: 'agents'}
     * ```
     * @see resolve.tint, pointSlot, renderAgents.pointWorld
     */
    pointWorld: splatPointWorld,
    /**
     * A 3D swarm in perspective, recording speed for the ramp resolve.
     *
     * @example
     * ```ts
     * splat: {kernel: splat.volume(simLayout, {shape: 'dot', outRes: 1024, maxSplatSize: 6, extQ: 2.6, names: {splat: 'swarmSplat', glow: 'swarmGlowProfile', profile: 'swarmProfile'}}), threads: 'agents'}
     * ```
     * @see resolve.ramp, renderAgents.volume
     */
    volume: splatVolume,
    /**
     * Particles standing on an image, drawn in the child's color through a camera. Dispatch it
     * over a 2D grid.
     *
     * @example
     * ```ts
     * splat: {kernel: splat.relief(fieldLayout, {shape: 'dot', splatRCap: 24, aaW: 0.75, minAlpha: 0.02, fp: 1024, wobbleFreq: 1.3, perspK: 0.9, name: 'reliefSplat'}), threads: 'grid'}
     * ```
     * @see resolve.weightedColor, renderAgents.relief
     */
    relief: splatRelief,
} as const

/**
 * The step that turns the shared canvas into the output picture and clears it for the next
 * frame. Pair `ramp` with `orientedWorld` or `volume`, `tint` with `pointWorld`,
 * `weightedColor` with `relief`.
 *
 * @example
 * ```ts
 * resolve: {kernel: resolve.tint(simLayout, {res: 1024, name: 'motesResolve'}), threads: 'fixed', size: [1024, 1024]}
 * ```
 * @tip Dispatch a resolve with `threads: 'fixed'` over the canvas size. It must touch every texel to clear it.
 * @see splat, renderAgents
 */
export const resolve = {
    /**
     * Rest-to-excited color from two colors, with optional motion trails.
     *
     * @example
     * ```ts
     * resolve: {kernel: resolve.ramp(simLayout, {colorSpace: 2, res: 1024, invGain: 1 / 255, trails: 'on', name: 'boidsResolve'}), threads: 'fixed', size: [1024, 1024]}
     * ```
     * @see splat.orientedWorld, splat.volume
     */
    ramp: resolveRamp,
    /**
     * One color, with an alpha from coverage.
     *
     * @example
     * ```ts
     * resolve: {kernel: resolve.tint(simLayout, {res: 1024, name: 'motesResolve'}), threads: 'fixed', size: [1024, 1024]}
     * ```
     * @see splat.pointWorld
     */
    tint: resolveTint,
    /**
     * The average of the colors that landed, nearer particles first.
     *
     * @example
     * ```ts
     * resolve: {kernel: resolve.weightedColor(fieldLayout, {fp: 1024, alphaK: 1.6, name: 'reliefResolve'}), threads: 'fixed', size: [outW, outH]}
     * ```
     * @see splat.relief
     */
    weightedColor: resolveWeightedColor,
} as const
