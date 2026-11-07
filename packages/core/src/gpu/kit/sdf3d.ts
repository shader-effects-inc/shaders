/**
 * 3D signed-distance-field primitives + orthographic raymarch (`'use gpu'` TGSL). It backs the
 * sphere3D/cube3D/… shape-effect shaders (Glass, Neon, Crystal, LiquidMetal, …) which raymarch a
 * 3D solid orthographically into the SAME 0–1 "SDF field" UV space the 2D analytic shapes live
 * in, then read the field as a plug-compatible `vec4` (silhouette distance / −chord in .r, hit
 * pattern coords in .g/.b, view depth in .a).
 *
 * `Math.PI` / `Math.SQRT2` are pre-folded to module-const number literals (a `Math.*` member
 * access inside a `'use gpu'` body is not proven to fold).
 *
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 * CONTENTS
 *   A. The pure-CPU resolvers/tables (shape types, defaults, bounding radius, sub-prop
 *      auto-animate/mouse resolution, field-resolution buckets) — plain TS.
 *   B. 18 3D SDF primitives as exported `'use gpu'` tgpu.fns (the CPU-golden targets).
 *   C. `buildRaymarchedFieldFn` — the orthographic march compiled into a resolvable
 *      `tgpu.fn([d.vec2f, RotStruct, d.f32], d.vec4f)`; the shape's size params are baked into the
 *      caller-supplied `sdfFn` (a `tgpu.fn([d.vec3f], d.f32)`) and rotation / bounding-radius
 *      arrive as call arguments.
 *   D. The compute pre-march graph (`makeVolumetricFieldLayout` + `buildVolumetricFieldKernel`),
 *      the field samplers (`buildFieldSampleGraph`: bicubic Catmull-Rom + 4-tap bilinear), and a
 *      device-side allocator/dispatcher (`createVolumetricFieldCompute`).
 *
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 * CONSUMER WIRES — deliberately NOT here, see the inline markers below:
 *   - `createAnalytic3dSdfSampler` / `createAnalytic3dSdfSetup` — the per-frame CPU setup that
 *     resolves the shape JSON (incl. auto-animate sub-props) into the march uniforms and dispatches
 *     the compute pass only on a state change. The consumer assembles these from a `MarchParams`
 *     uniform + this module's graph builders + the per-frame `writeParams`. The pure CPU resolvers
 *     they depend on (`resolveShapeSubProp`, `shape3dBoundingRadius`, `resolveActiveFieldRes`,
 *     `rotateVecCpu`, …) ARE here.
 *   - The KitTexture handoff: registering the compute-written `fieldTexture` as a
 *     `params.registerComputeTexture(...)` binding the FRAGMENT samples (the compute↔fragment
 *     handshake), and binding the fragment-side `sampleLayout` (from `buildFieldSampleGraph`) to
 *     that texture + a `SampleParams` uniform. The samplers here are self-contained `tgpu.fn`s
 *     backed by a kit-owned bind-group layout (the kit/blur pattern); the concrete texture + the
 *     per-frame domain uniform are bound by the consumer/composer.
 */
import {tgpu, d, std} from './index'
import type {InferInput} from 'typegpu/data'
import type {TgpuRoot, TgpuTexture} from 'typegpu'
import {createGuardedCompute, type KitComputePipeline} from '../compute'
import {isMobileGpuViewport} from '../../utilities/device'
import {getSdfContentBounds} from '../../utilities/sdfBounds'
import {call, expr, floatE} from '../composer'
// Shared with the renderer's top-level mouse drivers so shape sub-props have the same feel —
// and the same sub-stepped stability at the clamped 0.1s off-screen delta.
import {applySpring} from '../frame'
import type {Expr, GpuFragmentParams, KitTexture, GpuComputeStep} from '../contract'
import {
    createSdfDataTexture,
    createSvgSdfSampler, createSvgSdfSamplerWithGradients, createAnalyticSdfSampler, driveAnalyticSubProps,
    buildAnalyticSdfFn, analyticSubPropValues,
} from './sdf'
import {resolveShapeType, shapeContentExtent} from '../../utilities/shapeEffectBounds'
import {MAX_METABALLS} from '../../utilities/sdf3d'

// Re-exported so existing importers keep resolving it from here; the canonical definition lives
// in ../../utilities/device.
export {isMobileGpuViewport}

// Re-exported so kit consumers get the ball-count cap alongside SHAPE3D_DEFAULTS; the canonical
// definition lives in ../../utilities/sdf3d (the CPU-only mirror).
export {MAX_METABALLS}

// Math constants as plain number literals — the pattern for values referenced INSIDE a
// `'use gpu'` body.
const PI = 3.141592653589793 //  Math.PI
const TWO_PI = 6.283185307179586 //  2 * Math.PI

// ─── A. Pure-CPU shape metadata + resolvers (plain TS) ─────────

export const SHAPE3D_TYPES = [
    'sphere3D', 'cube3D', 'torus3D', 'octahedron3D', 'cylinder3D', 'capsule3D',
    'cone3D', 'pyramid3D', 'prism3D', 'ellipsoid3D', 'diamond3D', 'link3D',
    'gem3D', 'helix3D', 'metaballs3D', 'dodecahedron3D', 'hemisphere3D',
    'ribbon3D', 'blob3D', 'gyroscope3D'
] as const
export type Shape3DType = typeof SHAPE3D_TYPES[number]

export function is3dShapeType(shapeType: string): shapeType is Shape3DType {
    return (SHAPE3D_TYPES as readonly string[]).includes(shapeType)
}

// Custom-SVG 3D geometry: the user's uploaded SDF lifted into 3D by extrusion
// (slab with rounded bevel).
export const SVG3D_TYPES = ['svgExtrude3D'] as const
export type Svg3DType = typeof SVG3D_TYPES[number]

export function isSvg3dShapeType(shapeType: string): shapeType is Svg3DType {
    return shapeType === 'svgExtrude3D'
}

/** True when the shape type produces a volumetric (raymarched) field — analytic 3D or SVG 3D. */
export function isVolumetricShapeType(shapeType: string): boolean {
    return is3dShapeType(shapeType) || isSvg3dShapeType(shapeType)
}

/** Defaults for the SVG 3D sub-props stored in the shape JSON (shared by sampler, bounds, and UI). */
export const SVG3D_DEFAULTS: Record<string, number> = {
    depth: 0.12, bevel: 0.02, rotX: 15, rotY: 30, rotZ: 0
}

// ─── CPU auto-animate resolution (mirrors renderer.ts applyEasing/phase math) ──

const bounceEase = (t: number): number => {
    const n1 = 7.5625
    const d1 = 2.75
    if (t < 1 / d1) return n1 * t * t
    if (t < 2 / d1) { t -= 1.5 / d1; return n1 * t * t + 0.75 }
    if (t < 2.5 / d1) { t -= 2.25 / d1; return n1 * t * t + 0.9375 }
    t -= 2.625 / d1; return n1 * t * t + 0.984375
}

const applyEasing = (t: number, easing: string): number => {
    switch (easing) {
        case 'linear': return t
        case 'quad': return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
        case 'expo':
            if (t === 0) return 0
            if (t === 1) return 1
            return t < 0.5 ? Math.pow(2, 20 * t - 10) / 2 : (2 - Math.pow(2, -20 * t + 10)) / 2
        case 'bounce': return bounceEase(t)
        case 'sine':
        default: return (1 - Math.cos(Math.PI * t)) / 2
    }
}

/**
 * Per-frame context for resolving shape sub-props: time (auto-animate phase), pointer
 * (mouse drivers) and a persistent per-key spring store (mouse smoothing/momentum).
 */
export interface ShapeSubPropFrame {
    /** Wall-clock seconds since sampler creation (drives auto-animate phase). */
    elapsed: number
    /** Seconds since the previous frame (drives mouse spring integration). */
    deltaTime: number
    /** Normalised pointer position in [0,1]. */
    pointerX: number
    pointerY: number
    /** Persistent spring state per sub-prop key (mouse smoothing/momentum). */
    springs: Map<string, { current: number; velocity: number }>
}

/**
 * Resolve a shape sub-prop value that may be a number, an auto-animate config, or a
 * mouse config. Auto base rate matches the renderer's prop drivers (0.2 cycles/sec →
 * 5s full cycle at speed 1); mouse mapping (axis → spring → curve → output range)
 * mirrors the renderer's resolveMouseMapForProp so behaviour matches top-level props.
 */
export function resolveShapeSubProp(value: any, fallback: number, frame: ShapeSubPropFrame, key: string): number {
    if (typeof value === 'number') return value
    if (value && typeof value === 'object') {
        if (value.type === 'auto-animate') {
            const globalT = frame.elapsed * (value.speed ?? 1) * 0.2
            const t01 = ((globalT % 1) + 1) % 1
            const t = value.mode === 'loop' ? t01 : (t01 < 0.5 ? t01 * 2 : (1 - t01) * 2)
            const phase = applyEasing(t, value.easing ?? value.waveform ?? 'sine')
            const oMin = typeof value.outputMin === 'number' ? value.outputMin : fallback
            const oMax = typeof value.outputMax === 'number' ? value.outputMax : fallback
            return oMin + phase * (oMax - oMin)
        }
        if (value.type === 'mouse') {
            const target = value.axis === 'y' ? frame.pointerY : frame.pointerX
            let st = frame.springs.get(key)
            if (!st) { st = { current: target, velocity: 0 }; frame.springs.set(key, st) }
            const [np, nv] = applySpring(st.current, st.velocity, target, value.smoothing ?? 0, value.momentum ?? 0, frame.deltaTime)
            st.current = np; st.velocity = nv
            const smoothed = Math.max(0.001, np)
            const oMin = typeof value.outputMin === 'number' ? value.outputMin : fallback
            const oMax = typeof value.outputMax === 'number' ? value.outputMax : fallback
            const exponent = Math.pow(2, -(value.curve ?? 0) * 2)
            return oMin + Math.pow(smoothed, exponent) * (oMax - oMin)
        }
    }
    return fallback
}

/** Static bound for a sub-prop value: the largest magnitude it can take (for bounding boxes). */
export function shapeSubPropMax(value: any, fallback: number): number {
    if (typeof value === 'number') return value
    if (value && typeof value === 'object' && (value.type === 'auto-animate' || value.type === 'mouse')) {
        const oMin = typeof value.outputMin === 'number' ? value.outputMin : fallback
        const oMax = typeof value.outputMax === 'number' ? value.outputMax : fallback
        return Math.max(oMin, oMax)
    }
    return fallback
}

// ─── Shape parameter schemas ───────────────────────────────────────────────────
// Defaults live here so core (sampler + bounds) and the editor UI agree.

export const SHAPE3D_DEFAULTS: Record<Shape3DType, Record<string, number>> = {
    sphere3D:     { radius: 0.35, rotX: 0,   rotY: 0,  rotZ: 0 },
    cube3D:       { sizeX: 0.27, sizeY: 0.27, sizeZ: 0.27, rounding: 0.02, rotX: 25, rotY: 35, rotZ: 0 },
    torus3D:      { radius: 0.3, tube: 0.12, rotX: 55, rotY: 0, rotZ: 0 },
    octahedron3D: { radius: 0.42, rotX: 10, rotY: 25, rotZ: 0 },
    cylinder3D:   { radius: 0.22, height: 0.26, rounding: 0.02, rotX: 60, rotY: 0, rotZ: 30 },
    capsule3D:    { radius: 0.16, height: 0.2, rotX: 0, rotY: 0, rotZ: 35 },
    cone3D:       { radius: 0.3, topRadius: 0.02, height: 0.3, rotX: 25, rotY: 0, rotZ: 15 },
    pyramid3D:    { size: 0.3, height: 0.45, rotX: -10, rotY: 30, rotZ: 0 },
    prism3D:      { radius: 0.28, height: 0.14, rotX: 25, rotY: 30, rotZ: 0 },
    ellipsoid3D:  { radiusX: 0.38, radiusY: 0.22, radiusZ: 0.3, rotX: 0, rotY: 0, rotZ: 20 },
    diamond3D:    { radius: 0.3, height: 0.42, rotX: 15, rotY: 0, rotZ: 0 },
    link3D:       { radius: 0.18, length: 0.15, tube: 0.08, rotX: 20, rotY: 30, rotZ: 0 },
    gem3D:        { radius: 0.3, height: 0.32, facets: 8, rotX: 15, rotY: 0, rotZ: 0 },
    helix3D:      { radius: 0.22, tube: 0.06, pitch: 0.16, rotX: 20, rotY: 0, rotZ: 0 },
    metaballs3D:  { balls: 4, ballRadius: 0.16, spread: 0.2, blend: 0.12, speed: 1, rotX: 0, rotY: 0, rotZ: 0 },
    dodecahedron3D: { radius: 0.34, rotX: 20, rotY: 10, rotZ: 0 },
    hemisphere3D: { radius: 0.4, cut: 0, rotX: 30, rotY: 0, rotZ: 0 },
    ribbon3D:     { width: 0.15, thickness: 0.03, length: 0.38, wave: 0.1, waveFrequency: 7, twist: 140, speed: 1, rotX: 12, rotY: 18, rotZ: 0 },
    blob3D:       { radius: 0.3, wobble: 0.07, wobbliness: 5, speed: 1, rotX: 0, rotY: 0, rotZ: 0 },
    gyroscope3D:  { radius: 0.34, width: 0.1, thickness: 0.028, rounding: 0.006, core: 0.14, speed: 1, rotX: 15, rotY: 0, rotZ: 0 }
}

/**
 * Conservative bounding radius (field UV units) for a 3D shape config — radius of
 * the bounding sphere, which contains the silhouette under any rotation.
 * Sub-prop values may be auto-animate configs; the max output bound is used.
 */
export function shape3dBoundingRadius(cfg: Record<string, any>): number {
    const dflt = SHAPE3D_DEFAULTS[cfg.type as Shape3DType] ?? {}
    const n = (k: string, f = 0.3) => shapeSubPropMax(cfg[k], dflt[k] ?? f)
    switch (cfg.type) {
        case 'cube3D': {
            const r = Math.hypot(n('sizeX'), n('sizeY'), n('sizeZ')) + n('rounding', 0)
            return r
        }
        case 'torus3D': return n('radius') + n('tube')
        case 'cylinder3D': return Math.hypot(n('radius'), n('height')) + n('rounding', 0)
        case 'capsule3D': return n('height') + n('radius')
        case 'cone3D': return Math.hypot(Math.max(n('radius'), n('topRadius')), n('height'))
        case 'pyramid3D': return Math.hypot(n('size') * Math.SQRT2, n('height') * 0.5)
        case 'prism3D': return Math.hypot(n('radius') * 1.1548, n('height'))
        case 'ellipsoid3D': return Math.max(n('radiusX'), n('radiusY'), n('radiusZ'))
        case 'diamond3D': return Math.max(n('radius'), n('height'))
        case 'link3D': return n('length') + n('radius') + n('tube')
        case 'gem3D': return Math.max(n('radius'), n('height'))
        case 'helix3D': return Math.hypot(n('radius') + n('tube'), n('pitch') * 3 * 0.5 + n('tube'))
        case 'metaballs3D': return n('spread') + n('ballRadius', 0.16) + n('blend', 0.12)
        case 'dodecahedron3D': return n('radius') * 1.32
        case 'hemisphere3D': return n('radius')
        // yc/zc wave offsets are orthogonal axes with amplitudes amp / 0.6·amp → the offset vector
        // magnitude is ≤ 1.2·amp; the twisted cross-section stays inside its own diagonal.
        case 'ribbon3D': return Math.hypot(n('length', 0.38), n('wave', 0.1) * 1.2 + Math.hypot(n('width', 0.15), n('thickness', 0.03))) + 0.01
        case 'blob3D': return n('radius', 0.3) + n('wobble', 0.07) + 0.02
        case 'gyroscope3D': return n('radius', 0.34) + n('thickness', 0.028) * 0.5 + n('width', 0.1) * 0.5 + n('rounding', 0.006)
        case 'sphere3D':
        case 'octahedron3D':
        default: return n('radius', 0.35)
    }
}

/** Fixed winding count for the helix (not user-exposed — `pitch` alone drives the
 *  height). Kept as a constant so the SDF, bounds, and march agree. */
export const HELIX_TURNS = 3

/** CPU-value rotation holder (a type alias — the consumer re-assembles it on TypeGPU uniforms;
 *  see CONSUMER WIRES). */
export interface RotUniforms { cx: any; sx: any; cy: any; sy: any; cz: any; sz: any }

/**
 * How the march fills the surface-locked pattern coords (.g/.b):
 *  - 'none'      → not computed (left 0). The cheapest; use when the consumer only
 *                  reads .r (field/chord) and .a (depth). Skips ALL pattern work.
 *  - 'raw'       → oblique projection of the raw shape-local hit position. Seamless,
 *                  rotates with the solid, zero extra SDF evals.
 *  - 'triplanar' → per-face axis pick from the local surface normal. Costs SIX extra
 *                  sdf() evals per hit (a full normal probe). Only Crystal needs this.
 *
 * Defaulting to 'none' means a shader pays for pattern coords only by opting in —
 * the 6-eval triplanar probe no longer runs for the effects that ignore .g/.b.
 */
export type SurfacePatternMode = 'none' | 'raw' | 'triplanar'

/**
 * How the march measures the in-solid chord (`.r = −chord/2` inside the silhouette):
 *  - 'span'      → first-entry → LAST-exit, treating internal air gaps as glass. A rear lobe
 *                  passing behind a front lobe steps the field discontinuously at the rear
 *                  lobe's silhouette (mid-surface of the front lobe), which gradient-driven
 *                  effects render as smeared halos where shapes overlap on screen.
 *  - 'firstLobe' → first-entry → FIRST-exit: the optical thickness of only the front lobe.
 *                  The field stays continuous across a rear lobe's silhouette, so overlaps
 *                  stay clean; discontinuities remain only at the front lobe's own silhouette,
 *                  where an edge is visually expected. Also cheaper for thin/multi-lobe shapes
 *                  (the interior march is shorter than the backward trace from the bounding
 *                  sphere). Glass uses this.
 */
export type ChordMode = 'span' | 'firstLobe'

/** CPU mirror of the GPU `rotateVec3` (same ZXY order) for plain JS scalars — used to project
 *  an oriented bounding box onto the field axes for the aspect-fit domain. Returns
 *  the rotated vector as {x, y, z}. */
export function rotateVecCpu(
    vx: number, vy: number, vz: number,
    s: { cx: number; sx: number; cy: number; sy: number; cz: number; sz: number }
): { x: number; y: number; z: number } {
    const x1 = vx * s.cy + vz * s.sy
    const z1 = -vx * s.sy + vz * s.cy
    const y2 = vy * s.cx - z1 * s.sx
    const z2 = vy * s.sx + z1 * s.cx
    const x3 = x1 * s.cz - y2 * s.sz
    const y3 = x1 * s.sz + y2 * s.cz
    return { x: x3, y: y3, z: z2 }
}

// ─── B. 3D SDF primitives (TGSL, point in shape-local space) ──────────────────
// Sign convention: negative = inside, positive = outside, zero = boundary. Each is an exported
// DualFn so the golden tests can CPU-evaluate it. The caller bakes the size params into a wrapper
// `tgpu.fn([d.vec3f], d.f32)` that the march (C) consumes.

/** Sphere: `|p| - r`. */
export const sdSphere = tgpu.fn([d.vec3f, d.f32], d.f32)((p, r) => {
    'use gpu'
    return std.length(p) - r
})

/** Rounded box: half-extents (hx,hy,hz), corner rounding. */
export const sdRoundBox = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32, d.f32], d.f32)((p, hx, hy, hz, rounding) => {
    'use gpu'
    const qx = std.abs(p.x) - hx + rounding
    const qy = std.abs(p.y) - hy + rounding
    const qz = std.abs(p.z) - hz + rounding
    const outer = std.length(d.vec3f(std.max(qx, 0.0), std.max(qy, 0.0), std.max(qz, 0.0)))
    const inner = std.min(std.max(qx, std.max(qy, qz)), 0.0)
    return outer + inner - rounding
})

/** Torus: ring radius `ringR` (in xz), tube radius `tubeR` (about y). */
export const sdTorus = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, ringR, tubeR) => {
    'use gpu'
    const qx = std.length(d.vec2f(p.x, p.z)) - ringR
    return std.length(d.vec2f(qx, p.y)) - tubeR
})

/** Octahedron (iq, bound version): scaled L1 distance — Lipschitz-safe with the march's
 *  min-step guard. */
export const sdOctahedron = tgpu.fn([d.vec3f, d.f32], d.f32)((p, s) => {
    'use gpu'
    return (std.abs(p.x) + std.abs(p.y) + std.abs(p.z) - s) * 0.57735027
})

/** Rounded cylinder: radius `r`, half-height `h`, corner rounding. */
export const sdRoundCylinder = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32], d.f32)((p, r, h, rounding) => {
    'use gpu'
    const dx = std.length(d.vec2f(p.x, p.z)) - r + rounding
    const dy = std.abs(p.y) - h + rounding
    const inner = std.min(std.max(dx, dy), 0.0)
    const outer = std.length(d.vec2f(std.max(dx, 0.0), std.max(dy, 0.0)))
    return inner + outer - rounding
})

/** Capsule: radius `r`, half-height `h` (about y). */
export const sdCapsule = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, r, h) => {
    'use gpu'
    const cyc = std.clamp(p.y, h * -1.0, h)
    return std.length(d.vec3f(p.x, p.y - cyc, p.z)) - r
})

// Capped cone (iq, exact): bottom radius r1, top radius r2, half-height h, Y axis.
// r2 near zero → cone; r2 > 0 → frustum.
export const sdCappedCone = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32], d.f32)((p, r1, r2, h) => {
    'use gpu'
    const qx = std.length(d.vec2f(p.x, p.z))
    const qy = p.y
    const k2x = r2 - r1
    const k2y = h * 2.0
    const cax = qx - std.min(qx, std.select(r2, r1, qy < 0.0))
    const cay = std.abs(qy) - h
    const dotNum = (r2 - qx) * k2x + (h - qy) * k2y
    const dotDen = std.max(k2x * k2x + k2y * k2y, 0.000001)
    const t = std.clamp(dotNum / dotDen, 0.0, 1.0)
    const cbx = qx - r2 + k2x * t
    const cby = qy - h + k2y * t
    // inside = (cbx < 0) AND (cay < 0). Expressed without `&&` via a nested select
    // (the inner test is only reached when cbx < 0).
    const sInner = std.select(d.f32(1), d.f32(-1), cay < 0.0)
    const s = std.select(d.f32(1), sInner, cbx < 0.0)
    const dca = cax * cax + cay * cay
    const dcb = cbx * cbx + cby * cby
    return s * std.sqrt(std.min(dca, dcb))
})

// Square pyramid via plane intersection: base half-width b, total height ht,
// centered (base y=−ht/2, apex y=+ht/2). Exact inside; a lower bound outside
// near edges, which sphere tracing + bisection handle fine.
export const sdPyramid = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, b, ht) => {
    'use gpu'
    const h2 = ht * 0.5
    const denom = std.sqrt(ht * ht + b * b)
    const dx = (ht * std.abs(p.x) + b * (p.y - h2)) / denom
    const dz = (ht * std.abs(p.z) + b * (p.y - h2)) / denom
    const dBase = p.y * -1.0 - h2
    return std.max(std.max(dx, dz), dBase)
})

// Hexagonal prism (iq): r = inradius (across flats) in xy, he = half-depth in z.
export const sdHexPrism = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, r, he) => {
    'use gpu'
    const kx = -0.8660254
    const ky = 0.5
    const kz = 0.57735
    const ax = std.abs(p.x)
    const ay = std.abs(p.y)
    const az = std.abs(p.z)
    const dk = std.min(kx * ax + ky * ay, 0.0) * 2.0
    const px = ax - dk * kx
    const py = ay - dk * ky
    const cx = std.clamp(px, kz * r * -1.0, kz * r)
    const ex = px - cx
    const ey = py - r
    const dHex = std.sqrt(ex * ex + ey * ey) * std.sign(py - r)
    const dZ = az - he
    const inner = std.min(std.max(dHex, dZ), 0.0)
    const outer = std.length(d.vec2f(std.max(dHex, 0.0), std.max(dZ, 0.0)))
    return inner + outer
})

// Ellipsoid (iq approximation — exact at the boundary, scaled-gradient nearby).
export const sdEllipsoid = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32], d.f32)((p, rx, ry, rz) => {
    'use gpu'
    const k0 = std.length(d.vec3f(p.x / rx, p.y / ry, p.z / rz))
    const k1 = std.length(d.vec3f(p.x / (rx * rx), p.y / (ry * ry), p.z / (rz * rz)))
    return k0 * (k0 - 1.0) / std.max(k1, 0.000001)
})

// Bicone "diamond" gem: 2D triangle profile (girdle radius r, half-height h)
// revolved around Y — exact via signed distance to the profile segment.
export const sdBicone = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, r, h) => {
    'use gpu'
    const qx = std.length(d.vec2f(p.x, p.z))
    const qy = std.abs(p.y)
    const abx = r * -1.0
    const aby = h
    const apx = qx - r
    const apy = qy
    const den = std.max(abx * abx + aby * aby, 0.000001)
    const t = std.clamp((apx * abx + apy * aby) / den, 0.0, 1.0)
    const dxx = apx - abx * t
    const dyy = apy - aby * t
    const dist = std.sqrt(dxx * dxx + dyy * dyy)
    // inside ⇔ below the girdle→apex line: h·qx + r·qy < r·h
    return dist * std.sign(h * qx + r * qy - r * h)
})

// Chain link (iq): straight section half-length le, ring radius r1, tube radius r2.
export const sdLink = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32], d.f32)((p, le, r1, r2) => {
    'use gpu'
    const qy = std.max(std.abs(p.y) - le, 0.0)
    const ring = std.length(d.vec2f(p.x, qy)) - r1
    return std.length(d.vec2f(ring, p.z)) - r2
})

// Polynomial smooth-min (iq) — Lipschitz-preserving blend of two SDFs.
export const smin = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((a, b, k) => {
    'use gpu'
    const h = std.clamp((b - a) / k * 0.5 + 0.5, 0.0, 1.0)
    return std.mix(b, a, h) - k * h * (1.0 - h)
})

// Faceted gem (brilliant cut): a revolved bicone profile (girdle radius r, half-
// height h) intersected with a vertical n-gon prism (vertical facets) and capped
// by a flat table plane on the crown. sides = number of girdle facets.
export const sdGem = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32], d.f32)((p, r, h, sides) => {
    'use gpu'
    const base = sdBicone(p, r, h)
    // 2D regular-polygon distance in the xz plane (vertical prism, infinite in y).
    // Inradius r·cos(π/n) places the polygon vertices on the girdle circle of radius r.
    const lenXZ = std.length(d.vec2f(p.x, p.z))
    const ang = std.atan2(p.z, p.x)
    const sector = TWO_PI / sides
    const idx = std.floor(ang / sector + 0.5)
    const ca = ang - idx * sector
    const ri = r * std.cos(PI / sides)
    const facet = lenXZ * std.cos(ca) - ri
    const dd = std.max(base, facet)
    // Flat table: slice the crown with a horizontal plane just below the apex.
    const table = p.y - h * 0.45
    return std.max(dd, table)
})

// Helix / coil: tube radius `tube` swept along a coil of radius R climbing `pitch`
// per turn for `turns` turns. Euclidean distance to the nearest point on an INFINITE
// helical centreline — the nearest sweep angle is θ ≈ atan2(z,x) + 2πk for the
// height-nearest winding `k` (in-plane exact; the residual from the slight angular
// pull is < 2%, well within the march's tolerance). The infinite strand is then cut
// to length by intersecting (`max`) with a slab |y| ≤ pitch·turns/2. Both halves are
// valid distance fields, so the result never over-estimates → the sphere march can't
// tunnel through the thin tube (which is what an end-angle clamp did, leaving gaps).
export const sdHelix = tgpu.fn([d.vec3f, d.f32, d.f32, d.f32, d.f32], d.f32)((p, R, tube, pitch, turns) => {
    'use gpu'
    const psi = std.atan2(p.z, p.x)                              // [-π, π]
    const k = std.floor(p.y / pitch - psi / TWO_PI + 0.5)
    const theta = psi + TWO_PI * k
    const ccx = R * std.cos(theta)
    const ccy = pitch * theta / TWO_PI
    const ccz = R * std.sin(theta)
    const dStrand = std.length(d.vec3f(p.x - ccx, p.y - ccy, p.z - ccz)) - tube
    const slab = std.abs(p.y) - pitch * turns * 0.5
    return std.max(dStrand, slab)
})

// Metaballs: up to MAX_METABALLS spheres of radius `r` at CPU-animated centres b0..b7, merged
// with smooth-min of strength `k` so they bulge and fuse as they move. (Explicit vec3 args rather
// than an array param: MarchParams keeps NAMED mb0..mb7 fields — strict uniform layout,
// full-write only — so the caller passes the fields and the kernel loops over a local array
// built from them, the patternPaints local-`var`-array pattern.) The BALL COUNT is CPU-side:
// unused centres are parked far outside the march's bounding sphere (y = 99), where smin is
// exactly min (the blend only engages within `k` of the nearer distance) — so they contribute
// nothing and no count uniform/recompile is needed.
const MetaballArray = d.arrayOf(d.vec3f, MAX_METABALLS)
const METABALL_COUNT = MAX_METABALLS // same-module const number for the 'use gpu' loop bound
export const sdMetaballs = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.f32, d.f32], d.f32,
)((p, b0, b1, b2, b3, b4, b5, b6, b7, r, k) => {
    'use gpu'
    // `d.vec3f(bN)` copies — assigning the arg reference directly is invalid ("references
    // cannot be assigned").
    const balls = MetaballArray()
    balls[0] = d.vec3f(b0)
    balls[1] = d.vec3f(b1)
    balls[2] = d.vec3f(b2)
    balls[3] = d.vec3f(b3)
    balls[4] = d.vec3f(b4)
    balls[5] = d.vec3f(b5)
    balls[6] = d.vec3f(b6)
    balls[7] = d.vec3f(b7)
    let dd = std.length(p.sub(balls[0])) - r
    for (let i = 1; i < METABALL_COUNT; i++) {
        dd = smin(dd, std.length(p.sub(balls[i])) - r, k)
    }
    return dd
})

// Dodecahedron (12 pentagonal faces): intersection of half-spaces whose normals
// are the icosahedron's vertex directions (0,±a,±b) and cyclic. r = inradius.
export const sdDodecahedron = tgpu.fn([d.vec3f, d.f32], d.f32)((p, r) => {
    'use gpu'
    const q = d.vec3f(std.abs(p.x), std.abs(p.y), std.abs(p.z))
    const a = std.dot(q, d.vec3f(0.0, 0.5257311, 0.8506508))
    const b = std.dot(q, d.vec3f(0.8506508, 0.0, 0.5257311))
    const c = std.dot(q, d.vec3f(0.5257311, 0.8506508, 0.0))
    return std.max(std.max(a, b), c) - r
})

// Cut sphere (iq, exact): sphere of radius r sliced by the plane y = h, keeping the
// part below it. h = 0 → hemisphere; h < 0 → shallow cap; h > 0 → most of the sphere.
export const sdCutSphere = tgpu.fn([d.vec3f, d.f32, d.f32], d.f32)((p, r, h) => {
    'use gpu'
    const w = std.sqrt(std.max(r * r - h * h, 0.0))
    const qx = std.length(d.vec2f(p.x, p.z))
    const qy = p.y
    const s = std.max(
        (h - r) * qx * qx + w * w * (h + r - qy * 2.0),
        h * qx - w * qy
    )
    const case0 = std.length(d.vec2f(qx, qy)) - r
    const case1 = h - qy
    const case2 = std.length(d.vec2f(qx - w, qy - h))
    const inner = std.select(case2, case1, qx < w) // qx < w ? case1 : case2
    return std.select(inner, case0, s < 0.0)        // s < 0  ? case0 : inner
})

// Flowing ribbon: a twisted flat band along x whose centreline rides two CPU-animated sine waves
// (y and z, decorrelated phases) — the "simulated cloth ribbon". The cross-section is a rounded
// 2D box, counter-rotated by the twist angle at this x. Bending + twisting shrink true distances,
// so the result is scaled by a CPU-computed Lipschitz factor `lip` (≤ 1) to stay a lower bound the
// sphere march can't tunnel through.
//   w/th/len = half width/thickness/length; amp/freq/ph1/ph2 = the wave state;
//   twRate = twist radians per x unit; twPh = twist phase.
export const sdRibbon = tgpu.fn(
    [d.vec3f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32,
)((p, w, th, len, amp, freq, ph1, ph2, twRate, twPh, lip) => {
    'use gpu'
    const yc = amp * std.sin(freq * p.x + ph1)
    const zc = amp * 0.6 * std.sin(freq * 0.8 * p.x + ph2)
    const qy = p.y - yc
    const qz = p.z - zc
    const ang = twRate * p.x + twPh
    const c = std.cos(ang)
    const s = std.sin(ang)
    const ry = qy * c + qz * s
    const rz = qz * c - qy * s
    const round = th * 0.4
    const bx = std.abs(ry) - w + round
    const by = std.abs(rz) - th + round
    const d2 = std.length(d.vec2f(std.max(bx, 0.0), std.max(by, 0.0))) + std.min(std.max(bx, by), 0.0) - round
    const dx = std.abs(p.x) - len
    const dd = std.length(d.vec2f(std.max(d2, 0.0), std.max(dx, 0.0))) + std.min(std.max(d2, dx), 0.0)
    return dd * lip
})

// Wobbling blob: a sphere whose surface breathes with two CPU-phase-animated products of plane
// waves — a single connected organic body (unlike metaballs' fusing spheres). `lip` compensates
// the wave slope (CPU: 1 / (1 + amp·freq·k)).
export const sdWobbleBlob = tgpu.fn(
    [d.vec3f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32,
)((p, r, amp, freq, ph1, ph2, lip) => {
    'use gpu'
    const w = std.sin(p.x * freq + ph1) * std.sin(p.y * freq * 1.3 + ph2)
        + std.sin(p.y * freq * 0.8 + ph2 * 1.1) * std.sin(p.z * freq * 1.1 + ph1 * 0.9)
    const dd = std.length(p) - r - amp * w * 0.5
    return dd * lip
})

// One flat "wedding band" ring: a rounded-rectangle cross-section (radial half-thickness ht,
// axial half-width hw, edge rounding rnd) revolved around the local y axis at radius R. Exact
// (revolution of an exact 2D SDF).
export const sdFlatBand = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32,
)((radialDist, axialDist, R, ht, hw, rnd) => {
    'use gpu'
    const bx = std.abs(radialDist - R) - ht + rnd
    const by = std.abs(axialDist) - hw + rnd
    const outer = std.length(d.vec2f(std.max(bx, 0.0), std.max(by, 0.0)))
    const inner = std.min(std.max(bx, by), 0.0)
    return outer + inner - rnd
})

// Gyroscope: three nested FLAT rings (wedding-band profile: rectangular cross-section with
// beveled edges) at radius, 0.78·radius, 0.56·radius on different axes, each spinning about its
// own axis with CPU-animated angles (cos/sin pairs c1..s3), plus a core sphere. `width` is the
// full axial band width, `thick` the full radial thickness, `rnd` the edge bevel radius
// (CPU-clamped to half the smaller extent — at max the band becomes a tube). Rigid rotations of
// exact revolved profiles → Lipschitz-safe with no scale factor.
export const sdGyroscope = tgpu.fn(
    [d.vec3f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32,
)((p, radius, width, thick, rnd, core, c1, s1, c2, s2, c3, s3) => {
    'use gpu'
    const hw = width * 0.5
    const ht = thick * 0.5
    const dCore = std.length(p) - core
    // Ring 1: xz-plane band (axis y), spun about X.
    const p1y = p.y * c1 + p.z * s1
    const p1z = p.z * c1 - p.y * s1
    const d1 = sdFlatBand(std.length(d.vec2f(p.x, p1z)), p1y, radius, ht, hw, rnd)
    // Ring 2: xy-plane band (axis z), spun about Y.
    const p2x = p.x * c2 - p.z * s2
    const p2z = p.z * c2 + p.x * s2
    const d2 = sdFlatBand(std.length(d.vec2f(p2x, p.y)), p2z, radius * 0.78, ht, hw, rnd)
    // Ring 3: yz-plane band (axis x), spun about Z.
    const p3x = p.x * c3 + p.y * s3
    const p3y = p.y * c3 - p.x * s3
    const d3 = sdFlatBand(std.length(d.vec2f(p3y, p.z)), p3x, radius * 0.56, ht, hw, rnd)
    return std.min(std.min(dCore, d1), std.min(d2, d3))
})

// ─── C. Orthographic raymarch field ────────────────────────────────────────────
//
// A 3D shape is raymarched orthographically (camera looking down +z) through the
// same 0–1 "SDF field" UV space the 2D analytic shapes live in. The sampler
// returns a 2D scalar field that is plug-compatible with applyGlassEffect & co:
//
//   outside the silhouette → the minimum 3D distance encountered along the ray,
//     which for orthographic rays equals the 2D distance to the projected silhouette
//   inside the silhouette  → minus HALF THE CHORD LENGTH through the solid,
//     i.e. the real optical thickness of the glass along the view ray.
//
// Rotation is rigid, so instead of rotating the point inside the march loop we
// rotate the ray origin/direction once per sample — the SDFs stay axis-aligned.

/** CPU-precomputed sin/cos rotation, passed to the march as one struct argument (ZXY Euler). */
export const RotStruct = d.struct({
    cx: d.f32, sx: d.f32,
    cy: d.f32, sy: d.f32,
    cz: d.f32, sz: d.f32,
})

/** Rotate a vector by ZXY Euler angles using CPU-precomputed sin/cos uniforms (GPU). */
export const rotateVec3 = tgpu.fn([d.vec3f, RotStruct], d.vec3f)((v, rot) => {
    'use gpu'
    // Y axis
    const x1 = v.x * rot.cy + v.z * rot.sy
    const z1 = v.z * rot.cy - v.x * rot.sy
    // X axis
    const y2 = v.y * rot.cx - z1 * rot.sx
    const z2 = v.y * rot.sx + z1 * rot.cx
    // Z axis
    const x3 = x1 * rot.cz - y2 * rot.sz
    const y3 = x1 * rot.sz + y2 * rot.cz
    return d.vec3f(x3, y3, z2)
})

/** A shape SDF baked with its size params — the piece the march consumes. */
export type BakedSdf = (p: any) => any

/**
 * Shared orthographic raymarch core: turns any Lipschitz-safe 3D SDF into a 2D field sampler.
 * Returns a resolvable `tgpu.fn([d.vec2f, RotStruct, d.f32], d.vec4f)`:
 *   .r = scalar field (silhouette distance outside, −chord/2 inside)
 *   .g/.b = surface-locked pattern coords (per `patternMode`; 0 when 'none')
 *   .a = view-space hit depth (tEnter) along the ray
 *
 * `fuv` = field UV, `rot` = the CPU-precomputed rotation struct, `rBound` = a bounding-sphere
 * radius that always contains the solid (used to skip empty space analytically).
 *
 * The shape's size params are baked into `sdfFn` by the caller (a `tgpu.fn([d.vec3f], d.f32)`
 * reading its size uniforms); `patternMode` is a build-time JS string that selects the emitted
 * branch (only 'triplanar' emits the 6-eval normal probe; 'none' emits no pattern code).
 */
export function buildRaymarchedFieldFn(sdfFn: BakedSdf, patternMode: SurfacePatternMode = 'none', chordMode: ChordMode = 'span') {
    // ADAPTIVE march budget. Both sphere-trace loops early-out the instant the ray leaves the bounding
    // sphere (`t > tEnd` / `s > span`), so a shallow, face-on shape converges in a handful of steps no
    // matter how high the ceiling is — the budget is only ever SPENT by DEEP extrusions viewed at
    // GRAZING/oblique angles, where a ray travels far nearly PARALLEL to the surface and needs many
    // small steps to converge. A fixed cap exhausted there (streaky depth → noisy normals; missed
    // silhouette texels → notches) — the artifact seen on deep extrudes AND long analytic shapes alike,
    // and it's asymptotic (any fixed cap has a deeper case that breaks it). So the per-shape ceiling
    // scales with the bounding radius (`steps = clamp(rBound·SCALE, MIN, MAX)`, computed in the body):
    // deep shapes get room; shallow/moderate stay low; and even deep shapes only pay on their grazing
    // rays (the early-out breaks the rest). Mobile caps lower (VRAM/ALU). Bisection stays at 10.
    const STEP_MIN = isMobileGpuViewport() ? 64 : 96
    const STEP_MAX = isMobileGpuViewport() ? 224 : 384
    const STEP_SCALE = isMobileGpuViewport() ? 140 : 220

    const march = tgpu.fn([d.vec2f, RotStruct, d.f32], d.vec4f)((fuv, rot, rBound) => {
        'use gpu'
        // Field UV → shape-local ray. Y is flipped so positive rotX tips the top of the shape away
        // from the viewer (screen UVs are y-down here).
        const ro0 = d.vec3f(fuv.x - 0.5, 0.5 - fuv.y, -1.2)
        const rd0 = d.vec3f(0.0, 0.0, 1.0)
        const ro = rotateVec3(ro0, rot)
        const rd = rotateVec3(rd0, rot)

        // Per-shape step ceiling from the bounding radius (see STEP_* above): deep shapes get room,
        // shallow ones stay low; the loop early-outs keep the actual count far below this for all but
        // the grazing rays that need it.
        const steps = d.i32(std.clamp(rBound * d.f32(STEP_SCALE), d.f32(STEP_MIN), d.f32(STEP_MAX)))

        // Analytic bounding-sphere intersection: skip the empty space so the whole step budget is
        // spent near the surface. perp = exact distance from the ray line to the origin.
        const b = std.dot(ro, rd)
        const perp = std.sqrt(std.max(std.dot(ro, ro) - b * b, 0.0))
        const halfSpan = std.sqrt(std.max(rBound * rBound - perp * perp, 0.0))
        const tStart = std.max(b * -1.0 - halfSpan, 0.0)
        const tEnd = b * -1.0 + halfSpan + 0.01

        // Accumulators initialised from f32-typed expressions (a bare `10.0`/`0.0` literal binding
        // transpiles to i32 and truncates — the documented accumulator trap).
        let t = tStart
        let tPrev = tStart
        let minD = d.f32(10)
        let tMinD = tStart
        let hit = d.f32(0)

        // Hit test is d < 0 (strictly inside), not a small positive epsilon: the march must CROSS
        // the surface so the bisection bracket [tPrev, t] holds a true sign change.
        for (let i = 0; i < steps; i++) {
            const p = ro.add(rd.mul(t))
            const dd = sdfFn(p)
            if (dd < minD) {
                minD = dd
                tMinD = t
            }
            if (dd < 0.0) {
                hit = d.f32(1)
                break
            }
            if (t > tEnd) { break }
            tPrev = t
            t = t + std.max(dd, 0.002)
        }

        // A near MISS stores its closest approach to the shape (the outside distance). The march
        // only samples the ray at its sphere-tracing steps, so the sampled minimum JUMPS wherever a
        // neighbouring ray takes one step more or fewer — hard contour rings a short way outside a
        // 3D silhouette, visible in anything that ramps off the outside field (glows, flames, edge
        // feathering). A few golden-section steps around the nearest sample find the true closest
        // approach, making the outside field continuous. Far misses (> 0.15) skip it: they are
        // smooth enough and most of the texels.
        if (hit < 0.5 && minD < 0.15) {
            const reach = std.max(minD, 0.002) * 2.0
            let lo = tMinD - reach
            let hi = tMinD + reach
            for (let i = 0; i < 5; i++) {
                const m1 = hi - (hi - lo) * 0.618034
                const m2 = lo + (hi - lo) * 0.618034
                const f1 = sdfFn(ro.add(rd.mul(m1)))
                const f2 = sdfFn(ro.add(rd.mul(m2)))
                if (f1 < f2) { hi = m2 } else { lo = m1 }
            }
            minD = std.min(minD, sdfFn(ro.add(rd.mul((lo + hi) * 0.5))))
        }

        // Outputs — miss defaults; all refinement work below only runs for rays that actually hit.
        let fieldVal = minD
        let patternU = d.f32(0)
        let patternV = d.f32(0)
        // A miss reads as "far behind" (past the bounding-sphere exit), NOT 0 — otherwise
        // silhouette-adjacent taps would invert the rim normal inward.
        let viewDepth = tEnd + 0.4

        if (hit > 0.5) {
            // Bisection-refine the entry point between the last outside sample and the hit sample,
            // then one SECANT step through the bracket's field values. Bisection alone leaves an
            // error of bracket/1024 — up to ~5e-5 for a wide first step, which differentiates into
            // several degrees of per-texel normal noise (grain under strong refraction). The field
            // is near-linear across the final bracket, so its zero crossing is effectively exact.
            let eLo = tPrev
            let eHi = t
            let dLo = sdfFn(ro.add(rd.mul(eLo)))
            let dHi = sdfFn(ro.add(rd.mul(eHi)))
            for (let i = 0; i < 10; i++) {
                const mid = (eLo + eHi) * 0.5
                const dm = sdfFn(ro.add(rd.mul(mid)))
                if (dm > 0.0) {
                    eLo = mid
                    dLo = dm
                } else {
                    eHi = mid
                    dHi = dm
                }
            }
            const tEnter = eLo + (eHi - eLo) * std.clamp(dLo / std.max(dLo - dHi, 0.000000001), d.f32(0), d.f32(1))

            // Exit point — chordMode is a JS string, so exactly ONE branch is emitted (like
            // patternMode below).
            let tExit = tEnter
            if (chordMode === 'firstLobe') {
                // 'firstLobe': march FORWARD through the interior from the entry bracket's
                // inside end (eHi) to the FIRST boundary crossing. Interior sphere tracing
                // steps by |dd| (the boundary is at least that far away), so it can't skip
                // out of the lobe. See the ChordMode doc for why this keeps overlapping
                // lobes clean.
                //
                // The FIRST step is taken unconditionally: the entry bisection also moves eHi
                // on dm == 0, so sdf(eHi) can read EXACTLY 0 — a `dd >= 0` break there would
                // collapse the bracket to a zero-width chord, punching a per-texel "hole" in
                // the field that re-rolls every re-march (flickering speckles). A lobe thinner
                // than the 0.002 step floor still resolves: the exit lands inside the
                // [eHi, eHi + 0.002] bracket and the bisection finds it.
                let sInPrev = eHi
                let sIn = eHi + 0.002
                for (let i = 0; i < steps; i++) {
                    const dd = sdfFn(ro.add(rd.mul(sIn)))
                    if (dd >= 0.0) { break } // crossed out of the lobe → true bracket
                    sInPrev = sIn
                    sIn = sIn + std.max(dd * -1.0, 0.002)
                    if (sIn > tEnd) { break }
                }
                let fLo = sInPrev
                let fHi = sIn
                let gLo = sdfFn(ro.add(rd.mul(fLo)))
                let gHi = sdfFn(ro.add(rd.mul(fHi)))
                for (let i = 0; i < 10; i++) {
                    const mid = (fLo + fHi) * 0.5
                    const dm = sdfFn(ro.add(rd.mul(mid)))
                    if (dm < 0.0) {
                        fLo = mid
                        gLo = dm
                    } else {
                        fHi = mid
                        gHi = dm
                    }
                }
                // Secant through the bracket (fLo inside, fHi outside), as for the entry.
                tExit = fLo + (fHi - fLo) * std.clamp(gLo / std.min(gLo - gHi, -0.000000001), d.f32(0), d.f32(1))
            }
            if (chordMode === 'span') {
                // 'span': sphere-trace BACKWARD from the bounding-sphere exit toward the shape.
                // Tracing from outside converges as fast as the entry march. For non-convex shapes
                // (torus) this measures first-entry → last-exit, treating internal air gaps as glass.
                const roB = ro.add(rd.mul(tEnd)) // on the bounding sphere → outside the solid
                const span = tEnd - tStart + 0.01
                let s = d.f32(0)
                let sPrev = d.f32(0)
                for (let i = 0; i < steps; i++) {
                    const p = roB.sub(rd.mul(s))
                    const dd = sdfFn(p)
                    if (dd < 0.0) { break } // strictly inside → true bracket
                    if (s > span) { break }
                    sPrev = s
                    s = s + std.max(dd, 0.002)
                }

                // Refine the exit the same way as the entry (sPrev outside, s at the hit)
                let xLo = sPrev
                let xHi = s
                let hLo = sdfFn(roB.sub(rd.mul(xLo)))
                let hHi = sdfFn(roB.sub(rd.mul(xHi)))
                for (let i = 0; i < 10; i++) {
                    const mid = (xLo + xHi) * 0.5
                    const dm = sdfFn(roB.sub(rd.mul(mid)))
                    if (dm > 0.0) {
                        xLo = mid
                        hLo = dm
                    } else {
                        xHi = mid
                        hHi = dm
                    }
                }
                // Secant through the bracket (xLo outside, xHi inside), as for the entry.
                tExit = tEnd - (xLo + (xHi - xLo) * std.clamp(hLo / std.max(hLo - hHi, 0.000000001), d.f32(0), d.f32(1)))
            }

            const halfChord = std.max(tExit - tEnter, 0.0) * 0.5
            fieldVal = halfChord * -1.0
            viewDepth = tEnter

            // GBA (only Crystal reads .g/.b; everything else uses .r). patternMode is a JS string,
            // so these branches resolve at graph-build time — 'none' emits no pattern code and only
            // 'triplanar' emits the 6-eval normal probe.
            if (patternMode === 'raw') {
                // Oblique projection of the shape-local hit position. Folding hp.z in means faces
                // running "straight back" along the view still get 2D variation. The null direction
                // (-0.7,-0.5,1) isn't parallel to any axis-aligned face → every face gets detail;
                // still a single linear projection → seamless; still shape-local → rotates with it.
                const hp = ro.add(rd.mul(tEnter))
                patternU = hp.x + hp.z * 0.7
                patternV = hp.y + hp.z * 0.5
            }
            if (patternMode === 'triplanar') {
                const hp = ro.add(rd.mul(tEnter))
                const e2 = 0.004
                const nlx = sdfFn(d.vec3f(hp.x + e2, hp.y, hp.z)) - sdfFn(d.vec3f(hp.x - e2, hp.y, hp.z))
                const nly = sdfFn(d.vec3f(hp.x, hp.y + e2, hp.z)) - sdfFn(d.vec3f(hp.x, hp.y - e2, hp.z))
                const nlz = sdfFn(d.vec3f(hp.x, hp.y, hp.z + e2)) - sdfFn(d.vec3f(hp.x, hp.y, hp.z - e2))
                const anx = std.abs(nlx)
                const anyv = std.abs(nly)
                const anz = std.abs(nlz)
                // Per-face axis pick with a per-axis constant offset (adjacent faces don't share a
                // pattern). The `&&` (anx dominant) is a nested select: branch1 only when
                // anx>=anyv AND anx>=anz, else the anyv-vs-anz pick.
                const branch1 = d.vec2f(hp.y + 2.17, hp.z + 5.31)
                const branch2 = d.vec2f(hp.z + 7.73, hp.x + 1.13)
                const branch3 = d.vec2f(hp.x, hp.y)
                const branch23 = std.select(branch3, branch2, anyv >= anz)
                const innerPick = std.select(branch23, branch1, anx >= anz)
                const pattern = std.select(branch23, innerPick, anx >= anyv)
                patternU = pattern.x
                patternV = pattern.y
            }
        }

        return d.vec4f(fieldVal, patternU, patternV, viewDepth)
    }).$name('raymarchedField')

    return march
}

/** Thin wrapper for `buildRaymarchedFieldFn` (the march fn is the value itself; rotation +
 *  bounding radius are call arguments, not build-time closures). */
export function buildRaymarchedField(sdfFn: BakedSdf, patternMode: SurfacePatternMode = 'none', chordMode: ChordMode = 'span') {
    return buildRaymarchedFieldFn(sdfFn, patternMode, chordMode)
}

// ─── D. Compute pre-march + field samplers ─────────────────────────────────────

/** Field texture format. rgba32float is required (fp16 quantizes the depth channel → visible
 *  banding in the reconstructed normals) and is NOT filterable in WebGPU, hence the manual
 *  bilinear/Catmull-Rom in the samplers. */
export const VOLUMETRIC_FIELD_FORMAT = 'rgba32float' as const

/** Field-texture resolution. The near-binary edge mask of Neon/Holographic/Crystal exposes the
 *  texel lattice as jagged silhouettes once a field texel spans more than ~1 device pixel. 1536
 *  is the middle ground (1.5× the linear resolution of 1024, 2.25× the march cost, 36 MB fp32 vs
 *  16 MB) which — together with the aspect-fit domain — gets the strokes below device-pixel
 *  frequency without the 4× hit of 2048. The march only re-runs on shape-state changes. */
export const VOLUMETRIC_FIELD_RES = 1536

/** Mobile/tablet field-texture ALLOCATION (VRAM ceiling + activeRes cap). A coarse-pointer device
 *  gets a smaller field than the 1536 desktop default: its GPU can't afford the desktop march
 *  (cost scales with res²) and the texture is heavy (768² is 9 MB fp32 vs 1024²'s 16 MB vs 1536²'s
 *  36 MB). Why 768 and not 1024: on mobile the active region is sized at HALF the device-pixel
 *  footprint, so a typical phone shape already resolves to ≤640 — well under this cap. */
export const VOLUMETRIC_FIELD_RES_MOBILE = 768

/**
 * Pick the pre-march field resolution for this device, once, at shader-build time.
 *
 * The field texture is allocated when the sampler node is built and is NEVER reallocated
 * afterwards (the renderer keeps resize free of texture reallocation), so we can't track the live
 * canvas size. The one signal available synchronously is the device class, which is exactly what
 * separates "needs the full desktop field" from "a phone that shouldn't pay for a 2K texture".
 */
export function resolveVolumetricFieldRes(): number {
    return isMobileGpuViewport() ? VOLUMETRIC_FIELD_RES_MOBILE : VOLUMETRIC_FIELD_RES
}

/** Active-region resolution buckets — all multiples of 64 so a dispatch of activeRes² lands on an
 *  exact workgroup-64 grid. Spaced ~1.25–1.3× apart: coarse enough that a slow scale animation
 *  crosses a boundary only occasionally (each crossing is one re-march), fine enough that a small
 *  shape isn't over-marched. */
const ACTIVE_RES_BUCKETS = [256, 320, 384, 512, 640, 768, 896, 1024, 1280, 1536]

/**
 * Pick the active march/sample resolution for a shape from its on-screen footprint.
 *
 * The field domain covers the shape's AABB (spanX × spanY in field-UV). The shader maps field-UV
 * → screen with the `scale` uniform, so matching one field texel to one device pixel along the
 * tighter axis means activeRes ≈ max(spanX,spanY) · scale · Hdev. We snap up to a bucket and clamp
 * to the texture allocation. `prevRes` adds a shrink deadband so a shape hovering on a boundary
 * doesn't re-march every frame.
 */
export function resolveActiveFieldRes(
    spanXv: number, spanYv: number, scale: number, canvasHeightDevicePx: number, maxRes: number, prevRes: number
): number {
    if (!(canvasHeightDevicePx > 0) || !(scale > 0)) return maxRes // unknown footprint → full res
    // Mobile: target HALF the device-pixel footprint (a quarter of the texels marched + sampled).
    // The march cost scales with res², so this is the single biggest lever for an animated 3D
    // shape on a phone; the reconstructed reflection goes slightly softer — fine for reflective
    // (not near-binary-edge) surfaces at phone pixel densities.
    const footprintScale = isMobileGpuViewport() ? 0.5 : 1
    const targetPx = Math.max(spanXv, spanYv) * scale * canvasHeightDevicePx * footprintScale
    let res = maxRes
    for (const b of ACTIVE_RES_BUCKETS) {
        if (b > maxRes) break
        if (b >= targetPx) { res = b; break }
    }
    // Shrink deadband: only drop below the current res once the footprint has fallen well under it.
    if (prevRes > 0 && res < prevRes && targetPx > prevRes * 0.78) res = Math.min(prevRes, maxRes)
    return res
}

/**
 * March-pass uniform. Holds the rotation, bounding radius, aspect-fit domain (span/origin), active
 * resolution, AND the shape's size superset (pA..pD + MAX_METABALLS metaball centres) so the
 * caller's baked `sdfFn` reads its params from `layout.$.params.pA` etc. It is a superset; unused
 * fields are never read by the shape's kernel. (The consumer's per-frame CPU setup writes this
 * each time the shape state changes — see CONSUMER WIRES.)
 *
 * SLOT MAP — the mb0..mb7 vec3 slots are the metaballs3D centres; the other CPU-animated shapes
 * reuse them positionally as generic parameter carriers. The `update` switch in
 * `createAnalytic3dSdfSetup` packs them and the matching `sdfFn` branch unpacks them — keep the
 * two in sync with this table:
 *   metaballs3D: mb0..mb7 = ball centres (unused balls parked at y = 99)
 *   ribbon3D:    mb0 = (waveAmp, waveFreq, phase1) · mb1 = (phase2, twistRate, twistPhase)
 *                mb2.x = Lipschitz factor
 *   blob3D:      mb0 = (wobbleAmp, wobbliness, phase1) · mb1.x = phase2 · mb2.x = Lipschitz factor
 *   gyroscope3D: mb0/mb1/mb2 = (cos, sin, –) of the three ring angles · mb3.x = bevel radius
 */
export const MarchParams = d.struct({
    // Strict uniform layout (no uniform_buffer_standard_layout): a struct-typed member must be
    // 16-aligned with ≥ roundUp(16, sizeOf) bytes before the next member — pad the 24-byte
    // RotStruct to 32 so `rBound` can't land at offset 24 (invalid on Safari/FF/Chromium ≲ 150).
    // FULL-WRITE ONLY: never `.patch()` a MarchParams uniform — typegpu's partial writer does
    // not unwrap Decorated members, so a patch would treat `rot` as a leaf and corrupt it.
    rot: d.align(16, d.size(32, RotStruct)),
    rBound: d.f32,
    spanX: d.f32, spanY: d.f32,
    originX: d.f32, originY: d.f32,
    activeRes: d.u32,
    pA: d.f32, pB: d.f32, pC: d.f32, pD: d.f32,
    mb0: d.vec3f, mb1: d.vec3f, mb2: d.vec3f, mb3: d.vec3f,
    mb4: d.vec3f, mb5: d.vec3f, mb6: d.vec3f, mb7: d.vec3f,
})

/**
 * Build the GPU-free compute bind-group layout for the pre-march (kit/blur `build*Graph` pattern):
 *   - `field`  → the write-only rgba32float storage texture the march fills.
 *   - `params` → the `MarchParams` uniform.
 *
 * The caller threads the returned `layout` through both the baked `sdfFn` (which reads
 * `layout.$.params.pA` …) and `buildVolumetricFieldKernel`, so everything shares one layout / one
 * bind group. Returned separately from the kernel so the parent can bake `sdfFn` against
 * `layout.$.params` before the march + kernel are built.
 */
export function makeVolumetricFieldLayout() {
    const layout = tgpu.bindGroupLayout({
        field: {storageTexture: d.textureStorage2d(VOLUMETRIC_FIELD_FORMAT, 'write-only')},
        params: {uniform: MarchParams},
    })
    return {layout, MarchParams}
}

export type VolumetricFieldLayout = ReturnType<typeof makeVolumetricFieldLayout>['layout']

/**
 * The compute kernel that pre-marches the field into the storage texture. Dispatched 2D over the
 * ACTIVE block (see D3 rules: TGSL `/` is always float, so a 1D→2D index reconstruction is unsafe
 * — the guarded pipeline gives clean `(cx, cy)` u32 + an auto bounds guard, and we
 * `dispatchThreads(activeRes, activeRes)` each frame). Each active texel maps to its field-UV over
 * the aspect-fit domain (span/origin) at the active density, marches, and writes the top-left
 * `activeRes × activeRes` block of the maxRes texture.
 */
export function buildVolumetricFieldKernel(layout: VolumetricFieldLayout, marchFn: ReturnType<typeof buildRaymarchedFieldFn>) {
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const activeResF = d.f32(p.activeRes)
        const uv = d.vec2f(
            p.originX + (d.f32(cx) + 0.5) / activeResF * p.spanX,
            p.originY + (d.f32(cy) + 0.5) / activeResF * p.spanY,
        )
        const field = marchFn(uv, p.rot, p.rBound)
        // Storage-texture write: 2-arg textureStore (NO mip level).
        std.textureStore(layout.$.field, d.vec2u(cx, cy), field)
    }).$name('volumetricFieldMarch')
}

/**
 * Fragment-side sampler uniform: the aspect-fit domain + active resolution + bounding radius. (The
 * composer binds this + the field texture into the sampler layout — see CONSUMER WIRES.)
 */
export const SampleParams = d.struct({
    originX: d.f32, originY: d.f32,
    spanX: d.f32, spanY: d.f32,
    activeResF: d.f32,
    rBound: d.f32,
})

/**
 * Build the field samplers over a kit-owned bind-group layout (kit/blur pattern):
 *   - `field`  → the rgba32float texture bound SAMPLED (`d.texture2d(d.f32)`).
 *   - `params` → the `SampleParams` uniform (domain + active res + bounding radius).
 *
 * Returns two resolvable `tgpu.fn([d.vec2f], d.vec4f)` samplers:
 *   - `fieldSample`     — bicubic Catmull-Rom on .r/.a + inner-2×2 neighbourhood clamp + bilinear
 *                         .g/.b + far-field circle blend. Keeps the near-binary silhouette/normal
 *                         edges crisp under magnification.
 *   - `fieldSampleFast` — 4-tap bilinear (same domain map + far-field extension). Used by
 *                         finite-difference gradient taps, which don't benefit from the bicubic.
 *
 * rgba32float is not filterable, so texels are read with an integer `std.textureLoad(field, coords,
 * 0)` (3-arg SAMPLED read — level 0), NOT `.sample()`; there is no `/maxRes` normalization.
 *
 * CONSUMER WIRES: register the compute `fieldTexture` (from `createVolumetricFieldCompute`) as a
 * `params.registerComputeTexture(...)` KitTexture the fragment samples, bind it + a `SampleParams`
 * uniform to the returned `sampleLayout`, and drive the domain values each frame. The `texelSpan`
 * an effect uses for a fast-sampler early-out is `max(spanX, spanY) / activeRes` (CPU-computable
 * from the same values).
 */
export function buildFieldSampleGraph() {
    const sampleLayout = tgpu.bindGroupLayout({
        // rgba32float is 'unfilterable-float'; the TypeGPU default entry sampleType is 'float'
        // (filterable), which only validates against a float32 view on devices with the optional
        // `float32-filterable` feature (absent on most Android GPUs). The field is only ever read
        // via textureLoad below, so declare it unfilterable explicitly.
        field: {texture: d.texture2d(d.f32), sampleType: 'unfilterable-float'},
        params: {uniform: SampleParams},
    })

    // `nearestAt`: clamp the integer texel to the active block and load it (no normalization).
    const fetchTexel = tgpu.fn([d.f32, d.f32], d.vec4f)((xi, yi) => {
        'use gpu'
        const maxT = sampleLayout.$.params.activeResF - 1.0
        const tx = d.u32(std.clamp(xi, d.f32(0), maxT))
        const ty = d.u32(std.clamp(yi, d.f32(0), maxT))
        return std.textureLoad(sampleLayout.$.field, d.vec2u(tx, ty), 0)
    }).$name('fieldFetchTexel')

    const fieldSample = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
        'use gpu'
        const p = sampleLayout.$.params
        // The texture only covers the padded bounding square but effects sample far beyond it
        // (Glass shades the whole canvas). The domain border is guaranteed outside the silhouette,
        // so extending by the euclidean distance to the clamped point keeps the field growing
        // monotonically, like the march.
        const uxc = std.clamp(uv.x, p.originX, p.originX + p.spanX)
        const uyc = std.clamp(uv.y, p.originY, p.originY + p.spanY)
        const outsideDist = std.length(d.vec2f(uv.x - uxc, uv.y - uyc))
        const px = (uxc - p.originX) / p.spanX * p.activeResF - 0.5
        const py = (uyc - p.originY) / p.spanY * p.activeResF - 0.5
        const x0 = std.floor(px)
        const y0 = std.floor(py)
        const fx = px - x0
        const fy = py - y0

        // Catmull-Rom weights for the fractional offsets.
        const tx2 = fx * fx
        const tx3 = tx2 * fx
        const wx0 = tx3 * -0.5 + tx2 - fx * 0.5
        const wx1 = tx3 * 1.5 - tx2 * 2.5 + 1.0
        const wx2 = tx3 * -1.5 + tx2 * 2.0 + fx * 0.5
        const wx3 = tx3 * 0.5 - tx2 * 0.5
        const ty2 = fy * fy
        const ty3 = ty2 * fy
        const wy0 = ty3 * -0.5 + ty2 - fy * 0.5
        const wy1 = ty3 * 1.5 - ty2 * 2.5 + 1.0
        const wy2 = ty3 * -1.5 + ty2 * 2.0 + fy * 0.5
        const wy3 = ty3 * 0.5 - ty2 * 0.5

        // 4×4 neighbourhood fetch (kIJ = row I, col J → texel offset (J-1, I-1)).
        const k00 = fetchTexel(x0 - 1.0, y0 - 1.0)
        const k01 = fetchTexel(x0, y0 - 1.0)
        const k02 = fetchTexel(x0 + 1.0, y0 - 1.0)
        const k03 = fetchTexel(x0 + 2.0, y0 - 1.0)
        const k10 = fetchTexel(x0 - 1.0, y0)
        const k11 = fetchTexel(x0, y0)
        const k12 = fetchTexel(x0 + 1.0, y0)
        const k13 = fetchTexel(x0 + 2.0, y0)
        const k20 = fetchTexel(x0 - 1.0, y0 + 1.0)
        const k21 = fetchTexel(x0, y0 + 1.0)
        const k22 = fetchTexel(x0 + 1.0, y0 + 1.0)
        const k23 = fetchTexel(x0 + 2.0, y0 + 1.0)
        const k30 = fetchTexel(x0 - 1.0, y0 + 2.0)
        const k31 = fetchTexel(x0, y0 + 2.0)
        const k32 = fetchTexel(x0 + 1.0, y0 + 2.0)
        const k33 = fetchTexel(x0 + 2.0, y0 + 2.0)

        // Bicubic .r (silhouette mask) + .a (surface normal / crease depth).
        const rAcc =
            k00.r * wx0 * wy0 + k01.r * wx1 * wy0 + k02.r * wx2 * wy0 + k03.r * wx3 * wy0 +
            k10.r * wx0 * wy1 + k11.r * wx1 * wy1 + k12.r * wx2 * wy1 + k13.r * wx3 * wy1 +
            k20.r * wx0 * wy2 + k21.r * wx1 * wy2 + k22.r * wx2 * wy2 + k23.r * wx3 * wy2 +
            k30.r * wx0 * wy3 + k31.r * wx1 * wy3 + k32.r * wx2 * wy3 + k33.r * wx3 * wy3
        const aAcc =
            k00.a * wx0 * wy0 + k01.a * wx1 * wy0 + k02.a * wx2 * wy0 + k03.a * wx3 * wy0 +
            k10.a * wx0 * wy1 + k11.a * wx1 * wy1 + k12.a * wx2 * wy1 + k13.a * wx3 * wy1 +
            k20.a * wx0 * wy2 + k21.a * wx1 * wy2 + k22.a * wx2 * wy2 + k23.a * wx3 * wy2 +
            k30.a * wx0 * wy3 + k31.a * wx1 * wy3 + k32.a * wx2 * wy3 + k33.a * wx3 * wy3

        // Bilinear g/b (pattern coords) from the inner 2×2 — Catmull-Rom would ring their seams.
        const bl = std.mix(std.mix(k11, k12, fx), std.mix(k21, k22, fx), fy)

        // Neighbourhood-clamp the cubic .r/.a to the inner 2×2 min/max (Catmull-Rom overshoots
        // across the field's silhouette kink; the clamp removes the ringing).
        const rMin = std.min(std.min(k11.r, k12.r), std.min(k21.r, k22.r))
        const rMax = std.max(std.max(k11.r, k12.r), std.max(k21.r, k22.r))
        const aMin = std.min(std.min(k11.a, k12.a), std.min(k21.a, k22.a))
        const aMax = std.max(std.max(k11.a, k12.a), std.max(k21.a, k22.a))
        const aOut = std.clamp(aAcc, aMin, aMax)

        // Far-field: blend the border-extension distance toward the bounding-circle distance so a
        // distant soft falloff (Neon glow) is perfectly circular past the blend band.
        const rNear = std.clamp(rAcc, rMin, rMax) + outsideDist
        const rCircle = std.length(d.vec2f(uv.x - 0.5, uv.y - 0.5)) - p.rBound
        const rOut = std.mix(rNear, std.max(rCircle, rNear), std.smoothstep(d.f32(0), 0.25, outsideDist))
        return d.vec4f(rOut, bl.g, bl.b, aOut)
    }).$name('fieldSample')

    const fieldSampleFast = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
        'use gpu'
        const p = sampleLayout.$.params
        const uxc = std.clamp(uv.x, p.originX, p.originX + p.spanX)
        const uyc = std.clamp(uv.y, p.originY, p.originY + p.spanY)
        const outsideDist = std.length(d.vec2f(uv.x - uxc, uv.y - uyc))
        const px = (uxc - p.originX) / p.spanX * p.activeResF - 0.5
        const py = (uyc - p.originY) / p.spanY * p.activeResF - 0.5
        const x0 = std.floor(px)
        const y0 = std.floor(py)
        const fx = px - x0
        const fy = py - y0
        const t00 = fetchTexel(x0, y0)
        const t10 = fetchTexel(x0 + 1.0, y0)
        const t01 = fetchTexel(x0, y0 + 1.0)
        const t11 = fetchTexel(x0 + 1.0, y0 + 1.0)
        const bl = std.mix(std.mix(t00, t10, fx), std.mix(t01, t11, fx), fy)
        const rNear = bl.r + outsideDist
        const rCircle = std.length(d.vec2f(uv.x - 0.5, uv.y - 0.5)) - p.rBound
        const rOut = std.mix(rNear, std.max(rCircle, rNear), std.smoothstep(d.f32(0), 0.25, outsideDist))
        return d.vec4f(rOut, bl.g, bl.b, bl.a)
    }).$name('fieldSampleFast')

    return {sampleLayout, fieldSample, fieldSampleFast, SampleParams}
}

/** Device-side volumetric field pre-march: allocates the field texture + march uniform, builds the
 *  guarded compute pipeline, and exposes a per-frame `writeParams` + variable-count `dispatch`. */
export interface VolumetricFieldCompute {
    /** The guarded pipeline (already bound); dispatch it with `dispatch(activeRes)`. */
    computeStep: KitComputePipeline
    /** rgba32float, storage + sampled. The consumer registers this as a compute KitTexture the
     *  fragment samples (CONSUMER WIRES). */
    fieldTexture: TgpuTexture
    /** Fixed texture allocation (device-class dependent). */
    maxRes: number
    /** Current active linear resolution (≤ maxRes). */
    readonly activeRes: number
    /** Write the whole march uniform (the parent's per-frame CPU setup drives this). */
    writeParams(value: InferInput<typeof MarchParams>): void
    /** Dispatch a 2D `activeRes × activeRes` guarded grid, filling the top-left block. */
    dispatch(activeRes: number): void
}

/**
 * Allocate + wire the compute pre-march. `layout` + `kernel` come from `makeVolumetricFieldLayout`
 * / `buildVolumetricFieldKernel` (the caller bakes the shape `sdfFn` against `layout.$.params`
 * first). The guarded pipeline is built WITHOUT a pre-bound size so each frame can
 * `dispatch(activeRes)` a clean 2D grid.
 *
 * NOTE: this is the device-touching allocator (like kit/blur's `createGaussianBlurCompute`); it is
 * NOT exercised under vitest (no GPU) — only tsc-checked. The march/kernel/sampler CORRECTNESS is
 * verified on real GPU by the shape-effect shaders. CONSUMER WIRES: register `fieldTexture` as a
 * compute KitTexture + bind the fragment sampler layout (see `buildFieldSampleGraph`), and drive
 * `writeParams` / `dispatch` from the per-frame CPU setup (the createAnalytic3dSdfSetup equivalent).
 */
/** A MarchParams uniform handle (from `root.createUniform(MarchParams)`) the caller can share with
 *  the compute — so the setup's per-frame `update` writes the SAME buffer the kernel reads. */
export interface MarchParamsUniform {
    readonly buffer: unknown
    write(value: InferInput<typeof MarchParams>): void
}

/** Extra wiring for `createVolumetricFieldCompute`:
 *  - `paramsUniform`  → reuse a caller-owned MarchParams uniform (the shape setup writes it each
 *                       frame) instead of creating a fresh one. The kernel's bind group binds it.
 *  - `sdfSourceTexture` → the 2D SDF DATA texture the SVG-extrusion `sdfFn` samples inside the march
 *                       (`layout` MUST be a `makeVolumetricSvgFieldLayout()` layout with the
 *                       `sdfSource` entry). Bound into the SAME compute bind group. */
export interface VolumetricFieldComputeExtra {
    paramsUniform?: MarchParamsUniform
    sdfSourceTexture?: unknown
    /** Additional bind groups the march fn reads (a consumer's OWN param uniform — the voxel
     *  pre-march's `VoxelMarchParams`). Bound after the field/params group. */
    extraBindGroups?: unknown[]
}

export function createVolumetricFieldCompute(
    root: TgpuRoot,
    layout: VolumetricFieldLayout,
    kernel: ReturnType<typeof buildVolumetricFieldKernel>,
    onCleanup: (cb: () => void) => void,
    extra: VolumetricFieldComputeExtra = {},
): VolumetricFieldCompute {
    const maxRes = resolveVolumetricFieldRes()
    // rgba32float, storage (kernel write) + sampled (fragment read). Never reallocated on resize.
    const fieldTexture = root.createTexture({size: [maxRes, maxRes], format: VOLUMETRIC_FIELD_FORMAT}).$usage('storage', 'sampled')
    const paramsUniform = extra.paramsUniform ?? root.createUniform(MarchParams)
    onCleanup(() => fieldTexture.destroy())

    const bindGroup = root.createBindGroup(layout, {
        field: fieldTexture,
        params: paramsUniform.buffer,
        ...(extra.sdfSourceTexture ? {sdfSource: extra.sdfSourceTexture} : {}),
    } as never)

    // Guarded pipeline with NO pre-bound size — dispatched per frame at the active resolution.
    let pipeline = createGuardedCompute(root, (cx: number, cy: number) => {
        'use gpu'
        kernel(cx, cy)
    }, {bindGroup})
    for (const bg of extra.extraBindGroups ?? []) pipeline = pipeline.with(bg as never)

    let activeRes = maxRes
    return {
        computeStep: pipeline,
        fieldTexture,
        maxRes,
        get activeRes() { return activeRes },
        writeParams(value: InferInput<typeof MarchParams>) { paramsUniform.write(value) },
        dispatch(res: number) {
            activeRes = res
            pipeline.dispatchThreads(res, res)
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// E. Shape-effect CONSUMER API
// ═══════════════════════════════════════════════════════════════════════════════════════
//
// These wire the raymarch/compute/sampler pieces (A–D) into the shape-effect shaders (Glass,
// Crystal, Neon, Emboss, LiquidMetal, Plastic, Holographic, Frost, ThinFilm). The one binding that
// a consuming shader must close is documented as `CONSUMER WIRES`.

/**
 * SVG-extrusion variant of the march bind-group layout: same `field` + `params` as
 * `makeVolumetricFieldLayout`, PLUS a SAMPLED `sdfSource` entry — the 2D SDF DATA texture the
 * extrusion `sdfFn` reads (via `textureLoad`) inside the march. Kept as a SEPARATE function (rather
 * than an optional flag on `makeVolumetricFieldLayout`) so the analytic path's layout TYPE stays
 * free of the optional entry — the SVG `sdfFn` reads `layout.$.sdfSource` off THIS layout's precise
 * type, and `createVolumetricFieldCompute`'s `sdfSourceTexture` extra binds it into the same group.
 */
export function makeVolumetricSvgFieldLayout() {
    const layout = tgpu.bindGroupLayout({
        field: {storageTexture: d.textureStorage2d(VOLUMETRIC_FIELD_FORMAT, 'write-only')},
        params: {uniform: MarchParams},
        sdfSource: {texture: d.texture2d(d.f32)},
    })
    return {layout, MarchParams}
}

export type VolumetricSvgFieldLayout = ReturnType<typeof makeVolumetricSvgFieldLayout>['layout']

/**
 * FN-ARG field samplers — the CONSUMER-WIREABLE form of `buildFieldSampleGraph`. Same
 * bicubic/bilinear math, but the field texture + the six SampleParams (origin/span/activeRes/rBound)
 * arrive as FN ARGUMENTS instead of a captured kit-owned bind-group layout. This is the fn-arg
 * path: the consuming shape-effect FRAGMENT passes the compute field texture's `.accessor()` (a
 * group-1 `registerComputeTexture` binding) + its own SampleParams accessors (declared as
 * `extraFields`, see `VOLUMETRIC_FIELD_EXTRA_FIELDS`). The captured-layout `buildFieldSampleGraph`
 * cannot be used directly by a fragment because the pass manager only binds the four composer-pinned
 * groups (uniforms/textures/samplers/external) — it has no mechanism to bind a shader-owned 5th
 * bind group — so a fragment that referenced `sampleLayout.$.field` would resolve but never get
 * bound. The fn-arg form routes everything through the standard groups.
 *
 * rgba32float is not filterable → texels are read with integer `std.textureLoad(field, coords, 0)`.
 */
export function buildFieldSampleGraphArgs() {
    // Bicubic Catmull-Rom on .r/.a + inner-2×2 clamp + bilinear .g/.b + far-field circle blend.
    const fieldSampleArg = tgpu.fn(
        [d.vec2f, d.texture2d(d.f32), d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
        d.vec4f,
    )((uv, field, originX, originY, spanX, spanY, activeResF, rBound) => {
        'use gpu'
        const uxc = std.clamp(uv.x, originX, originX + spanX)
        const uyc = std.clamp(uv.y, originY, originY + spanY)
        const outsideDist = std.length(d.vec2f(uv.x - uxc, uv.y - uyc))
        const px = (uxc - originX) / spanX * activeResF - 0.5
        const py = (uyc - originY) / spanY * activeResF - 0.5
        const x0 = std.floor(px)
        const y0 = std.floor(py)
        const fx = px - x0
        const fy = py - y0

        const tx2 = fx * fx
        const tx3 = tx2 * fx
        const wx0 = tx3 * -0.5 + tx2 - fx * 0.5
        const wx1 = tx3 * 1.5 - tx2 * 2.5 + 1.0
        const wx2 = tx3 * -1.5 + tx2 * 2.0 + fx * 0.5
        const wx3 = tx3 * 0.5 - tx2 * 0.5
        const ty2 = fy * fy
        const ty3 = ty2 * fy
        const wy0 = ty3 * -0.5 + ty2 - fy * 0.5
        const wy1 = ty3 * 1.5 - ty2 * 2.5 + 1.0
        const wy2 = ty3 * -1.5 + ty2 * 2.0 + fy * 0.5
        const wy3 = ty3 * 0.5 - ty2 * 0.5

        const maxT = activeResF - 1.0
        const cxm = d.u32(std.clamp(x0 - 1.0, d.f32(0), maxT))
        const cx0 = d.u32(std.clamp(x0, d.f32(0), maxT))
        const cx1 = d.u32(std.clamp(x0 + 1.0, d.f32(0), maxT))
        const cx2 = d.u32(std.clamp(x0 + 2.0, d.f32(0), maxT))
        const cym = d.u32(std.clamp(y0 - 1.0, d.f32(0), maxT))
        const cy0 = d.u32(std.clamp(y0, d.f32(0), maxT))
        const cy1 = d.u32(std.clamp(y0 + 1.0, d.f32(0), maxT))
        const cy2 = d.u32(std.clamp(y0 + 2.0, d.f32(0), maxT))

        const k00 = std.textureLoad(field, d.vec2u(cxm, cym), 0)
        const k01 = std.textureLoad(field, d.vec2u(cx0, cym), 0)
        const k02 = std.textureLoad(field, d.vec2u(cx1, cym), 0)
        const k03 = std.textureLoad(field, d.vec2u(cx2, cym), 0)
        const k10 = std.textureLoad(field, d.vec2u(cxm, cy0), 0)
        const k11 = std.textureLoad(field, d.vec2u(cx0, cy0), 0)
        const k12 = std.textureLoad(field, d.vec2u(cx1, cy0), 0)
        const k13 = std.textureLoad(field, d.vec2u(cx2, cy0), 0)
        const k20 = std.textureLoad(field, d.vec2u(cxm, cy1), 0)
        const k21 = std.textureLoad(field, d.vec2u(cx0, cy1), 0)
        const k22 = std.textureLoad(field, d.vec2u(cx1, cy1), 0)
        const k23 = std.textureLoad(field, d.vec2u(cx2, cy1), 0)
        const k30 = std.textureLoad(field, d.vec2u(cxm, cy2), 0)
        const k31 = std.textureLoad(field, d.vec2u(cx0, cy2), 0)
        const k32 = std.textureLoad(field, d.vec2u(cx1, cy2), 0)
        const k33 = std.textureLoad(field, d.vec2u(cx2, cy2), 0)

        const rAcc =
            k00.r * wx0 * wy0 + k01.r * wx1 * wy0 + k02.r * wx2 * wy0 + k03.r * wx3 * wy0 +
            k10.r * wx0 * wy1 + k11.r * wx1 * wy1 + k12.r * wx2 * wy1 + k13.r * wx3 * wy1 +
            k20.r * wx0 * wy2 + k21.r * wx1 * wy2 + k22.r * wx2 * wy2 + k23.r * wx3 * wy2 +
            k30.r * wx0 * wy3 + k31.r * wx1 * wy3 + k32.r * wx2 * wy3 + k33.r * wx3 * wy3
        const aAcc =
            k00.a * wx0 * wy0 + k01.a * wx1 * wy0 + k02.a * wx2 * wy0 + k03.a * wx3 * wy0 +
            k10.a * wx0 * wy1 + k11.a * wx1 * wy1 + k12.a * wx2 * wy1 + k13.a * wx3 * wy1 +
            k20.a * wx0 * wy2 + k21.a * wx1 * wy2 + k22.a * wx2 * wy2 + k23.a * wx3 * wy2 +
            k30.a * wx0 * wy3 + k31.a * wx1 * wy3 + k32.a * wx2 * wy3 + k33.a * wx3 * wy3

        const bl = std.mix(std.mix(k11, k12, fx), std.mix(k21, k22, fx), fy)

        const rMin = std.min(std.min(k11.r, k12.r), std.min(k21.r, k22.r))
        const rMax = std.max(std.max(k11.r, k12.r), std.max(k21.r, k22.r))
        const aMin = std.min(std.min(k11.a, k12.a), std.min(k21.a, k22.a))
        const aMax = std.max(std.max(k11.a, k12.a), std.max(k21.a, k22.a))
        const aOut = std.clamp(aAcc, aMin, aMax)

        const rNear = std.clamp(rAcc, rMin, rMax) + outsideDist
        const rCircle = std.length(d.vec2f(uv.x - 0.5, uv.y - 0.5)) - rBound
        const rOut = std.mix(rNear, std.max(rCircle, rNear), std.smoothstep(d.f32(0), 0.25, outsideDist))
        return d.vec4f(rOut, bl.g, bl.b, aOut)
    }).$name('fieldSampleArg')

    // Bicubic value + ANALYTIC gradient in one tap: `.r` = the SOFT-clamped Catmull-Rom field,
    // `.g/.b` = a cubic B-spline gradient w.r.t. the sample UV (∂r/∂u, ∂r/∂v — the same units as
    // a forward difference of the sampler), `.a` = the hard-clamped Catmull-Rom depth (matching
    // fieldSampleArg). One 16-load tap replaces the centre + two offset taps of a
    // finite-difference stencil (48 loads → 16), and the gradient is texel-localised: a wide FD
    // stencil turns every chord discontinuity (overlapping lobes) into a stencil-wide band of
    // exploded gradients, this confines it to the reconstruction footprint.
    //
    // Two deliberate deviations from fieldSampleArg's Catmull-Rom + hard clamp:
    //   - GRADIENT from cubic B-SPLINE derivative weights (same 4×4 texels). Catmull-Rom
    //     interpolates, so at a slope kink in the chord field (facet boundaries, band rims,
    //     occlusion contours) its derivative rings — overshooting and sign-flipping with
    //     sub-texel phase — rendering every interior 3D crease as a stippled hairline. The
    //     B-spline basis is non-negative (approximating), so its derivative is ring-free and C1:
    //     creases resolve as smooth ~2-texel bevels.
    //   - VALUE stays exact Catmull-Rom (interpolating — an approximating value would BIAS the
    //     silhouette zero-crossing and draw a dark thin-chord outline around the shape), but the
    //     anti-ringing neighbourhood clamp is a SOFT clamp: the hard clamp saturates per texel
    //     cell, making the reconstruction C0 exactly on the silhouette kink so the contour
    //     stair-steps at the texel lattice (a thin bright fresnel rim shows every step). The
    //     polynomial soft knee keeps ringing bounded while staying C1, so the contour glides.
    //     Flat neighbourhoods have ~zero texel range → the knee collapses → no effect.
    // NOTE `.g/.b` are gradients, NOT the surface pattern coords — for patternMode 'none'
    // consumers (Glass) that channel is unused.
    const fieldSampleGradArg = tgpu.fn(
        [d.vec2f, d.texture2d(d.f32), d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
        d.vec4f,
    )((uv, field, originX, originY, spanX, spanY, activeResF, rBound) => {
        'use gpu'
        const uxc = std.clamp(uv.x, originX, originX + spanX)
        const uyc = std.clamp(uv.y, originY, originY + spanY)
        const outsideDist = std.length(d.vec2f(uv.x - uxc, uv.y - uyc))
        const px = (uxc - originX) / spanX * activeResF - 0.5
        const py = (uyc - originY) / spanY * activeResF - 0.5
        const x0 = std.floor(px)
        const y0 = std.floor(py)
        const fx = px - x0
        const fy = py - y0

        // Catmull-Rom weights (the `.r` VALUE + `.a` DEPTH reconstruction, matching fieldSampleArg).
        const tx2 = fx * fx
        const tx3 = tx2 * fx
        const wx0 = tx3 * -0.5 + tx2 - fx * 0.5
        const wx1 = tx3 * 1.5 - tx2 * 2.5 + 1.0
        const wx2 = tx3 * -1.5 + tx2 * 2.0 + fx * 0.5
        const wx3 = tx3 * 0.5 - tx2 * 0.5
        const ty2 = fy * fy
        const ty3 = ty2 * fy
        const wy0 = ty3 * -0.5 + ty2 - fy * 0.5
        const wy1 = ty3 * 1.5 - ty2 * 2.5 + 1.0
        const wy2 = ty3 * -1.5 + ty2 * 2.0 + fy * 0.5
        const wy3 = ty3 * 0.5 - ty2 * 0.5

        // Cubic B-spline weights + their derivatives (the GRADIENT reconstruction — ring-free,
        // see the header comment). Value weights: ((1−t)³, 3t³−6t²+4, −3t³+3t²+3t+1, t³)/6.
        const oxf = 1.0 - fx
        const bx0 = oxf * oxf * oxf * 0.16666667
        const bx1 = tx3 * 0.5 - tx2 + 0.66666667
        const bx2 = tx3 * -0.5 + tx2 * 0.5 + fx * 0.5 + 0.16666667
        const bx3 = tx3 * 0.16666667
        const dbx0 = oxf * oxf * -0.5
        const dbx1 = tx2 * 1.5 - fx * 2.0
        const dbx2 = tx2 * -1.5 + fx + 0.5
        const dbx3 = tx2 * 0.5
        const oyf = 1.0 - fy
        const by0 = oyf * oyf * oyf * 0.16666667
        const by1 = ty3 * 0.5 - ty2 + 0.66666667
        const by2 = ty3 * -0.5 + ty2 * 0.5 + fy * 0.5 + 0.16666667
        const by3 = ty3 * 0.16666667
        const dby0 = oyf * oyf * -0.5
        const dby1 = ty2 * 1.5 - fy * 2.0
        const dby2 = ty2 * -1.5 + fy + 0.5
        const dby3 = ty2 * 0.5

        const maxT = activeResF - 1.0
        const cxm = d.u32(std.clamp(x0 - 1.0, d.f32(0), maxT))
        const cx0 = d.u32(std.clamp(x0, d.f32(0), maxT))
        const cx1 = d.u32(std.clamp(x0 + 1.0, d.f32(0), maxT))
        const cx2 = d.u32(std.clamp(x0 + 2.0, d.f32(0), maxT))
        const cym = d.u32(std.clamp(y0 - 1.0, d.f32(0), maxT))
        const cy0 = d.u32(std.clamp(y0, d.f32(0), maxT))
        const cy1 = d.u32(std.clamp(y0 + 1.0, d.f32(0), maxT))
        const cy2 = d.u32(std.clamp(y0 + 2.0, d.f32(0), maxT))

        const k00 = std.textureLoad(field, d.vec2u(cxm, cym), 0)
        const k01 = std.textureLoad(field, d.vec2u(cx0, cym), 0)
        const k02 = std.textureLoad(field, d.vec2u(cx1, cym), 0)
        const k03 = std.textureLoad(field, d.vec2u(cx2, cym), 0)
        const k10 = std.textureLoad(field, d.vec2u(cxm, cy0), 0)
        const k11 = std.textureLoad(field, d.vec2u(cx0, cy0), 0)
        const k12 = std.textureLoad(field, d.vec2u(cx1, cy0), 0)
        const k13 = std.textureLoad(field, d.vec2u(cx2, cy0), 0)
        const k20 = std.textureLoad(field, d.vec2u(cxm, cy1), 0)
        const k21 = std.textureLoad(field, d.vec2u(cx0, cy1), 0)
        const k22 = std.textureLoad(field, d.vec2u(cx1, cy1), 0)
        const k23 = std.textureLoad(field, d.vec2u(cx2, cy1), 0)
        const k30 = std.textureLoad(field, d.vec2u(cxm, cy2), 0)
        const k31 = std.textureLoad(field, d.vec2u(cx0, cy2), 0)
        const k32 = std.textureLoad(field, d.vec2u(cx1, cy2), 0)
        const k33 = std.textureLoad(field, d.vec2u(cx2, cy2), 0)

        const rAcc =
            k00.r * wx0 * wy0 + k01.r * wx1 * wy0 + k02.r * wx2 * wy0 + k03.r * wx3 * wy0 +
            k10.r * wx0 * wy1 + k11.r * wx1 * wy1 + k12.r * wx2 * wy1 + k13.r * wx3 * wy1 +
            k20.r * wx0 * wy2 + k21.r * wx1 * wy2 + k22.r * wx2 * wy2 + k23.r * wx3 * wy2 +
            k30.r * wx0 * wy3 + k31.r * wx1 * wy3 + k32.r * wx2 * wy3 + k33.r * wx3 * wy3
        const aAcc =
            k00.a * wx0 * wy0 + k01.a * wx1 * wy0 + k02.a * wx2 * wy0 + k03.a * wx3 * wy0 +
            k10.a * wx0 * wy1 + k11.a * wx1 * wy1 + k12.a * wx2 * wy1 + k13.a * wx3 * wy1 +
            k20.a * wx0 * wy2 + k21.a * wx1 * wy2 + k22.a * wx2 * wy2 + k23.a * wx3 * wy2 +
            k30.a * wx0 * wy3 + k31.a * wx1 * wy3 + k32.a * wx2 * wy3 + k33.a * wx3 * wy3
        // Analytic B-spline patch derivative in texel space (chain-ruled to field-UV below).
        const drdpx =
            k00.r * dbx0 * by0 + k01.r * dbx1 * by0 + k02.r * dbx2 * by0 + k03.r * dbx3 * by0 +
            k10.r * dbx0 * by1 + k11.r * dbx1 * by1 + k12.r * dbx2 * by1 + k13.r * dbx3 * by1 +
            k20.r * dbx0 * by2 + k21.r * dbx1 * by2 + k22.r * dbx2 * by2 + k23.r * dbx3 * by2 +
            k30.r * dbx0 * by3 + k31.r * dbx1 * by3 + k32.r * dbx2 * by3 + k33.r * dbx3 * by3
        const drdpy =
            k00.r * bx0 * dby0 + k01.r * bx1 * dby0 + k02.r * bx2 * dby0 + k03.r * bx3 * dby0 +
            k10.r * bx0 * dby1 + k11.r * bx1 * dby1 + k12.r * bx2 * dby1 + k13.r * bx3 * dby1 +
            k20.r * bx0 * dby2 + k21.r * bx1 * dby2 + k22.r * bx2 * dby2 + k23.r * bx3 * dby2 +
            k30.r * bx0 * dby3 + k31.r * bx1 * dby3 + k32.r * bx2 * dby3 + k33.r * bx3 * dby3

        const rMin = std.min(std.min(k11.r, k12.r), std.min(k21.r, k22.r))
        const rMax = std.max(std.max(k11.r, k12.r), std.max(k21.r, k22.r))
        const aMin = std.min(std.min(k11.a, k12.a), std.min(k21.a, k22.a))
        const aMax = std.max(std.max(k11.a, k12.a), std.max(k21.a, k22.a))
        const aOut = std.clamp(aAcc, aMin, aMax)

        // SOFT neighbourhood clamp on `.r` (polynomial smooth-min/max — see the header comment).
        // The knee width scales with the local texel range: flat neighbourhoods → no effect.
        const kSoft = (rMax - rMin) * 0.3 + 0.000001
        const hCap = std.clamp((rMax - rAcc) / kSoft * 0.5 + 0.5, 0.0, 1.0)
        const rCap = std.mix(rMax, rAcc, hCap) - kSoft * hCap * (1.0 - hCap)
        const hFlr = std.clamp((rCap - rMin) / kSoft * 0.5 + 0.5, 0.0, 1.0)
        const rSoft = std.mix(rMin, rCap, hFlr) + kSoft * hFlr * (1.0 - hFlr)

        const rNear = rSoft + outsideDist
        const rCircle = std.length(d.vec2f(uv.x - 0.5, uv.y - 0.5)) - rBound
        const rOut = std.mix(rNear, std.max(rCircle, rNear), std.smoothstep(d.f32(0), 0.25, outsideDist))

        // Chain rule texels → field UV. Outside the domain the clamped texel coord stops moving
        // (∂px/∂u = 0), so the interior term is masked off; the border extension then contributes
        // exactly the outward unit direction of `outsideDist` (zero inside — numerator is 0).
        const insideX = std.select(d.f32(0), d.f32(1), std.abs(uv.x - uxc) < 0.0000001)
        const insideY = std.select(d.f32(0), d.f32(1), std.abs(uv.y - uyc) < 0.0000001)
        const invOut = 1.0 / std.max(outsideDist, 0.00001)
        const gU = drdpx * (activeResF / spanX) * insideX + (uv.x - uxc) * invOut
        const gV = drdpy * (activeResF / spanY) * insideY + (uv.y - uyc) * invOut

        return d.vec4f(rOut, gU, gV, aOut)
    }).$name('fieldSampleGradArg')

    // 4-tap bilinear (finite-difference gradient taps). Same domain map + far-field extension.
    const fieldSampleFastArg = tgpu.fn(
        [d.vec2f, d.texture2d(d.f32), d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
        d.vec4f,
    )((uv, field, originX, originY, spanX, spanY, activeResF, rBound) => {
        'use gpu'
        const uxc = std.clamp(uv.x, originX, originX + spanX)
        const uyc = std.clamp(uv.y, originY, originY + spanY)
        const outsideDist = std.length(d.vec2f(uv.x - uxc, uv.y - uyc))
        const px = (uxc - originX) / spanX * activeResF - 0.5
        const py = (uyc - originY) / spanY * activeResF - 0.5
        const x0 = std.floor(px)
        const y0 = std.floor(py)
        const fx = px - x0
        const fy = py - y0
        const maxT = activeResF - 1.0
        const cx0 = d.u32(std.clamp(x0, d.f32(0), maxT))
        const cx1 = d.u32(std.clamp(x0 + 1.0, d.f32(0), maxT))
        const cy0 = d.u32(std.clamp(y0, d.f32(0), maxT))
        const cy1 = d.u32(std.clamp(y0 + 1.0, d.f32(0), maxT))
        const t00 = std.textureLoad(field, d.vec2u(cx0, cy0), 0)
        const t10 = std.textureLoad(field, d.vec2u(cx1, cy0), 0)
        const t01 = std.textureLoad(field, d.vec2u(cx0, cy1), 0)
        const t11 = std.textureLoad(field, d.vec2u(cx1, cy1), 0)
        const bl = std.mix(std.mix(t00, t10, fx), std.mix(t01, t11, fx), fy)
        const rNear = bl.r + outsideDist
        const rCircle = std.length(d.vec2f(uv.x - 0.5, uv.y - 0.5)) - rBound
        const rOut = std.mix(rNear, std.max(rCircle, rNear), std.smoothstep(d.f32(0), 0.25, outsideDist))
        return d.vec4f(rOut, bl.g, bl.b, bl.a)
    }).$name('fieldSampleFastArg')

    // Unfiltered: the ONE texel under the UV (nearest, domain-clamped). For DISCRETE payload
    // channels a march packs into the field (ids, cell indices, quantised masks) — a bilinear
    // blend of two ids is neither id. `.r` still carries the far-field extension so a consumer
    // that reads only this tap keeps a monotone outside distance.
    const fieldSampleTexelArg = tgpu.fn(
        [d.vec2f, d.texture2d(d.f32), d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
        d.vec4f,
    )((uv, field, originX, originY, spanX, spanY, activeResF, rBound) => {
        'use gpu'
        const uxc = std.clamp(uv.x, originX, originX + spanX)
        const uyc = std.clamp(uv.y, originY, originY + spanY)
        const outsideDist = std.length(d.vec2f(uv.x - uxc, uv.y - uyc))
        const maxT = activeResF - 1.0
        const px = std.clamp(std.floor((uxc - originX) / spanX * activeResF), d.f32(0), maxT)
        const py = std.clamp(std.floor((uyc - originY) / spanY * activeResF), d.f32(0), maxT)
        const t = std.textureLoad(field, d.vec2u(d.u32(px), d.u32(py)), 0)
        const rNear = t.r + outsideDist
        const rCircle = std.length(d.vec2f(uv.x - 0.5, uv.y - 0.5)) - rBound
        const rOut = std.mix(rNear, std.max(rCircle, rNear), std.smoothstep(d.f32(0), 0.25, outsideDist))
        return d.vec4f(rOut, t.g, t.b, t.a)
    }).$name('fieldSampleTexelArg')

    // texelSpan = max(spanX, spanY) / activeRes (an effect's fast-sampler early-out threshold).

    // The marched DEPTH's analytic gradient (∂depth/∂u, ∂depth/∂v in field UV): the cubic B-spline
    // derivative of `.a` over the same 4×4 texels `fieldSampleGradArg` reads (cache-warm when both
    // are taken at one point). `.a` is reconstructed with a per-texel HARD clamp (C0 at every texel
    // boundary), so screen derivatives of it jump at the texel lattice — zoomed in, a material that
    // builds a front-face normal from them shows a dot per texel. The B-spline derivative is C1 and
    // ring-free. NOTE: missed rays store a far depth (`tEnd + 0.4`), so within ~2 texels of the
    // silhouette this mixes in the background — consumers fade it out there (`fieldTexelSpan`).
    const fieldDepthSlopeArg = tgpu.fn(
        [d.vec2f, d.texture2d(d.f32), d.f32, d.f32, d.f32, d.f32, d.f32],
        d.vec2f,
    )((uv, field, originX, originY, spanX, spanY, activeResF) => {
        'use gpu'
        const uxc = std.clamp(uv.x, originX, originX + spanX)
        const uyc = std.clamp(uv.y, originY, originY + spanY)
        const px = (uxc - originX) / spanX * activeResF - 0.5
        const py = (uyc - originY) / spanY * activeResF - 0.5
        const x0 = std.floor(px)
        const y0 = std.floor(py)
        const fx = px - x0
        const fy = py - y0
        const tx2 = fx * fx
        const tx3 = tx2 * fx
        const ty2 = fy * fy
        const ty3 = ty2 * fy
        const oxf = 1.0 - fx
        const bx0 = oxf * oxf * oxf * 0.16666667
        const bx1 = tx3 * 0.5 - tx2 + 0.66666667
        const bx2 = tx3 * -0.5 + tx2 * 0.5 + fx * 0.5 + 0.16666667
        const bx3 = tx3 * 0.16666667
        const dbx0 = oxf * oxf * -0.5
        const dbx1 = tx2 * 1.5 - fx * 2.0
        const dbx2 = tx2 * -1.5 + fx + 0.5
        const dbx3 = tx2 * 0.5
        const oyf = 1.0 - fy
        const by0 = oyf * oyf * oyf * 0.16666667
        const by1 = ty3 * 0.5 - ty2 + 0.66666667
        const by2 = ty3 * -0.5 + ty2 * 0.5 + fy * 0.5 + 0.16666667
        const by3 = ty3 * 0.16666667
        const dby0 = oyf * oyf * -0.5
        const dby1 = ty2 * 1.5 - fy * 2.0
        const dby2 = ty2 * -1.5 + fy + 0.5
        const dby3 = ty2 * 0.5

        const maxT = activeResF - 1.0
        const cxm = d.u32(std.clamp(x0 - 1.0, d.f32(0), maxT))
        const cx0 = d.u32(std.clamp(x0, d.f32(0), maxT))
        const cx1 = d.u32(std.clamp(x0 + 1.0, d.f32(0), maxT))
        const cx2 = d.u32(std.clamp(x0 + 2.0, d.f32(0), maxT))
        const cym = d.u32(std.clamp(y0 - 1.0, d.f32(0), maxT))
        const cy0 = d.u32(std.clamp(y0, d.f32(0), maxT))
        const cy1 = d.u32(std.clamp(y0 + 1.0, d.f32(0), maxT))
        const cy2 = d.u32(std.clamp(y0 + 2.0, d.f32(0), maxT))

        const a00 = std.textureLoad(field, d.vec2u(cxm, cym), 0).a
        const a01 = std.textureLoad(field, d.vec2u(cx0, cym), 0).a
        const a02 = std.textureLoad(field, d.vec2u(cx1, cym), 0).a
        const a03 = std.textureLoad(field, d.vec2u(cx2, cym), 0).a
        const a10 = std.textureLoad(field, d.vec2u(cxm, cy0), 0).a
        const a11 = std.textureLoad(field, d.vec2u(cx0, cy0), 0).a
        const a12 = std.textureLoad(field, d.vec2u(cx1, cy0), 0).a
        const a13 = std.textureLoad(field, d.vec2u(cx2, cy0), 0).a
        const a20 = std.textureLoad(field, d.vec2u(cxm, cy1), 0).a
        const a21 = std.textureLoad(field, d.vec2u(cx0, cy1), 0).a
        const a22 = std.textureLoad(field, d.vec2u(cx1, cy1), 0).a
        const a23 = std.textureLoad(field, d.vec2u(cx2, cy1), 0).a
        const a30 = std.textureLoad(field, d.vec2u(cxm, cy2), 0).a
        const a31 = std.textureLoad(field, d.vec2u(cx0, cy2), 0).a
        const a32 = std.textureLoad(field, d.vec2u(cx1, cy2), 0).a
        const a33 = std.textureLoad(field, d.vec2u(cx2, cy2), 0).a

        const dadpx =
            a00 * dbx0 * by0 + a01 * dbx1 * by0 + a02 * dbx2 * by0 + a03 * dbx3 * by0 +
            a10 * dbx0 * by1 + a11 * dbx1 * by1 + a12 * dbx2 * by1 + a13 * dbx3 * by1 +
            a20 * dbx0 * by2 + a21 * dbx1 * by2 + a22 * dbx2 * by2 + a23 * dbx3 * by2 +
            a30 * dbx0 * by3 + a31 * dbx1 * by3 + a32 * dbx2 * by3 + a33 * dbx3 * by3
        const dadpy =
            a00 * bx0 * dby0 + a01 * bx1 * dby0 + a02 * bx2 * dby0 + a03 * bx3 * dby0 +
            a10 * bx0 * dby1 + a11 * bx1 * dby1 + a12 * bx2 * dby1 + a13 * bx3 * dby1 +
            a20 * bx0 * dby2 + a21 * bx1 * dby2 + a22 * bx2 * dby2 + a23 * bx3 * dby2 +
            a30 * bx0 * dby3 + a31 * bx1 * dby3 + a32 * bx2 * dby3 + a33 * bx3 * dby3

        // Outside the domain the clamped texel coord stops moving: no depth slope there.
        const insideX = std.select(d.f32(0), d.f32(1), std.abs(uv.x - uxc) < 0.0000001)
        const insideY = std.select(d.f32(0), d.f32(1), std.abs(uv.y - uyc) < 0.0000001)
        return d.vec2f(dadpx * (activeResF / spanX) * insideX, dadpy * (activeResF / spanY) * insideY)
    }).$name('fieldDepthSlopeArg')

    const texelSpanFn = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((spanX, spanY, activeResF) => {
        'use gpu'
        return std.max(spanX, spanY) / activeResF
    }).$name('fieldTexelSpan')

    return {fieldSampleArg, fieldSampleGradArg, fieldSampleFastArg, fieldSampleTexelArg, fieldDepthSlopeArg, texelSpanFn}
}

/**
 * The six SampleParams a shape-effect shader must declare as `extraFields` so the field SAMPLER
 * (fn-arg form) can read the compute-written aspect-fit domain in its FRAGMENT. `initial` values are
 * inert (span 1 / full-res); `createVolumetricFieldComputeNode` writes the live values each frame
 * via `params.setExtraField`. CONSUMER WIRES: spread these into the shader definition's
 * `extraFields`, then pass `params.uniforms.<name>` into `buildVolumetricFieldSampler`.
 */
export const VOLUMETRIC_FIELD_EXTRA_FIELDS: Record<string, {schema: typeof d.f32; initial: number}> = {
    _vfOriginX: {schema: d.f32, initial: 0},
    _vfOriginY: {schema: d.f32, initial: 0},
    _vfSpanX: {schema: d.f32, initial: 1},
    _vfSpanY: {schema: d.f32, initial: 1},
    _vfActiveRes: {schema: d.f32, initial: 1},
    _vfRBound: {schema: d.f32, initial: 0.6},
}

/** The six SampleParams accessor Exprs the consumer's fragment reads from `params.uniforms._vf*`. */
export interface VolumetricFieldSampleParams {
    originX: Expr
    originY: Expr
    spanX: Expr
    spanY: Expr
    activeResF: Expr
    rBound: Expr
}

/** The field-sampler Exprs a shape-effect fragment consumes. */
export interface VolumetricFieldSamplers {
    /** Bicubic sampler — crisp silhouette/normal edges. `(uv) => vec4(field, gPattern, bPattern, depth)`. */
    volumetricFieldSampler: (uv: Expr) => Expr
    /** Bicubic value + ANALYTIC gradient in one 16-load tap: `(uv) => vec4(field, ∂field/∂u,
     *  ∂field/∂v, depth)`. Replaces a 3-tap finite-difference stencil (48 loads → 16) with a
     *  C1-smooth, texel-localised gradient — feed an effect's `bakedGradients` path. `.g/.b` are
     *  gradients, NOT pattern coords. */
    volumetricFieldSamplerGrad: (uv: Expr) => Expr
    /** 4-tap bilinear sampler (finite-difference gradient taps). */
    volumetricFieldSamplerFast: (uv: Expr) => Expr
    /** Unfiltered nearest-texel read — discrete payload channels (ids, packed cells). */
    volumetricFieldSamplerTexel: (uv: Expr) => Expr
    /** `max(spanX, spanY) / activeRes` — the fast-sampler early-out threshold. */
    volumetricFieldTexelSpan: Expr
    /** The marched depth's analytic gradient, `(uv) => vec2(∂depth/∂u, ∂depth/∂v)` (C1 B-spline;
     *  unreliable within ~2 texels of the silhouette — fade by `volumetricFieldTexelSpan`). */
    volumetricFieldDepthSlope: (uv: Expr) => Expr
}

/**
 * Build the field samplers for a shape-effect FRAGMENT. `field` is the compute-written field
 * KitTexture (`params.computeOutputs.volumetricFieldTexture` — a `registerComputeTexture` binding);
 * `p` are the SampleParams accessor Exprs (`params.uniforms._vf*`). Returns the three sampler Exprs
 * the effect samples. See `createVolumetricFieldComputeNode` for the compute side that fills them.
 */
export function buildVolumetricFieldSampler(field: KitTexture, p: VolumetricFieldSampleParams): VolumetricFieldSamplers {
    const {fieldSampleArg, fieldSampleGradArg, fieldSampleFastArg, fieldSampleTexelArg, fieldDepthSlopeArg, texelSpanFn} = buildFieldSampleGraphArgs()
    const acc = field.accessor()
    const argsFor = (uv: Expr): Expr[] => [uv, acc, p.originX, p.originY, p.spanX, p.spanY, p.activeResF, p.rBound]
    return {
        volumetricFieldSampler: (uv: Expr) => call(fieldSampleArg, 'vfSample', argsFor(uv)),
        volumetricFieldSamplerGrad: (uv: Expr) => call(fieldSampleGradArg, 'vfSampleGrad', argsFor(uv)),
        volumetricFieldSamplerFast: (uv: Expr) => call(fieldSampleFastArg, 'vfSampleFast', argsFor(uv)),
        volumetricFieldSamplerTexel: (uv: Expr) => call(fieldSampleTexelArg, 'vfSampleTexel', argsFor(uv)),
        volumetricFieldTexelSpan: call(texelSpanFn, 'vfTexelSpan', [p.spanX, p.spanY, p.activeResF]),
        volumetricFieldDepthSlope: (uv: Expr) => call(fieldDepthSlopeArg, 'vfDepthSlope', [uv, acc, p.originX, p.originY, p.spanX, p.spanY, p.activeResF]),
    }
}

/** How a consumer wants the SDF-field gradient delivered on each of the three shape paths. */
export interface ShapeFieldSamplerOptions {
    /** Surface-pattern coords baked into the ANALYTIC sampler (`.g/.b`). Match the shader's
     *  `createVolumetricFieldComputeNode` patternMode so both paths agree. */
    patternMode?: SurfacePatternMode
    /**
     * The second sampler the consumer takes its finite-difference neighbour taps with:
     *   - `'fast'` (default) — the cheap 4-tap bilinear sampler on the volumetric path, the same
     *     sampler as `sampler` on the flat paths. The metals/Frost/Water contract.
     *   - `'same'` — `gradSampler === sampler` everywhere (a consumer that takes no extra taps, or
     *     wants the bicubic tap for all three).
     */
    gradSampler?: 'fast' | 'same'
    /**
     * Which paths carry gradients baked into the CENTRE tap's `.g/.b` (so the consumer skips its
     * finite-difference taps — see the returned `bakedGradients`):
     *   - `'none'` (default) — nobody does; the consumer takes its own taps.
     *   - `'volumetric'` — only the compute path (`volumetricFieldSamplerGrad`). ThinFilm.
     *   - `'all'` — the compute path AND the flat-SVG path (`createSvgSdfSamplerWithGradients`);
     *     the analytic path still reports `bakedGradients: false`. Glass.
     */
    bakedGradients?: 'none' | 'volumetric' | 'all'
}

/** The resolved shape-field sampling surface a shape-effect fragment builds its taps from. */
export interface ShapeFieldSamplerResult {
    /** Centre-tap sampler: `(uv) => vec4(field/−chord, patternU|gradU, patternV|gradV, depth)`. */
    sampler: (uv: Expr) => Expr
    /** Neighbour-tap sampler (see `ShapeFieldSamplerOptions.gradSampler`). */
    gradSampler: (uv: Expr) => Expr
    /** Unfiltered nearest-texel tap for discrete payload channels (the volumetric path); the flat
     *  paths (analytic / SVG bilinear) alias `sampler`. */
    texelSampler: (uv: Expr) => Expr
    /** The active shape is a 3D / extruded field (`.r` is −chord/2, not a 2D distance). */
    volumetric: boolean
    /** The centre tap carries gradients in `.g/.b` — the consumer must skip its own taps. */
    bakedGradients: boolean
    /** The field as WGSL you can call from inside another fn (see `PortableFieldSampler`). */
    portable: PortableFieldSampler
    /** Volumetric path only: the marched depth's analytic gradient `(uv) => vec2` (see
     *  `volumetricFieldDepthSlope`). Undefined on the flat paths, where depth is constant. */
    depthSlope?: (uv: Expr) => Expr
    /** Volumetric path only: one field texel, in field UV. */
    texelSpan?: Expr
}

/**
 * The shape field in a form a separate WGSL fn can sample at any point (a custom body's
 * `shapeAt(p)`): the main-scope values the sample reads, passed in as arguments, plus the WGSL
 * that samples given their parameter names. The samplers above are Exprs over main-scope
 * accessors (`tex.$.<key>` raw texture refs the pass layout scans for, driver-override uniform
 * locals), which a linked fn cannot reference, so every input crosses as an argument. Uses the
 * cheap neighbour-quality tap on the volumetric path (a caller typically takes many).
 */
export interface PortableFieldSampler {
    /** Names this variant (shape path + type) for fn caching. */
    key: string
    /** The values passed ahead of the point, with their WGSL parameter types. */
    args: {value: Expr; type: 'f32' | 'texture_2d<f32>' | 'sampler'}[]
    /** External fns the sample WGSL calls, by the name it calls them. */
    uses: Record<string, unknown>
    /** WGSL sampling the field (`vec4f`, `.x` = signed distance / −chord/2) at `p`, given the args' parameter names. */
    sample: (argNames: string[], p: string) => string
}

// One instance per sampler fn, so every portable sampler links the same fn (no duplicate WGSL).
let portableVolumetricFn: unknown
const portableAnalyticFns = new Map<string, unknown>()

/**
 * Resolve the SDF field sampler for a shape-effect fragment: the volumetric → flat-SVG → analytic
 * routing every shader in the family (Glass, LiquidMetal, Chrome, Frost, Heatmap, …) used to
 * hand-write. Call it from `fragment` with the same `patternMode` the shader's
 * `createVolumetricFieldComputeNode` uses.
 *
 *   - **volumetric** — `params.computeOutputs.volumetricFieldTexture` exists, so the compute pass
 *     pre-marched a 3D/extruded field; samplers come from `buildVolumetricFieldSampler` against
 *     this node's `_vf*` SampleParams accessors.
 *   - **flat SVG** — a `shapeSdfUrl` prop; the 2D SDF data texture (`kit/sdf`).
 *   - **analytic** — the baked 2D SDF for the resolved shape type, with its animated sub-props
 *     driven per frame by `driveAnalyticSubProps`.
 *
 * A shader with no `shapeSdfUrl`/`shapeType` props still works (both read as `''` → analytic).
 */
export function resolveShapeFieldSampler(
    params: GpuFragmentParams,
    options: ShapeFieldSamplerOptions = {},
): ShapeFieldSamplerResult {
    const {patternMode = 'none', gradSampler: gradMode = 'fast', bakedGradients = 'none'} = options
    const {computeOutputs, uniforms, propValues, getCpuValue} = params

    const fieldTex = computeOutputs?.volumetricFieldTexture as KitTexture | undefined
    if (fieldTex) {
        const s = buildVolumetricFieldSampler(fieldTex, {
            originX: uniforms._vfOriginX, originY: uniforms._vfOriginY,
            spanX: uniforms._vfSpanX, spanY: uniforms._vfSpanY,
            activeResF: uniforms._vfActiveRes, rBound: uniforms._vfRBound,
        })
        const baked = bakedGradients !== 'none'
        const sampler = baked ? s.volumetricFieldSamplerGrad : s.volumetricFieldSampler
        portableVolumetricFn ??= buildFieldSampleGraphArgs().fieldSampleFastArg
        const vfArgs = [uniforms._vfOriginX, uniforms._vfOriginY, uniforms._vfSpanX, uniforms._vfSpanY, uniforms._vfActiveRes, uniforms._vfRBound]
        return {
            sampler,
            gradSampler: gradMode === 'same' ? sampler : s.volumetricFieldSamplerFast,
            texelSampler: s.volumetricFieldSamplerTexel,
            volumetric: true,
            bakedGradients: baked,
            depthSlope: s.volumetricFieldDepthSlope,
            texelSpan: s.volumetricFieldTexelSpan,
            portable: {
                key: 'volumetric',
                args: [
                    {value: fieldTex.accessor(), type: 'texture_2d<f32>'},
                    ...vfArgs.map((value) => ({value, type: 'f32' as const})),
                ],
                uses: {vfSampleFast: portableVolumetricFn},
                sample: (a, p) => `vfSampleFast(${p}, ${a.join(', ')})`,
            },
        }
    }

    const shapeSdfUrl = (propValues.shapeSdfUrl as string) || ''
    const shapeType = (propValues.shapeType as string) || ''

    if (shapeSdfUrl) {
        const bakedSvg = bakedGradients === 'all'
        const sampler = bakedSvg
            ? createSvgSdfSamplerWithGradients(params, shapeSdfUrl)
            : createSvgSdfSampler(params, shapeSdfUrl)
        return {
            sampler, gradSampler: sampler, texelSampler: sampler, volumetric: isVolumetricShapeType(shapeType), bakedGradients: bakedSvg,
            portable: {
                key: 'svg',
                args: [
                    {value: sampler.kitTexture.accessor(), type: 'texture_2d<f32>'},
                    {value: expr('samp.$.linearClamp'), type: 'sampler'},
                ],
                uses: {},
                sample: (a, p) => `textureSampleLevel(${a[0]}, ${a[1]}, ${p}, 0.0)`,
            },
        }
    }

    const st = resolveShapeType(shapeType, getCpuValue('shape'))
    const sub = driveAnalyticSubProps(params, () => getCpuValue('shape'))
    const sampler = createAnalyticSdfSampler(st, sub, patternMode)
    let analyticFn = portableAnalyticFns.get(st)
    if (!analyticFn) portableAnalyticFns.set(st, (analyticFn = buildAnalyticSdfFn(st)))
    const subArgs = [sub.radius, sub.sides, sub.rounding, sub.innerRatio, sub.rotation, sub.height, sub.offset, sub.aperture]
    return {
        sampler, gradSampler: sampler, texelSampler: sampler, volumetric: is3dShapeType(st), bakedGradients: false,
        portable: {
            key: `analytic:${st}`,
            args: subArgs.map((value) => ({value: value ?? floatE(0), type: 'f32' as const})),
            uses: {analyticSdf: analyticFn},
            sample: (a, p) => `analyticSdf(${p}, ${a.join(', ')})`,
        },
    }
}

// ─── CPU shape-JSON → MarchParams resolvers (analytic + SVG-extrusion 3D SDF setup) ─

/** A per-frame shape-effect volumetric setup: a baked `sdfFn` (reads its layout's MarchParams) plus
 *  the CPU resolver that writes the March uniform each frame from the shape config (auto-animate /
 *  mouse sub-props). The parent (`createVolumetricFieldComputeNode`) threads `layout` + `sdfFn`
 *  into the march/kernel/compute and drives `update` / `setActiveRes` / `getStateKey`. */
export interface VolumetricFieldSetup {
    /** The march bind-group layout the `sdfFn` reads its params off (analytic or SVG variant). */
    layout: VolumetricFieldLayout
    /** The MarchParams uniform `update` writes each frame (shared with the compute bind group). */
    marchParamsBuffer: MarchParamsUniform
    /** The shape SDF baked against `layout.$.params` — fed to `buildRaymarchedFieldFn`. */
    sdfFn: BakedSdf
    /** SVG-3D only: the 2D SDF DATA texture bound into the compute kernel (`sdfSource`). */
    sdfSourceTexture?: unknown
    /** Resolve the shape config + write the March uniform (advances the auto-animate clock). */
    update: (frameParams?: ShapeFrameParams) => void
    /** Set the active linear resolution + re-write the uniform (no time advance). */
    setActiveRes: (res: number) => void
    /** Dirty-key: the field re-marches only when this changes. */
    getStateKey: () => string
    /** The aspect-fit domain footprint the parent feeds `resolveActiveFieldRes` / SampleParams. */
    getFootprint: () => {spanX: number; spanY: number; originX: number; originY: number; rBound: number}
    /** The CPU-resolved shape rotation (the `rot` written to MarchParams) — for a consumer that
     *  composes its own view→shape transform on the CPU (the voxel pre-march's camera + light). */
    getRotation: () => {cx: number; sx: number; cy: number; sy: number; cz: number; sz: number}
    /** Dirty-key of the shape's GEOMETRY alone (state key minus rotation/domain) — for a consumer
     *  caching something in shape-local space (the voxel occupancy grid) that rotation can't change. */
    getGeometryKey?: () => string
}

/** Optional knobs shared by the volumetric setups. */
export interface VolumetricSetupOptions {
    /** Extra margin (field UV) added to the bounding radius AND the aspect-fit domain each update —
     *  for a consumer whose geometry puffs the solid outward (a voxel half-diagonal). Read live. */
    extraPad?: () => number
}

/** The per-frame params a setup's `update` reads (deltaTime → auto clock, pointer → mouse drivers). */
export interface ShapeFrameParams {
    deltaTime?: number
    pointer?: {x: number; y: number}
}

const DEG = Math.PI / 180

function parseShapeConfig(raw: unknown): Record<string, unknown> {
    if (raw && typeof raw === 'object') return raw as Record<string, unknown>
    if (typeof raw === 'string') {
        try {
            return JSON.parse(raw) as Record<string, unknown>
        } catch {
            return {}
        }
    }
    return {}
}

/**
 * Compile-time-selected analytic 3D SDF setup. Allocates the march layout + a MarchParams uniform,
 * bakes `sdfFn` for `shapeType` (one primitive, compile-time
 * branch, reading its size params off `layout.$.params`), and returns a per-frame `update` that
 * resolves the shape JSON (incl. auto-animate / mouse sub-props) and writes the uniform.
 */
export function createAnalytic3dSdfSetup(
    root: TgpuRoot,
    shapeType: string,
    initialConfig: Record<string, unknown>,
    getShapeConfig: () => unknown,
    opts: VolumetricSetupOptions = {},
): VolumetricFieldSetup {
    const defaults = SHAPE3D_DEFAULTS[shapeType as Shape3DType] ?? SHAPE3D_DEFAULTS.sphere3D
    const extraPad = opts.extraPad ?? (() => 0)
    const {layout} = makeVolumetricFieldLayout()
    const marchParamsBuffer = root.createUniform(MarchParams) as unknown as MarchParamsUniform

    // sdfFn: compile-time branch (shapeType is a JS string → the if-chain runs as JS at graph-build
    // time, emitting ONLY the selected primitive). Reads size params off layout.$.params.
    // The chain is `if / else if / … / else` (not bare early-returns): with sphere3D left as an
    // unguarded tail `return`, the transpiler emitted the taken branch AND the sphere tail, so every
    // non-sphere shape resolved to WGSL with a dead `return sdSphere(…)` after the branch → a "code is
    // unreachable" compile warning. Folding sphere into the trailing `else` emits exactly one reachable
    // `return` for any shapeType (sphere itself resolves to the else — no warning either way).
    const sdfFn: BakedSdf = tgpu.fn([d.vec3f], d.f32)((p) => {
        'use gpu'
        const pp = layout.$.params
        if (shapeType === 'cube3D') return sdRoundBox(p, pp.pA, pp.pB, pp.pC, pp.pD)
        else if (shapeType === 'torus3D') return sdTorus(p, pp.pA, pp.pB)
        else if (shapeType === 'octahedron3D') return sdOctahedron(p, pp.pA)
        else if (shapeType === 'cylinder3D') return sdRoundCylinder(p, pp.pA, pp.pB, pp.pD)
        else if (shapeType === 'capsule3D') return sdCapsule(p, pp.pA, pp.pB)
        else if (shapeType === 'cone3D') return sdCappedCone(p, pp.pA, pp.pC, pp.pB)
        else if (shapeType === 'pyramid3D') return sdPyramid(p, pp.pA, pp.pB)
        else if (shapeType === 'prism3D') return sdHexPrism(p, pp.pA, pp.pB)
        else if (shapeType === 'ellipsoid3D') return sdEllipsoid(p, pp.pA, pp.pB, pp.pC)
        else if (shapeType === 'diamond3D') return sdBicone(p, pp.pA, pp.pB)
        else if (shapeType === 'link3D') return sdLink(p, pp.pB, pp.pA, pp.pC)
        else if (shapeType === 'gem3D') return sdGem(p, pp.pA, pp.pB, pp.pC)
        else if (shapeType === 'helix3D') return sdHelix(p, pp.pA, pp.pB, pp.pC, d.f32(HELIX_TURNS))
        else if (shapeType === 'metaballs3D') return sdMetaballs(p, pp.mb0, pp.mb1, pp.mb2, pp.mb3, pp.mb4, pp.mb5, pp.mb6, pp.mb7, pp.pA, pp.pD)
        else if (shapeType === 'dodecahedron3D') return sdDodecahedron(p, pp.pA)
        else if (shapeType === 'hemisphere3D') return sdCutSphere(p, pp.pA, pp.pB)
        // The animated shapes reuse the mb vec3 slots as generic CPU-animated parameter carriers —
        // per-shape slot meanings are centralized in the MarchParams SLOT MAP above; the `update`
        // switch below is the packing side.
        else if (shapeType === 'ribbon3D') return sdRibbon(p, pp.pA, pp.pB, pp.pC, pp.mb0.x, pp.mb0.y, pp.mb0.z, pp.mb1.x, pp.mb1.y, pp.mb1.z, pp.mb2.x)
        else if (shapeType === 'blob3D') return sdWobbleBlob(p, pp.pA, pp.mb0.x, pp.mb0.y, pp.mb0.z, pp.mb1.x, pp.mb2.x)
        else if (shapeType === 'gyroscope3D') return sdGyroscope(p, pp.pA, pp.pB, pp.pD, pp.mb3.x, pp.pC, pp.mb0.x, pp.mb0.y, pp.mb1.x, pp.mb1.y, pp.mb2.x, pp.mb2.y)
        else return sdSphere(p, pp.pA)
    })

    // CPU state (mirrors the MarchParams struct fields).
    let elapsed = 0
    let lastShapeJson = ''
    let lastCfg: Record<string, unknown> = initialConfig
    const springs = new Map<string, {current: number; velocity: number}>()
    let pA = 0.35, pB = 0.3, pC = 0.3, pD = 0
    let rBound = 0.6
    let cx = 1, sx = 0, cy = 1, sy = 0, cz = 1, sz = 0
    let spanX = 1, spanY = 1, originX = 0, originY = 0
    let activeRes = resolveVolumetricFieldRes()
    const mb: [number, number, number][] = Array.from({length: MAX_METABALLS}, () => [0, 0, 0])

    const writeMarch = () => {
        marchParamsBuffer.write({
            rot: {cx, sx, cy, sy, cz, sz},
            rBound,
            spanX, spanY, originX, originY,
            activeRes,
            pA, pB, pC, pD,
            mb0: d.vec3f(mb[0][0], mb[0][1], mb[0][2]),
            mb1: d.vec3f(mb[1][0], mb[1][1], mb[1][2]),
            mb2: d.vec3f(mb[2][0], mb[2][1], mb[2][2]),
            mb3: d.vec3f(mb[3][0], mb[3][1], mb[3][2]),
            mb4: d.vec3f(mb[4][0], mb[4][1], mb[4][2]),
            mb5: d.vec3f(mb[5][0], mb[5][1], mb[5][2]),
            mb6: d.vec3f(mb[6][0], mb[6][1], mb[6][2]),
            mb7: d.vec3f(mb[7][0], mb[7][1], mb[7][2]),
        })
    }

    const update = (frameParams?: ShapeFrameParams) => {
        const deltaTime = frameParams?.deltaTime ?? 0
        elapsed += deltaTime
        const raw = getShapeConfig()
        if (raw && typeof raw === 'object') {
            lastCfg = raw as Record<string, unknown>
        } else if (typeof raw === 'string') {
            if (raw !== lastShapeJson) {
                lastShapeJson = raw
                try { lastCfg = JSON.parse(raw) } catch { /* keep previous */ }
            }
        }
        const frame: ShapeSubPropFrame = {
            elapsed, deltaTime,
            pointerX: frameParams?.pointer?.x ?? 0.5,
            pointerY: frameParams?.pointer?.y ?? 0.5,
            springs,
        }
        const sub = (k: string) => resolveShapeSubProp(lastCfg[k], defaults[k] ?? 0, frame, k)

        pB = 0.3; pC = 0.3; pD = 0
        for (let i = 0; i < mb.length; i++) mb[i] = [0, 0, 0]
        switch (shapeType) {
            case 'cube3D':
                pA = sub('sizeX'); pB = sub('sizeY'); pC = sub('sizeZ'); pD = sub('rounding'); break
            case 'torus3D':
                pA = sub('radius'); pB = sub('tube'); break
            case 'cylinder3D':
                pA = sub('radius'); pB = sub('height'); pD = sub('rounding'); break
            case 'capsule3D':
                pA = sub('radius'); pB = sub('height'); break
            case 'cone3D':
                pA = sub('radius'); pB = sub('height'); pC = sub('topRadius'); break
            case 'pyramid3D':
                pA = sub('size'); pB = sub('height'); break
            case 'prism3D':
            case 'diamond3D':
                pA = sub('radius'); pB = sub('height'); break
            case 'ellipsoid3D':
                pA = sub('radiusX'); pB = sub('radiusY'); pC = sub('radiusZ'); break
            case 'link3D':
                pA = sub('radius'); pB = sub('length'); pC = sub('tube'); break
            case 'gem3D':
                pA = sub('radius'); pB = sub('height'); pC = sub('facets'); break
            case 'helix3D':
                pA = sub('radius'); pB = sub('tube'); pC = sub('pitch'); break
            case 'metaballs3D': {
                pA = sub('ballRadius'); pB = sub('spread'); pD = sub('blend')
                // Ball count is CPU-resolved: active balls spread evenly around the ring (count 4
                // reproduces the original i·π/2 phasing exactly); the rest park far outside the
                // bounding sphere where smin degenerates to min (see sdMetaballs).
                const count = Math.min(MAX_METABALLS, Math.max(1, Math.round(sub('balls'))))
                const sp = pB, tt = elapsed * sub('speed') * 0.6
                for (let i = 0; i < mb.length; i++) {
                    if (i >= count) { mb[i] = [0, 99, 0]; continue }
                    const phase = i * (TWO_PI / count)
                    const ang = phase + tt
                    const wob = 0.75 + 0.25 * Math.sin(tt * 1.3 + i * 1.7)
                    mb[i] = [
                        sp * Math.cos(ang) * wob,
                        sp * 0.6 * Math.sin(tt * 0.9 + phase * 1.5),
                        sp * Math.sin(ang) * wob,
                    ]
                }
                break
            }
            case 'hemisphere3D':
                pA = sub('radius'); pB = sub('cut'); break
            case 'ribbon3D': {
                // Simulated cloth: two decorrelated travelling waves carry the centreline, the
                // amplitude breathes ±20%, and the twist phase drifts — all CPU state in mb0/mb1.
                // mb2.x is the Lipschitz factor for the wave slope + twist shear.
                pA = sub('width'); pB = sub('thickness'); pC = sub('length')
                const tt = elapsed * sub('speed')
                const amp = sub('wave') * (0.8 + 0.2 * Math.sin(tt * 0.9))
                const wf = sub('waveFrequency')
                const twRate = sub('twist') * DEG / Math.max(pC * 2, 0.01)
                mb[0] = [amp, wf, tt * 1.1]
                mb[1] = [tt * 0.7 + 2.1, twRate, tt * 0.45]
                const slope = Math.abs(amp) * wf + Math.abs(twRate) * Math.hypot(pA, pB)
                mb[2] = [1 / Math.sqrt(1 + slope * slope), 0, 0]
                break
            }
            case 'blob3D': {
                pA = sub('radius')
                const tt = elapsed * sub('speed')
                const amp = sub('wobble') * (0.75 + 0.25 * Math.sin(tt * 0.83))
                const wb = sub('wobbliness')
                mb[0] = [amp, wb, tt * 1.2]
                mb[1] = [tt * 0.77 + 1.3, 0, 0]
                mb[2] = [1 / (1 + Math.abs(amp) * wb * 1.3), 0, 0]
                break
            }
            case 'gyroscope3D': {
                pA = sub('radius'); pB = sub('width'); pC = sub('core'); pD = sub('thickness')
                // Bevel radius clamped to half the smaller cross-section extent (at max the
                // band degenerates to a tube — still valid).
                const rnd = Math.max(0, Math.min(sub('rounding'), Math.min(pB, pD) * 0.5))
                const tt = elapsed * sub('speed')
                const th1 = tt * 0.9
                const th2 = tt * 0.63 + 1.1
                const th3 = tt * 1.21 + 2.3
                mb[0] = [Math.cos(th1), Math.sin(th1), 0]
                mb[1] = [Math.cos(th2), Math.sin(th2), 0]
                mb[2] = [Math.cos(th3), Math.sin(th3), 0]
                mb[3] = [rnd, 0, 0]
                break
            }
            case 'sphere3D':
            case 'octahedron3D':
            case 'dodecahedron3D':
            default:
                pA = sub('radius'); break
        }
        rBound = shape3dBoundingRadius({...lastCfg, type: shapeType}) + 0.02 + extraPad()
        const rx = sub('rotX') * DEG, ry = sub('rotY') * DEG, rz = sub('rotZ') * DEG
        cx = Math.cos(rx); sx = Math.sin(rx)
        cy = Math.cos(ry); sy = Math.sin(ry)
        cz = Math.cos(rz); sz = Math.sin(rz)

        // Aspect-fit domain (isotropic for analytic primitives): rPad = rBound + 0.05.
        const rPad = rBound + 0.05
        spanX = rPad * 2; spanY = rPad * 2
        originX = 0.5 - rPad; originY = 0.5 - rPad
        writeMarch()
    }
    update()

    const setActiveRes = (res: number) => {
        if (res === activeRes) return
        activeRes = res
        writeMarch()
    }

    const getGeometryKey = () =>
        `${pA},${pB},${pC},${pD},${rBound},` + mb.map((m) => `${m[0]},${m[1]},${m[2]}`).join(',')
    const getStateKey = () => `${getGeometryKey()},${cx},${sx},${cy},${sy},${cz},${sz}`

    return {
        layout,
        marchParamsBuffer,
        sdfFn,
        update,
        setActiveRes,
        getStateKey,
        getFootprint: () => ({spanX, spanY, originX, originY, rBound}),
        getRotation: () => ({cx, sx, cy, sy, cz, sz}),
        getGeometryKey,
    }
}

/**
 * SVG-extrusion 3D SDF setup. Lifts the uploaded 2D SDF field into a solid by extrusion (slab +
 * rounded bevel) and raymarches it. The `sdfFn` SAMPLES the 2D SDF DATA texture (bound as the
 * layout's `sdfSource`) via a bilinear `textureLoad` INSIDE the march (rgba/r16 float texel reads,
 * explicit level 0).
 *
 * The bounding radius uses the full-field fallback (`hwf = hhf = 0.5`) rather than a content-tight
 * scan (`getSdfContentBounds`, which lives in `utilities/sdfBounds.ts`). Only march EFFICIENCY
 * differs, never correctness.
 */
export function createSvg3dSdfSetup(
    params: GpuFragmentParams,
    shapeSdfUrl: string,
    getShapeConfig: () => unknown,
    opts: VolumetricSetupOptions = {},
): VolumetricFieldSetup {
    const root = params.gpu.root
    const extraPad = opts.extraPad ?? (() => 0)
    const {texture, getVersion} = createSdfDataTexture(params, shapeSdfUrl)
    const {layout} = makeVolumetricSvgFieldLayout()
    const marchParamsBuffer = root.createUniform(MarchParams) as unknown as MarchParamsUniform

    // Extrusion sdf: samples the 2D field (sdfSource) via inline bilinear textureLoad, then
    // box-combines with the z-slab and rounds by the bevel. pA = halfDepth, pB = bevel.
    const sdfFn: BakedSdf = tgpu.fn([d.vec3f], d.f32)((p) => {
        'use gpu'
        // The source is the generated field (2048², 1024² on phones) or a legacy 512² `.bin`:
        // read its size rather than assuming one.
        const SIZE_F = d.f32(std.textureDimensions(layout.$.sdfSource).x)
        const MAX_TEXEL = SIZE_F - 1.0
        const pp = layout.$.params
        const halfDepth = pp.pA
        const bevel = pp.pB
        // Shape-local (x right, y up) → texel coords. Texture is stored y-down (screen orientation),
        // so y is un-flipped here (0.5 - y), matching the flat sampler. The texture handle
        // (`layout.$.sdfSource`) is referenced inline per read — a local `const src =
        // layout.$.sdfSource` transpiles to an invalid pointer-from-origin-handle.
        const u = (p.x + 0.5) * SIZE_F - 0.5
        const v = (0.5 - p.y) * SIZE_F - 0.5
        const x0 = std.floor(u)
        const y0 = std.floor(v)
        const fx = u - x0
        const fy = v - y0
        const cx0 = d.u32(std.clamp(x0, d.f32(0), MAX_TEXEL))
        const cx1 = d.u32(std.clamp(x0 + 1.0, d.f32(0), MAX_TEXEL))
        const cy0 = d.u32(std.clamp(y0, d.f32(0), MAX_TEXEL))
        const cy1 = d.u32(std.clamp(y0 + 1.0, d.f32(0), MAX_TEXEL))
        const s00 = std.textureLoad(layout.$.sdfSource, d.vec2u(cx0, cy0), 0).r
        const s10 = std.textureLoad(layout.$.sdfSource, d.vec2u(cx1, cy0), 0).r
        const s01 = std.textureLoad(layout.$.sdfSource, d.vec2u(cx0, cy1), 0).r
        const s11 = std.textureLoad(layout.$.sdfSource, d.vec2u(cx1, cy1), 0).r
        const bilinear = std.mix(std.mix(s00, s10, fx), std.mix(s01, s11, fx), fy)

        // Near the wall, a cubic B-spline instead of bilinear. A bilinear distance field is C0:
        // its slope jumps at every texel, so the extruded side walls come out FACETED (bands
        // running along the extrusion on curved walls, streaks on slanted ones) — magnified by
        // anything that reflects or refracts them. The B-spline is C2: smooth normals AND smooth
        // curvature, so even a mirror reflection glides. It approximates rather than interpolates
        // (a fraction of a texel, invisible at the generated 2048² field). Only within ~3 texels of
        // the wall, blended in so the field stays continuous; far away the march keeps 4 loads.
        let field2d = bilinear
        const nearWall = 1.0 - std.smoothstep(1.5 / SIZE_F, 3.0 / SIZE_F, std.abs(bilinear + bevel))
        if (nearWall > 0.0) {
            const ox = 1.0 - fx
            const bx0 = ox * ox * ox * 0.16666667
            const bx1 = fx * fx * fx * 0.5 - fx * fx + 0.66666667
            const bx2 = fx * fx * fx * -0.5 + fx * fx * 0.5 + fx * 0.5 + 0.16666667
            const bx3 = fx * fx * fx * 0.16666667
            const oy = 1.0 - fy
            const by0 = oy * oy * oy * 0.16666667
            const by1 = fy * fy * fy * 0.5 - fy * fy + 0.66666667
            const by2 = fy * fy * fy * -0.5 + fy * fy * 0.5 + fy * 0.5 + 0.16666667
            const by3 = fy * fy * fy * 0.16666667
            const cxm = d.u32(std.clamp(x0 - 1.0, d.f32(0), MAX_TEXEL))
            const cx2 = d.u32(std.clamp(x0 + 2.0, d.f32(0), MAX_TEXEL))
            const cym = d.u32(std.clamp(y0 - 1.0, d.f32(0), MAX_TEXEL))
            const cy2 = d.u32(std.clamp(y0 + 2.0, d.f32(0), MAX_TEXEL))
            const rowM = std.textureLoad(layout.$.sdfSource, d.vec2u(cxm, cym), 0).r * bx0
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx0, cym), 0).r * bx1
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx1, cym), 0).r * bx2
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx2, cym), 0).r * bx3
            const row0 = std.textureLoad(layout.$.sdfSource, d.vec2u(cxm, cy0), 0).r * bx0
                + s00 * bx1 + s10 * bx2
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx2, cy0), 0).r * bx3
            const row1 = std.textureLoad(layout.$.sdfSource, d.vec2u(cxm, cy1), 0).r * bx0
                + s01 * bx1 + s11 * bx2
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx2, cy1), 0).r * bx3
            const row2 = std.textureLoad(layout.$.sdfSource, d.vec2u(cxm, cy2), 0).r * bx0
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx0, cy2), 0).r * bx1
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx1, cy2), 0).r * bx2
                + std.textureLoad(layout.$.sdfSource, d.vec2u(cx2, cy2), 0).r * bx3
            field2d = std.mix(bilinear, rowM * by0 + row0 * by1 + row1 * by2 + row2 * by3, nearWall)
        }

        const d2 = field2d + bevel
        const wz = std.abs(p.z) - halfDepth + bevel
        const inner = std.min(std.max(d2, wz), d.f32(0))
        const outer = std.length(d.vec2f(std.max(d2, d.f32(0)), std.max(wz, d.f32(0))))
        return inner + outer - bevel
    })

    let elapsed = 0
    let lastShapeJson = ''
    let lastCfg: Record<string, unknown> = parseShapeConfig(getShapeConfig())
    const springs = new Map<string, {current: number; velocity: number}>()
    let halfDepth = 0.06, bevel = 0.02, rBound = 0.75
    let cx = 1, sx = 0, cy = 1, sy = 0, cz = 1, sz = 0
    let spanX = 1, spanY = 1, originX = 0, originY = 0
    let activeRes = resolveVolumetricFieldRes()

    const writeMarch = () => {
        marchParamsBuffer.write({
            rot: {cx, sx, cy, sy, cz, sz},
            rBound,
            spanX, spanY, originX, originY,
            activeRes,
            pA: halfDepth, pB: bevel, pC: 0, pD: 0,
            mb0: d.vec3f(0, 0, 0), mb1: d.vec3f(0, 0, 0), mb2: d.vec3f(0, 0, 0), mb3: d.vec3f(0, 0, 0),
            mb4: d.vec3f(0, 0, 0), mb5: d.vec3f(0, 0, 0), mb6: d.vec3f(0, 0, 0), mb7: d.vec3f(0, 0, 0),
        })
    }

    const update = (frameParams?: ShapeFrameParams) => {
        const deltaTime = frameParams?.deltaTime ?? 0
        elapsed += deltaTime
        const raw = getShapeConfig()
        if (raw && typeof raw === 'object') {
            lastCfg = raw as Record<string, unknown>
        } else if (typeof raw === 'string') {
            if (raw !== lastShapeJson) {
                lastShapeJson = raw
                try { lastCfg = JSON.parse(raw) } catch { /* keep previous */ }
            }
        }
        const frame: ShapeSubPropFrame = {
            elapsed, deltaTime,
            pointerX: frameParams?.pointer?.x ?? 0.5,
            pointerY: frameParams?.pointer?.y ?? 0.5,
            springs,
        }
        const sub = (k: string) => resolveShapeSubProp(lastCfg[k], SVG3D_DEFAULTS[k] ?? 0, frame, k)

        halfDepth = Math.max(0.001, sub('depth') * 0.5)
        bevel = Math.min(Math.max(0, sub('bevel')), halfDepth)

        // Content-tight bounds: shrink the field domain (and the march bounding sphere) to the logo's
        // actual extent instead of the full field, so the fixed field resolution + the 64-step march
        // budget are spent on the SHAPE, not empty margin. This sharpens the silhouette at every depth
        // — most at high depth, where the domain otherwise balloons with halfDepth and the deep
        // grazing edge tears. Falls back to the full field until the async scan resolves (or if the SDF
        // has no inside pixels); the +0.02 guards the anti-aliased silhouette + bevel from clipping.
        // getStateKey already folds in rBound/spanX/spanY, so the frame the scan lands re-marches.
        const cb = getSdfContentBounds(shapeSdfUrl)
        const hwf = cb ? Math.min(0.5, cb.hwf + 0.02) : 0.5
        const hhf = cb ? Math.min(0.5, cb.hhf + 0.02) : 0.5
        const pad = extraPad()
        rBound = Math.hypot(hwf, hhf, halfDepth) + bevel + 0.03 + pad

        const rx = sub('rotX') * DEG, ry = sub('rotY') * DEG, rz = sub('rotZ') * DEG
        cx = Math.cos(rx); sx = Math.sin(rx)
        cy = Math.cos(ry); sy = Math.sin(ry)
        cz = Math.cos(rz); sz = Math.sin(rz)

        // Project the oriented bounding box (local half-sizes hwf×hhf×halfDepth, rounded outward by
        // bevel) onto the field axes for the aspect-fit domain.
        const s = {cx, sx, cy, sy, cz, sz}
        const eX = rotateVecCpu(1, 0, 0, s)
        const eY = rotateVecCpu(0, 1, 0, s)
        const axisPad = bevel + 0.03 + pad
        const rBoundX = Math.abs(eX.x) * hwf + Math.abs(eX.y) * hhf + Math.abs(eX.z) * halfDepth + axisPad
        const rBoundY = Math.abs(eY.x) * hwf + Math.abs(eY.y) * hhf + Math.abs(eY.z) * halfDepth + axisPad
        const rPadX = rBoundX + 0.05
        const rPadY = rBoundY + 0.05
        spanX = rPadX * 2; spanY = rPadY * 2
        originX = 0.5 - rPadX; originY = 0.5 - rPadY
        writeMarch()
    }
    update()

    const setActiveRes = (res: number) => {
        if (res === activeRes) return
        activeRes = res
        writeMarch()
    }

    const getGeometryKey = () => `${halfDepth},${bevel},${rBound},${getVersion()}`
    const getStateKey = () =>
        `${getGeometryKey()},${spanX},${spanY},${cx},${sx},${cy},${sy},${cz},${sz}`

    return {
        layout: layout as unknown as VolumetricFieldLayout,
        marchParamsBuffer,
        sdfFn,
        sdfSourceTexture: texture.texture,
        update,
        setActiveRes,
        getStateKey,
        getFootprint: () => ({spanX, spanY, originX, originY, rBound}),
        getRotation: () => ({cx, sx, cy, sy, cz, sz}),
        getGeometryKey,
    }
}

/**
 * Flat analytic shape lifted into a slab (the 2D primitive extruded along z by `getHalfDepth()`,
 * no bevel, no rotation — the consumer's camera supplies any tilt). The volumetric counterpart of
 * `createAnalyticSdfSampler` for a consumer that wants EVERY shape on the compute path (the voxel
 * pre-march treats a flat circle as a one-layer-deep voxel disc). The eight analytic sub-props ride
 * the MarchParams slots: pA..pD = radius/sides/rounding/innerRatio · mb0 = (rotation, height,
 * offset) · mb1 = (aperture, halfDepth, –).
 */
export function createAnalytic2dExtrudeSetup(
    root: TgpuRoot,
    shapeType: string,
    getShapeConfig: () => unknown,
    getHalfDepth: () => number,
    opts: VolumetricSetupOptions = {},
): VolumetricFieldSetup {
    const extraPad = opts.extraPad ?? (() => 0)
    const {layout} = makeVolumetricFieldLayout()
    const marchParamsBuffer = root.createUniform(MarchParams) as unknown as MarchParamsUniform
    const analytic = buildAnalyticSdfFn(shapeType)

    const sdfFn: BakedSdf = tgpu.fn([d.vec3f], d.f32)((p) => {
        'use gpu'
        const pp = layout.$.params
        // Shape-local (x right, y up) → the analytic fn's field UV (y-down, centred at 0.5).
        const uv = d.vec2f(p.x + 0.5, 0.5 - p.y)
        const d2 = analytic(uv, pp.pA, pp.pB, pp.pC, pp.pD, pp.mb0.x, pp.mb0.y, pp.mb0.z, pp.mb1.x).x
        const wz = std.abs(p.z) - pp.mb1.y
        const inner = std.min(std.max(d2, wz), d.f32(0))
        const outer = std.length(d.vec2f(std.max(d2, d.f32(0)), std.max(wz, d.f32(0))))
        return inner + outer
    })

    let lastShapeJson = ''
    let lastCfg: Record<string, unknown> = parseShapeConfig(getShapeConfig())
    let sub = analyticSubPropValues(lastCfg)
    let halfDepth = 0.06, rBound = 0.6
    let spanX = 1, spanY = 1, originX = 0, originY = 0
    let activeRes = resolveVolumetricFieldRes()

    const writeMarch = () => {
        marchParamsBuffer.write({
            rot: {cx: 1, sx: 0, cy: 1, sy: 0, cz: 1, sz: 0},
            rBound,
            spanX, spanY, originX, originY,
            activeRes,
            pA: sub.radius, pB: sub.sides, pC: sub.rounding, pD: sub.innerRatio,
            mb0: d.vec3f(sub.rotation, sub.height, sub.offset),
            mb1: d.vec3f(sub.aperture, halfDepth, 0),
            mb2: d.vec3f(0, 0, 0), mb3: d.vec3f(0, 0, 0),
            mb4: d.vec3f(0, 0, 0), mb5: d.vec3f(0, 0, 0), mb6: d.vec3f(0, 0, 0), mb7: d.vec3f(0, 0, 0),
        })
    }

    const update = () => {
        const raw = getShapeConfig()
        if (raw && typeof raw === 'object') {
            lastCfg = raw as Record<string, unknown>
        } else if (typeof raw === 'string') {
            if (raw !== lastShapeJson) {
                lastShapeJson = raw
                try { lastCfg = JSON.parse(raw) } catch { /* keep previous */ }
            }
        }
        sub = analyticSubPropValues(lastCfg)
        halfDepth = Math.max(0.001, getHalfDepth())
        // Extent from the SAME resolved sub-props the SDF reads (radius may arrive as width /
        // bottomWidth, height defaults to the SDF's), so the bounding radius can never clip the slab.
        const {hwf, hhf} = shapeContentExtent({shape: {...lastCfg, radius: sub.radius, height: sub.height}})
        rBound = Math.hypot(hwf, hhf, halfDepth) + 0.02 + extraPad()
        const rPad = rBound + 0.05
        spanX = rPad * 2; spanY = rPad * 2
        originX = 0.5 - rPad; originY = 0.5 - rPad
        writeMarch()
    }
    update()

    const setActiveRes = (res: number) => {
        if (res === activeRes) return
        activeRes = res
        writeMarch()
    }

    const getStateKey = () =>
        `${sub.radius},${sub.sides},${sub.rounding},${sub.innerRatio},${sub.rotation},${sub.height},${sub.offset},${sub.aperture},${halfDepth},${rBound}`

    return {
        layout,
        marchParamsBuffer,
        sdfFn,
        update,
        setActiveRes,
        getStateKey,
        getFootprint: () => ({spanX, spanY, originX, originY, rBound}),
        getRotation: () => ({cx: 1, sx: 0, cy: 1, sy: 0, cz: 1, sz: 0}),
        getGeometryKey: getStateKey,
    }
}

/**
 * The shared shape-effect compute entry. When the active shape is volumetric (analytic 3D or SVG
 * extrude), pre-marches its field into a compute texture and
 * exposes the samplers. Returns null for flat shapes and when no device is present (GPU-free resolve
 * / WebGL) — the consumer then falls back to the inline analytic sampler.
 *
 * CONSUMER WIRES (the shape-effect shader that owns this — a later wave):
 *   1. Declare `...VOLUMETRIC_FIELD_EXTRA_FIELDS` in the shader definition's `extraFields` (the six
 *      SampleParams uniforms this node writes each frame via `setExtraField`).
 *   2. `compute: (params) => { const vf = createVolumetricFieldComputeNode(params, () =>
 *      params.getCpuValue('shape')); return vf && {outputs: vf.outputs, getComputeNodes:
 *      vf.getComputeNodes} }`.
 *   3. `fragment: (params) => { const fieldTex = params.computeOutputs?.volumetricFieldTexture; if
 *      (!fieldTex) { …inline analytic fallback… } const s = buildVolumetricFieldSampler(fieldTex,
 *      {originX: params.uniforms._vfOriginX, …}); …applyGlassEffect(s.volumetricFieldSampler, …)… }`.
 *
 * The one binding this can't close without a consuming fragment is the sampler↔fragment handoff
 * (step 3): the samplers are built in the FRAGMENT (a separate builder invocation from `compute`),
 * from the registered field texture (`outputs.volumetricFieldTexture`) + the fragment's own
 * `extraFields` accessors. Everything on the compute side (setup, march, kernel, dispatch, dirty-key,
 * SampleParams writes) is closed here.
 */
export function createVolumetricFieldComputeNode(
    params: GpuFragmentParams,
    getShapeConfig: () => unknown,
    patternMode: SurfacePatternMode = 'none',
    chordMode: ChordMode = 'span',
): {
    outputs: {volumetricFieldTexture: KitTexture}
    /** `frameParams` is typed `unknown` (narrowed to `ShapeFrameParams` inside) so the returned
     *  object satisfies the contract's `GpuComputeNode` shape directly — a shader's `compute` hook
     *  is just `compute: (params) => createVolumetricFieldComputeNode(params, …)`, no adapter. */
    getComputeNodes: (frameParams?: unknown) => GpuComputeStep[] | null
    /** The raw field TgpuTexture (rgba32float, storage+sampled) — for a consumer whose OWN compute
     *  kernels sample the field (SmokeFill's mask kernel). Bind it as a SAMPLED entry declared
     *  `sampleType: 'unfilterable-float'` (rgba32float is not filterable). */
    fieldTexture: TgpuTexture
    /** The aspect-fit domain + active res describing the field texture's CURRENT contents — the
     *  same values last published to the `_vf*` extraFields (initials before the first march).
     *  Read AFTER `getComputeNodes` each frame for a compute-side consumer's uniform. */
    getFieldDomain: () => {originX: number; originY: number; spanX: number; spanY: number; activeRes: number; rBound: number}
} | null {
    const root = params.gpu?.root
    if (!root) return null // GPU-free resolve / WebGL: consumer falls back to the inline march.

    const cfg = parseShapeConfig(getShapeConfig())
    const shapeSdfUrl = (params.getCpuValue('shapeSdfUrl') as string) || ''
    const shapeType = (params.getCpuValue('shapeType') as string) || (cfg.type as string) || ''

    let setup: VolumetricFieldSetup | null = null
    if (shapeSdfUrl && isSvg3dShapeType(shapeType)) {
        setup = createSvg3dSdfSetup(params, shapeSdfUrl, getShapeConfig)
    } else if (!shapeSdfUrl && is3dShapeType(shapeType)) {
        setup = createAnalytic3dSdfSetup(root, shapeType, cfg, getShapeConfig)
    }
    if (!setup) return null // flat shape → the consumer uses createAnalyticSdfSampler (kit/sdf.ts).

    const march = buildRaymarchedFieldFn(setup.sdfFn, patternMode, chordMode)
    const kernel = buildVolumetricFieldKernel(setup.layout, march)
    const compute = createVolumetricFieldCompute(root, setup.layout, kernel, params.onCleanup, {
        paramsUniform: setup.marchParamsBuffer,
        sdfSourceTexture: setup.sdfSourceTexture,
    })
    const fieldTexture = params.registerComputeTexture(compute.fieldTexture)

    let activeRes = compute.maxRes
    let lastKey = ''
    // Mobile re-march throttle: an auto-animated shape changes its state key every frame; on a
    // coarse-pointer device we march only every Nth CHANGED frame.
    const marchEvery = isMobileGpuViewport() ? 2 : 1
    let changedFrames = 0
    // Mirrors the last `_vf*` publication for `getFieldDomain` (VOLUMETRIC_FIELD_EXTRA_FIELDS initials).
    const domain = {originX: 0, originY: 0, spanX: 1, spanY: 1, activeRes: 1, rBound: 0.6}

    const getComputeNodes = (frameParams?: unknown): GpuComputeStep[] | null => {
        setup!.update(frameParams as ShapeFrameParams | undefined)
        const fp = setup!.getFootprint()
        const scale = (params.getCpuValue('scale') as number) ?? 1
        // `dimensions` is already the device-pixel backing size (the renderer owns DPR).
        const hDev = params.dimensions.height
        activeRes = resolveActiveFieldRes(fp.spanX, fp.spanY, scale, hDev, compute.maxRes, activeRes)
        const key = `${setup!.getStateKey()},${activeRes}`
        if (key === lastKey) return null // truly static → never re-march
        changedFrames++
        if (marchEvery > 1 && changedFrames % marchEvery !== 0) return null
        lastKey = key
        setup!.setActiveRes(activeRes)
        // Publish the aspect-fit domain to the consumer fragment's SampleParams extraFields.
        params.setExtraField('_vfOriginX', fp.originX)
        params.setExtraField('_vfOriginY', fp.originY)
        params.setExtraField('_vfSpanX', fp.spanX)
        params.setExtraField('_vfSpanY', fp.spanY)
        params.setExtraField('_vfActiveRes', activeRes)
        params.setExtraField('_vfRBound', fp.rBound)
        domain.originX = fp.originX
        domain.originY = fp.originY
        domain.spanX = fp.spanX
        domain.spanY = fp.spanY
        domain.activeRes = activeRes
        domain.rBound = fp.rBound
        // Dispatch the active block (activeRes²) — a thunk since the pipeline has no pre-bound size.
        return [() => compute.dispatch(activeRes)]
    }

    return {
        outputs: {volumetricFieldTexture: fieldTexture},
        getComputeNodes,
        fieldTexture: compute.fieldTexture,
        getFieldDomain: () => ({...domain}),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// E. Param-threaded raymarch primitives — closed 3D solids for a PERSPECTIVE sphere-march whose
//    shape parameters ride in as packed vec4 fn-args each march step (a fragment tgpu.fn cannot
//    read node uniforms), the signature `effects/raymarch3d.ParamThreadedSdf` threads:
//      sdf:       (p,            rp0, rp1, rp2, t) -> f32
//      surfaceUV: (hitPos, normal, rp0, rp1, rp2, t) -> vec2f
//    Rotation shapes pack CPU-precomputed Euler cos/sin + sizes:
//      rp0 = (cx, sx, cy, sy)   rp1 = (cz, sz, sizeA, sizeB)   rp2 = (sizeC, sizeD, _, _)
//    The twisted ribbon carries no 3D rotation — a 2D angle frame + a twisted capsule section:
//      rp0 = (angleRad, twist, halfW, width), rp1 = (thickness, phase, lipschitz, _), t = time.
//    Distinct .$name pins per fn: the trace invokes the SDF through a closure alias, so typegpu
//    cannot derive the const name (two shapes in one composition would otherwise collide).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

// Twist advance per unit animated time of the twisted-ribbon section (part of the primitive's
// parameterization — the ribbon's `phase` arg offsets it per instance).
const TWISTED_RIBBON_TWIST_RATE = 0.08

/** Euler rotation rotateZ(rotateX(rotateY(p))) from packed cos/sin (rp0 = cx,sx,cy,sy; rp1.xy = cz,sz). Pure. */
export const eulerRotatePacked = tgpu.fn([d.vec3f, d.vec4f, d.vec4f], d.vec3f)((p, rp0, rp1) => {
    'use gpu'
    const cx = rp0.x
    const sx = rp0.y
    const cy = rp0.z
    const sy = rp0.w
    const cz = rp1.x
    const sz = rp1.y
    const py = d.vec3f(p.x * cy + p.z * sy, p.y, p.x * sy * -1.0 + p.z * cy)
    const px = d.vec3f(py.x, py.y * cx - py.z * sx, py.y * sx + py.z * cx)
    return d.vec3f(px.x * cz - px.y * sz, px.x * sz + px.y * cz, px.z)
}).$name('eulerRotatePacked')

export const paramSphereSdf = tgpu.fn([d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.f32)((p, rp0, rp1, _rp2, _t) => {
    'use gpu'
    return std.length(eulerRotatePacked(p, rp0, rp1)) - rp1.z
}).$name('paramSphereSdf')
/** Spherical (longitude/latitude) surface parameterization. */
export const sphereSurfaceUV = tgpu.fn([d.vec3f, d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec2f)((h, _n, rp0, rp1, _rp2, _t) => {
    'use gpu'
    const n = std.normalize(eulerRotatePacked(h, rp0, rp1))
    return d.vec2f(std.atan2(n.z, n.x) / 6.283185307179586 + 0.5, (n.y + 1.0) * 0.5)
}).$name('sphereSurfaceUV')

export const paramTorusSdf = tgpu.fn([d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.f32)((p, rp0, rp1, _rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(p, rp0, rp1)
    const q = d.vec2f(std.length(d.vec2f(rp.x, rp.z)) - rp1.z, rp.y)
    return std.length(q) - rp1.w
}).$name('paramTorusSdf')
/** Toroidal (ring angle / tube angle) surface parameterization. */
export const torusSurfaceUV = tgpu.fn([d.vec3f, d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec2f)((h, _n, rp0, rp1, _rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(h, rp0, rp1)
    const u = std.atan2(rp.z, rp.x) / 6.283185307179586 + 0.5
    const ringDist = std.length(d.vec2f(rp.x, rp.z)) - rp1.z
    return d.vec2f(u, std.atan2(rp.y, ringDist) / 6.283185307179586 + 0.5)
}).$name('torusSurfaceUV')

export const paramBoxSdf = tgpu.fn([d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.f32)((p, rp0, rp1, rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(p, rp0, rp1)
    const rounding = rp2.y
    const qx = std.abs(rp.x) - rp1.z + rounding
    const qy = std.abs(rp.y) - rp1.w + rounding
    const qz = std.abs(rp.z) - rp2.x + rounding
    const outer = std.length(d.vec3f(std.max(qx, d.f32(0)), std.max(qy, d.f32(0)), std.max(qz, d.f32(0))))
    const inner = std.min(std.max(qx, std.max(qy, qz)), d.f32(0))
    return outer + inner - rounding
}).$name('paramBoxSdf')
/** Dominant-face cube projection: picks the face by the largest |normal| axis, scaled to 0–1. */
export const boxFaceUV = tgpu.fn([d.vec3f, d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec2f)((h, n, rp0, rp1, rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(h, rp0, rp1)
    const rn = eulerRotatePacked(n, rp0, rp1)
    const an = d.vec3f(std.abs(rn.x), std.abs(rn.y), std.abs(rn.z))
    const hx = rp1.z
    const hy = rp1.w
    const hz = rp2.x
    const scaleX = 0.5 / std.max(hy, hz)
    const scaleY = 0.5 / std.max(hx, hz)
    const scaleZ = 0.5 / std.max(hx, hy)
    const uvX = d.vec2f(rp.z * scaleX + 0.5, rp.y * scaleX + 0.5)
    const uvY = d.vec2f(rp.x * scaleY + 0.5, rp.z * scaleY + 0.5)
    const uvZ = d.vec2f(rp.x * scaleZ + 0.5, rp.y * scaleZ + 0.5)
    const pickXZ = std.select(uvZ, uvX, an.x > an.z)
    const pickYZ = std.select(uvZ, uvY, an.y > an.z)
    return std.select(pickYZ, pickXZ, an.x > an.y)
}).$name('boxFaceUV')

export const paramCapsuleSdf = tgpu.fn([d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.f32)((p, rp0, rp1, _rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(p, rp0, rp1)
    const radius = rp1.z
    const halfHeight = rp1.w
    const clampedY = std.clamp(rp.y, halfHeight * -1.0, halfHeight)
    return std.length(d.vec3f(rp.x, rp.y - clampedY, rp.z)) - radius
}).$name('paramCapsuleSdf')
/** Cylindrical (azimuth / axial span) surface parameterization. */
export const capsuleSurfaceUV = tgpu.fn([d.vec3f, d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec2f)((h, _n, rp0, rp1, _rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(h, rp0, rp1)
    const u = std.atan2(rp.z, rp.x) / 6.283185307179586 + 0.5
    const span = rp1.w + rp1.z
    return d.vec2f(u, std.clamp(rp.y / span * 0.5 + 0.5, 0.0, 1.0))
}).$name('capsuleSurfaceUV')

export const paramMobiusSdf = tgpu.fn([d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.f32)((p, rp0, rp1, rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(p, rp0, rp1)
    const ringRadius = rp1.z
    const halfWidth = rp1.w
    const thickness = rp2.x
    const phi = std.atan2(rp.z, rp.x)
    const spine = d.vec3f(std.cos(phi) * ringRadius, 0.0, std.sin(phi) * ringRadius)
    const dvec = d.vec3f(rp.x - spine.x, rp.y - spine.y, rp.z - spine.z)
    const halfPhi = phi * 0.5
    const ev = d.vec3f(std.cos(phi) * std.cos(halfPhi), std.sin(halfPhi), std.sin(phi) * std.cos(halfPhi))
    const vProj = std.dot(dvec, ev)
    const du = std.max(std.abs(vProj) - halfWidth, d.f32(0))
    const perp = d.vec3f(dvec.x - ev.x * vProj, dvec.y - ev.y * vProj, dvec.z - ev.z * vProj)
    return std.length(d.vec2f(du, std.length(perp))) - thickness
}).$name('paramMobiusSdf')
/** Möbius-band (ring angle / half-twisted width) surface parameterization. */
export const mobiusSurfaceUV = tgpu.fn([d.vec3f, d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec2f)((h, _n, rp0, rp1, _rp2, _t) => {
    'use gpu'
    const rp = eulerRotatePacked(h, rp0, rp1)
    const ringRadius = rp1.z
    const halfWidth = rp1.w
    const phi = std.atan2(rp.z, rp.x)
    const spine = d.vec3f(std.cos(phi) * ringRadius, 0.0, std.sin(phi) * ringRadius)
    const dvec = d.vec3f(rp.x - spine.x, rp.y - spine.y, rp.z - spine.z)
    const halfPhi = phi * 0.5
    const ev = d.vec3f(std.cos(phi) * std.cos(halfPhi), std.sin(halfPhi), std.sin(phi) * std.cos(halfPhi))
    const vProj = std.dot(dvec, ev)
    return d.vec2f(phi / 6.283185307179586 + 0.5, vProj / halfWidth * 0.5 + 0.5)
}).$name('mobiusSurfaceUV')

// Twisted ribbon: no 3D rotation — a 2D `angleRad` frame + a twisted capsule cross-section
// animated by `t`. The `/ lipschitz` scale keeps the march from tunnelling through fast twists
// (the SDF is approximate under twist; the CPU passes lipschitz = max(halfW·twist, 1)).
export const paramTwistedRibbonSdf = tgpu.fn([d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.f32)((p, rp0, rp1, _rp2, t) => {
    'use gpu'
    const angleRad = rp0.x
    const twist = rp0.y
    const halfW = rp0.z
    const thickness = rp1.x
    const phase = rp1.y
    const lipschitz = rp1.z
    const cosA = std.cos(angleRad)
    const sinA = std.sin(angleRad)
    const rpx = p.x * cosA + p.y * sinA
    const rpy = p.x * sinA * -1.0 + p.y * cosA
    const twistAngle = rpx * twist + phase + t * TWISTED_RIBBON_TWIST_RATE
    const ca = std.cos(twistAngle)
    const sa = std.sin(twistAngle)
    const localU = rpy * ca + p.z * sa
    const localV = rpy * sa * -1.0 + p.z * ca
    const du = std.max(std.abs(localU) - halfW, d.f32(0))
    return (std.length(d.vec2f(du, localV)) - thickness) / lipschitz
}).$name('paramTwistedRibbonSdf')
/** Twisted-ribbon (along-spine / across-width) surface parameterization. */
export const twistedRibbonSurfaceUV = tgpu.fn([d.vec3f, d.vec3f, d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec2f)((h, _n, rp0, rp1, _rp2, t) => {
    'use gpu'
    const angleRad = rp0.x
    const twist = rp0.y
    const halfW = rp0.z
    const width = rp0.w
    const phase = rp1.y
    const cosA = std.cos(angleRad)
    const sinA = std.sin(angleRad)
    const rpx = h.x * cosA + h.y * sinA
    const rpy = h.x * sinA * -1.0 + h.y * cosA
    const twistAngle = rpx * twist + phase + t * TWISTED_RIBBON_TWIST_RATE
    const ca = std.cos(twistAngle)
    const sa = std.sin(twistAngle)
    const localU = rpy * ca + h.z * sa
    const localUClamped = std.max(std.min(localU, halfW), halfW * -1.0)
    return d.vec2f(rpx * 0.15 + 0.5, localUClamped / width + 0.5)
}).$name('twistedRibbonSurfaceUV')

/**
 * Packed vec4 param-carrier extraFields for a param-threaded trace consumer: `<prefix>0…N-1`,
 * one vec4f per initial. The CPU driver writes them per frame via `params.setExtraField`.
 */
export function packedParamExtraFields(prefix: string, initials: number[][]): Record<string, {schema: d.Vec4f; initial: number[]}> {
    const out: Record<string, {schema: d.Vec4f; initial: number[]}> = {}
    initials.forEach((initial, i) => { out[`${prefix}${i}`] = {schema: d.vec4f, initial} })
    return out
}
