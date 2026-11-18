import { isMobileGpuViewport } from './device'
// frame.ts is pure math (type-only typegpu import) — safe for this CPU-only mirror.
import { applySpring } from '../gpu/frame'
import { isMesh3dShapeType } from './meshShape'

// Re-exported so existing importers (`@coreroot/utilities/sdf3d`) keep resolving it
// from here; the canonical definition now lives in ./device.
export { isMobileGpuViewport }

// ─── 3D analytic SDF shapes for the shape-effect shaders (Glass, Emboss, Neon…) ─
//
// A 3D shape is raymarched orthographically (camera looking down +z) through the
// same 0–1 "SDF field" UV space the 2D analytic shapes live in. The sampler
// returns a 2D scalar field that is plug-compatible with applyGlassEffect & co:
//
//   outside the silhouette → the minimum 3D distance encountered along the ray,
//     which for orthographic rays equals the 2D distance to the projected
//     silhouette (distance from a line to a set = distance in the projection plane)
//   inside the silhouette  → minus HALF THE CHORD LENGTH through the solid,
//     i.e. the real optical thickness of the glass along the view ray. For a
//     sphere this is exactly the lens height field −√(r²−d²); for a face-on cube
//     it is a flat slab (no interior refraction, strong edges) — physically the
//     right look for refraction, highlights, and fresnel.
//
// Rotation is rigid, so instead of rotating the point inside the march loop we
// rotate the ray origin/direction once per sample — the SDFs stay axis-aligned.
//
// Sub-prop values inside the shape JSON (sizes and rotations) may be either
// plain numbers or AutoAnimateConfig objects ({ type: 'auto-animate', … }).
// Auto-animate configs are resolved on the CPU each frame, mirroring the
// renderer's prop-driver math — this is what powers spinning/rocking/pulsing
// 3D glass without any renderer involvement.

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

/** True when the shape type produces a volumetric field — analytic 3D, SVG 3D, or a 3D model
 *  (`mesh3D`, drawn into the field rather than raymarched; see `gpu/kit/meshField.ts`). */
export function isVolumetricShapeType(shapeType: string): boolean {
    return is3dShapeType(shapeType) || isSvg3dShapeType(shapeType) || isMesh3dShapeType(shapeType)
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

/**
 * Maximum metaballs3D ball count. Canonical here (this module is the CPU-only mirror the GPU kit
 * imports); shared by the `sdMetaballs` kernel loop and the CPU pack/clamp in
 * `gpu/kit/sdf3d.ts`. Raising it also means adding matching named `mbN: d.vec3f` fields to
 * `MarchParams` (strict uniform layout — named fields, full-write only) and threading them
 * through `sdMetaballs`' explicit args and the two `writeMarch` sites.
 */
export const MAX_METABALLS = 8

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
    const d = SHAPE3D_DEFAULTS[cfg.type as Shape3DType] ?? {}
    const n = (k: string, f = 0.3) => shapeSubPropMax(cfg[k], d[k] ?? f)
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
        case 'ribbon3D': return Math.hypot(n('length', 0.38), n('wave', 0.1) * 1.2 + Math.hypot(n('width', 0.15), n('thickness', 0.03))) + 0.01
        case 'blob3D': return n('radius', 0.3) + n('wobble', 0.07) + 0.02
        case 'gyroscope3D': return n('radius', 0.34) + n('thickness', 0.028) * 0.5 + n('width', 0.1) * 0.5 + n('rounding', 0.006)
        case 'sphere3D':
        case 'octahedron3D':
        default: return n('radius', 0.35)
    }
}

