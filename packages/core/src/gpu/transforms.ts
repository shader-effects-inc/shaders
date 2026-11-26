/**
 * GPU-side prop transforms. The value-producing color/position transforms return a `typegpu`
 * CPU-side vector instance (`d.vec4f` / `d.vec2f`) that the uniform store writes straight into
 * the packed struct field. The scalar transforms (boolean/angle/edges/colorSpace) are already
 * pure numbers, so they are re-exported unchanged.
 *
 * `gpuTransformFor(transform)` maps a `PropConfig.transform` reference to its GPU-side
 * equivalent. This lets a shader keep its EXISTING `props` definition untouched — the props
 * reference the shared transforms (`transformColor`, `transformPosition`, …); the bridge
 * (`uniformBridge.ts`) swaps each one for the GPU transform at registration time.
 *
 * This module statically imports the shared transform function references from
 * `utilities/transformations.ts` for identity-based mapping (and re-exports the pure scalar
 * transforms from there). The value-producing math here (`transformColorGpu` /
 * `transformPositionGpu`) builds `d.vec*f` instances via colorjs.io.
 */
import Color from 'colorjs.io'

/** A colour channel as a finite number: colorjs.io gives `null` for `none` and NaN for some keywords. */
const finiteOr0 = (x: number | null | undefined): number => (typeof x === 'number' && Number.isFinite(x) ? x : 0)
import * as d from 'typegpu/data'

import type {DimensionalValue} from '../types'
import {
    transformColor,
    transformPosition,
    transformBoolean,
    transformAngle,
    transformEdges,
    transformColorSpace,
    transformStrokePosition,
    setColorSpaceMode as setColorSpaceModeBase,
    type PositionAxisInput,
} from '../utilities/transformations'
import {colorStopsTransform} from '../utilities/colorStops'
import {listPropTransform} from '../utilities/listProps'

// Re-export the already-pure scalar transforms so shaders import EVERYTHING from one place
// (`@coreroot/gpu/porters`). These are pure number math, so the GPU-side equivalent IS the
// shared function.
export {transformBoolean, transformAngle, transformEdges, transformColorSpace, transformStrokePosition}
export {colorStopsTransform, listPropTransform}
export type {PositionAxisInput}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Color-space mode (GPU-side single source of truth)
// ═══════════════════════════════════════════════════════════════════════════════════════
//
// `transformColorGpu` parses into `p3-linear` (default) or `srgb-linear` (Figma-matching).
// The mode is a rendering-wide setting fed by the renderer at initialize. We own it here
// rather than reading the private mode global inside `transformations.ts` (which exports no
// getter). `setColorSpaceModeGpu` also forwards to the `transformations.ts` setter so the
// shared `transformColor` stays in lockstep.
//
// NOTE: `initialize()` (gpu/index.ts) imports `setColorSpaceMode` from `transformations.ts`
// directly rather than `setColorSpaceModeGpu`, so until it is repointed here a renderer
// initialised with `colorSpace: 'srgb'` drives the GPU path at the default `p3-linear` (only
// the rare srgb renderer is affected; the renderer default is p3-linear).
let gpuColorSpaceMode: 'p3-linear' | 'srgb' = 'p3-linear'

export function setColorSpaceModeGpu(mode: 'p3-linear' | 'srgb'): void {
    gpuColorSpaceMode = mode
    // Keep the shared `transformColor` mode in lockstep (still used by non-GPU code paths).
    try {
        setColorSpaceModeBase(mode)
    } catch {
        /* best-effort */
    }
}

export function getColorSpaceModeGpu(): 'p3-linear' | 'srgb' {
    return gpuColorSpaceMode
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// transformColorGpu — parseColorChannels + a d.vec4f wrapper
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Parse a CSS color string into finite RGBA components in the active color space (via
 * colorjs.io). Defensive against colorjs.io throwing on malformed input (→ transparent black)
 * and against CSS values whose channels are NaN by spec ('transparent') poisoning the packed
 * field (→ 0).
 */
function parseColorChannelsGpu(value: string): {r: number; g: number; b: number; a: number} {
    try {
        const parsed = new Color(value)
        const targetSpace = gpuColorSpaceMode === 'srgb' ? 'srgb-linear' : 'p3-linear'
        const coords = parsed.to(targetSpace).coords
        // colorjs.io returns a BOXED Number object for functional notations (rgba()/hsl()/rgb(…/…)),
        // so a raw Number.isFinite(parsed.alpha) is false → alpha silently collapsed to 0 (the color
        // rendered fully transparent). Number() unboxes it (preserving the value) before the check;
        // a genuinely non-finite alpha defaults OPAQUE, not invisible.
        const alpha = Number(parsed.alpha)
        return {
            r: finiteOr0(coords?.[0]),
            g: finiteOr0(coords?.[1]),
            b: finiteOr0(coords?.[2]),
            a: Number.isFinite(alpha) ? alpha : 1,
        }
    } catch {
        return {r: 0, g: 0, b: 0, a: 0}
    }
}

/**
 * GPU-side `transformColor`: CSS string → `d.vec4f` CPU instance. Uses the colorjs.io pipeline
 * + active color-space mode, with the NaN→0 guard.
 */
export function transformColorGpu(value: string): d.v4f {
    const {r, g, b, a} = parseColorChannelsGpu(value)
    return d.vec4f(r, g, b, a)
}

/** Convenience for the color-stops packer: CSS string → flat `[r, g, b, a]` tuple. */
export function colorToRGBA(value: string): [number, number, number, number] {
    const c = transformColorGpu(value)
    return [c.x, c.y, c.z, c.w]
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// transformPositionGpu — transformPosition + a d.vec2f wrapper
// ═══════════════════════════════════════════════════════════════════════════════════════

/** Parse one position axis (number | DimensionalValue | CSS keyword) into a UV number. */
function parsePositionAxis(val: PositionAxisInput, isVertical: boolean): number {
    if (typeof val === 'number') return val
    // Defensive: a DimensionalValue should be resolved to a plain number by the renderer before
    // reaching here; if not, fall back to its raw value (treated as UV).
    if (typeof val === 'object' && val !== null) return (val as DimensionalValue).value

    const str = val.toLowerCase().trim()
    if (isVertical) {
        if (str === 'top') return 0
        if (str === 'bottom') return 1
        if (str === 'center') return 0.5
    } else {
        if (str === 'left') return 0
        if (str === 'right') return 1
        if (str === 'center') return 0.5
    }
    console.warn(`Invalid position value: ${val}. Defaulting to center.`)
    return 0.5
}

/**
 * GPU-side `transformPosition`: `{x, y}` (or a CSS-style string) → `d.vec2f(x, 1 - y)` CPU
 * instance. Y-flip applied (y=0 top, y=1 bottom). DimensionalValue axis inputs are resolved to
 * UV numbers.
 */
export function transformPositionGpu(value: {x: PositionAxisInput; y: PositionAxisInput} | string): d.v2f {
    let x = 0.5
    let y = 0.5

    if (typeof value === 'string') {
        const parts = value.toLowerCase().trim().split(/\s+/)
        if (parts.includes('left')) x = 0
        else if (parts.includes('right')) x = 1
        else if (parts.includes('center') || (parts.length === 1 && (parts[0] === 'top' || parts[0] === 'bottom'))) x = 0.5

        if (parts.includes('top')) y = 0
        else if (parts.includes('bottom')) y = 1
        else if (parts.includes('center') || (parts.length === 1 && (parts[0] === 'left' || parts[0] === 'right'))) y = 0.5

        if (parts.length === 1 && parts[0] === 'center') {
            x = 0.5
            y = 0.5
        }
    } else if (typeof value === 'object' && value !== null) {
        x = parsePositionAxis(value.x, false)
        y = parsePositionAxis(value.y, true)
    } else {
        console.warn(`Invalid position value provided: ${value}. Defaulting to center.`)
    }

    return d.vec2f(x, 1.0 - y)
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// gpuTransformFor — transform reference → GPU-side equivalent
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * A GPU-side transform: takes a raw prop value and returns the FINAL value written to the
 * packed struct (a number, or a `d.vec*f` instance). The uniform store applies it (see
 * `FieldHandle.setFromRaw`), so it must be a pure value → value function.
 */
export type GpuTransform = (value: unknown) => unknown

/**
 * Identity transform (value → value). Not returned by `gpuTransformFor` — kept as a small
 * exported utility some callers/tests reference.
 */
export const identityTransform: GpuTransform = (value) => value

/**
 * Map a `PropConfig.transform` reference to the GPU-side transform the uniform store should
 * apply. Shaders keep their transform-referencing props; the bridge calls this to swap in the
 * GPU equivalent (for the value-producing transforms) or to hand back the transform itself
 * (for the pure scalar ones).
 *
 *   transformColor      → transformColorGpu       (d.vec4f, not an {r,g,b,a} object)
 *   transformPosition   → transformPositionGpu    (d.vec2f, Y-flip applied)
 *   colorStopsTransform → colorStopsTransform      (marker — the bridge expands the prop into
 *                                                   array fields; never applied as a field xform)
 *   undefined           → undefined                (no transform)
 *   anything else       → itself                   (called directly to produce the field value)
 *
 * The "anything else" branch is the key: EVERY other prop transform in the library is a pure
 * value→value function producing a packable scalar — the shared scalar transforms
 * (transformBoolean/Angle/Edges/ColorSpace/StrokePosition), the inline numeric transforms
 * (`v => v * 0.5`, `v => 0.1 + v * 1.4`, deg→rad), the boolean→float transforms (`v => v ? 1 : 0`),
 * and the per-shader enum string→number maps. Their GPU-side equivalent IS the function itself, so
 * we return it and let the uniform store apply it (seed + `setFromRaw`). This is the single point at
 * which a prop's declared transform takes effect, so `uniforms.<prop>`, `propValues.<prop>` and
 * `getCpuValue('<prop>')` all observe the SAME transformed value.
 */
export function gpuTransformFor(propTransform: ((value: never) => unknown) | undefined): GpuTransform | undefined {
    if (!propTransform) return undefined
    if (propTransform === (transformColor as unknown)) return transformColorGpu as GpuTransform
    if (propTransform === (transformPosition as unknown)) return transformPositionGpu as GpuTransform
    if (propTransform === (colorStopsTransform as unknown)) return colorStopsTransform as GpuTransform
    if (propTransform === (listPropTransform as unknown)) return listPropTransform as GpuTransform
    // Every other prop transform is a pure scalar-producing function; the GPU equivalent is itself.
    return propTransform as GpuTransform
}
