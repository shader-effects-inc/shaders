import type { BoundingBoxConfig, BoundingBoxDimension, BoundingBoxInput } from '../types'
import { lengthToPx, toDimensional, warnUnitOnce } from './units'

/**
 * Full-frame identity bounding box. A box with these values covers the whole viewport,
 * which renders the same as having no box at all.
 */
export const DEFAULT_BOUNDING_BOX: BoundingBoxConfig = {
    x: { value: 0, unit: 'uv' },
    y: { value: 0, unit: 'uv' },
    width: { value: 1, unit: 'uv' },
    height: { value: 1, unit: 'uv' },
    origin: 'top-left',
    rotation: 0
}

/**
 * One authored box length → its stored object form.
 * - a bare number is PIXELS (as in CSS and React's `style` prop)
 * - a CSS-like string (`'50%'`, `'2rem'`) is parsed
 * - an object keeps its value/unit; a missing half comes from `fallback` (a partial `{ value }`
 *   has always read in the fallback's unit)
 * - anything unreadable falls back entirely (with a one-time warning for a bad string)
 */
function toBoxDimension(d: unknown, fallback: BoundingBoxDimension): BoundingBoxDimension {
    if (typeof d === 'number') return Number.isFinite(d) ? { value: d, unit: 'px' } : { ...fallback }
    if (typeof d === 'string') {
        const parsed = toDimensional(d)
        if (parsed) return parsed
        warnUnitOnce(`"${d}" isn't a valid bounding-box length — use a number (px) or a string like '50%', '24px', '10vw', '2rem'.`)
        return { ...fallback }
    }
    if (d && typeof d === 'object') {
        const o = d as Partial<BoundingBoxDimension>
        const value = typeof o.value === 'number' ? o.value : fallback.value
        const unit = typeof o.unit === 'string' && toDimensional({ value, unit: o.unit }) ? o.unit : fallback.unit
        return { value, unit }
    }
    return { ...fallback }
}

/**
 * Merges a user-supplied (possibly partial) bounding box with the full-frame defaults so the
 * renderer always receives a complete BoundingBoxConfig — the GPU path dereferences every
 * dimension (`bbox.x.unit`, …) and would crash on a partial object. Lengths may be authored in
 * any accepted form (see toBoxDimension) and are normalized to their object form here, so every
 * reader downstream sees `{ value, unit }`. Returns undefined when no box is given: absence keeps
 * the bounding-box render path disabled entirely, so a merged default is never forced onto layers
 * that don't use one.
 */
export function resolveBoundingBox(bbox?: BoundingBoxInput | null): BoundingBoxConfig | undefined {
    if (!bbox || typeof bbox !== 'object') return undefined
    return {
        x: toBoxDimension(bbox.x, DEFAULT_BOUNDING_BOX.x),
        y: toBoxDimension(bbox.y, DEFAULT_BOUNDING_BOX.y),
        width: toBoxDimension(bbox.width, DEFAULT_BOUNDING_BOX.width),
        height: toBoxDimension(bbox.height, DEFAULT_BOUNDING_BOX.height),
        origin: bbox.origin ?? 'top-left',
        rotation: bbox.rotation ?? 0,
        ...(bbox.cornerRadius !== undefined && bbox.cornerRadius !== null
            ? { cornerRadius: toBoxDimension(bbox.cornerRadius, { value: 0, unit: 'uv' }) }
            : {}),
        ...(bbox.lockAspect !== undefined ? { lockAspect: bbox.lockAspect } : {})
    }
}

/**
 * A box length as a fraction of the canvas on its axis (x / width → width, y / height → height).
 * `uv` already is that fraction; every other unit converts through CSS pixels, with `%` measuring
 * the containing box on the same axis (the canvas, until frames exist). Accepts the authored forms
 * too (bare number = px) so a caller that skipped resolveBoundingBox still reads correctly.
 */
export function boxLengthUV(d: unknown, axis: 'x' | 'y', canvasWidth: number, canvasHeight: number): number {
    const dim = axis === 'x' ? canvasWidth : canvasHeight
    if (typeof d === 'number') return dim > 0 ? d / dim : 0
    const v = toDimensional(d)
    if (!v) {
        // A partial / unit-less object has always read as uv.
        const raw = (d as { value?: unknown } | null)?.value
        return typeof raw === 'number' ? raw : 0
    }
    if (v.unit === 'uv') return v.value
    const px = lengthToPx(v, axis, { canvasWidth, canvasHeight }) ?? 0
    return dim > 0 ? px / dim : 0
}

/**
 * The clip window's corner radius in canvas-height UV. `uv` is already that; every other length
 * converts through pixels. `%` isn't accepted yet: in CSS a % corner radius is per-axis
 * (elliptical), and this clip window only draws round corners — it falls back to square corners
 * until elliptical corners exist.
 */
export function boxCornerRadiusUV(d: unknown, canvasWidth: number, canvasHeight: number): number {
    if (d === undefined || d === null) return 0
    if (typeof d === 'number') return canvasHeight > 0 ? d / canvasHeight : 0
    const v = toDimensional(d)
    if (!v) {
        const raw = (d as { value?: unknown } | null)?.value
        return typeof raw === 'number' ? raw : 0
    }
    if (v.unit === 'uv') return v.value
    const px = lengthToPx(v, 'none', { canvasWidth, canvasHeight })
    if (px == null) {
        warnUnitOnce(`"${v.unit}" isn't supported on boundingBox.cornerRadius — % corners are elliptical in CSS and not supported yet. Using square corners.`)
        return 0
    }
    return canvasHeight > 0 ? px / canvasHeight : 0
}

/** Whether a box has a length that depends on the browser viewport or root font size. */
export function boundingBoxUsesViewportUnits(bbox?: Partial<BoundingBoxConfig> | null): boolean {
    if (!bbox) return false
    for (const d of [bbox.x, bbox.y, bbox.width, bbox.height, bbox.cornerRadius]) {
        const v = toDimensional(d)
        if (v && (v.unit === 'vw' || v.unit === 'vh' || v.unit === 'vmin' || v.unit === 'vmax' || v.unit === 'rem')) return true
    }
    return false
}

/**
 * True when a box covers the whole frame with no rotation or corner rounding — i.e. it renders
 * identically to having no box at all. Such a box is a real cost: the clip mask, the resize-fit
 * UV mapping and (for a Group) a full RTT round-trip + resample are all emitted for an identity
 * transform. `boundingBox={{ origin: 'center' }}` merges to exactly this, so it is a common
 * shape to hit; both the composer and the renderer treat it as "no box".
 *
 * Only canvas-relative units count as identity: `uv`, and `%` / `cw` / `ch`, which are the same
 * fraction of the canvas written ×100 (offset 0, extent 100 on the matching axis). A px or
 * viewport-relative box that happens to equal the frame right now would flip in and out of
 * identity on every resize, and crossing that boundary is a recompile.
 *
 * Origin is irrelevant here: with x=0, y=0 and a full-frame size every origin resolves the box's
 * top-left to (0,0) (`resolveTopLeft` in kit/uvTransform).
 */
export function isIdentityBoundingBox(bbox?: Partial<BoundingBoxConfig> | null): boolean {
    if (!bbox || typeof bbox !== 'object') return false
    const canvasUnit = (axis: 'x' | 'y') => axis === 'x' ? 'cw' : 'ch'
    const isFrac = (d: BoundingBoxDimension | undefined, frac: number, axis: 'x' | 'y'): boolean =>
        !!d && typeof d === 'object' && (
            (d.unit === 'uv' && d.value === frac) ||
            ((d.unit === '%' || d.unit === canvasUnit(axis)) && d.value === frac * 100)
        )
    return (
        isFrac(bbox.x, 0, 'x') &&
        isFrac(bbox.y, 0, 'y') &&
        isFrac(bbox.width, 1, 'x') &&
        isFrac(bbox.height, 1, 'y') &&
        (bbox.rotation ?? 0) === 0 &&
        (bbox.cornerRadius?.value ?? 0) === 0
    )
}
