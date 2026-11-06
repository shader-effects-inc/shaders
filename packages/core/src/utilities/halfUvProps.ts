import { lengthToPx, toDimensional, type LengthAxis } from './units'

/**
 * Reading a HALF-EXTENT size prop that may carry pixels.
 *
 * A shape's size props are half-extents in canvas-height units, and a user may switch one to
 * absolute pixels — at which point the prop value becomes a `DimensionalValue` carrying the FULL
 * pixel size (the same number the Design Editor's px chip shows), not the half. A `computeBounds` /
 * `writeBounds` pair therefore has to normalize before it can do arithmetic, and it has to do so
 * IDENTICALLY on both sides or a resize round-trip drifts.
 *
 * Trapezoid had this as a local closure shared between its two bounds callbacks. It lives here so
 * the next `computeBounds` shape that grows px-capable size props can reuse the exact reading rule
 * rather than re-deriving the ÷2 (which is the easy half to get wrong).
 *
 * NOT adopted by Ring or Blob: neither handles px size props at all today (both read
 * `props.<size> as number`), so pointing them at this helper would ADD px support — a behaviour
 * change, not an extraction. Left for whenever that support is wanted deliberately.
 */
/**
 * Normalize a half-extent size prop to half-UV (canvas-height units).
 *
 * - a value with a real unit (`{value, unit: 'px'}`, `'50vw'`, `'40%'`, … — see utilities/units)
 *   → the value is the FULL size, so `px / (2 · canvasHeight)`. A `%` measures `axis` (the box
 *   width for 'x', height for 'y'); on a prop with no axis it's unsupported → `fallback`.
 * - any other `{value: number}` → already the half-UV value (this is Trapezoid's original rule: a
 *   value object whose unit is `'uv'`, missing or unknown is read as UV).
 * - a plain number → already the half-UV value.
 * - anything else → `fallback` (the prop's default), never NaN.
 *
 * "Never NaN" is checked, not assumed: a non-finite numeric value (a dynamic prop resolving to NaN
 * mid-frame) and a px read against a zero or non-finite `canvasHeight` (the frame before the canvas
 * has been measured) both fall back rather than poisoning the bounds arithmetic downstream. The
 * `fallback` itself is normalized at the boundary too (→ 0.5, a half-canvas half-extent), so the
 * promise holds even for a caller that computes its fallback — every call site today passes a
 * literal, making that branch unreachable, but the contract shouldn't depend on it.
 */
export function halfUvFromProp(
    value: unknown,
    canvasHeight: number,
    fallback: number,
    axis: LengthAxis = 'none',
    canvasWidth: number = 0
): number {
    const safeFallback = Number.isFinite(fallback) ? fallback : 0.5
    const d = toDimensional(value)
    // `uv` (object or `'0.3uv'` string) already IS the half-UV value.
    if (d && d.unit === 'uv') return Number.isFinite(d.value) ? d.value : safeFallback
    if (d) {
        if (!Number.isFinite(d.value)) return safeFallback
        if (!(Number.isFinite(canvasHeight) && canvasHeight !== 0)) return safeFallback
        // Any real unit carries the FULL size; convert to px, then to the half-UV.
        const px = lengthToPx(d, axis, { canvasWidth: canvasWidth > 0 ? canvasWidth : canvasHeight, canvasHeight })
        return px != null && Number.isFinite(px) ? px / (2 * canvasHeight) : safeFallback
    }
    if (value != null && typeof value === 'object' && typeof (value as {value?: unknown}).value === 'number') {
        const dv = value as {value: number}
        return Number.isFinite(dv.value) ? dv.value : safeFallback
    }
    return typeof value === 'number' && Number.isFinite(value) ? value : safeFallback
}
