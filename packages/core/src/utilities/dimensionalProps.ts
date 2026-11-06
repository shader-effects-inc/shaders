import type { BoundingBoxDeclaration, BoundingBoxOrigin, BoundingBoxPropBindings, ComponentDefinition, DimensionalValue } from '../types'
import { isDimensionalValue, lengthToPx, toDimensional, usesViewportUnits, warnUnitOnce, type LengthAxis } from './units'

/**
 * Shape position/size unit resolution.
 *
 * Shape position props (`center`) and size props (`radius`/`width`/`height`/…) may carry a unit —
 * a DimensionalValue `{ value, unit }` or a CSS-like string (`'50vw'`); see utilities/units.ts for
 * the unit list. Positions may be measured from a reference edge via the shape's `origin` prop.
 * Neither the prop `transform` functions nor GPU code can see the live canvas size, so unit → UV
 * conversion and origin resolution happen here, in plain JS, against the current canvas
 * dimensions — and re-run on resize (and, for viewport / rem units, on window resize).
 *
 * Every unit converts to CSS pixels first (`lengthToPx`); each conversion below then turns pixels
 * into the prop's UV space exactly as it always has for `px`. A plain number (and `unit: 'uv'`)
 * keeps its legacy meaning, so existing presets resolve byte-identically.
 *
 * Which props are dimensional, and how each converts, is derived entirely from the shader's
 * `boundingBoxDeclaration.propBindings` (the same source the Design Editor overlay uses), so this
 * stays in lockstep with `bboxPropBindings.ts` in dotcom:
 *   - position-x / position-y  → position prop ({x,y}); px divides by canvasWidth / canvasHeight
 *   - canvas-height            → size prop; px = value / canvasHeight
 *   - half-canvas-height       → size prop; px = value / (2 * canvasHeight)
 *
 * `%` resolves against the containing box — the canvas, until frames exist — and only on props
 * tied to an axis: a width binding (or `ui.dimensionalAxis: 'x'`) measures the box width, a height
 * binding the box height. A prop with no axis (a radius bound to both, a softness) ignores a `%`
 * value and falls back to its default, the way CSS drops a value a property doesn't accept.
 */

// Module-global canvas dimensions, mirroring setColorSpaceMode. The root <Shader> component
// (and the design-editor preset renderer) set these before building uniforms; the renderer
// re-resolves against its own live dimensions on every resize.
let curWidth = 0
let curHeight = 0
export function setShaderCanvasDimensions(width: number, height: number): void {
    if (width > 0) curWidth = width
    if (height > 0) curHeight = height
}
export function getShaderCanvasDimensions(): { width: number; height: number } {
    return { width: curWidth, height: curHeight }
}

export { isDimensionalValue, toDimensional, usesViewportUnits }

export type SizeConversion = 'canvas-height' | 'half-canvas-height' | 'canvas-width' | 'count-canvas-height'

/** A marker-declared size prop: its px → UV conversion and the axis a `%` value measures. */
export interface ExtraSizeProp {
    conversion: SizeConversion
    axis: LengthAxis
}

/** Extra (non-bbox) size props. A bare SizeConversion entry means "no axis". */
export type ExtraSizeProps = Map<string, SizeConversion | ExtraSizeProp>

interface SizePlanEntry {
    conversion: SizeConversion
    axis: LengthAxis
}

export interface DimensionalPlan {
    /** Props that are {x,y} positions (resolved with origin). */
    positionProps: Set<string>
    /** Size props (scalar) keyed to their px→UV conversion. */
    sizeProps: Map<string, SizeConversion>
    /** The axis a `%` value on each size prop measures against ('none' = `%` unsupported). */
    sizeAxes: Map<string, LengthAxis>
}

const SIZE_CONVERSIONS = new Set<string>(['canvas-height', 'half-canvas-height', 'canvas-width', 'count-canvas-height'])

/**
 * Derives the extra (non-bbox) dimensional size props from a component's prop configs.
 * Any prop whose `ui.dimensional` marker is set becomes a scalar size prop keyed to that
 * conversion — this is what lets an ordinary scalar prop (e.g. softness, rounding) opt into
 * resize-stable units with a single declaration, with no per-prop renderer plumbing.
 * `ui.dimensionalAxis` names the axis `%` measures; 'canvas-width' implies 'x'.
 */
export function buildExtraSizeProps(
    props: Record<string, { ui?: { dimensional?: SizeConversion; dimensionalAxis?: 'x' | 'y' } }> | undefined
): Map<string, ExtraSizeProp> | undefined {
    if (!props) return undefined
    let out: Map<string, ExtraSizeProp> | undefined
    for (const [name, cfg] of Object.entries(props)) {
        const d = cfg?.ui?.dimensional
        if (d && SIZE_CONVERSIONS.has(d)) {
            const axis: LengthAxis = cfg.ui?.dimensionalAxis ?? (d === 'canvas-width' ? 'x' : 'none')
            ;(out ??= new Map()).set(name, { conversion: d, axis })
        }
    }
    return out
}

const extraEntry = (v: SizeConversion | ExtraSizeProp): SizePlanEntry =>
    typeof v === 'string' ? { conversion: v, axis: v === 'canvas-width' ? 'x' : 'none' } : v

/**
 * The axis a `%` on a width/height-bound size prop measures. A prop bound to BOTH axes (a circle's
 * radius drives width and height) has no single axis, so `%` is unsupported there.
 */
export function sizeBindingAxis(pb: BoundingBoxPropBindings | undefined, prop: string): LengthAxis {
    const onX = pb?.width?.prop === prop
    const onY = pb?.height?.prop === prop
    return onX && !onY ? 'x' : onY && !onX ? 'y' : 'none'
}

/**
 * Builds the dimensional resolution plan from a shader's bbox prop bindings, optionally merging
 * in `extraSizeProps` (marker-declared scalar props from buildExtraSizeProps).
 * Returns null only when there are no prop bindings AND no extra size props (filters/generators
 * position via the `boundingBox` object instead, resolved separately in uvTransform).
 */
export function buildDimensionalPlan(
    decl: BoundingBoxDeclaration | undefined,
    extraSizeProps?: ExtraSizeProps
): DimensionalPlan | null {
    const pb = decl?.propBindings
    const positionProps = new Set<string>()
    const sizeProps = new Map<string, SizeConversion>()
    const sizeAxes = new Map<string, LengthAxis>()
    if (pb) {
        for (const axis of ['x', 'y', 'width', 'height'] as const) {
            const b = pb[axis]
            if (!b) continue
            if (b.as === 'position-x' || b.as === 'position-y') positionProps.add(b.prop)
            else if (b.as === 'canvas-height' || b.as === 'half-canvas-height') {
                sizeProps.set(b.prop, b.as)
                sizeAxes.set(b.prop, sizeBindingAxis(pb, b.prop))
            }
        }
    }
    if (extraSizeProps) {
        for (const [prop, raw] of extraSizeProps) {
            const e = extraEntry(raw)
            sizeProps.set(prop, e.conversion)
            sizeAxes.set(prop, e.axis)
        }
    }
    return positionProps.size > 0 || sizeProps.size > 0 ? { positionProps, sizeProps, sizeAxes } : null
}

type Edge = 'start' | 'end' | 'center'

// Per-axis decomposition: each axis is independent. left/top → start, right/bottom → end,
// neither (an axis named 'center', e.g. 'center-left' or 'top-center') → center.
// NB: named `resolveOriginEdges` (not `originEdges`) to avoid a top-level identifier clash with
// dotcom's own `originEdges` (utils/bboxOrigin) when a dev bundler hoists both into one scope.
const resolveOriginEdges = (origin: BoundingBoxOrigin): { ox: Edge; oy: Edge } => ({
    ox: origin.includes('left') ? 'start' : origin.includes('right') ? 'end' : 'center',
    oy: origin.includes('top') ? 'start' : origin.includes('bottom') ? 'end' : 'center'
})

// Size-aware origin resolution, Y-down. The stored position is the absolute canvas-UV position of
// the box's anchor point (the corner/edge named by `origin`); the true render centre is recovered by
// offsetting with the box half-extent `he` (per-axis, UV). For the default 'center' origin he is
// zero and resolveOriginCenter is the identity, so centred shapes resolve unchanged. The editor's
// bboxOrigin.ts holds the mirror copy (resolveCenter/storeCenter). Named `resolveOriginCenter` (NOT
// dotcom's `resolveCenter`) to avoid a top-level identifier clash when a dev bundler hoists both
// modules into one scope — the same reason resolveOriginPoint/resolveOriginEdges were renamed.
const resolveOriginCenter = (anchorUV: number, edge: Edge, he: number): number =>
    edge === 'start' ? anchorUV + he : edge === 'end' ? anchorUV - he : anchorUV

/**
 * CSS pixels of a size value that may carry any unit. `uvDim` is what one legacy UV unit measures
 * in pixels for this prop (canvas height for 'canvas-height', …). Returns null when the value is a
 * unit the prop can't take (`%` with no axis) or isn't a length at all.
 */
export function sizeValuePx(v: unknown, uvDim: number, axis: LengthAxis, width: number, height: number): number | null {
    if (typeof v === 'number') return Number.isFinite(v) ? v * uvDim : null
    const d = toDimensional(v)
    if (!d) return null
    if (d.unit === 'uv') return d.value * uvDim
    return lengthToPx(d, axis, { canvasWidth: width, canvasHeight: height })
}

// Pixel size of a width/height axis binding, mirroring dotcom's `bindingToPx` size cases. A value
// may be a plain UV number or a unit-carrying value. Used by boxHalfExtentsUV when a shape's box is
// derived from its width/height propBindings (Circle/Star/RoundedRect) rather than computeBounds.
// `axis` is the axis a `%` measures (see bindingAxis); unsupported / non-length values read as 0.
export function bindingSizePx(binding: { as: string }, v: any, width: number, height: number, axis: LengthAxis = 'none'): number {
    const uvDim = binding.as === 'canvas-height' ? height
        : binding.as === 'half-canvas-height' ? 2 * height
        : binding.as === 'canvas-width' ? width
        : 0
    if (uvDim === 0) return 0
    return sizeValuePx(v, uvDim, axis, width, height) ?? 0
}

/**
 * Per-axis half-extent of a shape's bounding box, in canvas-UV. Used to recover the true render
 * centre from the stored anchor (resolveOriginCenter) and re-store it (storeCenter). Returns zero when no
 * size information is available. `props` may carry raw (unit-carrying) size values — both
 * computeBounds and bindingSizePx handle that.
 */
export function boxHalfExtentsUV(
    decl: BoundingBoxDeclaration | undefined,
    props: Record<string, any>,
    width: number,
    height: number
): { hwx: number; hhy: number } {
    if (decl?.computeBounds) {
        const b = decl.computeBounds(props, width, height)
        return {
            hwx: width > 0 ? (b.widthPx / width) / 2 : 0,
            hhy: height > 0 ? (b.heightPx / height) / 2 : 0
        }
    }
    const pb = decl?.propBindings
    let hwx = 0, hhy = 0
    if (pb?.width && width > 0) hwx = bindingSizePx(pb.width, props[pb.width.prop], width, height, sizeBindingAxis(pb, pb.width.prop)) / (2 * width)
    if (pb?.height && height > 0) hhy = bindingSizePx(pb.height, props[pb.height.prop], width, height, sizeBindingAxis(pb, pb.height.prop)) / (2 * height)
    return { hwx, hhy }
}

/**
 * Resolves one position axis (number | DimensionalValue | length string | keyword) into a UV
 * number, applying the size-aware origin offset `he`. Keyword strings (e.g. 'center') and anything
 * else that isn't a length are returned untouched for the transform to parse.
 */
function resolvePositionAxis(v: any, dim: number, edge: Edge, he: number, axis: 'x' | 'y', width: number, height: number): any {
    let anchorUV: number
    const d = toDimensional(v)
    if (d && d.unit !== 'uv') {
        // Every real unit is a gap INSET from the named edge (resize-stable), NOT an
        // absolute-from-top-left position — a far-edge anchor stored as absolute px would drift on
        // resize. Convert the gap to the absolute anchor UV: end → 1-gap, centre → 0.5+gap,
        // start → gap. (uv values already ARE the absolute anchor fraction, so they stay
        // resize-stable directly.)
        const px = lengthToPx(d, axis, { canvasWidth: width, canvasHeight: height }) ?? 0
        const gap = dim > 0 ? px / dim : 0
        anchorUV = edge === 'end' ? 1 - gap : edge === 'center' ? 0.5 + gap : gap
    } else if (d) {
        anchorUV = d.value
    } else if (typeof v === 'number') {
        anchorUV = v
    } else {
        return v
    }
    return resolveOriginCenter(anchorUV, edge, he)
}

/**
 * Resolves a size value carrying a unit to the prop's UV space, or `undefined` when the prop
 * can't take that unit (a `%` on a prop with no axis — the caller falls back to the default).
 */
function resolveSize(
    propName: string,
    v: DimensionalValue,
    conversion: SizeConversion,
    axis: LengthAxis,
    width: number,
    height: number
): number | undefined {
    if (v.unit === 'uv') return v.value
    const px = lengthToPx(v, axis, { canvasWidth: width, canvasHeight: height })
    if (px == null) {
        warnUnitOnce(`"${v.unit}" isn't supported on "${propName}" — % only works on props tied to an axis (x, y, width, height). Using the default instead.`)
        return undefined
    }
    if (conversion === 'canvas-width') return width > 0 ? px / width : v.value
    if (height <= 0) return v.value
    // Inverse (count) conversion: a length sets the size of one unit, so the resolved count is
    // canvasHeight / px. Guard a zero length so we never divide by zero.
    if (conversion === 'count-canvas-height') return px > 0 ? height / px : v.value
    return conversion === 'canvas-height' ? px / height : px / (2 * height)
}

const hasUnit = (v: unknown): boolean => toDimensional(v) != null

/**
 * Returns a props object where dimensional shape props are resolved to plain UV values
 * (numbers / {x,y} numbers) ready for the existing prop transforms, applying the `origin` prop.
 * A size prop given a unit it can't take is removed, so the consumer's default applies.
 *
 * The default origin ('center') resolves the stored centre as-is (zero half-extent), so a centred
 * shape with plain-number props is returned unchanged (zero overhead) — identical to the legacy
 * default. Non-centre origins recover the render centre from the stored anchor via the box extent.
 */
export function resolveDimensionalProps(
    decl: BoundingBoxDeclaration | undefined,
    rawProps: Record<string, any>,
    width: number,
    height: number,
    extraSizeProps?: ExtraSizeProps
): Record<string, any> {
    const plan = buildDimensionalPlan(decl, extraSizeProps)
    if (!plan) return rawProps

    const origin: BoundingBoxOrigin = (rawProps.origin as BoundingBoxOrigin) ?? 'center'
    const { ox, oy } = resolveOriginEdges(origin)
    const isDefaultOrigin = origin === 'center'
    // Half-extent only matters for non-centre origins; skip the computeBounds/binding work otherwise.
    const he = isDefaultOrigin ? { hwx: 0, hhy: 0 } : boxHalfExtentsUV(decl, rawProps, width, height)

    let out: Record<string, any> | null = null
    const ensure = () => (out ??= { ...rawProps })

    for (const prop of plan.positionProps) {
        const val = rawProps[prop]
        if (!val || typeof val !== 'object') continue
        const hasDim = hasUnit(val.x) || hasUnit(val.y)
        // Identity fast-path: default (centre) origin + no units → leave untouched (legacy behaviour)
        if (isDefaultOrigin && !hasDim) continue
        ensure()[prop] = {
            ...val,
            x: resolvePositionAxis(val.x, width, ox, he.hwx, 'x', width, height),
            y: resolvePositionAxis(val.y, height, oy, he.hhy, 'y', width, height)
        }
    }

    for (const [prop, conversion] of plan.sizeProps) {
        const d = toDimensional(rawProps[prop])
        if (!d) continue
        const resolved = resolveSize(prop, d, conversion, plan.sizeAxes.get(prop) ?? 'none', width, height)
        if (resolved === undefined) delete ensure()[prop]
        else ensure()[prop] = resolved
    }

    return out ?? rawProps
}

/**
 * Resolves a single dimensional prop value to its plain UV equivalent, given the shader
 * declaration, the prop name, the current origin, and live canvas dimensions. Used by the
 * renderer's update + resize paths to re-resolve from stashed raw values. Returns `undefined`
 * when a size prop was given a unit it can't take — the caller substitutes the prop default.
 */
export function resolveDimensionalProp(
    decl: BoundingBoxDeclaration | undefined,
    propName: string,
    rawValue: any,
    origin: BoundingBoxOrigin,
    width: number,
    height: number,
    extraSizeProps?: ExtraSizeProps,
    he?: { hwx: number; hhy: number }
): any {
    const plan = buildDimensionalPlan(decl, extraSizeProps)
    if (!plan) return rawValue
    const { ox, oy } = resolveOriginEdges(origin)
    if (plan.positionProps.has(propName)) {
        if (!rawValue || typeof rawValue !== 'object') return rawValue
        return {
            ...rawValue,
            x: resolvePositionAxis(rawValue.x, width, ox, he?.hwx ?? 0, 'x', width, height),
            y: resolvePositionAxis(rawValue.y, height, oy, he?.hhy ?? 0, 'y', width, height)
        }
    }
    const conversion = plan.sizeProps.get(propName)
    if (!conversion) return rawValue
    const d = toDimensional(rawValue)
    if (!d) return rawValue
    return resolveSize(propName, d, conversion, plan.sizeAxes.get(propName) ?? 'none', width, height)
}

/** Whether a prop is a dimensional (position/size) prop for the given declaration. */
export function isDimensionalProp(
    decl: BoundingBoxDeclaration | undefined,
    propName: string,
    extraSizeProps?: ExtraSizeProps
): boolean {
    const plan = buildDimensionalPlan(decl, extraSizeProps)
    if (!plan) return false
    return plan.positionProps.has(propName) || plan.sizeProps.has(propName)
}

/** Convenience: the set of all dimensional prop names for a declaration. */
export function dimensionalPropNames(
    decl: BoundingBoxDeclaration | undefined,
    extraSizeProps?: ExtraSizeProps
): string[] {
    const plan = buildDimensionalPlan(decl, extraSizeProps)
    if (!plan) return []
    return [...plan.positionProps, ...plan.sizeProps.keys()]
}

// Mark a component (for tooling) — no-op placeholder kept for symmetry with potential future
// per-component caching. Exported types re-export for downstream use.
export type { ComponentDefinition }
