import type { BoundingBoxConfig, BoundingBoxDeclaration } from '../types'
import { bindingSizePx, sizeBindingAxis } from './dimensionalProps'
import { boxLengthUV } from './boundingBox'
import { getNaturalSize } from './naturalSize'

/**
 * Layout measurement — "child, how big are you?"
 *
 * The single sizing question a layout pass asks of a layer. One implementation, four answers,
 * tried in order:
 *
 *   1. `computeBounds`   — content-tight measurement (Text run, Blob, Ring, shape effects).
 *   2. `propBindings`    — size props converted through their declared binding (Circle radius…).
 *   3. explicit box      — an authored `boundingBox` on the node metadata (any layer, incl.
 *                          media and resize-fit generators sized deliberately).
 *   4. natural size      — media intrinsic dimensions from the naturalSize registry
 *                          (the `<img naturalWidth>` analog).
 *
 * A layer none of these can answer for (filters, full-frame generators) returns null — it is
 * OUT OF FLOW: no slot in a stack, no contribution to a group's content size.
 *
 * `availableWidthPx` is accepted but currently unused by every measurer — it is the future slot
 * for text wrapping ("how tall are you GIVEN this width"). Callers must pass it today so the
 * signature never changes when wrapping lands.
 */

export interface MeasureInput {
    /** Component type name. Informational only; measurement never looks a type up by name. */
    type?: string
    /** The shader definition's `naturalSizeKey` declaration (media layers only). Passed in by the
     *  caller, who holds the definition, so measurement never reaches into the full shader
     *  registry; doing that pulled every shader definition into any bundle using the renderer. */
    naturalSizeKey?: NaturalSizeKeyDecl
    /** True for filters (requiresChild) — they NEVER answer, even with an explicit box. A
     *  filter's box is a clip WINDOW over the composite below ("where the effect applies"),
     *  not an element size; giving it a flow slot stacks a distortion of its own siblings. */
    requiresChild?: boolean
    /** The shader's runtime bounding-box declaration (with computeBounds, when defined). */
    decl: BoundingBoxDeclaration | undefined
    /** The node's current props (raw — measurers handle DimensionalValues themselves). */
    props: Record<string, any>
    /** The node's authored boundingBox metadata, if any. */
    boundingBox?: Partial<BoundingBoxConfig> | null
    /** Width offered by the layout container. Unused until text wrapping; pass it anyway. */
    availableWidthPx: number
}

export interface MeasuredSize {
    widthPx: number
    heightPx: number
}

/** A shader definition's `naturalSizeKey` declaration (see `gpu/contract.ts`). */
export type NaturalSizeKeyDecl = {fromProp: string} | {fixed: string}

/**
 * Natural-size registry key for a media layer, or null when the layer is not media.
 *
 * Declaration-driven: the shader states how its key is derived via `naturalSizeKey` on its
 * definition, and the caller hands that declaration in with the node.
 */
function mediaKey(decl: NaturalSizeKeyDecl | undefined, props: Record<string, any>): string | null {
    if (!decl) return null
    if ('fixed' in decl) return decl.fixed || null
    const value = props[decl.fromProp]
    return typeof value === 'string' && value ? value : null
}

/** Explicit authored box → px size, when BOTH dimensions were actually set. */
function explicitBoxSize(bbox: Partial<BoundingBoxConfig> | null | undefined, cw: number, ch: number): MeasuredSize | null {
    if (!bbox || typeof bbox !== 'object') return null
    const dimPx = (d: unknown, axis: 'x' | 'y', dim: number): number | null => {
        if (d === undefined || d === null) return null
        if (typeof d === 'object' && typeof (d as { value?: unknown }).value !== 'number') return null
        return boxLengthUV(d, axis, cw, ch) * dim
    }
    const w = dimPx(bbox.width, 'x', cw)
    const h = dimPx(bbox.height, 'y', ch)
    return w != null && h != null && w > 0 && h > 0 ? { widthPx: w, heightPx: h } : null
}

/**
 * Measure a layer for layout. Returns null when the layer has no answerable size (out of flow).
 */
export function measureNode(input: MeasureInput, cw: number, ch: number): MeasuredSize | null {
    const { decl, props, boundingBox } = input
    if (!(cw > 0) || !(ch > 0)) return null

    // Filters are out of flow by nature — a boxed filter is a region-limited effect over the
    // stack (backdrop-filter semantics), never a stacked element.
    if (input.requiresChild) return null

    // 1. Content-tight measurement.
    if (decl?.computeBounds) {
        const b = decl.computeBounds(props, cw, ch)
        if (b.widthPx > 0 && b.heightPx > 0) return { widthPx: b.widthPx, heightPx: b.heightPx }
        return null
    }

    // 2. Size prop bindings.
    const pb = decl?.propBindings
    if (pb?.width && pb?.height) {
        const w = bindingSizePx(pb.width, props[pb.width.prop], cw, ch, sizeBindingAxis(pb, pb.width.prop))
        const h = bindingSizePx(pb.height, props[pb.height.prop], cw, ch, sizeBindingAxis(pb, pb.height.prop))
        if (w > 0 && h > 0) return { widthPx: w, heightPx: h }
        return null
    }

    // 3. Explicit authored box (deliberately-sized media / generators).
    const boxed = explicitBoxSize(boundingBox, cw, ch)
    if (boxed) return boxed

    // 4. Media intrinsic size.
    const key = mediaKey(input.naturalSizeKey, props)
    if (key) {
        const natural = getNaturalSize(key)
        if (natural) return { widthPx: natural.width, heightPx: natural.height }
    }

    // Out of flow.
    return null
}
