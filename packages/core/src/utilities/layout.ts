import type { BoundingBoxConfig, BoundingBoxDeclaration, BoundingBoxOrigin, DimensionalValue, LayoutConfig } from '../types'
import { measureNode, type MeasuredSize } from './measure'
import { lengthToPx, toDimensional } from './units'

/**
 * Pure layout arithmetic — measure results in, absolute centre positions out.
 *
 * One pass, one direction: measure → arrange (stack along the main axis with a gap, align on
 * the cross axis) → anchor (pin the whole block on the canvas). No reflow, no wrapping, no
 * grow/shrink — children are rigid boxes (text does not wrap yet; the measure contract already
 * carries availableWidth for the day it does).
 *
 * All values are canvas pixels. The renderer (or any host) converts the returned centres into
 * each child's native position prop.
 */

export interface LayoutItem {
    /** Caller's identifier for the child (node id). */
    id: string
    size: MeasuredSize
}

export interface LayoutPlacement {
    centerXPx: number
    centerYPx: number
}

export interface ResolvedLayout {
    mode: 'column' | 'row'
    gapPx: number
    align: 'start' | 'center' | 'end'
    anchor: BoundingBoxOrigin
    anchorOffsetX: number
    anchorOffsetY: number
}

/**
 * Gap value → px along the main axis. Number = px (the CSS-shaped default); a `uv` value is a
 * fraction of the main axis; any other unit (incl. a length string) converts as CSS does, with `%`
 * measuring the main axis.
 */
function gapToPx(gap: number | DimensionalValue | string | undefined, axis: 'x' | 'y', cw: number, ch: number): number {
    if (gap == null) return 0
    if (typeof gap === 'number') return gap
    const d = toDimensional(gap)
    if (!d) return 0
    if (d.unit === 'uv') return d.value * (axis === 'y' ? ch : cw)
    return lengthToPx(d, axis, { canvasWidth: cw, canvasHeight: ch }) ?? 0
}

/** Fill defaults; null when the config doesn't activate layout. */
export function resolveLayoutConfig(cfg: LayoutConfig | undefined, cw: number, ch: number): ResolvedLayout | null {
    if (!cfg || (cfg.mode !== 'column' && cfg.mode !== 'row')) return null
    return {
        mode: cfg.mode,
        gapPx: Math.max(0, gapToPx(cfg.gap, cfg.mode === 'column' ? 'y' : 'x', cw, ch)),
        align: cfg.align ?? 'center',
        anchor: cfg.anchor ?? 'center',
        anchorOffsetX: cfg.anchorOffset?.x ?? 0,
        anchorOffsetY: cfg.anchorOffset?.y ?? 0,
    }
}

// Anchor-edge decomposition (same convention as bounding-box origins).
type Edge = 'start' | 'end' | 'center'
const anchorEdges = (origin: BoundingBoxOrigin): { ox: Edge; oy: Edge } => ({
    ox: origin.includes('left') ? 'start' : origin.includes('right') ? 'end' : 'center',
    oy: origin.includes('top') ? 'start' : origin.includes('bottom') ? 'end' : 'center',
})

/**
 * Block top-left along one axis: pin to the named edge with a resize-stable px gap-inset
 * (start → gap from the leading edge, end → gap from the trailing edge, center → centred + gap).
 */
const anchorTopLeft = (edge: Edge, blockSize: number, canvasDim: number, offsetPx: number): number =>
    edge === 'start' ? offsetPx
        : edge === 'end' ? canvasDim - blockSize - offsetPx
            : (canvasDim - blockSize) / 2 + offsetPx

/** The block's own size: main axis = stacked sizes + gaps, cross axis = widest child. */
export function layoutBlockSize(layout: ResolvedLayout, items: LayoutItem[]): MeasuredSize {
    let main = 0
    let cross = 0
    for (const item of items) {
        const m = layout.mode === 'column' ? item.size.heightPx : item.size.widthPx
        const c = layout.mode === 'column' ? item.size.widthPx : item.size.heightPx
        main += m
        cross = Math.max(cross, c)
    }
    if (items.length > 1) main += layout.gapPx * (items.length - 1)
    return layout.mode === 'column'
        ? { widthPx: cross, heightPx: main }
        : { widthPx: main, heightPx: cross }
}

/**
 * Arrange + anchor. Items stack in array order (the caller passes render order); the returned
 * map holds each child's absolute render CENTRE in canvas px.
 */
export function computeLayout(
    layout: ResolvedLayout,
    items: LayoutItem[],
    cw: number,
    ch: number
): Map<string, LayoutPlacement> {
    const placements = new Map<string, LayoutPlacement>()
    if (items.length === 0) return placements

    const block = layoutBlockSize(layout, items)
    const { ox, oy } = anchorEdges(layout.anchor)
    const blockLeft = anchorTopLeft(ox, block.widthPx, cw, layout.anchorOffsetX)
    const blockTop = anchorTopLeft(oy, block.heightPx, ch, layout.anchorOffsetY)

    // Cross-axis position of a child within the block.
    const crossPos = (childCross: number, blockCross: number): number =>
        layout.align === 'start' ? 0
            : layout.align === 'end' ? blockCross - childCross
                : (blockCross - childCross) / 2

    let cursor = 0
    for (const item of items) {
        if (layout.mode === 'column') {
            const x = blockLeft + crossPos(item.size.widthPx, block.widthPx) + item.size.widthPx / 2
            const y = blockTop + cursor + item.size.heightPx / 2
            placements.set(item.id, { centerXPx: x, centerYPx: y })
            cursor += item.size.heightPx + layout.gapPx
        } else {
            const x = blockLeft + cursor + item.size.widthPx / 2
            const y = blockTop + crossPos(item.size.heightPx, block.heightPx) + item.size.heightPx / 2
            placements.set(item.id, { centerXPx: x, centerYPx: y })
            cursor += item.size.widthPx + layout.gapPx
        }
    }
    return placements
}

// ─── Tree layout ──────────────────────────────────────────────────────────────
//
// The renderer-independent walk: given a minimal view of the node tree, compute every write a
// layout pass must make. Pure — the renderer (or a test) builds the views and applies the writes.

/** Minimal node view the layout walk needs. Children in RENDER ORDER. */
export interface LayoutNodeView {
    id: string
    componentName: string
    visible: boolean
    /** Per-child escape hatch — absolute children are out of flow. */
    absolute?: boolean
    /** Filters (requiresChild) never take a flow slot — even with an explicit box. */
    requiresChild?: boolean
    /** Group layout config, when this node is a Group. */
    layout?: LayoutConfig
    decl: BoundingBoxDeclaration | undefined
    /** Raw prop snapshot (pre-transform values, _rawDimensional where present). */
    props: Record<string, any>
    boundingBox?: Partial<BoundingBoxConfig> | null
    /** True when a mouse/map driver owns the position prop — the slot is kept, the write skipped. */
    positionDriven?: boolean
    children: LayoutNodeView[]
}

/** One write the layout pass wants applied. */
export type LayoutWrite =
    | {
        /** Write the child's native position prop (absolute canvas-UV render centre). */
        kind: 'position'
        id: string
        prop: string
        value: { x: number; y: number }
    }
    | {
        /** Position a box-measured child (media/generator): full box, px, top-left origin. */
        kind: 'box'
        id: string
        xPx: number
        yPx: number
        widthPx: number
        heightPx: number
    }

/**
 * A child's size for layout. A nested layout group measures as its own content block
 * (recursion); anything else answers through measureNode.
 */
export function measureChild(view: LayoutNodeView, cw: number, ch: number): MeasuredSize | null {
    const nested = resolveLayoutConfig(view.layout, cw, ch)
    if (nested) {
        const items = flowItems(view, cw, ch)
        if (items.length === 0) return null
        return layoutBlockSize(nested, items)
    }
    return measureNode({
        type: view.componentName,
        decl: view.decl,
        props: view.props,
        boundingBox: view.boundingBox,
        requiresChild: view.requiresChild,
        availableWidthPx: cw,
    }, cw, ch)
}

function flowItems(group: LayoutNodeView, cw: number, ch: number): Array<LayoutItem & { view: LayoutNodeView }> {
    const items: Array<LayoutItem & { view: LayoutNodeView }> = []
    for (const child of group.children) {
        if (child.visible === false || child.absolute) continue
        const size = measureChild(child, cw, ch)
        if (size) items.push({ id: child.id, size, view: child })
    }
    return items
}

/**
 * The offset between a shape's VISUAL box centre and its stored centre prop, when its
 * computeBounds declares one (Text: the ink box sits off the font-box centre the renderer
 * uses). Zero for centred shapes (Blob, Ring, shape effects).
 */
export function computeBoundsShift(view: LayoutNodeView, cw: number, ch: number): { x: number; y: number } {
    if (!view.decl?.computeBounds) return { x: 0, y: 0 }
    const b = view.decl.computeBounds(view.props, cw, ch)
    const cx = typeof view.props.center?.x === 'number' ? view.props.center.x : 0.5
    const cy = typeof view.props.center?.y === 'number' ? view.props.center.y : 0.5
    return { x: b.centerXPx - cx * cw, y: b.centerYPx - cy * ch }
}

/** Placed content-block rect of a layout group (canvas px, top-left). */
export interface GroupBlockRect {
    leftPx: number
    topPx: number
    widthPx: number
    heightPx: number
}
type RectCollector = Map<string, GroupBlockRect>

/** Emit the writes that place one child at an absolute centre. */
function placeChild(view: LayoutNodeView, size: MeasuredSize, centerXPx: number, centerYPx: number, cw: number, ch: number, out: LayoutWrite[], rects?: RectCollector): void {
    // Nested layout group: place ITS children around the given centre (anchor config overridden
    // by the parent's slot), recursively.
    const nested = resolveLayoutConfig(view.layout, cw, ch)
    if (nested) {
        placeGroup(view, nested, cw, ch, out, { centerXPx, centerYPx }, rects)
        return
    }
    const posProp = view.decl?.propBindings?.x?.prop ?? view.decl?.propBindings?.y?.prop
    if (posProp) {
        if (!view.positionDriven) {
            // The slot centre targets the VISUAL box; invert any computeBounds shift so the
            // stored centre puts the ink there (Text), identity for centred shapes.
            const shift = computeBoundsShift(view, cw, ch)
            out.push({ kind: 'position', id: view.id, prop: posProp, value: { x: (centerXPx - shift.x) / cw, y: (centerYPx - shift.y) / ch } })
        }
        return
    }
    // Box-measured child (media/generator): drive its box outright.
    out.push({
        kind: 'box',
        id: view.id,
        xPx: centerXPx - size.widthPx / 2,
        yPx: centerYPx - size.heightPx / 2,
        widthPx: size.widthPx,
        heightPx: size.heightPx,
    })
}

function placeGroup(
    group: LayoutNodeView,
    layout: ResolvedLayout,
    cw: number,
    ch: number,
    out: LayoutWrite[],
    /** Parent-imposed block centre (nested groups); absent → anchor per config. */
    at?: { centerXPx: number; centerYPx: number },
    rects?: RectCollector,
    /** Precomputed flow items — pass when the caller already measured them (avoids remeasuring). */
    precomputedItems?: Array<LayoutItem & { view: LayoutNodeView }>
): void {
    const items = precomputedItems ?? flowItems(group, cw, ch)
    if (items.length === 0) return
    let effective = layout
    if (at) {
        // Convert the imposed centre to an equivalent top-left anchor: pin to 'top-left' with the
        // block's top-left as the offset — reuses the exact anchoring math.
        const block = layoutBlockSize(layout, items)
        effective = {
            ...layout,
            anchor: 'top-left',
            anchorOffsetX: at.centerXPx - block.widthPx / 2,
            anchorOffsetY: at.centerYPx - block.heightPx / 2,
        }
    }
    // Record where this group's block actually LANDED — for a nested group that is the
    // parent-imposed slot, not its own anchor config (the Design Editor draws this rect).
    if (rects) {
        const block = layoutBlockSize(effective, items)
        const { ox, oy } = anchorEdges(effective.anchor)
        rects.set(group.id, {
            leftPx: anchorTopLeft(ox, block.widthPx, cw, effective.anchorOffsetX),
            topPx: anchorTopLeft(oy, block.heightPx, ch, effective.anchorOffsetY),
            widthPx: block.widthPx,
            heightPx: block.heightPx,
        })
    }
    const placements = computeLayout(effective, items, cw, ch)
    for (const item of items) {
        const p = placements.get(item.id)
        if (p) placeChild(item.view, item.size, p.centerXPx, p.centerYPx, cw, ch, out, rects)
    }
}

/**
 * Inverse of the block anchoring: given a dragged block top-left, the anchorOffset that puts it
 * there under the group's current anchor (px, resize-stable gap-inset convention). Lets the
 * editor translate "user dragged the group's block box" into layout config.
 */
export function anchorOffsetForBlockTopLeft(
    group: LayoutNodeView,
    leftPx: number,
    topPx: number,
    cw: number,
    ch: number
): { x: number; y: number } | null {
    const layout = resolveLayoutConfig(group.layout, cw, ch)
    if (!layout) return null
    const items = flowItems(group, cw, ch)
    if (items.length === 0) return null
    const block = layoutBlockSize(layout, items)
    const { ox, oy } = anchorEdges(layout.anchor)
    const invert = (edge: Edge, pos: number, size: number, dim: number): number =>
        edge === 'start' ? pos
            : edge === 'end' ? dim - pos - size
                : pos - (dim - size) / 2
    return {
        x: invert(ox, leftPx, block.widthPx, cw),
        y: invert(oy, topPx, block.heightPx, ch),
    }
}

/**
 * Walk the whole tree and compute every layout write. Only TOP-LEVEL layout groups anchor
 * themselves; nested layout groups are placed by their parent's flow. Non-layout subtrees are
 * still traversed (a layout group may sit anywhere in the tree).
 */
export function computeTreeLayout(roots: LayoutNodeView[], cw: number, ch: number, rects?: Map<string, GroupBlockRect>): LayoutWrite[] {
    const out: LayoutWrite[] = []
    const walk = (view: LayoutNodeView): void => {
        const layout = resolveLayoutConfig(view.layout, cw, ch)
        if (layout) {
            // Measure the flow ONCE — placeGroup reuses these items, and their ids decide which
            // children still need walking (re-calling measureChild here remeasured every nested
            // group subtree and Text block a second time per pass).
            const items = flowItems(view, cw, ch)
            placeGroup(view, layout, cw, ch, out, undefined, rects, items)
            // Children the flow did NOT place (absolute escapes, unmeasurable layers) may still
            // contain their own layout groups — keep walking those. Flow-placed children were
            // handled (recursively) by placeGroup.
            const placed = new Set(items.map(i => i.id))
            for (const child of view.children) {
                if (child.visible === false) continue
                if (!placed.has(child.id)) walk(child)
            }
            return
        }
        for (const child of view.children) {
            if (child.visible !== false) walk(child)
        }
    }
    for (const root of roots) {
        if (root.visible !== false) walk(root)
    }
    return out
}

/**
 * The PLACED content-block rect of every active layout group in the tree — top-level groups
 * anchored by their own config, NESTED groups where their parent's flow put them. This is what
 * the Design Editor draws when a layout group is selected (a nested row's box must sit in its
 * slot in the parent column, not at its own default anchor).
 */
export function computeGroupBlockRects(
    roots: LayoutNodeView[],
    cw: number,
    ch: number
): Map<string, GroupBlockRect> {
    const rects = new Map<string, GroupBlockRect>()
    computeTreeLayout(roots, cw, ch, rects)
    return rects
}
