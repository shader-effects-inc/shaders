/**
 * Group layout gates — the pure measure→arrange→anchor walk (utilities/layout.ts).
 * Covers: column/row stacking with gaps, cross-axis alignment, anchoring (centre + edge with
 * resize-stable offsets), the absolute escape hatch, out-of-flow filters, box-measured media,
 * and nested layout groups (a row of columns).
 */
import { describe, it, expect } from 'vitest'
import { computeLayout, computeTreeLayout, resolveLayoutConfig, layoutBlockSize, computeGroupBlockRects, type LayoutNodeView } from '../../utilities/layout'
import { registerNaturalSize } from '../../utilities/naturalSize'
import { componentDefinition as Circle } from '../../shaders/Circle'
import { componentDefinition as Saturation } from '../../shaders/Saturation'
import { componentDefinition as ImageTexture } from '../../shaders/ImageTexture'
import { componentDefinition as Group } from '../../shaders/Group'

const CW = 1920
const CH = 1080

const circle = (id: string, radius: number, extra: Partial<LayoutNodeView> = {}): LayoutNodeView => ({
    id, componentName: 'Circle', visible: true,
    decl: Circle.boundingBoxDeclaration,
    props: { radius, center: { x: 0.5, y: 0.5 } },
    children: [],
    ...extra,
})

const group = (id: string, layout: any, children: LayoutNodeView[], extra: Partial<LayoutNodeView> = {}): LayoutNodeView => ({
    id, componentName: 'Group', visible: true,
    decl: Group.boundingBoxDeclaration,
    props: {}, layout, children,
    ...extra,
})

describe('computeLayout — column arithmetic', () => {
    const layout = resolveLayoutConfig({ mode: 'column', gap: 24, anchor: 'center' }, CW, CH)!
    const items = [
        { id: 'a', size: { widthPx: 400, heightPx: 100 } },
        { id: 'b', size: { widthPx: 600, heightPx: 50 } },
    ]

    it('stacks with the gap and centres the block', () => {
        // block = 600 × (100+24+50=174), centred → top = (1080-174)/2 = 453
        const p = computeLayout(layout, items, CW, CH)
        expect(p.get('a')).toEqual({ centerXPx: 960, centerYPx: 453 + 50 })
        expect(p.get('b')).toEqual({ centerXPx: 960, centerYPx: 453 + 100 + 24 + 25 })
    })

    it('align start/end place children against the block edge', () => {
        const start = computeLayout({ ...layout, align: 'start' }, items, CW, CH)
        const end = computeLayout({ ...layout, align: 'end' }, items, CW, CH)
        const blockLeft = (CW - 600) / 2
        expect(start.get('a')!.centerXPx).toBe(blockLeft + 200)
        expect(end.get('a')!.centerXPx).toBe(blockLeft + 600 - 200)
    })

    it('edge anchor with px offset is a resize-stable inset', () => {
        const anchored = resolveLayoutConfig({ mode: 'column', gap: 0, anchor: 'bottom-left', anchorOffset: { x: 32, y: 48 } }, CW, CH)!
        const p = computeLayout(anchored, [{ id: 'a', size: { widthPx: 100, heightPx: 60 } }], CW, CH)
        expect(p.get('a')).toEqual({ centerXPx: 32 + 50, centerYPx: CH - 48 - 30 })
    })
})

describe('computeLayout — row', () => {
    it('stacks horizontally, cross-centres vertically', () => {
        const layout = resolveLayoutConfig({ mode: 'row', gap: 10 }, CW, CH)!
        const items = [
            { id: 'a', size: { widthPx: 100, heightPx: 80 } },
            { id: 'b', size: { widthPx: 200, heightPx: 40 } },
        ]
        // block 310 × 80, centred → left (1920-310)/2 = 805, top (1080-80)/2 = 500
        const p = computeLayout(layout, items, CW, CH)
        expect(p.get('a')).toEqual({ centerXPx: 805 + 50, centerYPx: 540 })
        expect(p.get('b')).toEqual({ centerXPx: 805 + 100 + 10 + 100, centerYPx: 540 })
    })
})

describe('computeTreeLayout — the full walk', () => {
    it('a BOXED filter still takes no slot (its box is a clip window, not an element size)', () => {
        const tree = group('g', { mode: 'column', gap: 0 }, [
            circle('c1', 0.2),
            {
                id: 'wave', componentName: 'WaveDistortion', visible: true, requiresChild: true,
                decl: Saturation.boundingBoxDeclaration, props: {},
                boundingBox: { width: { value: 0.3, unit: 'uv' as const }, height: { value: 0.3, unit: 'uv' as const } },
                children: []
            },
            circle('c2', 0.2),
        ])
        const writes = computeTreeLayout([tree], CW, CH)
        expect(writes.map(w => w.id)).toEqual(['c1', 'c2'])
        // Two-slot stack — the boxed filter contributed nothing.
        const w2 = writes[1] as any
        expect(w2.value.y * CH).toBeCloseTo(432 + 216)
    })

    it('a boxed GENERATOR takes a slot (it renders content — a div with a background)', () => {
        const tree = group('g', { mode: 'column', gap: 0 }, [
            {
                id: 'solid', componentName: 'SolidColor', visible: true,
                decl: undefined, props: {},
                boundingBox: { width: { value: 200, unit: 'px' as const }, height: { value: 100, unit: 'px' as const } },
                children: []
            },
            circle('c1', 0.2),
        ])
        const writes = computeTreeLayout([tree], CW, CH)
        const solidWrite = writes.find(w => w.id === 'solid') as any
        expect(solidWrite.kind).toBe('box')
        expect(solidWrite.heightPx).toBe(100)
        // block = 216w × (100+216) centred
        expect(solidWrite.yPx).toBeCloseTo((CH - 316) / 2)
    })

    it('positions measurable children, skips filters (out of flow) without a slot', () => {
        const tree = group('g', { mode: 'column', gap: 0 }, [
            circle('c1', 0.2),  // 216 px square
            { id: 'sat', componentName: 'Saturation', visible: true, requiresChild: true, decl: Saturation.boundingBoxDeclaration, props: {}, children: [] },
            circle('c2', 0.2),
        ])
        const writes = computeTreeLayout([tree], CW, CH)
        expect(writes.map(w => w.id)).toEqual(['c1', 'c2'])
        // block = 216×432 centred: c1 centre y = (1080-432)/2 + 108 = 432
        const w1 = writes[0] as any
        expect(w1.kind).toBe('position')
        expect(w1.prop).toBe('center')
        expect(w1.value.y * CH).toBeCloseTo(432)
        expect(w1.value.x * CW).toBeCloseTo(960)
    })

    it('absolute children keep their own position (no write, no slot)', () => {
        const tree = group('g', { mode: 'column', gap: 0 }, [
            circle('c1', 0.2),
            circle('escaped', 0.2, { absolute: true }),
            circle('c2', 0.2),
        ])
        const writes = computeTreeLayout([tree], CW, CH)
        expect(writes.map(w => w.id)).toEqual(['c1', 'c2'])
        // Only two slots: identical to the two-child stack.
        const w2 = writes[1] as any
        expect(w2.value.y * CH).toBeCloseTo(432 + 216)
    })

    it('position-driven children hold a slot but receive no write', () => {
        const tree = group('g', { mode: 'column', gap: 0 }, [
            circle('c1', 0.2),
            circle('driven', 0.2, { positionDriven: true }),
        ])
        const writes = computeTreeLayout([tree], CW, CH)
        expect(writes.map(w => w.id)).toEqual(['c1'])
        // c1 still measures a TWO-child block (the driven child keeps its slot).
        const w1 = writes[0] as any
        expect(w1.value.y * CH).toBeCloseTo((CH - 432) / 2 + 108)
    })

    it('media with a natural size gets a box write', () => {
        registerNaturalSize('https://x/logo.png', 300, 120)
        const tree = group('g', { mode: 'column', gap: 20 }, [
            { id: 'img', componentName: 'ImageTexture', visible: true, decl: ImageTexture.boundingBoxDeclaration, naturalSizeKey: ImageTexture.naturalSizeKey, props: { url: 'https://x/logo.png' }, children: [] },
            circle('c1', 0.1),
        ])
        const writes = computeTreeLayout([tree], CW, CH)
        const imgWrite = writes.find(w => w.id === 'img') as any
        expect(imgWrite.kind).toBe('box')
        expect(imgWrite.widthPx).toBe(300)
        expect(imgWrite.heightPx).toBe(120)
        // circle r=0.1 → 108px; block = 300 × (120+20+108=248) centred
        expect(imgWrite.yPx).toBeCloseTo((CH - 248) / 2)
        expect(imgWrite.xPx).toBeCloseTo((CW - 300) / 2)
    })

    it('nested layout groups: a column inside a column measures as its block', () => {
        const inner = group('inner', { mode: 'column', gap: 10 }, [
            circle('i1', 0.1),  // 108
            circle('i2', 0.1),  // 108
        ])
        const outer = group('outer', { mode: 'column', gap: 50 }, [
            circle('o1', 0.2),  // 216
            inner,              // block 108 × 226
        ])
        const writes = computeTreeLayout([outer], CW, CH)
        expect(writes.map(w => w.id)).toEqual(['o1', 'i1', 'i2'])
        // outer block: width max(216,108)=216, height 216+50+226=492 → top = (1080-492)/2 = 294
        const o1 = writes[0] as any
        expect(o1.value.y * CH).toBeCloseTo(294 + 108)
        // inner block sits at cursor 216+50=266 → its i1 centre = 294+266+54
        const i1 = writes[1] as any
        expect(i1.value.y * CH).toBeCloseTo(294 + 266 + 54)
        expect(i1.value.x * CW).toBeCloseTo(960)
    })

    it('computeGroupBlockRects: a NESTED group reports its SLOT rect, not its own anchor', () => {
        const inner = group('inner', { mode: 'column', gap: 10 }, [
            circle('i1', 0.1),  // 108
            circle('i2', 0.1),  // 108
        ])
        const outer = group('outer', { mode: 'column', gap: 50 }, [
            circle('o1', 0.2),  // 216
            inner,              // block 108 × 226
        ])
        const rects = computeGroupBlockRects([outer], CW, CH)
        // outer block: 216 × 492 centred → top 294; inner slot starts at cursor 216+50=266
        expect(rects.get('outer')).toEqual({
            leftPx: (CW - 216) / 2, topPx: (CH - 492) / 2, widthPx: 216, heightPx: 492
        })
        const innerRect = rects.get('inner')!
        // Inner sits IN ITS SLOT (cross-centred in the outer block), NOT at its own default
        // centre anchor — that self-anchoring bug drew nested groups dead centre in the editor.
        expect(innerRect.topPx).toBeCloseTo(294 + 266)
        expect(innerRect.leftPx).toBeCloseTo((CW - 108) / 2)
        expect(innerRect.widthPx).toBeCloseTo(108)
        expect(innerRect.heightPx).toBeCloseTo(226)
        expect(innerRect.topPx).not.toBeCloseTo((CH - 226) / 2)  // ≠ self-anchored centre
    })

    it('a layout group nested under a PLAIN group is still found and anchors itself', () => {
        const plain: LayoutNodeView = {
            id: 'plain', componentName: 'Group', visible: true,
            decl: Group.boundingBoxDeclaration, props: {},
            children: [group('g', { mode: 'row', gap: 0 }, [circle('c', 0.1)])],
        }
        const writes = computeTreeLayout([plain], CW, CH)
        expect(writes.map(w => w.id)).toEqual(['c'])
    })

    it('mode none / no layout emits nothing', () => {
        expect(computeTreeLayout([group('g', { mode: 'none' }, [circle('c', 0.2)])], CW, CH)).toEqual([])
        expect(computeTreeLayout([circle('c', 0.2)], CW, CH)).toEqual([])
    })
})

describe('layoutBlockSize', () => {
    it('uv gap resolves against the main axis', () => {
        const layout = resolveLayoutConfig({ mode: 'column', gap: { value: 0.1, unit: 'uv' } }, CW, CH)!
        expect(layout.gapPx).toBe(108)
        const block = layoutBlockSize(layout, [
            { id: 'a', size: { widthPx: 10, heightPx: 10 } },
            { id: 'b', size: { widthPx: 20, heightPx: 10 } },
        ])
        expect(block).toEqual({ widthPx: 20, heightPx: 128 })
    })
})
