import {describe, it, expect} from 'vitest'
import {packSlots} from '@coreroot/gpu/kit/host/glyphAtlas'

/**
 * Glyph atlas packing: every raster gets a place with a clear gutter around it, nothing
 * overlaps, the atlas stays within the device limit and roughly square, and an impossible
 * request reports so instead of overlapping.
 */
const GUTTER = 4

function overlaps(a: {x: number, y: number, w: number, h: number}, b: {x: number, y: number, w: number, h: number}) {
    return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
}

describe('packSlots', () => {
    it('places every raster inside the atlas with gutters and no overlap', () => {
        const sizes = [{w: 640, h: 128}, {w: 64, h: 64}, {w: 1280, h: 320}, {w: 192, h: 64}, {w: 320, h: 192}]
        const packed = packSlots(sizes, 8192)!
        expect(packed).not.toBeNull()
        const boxes = sizes.map((s, i) => ({...packed.positions[i]!, w: s.w, h: s.h}))
        for (const b of boxes) {
            expect(b.x).toBeGreaterThanOrEqual(GUTTER)
            expect(b.y).toBeGreaterThanOrEqual(GUTTER)
            expect(b.x + b.w + GUTTER).toBeLessThanOrEqual(packed.width)
            expect(b.y + b.h + GUTTER).toBeLessThanOrEqual(packed.height)
        }
        // Grown by the gutter, no two rasters touch
        const padded = boxes.map(b => ({x: b.x - GUTTER, y: b.y - GUTTER, w: b.w + GUTTER * 2, h: b.h + GUTTER * 2}))
        for (let i = 0; i < padded.length; i++) {
            for (let j = i + 1; j < padded.length; j++) expect(overlaps(padded[i]!, padded[j]!)).toBe(false)
        }
        expect(packed.height).toBeLessThanOrEqual(packed.width * 2)
    })

    it('stays within the device limit, or reports that it cannot', () => {
        expect(packSlots([{w: 3000, h: 3000}, {w: 3000, h: 3000}], 4096)).toBeNull()
        const packed = packSlots([{w: 4000, h: 900}, {w: 4000, h: 900}], 4096)!
        expect(packed.width).toBeLessThanOrEqual(4096)
        expect(packed.height).toBeLessThanOrEqual(4096)
    })

    it('handles an empty atlas', () => {
        expect(packSlots([], 8192)).toMatchObject({positions: []})
    })
})
