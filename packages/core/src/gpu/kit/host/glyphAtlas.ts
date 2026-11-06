/**
 * The glyph atlas: every Text layer in a composition rasters into its own canvas, and all of
 * those rasters share ONE single-channel GPU texture (and so one texture binding — a shader with
 * twenty text layers costs one of the ~16 sampled-texture slots a pass gets, not twenty).
 *
 * Coverage only: the texture is `r8unorm`, written from white-on-transparent rasters with
 * premultiplied alpha, so red = coverage — a quarter of the memory of RGBA. No mip chain: a
 * raster is always re-made at the canvas's own density (see Text), so it is never minified
 * far enough to need one, and the sampler filters it explicitly.
 *
 * Each slot reserves room rounded up to a bucket, so a raster that changes size a little (typing)
 * re-uploads in place; outgrowing its room repacks the whole atlas into a fresh texture. Uploads
 * are synchronous: a capture re-rasters at its own resolution inside the frame it draws, so the
 * atlas has to hold the new raster before that frame samples it.
 */
import type {GpuFragmentParams, GpuMediaTexture, KitTexture} from '../../contract'

/** Where a slot's raster sits, in atlas UV: left, top, width, height. Zero size = nothing. */
export type AtlasRect = [number, number, number, number]

export interface GlyphAtlasSlot {
    /** Show the top-left `width`×`height` of `source` (white glyphs on transparent). */
    set(source: HTMLCanvasElement, width: number, height: number): void
    /** Show nothing. */
    clear(): void
}

export interface GlyphAtlas {
    readonly kit: KitTexture
    /** A slot for one layer's raster; `onRect` hears where it sits whenever that changes. */
    slot(onRect: (rect: AtlasRect) => void): GlyphAtlasSlot
}

/** Clear texels around every raster: wider than any filter tap reaches past a raster's edge. */
const GUTTER = 4
/** Slot room is rounded up to this, so small size changes re-upload in place. */
const BUCKET = 64
const MIN_ATLAS = 256

interface SlotState {
    source: HTMLCanvasElement | null
    w: number
    h: number
    /** Reserved room (excluding gutter). */
    capW: number
    capH: number
    x: number
    y: number
    placed: boolean
    /** The raster changed and must be uploaded. */
    dirty: boolean
    /** The raster shrank inside its room: stale texels must be cleared first. */
    shrank: boolean
    onRect: (rect: AtlasRect) => void
}

const roundUp = (v: number, step: number) => Math.ceil(v / step) * step

/** The composition's glyph atlas (created on first use, shared by every caller). */
export function glyphAtlasFor(params: GpuFragmentParams): GlyphAtlas {
    return params.shared('kit:glyph-atlas', () => createGlyphAtlas(params))
}

/**
 * Shelf-pack slots (tallest first) into the narrowest power-of-two width that keeps the atlas
 * roughly square. Returns null when they can't fit within `maxAxis`.
 */
export function packSlots(sizes: Array<{w: number, h: number}>, maxAxis: number): {width: number, height: number, positions: Array<{x: number, y: number}>} | null {
    const order = sizes.map((s, i) => ({...s, i})).sort((a, b) => b.h - a.h)
    const widest = Math.max(0, ...sizes.map(s => s.w)) + GUTTER * 2
    const area = sizes.reduce((sum, s) => sum + (s.w + GUTTER * 2) * (s.h + GUTTER * 2), 0)
    let width = MIN_ATLAS
    while (width < widest || width * width < area) width *= 2
    if (width > maxAxis) {
        if (widest > maxAxis) return null
        width = maxAxis
    }
    for (;;) {
        const positions = new Array<{x: number, y: number}>(sizes.length)
        let x = 0
        let y = 0
        let shelf = 0
        for (const s of order) {
            const w = s.w + GUTTER * 2
            const h = s.h + GUTTER * 2
            if (x + w > width) {
                x = 0
                y += shelf
                shelf = 0
            }
            positions[s.i] = {x: x + GUTTER, y: y + GUTTER}
            x += w
            shelf = Math.max(shelf, h)
        }
        const height = roundUp(Math.max(y + shelf, 2), BUCKET)
        if (height <= maxAxis && (height <= width * 2 || width >= maxAxis)) return {width, height, positions}
        if (width >= maxAxis) return null
        width = Math.min(width * 2, maxAxis)
    }
}

function createGlyphAtlas(params: GpuFragmentParams): GlyphAtlas {
    const {createMediaTexture, registerMediaTexture, onCleanup, gpu} = params
    const device = gpu?.device as GPUDevice | undefined
    const maxAxis = Math.min(device?.limits?.maxTextureDimension2D ?? 8192, 16384)
    const make = (width: number, height: number) =>
        createMediaTexture({width, height, format: 'r8unorm', label: 'Text:glyph-atlas'})

    let texture: GpuMediaTexture = make(2, 2)
    let disposed = false
    let relayout = false
    const slots = new Set<SlotState>()
    const kit = registerMediaTexture(() => texture.texture)

    onCleanup(() => {
        disposed = true
        texture.destroy()
    })

    const queue = (): GPUQueue | null => {
        const q = device?.queue as GPUQueue | undefined
        return typeof q?.copyExternalImageToTexture === 'function' ? q : null
    }

    const rectOf = (s: SlotState): AtlasRect => (s.source && s.placed
        ? [s.x / texture.width, s.y / texture.height, s.w / texture.width, s.h / texture.height]
        : [0, 0, 0, 0])

    const upload = (s: SlotState, q: GPUQueue) => {
        if (!s.source || !s.placed) return
        const target = texture.unwrap()
        if (s.shrank) {
            // Clear the whole room, so no texel of a larger raster survives next to the new one
            q.writeTexture({texture: target, origin: {x: s.x, y: s.y}}, new Uint8Array(s.capW * s.capH), {bytesPerRow: s.capW}, {width: s.capW, height: s.capH})
        }
        q.copyExternalImageToTexture(
            {source: s.source, origin: {x: 0, y: 0}},
            {texture: target, origin: {x: s.x, y: s.y}, premultipliedAlpha: true},
            {width: s.w, height: s.h},
        )
    }

    const repack = () => {
        const live = [...slots].filter(s => s.source)
        for (const s of live) {
            s.capW = roundUp(s.w, BUCKET)
            s.capH = roundUp(s.h, BUCKET)
        }
        const packed = packSlots(live.map(s => ({w: s.capW, h: s.capH})), maxAxis)
        // Past the device's largest texture: keep what fits by packing the biggest rasters
        // first and dropping the rest (they draw nothing) rather than failing the whole atlas.
        let fitting = live
        let layout = packed
        while (!layout && fitting.length > 1) {
            fitting = [...fitting].sort((a, b) => b.capW * b.capH - a.capW * a.capH).slice(0, -1)
            layout = packSlots(fitting.map(s => ({w: s.capW, h: s.capH})), maxAxis)
        }
        const next = make(layout?.width ?? 2, layout?.height ?? 2)
        texture.destroy()
        texture = next
        for (const s of slots) s.placed = false
        fitting.forEach((s, i) => {
            const at = layout?.positions[i]
            if (!at) return
            s.x = at.x
            s.y = at.y
            s.placed = true
            s.dirty = true
            s.shrank = false
        })
    }

    const flush = () => {
        if (disposed) return
        if (relayout) {
            relayout = false
            repack()
        }
        const q = queue()
        for (const s of slots) {
            if (s.dirty && q) upload(s, q)
            s.dirty = false
            s.shrank = false
            s.onRect(rectOf(s))
        }
    }

    return {
        kit,
        slot(onRect) {
            const s: SlotState = {source: null, w: 0, h: 0, capW: 0, capH: 0, x: 0, y: 0, placed: false, dirty: false, shrank: false, onRect}
            slots.add(s)
            onCleanup(() => slots.delete(s))
            return {
                set(source, width, height) {
                    const w = Math.max(1, Math.round(width))
                    const h = Math.max(1, Math.round(height))
                    if (!s.placed || w > s.capW || h > s.capH) relayout = true
                    else if (w < s.w || h < s.h) s.shrank = true
                    s.source = source
                    s.w = w
                    s.h = h
                    s.dirty = true
                    flush()
                },
                clear() {
                    s.source = null
                    s.placed = false
                    flush()
                },
            }
        },
    }
}
