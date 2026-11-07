// Content-tight bounds for custom-SVG shape effects (core half of WS4 SVG).
//
// `shapeContentExtent` (shapeEffectBounds.ts) is sync and can't load the SDF, so it reads a cache
// filled here by a one-shot async scan, keyed by URL (`svgShapeBoundsUrl`): the SVG itself —
// rasterised with the field generator's placement, its coverage bbox — or, for a preset that only
// has one, the legacy `.bin` field's inside pixels. This mirrors the
// Design Editor's reactive `svgSdfBounds.ts`, but the renderer has no Vue reactivity — so callers
// pass an `onReady` callback (the renderer's per-node dimensional re-resolve) that fires when the
// scan resolves, letting an origin-anchored shape re-anchor to its real content extent (the box
// "snaps in" instead of staying on the full-field fallback).
//
// Scope: half-extents only, assuming the content is CENTRED in the field (generateSdf letterboxes
// the SVG centred). Off-centre content sizes correctly but stays centred on `center` — a noted
// refinement shared with the editor scanner.

// Edge length of the square SVG-SDF data field. Must match the kit's SDF texture size
// (`gpu/kit/sdf.ts` SVG_SDF_SIZE) — the .bin fields this scans are generated at that resolution.
const SVG_SDF_SIZE = 512

export interface SdfContentExtent { hwf: number; hhf: number }  // half-extents in field-UV (full = 0.5)

// `null` = scanned but no inside pixels (full-field fallback). Absent = not yet scanned.
const cache: Record<string, SdfContentExtent | null> = {}
const pending = new Set<string>()
const waiters: Record<string, Array<() => void>> = {}

/**
 * Content half-extents for an SVG SDF url, or null until scanned (caller falls back to full field).
 * Triggers a one-shot async scan on first request; `onReady` (if given) is invoked once when that
 * scan resolves, so the renderer can re-resolve the node's anchored position against the real extent.
 */
export function getSdfContentBounds(url: string, onReady?: () => void): SdfContentExtent | null {
    if (!url) return null
    if (url in cache) return cache[url]
    if (onReady) (waiters[url] ??= []).push(onReady)
    if (!pending.has(url)) { pending.add(url); void scan(url) }
    return null
}

/** Field padding around the SVG (matches `gpu/kit/svgSdfGenerate.ts` and the old `.bin` generator). */
const SVG_PADDING = 0.05

/** Content half-extents of an SVG, letterboxed into a square field exactly as the generator does. */
async function scanSvg(blob: Blob): Promise<SdfContentExtent | null> {
    if (typeof Image === 'undefined') return null
    const objectUrl = URL.createObjectURL(blob)
    try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
            const el = new Image()
            el.onload = () => resolve(el)
            el.onerror = () => reject(new Error('SVG failed to decode'))
            el.src = objectUrl
        })
        const size = SVG_SDF_SIZE
        const canvas: OffscreenCanvas | HTMLCanvasElement = typeof OffscreenCanvas !== 'undefined'
            ? new OffscreenCanvas(size, size)
            : Object.assign(document.createElement('canvas'), { width: size, height: size })
        const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null
        if (!ctx) return null
        const content = size * (1 - 2 * SVG_PADDING)
        const w0 = img.naturalWidth || size
        const h0 = img.naturalHeight || size
        const s = content / Math.max(w0, h0)
        ctx.drawImage(img, (size - w0 * s) / 2, (size - h0 * s) / 2, w0 * s, h0 * s)
        const px = ctx.getImageData(0, 0, size, size).data
        let minX = size, minY = size, maxX = -1, maxY = -1
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (px[(y * size + x) * 4 + 3] >= 128) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        return maxX < 0 ? null : { hwf: (maxX - minX + 1) / size / 2, hhf: (maxY - minY + 1) / size / 2 }
    } finally {
        URL.revokeObjectURL(objectUrl)
    }
}

async function scan(url: string): Promise<void> {
    try {
        const res = await fetch(url, { mode: 'cors' })
        if (!res.ok) { cache[url] = null; return }
        const blob = await res.blob()
        const binBytes = [SVG_SDF_SIZE * SVG_SDF_SIZE * 2, SVG_SDF_SIZE * SVG_SDF_SIZE * 4]
        const looksSvg = blob.type.includes('svg') || (!binBytes.includes(blob.size)
            && (await blob.slice(0, 512).text()).replace(/^\uFEFF/, '').trimStart().startsWith('<'))
        if (looksSvg) {
            cache[url] = await scanSvg(blob.type.includes('svg') ? blob : new Blob([blob], { type: 'image/svg+xml' }))
            return
        }
        const buf = await blob.arrayBuffer()
        const pixelCount = SVG_SDF_SIZE * SVG_SDF_SIZE
        // Decode to signed distance, matching loadSdfFromUrl: compact Uint16 or legacy Float32.
        if (buf.byteLength !== pixelCount * 2 && buf.byteLength !== pixelCount * 4) {
            cache[url] = null  // not a valid SDF binary (e.g. raw SVG) → full-field fallback
            return
        }
        const valueAt: (i: number) => number =
            buf.byteLength === pixelCount * 2
                ? (() => { const u = new Uint16Array(buf); return (i: number) => u[i] / 32767.5 - 1 })()
                : (() => { const f = new Float32Array(buf); return (i: number) => f[i] })()

        // Inside the shape = negative distance. Find the bbox of inside pixels.
        let minX = SVG_SDF_SIZE, minY = SVG_SDF_SIZE, maxX = -1, maxY = -1
        for (let y = 0; y < SVG_SDF_SIZE; y++) {
            const row = y * SVG_SDF_SIZE
            for (let x = 0; x < SVG_SDF_SIZE; x++) {
                if (valueAt(row + x) < 0) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        cache[url] = maxX < 0
            ? null  // no inside pixels → full field
            : { hwf: (maxX - minX + 1) / SVG_SDF_SIZE / 2, hhf: (maxY - minY + 1) / SVG_SDF_SIZE / 2 }
    } catch {
        cache[url] = null
    } finally {
        pending.delete(url)
        const fns = waiters[url]
        if (fns) { delete waiters[url]; for (const fn of fns) fn() }
    }
}

/** True once `url` has been scanned (hit or full-field miss) — i.e. `getSdfContentBounds` is final. */
export function isSdfContentBoundsScanned(url: string): boolean {
    return !!url && url in cache
}
