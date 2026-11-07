/**
 * The SVG's outline as exact line segments, for the extrusion `sdfFn` to measure its walls against
 * instead of the sampled 2D field.
 *
 * Why: an extruded solid's walls are the 2D field's zero contour swept along z, so any per-texel
 * error in that field becomes a wall that wobbles from texel row to texel row. On a shape turned
 * about one axis every screen row reads one field row, so the wobble shows as streaks — magnified by
 * glass. Near the wall the march now measures the distance to the real outline (the path itself,
 * flattened to 0.005 texel), so the walls are exact geometry; the field stays for the cheap far-field
 * skip, where its error can't show.
 *
 * How: the SVG is parsed (scripts, handlers and links stripped) and mounted invisibly so the browser
 * resolves each element's transform and the viewBox (`getScreenCTM`), in the same letterboxed
 * placement the raster uses. Path data and basic shapes are flattened here (lines and vertices exact;
 * curves and arcs to a texel tolerance — `getPointAtLength` re-walks the path per call, seconds for a
 * logo). Every piece (densified to ≤ 1 texel) is checked against the raster's coverage on both sides:
 * pieces with the shape on one side only are boundary (oriented so the inside is on the left),
 * anything else — an edge buried inside an overlapping shape, an evenodd seam — is dropped. Runs are
 * re-simplified (Douglas–Peucker, which recovers the exact vertices) and bucketed into a coarse grid
 * of per-cell candidate lists.
 *
 * Content the walk can't reproduce exactly (strokes, text, `<use>`, images, masks, clips, filters,
 * patterns) returns null: the extrusion keeps the sampled field. Pure CPU/DOM; no GPU here.
 */

/** Grid cells per side over the field's unit square. */
export const OUTLINE_GRID = 64
/** Segment capacity of the GPU buffer (`vec4f` per segment: a.xy, b.xy in field UV). */
export const OUTLINE_MAX_SEGMENTS = 8192
/** u32 capacity of the grid buffer: `OUTLINE_GRID²` (offset, count) pairs, then the index lists. */
export const OUTLINE_GRID_WORDS = 1 << 18

/** Flattened outline in field UV (y down), the inside on each segment's left (+(t.y, −t.x)). */
export interface SvgOutline {
    /** a.x, a.y, b.x, b.y per segment. */
    segments: Float32Array<ArrayBuffer>
    count: number
}

/** Where the raster put the SVG: canvas px = offset + svg px × scale, on a `size`² canvas. */
export interface OutlinePlacement {
    size: number
    width: number
    height: number
    scale: number
}

/** Curve/arc flattening tolerance and the re-simplify tolerance, in field texels. */
const FLATTEN_TEXELS = 0.005
const SIMPLIFY_TEXELS = 0.005
/** Boundary-test piece length (texels) and the probe offset either side of it. */
const PIECE_TEXELS = 1
const PROBE_TEXELS = 1.5
const MAX_POINTS = 400_000
const UNSUPPORTED = 'text, use, image, mask, clipPath, filter, pattern, foreignObject, switch'

function sanitise(svg: SVGSVGElement): void {
    svg.querySelectorAll('script, foreignObject').forEach((el) => el.remove())
    for (const el of [svg, ...Array.from(svg.querySelectorAll('*'))]) {
        for (const attr of Array.from(el.attributes)) {
            const name = attr.name.toLowerCase()
            if (name.startsWith('on') || name === 'href' || name === 'xlink:href') el.removeAttribute(attr.name)
        }
    }
}

function simplify(pts: number[][], tol: number): number[][] {
    if (pts.length < 3) return pts
    const keep = new Uint8Array(pts.length)
    keep[0] = 1
    keep[pts.length - 1] = 1
    const stack: [number, number][] = [[0, pts.length - 1]]
    while (stack.length) {
        const [i0, i1] = stack.pop()!
        const [ax, ay] = pts[i0]
        const [bx, by] = pts[i1]
        const dx = bx - ax
        const dy = by - ay
        const len = Math.hypot(dx, dy) || 1e-12
        let worst = -1
        let worstD = tol
        for (let i = i0 + 1; i < i1; i++) {
            const [px, py] = pts[i]
            const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (len * len)))
            const dist = Math.hypot(px - ax - dx * t, py - ay - dy * t)
            if (dist > worstD) { worstD = dist; worst = i }
        }
        if (worst >= 0) {
            keep[worst] = 1
            stack.push([i0, worst], [worst, i1])
        }
    }
    return pts.filter((_, i) => keep[i])
}

type Pt = [number, number]

/** Arc (SVG endpoint form) → points after the start, flattened to `tol` (user units). */
function flattenArc(out: Pt[], x1: number, y1: number, rxIn: number, ryIn: number, phiDeg: number, large: boolean, sweep: boolean, x2: number, y2: number, tol: number): void {
    let rx = Math.abs(rxIn)
    let ry = Math.abs(ryIn)
    if (rx < 1e-12 || ry < 1e-12 || (x1 === x2 && y1 === y2)) { out.push([x2, y2]); return }
    const phi = phiDeg * Math.PI / 180
    const cp = Math.cos(phi), sp = Math.sin(phi)
    const dx = (x1 - x2) / 2, dy = (y1 - y2) / 2
    const x1p = cp * dx + sp * dy, y1p = -sp * dx + cp * dy
    const lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if (lam > 1) { const k = Math.sqrt(lam); rx *= k; ry *= k }
    const num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    const den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    let co = Math.sqrt(Math.max(0, num / den))
    if (large === sweep) co = -co
    const cxp = co * rx * y1p / ry, cyp = -co * ry * x1p / rx
    const cx = cp * cxp - sp * cyp + (x1 + x2) / 2, cy = sp * cxp + cp * cyp + (y1 + y2) / 2
    const ang = (ux: number, uy: number, vx: number, vy: number) => Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    const th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    let dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if (!sweep && dth > 0) dth -= 2 * Math.PI
    if (sweep && dth < 0) dth += 2 * Math.PI
    const r = Math.max(rx, ry)
    const step = 2 * Math.acos(Math.max(-1, 1 - tol / r)) || 0.1
    const n = Math.min(4096, Math.max(1, Math.ceil(Math.abs(dth) / step)))
    for (let i = 1; i <= n; i++) {
        const t = th1 + dth * i / n
        const ex = rx * Math.cos(t), ey = ry * Math.sin(t)
        out.push(i === n ? [x2, y2] : [cp * ex - sp * ey + cx, sp * ex + cp * ey + cy])
    }
}

/** Cubic Bézier → points after the start, flattened to `tol` (the second-derivative bound). */
function flattenCubic(out: Pt[], p0: Pt, p1: Pt, p2: Pt, p3: Pt, tol: number): void {
    const ddx = Math.max(Math.abs(p0[0] - 2 * p1[0] + p2[0]), Math.abs(p1[0] - 2 * p2[0] + p3[0]))
    const ddy = Math.max(Math.abs(p0[1] - 2 * p1[1] + p2[1]), Math.abs(p1[1] - 2 * p2[1] + p3[1]))
    const n = Math.min(4096, Math.max(1, Math.ceil(Math.sqrt(6 * Math.hypot(ddx, ddy) / (8 * tol)))))
    for (let i = 1; i <= n; i++) {
        const t = i / n, u = 1 - t
        const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, e = t * t * t
        out.push([a * p0[0] + b * p1[0] + c * p2[0] + e * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + e * p3[1]])
    }
}

/** Path data → closed subpaths (fill closes every subpath), in user units. */
function flattenPathData(dAttr: string, tol: number): Pt[][] {
    const tokens = dAttr.match(/[a-zA-Z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g) ?? []
    const subpaths: Pt[][] = []
    let cur: Pt[] = []
    let x = 0, y = 0, sx = 0, sy = 0
    let lastCtrl: Pt | null = null
    let lastQuad: Pt | null = null
    let cmd = ''
    let i = 0
    const isCmd = (t: string | undefined) => t !== undefined && /^[a-zA-Z]$/.test(t)
    // Arc flags may be packed ("a.84.84 0 00-.419.726"): read one flag character at a time.
    let pendingFlags = ''
    const num = () => Number(tokens[i++])
    const flag = () => {
        if (!pendingFlags) pendingFlags = tokens[i++] ?? '0'
        const f = pendingFlags[0]
        pendingFlags = pendingFlags.slice(1)
        return f === '1'
    }
    const close = () => { if (cur.length > 1) subpaths.push(cur); cur = [] }
    while (i < tokens.length) {
        if (isCmd(tokens[i])) cmd = tokens[i++]
        else if (!cmd) { i++; continue }
        const rel = cmd === cmd.toLowerCase()
        const C = cmd.toUpperCase()
        const ox = rel ? x : 0, oy = rel ? y : 0
        if (C === 'Z') { cur.push([sx, sy]); x = sx; y = sy; close(); lastCtrl = lastQuad = null; continue }
        if (i >= tokens.length || isCmd(tokens[i])) { if (C !== 'Z') cmd = ''; continue }
        let ctrl: Pt | null = null
        let quad: Pt | null = null
        if (C === 'M') {
            close()
            x = num() + ox; y = num() + oy; sx = x; sy = y
            cur = [[x, y]]
            cmd = rel ? 'l' : 'L'
        } else if (C === 'L') { x = num() + ox; y = num() + oy; cur.push([x, y]) }
        else if (C === 'H') { x = num() + ox; cur.push([x, y]) }
        else if (C === 'V') { y = num() + oy; cur.push([x, y]) }
        else if (C === 'C' || C === 'S') {
            const p1: Pt = C === 'C' ? [num() + ox, num() + oy] : (lastCtrl ? [2 * x - lastCtrl[0], 2 * y - lastCtrl[1]] : [x, y])
            const p2: Pt = [num() + ox, num() + oy]
            const p3: Pt = [num() + ox, num() + oy]
            flattenCubic(cur, [x, y], p1, p2, p3, tol)
            ctrl = p2; x = p3[0]; y = p3[1]
        } else if (C === 'Q' || C === 'T') {
            const q: Pt = C === 'Q' ? [num() + ox, num() + oy] : (lastQuad ? [2 * x - lastQuad[0], 2 * y - lastQuad[1]] : [x, y])
            const p3: Pt = [num() + ox, num() + oy]
            flattenCubic(cur, [x, y], [x + 2 / 3 * (q[0] - x), y + 2 / 3 * (q[1] - y)], [p3[0] + 2 / 3 * (q[0] - p3[0]), p3[1] + 2 / 3 * (q[1] - p3[1])], p3, tol)
            quad = q; x = p3[0]; y = p3[1]
        } else if (C === 'A') {
            const rx = num(), ry = num(), rot = num()
            const large = flag(), sweep = flag()
            if (pendingFlags) { tokens.splice(i, 0, pendingFlags); pendingFlags = '' } // "0112" → flags 0 1, then 12
            const x2 = num() + ox, y2 = num() + oy
            flattenArc(cur, x, y, rx, ry, rot, large, sweep, x2, y2, tol)
            x = x2; y = y2
        } else { i++; continue }
        if (cur.length === 0) cur = [[x, y]]
        lastCtrl = ctrl
        lastQuad = quad
    }
    close()
    return subpaths
}

/** A basic shape as closed polylines (user units), or null for an element we don't flatten. */
function flattenElement(el: SVGGeometryElement, tol: number): Pt[][] | null {
    const name = el.nodeName.toLowerCase()
    const len = (a: {baseVal: {value: number}}) => a.baseVal.value
    if (name === 'path') return flattenPathData(el.getAttribute('d') ?? '', tol)
    if (name === 'polygon' || name === 'polyline') {
        const pts = Array.from((el as SVGPolygonElement).points).map((p) => [p.x, p.y] as Pt)
        return pts.length > 2 ? [[...pts, pts[0]]] : []
    }
    if (name === 'circle' || name === 'ellipse') {
        const c = el as SVGCircleElement & SVGEllipseElement
        const cx = len(c.cx), cy = len(c.cy)
        const rx = name === 'circle' ? len(c.r) : len(c.rx)
        const ry = name === 'circle' ? len(c.r) : len(c.ry)
        if (!(rx > 0 && ry > 0)) return []
        const out: Pt[] = [[cx + rx, cy]]
        flattenArc(out, cx + rx, cy, rx, ry, 0, false, true, cx - rx, cy, tol)
        flattenArc(out, cx - rx, cy, rx, ry, 0, false, true, cx + rx, cy, tol)
        return [out]
    }
    if (name === 'rect') {
        const r = el as SVGRectElement
        const x = len(r.x), y = len(r.y), w = len(r.width), h = len(r.height)
        if (!(w > 0 && h > 0)) return []
        let rx = len(r.rx), ry = len(r.ry)
        if (!r.hasAttribute('rx')) rx = ry
        if (!r.hasAttribute('ry')) ry = rx
        rx = Math.min(rx, w / 2); ry = Math.min(ry, h / 2)
        if (!(rx > 0 && ry > 0)) return [[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]]
        const out: Pt[] = [[x + rx, y], [x + w - rx, y]]
        flattenArc(out, x + w - rx, y, rx, ry, 0, false, true, x + w, y + ry, tol)
        out.push([x + w, y + h - ry])
        flattenArc(out, x + w, y + h - ry, rx, ry, 0, false, true, x + w - rx, y + h, tol)
        out.push([x + rx, y + h])
        flattenArc(out, x + rx, y + h, rx, ry, 0, false, true, x, y + h - ry, tol)
        out.push([x, y + ry])
        flattenArc(out, x, y + ry, rx, ry, 0, false, true, x + rx, y, tol)
        return [out]
    }
    return null
}

/**
 * Extract the outline of `svgText` as placed by the raster, checked against its coverage
 * (`rgba`, the raster canvas's pixels). Null when the SVG has content the walk can't reproduce,
 * no boundary at all, or more segments than the buffer holds.
 */
export function extractSvgOutline(svgText: string, rgba: Uint8ClampedArray, place: OutlinePlacement): SvgOutline | null {
    if (typeof document === 'undefined' || typeof DOMParser === 'undefined') return null
    const parsed = new DOMParser().parseFromString(svgText, 'image/svg+xml')
    const svg = parsed.documentElement as unknown as SVGSVGElement
    if (!svg || svg.nodeName.toLowerCase() !== 'svg' || parsed.querySelector('parsererror')) return null
    if (svg.querySelector(UNSUPPORTED)) return null
    sanitise(svg)

    const {size, width, height, scale} = place
    const offX = (size - width * scale) / 2
    const offY = (size - height * scale) / 2
    const alphaAt = (x: number, y: number) => {
        const xi = Math.min(size - 1, Math.max(0, Math.floor(x)))
        const yi = Math.min(size - 1, Math.max(0, Math.floor(y)))
        return rgba[(yi * size + xi) * 4 + 3] / 255
    }

    const host = document.createElement('div')
    host.setAttribute('style', 'position:fixed;left:0;top:0;width:0;height:0;overflow:hidden;opacity:0;pointer-events:none')
    const live = document.importNode(svg, true) as SVGSVGElement
    live.setAttribute('width', `${width}px`)
    live.setAttribute('height', `${height}px`)
    host.appendChild(live)
    document.body.appendChild(host)

    const runs: number[][][] = []
    try {
        const origin = live.getBoundingClientRect()
        const shapes = Array.from(live.querySelectorAll('path, rect, circle, ellipse, polygon, polyline, line')) as SVGGeometryElement[]
        let budget = MAX_POINTS
        for (const el of shapes) {
            if (el.closest('defs, symbol, marker')) continue
            const style = getComputedStyle(el)
            if (style.display === 'none' || style.visibility === 'hidden') continue
            if (style.stroke !== 'none' && parseFloat(style.strokeWidth) > 0 && parseFloat(style.strokeOpacity || '1') > 0) return null
            if (style.fill === 'none' || el.nodeName.toLowerCase() === 'line') continue
            const ctm = el.getScreenCTM()
            if (!ctm) continue
            const unitTexels = Math.sqrt(Math.abs(ctm.a * ctm.d - ctm.b * ctm.c)) * scale
            const polys = flattenElement(el, FLATTEN_TEXELS / Math.max(unitTexels, 1e-9))
            if (!polys) return null
            const toCanvas = ([ux, uy]: Pt): Pt => [
                offX + (ctm.a * ux + ctm.c * uy + ctm.e - origin.left) * scale,
                offY + (ctm.b * ux + ctm.d * uy + ctm.f - origin.top) * scale,
            ]
            for (const poly of polys) {
                budget -= poly.length
                if (budget < 0) return null
                // Densify to ≤ PIECE_TEXELS, test each piece against the coverage either side, and
                // keep the boundary runs, oriented so the shape is on their left.
                let run: Pt[] = []
                let runLeft: boolean | null = null
                const flush = () => {
                    if (run.length >= 2) runs.push(runLeft ? run : run.reverse())
                    run = []
                    runLeft = null
                }
                let prev = toCanvas(poly[0])
                for (let k = 1; k < poly.length; k++) {
                    const next = toCanvas(poly[k])
                    const dx = next[0] - prev[0]
                    const dy = next[1] - prev[1]
                    const len = Math.hypot(dx, dy)
                    if (len < 1e-9) continue
                    const pieces = Math.max(1, Math.ceil(len / PIECE_TEXELS))
                    budget -= pieces
                    if (budget < 0) return null
                    const nx = dy / len
                    const ny = -dx / len
                    for (let j = 0; j < pieces; j++) {
                        const a: Pt = [prev[0] + dx * j / pieces, prev[1] + dy * j / pieces]
                        const b: Pt = j === pieces - 1 ? next : [prev[0] + dx * (j + 1) / pieces, prev[1] + dy * (j + 1) / pieces]
                        const mx = (a[0] + b[0]) / 2
                        const my = (a[1] + b[1]) / 2
                        const aL = alphaAt(mx + nx * PROBE_TEXELS, my + ny * PROBE_TEXELS)
                        const aR = alphaAt(mx - nx * PROBE_TEXELS, my - ny * PROBE_TEXELS)
                        if (Math.abs(aL - aR) < 0.5) { flush(); continue } // not a boundary here
                        const left = aL > aR
                        if (runLeft !== null && left !== runLeft) flush()
                        if (run.length === 0) run.push(a)
                        runLeft = left
                        run.push(b)
                    }
                    prev = next
                }
                flush()
            }
        }
    } finally {
        host.remove()
    }

    const segments = new Float32Array(OUTLINE_MAX_SEGMENTS * 4)
    let count = 0
    for (const run of runs) {
        const pts = simplify(run, SIMPLIFY_TEXELS)
        for (let i = 0; i + 1 < pts.length; i++) {
            if (count >= OUTLINE_MAX_SEGMENTS) return null
            segments.set([pts[i][0] / size, pts[i][1] / size, pts[i + 1][0] / size, pts[i + 1][1] / size], count * 4)
            count++
        }
    }
    return count > 0 ? {segments, count} : null
}

/**
 * Bucket the outline into the grid: each cell lists every segment within `band` (field UV) of it,
 * so any point of the cell within `band` of the outline finds its nearest segment in its own cell.
 * Null if the lists overflow the buffer.
 */
export function buildOutlineGrid(outline: SvgOutline, band: number): Uint32Array<ArrayBuffer> | null {
    const G = OUTLINE_GRID
    const cellSize = 1 / G
    const halfDiag = cellSize * Math.SQRT1_2
    const lists: number[][] = Array.from({length: G * G}, () => [])
    const s = outline.segments
    for (let k = 0; k < outline.count; k++) {
        const ax = s[k * 4], ay = s[k * 4 + 1], bx = s[k * 4 + 2], by = s[k * 4 + 3]
        const x0 = Math.max(0, Math.floor((Math.min(ax, bx) - band) * G))
        const x1 = Math.min(G - 1, Math.floor((Math.max(ax, bx) + band) * G))
        const y0 = Math.max(0, Math.floor((Math.min(ay, by) - band) * G))
        const y1 = Math.min(G - 1, Math.floor((Math.max(ay, by) + band) * G))
        const dx = bx - ax
        const dy = by - ay
        const len2 = dx * dx + dy * dy || 1e-12
        for (let cy = y0; cy <= y1; cy++) {
            for (let cx = x0; cx <= x1; cx++) {
                const px = (cx + 0.5) * cellSize
                const py = (cy + 0.5) * cellSize
                const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2))
                if (Math.hypot(px - ax - dx * t, py - ay - dy * t) <= band + halfDiag) lists[cy * G + cx].push(k)
            }
        }
    }
    const words = new Uint32Array(OUTLINE_GRID_WORDS)
    let cursor = G * G * 2
    for (let c = 0; c < G * G; c++) {
        const list = lists[c]
        if (cursor + list.length > OUTLINE_GRID_WORDS) return null
        words[c * 2] = cursor
        words[c * 2 + 1] = list.length
        words.set(list, cursor)
        cursor += list.length
    }
    return words
}
