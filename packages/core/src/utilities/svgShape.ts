// Is this shape a custom SVG, and where does it come from?
//
// The engine builds an SVG shape's distance field (and, for extrusions, its exact outline) from the
// SVG itself in real time (`gpu/kit/svgSdfGenerate.ts`). Two props can name the source:
//  - `shape.svgUrl` — the SVG the shape picker uploaded;
//  - `shapeSdfUrl` — the shape's source URL: an SVG (a logo override writes one here, and it wins),
//    or, in presets saved before real-time generation, a pre-generated `.bin` field (kept as the
//    fallback, and preferred over a `svgUrl` that describes a different shape).
// Either makes a shape an SVG shape; every routing decision goes through `isSvgShape`, never
// through `shapeSdfUrl` alone (new uploads don't have one).

const parseCache = new Map<string, Record<string, unknown>>()

function shapeConfig(shape: unknown): Record<string, unknown> {
    if (shape && typeof shape === 'object') return shape as Record<string, unknown>
    if (typeof shape !== 'string' || !shape) return {}
    let cfg = parseCache.get(shape)
    if (!cfg) {
        try { cfg = JSON.parse(shape) as Record<string, unknown> } catch { cfg = {} }
        if (parseCache.size >= 64) parseCache.clear()
        parseCache.set(shape, cfg)
    }
    return cfg
}

/** The SVG URL a `shape` prop (JSON string or object) carries, or ''. */
export function svgUrlOfShape(shape: unknown): string {
    const url = shapeConfig(shape).svgUrl
    return typeof url === 'string' ? url : ''
}

/** True when the shape is a custom SVG: an `svgUrl` in its config, or a legacy `.bin` field. */
export function isSvgShape(shapeSdfUrl: unknown, shape: unknown): boolean {
    return (typeof shapeSdfUrl === 'string' && shapeSdfUrl !== '') || svgUrlOfShape(shape) !== ''
}

/** The URL content bounds are scanned from: `shapeSdfUrl` (an SVG override, or a `.bin` — the
 *  authority when a stale `svgUrl` disagrees) when set, else the shape's `svgUrl`. */
export function svgShapeBoundsUrl(shapeSdfUrl: unknown, shape: unknown): string {
    return (typeof shapeSdfUrl === 'string' && shapeSdfUrl) || svgUrlOfShape(shape)
}
