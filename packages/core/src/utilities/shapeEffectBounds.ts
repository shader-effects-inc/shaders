import type { BoundingBoxDeclaration } from '../types'
import { getSdfContentBounds } from './sdfBounds'
import { is3dShapeType, isSvg3dShapeType, shape3dBoundingRadius, shapeSubPropMax, SVG3D_DEFAULTS } from './sdf3d'
import { isSvgShape, svgShapeBoundsUrl } from './svgShape'

/**
 * Shared bounding-box declaration for the SDF "shape effect" shaders — Glass, Emboss, ThinFilm,
 * Neon, Crystal, SmokeFill. These place an SDF shape via an isotropic `scale`, a `center`, and a
 * `rotation`, with no native width/height. The Design Editor drives them through the
 * propBindings + computeBounds path (like Blob/Ring) so the bounding box scales and rotates them.
 *
 * Geometry: each shader remaps screen UV into the 512² SDF field as
 *   sdfUV = (uv - center) · aspect / scale + 0.5   (x),   (uv - center) / scale + 0.5   (y)
 * so the full field spans a `scale·ch` px **square** ((scale/aspect)·cw = scale·ch). The box
 * therefore has extent `scale·ch` and `scale = widthPx / ch` — mirroring Blob's size↔diameter.
 *
 * NOTE: this box matches the FULL field, which letterboxes the shape with padding — so it's looser
 * than the shape itself. Content-tight bounds (scanning the SDF's zero-crossing extent and locking
 * to the real content aspect ratio) are a future refinement; until then the box is a square that
 * fully contains the shape.
 */
export const shapeEffectBoundingBoxDeclaration: BoundingBoxDeclaration = {
    propBindings: {
        x: { prop: 'center', as: 'position-x' },
        y: { prop: 'center', as: 'position-y' },
        rotation: { prop: 'rotation', as: 'degrees' }
    },
    computeBounds(props, cw, ch) {
        const scale = (props.scale as number) ?? 1
        // typeof guard: px-unit DimensionalValue axes baseline to 0.5, never NaN.
        const cx = typeof props.center?.x === 'number' ? props.center.x : 0.5
        const cy = typeof props.center?.y === 'number' ? props.center.y : 0.5
        // Content-tight bounds: the box hugs the shape's extent within the 512² field instead of
        // the full field. Analytic shapes are centred at 0.5 in the field, so the box centre is the
        // shape centre (no offset). 1 field-UV maps to scale·ch px (the field is square), so a half
        // extent hwf → 2·hwf·scale·ch px. (SVG shapes use the full field, not content-tight bounds.)
        const { hwf, hhf } = shapeContentExtent(props)
        const fieldPx = scale * ch
        return {
            centerXPx: cx * cw,
            centerYPx: cy * ch,
            widthPx: 2 * hwf * fieldPx,
            heightPx: 2 * hhf * fieldPx,
            rotationDeg: (props.rotation as number) ?? 0
        }
    },
    writeBounds(bounds, currentProps, cw, ch) {
        const center = { ...(currentProps.center ?? { x: 0.5, y: 0.5 }), x: bounds.centerXPx / cw, y: bounds.centerYPx / ch }
        const scale = (currentProps.scale as number) ?? 1

        // Non-uniform shapes (rounded rect, ellipse): the box drives the shape's own half-width
        // (radius) and half-height directly, so W/H resize independently. `scale` stays a multiplier.
        const cfg = isNonUniformShapeEffect(currentProps) ? parseShapeConfig(currentProps.shape) : null
        if (cfg && scale > 0) {
            return {
                center,
                rotation: bounds.rotationDeg,
                shape: JSON.stringify({
                    ...cfg,
                    radius: Math.max(0.001, bounds.widthPx / (2 * scale * ch)),
                    height: Math.max(0.001, bounds.heightPx / (2 * scale * ch))
                })
            }
        }

        // Uniform shapes: the box drives the isotropic scale.
        const { hwf } = shapeContentExtent(currentProps)
        return {
            center,
            scale: hwf > 0 ? Math.max(0.01, bounds.widthPx / (2 * hwf * ch)) : 0.01,
            rotation: bounds.rotationDeg
        }
    }
}

/** Parse the serialized analytic shape config (JSON), or null. Cached on the exact string:
 *  the bounds path runs EVERY frame per bounds-consuming node (Repeater's child-bounds refresh),
 *  and the shape JSON rarely changes — re-parsing it per frame was pure GC churn (mirrors the
 *  `lastShapeJson` guard in kit/sdf3d's field setup). Capped small; cleared wholesale on overflow. */
const shapeParseCache = new Map<string, any>()
function parseShapeConfig(shape: any): any {
    if (typeof shape !== 'string') return shape ?? null
    if (shapeParseCache.has(shape)) return shapeParseCache.get(shape)
    let parsed: any = null
    try { parsed = JSON.parse(shape) } catch { parsed = null }
    if (shapeParseCache.size >= 64) shapeParseCache.clear()
    shapeParseCache.set(shape, parsed)
    return parsed
}

/**
 * True when the shape effect's analytic shape has independent width/height (rounded rect, ellipse),
 * so its bounding box should resize non-uniformly (W → radius, H → height) instead of via the
 * isotropic scale. SVG and single-radius shapes (circle/polygon/star/…) stay uniform.
 */
export function isNonUniformShapeEffect(props: Record<string, any>): boolean {
    if (isSvgShape(props.shapeSdfUrl, props.shape)) return false
    const t = parseShapeConfig(props.shape)?.type
    return t === 'roundedRectSDF' || t === 'ellipseSDF'
}

/**
 * The shape's content half-extents within the 512² SDF field (field-UV, where the full field is
 * 0.5). Analytic shapes are bounded by their radius (always-contains — a wrong formula can only be
 * slightly loose, never tighter than the shape). roundedRect/ellipse use radius (half-width) ×
 * height (half-height). Custom SVG uses scanned content bounds: the editor injects reactive
 * `_shapeBounds` (so its overlay snaps in), and the renderer fills `getSdfContentBounds`'s cache
 * (re-resolving the node when its scan resolves) — both content-tight, so anchor/render/overlay
 * agree. Falls back to the full field until whichever scan resolves.
 */
export function shapeContentExtent(props: Record<string, any>): { hwf: number; hhf: number } {
    if (isSvgShape(props.shapeSdfUrl, props.shape)) {
        const sb = props._shapeBounds ?? getSdfContentBounds(svgShapeBoundsUrl(props.shapeSdfUrl, props.shape))
        const base = (sb && typeof sb.hwf === 'number' && typeof sb.hhf === 'number')
            ? { hwf: sb.hwf, hhf: sb.hhf }
            : { hwf: 0.5, hhf: 0.5 }  // not scanned yet → full field (box snaps in on resolve)
        // SVG lifted into 3D (extrude): rotation-safe square from the
        // bounding-sphere radius (rotations may be auto-animated, so the box
        // must contain the silhouette under any orientation).
        const t = props.shapeType
        if (typeof t === 'string' && isSvg3dShapeType(t)) {
            const cfg = parseShapeConfig(props.shape) ?? {}
            const sub = (k: string) => shapeSubPropMax(cfg[k], SVG3D_DEFAULTS[k] ?? 0)
            const br = Math.hypot(base.hwf, base.hhf, sub('depth') * 0.5) + sub('bevel') + 0.02
            return { hwf: br, hhf: br }
        }
        return base
    }
    const cfg: any = parseShapeConfig(props.shape)
    if (!cfg || typeof cfg !== 'object') return { hwf: 0.5, hhf: 0.5 }
    // 3D shapes: bounding-sphere radius contains the silhouette under any rotation
    // (rotations may be auto-animated, so the box must hold for all orientations).
    if (typeof cfg.type === 'string' && is3dShapeType(cfg.type)) {
        const br = shape3dBoundingRadius(cfg)
        return { hwf: br, hhf: br }
    }
    const r = typeof cfg.radius === 'number' ? cfg.radius : 0.35
    switch (cfg.type) {
        case 'roundedRectSDF':
        case 'ellipseSDF': {
            const h = typeof cfg.height === 'number' ? cfg.height : r
            return { hwf: r, hhf: h }
        }
        case 'ringSDF': {
            const t = typeof cfg.thickness === 'number' ? cfg.thickness : 0
            return { hwf: r + t, hhf: r + t }
        }
        case 'trapezoidSDF': {
            // bottomWidth/topWidth are half-widths, height is the half-height (see trapezoidSdf).
            const bw = typeof cfg.bottomWidth === 'number' ? cfg.bottomWidth : 0.3
            const tw = typeof cfg.topWidth === 'number' ? cfg.topWidth : 0.2
            const h = typeof cfg.height === 'number' ? cfg.height : 0.25
            return { hwf: Math.max(bw, tw), hhf: h }
        }
        case 'vesicaSDF': {
            // Lens of two circles offset by d = radius·spread: half-width r(1-spread),
            // half-height r·√(1-spread²).
            const spread = typeof cfg.spread === 'number' ? Math.min(1, Math.max(0, cfg.spread)) : 0.4
            return { hwf: r * (1 - spread), hhf: r * Math.sqrt(Math.max(0, 1 - spread * spread)) }
        }
        case 'teardropSDF': {
            // Bulb radius r at the base, point a distance `height` away; centred, so
            // the half-height is (height + r)/2 and the half-width is the bulb radius.
            const h = typeof cfg.height === 'number' ? cfg.height : 0.4
            return { hwf: r, hhf: (h + r) / 2 }
        }
        case 'parallelogramSDF': {
            // width/height are half-extents; skew adds horizontal reach at the top.
            const w = typeof cfg.width === 'number' ? cfg.width : 0.32
            const h = typeof cfg.height === 'number' ? cfg.height : 0.22
            const sk = typeof cfg.skew === 'number' ? cfg.skew : 0
            return { hwf: w + Math.abs(sk), hhf: h }
        }
        // circle / polygon / star / flower / cross / crescent / heart / arc / unknown:
        // bound by the radius circle (always contains the shape).
        default:
            return { hwf: r, hhf: r }
    }
}

/**
 * Resolve the analytic shape type for the SDF shape-effect shaders: the compile-time
 * `shapeType` prop wins; otherwise the `shape` config (object or JSON-string form) is
 * consulted. Only a non-empty string `type` is trusted — bad JSON, missing fields, or an
 * object-valued `type` fall back to circleSDF instead of leaking into generated sampler
 * names (`analyticSdf_[object Object]`).
 */
export function resolveShapeType(shapeType: string, shape: unknown): string {
    if (shapeType) return shapeType
    let parsed: unknown = shape
    if (typeof shape === 'string') {
        try { parsed = JSON.parse(shape) } catch { return 'circleSDF' }
    }
    if (parsed && typeof parsed === 'object') {
        const t = (parsed as Record<string, unknown>).type
        if (typeof t === 'string' && t) return t
    }
    return 'circleSDF'
}
