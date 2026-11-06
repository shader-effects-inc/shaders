import type { DimensionalUnit, DimensionalValue } from '../types'

/**
 * CSS-style length units for dimensional props.
 *
 * A dimensional prop (a shape's position/size, or any prop marked `ui.dimensional`) may carry a
 * unit, written either as a CSS-like string (`'50vw'`, `'12px'`, `'-10%'`) or in the stored object
 * form (`{ value: 50, unit: 'vw' }`). A plain number keeps its original meaning (UV — canvas
 * relative), so every existing preset resolves byte-identically.
 *
 *   px                     CSS pixels
 *   %                      the containing box (the nearest frame; the canvas until frames exist),
 *                          on props tied to an axis only: x / width → its width, y / height → its height
 *   cw  ch  cmin  cmax     1% of the canvas width / height / smaller side / larger side
 *   vw  vh  vmin  vmax     1% of the browser viewport, measured exactly as CSS does
 *   rem                    the root element's font size
 *   uv                     the legacy unit (a bare number); each conversion keeps its own meaning
 *
 * Everything here converts to CSS pixels; the callers in dimensionalProps turn pixels into the
 * shader's UV space with the conversion they already had for `px`.
 */

export const DIMENSIONAL_UNITS: readonly DimensionalUnit[] = [
    'px', '%', 'cw', 'ch', 'cmin', 'cmax', 'vw', 'vh', 'vmin', 'vmax', 'rem', 'uv'
]

const UNIT_SET = new Set<string>(DIMENSIONAL_UNITS)

const VIEWPORT_UNITS = new Set<string>(['vw', 'vh', 'vmin', 'vmax', 'rem'])

// A finite CSS number (`12`, `-0.5`, `.5`, `1e3`) followed by a unit. The unit is required — a
// unitless string isn't a length, except a bare `'0'`, which CSS accepts in any unit.
// The number grammar is deliberately unambiguous (`\d+(?:\.\d*)?`, never `\d+\.?\d*`): with an
// optional dot, a long run of digits can split between two `\d` groups in n ways, so a failing
// match backtracks quadratically (CodeQL js/polynomial-redos).
const LENGTH_RE = /^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(px|%|cw|ch|cmin|cmax|vw|vh|vmin|vmax|rem|uv)\s*$/i

// No real length comes close; caps the work on arbitrary input (e.g. preset JSON) regardless.
const MAX_LENGTH_STRING = 64

/** Whether `v` is a dimensional value in its stored object form. */
export function isDimensionalValue(v: any): v is DimensionalValue {
    return v != null && typeof v === 'object' &&
        typeof v.value === 'number' && typeof v.unit === 'string' && UNIT_SET.has(v.unit)
}

/** Parses a CSS-like length string (`'50vw'`) into its object form; null when it isn't one. */
export function parseLength(s: string): DimensionalValue | null {
    if (s.length > MAX_LENGTH_STRING) return null
    if (s.trim() === '0') return { value: 0, unit: 'px' }
    const m = LENGTH_RE.exec(s)
    if (!m) return null
    const value = Number(m[1])
    if (!Number.isFinite(value)) return null
    return { value, unit: m[2].toLowerCase() as DimensionalUnit }
}

/**
 * Normalizes a raw prop value to a DimensionalValue when it carries a unit (object or string).
 * Plain numbers and everything else return null — callers keep their legacy handling for those.
 */
export function toDimensional(v: unknown): DimensionalValue | null {
    if (isDimensionalValue(v)) return v
    if (typeof v === 'string') return parseLength(v)
    return null
}

/** Whether a raw value carries a unit that depends on the browser viewport or root font size. */
export function usesViewportUnits(v: unknown): boolean {
    const d = toDimensional(v)
    if (d) return VIEWPORT_UNITS.has(d.unit)
    if (v != null && typeof v === 'object') {
        const o = v as { x?: unknown; y?: unknown }
        return usesViewportUnits(o.x) || usesViewportUnits(o.y)
    }
    return false
}

// ── Environment: viewport + root font size ─────────────────────────────────────────────

interface UnitEnvironment {
    viewportWidth: number
    viewportHeight: number
    rem: number
}

// Read lazily and cached: the first vw/vh/rem conversion measures, the renderer invalidates on
// window resize. A per-frame resolve (auto-animate bounds) therefore never touches the DOM.
let cachedEnv: UnitEnvironment | null = null
let envOverride: Partial<UnitEnvironment> | null = null
let viewportProbe: HTMLElement | null = null

/**
 * The CSS viewport. `vw`/`vh` are measured from a hidden fixed `100vw × 100vh` element so they
 * match CSS exactly — including the mobile large-viewport rule, where `innerHeight` changes as the
 * address bar collapses but `100vh` doesn't. Created once, only if a viewport unit is ever used.
 */
function measureViewport(): { width: number; height: number } | null {
    if (typeof document === 'undefined' || !document.body) {
        return typeof window !== 'undefined' && window.innerWidth > 0
            ? { width: window.innerWidth, height: window.innerHeight }
            : null
    }
    if (!viewportProbe || !viewportProbe.isConnected) {
        const el = document.createElement('div')
        el.setAttribute('aria-hidden', 'true')
        el.setAttribute('data-shaders-viewport-probe', '')
        el.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;' +
            'visibility:hidden;pointer-events:none;z-index:-1;contain:strict;'
        document.body.appendChild(el)
        viewportProbe = el
    }
    const w = viewportProbe.offsetWidth
    const h = viewportProbe.offsetHeight
    if (w > 0 && h > 0) return { width: w, height: h }
    return typeof window !== 'undefined' && window.innerWidth > 0
        ? { width: window.innerWidth, height: window.innerHeight }
        : null
}

function measureRem(): number | null {
    if (typeof document === 'undefined' || typeof getComputedStyle === 'undefined' || !document.documentElement) return null
    const px = parseFloat(getComputedStyle(document.documentElement).fontSize)
    return Number.isFinite(px) && px > 0 ? px : null
}

/**
 * The current viewport size and root font size. Without a browser (SSR, workers, headless) the
 * viewport falls back to the canvas size and `rem` to 16.
 */
export function getUnitEnvironment(canvasWidth: number, canvasHeight: number): UnitEnvironment {
    if (!cachedEnv) {
        const vp = measureViewport()
        cachedEnv = {
            viewportWidth: vp?.width ?? 0,
            viewportHeight: vp?.height ?? 0,
            rem: measureRem() ?? 16
        }
    }
    const env = envOverride ? { ...cachedEnv, ...envOverride } : cachedEnv
    return {
        viewportWidth: env.viewportWidth > 0 ? env.viewportWidth : canvasWidth,
        viewportHeight: env.viewportHeight > 0 ? env.viewportHeight : canvasHeight,
        rem: env.rem > 0 ? env.rem : 16
    }
}

/** Drops the cached viewport / root font size so the next conversion re-measures. */
export function invalidateUnitEnvironment(): void {
    cachedEnv = null
}

/** Pins the viewport / root font size (tests, headless renders). Pass null to clear. */
export function setUnitEnvironmentOverride(env: Partial<UnitEnvironment> | null): void {
    envOverride = env
}

// ── Conversion ─────────────────────────────────────────────────────────────────────────

/**
 * Which side of the containing box a `%` value measures against. Props with no axis (a radius,
 * a stroke, a softness) don't accept `%` — CSS refuses it on such properties too.
 */
export type LengthAxis = 'x' | 'y' | 'none'

export interface LengthContext {
    /** Canvas size, CSS px. */
    canvasWidth: number
    canvasHeight: number
    /** The containing box `%` resolves against, CSS px. Defaults to the canvas (no frames yet). */
    boxWidth?: number
    boxHeight?: number
}

const warned = new Set<string>()

/** Warns once per distinct message; a bad unit on a prop animated every frame stays one line. */
export function warnUnitOnce(message: string): void {
    if (warned.has(message)) return
    warned.add(message)
    console.warn(`[shaders] ${message}`)
}

/**
 * Converts a unit-carrying value to CSS pixels. Returns null for `uv` (the caller owns that
 * conversion — it differs per prop) and for a `%` on a prop with no axis (unsupported).
 */
export function lengthToPx(v: DimensionalValue, axis: LengthAxis, ctx: LengthContext): number | null {
    const { canvasWidth: cw, canvasHeight: ch } = ctx
    switch (v.unit) {
        case 'px': return v.value
        case '%': {
            if (axis === 'none') return null
            const box = axis === 'x' ? (ctx.boxWidth ?? cw) : (ctx.boxHeight ?? ch)
            return v.value / 100 * box
        }
        case 'cw': return v.value / 100 * cw
        case 'ch': return v.value / 100 * ch
        case 'cmin': return v.value / 100 * Math.min(cw, ch)
        case 'cmax': return v.value / 100 * Math.max(cw, ch)
        case 'vw':
        case 'vh':
        case 'vmin':
        case 'vmax':
        case 'rem': {
            const env = getUnitEnvironment(cw, ch)
            if (v.unit === 'rem') return v.value * env.rem
            const side = v.unit === 'vw' ? env.viewportWidth
                : v.unit === 'vh' ? env.viewportHeight
                : v.unit === 'vmin' ? Math.min(env.viewportWidth, env.viewportHeight)
                : Math.max(env.viewportWidth, env.viewportHeight)
            return v.value / 100 * side
        }
        default: return null
    }
}
