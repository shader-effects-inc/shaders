/**
 * `text`: a line of type as a field the body reads by name. White where the glyphs are, with
 * anti-aliased coverage in alpha. Feed it to other inputs (`blur(label)` for a glow) or use it as
 * a mask, a reveal target, or a spawn shape for particles.
 */
// Maintainer notes. Rasterised on the CPU with Canvas 2D at the canvas resolution and uploaded
// as a media texture; re-rasterised only when the string, a typography value or the canvas size
// changes. With typography on, the layout is the Text component's exactly (measured block, font
// box centred on `center`, per-line alignment, tracking, case, rotation), so the bounding box the
// editor draws from the shared measure (utilities/typography) hugs what is drawn here.
import {isPropRef, withBuilder, type InputNode} from './types'
import type {PropRef} from '../values'
import {loadGoogleFont} from '../../utilities/fonts'
import {bumpMeasureGeneration, measureTextBlock, supportsCanvasLetterSpacing, type TextTransformMode} from '../../utilities/textMeasure'
import {TYPOGRAPHY_FIELDS, type TypographyDefaults, type TypographyField} from '../../utilities/typography'

export interface TextInput extends InputNode {
    readonly kind: 'text'
    readonly text: string | PropRef
    /** The typography controls this text reads from its component (added to it as props). */
    readonly typography: readonly TypographyField[]
    /** The typography's starting values (the added props' defaults, and the fallback for the rest). */
    readonly defaults: Required<Pick<TypographyDefaults, 'fontFamily' | 'fontWeight' | 'fontSize' | 'letterSpacing' | 'textAlign' | 'center'>>
}

/** `'800 Inter, system-ui'` → weight 800, family Inter. */
function parseFont(font: string | undefined): {fontFamily?: string; fontWeight?: number} {
    if (!font) return {}
    const m = /^\s*(?:(\d{3})\s+)?["']?([^,"']+)/.exec(font)
    return m ? {fontWeight: m[1] ? Number(m[1]) : undefined, fontFamily: m[2]!.trim()} : {}
}

/**
 * Text as a field: `label(uv).a` is the coverage (white glyphs, anti-aliased). `source` is a string
 * or a string prop. Feed it to other inputs (`blur(label)` for a glow), use it as a mask or a reveal
 * target, or as a spawn shape for particles (`samplePoints(label)`).
 *
 * By default it carries the Text component's typography: the component gets Font, Weight, Italic,
 * Transform, Size, Letter Spacing, Line Height, Align, Max Width, Center and Rotation controls (the
 * same ones the Text layer has), a bounding box that hugs the ink and scales the font from its
 * corners, and double-click editing of the text prop on the canvas (its panel field is hidden, as on Text, unless the prop sets `ui.hidden: false`). `typography: [...]` keeps only
 * some (an ASCII field might want just `['fontFamily', 'fontWeight']`); `typography: false` turns
 * them off and draws with `font`, `size`, `align`, `position` and `tracking` as given.
 * With typography on, `font` ('800 Inter'), `size` (font size as a fraction of the canvas height),
 * `align`, `position` and `tracking` set the controls' starting values.
 *
 * @example
 * ```ts
 * // An input that feeds another is a const above the definition (an object literal can't
 * // refer to its own keys).
 * const label = text(p('title'), {font: '700 Inter', size: 0.2})
 * // …
 * inputs: {label, glow: blur(label, {radius: 30})},
 * paint: wgsl`return vec4f(vec3f(1.0, 0.5, 0.2) * (label(uv).a + glow(uv).a * 0.8), 1.0);`,
 * ```
 */
export function text(source: string | PropRef, opts: {
    font?: string
    size?: number
    align?: 'left' | 'center' | 'right'
    position?: readonly [number, number]
    tracking?: number
    typography?: boolean | readonly TypographyField[]
} = {}): TextInput {
    const parsed = parseFont(opts.font)
    const typography = opts.typography === false ? []
        : Array.isArray(opts.typography) ? TYPOGRAPHY_FIELDS.filter((f) => (opts.typography as readonly string[]).includes(f))
            : [...TYPOGRAPHY_FIELDS]
    const node = {
        kind: 'text' as const,
        text: source,
        typography,
        defaults: {
            fontFamily: parsed.fontFamily ?? 'Inter',
            fontWeight: parsed.fontWeight ?? 600,
            fontSize: Math.max(0.01, Math.min(1, opts.size ?? 0.18)),
            letterSpacing: opts.tracking ?? 0,
            textAlign: opts.align ?? 'center',
            center: {x: opts.position?.[0] ?? 0.5, y: opts.position?.[1] ?? 0.5},
        },
    }
    return withBuilder(node, (ctx) => {
        let size = ctx.canvas()
        let media = ctx.params.createMediaTexture({width: Math.max(1, size.width), height: Math.max(1, size.height), label: `${ctx.name} (text)`})
        ctx.onCleanup(() => media.destroy())
        let drawnKey = ''
        let fontEpoch = 0
        let loadingFont = ''
        ctx.onResize((next) => {
            size = next
        })
        /** A typography value: the component's prop when this text carries that control, else the default. */
        const read = <T>(field: TypographyField, fallback: T): T => {
            if (!node.typography.includes(field) || !ctx.props[field]) return fallback
            const v = ctx.params.getCpuValue(field)
            return (v ?? fallback) as T
        }
        const raster = () => {
            const value = isPropRef(node.text) ? String(ctx.params.getCpuValue(node.text.name) ?? '') : node.text
            const d = node.defaults
            const family = String(read('fontFamily', d.fontFamily))
            const weight = Number(read('fontWeight', d.fontWeight))
            const italic = Number(read<unknown>('italic', 0)) > 0.5 || read<unknown>('italic', false) === true
            const transform = String(read('textTransform', 'none')) as TextTransformMode
            const fontSizeUV = Number(read('fontSize', d.fontSize))
            const tracking = Number(read('letterSpacing', d.letterSpacing))
            const lineHeight = Number(read('lineHeight', 1.2)) || 1.2
            const align = String(read('textAlign', d.textAlign))
            const widthUV = Number(read('width', 0))
            const c = read<unknown>('center', d.center) as {x?: number; y?: number} | readonly number[]
            const cx = Array.isArray(c) ? Number(c[0]) : Number((c as {x?: number}).x ?? 0.5)
            const rawY = Array.isArray(c) ? Number(c[1]) : Number((c as {y?: number}).y ?? 0.5)
            // The prop's live value is the GPU form (transformPosition flips y: 0 = bottom); the
            // canvas draws top-down, like the default here and the bounding box.
            const fromProp = node.typography.includes('center') && !!ctx.props.center
            const cy = fromProp ? 1 - rawY : rawY
            const rotation = Number(read('rotation', 0))
            const w = Math.max(1, Math.round(size.width))
            const h = Math.max(1, Math.round(size.height))
            const key = [value, family, weight, italic, transform, fontSizeUV, tracking, lineHeight, align, widthUV, cx, cy, rotation, w, h, fontEpoch].join('|')
            if (key === drawnKey) return

            // The real font: draw with the fallback now, again once it has loaded.
            const fontKey = `${family}|${weight}|${italic}`
            if (fontKey !== loadingFont) {
                loadingFont = fontKey
                void loadGoogleFont(family, weight, italic).then(() => {
                    bumpMeasureGeneration()
                    fontEpoch++
                }).catch(() => undefined)
            }

            if (media.width !== w || media.height !== h) {
                const old = media
                media = ctx.params.createMediaTexture({width: w, height: h, label: `${ctx.name} (text)`})
                old.destroy()
            }
            const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : null
            const g = canvas?.getContext('2d') as OffscreenCanvasRenderingContext2D | null | undefined
            if (!canvas || !g) {
                ctx.report('text needs OffscreenCanvas (not available here)')
                return
            }
            g.clearRect(0, 0, w, h)
            const fontSizePx = fontSizeUV * h
            if (value.trim() && fontSizePx > 0) {
                const wrapPx = widthUV * w
                const m = measureTextBlock({text: value, fontFamily: family, fontWeight: weight, italic, letterSpacingEm: tracking, textTransform: transform},
                    lineHeight, wrapPx > 0 ? wrapPx * 100 / fontSizePx : 0)
                const f = fontSizePx / 100
                const contentW = (wrapPx > 0 ? Math.max(wrapPx * 100 / fontSizePx, m.widthPer100) : m.widthPer100) * f
                const blockH = Math.max(fontSizePx * lineHeight, m.blockHeightPer100 * f)
                g.fillStyle = '#ffffff'
                g.font = `${italic ? 'italic ' : ''}${weight} ${fontSizePx}px "${family}", sans-serif`
                if (supportsCanvasLetterSpacing()) (g as unknown as {letterSpacing: string}).letterSpacing = `${tracking * fontSizePx}px`
                g.textAlign = 'left'
                g.textBaseline = 'alphabetic'
                // The Text component's layout: the FONT-box block centred on `center`, lines at
                // `lineHeight` pitch, each placed by `align` within the content width.
                g.translate(cx * w, cy * h)
                if (rotation) g.rotate(rotation * Math.PI / 180)
                const top = -blockH / 2
                const left = -contentW / 2
                for (let i = 0; i < m.lines.length; i++) {
                    const line = m.lines[i]!
                    if (!line.trim()) continue
                    const lineW = m.lineWidthsPer100[i]! * f
                    const x = align === 'left' ? left : align === 'right' ? left + contentW - lineW : left + (contentW - lineW) / 2
                    g.fillText(line, x, top + m.ascentPer100 * f + i * m.pitchPer100 * f)
                }
            }
            media.write(canvas)
            drawnKey = key
        }
        return {
            outputs: {'': {current: () => media.texture, premultiplied: false, format: 'rgba8unorm', size: () => ({width: media.width, height: media.height})}},
            steps() {
                raster()
                return []
            },
        }
    })
}

/**
 * Where particles should appear: inside a source's opaque area (a `text`, a layer prop, a
 * `pass`). Hand it to `simulate.particles` as `spawn`.
 *
 * @example
 * ```ts
 * inputs: {
 *   logo: text('SHADERS', {size: 0.3}),
 *   dust: simulate.particles({count: 30000, spawn: samplePoints(logo), forces: [{spring: 3}, {cursor: 1}, {drag: 0.6}]}),
 * },
 * ```
 */
export function samplePoints(source: InputNode | PropRef): {readonly fromShape: InputNode | PropRef} {
    return {fromShape: source}
}
