/**
 * `text`: a line of type as a field the body reads by name. White where the glyphs are, with
 * anti-aliased coverage in alpha. Feed it to other inputs (`blur(label)` for a glow) or use it as
 * a mask, a reveal target, or a spawn shape for particles.
 */
// Maintainer notes. Rasterised on the CPU with Canvas 2D at the canvas resolution and uploaded
// as a media texture; re-rasterised only when the string, the font or the canvas size changes.
// Fonts must already be loaded by the page (the Text shader's measure cache is not involved).
import {isPropRef, withBuilder, type InputNode} from './types'
import type {PropRef} from '../values'

export interface TextInput extends InputNode {
    readonly kind: 'text'
    readonly text: string | PropRef
    readonly font: string
    readonly size: number
    readonly align: 'left' | 'center' | 'right'
    readonly position: readonly [number, number]
    readonly tracking: number
}

/**
 * A line of text as a field: `label(uv).a` is the coverage. `text` is a string or a string
 * prop; `font` a CSS font family (and weight, e.g. `'600 Inter'`); `size` the cap height as a
 * fraction of the canvas height; `position` where its anchor sits in uv.
 *
 * @example
 * ```ts
 * inputs: {label: text(p('title'), {font: '700 Inter', size: 0.2}), glow: blur(label, {radius: 30})},
 * paint: wgsl`return vec4f(vec3f(1.0, 0.5, 0.2) * (label(uv).a + glow(uv).a * 0.8), 1.0);`,
 * ```
 */
export function text(source: string | PropRef, opts: {
    font?: string
    size?: number
    align?: 'left' | 'center' | 'right'
    position?: readonly [number, number]
    tracking?: number
} = {}): TextInput {
    const node = {
        kind: 'text' as const,
        text: source,
        font: opts.font ?? '600 Inter, system-ui, sans-serif',
        size: Math.max(0.01, Math.min(1, opts.size ?? 0.18)),
        align: opts.align ?? 'center',
        position: opts.position ?? [0.5, 0.5],
        tracking: opts.tracking ?? 0,
    }
    return withBuilder(node, (ctx) => {
        let size = ctx.canvas()
        let media = ctx.params.createMediaTexture({width: Math.max(1, size.width), height: Math.max(1, size.height), label: `${ctx.name} (text)`})
        ctx.onCleanup(() => media.destroy())
        let drawn = ''
        let dirty = true
        ctx.onResize((next) => {
            size = next
            dirty = true
        })
        const raster = () => {
            const value = isPropRef(node.text) ? String(ctx.params.getCpuValue(node.text.name) ?? '') : node.text
            if (!dirty && value === drawn) return
            const w = Math.max(1, Math.round(size.width))
            const h = Math.max(1, Math.round(size.height))
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
            const px = Math.round(node.size * h)
            g.clearRect(0, 0, w, h)
            g.fillStyle = '#ffffff'
            g.font = `${px}px ${node.font}`.replace(/^(\d+)px (\d{3}) /, '$2 $1px ')
            if (node.tracking) (g as unknown as {letterSpacing: string}).letterSpacing = `${node.tracking * px}px`
            g.textAlign = node.align
            g.textBaseline = 'middle'
            g.fillText(value, node.position[0] * w, node.position[1] * h)
            media.write(canvas)
            drawn = value
            dirty = false
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
