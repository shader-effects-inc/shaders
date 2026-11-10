/**
 * Framework-side helpers for `capturesDOM` components (HTMLInCanvas).
 *
 * A capturesDOM component renders its children inside a `<div drawable>` wrapper that must live
 * INSIDE the `<Shader>`'s own canvas: html-in-canvas only draws an element into its nearest ancestor
 * canvas, and that canvas also carries `content="drawable"` (set by the shader host in core). Vue's
 * `<Shader>` already renders its slot inside the canvas, so its template renders the wrapper in
 * place; the React, Svelte and Solid `<Shader>`s render children next to the canvas, so their
 * templates portal the wrapper into the canvas found by {@link findShaderCanvas}.
 *
 * Where the browser can't capture HTML (no html-in-canvas: Firefox, Safari, Chromium < 157 without
 * the flag, Electron today; or no WebGPU at all) the HTML is shown as ordinary DOM instead: the
 * wrapper goes into the `.shader` container, absolutely positioned over the canvas. It stays visible
 * and interactive; it just isn't shaded. See {@link findCaptureHost}.
 */

/**
 * Inline style for the drawable wrapper: fill the canvas box so the HTML lays out at the shader's
 * CSS size and the capture maps onto the canvas 1:1. It is also a size container, so the HTML can
 * size itself to the shader with container units (`cqw`, `cqh`, `cqmin`), which keeps a layout
 * proportional when the shader is shown scaled (editor frames, thumbnails).
 */
export const DRAWABLE_WRAPPER_STYLE = 'display:block;width:100%;height:100%;box-sizing:border-box;container-type:size;'

/** The same style as a camelCase object, for JSX `style` props. */
export const DRAWABLE_WRAPPER_STYLE_OBJECT = {
    display: 'block',
    width: '100%',
    height: '100%',
    boxSizing: 'border-box',
    containerType: 'size',
} as const

/**
 * Inline style for the fallback wrapper: the same box as {@link DRAWABLE_WRAPPER_STYLE}, laid over
 * the canvas so the HTML sits exactly where the captured layer would have been.
 */
export const OVERLAY_WRAPPER_STYLE = 'position:absolute;inset:0;display:block;width:100%;height:100%;box-sizing:border-box;container-type:size;'

/** The same style as a camelCase object, for JSX `style` props. */
export const OVERLAY_WRAPPER_STYLE_OBJECT = {
    position: 'absolute',
    inset: 0,
    display: 'block',
    width: '100%',
    height: '100%',
    boxSizing: 'border-box',
    containerType: 'size',
} as const

/**
 * True when this browser can draw HTML into a WebGPU texture (html-in-canvas's WebGPU path, Chrome
 * 157+). A prototype check, so it's synchronous and needs no device. No WebGPU at all → false.
 */
export function supportsHtmlInCanvas(): boolean {
    return (
        typeof HTMLCanvasElement !== 'undefined' &&
        'requestPaint' in HTMLCanvasElement.prototype &&
        typeof GPUQueue !== 'undefined' &&
        'drawElementImageToTexture' in GPUQueue.prototype
    )
}

/** Where a capturesDOM component mounts its HTML wrapper. */
export interface CaptureHost {
    /** The element to append the wrapper to. */
    element: HTMLElement
    /**
     * `true`: `element` is the <Shader> canvas — mark the wrapper `drawable`, style it
     * {@link DRAWABLE_WRAPPER_STYLE}, and hand it to the renderer to capture.
     * `false`: `element` is the `.shader` container — style the wrapper {@link OVERLAY_WRAPPER_STYLE}
     * and show it as plain HTML over the canvas.
     */
    drawable: boolean
}

/**
 * Resolve where a capturesDOM component's HTML goes, from any element it rendered in place (its
 * order marker). In fallback mode the `.shader` container gets `position: relative` when it has no
 * positioning of its own, so the overlay has something to sit in; an author's own positioning
 * (absolute, fixed, sticky…) is left alone.
 */
export function findCaptureHost(from: Element | null | undefined): CaptureHost | null {
    const canvas = findShaderCanvas(from)
    if (!canvas) return null
    if (supportsHtmlInCanvas()) return {element: canvas, drawable: true}
    const container = canvas.parentElement
    if (!container) return null
    const position = getComputedStyle(container).position
    if (!position || position === 'static') container.style.position = 'relative'
    return {element: container, drawable: false}
}

/**
 * The `<Shader>` canvas that owns `from` (any element the component rendered in place, e.g. its
 * order marker): the enclosing renderer canvas when `from` is inside it (Vue), otherwise the canvas
 * that is a direct child of the enclosing `.shader` container (React, Svelte, Solid).
 */
export function findShaderCanvas(from: Element | null | undefined): HTMLCanvasElement | null {
    if (!from) return null
    const enclosing = from.closest('canvas[data-renderer="shaders"]')
    if (enclosing) return enclosing as HTMLCanvasElement
    const host = from.closest('.shader')
    return host?.querySelector<HTMLCanvasElement>(':scope > canvas[data-renderer="shaders"]') ?? null
}
