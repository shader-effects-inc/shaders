import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/porters'
import {call, vec4, debugWarn} from '@coreroot/gpu/porters'
import {defineShader} from '@coreroot/std'
import {tonemap} from '@coreroot/gpu/kit'
import {createSwappableMediaTexture, type SwappableMediaTexture} from '@coreroot/gpu/kit/host/mediaLifecycle'

export interface ComponentProps {}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Host recipe: DOM capture via the WICG html-in-canvas API (shipped in Chrome 157). Pure CPU
// machinery; the GPU tail below only samples the captured texture.
//
// The model (https://github.com/WICG/html-in-canvas):
//   - `<canvas content="drawable">` lays out its descendants (they are NOT painted to the page);
//   - a descendant carrying the `drawable` attribute gets its rendering recorded as a snapshot on
//     every rendering update;
//   - the canvas fires `paint` (listing the `changedElements`) after the browser's Paint step;
//     `requestPaint()` forces one even when nothing changed;
//   - `GPUQueue.drawElementImageToTexture({source}, {texture, size})` re-rasterises the latest
//     snapshot (a vector record, so `size` gives crisp output at any resolution) into a texture
//     with RENDER_ATTACHMENT | COPY_DST usage;
//   - descendants are not hit-testable and have no accessibility geometry until
//     `canvas.updateElementGeometry(el, {canvasTransform})` says where they are drawn (CSS px).
//
// Our shape: the framework template renders the component's children inside a `<div drawable>`
// wrapper sized 100% × 100%, placed INSIDE the `<Shader>`'s own canvas (the WebGPU canvas — the
// same arrangement as the WICG WebGPU demo). So the HTML lays out at the shader's real CSS size, the
// capture is rasterised at the canvas's backing-buffer size (one texel per output pixel, sampled at
// the screen UV), and the wrapper maps onto the canvas 1:1, which makes its content clickable,
// focusable and selectable where it appears (identity canvas transform).
//
// Without html-in-canvas (or WebGPU) the templates never put the wrapper in the canvas: they lay it
// over the canvas as plain HTML instead (`findCaptureHost`), and this host sees no drawable wrapper
// and leaves the layer transparent.
// ═══════════════════════════════════════════════════════════════════════════════════════

/** `HTMLCanvasElement` additions from html-in-canvas (not yet in lib.dom). */
interface DrawableCanvas extends HTMLCanvasElement {
    requestPaint(): void
    updateElementGeometry(element: Element, options?: {canvasTransform?: DOMMatrixInit}): void
}

/** `GPUQueue` addition from html-in-canvas (not yet in @webgpu/types). */
interface ElementImageQueue {
    drawElementImageToTexture(
        source: {source: Element},
        destination: {texture: GPUTexture; size: {width: number; height: number}},
    ): void
}

/** `CanvasPaintEvent` — `changedElements` lists the drawable descendants whose snapshot changed. */
type CanvasPaintEvent = Event & {readonly changedElements?: readonly Element[]}

/**
 * Every HTMLInCanvas composition sharing one `<Shader>` canvas holds a reference on its
 * `content="drawable"` attribute. The pipeline cache keeps up to four compositions alive and
 * disposes them only on eviction, so a recompose never drops the count to zero (dropping the
 * attribute would discard every snapshot and blank the layer for a frame or two). The attribute's
 * prior value is restored when the last holder lets go (renderer teardown).
 */
const drawableCanvasHolds = new WeakMap<HTMLCanvasElement, {count: number; prior: string | null}>()

function holdDrawableCanvas(canvas: HTMLCanvasElement): () => void {
    const hold = drawableCanvasHolds.get(canvas)
    if (hold) hold.count++
    else {
        drawableCanvasHolds.set(canvas, {count: 1, prior: canvas.getAttribute('content')})
        canvas.setAttribute('content', 'drawable')
    }
    let released = false
    return () => {
        if (released) return
        released = true
        const current = drawableCanvasHolds.get(canvas)
        if (!current || --current.count > 0) return
        drawableCanvasHolds.delete(canvas)
        if (current.prior === null) canvas.removeAttribute('content')
        else canvas.setAttribute('content', current.prior)
    }
}

/** True when this browser + device implement html-in-canvas's WebGPU path. */
function supportsDrawElement(canvas: HTMLCanvasElement, queue: GPUQueue): boolean {
    return (
        typeof (canvas as Partial<DrawableCanvas>).requestPaint === 'function' &&
        typeof (queue as unknown as Partial<ElementImageQueue>).drawElementImageToTexture === 'function'
    )
}

/**
 * Create the DOM-capture host and return the media texture the fragment samples. The texture is
 * always valid to bind: it stays transparent until the first snapshot lands, and for good where
 * the API is missing (Firefox, Safari, Chromium < 157), where the templates show the HTML as plain
 * DOM over the canvas instead. Diagnostics go through `debugWarn` only.
 */
function createDomCaptureHost(params: GpuFragmentParams): SwappableMediaTexture {
    const {gpu, canvas, domElement, onBeforeRender, onCleanup} = params
    const tex = createSwappableMediaTexture(params, {
        label: 'HTMLInCanvas',
        // The canvas's backing buffer — the size every capture is rasterised at.
        initial: {width: canvas?.width || 1, height: canvas?.height || 1},
        // Written via `unwrap()` + `drawElementImageToTexture`, bypassing `write()` — a mip chain
        // here would leave the upper mips unpopulated.
        mipmaps: false,
    })

    if (!canvas || !domElement || typeof window === 'undefined') return tex
    const queue = gpu.device.queue
    if (!supportsDrawElement(canvas, queue)) {
        debugWarn('[HTMLInCanvas] html-in-canvas is not available in this browser (needs Chrome 157+); the layer stays transparent.')
        return tex
    }
    // The wrapper must be drawn into its NEAREST ancestor canvas, which has to be ours.
    if (domElement.parentElement?.closest('canvas') !== canvas) {
        debugWarn('[HTMLInCanvas] the drawable wrapper is not inside the <Shader> canvas; nothing to capture.')
        return tex
    }

    const drawableCanvas = canvas as DrawableCanvas
    const elementQueue = queue as unknown as ElementImageQueue
    const releaseCanvas = holdDrawableCanvas(canvas)

    // `snapshotRecorded`: a paint has recorded the wrapper at least once, so drawing can't throw
    // "no cached paint record". `stale`: the texture doesn't hold the latest snapshot at its size.
    let snapshotRecorded = false
    let stale = true
    let geometrySynced = false
    let hasLoggedError = false
    // A paint we asked for (setup) carries an empty `changedElements` when nothing changed, but it
    // still means a snapshot exists for this new composition's (empty) texture to copy.
    let paintRequested = true

    // Copy only on `paint`, never per frame: it fires on every change to the wrapper's rendering
    // (framework updates, CSS animations and transitions, caret blink, image loads, hover).
    // The copy itself runs in onBeforeRender, which only the ACTIVE composition drains — cached,
    // inactive compositions just keep their flags current, so swapping back recopies if needed.
    const onPaint = (event: Event): void => {
        const changed = (event as CanvasPaintEvent).changedElements
        if (!paintRequested && changed && !changed.includes(domElement)) return
        paintRequested = false
        snapshotRecorded = true
        stale = true
    }
    canvas.addEventListener('paint', onPaint)
    drawableCanvas.requestPaint()

    onBeforeRender(() => {
        if (tex.disposed) return
        const width = Math.max(1, canvas.width)
        const height = Math.max(1, canvas.height)
        if (tex.width !== width || tex.height !== height) {
            // Backing buffer resized (CSS resize, zoom, quality scaling). Swapping makes a fresh,
            // empty texture — redraw the current snapshot into it at the new resolution.
            tex.ensureSize(width, height)
            stale = true
        }
        if (!stale || !snapshotRecorded) return
        stale = false
        try {
            elementQueue.drawElementImageToTexture(
                {source: domElement},
                {texture: tex.unwrap(), size: {width, height}},
            )
        } catch (e) {
            // e.g. the wrapper is display:none (no snapshot). The next paint re-arms the copy.
            if (!hasLoggedError) {
                debugWarn('[HTMLInCanvas] drawElementImageToTexture failed:', e)
                hasLoggedError = true
            }
            return
        }
        if (!geometrySynced) {
            // The wrapper covers the canvas 1:1, so the identity transform puts its content (for
            // hit testing and accessibility) exactly where it appears. Set once — it never changes.
            try {
                drawableCanvas.updateElementGeometry(domElement, {canvasTransform: new DOMMatrix()})
            } catch {
                /* geometry is best-effort — the texture is what matters */
            }
            geometrySynced = true
        }
    })

    onCleanup(() => {
        canvas.removeEventListener('paint', onPaint)
        releaseCanvas()
    })

    return tex
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: 'HTMLInCanvas',
    role: 'media',
    species: 'custom',
    deprecatedNames: ['DOMTexture'],
    boundingBoxDeclaration: {aspectRatio: null},
    category: 'Textures',
    description: 'A generator that turns its HTML children into a live layer, laid out at the canvas size and still interactive (Chrome 157+).',
    // Not `experimental`: html-in-canvas shipped in Chrome 157, and where it's missing the framework
    // templates show the HTML as plain DOM over the canvas (see utilities/htmlInCanvas.ts), so the
    // component degrades instead of breaking.
    capturesDOM: true,
    props: {},

    // The DOM-capture host (CPU, above) + a two-step GPU tail: sample the captured DOM at the
    // screen UV (the capture covers the full canvas, texel-for-pixel), then decode sRGB → linear
    // (alpha passes through; the copy writes straight alpha like every other media upload).
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const tex = createDomCaptureHost(params)
        const sampled = tex.kit.sample(params.ctx.uv)
        return vec4(call(tonemap.srgbToLinear, 'srgbToLinear', [sampled.member('rgb')]), sampled.member('a'))
    }}
})

export default componentDefinition
