import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/porters'
import {call, vec4} from '@coreroot/gpu/porters'
import {defineShader} from '@coreroot/std'
import {tonemap} from '@coreroot/gpu/kit'
import {createSwappableMediaTexture, type SwappableMediaTexture} from '@coreroot/gpu/kit/host/mediaLifecycle'

export interface ComponentProps {}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Host recipe: DOM capture — the WICG html-in-canvas copy loop. Pure CPU machinery; the
// GPU tail below only samples the captured texture.
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Create the DOM-capture host. Needs only the raw GPUDevice + the raw GPUTexture behind a media
 * texture, then `queue.copyElementImageToTexture({source: element}, {destination: {texture}})`
 * (WICG html-in-canvas, Chrome Canary flag) — accessed via `gpu.device` + `mediaTexture.unwrap()`.
 * The spec moved from positional `(element, {texture})` to the two-dictionary form
 * (GPUCopyElementImageSource / GPUCopyElementImageDestination); older Canary builds still
 * expect the positional form, so the copy falls back once on TypeError and sticks. The media
 * texture is sized to the captured element (swapped on resize — the pass manager rebuilds the
 * bind group). Defensive no-op when the API is absent (non-Canary): the texture stays
 * zero-initialised (transparent), matching the caught-error behaviour.
 */
function createDomCaptureHost(params: GpuFragmentParams): SwappableMediaTexture {
    const {gpu, domCanvas, onBeforeRender, onResize, onCleanup} = params
    const device = gpu.device
    const initialDpr = Math.min(window.devicePixelRatio, 2)
    // Swappable so a resize can re-allocate at the element's new physical size; the WICG copy
    // needs the raw GPUTexture, which is what `unwrap()` is for.
    const tex = createSwappableMediaTexture(params, {
        label: 'HTMLInCanvas',
        initial: {
            width: Math.max(1, Math.round(window.innerWidth * initialDpr)),
            height: Math.max(1, Math.round(window.innerHeight * initialDpr)),
        },
    // The WICG copy below writes via `unwrap()` + `copyElementImageToTexture` directly,
            // bypassing `write()` — a mip chain here would leave upper mips unpopulated garbage.
            mipmaps: false,
        })

    let contentElement: Element | null = null
    let hasLoggedError = false
    let paintSetup = false
    let hasFreshSnapshot = false
    // Pre-dictionary Canary builds take (element, {texture}); current builds take
    // ({source}, {destination: {texture}}). Detected once on first TypeError, then sticky.
    let useLegacySignature = false

    const ensureTexSize = (w: number, h: number): void => tex.ensureSize(w, h)

    onBeforeRender(() => {
        if (!domCanvas || tex.disposed) return

        // Find the direct child element once — it must be a direct child of the layoutsubtree
        // canvas for copyElementImageToTexture to work. Size the media texture to the element's
        // physical render pixels (copyElementImageToTexture copies at that size; `100vw` inside a
        // layoutsubtree may differ from window.innerWidth). Swapping recreates the GPU texture.
        if (!contentElement) {
            contentElement = domCanvas.firstElementChild
            if (contentElement) {
                const rect = (contentElement as HTMLElement).getBoundingClientRect()
                const physW = Math.round(rect.width * window.devicePixelRatio)
                const physH = Math.round(rect.height * window.devicePixelRatio)
                if (physW > 0 && physH > 0) ensureTexSize(physW, physH)
            }
        }
        if (!contentElement) return

        // One-time setup: initialize the 2D context (activates the layoutsubtree), register the
        // onpaint handler, and trigger the initial snapshot.
        if (!paintSetup) {
            domCanvas.getContext('2d')
            ;(domCanvas as unknown as {onpaint: (() => void) | null}).onpaint = () => {
                hasFreshSnapshot = true
            }
            ;(domCanvas as unknown as {requestPaint?: () => void}).requestPaint?.()
            paintSetup = true
            return
        }

        // Copy element → GPU texture once a fresh snapshot is confirmed. onpaint fires
        // automatically when DOM content changes (framework reactivity, CSS animations, etc.) —
        // do NOT call requestPaint() here every frame.
        if (hasFreshSnapshot && contentElement) {
            const copyFn = (device.queue as unknown as {copyElementImageToTexture?: (...args: unknown[]) => void}).copyElementImageToTexture
            if (typeof copyFn !== 'function') {
                // Defensive no-op when the WICG API is unavailable (non-Canary) — stay transparent.
                hasFreshSnapshot = false
                return
            }
            try {
                if (useLegacySignature) {
                    copyFn.call(device.queue, contentElement, {texture: tex.unwrap()})
                } else {
                    try {
                        copyFn.call(device.queue, {source: contentElement}, {destination: {texture: tex.unwrap()}})
                    } catch (e) {
                        if (!(e instanceof TypeError)) throw e
                        // Dictionary form rejected → older Canary. Retry positional and stick.
                        copyFn.call(device.queue, contentElement, {texture: tex.unwrap()})
                        useLegacySignature = true
                    }
                }
                hasFreshSnapshot = false
            } catch (e) {
                if (!hasLoggedError) {
                    console.warn(
                        '[HTMLInCanvas] copyElementImageToTexture failed.\n' +
                        'Requires Chrome Canary with chrome://flags/#canvas-draw-element enabled.\n' +
                        'The element must be a direct child of a <canvas layoutsubtree>.',
                        e
                    )
                    hasLoggedError = true
                }
            }
        }
    })

    onResize(() => {
        const newDpr = Math.min(window.devicePixelRatio, 2)
        ensureTexSize(Math.round(window.innerWidth * newDpr), Math.round(window.innerHeight * newDpr))
        // Force a re-measure + resize to the element on the next frame, and re-request a paint.
        contentElement = null
        hasFreshSnapshot = false
        if (domCanvas) (domCanvas as unknown as {requestPaint?: () => void}).requestPaint?.()
    })

    onCleanup(() => {
        if (domCanvas) (domCanvas as unknown as {onpaint: (() => void) | null}).onpaint = null
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
    description: 'Render live HTML/DOM content as a WebGPU texture layer via the html-in-canvas API. Requires Chrome Canary with chrome://flags/#canvas-draw-element enabled.',
    capturesDOM: true,
    experimental: {
        message: 'Requires Chrome Canary with chrome://flags/#canvas-draw-element enabled. Do not use for production use.',
        docs: 'This component is powered by the WICG html-in-canvas proposal, which is currently only available in Chrome Canary behind a feature flag. It is not suitable for production use and may change as the specification evolves.',
        docsLink: {
            url: 'https://github.com/WICG/html-in-canvas',
            label: 'View the spec'
        }
    } as ComponentProps,
    props: {},

    // The DOM-capture host (CPU, above) + a two-step GPU tail: sample the captured DOM at the
    // screen UV (full canvas, no fit), then decode sRGB → linear (alpha passes through).
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const tex = createDomCaptureHost(params)
        const sampled = tex.kit.sample(params.ctx.uv)
        return vec4(call(tonemap.srgbToLinear, 'srgbToLinear', [sampled.member('rgb')]), sampled.member('a'))
    }}
})

export default componentDefinition
