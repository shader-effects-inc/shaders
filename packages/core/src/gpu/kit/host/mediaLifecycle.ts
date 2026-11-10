/**
 * CPU-side media lifecycle helpers — the loading half of the texture shaders.
 *
 * None of this is GPU code. It is the bookkeeping a media shader does around
 * `createMediaTexture` / `registerExternalTexture`: allocate a placeholder, fetch and decode a
 * source, swap the backing texture when the size changes, and tear the whole thing down on cleanup
 * without writing into a destroyed texture. It lives under `gpu/kit/host/` because shader
 * definitions are its only consumers (PRIMITIVES.md C9).
 *
 * The four pieces here were hand-rolled in seven shaders and had drifted:
 *
 *   - **`createSwappableMediaTexture`** — ImageTexture's `applyImage`, Text's and HTMLInCanvas's
 *     near-identical `ensureTexSize`, and Ascii's / ObjectTracker's fixed-size halves. Only Text and
 *     HTMLInCanvas skipped the realloc when the size was unchanged; only ImageTexture and Text had a
 *     dispose guard on the write path.
 *   - **`createUrlSourceLoader`** — the `currentUrl` / `isLoading` / `isDisposed` state machine that
 *     ImageTexture and VideoTexture each carried a copy of.
 *   - **`decodeImageSource`** — ImageTexture's bitmap-vs-SVG fork, including the distinction that
 *     makes an SVG measurable (raster size for the texture, INTRINSIC size for layout).
 *   - **`createVideoElementSource`** — VideoTexture's and WebcamTexture's video-element lifecycle and
 *     the shared readiness gate, and the ONE place a Firefox `importExternalTexture` fallback can
 *     later land.
 *
 * Structural-vs-runtime (C5) does not apply — this is all CPU — but two lifecycle rules do, and both
 * are load-bearing:
 *
 *   1. **Check disposal after every `await`.** A shader can be cleaned up while a fetch is in flight;
 *      writing into a destroyed texture is a WebGPU validation error. The `ctx.isDisposed()` hook on
 *      the loader callbacks exists for exactly this, and the texture handle guards itself as well.
 *   2. **A failed load does not commit its URL.** The per-frame URL watch retries a URL that never
 *      committed — which is why a permanently broken URL retries (and logs) every frame. That is
 *      pre-existing behaviour, preserved here deliberately; the fix belongs with the retry policy,
 *      not with this extraction.
 */
import type {GpuFragmentParams, GpuMediaTexture, KitTexture} from '../../contract'
import {registerNaturalSize} from '../../../utilities/naturalSize'
import {debugWarn} from '../../support'

// ── Swappable media texture ────────────────────────────────────────────────────────────────────

/** Full mip chain for a texture whose long axis is `size` texels (1×1 → 1 level). */
function mipLevelCountFor(width: number, height: number): number {
    return Math.floor(Math.log2(Math.max(width, height))) + 1
}

export interface SwappableMediaTextureOptions {
    /** Texture label (shows up in WebGPU error messages / the texture manager's accounting). */
    label: string
    /**
     * Size of the initial placeholder. Default 1×1 — a transparent texel, so the pass always has
     * something valid to bind and the layer renders transparent until content lands. Text and
     * HTMLInCanvas start larger only because their first write is imminent.
     */
    initial?: {width: number; height: number}
    /** Base format, forwarded to `createMediaTexture`. Default `rgba8unorm`. */
    format?: GPUTextureFormat
    /**
     * Allocate a full mip chain and regenerate it on every `write()` — default `true`. Set `false`
     * for a texture whose upload path never goes through `write()` (HTMLInCanvas uses `unwrap()` +
     * `drawElementImageToTexture` directly): a mip chain with only level 0 ever populated would
     * sample garbage/zero data from the unpopulated upper mips on minification.
     */
    mipmaps?: boolean
}

export interface SwappableMediaTexture {
    /** The registered `KitTexture` for the fragment builder. Follows swaps — register it ONCE. */
    readonly kit: KitTexture
    /**
     * Ensure the backing texture is exactly `width`×`height` (each floored at 1), allocating and
     * swapping when it is not. Destroying the old texture is safe: WebGPU keeps it alive for frames
     * already submitted. No-op after cleanup.
     */
    ensureSize(width: number, height: number): void
    /** Upload pixel content (ImageBitmap / HTMLCanvasElement / ImageData / TypedArray). */
    write(source: unknown): void
    /** The raw `GPUTexture`, for `queue.drawElementImageToTexture` (HTMLInCanvas). */
    unwrap(): GPUTexture
    readonly width: number
    readonly height: number
    /** True once the shader has been cleaned up — check it after an await before writing. */
    readonly disposed: boolean
}

/**
 * A media texture that can change size: allocate → swap the getter → destroy the previous one, with
 * the placeholder and the cleanup wired up.
 *
 * Media textures are STATIC-sized, so a new image at its native resolution, a re-rastered text run
 * or a resized DOM capture means a NEW texture. The pass manager notices the backing texture's
 * identity changed and rebuilds only the sampling pass's bind group — no recompose, no pipeline
 * rebuild. Shaders whose texture never changes size (Ascii's and ObjectTracker's glyph atlases) use
 * this too and simply never call `ensureSize`; they get the dispose-guarded `write` for free.
 */
export function createSwappableMediaTexture(
    params: GpuFragmentParams,
    opts: SwappableMediaTextureOptions,
): SwappableMediaTexture {
    const {createMediaTexture, registerMediaTexture, onCleanup} = params
    const wantsMipmaps = opts.mipmaps ?? true
    const make = (width: number, height: number): GpuMediaTexture =>
        createMediaTexture({
            width,
            height,
            label: opts.label,
            ...(wantsMipmaps ? {mipLevelCount: mipLevelCountFor(width, height)} : {}),
            ...(opts.format ? {format: opts.format} : {}),
        })

    let current = make(Math.max(1, opts.initial?.width ?? 1), Math.max(1, opts.initial?.height ?? 1))
    let isDisposed = false

    const kit: KitTexture = registerMediaTexture(() => current.texture)

    onCleanup(() => {
        isDisposed = true
        current.destroy()
    })

    return {
        kit,
        ensureSize(width, height) {
            const w = Math.max(1, width)
            const h = Math.max(1, height)
            if (isDisposed || (current.width === w && current.height === h)) return
            const next = make(w, h)
            const previous = current
            current = next
            previous.destroy()
        },
        write(source) {
            if (isDisposed) return
            current.write(source)
            // Content-change-driven, not per-frame (see call sites) — safe to regenerate here.
            if (wantsMipmaps) current.generateMipmaps()
        },
        unwrap: () => current.unwrap(),
        get width() { return current.width },
        get height() { return current.height },
        get disposed() { return isDisposed },
    }
}

// ── URL source loader ──────────────────────────────────────────────────────────────────────────

/** What a `load` callback is handed alongside the URL. */
export interface UrlLoadContext {
    /**
     * True once the shader has been cleaned up. Check it after EVERY await and bail — the textures
     * the load would write into are gone.
     */
    isDisposed(): boolean
    /**
     * Mark this URL as loaded, which stops the per-frame watch from re-requesting it. Call it after
     * the last await that could fail, so a failed load stays retryable (see the module note).
     */
    commit(): void
}

export interface UrlSourceLoaderOptions {
    /** Name of the URL prop, read live via `getCpuValue`. */
    prop: string
    /** Do the load. Owns its own decode/apply; the loader owns only the state machine. */
    load(url: string, ctx: UrlLoadContext): Promise<void>
    /** Reported for anything `load` throws. Error TAXONOMY stays per-shader by design. */
    onError?(error: unknown, url: string): void
}

export interface UrlSourceLoader {
    /** The URL currently loaded (empty until the first `commit`). */
    currentUrl(): string
    /** A load is in flight — the per-frame watch stands down while it is. */
    isLoading(): boolean
    /** Force a load of `url` now, bypassing the per-frame watch. Used by the kickoff and by tests. */
    request(url: string): void
}

/**
 * The URL-prop state machine: kick off the initial load, watch the prop for changes each frame, and
 * keep exactly one load in flight.
 *
 * NOTE the `setTimeout(…, 0)` kickoff: the load must not start until the node's uniforms are
 * populated, because `getCpuValue` reads the live handle. Deferring by a macrotask is the workaround,
 * not the fix — the fix is an `onUniformsReady` hook on `GpuFragmentParams` (PRIMITIVES_PLAN.md
 * Phase 12 item 3), at which point this function is the single site that changes.
 */
export function createUrlSourceLoader(
    params: GpuFragmentParams,
    opts: UrlSourceLoaderOptions,
): UrlSourceLoader {
    const {getCpuValue, onBeforeRender, onCleanup} = params
    let currentUrl = ''
    let isLoading = false
    let isDisposed = false
    let lastFailedUrl: string | null = null

    const run = (url: string): void => {
        if (!url || url.trim() === '' || isLoading || isDisposed) return
        isLoading = true
        const loadCtx: UrlLoadContext = {
            isDisposed: () => isDisposed,
            commit: () => { currentUrl = url },
        }
        void (async () => {
            try {
                await opts.load(url, loadCtx)
            } catch (error) {
                opts.onError?.(error, url)
            } finally {
                isLoading = false
                // Anything that finished without committing (threw, or bailed on a dispose/stale
                // check) is a failure. Remembering it is what bounds the per-frame watch below.
                lastFailedUrl = currentUrl === url ? null : url
            }
        })()
    }

    const readUrl = (): string => (getCpuValue(opts.prop) as string) ?? ''

    // Kick off the initial load once uniforms are populated (see the Phase 12 note above).
    setTimeout(() => {
        const initialUrl = readUrl()
        if (initialUrl && initialUrl.trim() !== '') run(initialUrl)
    }, 0)

    // Watch the prop each frame. A URL that failed never committed, so `newUrl !== currentUrl` stays
    // true forever — hence the `lastFailedUrl` gate: one automatic attempt per URL, not one per frame.
    // Editing the prop to anything else (including back to the failed URL) re-arms it, and `request`
    // bypasses the gate entirely so a caller can retry deliberately.
    onBeforeRender(() => {
        const newUrl = readUrl()
        if (newUrl === currentUrl || isLoading) return
        if (newUrl === lastFailedUrl) return
        lastFailedUrl = null
        run(newUrl)
    })

    onCleanup(() => {
        isDisposed = true
        currentUrl = ''
    })

    return {
        currentUrl: () => currentUrl,
        isLoading: () => isLoading,
        request: run,
    }
}

// ── Image decode ───────────────────────────────────────────────────────────────────────────────

/**
 * A decoded image ready to upload. The two size pairs are NOT interchangeable:
 *
 *   - `width`/`height` are the RASTER size — what the GPU texture must be.
 *   - `naturalWidth`/`naturalHeight` are the INTRINSIC size — the layout-facing "how big is this
 *     element" answer that goes to the natural-size registry. For a bitmap they are the same; for an
 *     SVG they differ, and conflating them makes a 32px logo slot at 2048px in a layout column.
 */
export interface DecodedImageSource {
    source: ImageBitmap | HTMLCanvasElement
    width: number
    height: number
    naturalWidth: number
    naturalHeight: number
    /** Release transient decode resources (closes an `ImageBitmap`). Safe to call twice. */
    close(): void
}

/** Reference resolution an SVG is rasterized at — its long axis, in texels. */
const SVG_RASTER_SIZE = 2048

function isSvgUrl(url: string): boolean {
    return /\.svg($|\?|#)/i.test(url)
}

/**
 * Rasterize an SVG (which has no intrinsic bitmap size) into a canvas at a fixed reference
 * resolution, long axis at `SVG_RASTER_SIZE`, aspect preserved.
 */
function rasterizeSvg(url: string): Promise<DecodedImageSource> {
    return new Promise((resolve, reject) => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
            const naturalW = img.naturalWidth || SVG_RASTER_SIZE
            const naturalH = img.naturalHeight || SVG_RASTER_SIZE
            const aspect = naturalW / naturalH
            const canvasW = aspect >= 1 ? SVG_RASTER_SIZE : Math.round(SVG_RASTER_SIZE * aspect)
            const canvasH = aspect >= 1 ? Math.round(SVG_RASTER_SIZE / aspect) : SVG_RASTER_SIZE
            const canvas = document.createElement('canvas')
            canvas.width = canvasW
            canvas.height = canvasH
            const ctx = canvas.getContext('2d')
            if (!ctx) {
                reject(new Error('Could not get 2D context'))
                return
            }
            ctx.drawImage(img, 0, 0, canvasW, canvasH)
            resolve({
                source: canvas,
                width: canvasW,
                height: canvasH,
                naturalWidth: naturalW,
                naturalHeight: naturalH,
                close: () => { /* a canvas needs no release */ },
            })
        }
        img.onerror = () => reject(new Error(`SVG image failed to load: ${url}`))
        img.src = url
    })
}

/**
 * Fetch + decode an image URL to something uploadable. SVGs go through a canvas rasterization (with
 * the intrinsic/raster split above); everything else is a CORS fetch → blob → `createImageBitmap`.
 */
export async function decodeImageSource(url: string): Promise<DecodedImageSource> {
    if (isSvgUrl(url)) return rasterizeSvg(url)
    const response = await fetch(url, {mode: 'cors'})
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const blob = await response.blob()
    const bitmap = await createImageBitmap(blob)
    let closed = false
    return {
        source: bitmap,
        width: bitmap.width,
        height: bitmap.height,
        naturalWidth: bitmap.width,
        naturalHeight: bitmap.height,
        close: () => {
            if (closed) return
            closed = true
            bitmap.close()
        },
    }
}

// ── Video element source (URL video + webcam) ──────────────────────────────────────────────────

/** Where the video frames come from. */
export type VideoSourceSpec =
    | {kind: 'url'; prop: string}
    | {kind: 'webcam'; constraints?: MediaStreamConstraints}

/**
 * Which step failed, so a shader's `onError` can keep its own message taxonomy. `'acquire'` covers
 * everything up to and including metadata (a bad URL, a load timeout, a denied camera) and is
 * reported ONCE per attempt; `'autoplay'` is the browser refusing to start playback, which is not
 * necessarily fatal — see `autoplayRequired`.
 */
export type VideoLoadStage = 'acquire' | 'autoplay'

export interface VideoElementSourceOptions {
    source: VideoSourceSpec
    /**
     * Keep `video.loop` synced to a boolean prop each frame (URL sources only). The prop packs to an
     * f32 field (1 = true), and an UNSET value reads as true — both current consumers default their
     * loop prop to true.
     */
    loop?: {prop: string}
    /**
     * Key the source's pixel dimensions are published under in the natural-size registry. Defaults
     * to the URL for a URL source; a webcam has no URL and must pass its own fixed key.
     */
    naturalSizeKey?: string
    /**
     * Milliseconds to wait for `loadedmetadata` before giving up. Default 0 = wait indefinitely
     * (the webcam's behaviour — a permission prompt can legitimately take a while).
     */
    metadataTimeoutMs?: number
    /**
     * Whether a rejected `play()` aborts the load. False (VideoTexture) reports it and keeps the
     * element, since a browser autoplay block still leaves a decodable first frame; true
     * (WebcamTexture) discards it. The two consumers genuinely differ here, so there is no default
     * winner beyond the more forgiving one.
     */
    autoplayRequired?: boolean
    onError?(error: unknown, info: {url: string | null; stage: VideoLoadStage}): void
}

export interface VideoElementSource {
    /**
     * The readiness-gated getter to hand `registerExternalTexture`. Returns the element only when it
     * has a decodable frame; the pass manager skips the pass for a frame where it is null.
     */
    getSource(): HTMLVideoElement | null
    /** The live element, or null before the first successful load. */
    element(): HTMLVideoElement | null
}

/** Create a muted, inline-playing video element with the shared attributes. */
function makeVideoElement(): HTMLVideoElement {
    const video = document.createElement('video')
    video.playsInline = true
    video.muted = true // always muted, for autoplay
    return video
}

/** Resolve once `loadedmetadata` fires; reject on error, or after `timeoutMs` when non-zero. */
function awaitMetadata(video: HTMLVideoElement, timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve()
        video.onerror = () => reject(new Error('Failed to load video'))
        if (timeoutMs > 0) setTimeout(() => reject(new Error('Video load timeout')), timeoutMs)
    })
}

/**
 * The video-element half of VideoTexture and WebcamTexture: acquire a source (a URL or
 * `getUserMedia`), wait for metadata, start playback, publish the frame size for layout, and gate
 * the element behind a readiness check so the renderer never imports an undecodable frame.
 *
 * The GPU side is identical for both consumers — a per-frame zero-copy `importExternalTexture` — so
 * only the acquisition differs, and both paths live here rather than in the two shaders.
 */
export function createVideoElementSource(
    params: GpuFragmentParams,
    opts: VideoElementSourceOptions,
): VideoElementSource {
    const {getCpuValue, onBeforeRender, onCleanup} = params
    const metadataTimeoutMs = opts.metadataTimeoutMs ?? 0
    const autoplayRequired = opts.autoplayRequired ?? false

    let videoElement: HTMLVideoElement | null = null
    let mediaStream: MediaStream | null = null
    let isDisposed = false

    /** `loop` packs to an f32 field (1 = true, -1/0 = false); an unset value means true. */
    const readLoop = (): boolean => {
        if (!opts.loop) return false
        const raw = getCpuValue(opts.loop.prop)
        return raw === true || raw === undefined || (typeof raw === 'number' && raw > 0)
    }

    /** Play, publish the frame size, and adopt the element. Shared tail of both acquisition paths. */
    const adopt = async (video: HTMLVideoElement, url: string | null): Promise<void> => {
        try {
            // Start playback before the first sample, to avoid a black first frame.
            await video.play()
        } catch (error) {
            opts.onError?.(error, {url, stage: 'autoplay'})
            if (autoplayRequired) throw error
        }
        if (isDisposed) return

        // Swap in the new element, retiring the previous one.
        if (videoElement && videoElement !== video) {
            videoElement.pause()
            videoElement.src = ''
        }
        videoElement = video
        const key = opts.naturalSizeKey ?? url
        if (key) registerNaturalSize(key, video.videoWidth, video.videoHeight)
    }

    if (opts.source.kind === 'url') {
        const urlProp = opts.source.prop
        createUrlSourceLoader(params, {
            prop: urlProp,
            async load(url, ctx) {
                const video = makeVideoElement()
                video.src = url
                video.crossOrigin = 'anonymous'
                video.loop = readLoop()
                try {
                    await awaitMetadata(video, metadataTimeoutMs)
                } catch (error) {
                    // Release the element before rethrowing; the loader reports it once, as 'acquire'.
                    video.src = ''
                    throw error
                }
                if (ctx.isDisposed()) {
                    video.src = ''
                    return
                }
                ctx.commit()
                await adopt(video, url)
            },
            onError(error, url) {
                opts.onError?.(error, {url, stage: 'acquire'})
            },
        })

        if (opts.loop) {
            onBeforeRender(() => {
                if (!videoElement) return
                const loopValue = readLoop()
                if (videoElement.loop !== loopValue) videoElement.loop = loopValue
            })
        }

        onCleanup(() => {
            isDisposed = true
            if (videoElement) {
                videoElement.pause()
                videoElement.src = ''
                videoElement = null
            }
        })
    } else {
        const constraints = opts.source.constraints ?? {video: true, audio: false}
        let isInitialized = false

        const startWebcam = async (): Promise<void> => {
            if (isInitialized || isDisposed) return
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
                if (isDisposed) {
                    mediaStream.getTracks().forEach((track) => track.stop())
                    mediaStream = null
                    return
                }
                const video = makeVideoElement()
                video.srcObject = mediaStream
                await awaitMetadata(video, metadataTimeoutMs)
                if (isDisposed) return
                await adopt(video, null)
                if (isDisposed) return
                isInitialized = true
                debugWarn(`[media] Video source started: ${video.videoWidth}x${video.videoHeight}`)
            } catch (error) {
                opts.onError?.(error, {url: null, stage: 'acquire'})
            }
        }

        // Deferred for the same reason as the URL loader's kickoff (Phase 12: onUniformsReady).
        setTimeout(() => { void startWebcam() }, 0)
        // A no-op registration, kept so the frame loop's callback set drains the same way it does
        // for a URL source (which registers a URL watch here).
        onBeforeRender(() => {})

        onCleanup(() => {
            isDisposed = true
            if (mediaStream) {
                mediaStream.getTracks().forEach((track) => track.stop())
                mediaStream = null
            }
            if (videoElement) {
                videoElement.srcObject = null
                videoElement = null
            }
            isInitialized = false
        })
    }

    const readySource = (): HTMLVideoElement | null => {
        if (!videoElement || isDisposed) return null
        // readyState < HAVE_CURRENT_DATA, or metadata not yet resolved → nothing decodable to import.
        if (videoElement.readyState < 2 || videoElement.videoWidth === 0) return null
        // TODO(firefox): `importExternalTexture` is Chromium-only. The fallback is a media texture
        // written per frame with `texture.write(video)` and sampled as a regular texture. It belongs
        // here — one site, both consumers — behind a capability probe.
        return videoElement
    }

    return {
        getSource: readySource,
        element: () => videoElement,
    }
}
