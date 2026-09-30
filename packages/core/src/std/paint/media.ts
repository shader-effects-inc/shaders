/**
 * std/paint/media — pixels from an image, a video or the camera instead of from math.
 *
 * Each word loads its source, fits it into the layer like CSS `object-fit`, and returns the
 * GPU half of a definition. Spread the result into `defineShader({...})` next to your props,
 * with `role: 'media'` and `species: 'custom'`. Every word takes a `fit` prop
 * (`'cover'`, `'contain'`, `'fill'`, `'scale-down'`, marked `compileTime`) and
 * `decode: 'srgb-linear-alpha-cut'`, which reads the source as sRGB and makes the letterbox
 * transparent.
 */
// Maintainer notes (not part of the reference):
//  - Every media word is the same two halves: a HOST LIFECYCLE (kit/host/mediaLifecycle): a
//    swappable media texture or a readiness-gated external-texture source, created once at
//    composition and torn down in its onCleanup; and the shared GPU FIT SURFACE (kit/media's
//    `mediaSurface`): object-fit UV scale → fitted UV → sample → decode, with the media's pixel
//    size read on the GPU via `textureDimensions` (no CPU aspect uniforms).
//  - The words return `{gpu: {fragment}}` for the definition to spread (the compute-backed-blur
//    precedent, minus the compute half). Declarative lifecycle flags (`naturalSizeKey`,
//    `acceptsUVContext`, resize-fit bounds) stay declared on the definition — a word never owns
//    an engine flag.
//  - Error TAXONOMY stays per-shader by design (only ImageTexture knows how to phrase a bad image
//    URL; only WebcamTexture knows a camera permission denial), so each word takes the
//    consumer's `onError` as a slot.
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call} from '../../gpu/composer'
import {media} from '../../gpu/kit/index'
import {
    createSwappableMediaTexture,
    createUrlSourceLoader,
    createVideoElementSource,
    decodeImageSource,
} from '../../gpu/kit/host/mediaLifecycle'
import {registerNaturalSize} from '../../utilities/naturalSize'
import type {PropRef} from '../values'

/** What a media word returns: the GPU half of a definition. Spread it into `defineShader({...})`. */
export interface MediaSurfaceHalf {
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/** The fit-surface slots every media word shares. */
interface MediaFitSlots {
    /** The `compileTime` object-fit prop: `'cover'`, `'contain'`, `'fill'` or `'scale-down'`. */
    fit: PropRef
    /** How the sample is decoded. Always `'srgb-linear-alpha-cut'`: sRGB read as linear, letterbox made transparent. */
    // Required rather than defaulted so each consumer states its choice (kit/media's D-1 rule).
    decode: 'srgb-linear-alpha-cut'
}

/** Assemble the shared fit surface against the composition params. */
function fitSurface(
    params: GpuFragmentParams,
    slots: MediaFitSlots,
    texture: media.MediaSurfaceOptions['texture'],
    extra?: Pick<media.MediaSurfaceOptions, 'allowNone' | 'uvPostProcess'>,
): Expr {
    const {ctx, propValues, uvContext, effectiveViewportSize} = params
    return media.mediaSurface({
        texture,
        uv: uvContext ?? ctx.uv,
        viewport: effectiveViewportSize ?? ctx.viewportSize,
        fit: propValues[slots.fit.name],
        decode: slots.decode,
        ...extra,
    })
}

/**
 * An image from a URL prop, fitted into the layer.
 *
 * The layer is transparent until the image loads, then shows it at its real aspect ratio.
 * Set `naturalSizeKey: {fromProp: 'url'}` on the definition so layout can measure the image.
 * `onError` receives load failures with the URL so you can phrase the message.
 *
 * @example
 * ```ts
 * ...imageMedia({src: p('url'), fit: p('objectFit'), decode: 'srgb-linear-alpha-cut', label: 'Photo'})
 * ```
 * @see videoMedia, webcamMedia
 */
export function imageMedia(opts: MediaFitSlots & {
    /** The URL prop. */
    src: PropRef
    /** A debug label for the texture, shown in GPU tooling. */
    label: string
    /** Called when the image fails to load. */
    onError?: (error: unknown, url: string) => void
}): MediaSurfaceHalf {
    // Fetched → decoded → written into a media texture sized to the image's NATIVE pixels. A 1×1
    // transparent placeholder shows until the image lands; the real texture is swapped in when it
    // loads (the pass manager rebuilds the sampling bind group on the swap — no recompose), which
    // is what makes `textureDimensions` report the real aspect. sRGB samples are decoded to linear
    // in-shader (the media analog of the external-texture decode).
    return {gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const tex = createSwappableMediaTexture(params, {label: opts.label})

        createUrlSourceLoader(params, {
            prop: opts.src.name,
            async load(url, loadCtx) {
                const decoded = await decodeImageSource(url)
                if (loadCtx.isDisposed()) {
                    decoded.close()
                    return
                }
                loadCtx.commit()
                // Layout measures the source's INTRINSIC size (a 32px SVG logo slots at 32px), not
                // the raster the texture is baked at — for a bitmap the two are the same.
                registerNaturalSize(url, decoded.naturalWidth, decoded.naturalHeight)
                tex.ensureSize(decoded.width, decoded.height)
                tex.write(decoded.source)
                decoded.close()
            },
            onError: opts.onError,
        })

        return fitSurface(params, opts, tex.kit)
    }}}
}

/**
 * A video from a URL prop, playing in the layer.
 *
 * The video plays muted and inline. `loop` is a boolean prop. `metadataTimeoutMs` is how long
 * to wait for the video's metadata before reporting a failure. Set
 * `naturalSizeKey: {fromProp: 'url'}` on the definition. `onError` tells you whether loading
 * (`'acquire'`) or playback (`'autoplay'`) failed.
 *
 * @example
 * ```ts
 * ...videoMedia({src: p('url'), fit: p('objectFit'), loop: p('loop'), decode: 'srgb-linear-alpha-cut', metadataTimeoutMs: 10000})
 * ```
 * @see imageMedia, webcamMedia
 */
export function videoMedia(opts: MediaFitSlots & {
    /** The URL prop. */
    src: PropRef
    /** The loop prop, synced to the video each frame. */
    loop: PropRef
    /** Milliseconds to wait for the video's metadata before giving up. */
    metadataTimeoutMs: number
    onError?: (error: unknown, info: {url: string | null; stage: 'acquire' | 'autoplay'}) => void
}): MediaSurfaceHalf {
    // The shared video-element lifecycle (created once at composition, torn down in its onCleanup)
    // feeding a per-frame zero-copy `importExternalTexture` sample. The registered getter returns
    // the element only when it has a decodable frame; the pass manager re-imports it each frame
    // and skips the pass while it is null (the WGSL is fixed, so readiness never recompiles).
    // Firefox's CanvasTexture copy-path fallback is not implemented — its single site is in the
    // host module.
    return {gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const source = createVideoElementSource(params, {
            source: {kind: 'url', prop: opts.src.name},
            loop: {prop: opts.loop.name},
            metadataTimeoutMs: opts.metadataTimeoutMs,
            onError: opts.onError,
        })
        return fitSurface(params, opts, params.registerExternalTexture(source.getSource))
    }}}
}

/**
 * The user's camera, live in the layer.
 *
 * Asks for camera permission with your `constraints`. `mirror` is a boolean prop for a selfie
 * flip that updates instantly. This word also accepts `fit: 'none'` to show the feed at its
 * native pixel size. Set `naturalSizeKey: {fixed: '<key>'}` on the definition and pass the
 * same key here. `onError` receives permission denials and playback failures.
 *
 * @example
 * ```ts
 * ...webcamMedia({fit: p('objectFit'), mirror: p('mirror'), decode: 'srgb-linear-alpha-cut', constraints: {video: {facingMode: 'user'}}, naturalSizeKey: 'webcam'})
 * ```
 * @see videoMedia, imageMedia
 */
export function webcamMedia(opts: MediaFitSlots & {
    /** The selfie-mirror boolean prop. */
    mirror: PropRef
    /** The `getUserMedia` constraints to request the camera with. */
    constraints: MediaStreamConstraints
    /** The key the definition's `naturalSizeKey: {fixed}` uses (a webcam has no URL). */
    naturalSizeKey: string
    onError?: (error: unknown, info: {url: string | null; stage: 'acquire' | 'autoplay'}) => void
}): MediaSurfaceHalf {
    // The exact `videoMedia` path, only the source is `getUserMedia` instead of a URL (which is
    // why both share the host's `createVideoElementSource`). Autoplay is required — a webcam
    // element that would not play is not a usable feed, unlike a URL video where a blocked
    // autoplay still yields a first frame. `mirror` is a runtime uniform (selfie flip must not
    // recompose) applied to the fitted UV BEFORE the letterbox alpha cut, so the cut sees the
    // same UV that was sampled.
    return {gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const source = createVideoElementSource(params, {
            source: {kind: 'webcam', constraints: opts.constraints},
            naturalSizeKey: opts.naturalSizeKey,
            autoplayRequired: true,
            onError: opts.onError,
        })
        return fitSurface(params, opts, params.registerExternalTexture(source.getSource), {
            allowNone: true,
            uvPostProcess: (uv) => call(media.applyMirror, 'applyMirror', [uv, params.uniforms[opts.mirror.name]]),
        })
    }}}
}
