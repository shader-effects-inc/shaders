/**
 * Texture + sampler managers.
 *
 * `createTextureManager(root)` owns, for one renderer:
 *   - shared samplers (linear/nearest × clamp/repeat), created lazily and cached;
 *   - media textures (image/canvas/DOM upload targets, optional sRGB view) with `.write(...)`;
 *   - data textures (TypedArray-backed, float/byte formats) for compute/lookup inputs;
 *   - render textures (rgba16float, render+sampled) for RTT boundaries, with resize;
 *   - external-texture bindings (video/webcam) that re-import + rebuild their bind group each
 *     frame;
 *   - a live texture count (feeds performanceTracker's textureCount later).
 *
 * The composer/passManager decide layouts, sampling, and per-frame RTT sizing; this module
 * is purely lifecycle + the WebGPU idioms around it. sRGB view creation is the standard
 * "linear format + `-srgb` viewFormat + srgb view" idiom, built here from the raw GPUTextureView
 * so it is robust regardless of how TypeGPU's typed-view API settles.
 */
import type {TgpuRoot, TgpuBindGroup, TgpuBindGroupLayout} from 'typegpu'
import type {TgpuTexture, TgpuFixedSampler} from 'typegpu'

/** Formats a data texture may take (float/byte, for compute/lookup inputs). */
export type DataTextureFormat =
    | 'r32float'
    | 'r16float'
    | 'rg16float'
    | 'rg32float'
    | 'rgba8unorm'
    | 'rgba16float'
    | 'rgba32float'

type AnyImageSource =
    | HTMLCanvasElement
    | HTMLImageElement
    | HTMLVideoElement
    | ImageBitmap
    | ImageData
    | OffscreenCanvas
    | VideoFrame

/** A managed texture wrapper: the TypeGPU texture plus lifecycle + upload + view helpers. */
export interface ManagedTexture {
    /** The underlying TypeGPU texture (already `$usage`-flagged). */
    readonly texture: TgpuTexture
    readonly width: number
    readonly height: number
    readonly format: GPUTextureFormat
    /** Re-upload pixel data. Media: an image source; data: a TypedArray/ArrayBuffer. */
    write(source: AnyImageSource | ArrayBuffer | ArrayBufferView): void
    /** The raw GPUTexture (for HTMLInCanvas's `queue.drawElementImageToTexture`). */
    unwrap(): GPUTexture
    /**
     * A raw sRGB GPUTextureView when the texture was created with `srgb: true` (media only);
     * `undefined` otherwise. Sampling through it applies the sRGB→linear EOTF in hardware.
     */
    createSrgbView(): GPUTextureView | undefined
    /** Regenerate mipmaps (no-op unless created with sampled+render and mipLevelCount > 1). */
    generateMipmaps(): void
    /** Destroy the GPU texture and drop it from the live count. */
    destroy(): void
}

/** A render-to-texture target that can be resized (destroy + recreate). */
export interface RenderTexture extends ManagedTexture {
    /** Resize by destroying and recreating the backing texture. Returns false if already that size (no-op). */
    resize(width: number, height: number): boolean
}

/** Per-frame external-texture binding (video/webcam). */
export interface ExternalTextureBinding {
    /**
     * Import the current source and (re)build the bind group for this frame. Returns the fresh
     * bind group, or `null` when there is no current source (nothing to render this frame).
     */
    update(): TgpuBindGroup | null
    /** The bind group from the last successful `update()`, or `null`. */
    readonly current: TgpuBindGroup | null
    destroy(): void
}

export interface CreateMediaTextureOptions {
    width: number
    height: number
    /** Base (linear) format. Default `rgba8unorm`. */
    format?: GPUTextureFormat
    /** Add an `-srgb` viewFormat + expose `createSrgbView()`. */
    srgb?: boolean
    mipLevelCount?: number
    label?: string
}

export interface CreateDataTextureOptions {
    width: number
    height: number
    format: DataTextureFormat
    /** Optional initial pixel data. */
    data?: ArrayBufferView
    label?: string
}

export interface CreateRenderTextureOptions {
    width: number
    height: number
    /** Default `rgba16float` (the RTT target format). */
    format?: GPUTextureFormat
    mipLevelCount?: number
    label?: string
}

export interface CreateExternalTextureBindingOptions {
    /** The bind group layout whose entry `entryKey` is a `d.textureExternal()`. */
    layout: TgpuBindGroupLayout
    /** The external-texture entry key in `layout`. */
    entryKey: string
    /** Returns the live source each frame, or null/undefined when unavailable. */
    getSource: () => HTMLVideoElement | VideoFrame | null | undefined
    /** Other (constant) bind group entries — samplers, uniform buffers, etc. */
    staticEntries?: Record<string, unknown>
}

/** Shared sampler set. Created lazily; each getter caches. */
export interface SharedSamplers {
    readonly linearClamp: TgpuFixedSampler
    readonly nearestClamp: TgpuFixedSampler
    readonly linearRepeat: TgpuFixedSampler
    readonly nearestRepeat: TgpuFixedSampler
}

export interface TextureManager {
    readonly samplers: SharedSamplers
    createMediaTexture(options: CreateMediaTextureOptions): ManagedTexture
    createDataTexture(options: CreateDataTextureOptions): ManagedTexture
    createRenderTexture(options: CreateRenderTextureOptions): RenderTexture
    createExternalTextureBinding(options: CreateExternalTextureBindingOptions): ExternalTextureBinding
    /** Number of live GPU textures this manager currently owns. */
    readonly textureCount: number
    /** Destroy every texture (and drop samplers) this manager created. */
    destroy(): void
}

/** srgb viewFormat for a linear base format, e.g. `rgba8unorm` → `rgba8unorm-srgb`. */
function srgbViewFormat(format: GPUTextureFormat): GPUTextureFormat {
    return `${format}-srgb` as GPUTextureFormat
}

/** Raw texture-creation props (runtime formats, not literal types). */
interface RawTextureProps {
    size: [number, number]
    format: GPUTextureFormat
    viewFormats?: GPUTextureFormat[]
    mipLevelCount?: number
}

export function createTextureManager(root: TgpuRoot): TextureManager {
    const live = new Set<ManagedTexture>()

    /**
     * Create a texture from runtime (non-literal) props + usages. `root.createTexture`'s
     * heavily-generic signature is built for compile-time-literal formats; we pass runtime
     * strings, so create through a loosened signature and re-flag usages afterwards.
     */
    function createRaw(props: RawTextureProps, ...usages: ('sampled' | 'render' | 'storage')[]): TgpuTexture {
        // Call THROUGH `root` (via .call) — never through a detached alias. TypeGPU
        // captures the root as the texture's `branch` (INTERNAL_createTexture(props, this)) for lazy
        // GPU-texture creation; a `const create = root.createTexture; create(props)` call passes
        // `this = undefined`, so `branch` is undefined and the texture's `unwrap()` throws
        // `Cannot read properties of undefined (reading 'device')` the first time it's rendered.
        const create = root.createTexture as unknown as (p: RawTextureProps) => TgpuTexture
        const tex = create.call(root, props) as TgpuTexture & {$usage: (...u: string[]) => TgpuTexture}
        return tex.$usage(...usages)
    }

    // ---- Shared samplers (lazy + cached) --------------------------------------------------
    let linearClamp: TgpuFixedSampler | undefined
    let nearestClamp: TgpuFixedSampler | undefined
    let linearRepeat: TgpuFixedSampler | undefined
    let nearestRepeat: TgpuFixedSampler | undefined

    const samplers: SharedSamplers = {
        get linearClamp() {
            return (linearClamp ??= root.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
                mipmapFilter: 'linear',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            }))
        },
        get nearestClamp() {
            return (nearestClamp ??= root.createSampler({
                magFilter: 'nearest',
                minFilter: 'nearest',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            }))
        },
        get linearRepeat() {
            return (linearRepeat ??= root.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
                mipmapFilter: 'linear',
                addressModeU: 'repeat',
                addressModeV: 'repeat',
            }))
        },
        get nearestRepeat() {
            return (nearestRepeat ??= root.createSampler({
                magFilter: 'nearest',
                minFilter: 'nearest',
                addressModeU: 'repeat',
                addressModeV: 'repeat',
            }))
        },
    }

    // ---- Shared managed-texture wrapper ---------------------------------------------------
    function wrap(
        texture: TgpuTexture,
        width: number,
        height: number,
        format: GPUTextureFormat,
        srgb: boolean,
    ): ManagedTexture {
        const managed: ManagedTexture = {
            texture,
            width,
            height,
            format,
            write(source) {
                // TypeGPU's texture.write accepts both image sources and TypedArray/ArrayBuffer.
                ;(texture as {write: (s: unknown) => void}).write(source)
            },
            unwrap() {
                return root.unwrap(texture)
            },
            createSrgbView() {
                if (!srgb) return undefined
                // Raw sRGB view via the unwrapped GPUTexture — the standard WebGPU idiom, robust
                // regardless of how the TypeGPU typed-view API settles.
                return root.unwrap(texture).createView({format: srgbViewFormat(format)})
            },
            generateMipmaps() {
                const t = texture as {generateMipmaps?: () => void}
                t.generateMipmaps?.()
            },
            destroy() {
                if (!live.has(managed)) return
                texture.destroy()
                live.delete(managed)
            },
        }
        live.add(managed)
        return managed
    }

    // ---- Media textures (image/canvas/DOM upload targets) --------------------------------
    function createMediaTexture(options: CreateMediaTextureOptions): ManagedTexture {
        const {width, height, srgb = false, mipLevelCount} = options
        const format = options.format ?? 'rgba8unorm'
        const texture = createRaw(
            {
                size: [width, height],
                format,
                ...(srgb ? {viewFormats: [srgbViewFormat(format)]} : {}),
                ...(mipLevelCount ? {mipLevelCount} : {}),
            },
            'sampled',
            'render',
        )
        return wrap(texture, width, height, format, srgb)
    }

    // ---- Data textures (TypedArray-backed) ------------------------------------------------
    function createDataTexture(options: CreateDataTextureOptions): ManagedTexture {
        const {width, height, format, data} = options
        const texture = createRaw({size: [width, height], format}, 'sampled')
        const managed = wrap(texture, width, height, format, false)
        if (data) managed.write(data)
        return managed
    }

    // ---- Render textures (RTT) ------------------------------------------------------------
    function createRenderTexture(options: CreateRenderTextureOptions): RenderTexture {
        const format = options.format ?? 'rgba16float'
        const {mipLevelCount} = options
        let width = options.width
        let height = options.height

        function build(w: number, h: number): ManagedTexture {
            const texture = createRaw(
                {size: [w, h], format, ...(mipLevelCount ? {mipLevelCount} : {})},
                'sampled',
                'render',
            )
            return wrap(texture, w, h, format, false)
        }

        let inner = build(width, height)
        const rt: RenderTexture = {
            get texture() {
                return inner.texture
            },
            get width() {
                return width
            },
            get height() {
                return height
            },
            format,
            write(source) {
                inner.write(source)
            },
            unwrap() {
                return inner.unwrap()
            },
            createSrgbView() {
                return inner.createSrgbView()
            },
            generateMipmaps() {
                inner.generateMipmaps()
            },
            resize(w, h) {
                if (w === width && h === height) return false
                inner.destroy()
                width = w
                height = h
                inner = build(w, h)
                return true
            },
            destroy() {
                inner.destroy()
            },
        }
        return rt
    }

    // ---- External textures (video/webcam) ------------------------------------------------
    function createExternalTextureBinding(
        options: CreateExternalTextureBindingOptions,
    ): ExternalTextureBinding {
        const {layout, entryKey, getSource, staticEntries = {}} = options
        let current: TgpuBindGroup | null = null
        const binding: ExternalTextureBinding = {
            update() {
                const source = getSource()
                if (!source) {
                    current = null
                    return null
                }
                // External textures expire after the frame/task, so re-import + rebuild the bind
                // group every frame. A video element can transiently lose its backing resource
                // (source swap / element re-parent race) and importExternalTexture then THROWS —
                // treat that exactly like "not ready": null → the pass is skipped this frame.
                let external: GPUExternalTexture
                try {
                    external = root.device.importExternalTexture({source})
                } catch {
                    current = null
                    return null
                }
                current = root.createBindGroup(layout as never, {
                    ...staticEntries,
                    [entryKey]: external,
                } as never)
                return current
            },
            get current() {
                return current
            },
            destroy() {
                current = null
            },
        }
        return binding
    }

    return {
        samplers,
        createMediaTexture,
        createDataTexture,
        createRenderTexture,
        createExternalTextureBinding,
        get textureCount() {
            return live.size
        },
        destroy() {
            for (const t of [...live]) t.destroy()
            live.clear()
        },
    }
}
