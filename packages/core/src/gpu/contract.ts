/**
 * The GPU shader contract (`GpuShaderDefinition`) + the KitExpr IR.
 *
 * This is the interface the composer consumes. Everything DECLARATIVE
 * (name/category/requiresRTT/requiresChild/blendWithChildren/acceptsUVContext/
 * providesUVContextViaCompute/wantsBoundsParams/boundingBoxDeclaration/props/…) is the
 * shader's static metadata; the GPU-code-producing members are expressed as `KitExpr`s.
 *
 * ─────────────────────────────────────────────────────────────────────────────────────
 * HOW TEXTURES REACH SHADER CODE:
 *
 *   1. GLUE-LEVEL sampling (child RTT, mask textures, map-source textures — everything the
 *      COMPOSER samples): done in the runtime-emitted raw-WGSL glue, which references the
 *      composition texture bind-group layout accessor directly, e.g.
 *      `textureSample(tex.$.rtt_0, samp.$.linearClamp, uv)`. `tgpu.resolve`'s linker
 *      rewrites `tex.$.rtt_0` → the resolved binding name.
 *   2. SHADER-BODY sampling (media shaders, compute-output-reading distortions): the
 *      shader's `'use gpu'` fn declares TEXTURE-typed parameters (`d.texture2d(d.f32)` etc. —
 *      valid TGSL params, resolving to `texture_2d<f32>`); the glue passes the layout texture
 *      accessor as the call argument. The SAMPLER is NOT a parameter — `d.sampler` is rejected
 *      as a TGSL fn param ("not resolvable") — so the shader fn captures a sampler from the
 *      kit-owned shared-sampler layout accessor (`sharedSamplerLayout.$.linearClamp`) inside
 *      its body.
 *   3. `tgpu.slot`-per-texture does NOT work: filling a slot with a layout texture accessor
 *      (`fn.with(slot, layout.$.tex)`) throws "Accessed view outside of codegen mode",
 *      because reading a texture view's `.$` is only legal inside a codegen context (TypeGPU
 *      0.11.9). fn-args are the answer for shader-body sampling; glue-level sampling covers
 *      the composer.
 */
import type {TgpuBindGroupLayout} from 'typegpu'
import type {WgslStruct, AnyWgslData} from 'typegpu/data'
import type {
    MaskType,
    NodeMetadata,
    BoundingBoxDeclaration,
    ComponentProps,
    PropConfig,
    CleanupCallback,
    RenderCallback,
    ResizeCallback,
    BlendMode,
} from '../types'

// ═══════════════════════════════════════════════════════════════════════════════════════
// KitExpr — the GPU-expression IR (the "small string-builder")
// ═══════════════════════════════════════════════════════════════════════════════════════
//
// The composition glue is a runtime-emitted WGSL fragment. The composer builds a tiny
// expression tree of `Expr`s and serialises it to a WGSL string + an externals map (fn
// objects, bind-group layouts). It is deliberately minimal: it only models fn calls,
// field/member access, texture samples, literals, and vec construction — exactly the shapes
// `composeNodeTree`'s glue produces (blend folds, mask application, alpha cuts, driver
// overrides, the tonemap tail).
//
// An `Expr` is opaque; it carries an `emit(ctx)` that returns the WGSL text for the
// expression and registers any externals it needs into the `EmitContext`. Shader
// `fragment` builders receive `KitExpr` accessors (props, ctx fields, childNode, textures)
// and return a `KitExpr` — they never see the WGSL text.

/** The context threaded through `Expr.emit` — implemented by the composer's serialiser. */
export interface EmitContext {
    /**
     * Register an external (a `TgpuFn`, a bind-group layout, or a plain resolvable) and
     * return the STABLE identifier to reference it by in the WGSL template. Identical
     * externals (same object) dedupe to the same name, so a blend fn used 20 times is
     * declared once. `hint` seeds a readable name.
     */
    external(value: unknown, hint: string): string
    /** Append a statement to the current fragment body's preamble (driver-override locals). */
    statement(wgsl: string): void
    /** A fresh unique local-variable name (driver-override copies, RTT sample temporaries). */
    freshLocal(hint: string): string
    /**
     * Run `factory` at most once per `key` within this fragment and cache its result. Used
     * for driver-override locals: the first reference to a driver-overridden node's `props`
     * declares `var p = …; p.field = …;` and subsequent references reuse `p`.
     */
    memo(key: string, factory: () => string): string
}

/**
 * A GPU-side expression. The composer's currency. Immutable;
 * the fluent helpers return new `Expr`s. `emit` is package-internal — shader code only ever
 * constructs `Expr`s via the factories in `composer.ts` (`expr`, `call`, `vec4`, …) and the
 * `params` handed to `fragment`.
 */
export class Expr {
    /** @internal */
    readonly _emit: (ctx: EmitContext) => string

    constructor(emit: (ctx: EmitContext) => string) {
        this._emit = emit
    }

    /** `expr.<prop>` — member/swizzle access (`.rgb`, `.a`, `.xyz`, `.x`, or a struct field). */
    member(prop: string): Expr {
        return new Expr((ctx) => `${wrap(this._emit(ctx))}.${prop}`)
    }

    /** `expr * other` — scalar/vector multiply (WGSL infix; both operands parenthesised). */
    mul(other: Expr | number): Expr {
        return new Expr((ctx) => `(${this._emit(ctx)} * ${emitOperand(other, ctx)})`)
    }

    /** `expr + other`. */
    add(other: Expr | number): Expr {
        return new Expr((ctx) => `(${this._emit(ctx)} + ${emitOperand(other, ctx)})`)
    }
}

/** True when a value is a KitExpr. */
export function isExpr(value: unknown): value is Expr {
    return value instanceof Expr
}

function wrap(text: string): string {
    // Only wrap compound expressions; a bare identifier/access chain needs no parens.
    return /^[\w.$]+$/.test(text) ? text : `(${text})`
}

function emitOperand(operand: Expr | number, ctx: EmitContext): string {
    if (typeof operand === 'number') return formatFloat(operand)
    return operand._emit(ctx)
}

/** WGSL float literal — always with a decimal so it types as f32, never i32. */
export function formatFloat(n: number): string {
    if (!Number.isFinite(n)) return n > 0 ? '3.4e38' : n < 0 ? '-3.4e38' : '0.0'
    return Number.isInteger(n) ? `${n}.0` : `${n}`
}

/** A texture handle the composer/shaders sample. Backed by a texture bind-group entry. */
export interface KitTexture {
    /** The bind-group layout entry key (e.g. `rtt_0`, `media_2`, `video_0`). */
    readonly key: string
    /**
     * A texture sample at the given UV. For a regular sampled texture this is
     * `textureSample(tex.$.<key>, samp.$.<sampler>, uv)`; for an external (video/webcam)
     * texture it is `textureSampleBaseClampToEdge(ext.$.<key>, samp.$.<sampler>, uv)` (the
     * only sampling builtin external textures support — clamp-to-edge is implicit).
     */
    sample(uv: Expr, sampler?: SharedSamplerName): Expr
    /**
     * An explicit-LOD sample (`textureSampleLevel(…, 0)`): identical to `sample` on a
     * single-mip texture, but legal in NON-UNIFORM control flow (inside a `guarded` branch)
     * where the implicit-derivative `textureSample` is a WGSL validation error. External
     * textures alias `sample` (`textureSampleBaseClampToEdge` takes no derivatives and is
     * already uniform-flow-safe).
     */
    sampleLevel(uv: Expr, sampler?: SharedSamplerName): Expr
    /** The raw accessor expression (`tex.$.<key>` / `ext.$.<key>`) for shader-body fn args. */
    accessor(): Expr
    /**
     * The texture's pixel dimensions as a `vec2f` Expr (`vec2f(textureDimensions(<accessor>))`).
     * For external textures this is the CURRENT video frame's size, read on the GPU every frame
     * (no CPU-tracked `videoWidth`/`videoHeight` uniforms needed).
     */
    dimensions(): Expr
}

/** The four shared samplers the kit exposes (kit-owned fixed sampler layout). */
export type SharedSamplerName = 'linearClamp' | 'nearestClamp' | 'linearRepeat' | 'nearestRepeat'

/**
 * The resolved remap window of a map-driven prop (post dynamic-bound resolution) — the same
 * five values the fragment path reads from the `_map_<prop>_*` synthetic uniforms. `window()` on a
 * `GpuMapInfo` re-reads this each call so a per-frame change (e.g. a resized px-unit output bound)
 * is reflected without a recompose.
 */
export interface GpuMapWindow {
    inputMin: number
    inputMax: number
    outputMin: number
    outputMax: number
    curve: number
}

/**
 * The "compute-map interplay" handle — everything a COMPUTE hook needs to reproduce the
 * fragment path's map resolve (sample the map source → extract the channel → remap) for a
 * map-driven prop, per compute pixel. Obtained via `GpuFragmentParams.getMapInfo(prop)`; returns
 * null when `prop` has no active `type:'map'` driver.
 */
export interface GpuMapInfo {
    /**
     * The map SOURCE rendered to an RTT boundary, as a KitTexture. Bind it LATE inside the compute
     * node's `bindInputs` by resolving `sourceTexture.key` (exactly like the child RTT — the pass
     * manager allocates it after composition), then feed the bound texture to the fill kernel.
     */
    sourceTexture: KitTexture
    /** Channel to extract as the driving scalar ('luminance'|'luminanceInverted'|'alpha'|'alphaInverted'). */
    channel: string
    /** The live resolved remap window (re-read each call). Matches the fragment `_map_<prop>_*` uniforms. */
    window(): GpuMapWindow
}

/** Options for a shader-owned media texture (image/canvas/DOM upload target). */
export interface GpuMediaTextureOptions {
    width: number
    height: number
    /** Base (linear) format. Default `rgba8unorm`. */
    format?: GPUTextureFormat
    /** Add an `-srgb` viewFormat + expose `createSrgbView()` (rarely needed — decode in-shader). */
    srgb?: boolean
    mipLevelCount?: number
    label?: string
}

/**
 * Options for a shader-owned DATA texture — a fixed-size GPU texture uploaded from a raw
 * CPU `ArrayBufferView` (via `device.queue.writeTexture`, NOT `copyExternalImageToTexture`), for
 * float fields the shader computes/loads rather than image content. The SVG-SDF samplers use this
 * (an `r16float`/`r32float` field loaded from a `.bin`). Distinct from a media texture only in the
 * upload path + the float-only `format`; the returned handle is the same `GpuMediaTexture` shape.
 */
export interface GpuDataTextureOptions {
    width: number
    height: number
    /**
     * `r16float` (single-channel, always linear-filterable) or `r32float` (needs `float32-filterable`),
     * `rgba16float` for a filterable multi-channel field texture (ChromaFlow's flow+density,
     * PixelThrow's 2D displacement) — half-encode each texel with kit `toHalfFloat` before upload —
     * or `rgba8unorm` for byte-valued lookup data (FilmStock's color LUT atlas).
     */
    format: 'r16float' | 'r32float' | 'rgba16float' | 'rgba8unorm'
    /** Optional initial pixel data. */
    data?: ArrayBufferView
    label?: string
}

/**
 * A shader-owned media texture: a fixed-size GPU texture the shader uploads image/canvas/video-frame
 * content into (`write`), and whose raw `GPUTexture` it can reach (`unwrap`, for HTMLInCanvas's
 * `queue.copyElementImageToTexture`). Media textures are STATIC-sized — to change the size (a new
 * image at its native resolution, a re-rastered text run, a resized DOM capture) create a NEW one and
 * point the registered getter at it (see `registerMediaTexture`); the pass manager rebuilds the
 * sampling pass's bind group when it sees the backing texture change (no recompose).
 */
export interface GpuMediaTexture {
    /** The underlying TypeGPU texture — pass to `registerMediaTexture`'s getter / bind path. */
    readonly texture: unknown
    readonly width: number
    readonly height: number
    /** Upload pixel content (an ImageBitmap / HTMLCanvasElement / ImageData / TypedArray). */
    write(source: unknown): void
    /** The raw `GPUTexture` (for `queue.copyElementImageToTexture`). */
    unwrap(): GPUTexture
    /** Regenerate the mip chain from mip 0 — a no-op unless `mipLevelCount` > 1. */
    generateMipmaps(): void
    /** Destroy the GPU texture (WebGPU keeps it alive for already-submitted frames). */
    destroy(): void
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Fragment / compute / uvRemap builder params (KitExpr currency)
// ═══════════════════════════════════════════════════════════════════════════════════════

/** System/context accessors a fragment builder reads (the `_sys` block + the uv varying). */
export interface KitCtx {
    /** The fullscreen-pass UV varying (`in.uv`), Y-orientation resolved once by the composer. */
    readonly uv: Expr
    /** `_sys.time` — CPU-accumulated seconds. */
    readonly time: Expr
    /** `_sys.viewportSize` — device-pixel backing-buffer size (vec2f). */
    readonly viewportSize: Expr
    /** `_sys.logicalViewportSize` — CSS-pixel authored-frame size (vec2f). */
    readonly logicalViewportSize: Expr
    /** `_sys.aspect` — width / height. */
    readonly aspect: Expr
    /** `_sys.pointer` — pointer position in UV space (vec2f). */
    readonly pointer: Expr
}

/**
 * Params passed to a fragment builder. The builder runs at COMPOSITION time and returns a
 * `KitExpr` — inside it authors call their pre-transpiled `'use gpu'` fns, wiring
 * `props`/`uniforms`/`ctx`/textures as KitExpr arguments.
 */
export interface GpuFragmentParams {
    /**
     * The node's props as a single struct accessor (`uni.$.uniforms.n_<id>`), OR a local
     * driver-overridden copy when the node has active map/mouse/auto drivers. Pass this as
     * the `props` argument of the shader's `'use gpu'` fn.
     */
    props: Expr
    /**
     * Record that this composition contains USER-AUTHORED WGSL (a `wgsl` body). The renderer
     * then builds and first-renders the composition inside a validation error scope, so a
     * body that fails to compile marks only that composition broken instead of tripping the
     * fatal uncaptured-error limit. Set by the std `wgsl` lowering; hosts without a renderer
     * (tests, offline resolve) may omit it.
     */
    noteCustomWgsl?: () => void
    /**
     * Per-prop KitExpr accessors (`props.<field>`), keyed by prop name. For builders that
     * reference individual fields (aspect math) rather than the whole struct. Driver-overridden
     * props resolve to the local copy's field.
     */
    uniforms: Record<string, Expr>
    /**
     * Per-prop CPU VALUES (post-transform), keyed by prop name — the mirror of `uniforms`, for
     * COMPILE-TIME BRANCHING. A `compileTime` prop (e.g. `edges`, `colorSpace`, a stop count)
     * is read here as a plain number/enum so the builder can JS-branch and emit specialised
     * WGSL. These values are part of the pipeline structural hash, so a change recomposes → the
     * builder re-runs. Scalars come through as numbers; vector fields as the store's live view
     * (rarely needed at build time — prefer the `uniforms`/`props` accessors for GPU math).
     */
    propValues: Record<string, unknown>
    /** Composed children — an RTT texture sample expression when this node consumes a child. */
    childNode?: Expr
    /** System/context accessors. */
    ctx: KitCtx
    /** Composed UV from a UV-propagating parent (`uvContext ?? ctx.uv` idiom). */
    uvContext?: Expr
    /**
     * True on a `providesUVContextViaCompute` LINK when `tryComposeViaUVContext` has PRE-EVALUATED
     * the child generator at this link's looked-up UV (so `childNode` is already the color at the right
     * UV — composite it directly). Absent → the link re-samples its RTT child at the looked-up UV.
     * Surface3D is the first consumer.
     */
    uvPropagationActive?: boolean
    /** Pre-computed bbox clip mask (float 1 inside / 0 outside) for early-discard. */
    bboxMask?: Expr
    /** Box pixel dimensions as the effective viewport (generator resize-fit). */
    effectiveViewportSize?: Expr
    /** CSS-pixel authored-frame size, for resolution-stable feature counts. */
    logicalViewportSize?: Expr
    /** Compute-pass outputs (storage textures / buffers) forwarded to the fragment. */
    computeOutputs?: Record<string, KitTexture | GpuStorageBuffer>
    /** Live child-shape bounds (Repeater). */
    childBoundsParams?: GpuBoundsParams
    /** This node's own active bbox bounds (Repeater). */
    ownBoundsParams?: GpuBoundsParams
    /**
     * Register a texture the shader BODY samples via a fn-arg (media textures).
     *
     * @deprecated Zero consumers — every shader reaches a texture through the `register*` family
     * (`registerTexture` / `registerExternalTexture` / `registerStorageTexture` / `convertToTexture`),
     * which allocates the binding as well as declaring it. Kept only for contract stability; use a
     * `register*` call instead.
     */
    declareTexture(tex: KitTexture): void
    /**
     * Register an EXTERNAL texture (video/webcam) the shader samples. Returns a KitTexture
     * whose `.sample(uv)` emits `textureSampleBaseClampToEdge(ext.$.<key>, …)` and whose
     * `.dimensions()` emits `textureDimensions(ext.$.<key>)` (the live frame size). `getSource`
     * is polled EACH FRAME by the pass manager: it returns the live `HTMLVideoElement`/`VideoFrame`
     * when a decodable frame exists, or `null` when unavailable — in which case the pass is skipped
     * for that frame (nothing to bind). The external texture is re-imported + its group-3 bind group
     * rebuilt every frame; the emitted WGSL is fixed, so readiness never triggers a recompile.
     */
    registerExternalTexture(getSource: () => unknown): KitTexture
    /** Convert a composed expression to an RTT texture (registers a boundary). */
    convertToTexture(expr: Expr): KitTexture
    /**
     * Register a compute-WRITTEN texture (created with `storage`+`sampled` usage) so the
     * fragment / uvRemap can sample it via the returned KitTexture. Registers a `kind:'compute'`
     * texture binding; the pass manager binds the given texture into the sampling pass's texture
     * group. This is how a compute hook's storage-texture outputs (Blur's blurred buffer, Glow's
     * bloom pyramid, …) reach the fragment.
     */
    registerComputeTexture(texture: unknown): KitTexture
    /**
     * Create a shader-owned media texture (fixed-size GPU upload target). Wired to the
     * renderer's TextureManager, so it's counted + destroyed with the renderer. A media shader
     * (ImageTexture/Text/HTMLInCanvas) creates a placeholder here at composition, uploads content via
     * `write` / `queue.copyElementImageToTexture`, and swaps to a fresh (differently-sized) texture
     * as content loads — see `registerMediaTexture`.
     */
    createMediaTexture(options: GpuMediaTextureOptions): GpuMediaTexture
    /**
     * Create a shader-owned DATA texture (r16float/r32float) uploaded from a raw
     * `ArrayBufferView`. Same lifecycle + return shape as `createMediaTexture`, but the write path is
     * `queue.writeTexture` (raw float data), not an image copy. The SVG-SDF samplers create one here,
     * async-load the `.bin` field into it, and register it via `registerMediaTexture` for the fragment
     * to sample. Wired to the renderer's TextureManager (`createDataTexture`); throws if none injected.
     */
    createDataTexture(options: GpuDataTextureOptions): GpuMediaTexture
    /**
     * Register a media texture the shader BODY samples. `getTexture` returns the CURRENT backing
     * TypeGPU texture each time it is read — a media shader keeps a mutable `current` and reassigns it
     * when it swaps to a differently-sized texture (image loaded at native size, text re-rastered, DOM
     * resized). The pass manager reads the getter every frame and rebuilds ONLY the sampling pass's
     * group-1 bind group when the backing texture identity changes (cheap — no pipeline rebuild, no
     * recompose). Returns the `KitTexture` for the body's `.sample(uv)` / `.dimensions()`.
     */
    registerMediaTexture(getTexture: () => unknown): KitTexture
    /**
     * The LIVE post-transform CPU value of a prop, re-read on every call — for per-frame
     * compute updates (Blur's radius from intensity). Unlike `propValues` (a compose-time snapshot
     * for structural branching), this reads the node's live uniform handle, so a reactive value
     * that changed since composition (without recompose) is reflected. Returns undefined for an
     * unknown / cpu-only prop. `getCpuValue('_animTime')` (and `'_animTime_<key>'`) returns the
     * node's animated-time accumulator for the current frame — valid inside `onBeforeRender` /
     * compute hooks on a live renderer, undefined in GPU-free composition.
     */
    getCpuValue(prop: string): unknown
    /**
     * Compute-map interplay. For a MAP-driven prop (`metadata.maps[prop].type === 'map'`),
     * returns everything a compute pass needs to reproduce the fragment path's map resolve per
     * compute pixel — the map SOURCE texture (a late-bound RTT KitTexture, bound in the compute
     * node's `bindInputs`), the channel to extract, and a live remap-window reader. Returns null
     * when `prop` has no active map driver (the caller falls back to its static path). The source
     * RTT is shared with the fragment override (one boundary per node+prop). See Blur's compute
     * hook for the reference consumer.
     */
    getMapInfo(prop: string): GpuMapInfo | null
    /**
     * LAYER-typed prop resolver. For a prop whose VALUE is another layer's custom id (ui type
     * 'layer' — the layer-selector prop), returns that layer's composed output as a shared RTT
     * KitTexture — the same boundary the map machinery registers, so the texture is shared when
     * the layer is also rendered normally or consumed by a prop map. Returns null when the prop
     * is empty or the id doesn't resolve, so the shader degrades to its no-source path. Declare
     * the prop `compileTime`: the selected id is then part of the structural hash, so switching
     * layers recomposes and this re-resolves. DisplacementMap is the reference consumer.
     */
    getLayerTexture(prop: string): KitTexture | null
    /**
     * Write an `extraFields` field's CPU value (a scalar or a vecNf's components) on this
     * node's LIVE handle — call from an `onBeforeRender` callback to drive a derived uniform each
     * frame (Blob's normalized light direction). Patches the packed buffer for the next flush, so
     * the value is picked up the SAME frame (frame order: onBeforeRender → flush → render). No-op
     * for an undeclared name or before a composition is bound. Read the same field back in the
     * builder via `uniforms.<name>`.
     */
    setExtraField(name: string, value: number | number[]): void

    // Lifecycle.
    onCleanup: (cb: CleanupCallback) => void
    onBeforeRender: (cb: RenderCallback) => void
    onAfterRender: (cb: RenderCallback) => void
    onResize: (cb: ResizeCallback) => void
    canvas: HTMLCanvasElement
    dimensions: {width: number; height: number}
    domCanvas?: HTMLCanvasElement
    /** Raw device + root for HTMLInCanvas / texture uploads. */
    gpu: {device: GPUDevice; root: import('typegpu').TgpuRoot}
}

/** UV-space rectangle bounds as KitExpr accessors (Repeater). */
export interface GpuBoundsParams {
    centerX: Expr
    centerY: Expr
    halfWidth: Expr
    halfHeight: Expr
    rotation: Expr
}

/** A storage buffer produced by a compute pass (particle/simulation state). */
export interface GpuStorageBuffer {
    readonly key: string
    accessor(): Expr
}

/** A fragment builder — runs at composition, returns the node's color expression. */
export type GpuFragment = (params: GpuFragmentParams) => Expr

/** Pure coordinate bend (KitExpr currency). */
export type GpuUvRemap = (ctx: {
    uv: Expr
    mask: Expr
    uniforms: Record<string, Expr>
    /** Per-prop CPU values (post-transform) for compile-time branching — mirror of `uniforms` (see GpuFragmentParams). */
    propValues: Record<string, unknown>
    props: Expr
    aspect: Expr
    onBeforeRender: (cb: RenderCallback) => void
    computeOutputs?: Record<string, KitTexture | GpuStorageBuffer>
}) => {uv: Expr; mask: Expr}

/** One compute step — a dispatchable kit pipeline or an inline thunk. */
export type GpuComputeStep = import('./compute').ComputeStep

/** A compute node — KitTexture/GpuStorageBuffer outputs. */
export type GpuComputeNode = (params: GpuFragmentParams) => {
    outputs: Record<string, KitTexture | GpuStorageBuffer>
    getComputeNodes: (frameParams: unknown) => GpuComputeStep[] | null
    /**
     * (Re)bind compute-pass INPUTS that read composition RTT boundaries. A compute
     * hook that consumes its child via `convertToTexture(childNode)` can only build its input
     * bind group once the physical RTT texture exists — and that is allocated by the pass manager
     * AFTER composition. The composer records this callback per compute node; the pass manager
     * invokes it right after allocating RTT textures (and again on every recompose), passing a
     * resolver from a texture key (the `KitTexture.key` returned by `convertToTexture`) to the
     * bound texture. Blur is the first shader to need this; see gpu/passManager `setComposition`.
     */
    bindInputs?: (resolve: (key: string) => {texture: unknown} | undefined) => void
} | null

/**
 * Custom per-prop sample UVs for prop maps (DotGrid, Grid). Returns a per-prop map of
 * the UV a map-driven prop's SOURCE texture should be sampled at (e.g. the cell-CENTER UV for a
 * tiled pattern, so a mapped `dotSize`/`thickness` reads one value per cell rather than varying
 * per-fragment and clipping the shape at source boundaries). Any prop NOT in the returned map
 * falls back to the fragment UV.
 *
 * Receives `ctx` (the same `KitCtx` a fragment builder gets) so the cell-center math can read
 * `ctx.uv` / `ctx.viewportSize`. `uniforms` are the node's BASE (non-driver-overridden) prop
 * accessors (the sample-UV math reads static prop values like `density`/`cells`/`rotation`,
 * never the mapped value being resolved).
 */
export type GpuMapSampleUVs = (params: {
    uniforms: Record<string, Expr>
    ctx: KitCtx
}) => Record<string, Expr>

// ═══════════════════════════════════════════════════════════════════════════════════════
// GpuShaderDefinition — the whole contract
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * A field the uniform store registers for this shader. These are derived from the shader's
 * `props` (default + transform). The `propsSchema` is the composer's view of the node's struct;
 * individual `FieldInit`s carry the transform + cpu-only marking for the store.
 */
export interface GpuPropField {
    name: string
    /** WGSL schema (omit to infer from the transformed default). */
    schema?: AnyWgslData
    /** CPU transform (transformColor/transformPosition/…) — retained on the FieldHandle. */
    transform?: (value: unknown) => unknown
    /** CPU-only prop (string/JSON/select) — no struct field. */
    cpu?: boolean
}

/**
 * The GPU shader definition. Declarative fields carry the shader's static metadata;
 * GPU-code members take KitExpr currency.
 */
export interface GpuShaderDefinition<T extends ComponentProps = ComponentProps> {
    // ── Declarative ───────────────────────────────────────────────────────────────────
    name: string
    /**
     * Former public names of this shader (a rename's back-compat surface). Each old name keeps
     * working everywhere a name is currency — framework named exports (alias re-exports of the
     * SAME component, no duplication), `shaders/core/<OldName>` deep imports, JS-package `type`
     * strings, registry lookups (`getShaderByName`), and saved presets (dotcom migrates the type
     * to the canonical name at load) — but is never listed in UI surfaces (component picker,
     * docs, metadata), which enumerate canonical names only. Wired through the registry +
     * component generators; see HTMLInCanvas (né DOMTexture) for the reference rename.
     */
    deprecatedNames?: string[]
    category?: string
    description?: string
    /**
     * A content fingerprint for definitions that can change WITHOUT changing name — a
     * user-authored `wgsl` body edited live. The composer folds it into the structural hash, so
     * swapping in a new definition object under the same name recomposes instead of hitting
     * the pipeline cache. `defineShader` stamps it for `wgsl` bodies; library shaders omit it.
     */
    revision?: string
    requiresRTT?: boolean
    requiresChild?: boolean
    // A GENERATOR (requiresChild falsy) that can OPTIONALLY consume a nested child — it works fully
    // standalone, but when a child is nested it uses it (e.g. ReactionDiffusion reads the child's
    // luminance to bias the sim). Purely an editor hint: it makes the node a valid nest/drop target
    // WITHOUT ghosting it as "needs input" when childless. Does NOT affect compositing (that stays
    // keyed on requiresChild) — the fragment decides whether the child is shown.
    acceptsOptionalChild?: boolean
    blendWithChildren?: boolean
    usesPointer?: boolean
    acceptsUVContext?: boolean
    providesUVContextViaCompute?: boolean
    capturesDOM?: boolean
    wantsBoundsParams?: boolean
    boundingBoxDeclaration?: BoundingBoxDeclaration
    /**
     * MEDIA shaders only: how this shader's entry in the natural-size registry
     * (`utilities/naturalSize.ts`) is keyed, so layout can measure it at its intrinsic pixel size —
     * the `<img naturalWidth>` answer. `{fromProp}` reads the key from a URL-valued prop (empty /
     * non-string ⇒ unmeasurable, out of flow until it loads); `{fixed}` is for a source with no URL
     * (the webcam). Purely declarative: the shader still PUBLISHES its dimensions by calling
     * `registerNaturalSize`, and `measureNode` reads this to know which key to look them up under.
     * Omit it for everything that is not media — that is what makes a layer unmeasurable by default.
     */
    naturalSizeKey?: {fromProp: string} | {fixed: string}
    experimental?: ComponentProps
    /**
     * Declares per-node ANIMATED TIME. When set, the renderer registers a synthetic `_animTime`
     * f32 field on this node's struct and advances it CPU-side every frame by
     * `deltaTime * <speed prop>.value` (speed=0 PAUSES — no rewind). `speed` names the prop the
     * accumulation rate reads.
     * Read the accumulated value inside the `fragment` / `uvRemap` builder via the porters
     * facade's `animatedTime(params[, seedName])` — do NOT read `ctx.time` for speed-controlled
     * motion (`ctx.time` is the global, un-paused, un-scaled clock). Seed is added on the GPU by
     * the helper, so it stays an ordinary uniform and is not named here.
     */
    animatedTime?: {speed: string}
    /**
     * ADDITIONAL per-node animated-time clocks beyond the primary `_animTime`, for shaders
     * with two+ INDEPENDENT flow clocks (FlowField: a flow-drift speed AND an evolution speed).
     * Each entry `{<key>: '<speedProp>'}` registers a synthetic `_animTime_<key>` f32 field,
     * advanced every frame by `deltaTime * <speedProp>.value` on its OWN accumulator (keyed on the
     * speed prop, distinct from the primary), exactly like `_animTime`. speed=0 pauses. Read the
     * accumulated value in the builder via `animatedTime(params, seed?, '_animTime_<key>')`. Not a
     * prop → never in the structural hash, so a per-frame change patches in place.
     */
    extraAnimatedTimes?: Record<string, string>
    /**
     * EXTRA per-node uniform fields the shader DERIVES on the CPU each frame and READS on the
     * GPU — for values that are neither a prop nor a system uniform (Blob's normalized light
     * direction, precomputed so `normalize()` doesn't run per fragment). A shader cannot add
     * ad-hoc uniforms inside its builder, so it DECLARES them here. Each entry registers an
     * f32/vecNf field on this node's `n_<id>` struct (exactly like `_animTime`), seeded with
     * `initial`. The builder reads it as `params.uniforms.<name>` (a GPU accessor Expr); an
     * `onBeforeRender` callback writes it each frame via `params.setExtraField(<name>, value)`.
     * The value changes per frame WITHOUT recomposing (it is not a prop, so not in the structural
     * hash) — same contract as `_animTime`.
     */
    extraFields?: Record<string, {schema: AnyWgslData; initial: number | number[]}>
    props: {[K in keyof T]: PropConfig<T[K]>}

    // ── GPU code (KitExpr currency) ───────────────────────────────────────────────────
    /**
     * The node's per-node uniform struct (the `n_<id>` sub-struct of the packed buffer).
     * The composer nests it under a runtime key; shader fns take an instance as their
     * `props` arg. Optional: when omitted the composer infers it from `propFields`.
     */
    propsSchema?: WgslStruct
    /** Store field descriptors (transforms / cpu-only markers) derived from `props`. */
    propFields?: GpuPropField[]
    /** The fragment builder. */
    fragment: GpuFragment
    /** Optional pure-analytic UV bend. */
    uvRemap?: GpuUvRemap
    /** Optional compute pass. */
    compute?: GpuComputeNode
    /** Optional custom per-prop sample UVs for prop maps. */
    mapSampleUVs?: GpuMapSampleUVs
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// RegistryView — the minimal node-registry shape the composer walks
// ═══════════════════════════════════════════════════════════════════════════════════════
//
// The composer does NOT own the live registry — frame.ts / index.ts wire the real one in.
// It reads this minimal projection: enough to walk parents→children, sort by render order,
// read blend/mask/opacity/transform/bbox metadata, and reach each node's shader definition +
// uniform handles.

/** A driver config attached to a prop (map/mouse/auto) — opaque here; resolved by the composer. */
export type PropDriverConfig = import('../types').PropDriver

/** One node as the composer sees it. */
export interface RegistryNode {
    /** Internal unique node id. */
    id: string
    /** Custom id used for mask/map source references (`metadata.id`, sanitised). */
    customId?: string
    componentName: string
    parentId: string | null
    definition: GpuShaderDefinition
    /** Blend/opacity/visible/mask/maps/transform/boundingBox — the composition metadata. */
    metadata: NodeMetadata
    /**
     * The node's uniform handles keyed by prop name (from the uniform store). The composer
     * reads `.accessorPath` / `store.gpuAccessor(handle)` to emit GPU references, and reads
     * `.value` for compile-time-relevant prop values (hashing, bbox math).
     */
    handles: Record<string, UniformHandleView>
    /** Per-node DOM-capture host canvas (HTMLInCanvas `capturesDOM`) — forwarded to `params.domCanvas`. */
    domCanvas?: HTMLCanvasElement
}

/** The subset of a FieldHandle / ArrayFieldHandle the composer needs. */
export interface UniformHandleView {
    readonly accessorPath: string
    readonly cpu: boolean
    readonly value: unknown
}

/** The composer's read-only view over the node registry. */
export interface RegistryView {
    rootId: string | null
    /** All nodes by internal id. */
    getNode(id: string): RegistryNode | undefined
    /** Direct children of a node, unsorted. */
    getChildren(parentId: string): RegistryNode[]
    /** Resolve a custom id (mask/map source) to an internal node id. */
    resolveCustomId(customId: string): string | null
    /** The uniform store (for `gpuAccessor` + the combined layout). */
    store: {
        gpuAccessor(handle: {accessorPath: string}, layoutVar?: string): string
        readonly layout?: TgpuBindGroupLayout
    }
}

// Re-exports so consumers import contract types from one place.
export type {BlendMode, MaskType}
