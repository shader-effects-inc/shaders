/**
 * `composeNodeTree` — walks the node registry and emits a `CompositionIR`.
 *
 * The composition algorithm (the replace-vs-blend / mask / bbox / transform / driver /
 * analytic-fold logic) assembles a tree of `KitExpr`s and records RTT boundaries.
 *
 * Output: a `CompositionIR` — an ordered list of GPU passes (compute steps, RTT passes in
 * leaves→root order, one final pass with the tonemap + sRGB-OETF tail). Each pass carries a
 * built `tgpu.fragmentFn` entry (raw-WGSL glue + a `$uses` externals map) plus the texture
 * keys it reads. The pass manager (passManager.ts) allocates the RTT textures and executes.
 *
 * Glue emission:
 *   - uniform refs → `store.gpuAccessor(handle, 'uni')` → `uni.$.uniforms.n_x.field`
 *     (NEVER hand-built; the `.uniforms` entry-key segment is the store's bind-group key).
 *   - blend folds → `blendModes[mode](base, overlay, opacity)` externals.
 *   - masks → `maskFunctions[type](target, mask)` externals.
 *   - textures → `textureSample(tex.$.<key>, samp.$.<sampler>, uv)` referencing the
 *     composition texture layout + the kit shared-sampler layout directly (no slots).
 *   - `tgpu.resolve`'s linker rewrites those accessor prefixes to the real binding names.
 */
import tgpu, {type TgpuBindGroupLayout, type TgpuFragmentFn} from 'typegpu'
import * as d from 'typegpu/data'
import {blendModes, unpremultiplyAlpha as unpremultiplyAlphaFn, type BlendMode} from './kit/blend'
import {maskFunctions, type MaskType} from './kit/mask'
import {tonemapFns, linearToSrgb, type ToneMappingMode} from './kit/tonemap'
import {nodeKey} from './uniformStore'
import {debugError} from './support'
import {needsTransformation} from './kit/uvTransform'
import {isIdentityBoundingBox} from '../utilities/boundingBox'
import type {ComputeStep} from './compute'
import {
    Expr,
    formatFloat,
    type EmitContext,
    type KitCtx,
    type KitTexture,
    type SharedSamplerName,
    type GpuFragmentParams,
    type GpuBoundsParams,
    type GpuMediaTexture,
    type GpuMapInfo,
    type GpuStorageBuffer,
    type RegistryNode,
    type RegistryView,
} from './contract'

// ═══════════════════════════════════════════════════════════════════════════════════════
// Shared sampler layout (kit ABI). Composition-independent, pinned to group 2. Both
// the glue and shader bodies reference this so a build-time shader fn can capture a sampler
// accessor (TGSL rejects sampler fn params). passManager binds the real GPUSamplers to it.
// ═══════════════════════════════════════════════════════════════════════════════════════
export const SHARED_SAMPLER_LAYOUT = tgpu
    .bindGroupLayout({
        linearClamp: {sampler: 'filtering'},
        nearestClamp: {sampler: 'filtering'},
        linearRepeat: {sampler: 'filtering'},
        nearestRepeat: {sampler: 'filtering'},
    })
    .$idx(2)

/** Group indices the composer pins. uniforms=0, textures=1, samplers=2, external=3. */
export const BIND_GROUPS = {uniforms: 0, textures: 1, samplers: 2, external: 3} as const

// ═══════════════════════════════════════════════════════════════════════════════════════
// KitExpr factories — the "small string-builder"
// ═══════════════════════════════════════════════════════════════════════════════════════

/** A raw WGSL fragment (identifier / literal / pre-formed expression). */
export function expr(wgsl: string): Expr {
    return new Expr(() => wgsl)
}

/** A WGSL float literal Expr (decimal-safe). */
export function floatE(n: number): Expr {
    return new Expr(() => formatFloat(n))
}

/** A call to a `tgpu.fn` external: `<name>(arg0, arg1, …)`. Dedupes the fn by identity. */
export function call(fn: unknown, hint: string, args: Expr[]): Expr {
    return new Expr((ctx) => {
        const name = ctx.external(fn, hint)
        return `${name}(${args.map((a) => a._emit(ctx)).join(', ')})`
    })
}

/** `vec4f(rgb, a)` or `vec4f(x, y, z, w)`. */
export function vec4(...parts: (Expr | number)[]): Expr {
    return new Expr((ctx) => `vec4f(${parts.map((p) => operand(p, ctx)).join(', ')})`)
}

/** `mix(a, b, t)` — WGSL built-in, no external. */
export function mixExpr(a: Expr, b: Expr, t: Expr | number): Expr {
    return new Expr((ctx) => `mix(${a._emit(ctx)}, ${b._emit(ctx)}, ${operand(t, ctx)})`)
}

/**
 * A WGSL fixed-size array constructor `array<elemType, N>(e0, e1, …)`, for passing a run of
 * builder-computed scalar/vec Exprs into a `tgpu.fn` that takes a `d.arrayOf(...)` param. Used
 * by Dither's Floyd-Steinberg block diffusion (64 RTT-sampled luminances → the diffusion body's
 * `array<f32,64>` param). `elemType` is the WGSL element type (e.g. `f32`, `vec2f`).
 */
export function arrayExpr(elemType: string, elems: Expr[]): Expr {
    return new Expr((ctx) => `array<${elemType}, ${elems.length}>(${elems.map((e) => e._emit(ctx)).join(', ')})`)
}

/**
 * Hoist an Expr into a `let` local, emitted (memoised) once per fragment. Without this,
 * `Expr` composition re-emits the whole subtree at every use site (`member()` has no CSE) —
 * a builder that reads several members of one expensive call (a folded distortion coordinate,
 * a 16-load field sample) must hoist it or pay the full subtree once per read.
 */
let hoistCounter = 0
export function asLocal(e: Expr, hint: string): Expr {
    const id = hoistCounter++
    return new Expr((ctx) =>
        ctx.memo(`local:${id}`, () => {
            const name = ctx.freshLocal(hint)
            ctx.statement(`let ${name} = ${e._emit(ctx)};`)
            return name
        }),
    )
}

/** Transparent-black `vec4f(0.0, 0.0, 0.0, 0.0)`. */
export const ZERO = expr('vec4f(0.0, 0.0, 0.0, 0.0)')
/** `vec4f(1.0, 1.0, 1.0, 0.0)` — the first-child blend base. */
export const WHITE0 = expr('vec4f(1.0, 1.0, 1.0, 0.0)')

function operand(o: Expr | number, ctx: EmitContext): string {
    return typeof o === 'number' ? formatFloat(o) : o._emit(ctx)
}

// ── Composition primitives (blend / mask / mix) ─────────────────────────────────────────

/** `blendModes[mode](base, overlay, opacity)`. */
export function applyBlend(base: Expr, overlay: Expr, mode: BlendMode, opacity: Expr): Expr {
    const fn = blendModes[mode] ?? blendModes.normal
    return call(fn, `blend_${sanitizeName(mode)}`, [base, overlay, opacity])
}

/** `maskFunctions[type](target, mask)`. */
export function applyMaskFn(target: Expr, mask: Expr, type: MaskType): Expr {
    const fn = maskFunctions[type] ?? maskFunctions.alpha
    return call(fn, `mask_${sanitizeName(type)}`, [target, mask])
}

function sanitizeName(s: string): string {
    return s.replace(/[^a-zA-Z0-9_]/g, '_')
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// CompositionIR
// ═══════════════════════════════════════════════════════════════════════════════════════

/** A built fragment entry + its resolvable form (for GPU-free snapshots) + texture reads. */
export interface FragmentSpec {
    /** The raw-WGSL body block (`{ …statements… return …; }`). */
    body: string
    /** The `$uses` externals map (fn objects + bind-group layouts). */
    externals: Record<string, unknown>
    /** The built `tgpu.fragmentFn` entry (GPU-free to build + resolve). */
    entry: TgpuFragmentFn
    /** Composition texture keys this pass samples (NEVER its own RTT output — no r/w hazard). */
    reads: string[]
    /** External (video/webcam) texture keys this pass samples. */
    externalReads: string[]
    /** Whether this pass binds the shared sampler group. */
    usesSamplers: boolean
    /** This pass's OWN texture bind-group layout (only its `reads` keys), pinned to group 1. */
    textureLayout?: TgpuBindGroupLayout
    /** This pass's OWN external-texture layout (only its `externalReads` keys), group 3. */
    externalLayout?: TgpuBindGroupLayout
}

/** An RTT pass: render `fragment` into the texture bound at `textureKey`. */
export interface RttPassSpec {
    textureKey: string
    fragment: FragmentSpec
}

/** A registered texture the composition needs bound (RTT target or shader-owned media). */
export interface TextureBinding {
    key: string
    kind: 'rtt' | 'media' | 'compute'
    /** For media/compute textures: the shader-provided resource passManager binds. Absent for RTT. */
    resource?: unknown
}

/** A registered external texture (video/webcam) — per-frame re-import. */
export interface ExternalTextureBinding {
    key: string
    /** Returns the live source each frame (HTMLVideoElement / VideoFrame). */
    getSource: () => unknown
}

/** A compute step provider for one node (drained per frame by frame.ts / passManager). */
export interface ComputeSpec {
    nodeId: string
    getComputeNodes: (frameParams: unknown) => ComputeStep[] | null
    /**
     * (Re)bind this compute pass's RTT-boundary inputs once the pass manager has
     * allocated them (see contract `GpuComputeNode.bindInputs`). Called in `setComposition`.
     */
    bindInputs?: (resolve: (key: string) => {texture: unknown} | undefined) => void
}

export interface CompositionIR {
    /** Compute passes, in registry order (dispatched first each frame). */
    computeSteps: ComputeSpec[]
    /** RTT passes, leaves→root (a valid topological order by construction). */
    rttPasses: RttPassSpec[]
    /** The final fullscreen pass → canvas view (tonemap + sRGB OETF tail). */
    finalPass: FragmentSpec
    /** All texture bindings (RTT targets + shader-owned media/compute textures). */
    textures: TextureBinding[]
    /** External (video/webcam) texture bindings. */
    externalTextures: ExternalTextureBinding[]
    /**
     * Composition-wide bind-group layouts. `uniforms` (group 0) + `samplers` (group 2) are
     * shared across all passes; texture (group 1) + external (group 3) layouts are per-pass
     * and live on each `FragmentSpec` (only the keys that pass reads — never its own output).
     */
    layouts: {
        uniforms?: TgpuBindGroupLayout
        samplers: TgpuBindGroupLayout
    }
    /** Lifecycle callbacks collected during composition (frame.ts drains these). */
    onBeforeRender: ((p: unknown) => void)[]
    onAfterRender: ((p: unknown) => void)[]
    onResize: ((p: unknown) => void)[]
    onCleanup: (() => void)[]
    /** The set of node ids included in this composition (opacity-transition tracking). */
    composedNodeIds: Set<string>
    /** True when a builder lowered user-authored WGSL into this composition (see `GpuFragmentParams.noteCustomWgsl`). */
    usesCustomWgsl: boolean
}

export interface ComposeOptions {
    /** Tone mapping curve applied at the final-pass tail (part of the structural hash). */
    toneMapping?: ToneMappingMode
    /**
     * Flip the fragment UV's Y once, globally. Default TRUE: typegpu's `fullScreenTriangle` uv
     * is top-left origin (uv.y=0 at top) while shader math is written against a bottom-left
     * origin. Flipping once here makes `ctx.uv` bottom-left, so shaders need zero Y changes, and
     * RTT write/sample round-trips stay self-consistent (both sides use the same flipped
     * coordinate).
     */
    flipY?: boolean
    /**
     * Premultiply RGB by alpha at the final tail. Default OFF; shaders emit straight alpha and
     * the canvas is `premultiplied`, so the renderer enables it.
     */
    premultiplyAlpha?: boolean
    /** Canvas element / dimensions / gpu — threaded to fragment builders' params. */
    canvas?: HTMLCanvasElement
    dimensions?: {width: number; height: number}
    gpu?: {device: GPUDevice; root: import('typegpu').TgpuRoot}
    /**
     * Write a node's `extraFields` value on its live uniform handle. The composer has no
     * store-write access (only `gpuAccessor`), so the renderer injects this; `params.setExtraField`
     * routes through it. `null`/absent → `setExtraField` is a harmless no-op (tests that never
     * render). Not part of the structural hash (a function identity).
     */
    writeExtraField?: (nodeId: string, name: string, value: number | number[]) => void
    /**
     * Create a shader-owned media texture. The renderer injects its TextureManager's
     * `createMediaTexture` (the composer has no texture-manager access); `params.createMediaTexture`
     * routes here. Absent → `params.createMediaTexture` throws (no media shader can compose without a
     * device — tests that never render also never call it).
     */
    createMediaTexture?: (options: import('./contract').GpuMediaTextureOptions) => GpuMediaTexture
    /**
     * Create a shader-owned DATA texture (r16float/r32float) via the renderer's TextureManager
     * — the raw-float upload sibling of `createMediaTexture` (SVG-SDF field textures). Absent →
     * `params.createDataTexture` throws (no SVG-SDF shader can compose without a device).
     */
    createDataTexture?: (options: import('./contract').GpuDataTextureOptions) => GpuMediaTexture
    /**
     * Resolve the LIVE remap window (post dynamic-bound resolution) of a node's map-driven
     * prop — the same five values the renderer writes into the `_map_<prop>_*` synthetic uniforms
     * each frame. The composer has no access to the renderer's resolved `mapValues`, so the renderer
     * injects this; `params.getMapInfo(prop).window()` routes through it. Absent → `window()` falls
     * back to the raw driver-config numbers (correct for static maps). Not part of the structural
     * hash (a function identity).
     */
    resolveMapWindow?: (nodeId: string, prop: string) => import('./contract').GpuMapWindow | undefined
    /**
     * Resolve the LIVE driver-resolved CPU value of a mouse-position / mouse / auto-animate driven
     * prop — the same value the fragment reads from the `_smoothed_<prop>` synthetic field. A COMPUTE
     * hook reads its driven props through `params.getCpuValue(prop)`; a maps-routed prop's static
     * handle never moves, so without this the compute sim (e.g. Smoke's emitter) is frozen at the
     * default while the fragment path animates. The renderer injects this from its per-frame CPU
     * driver state; absent, or the prop has no such driver → undefined and `getCpuValue` falls back to
     * the static handle (correct for GPU-free tests / static maps). Also resolves the synthetic
     * `_animTime` / `_animTime_<key>` fields (the node's animated-time accumulator this frame) so
     * per-frame CPU work can share the fragment's clock. Not part of the structural hash.
     */
    resolveDriverCpuValue?: (nodeId: string, prop: string) => unknown
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Fragment serialiser — KitExpr tree → WGSL body + externals (a fresh EmitContext per pass)
// ═══════════════════════════════════════════════════════════════════════════════════════

function serialiseFragment(
    root: Expr,
    tail: (composed: string, ctx: EmitContext) => string,
    flipY: boolean,
    uniformLayout?: TgpuBindGroupLayout,
): {body: string; externals: Record<string, unknown>} {
    const externals: Record<string, unknown> = {}
    const byIdentity = new Map<unknown, string>()
    const usedNames = new Set<string>()
    const statements: string[] = []
    const memos = new Map<string, string>()
    let localCounter = 0

    const ctx: EmitContext = {
        external(value, hint) {
            const existing = byIdentity.get(value)
            if (existing) return existing
            let name = sanitizeName(hint)
            if (usedNames.has(name)) {
                let i = 1
                while (usedNames.has(`${name}_${i}`)) i++
                name = `${name}_${i}`
            }
            usedNames.add(name)
            byIdentity.set(value, name)
            externals[name] = value
            return name
        },
        statement(wgsl) {
            statements.push(wgsl)
        },
        freshLocal(hint) {
            return `${sanitizeName(hint)}_${localCounter++}`
        },
        memo(key, factory) {
            const hit = memos.get(key)
            if (hit !== undefined) return hit
            const v = factory()
            memos.set(key, v)
            return v
        },
    }

    // The uv varying local — declared once, referenced by ctx.uv (Y-flip applied here).
    const uvInit = flipY ? 'vec2f(in.uv.x, 1.0 - in.uv.y)' : 'in.uv'
    statements.push(`var uv = ${uvInit};`)

    const composedText = root._emit(ctx)
    const returnExpr = tail(composedText, ctx)

    // Group-0 anchor. The composition PINS uniforms to bind group 0, textures to 1, samplers to 2.
    // A pass that samples a texture/sampler but reads NO uniform (e.g. Blur's final pass composites
    // two textures with no uniform math) leaves group 0 empty. TypeGPU then fills group 0 with an
    // empty placeholder layout that the composition's uniform bind group can't satisfy → a draw-time
    // "missing bind groups" error. Anchoring group 0 with a discarded read of the always-present
    // `_sys.time` uniform keeps the uniform layout at group 0 (matching what passManager binds).
    const scan = `${statements.join('\n')}\n${composedText}\n${returnExpr}`
    if (uniformLayout && /\b(?:tex|samp|ext)\.\$\./.test(scan) && !/\buni\.\$\./.test(scan)) {
        ctx.external(uniformLayout, 'uni')
        statements.unshift('_ = uni.$.uniforms._sys.time;')
    }

    const body = `{\n  ${statements.join('\n  ')}\n  return ${returnExpr};\n}`
    return {body, externals}
}

/** Mutable holder the Expr emit closures read at serialise time (layouts built post-walk). */
interface LayoutRefs {
    uniform?: TgpuBindGroupLayout
    sampler: TgpuBindGroupLayout
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// The composer
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Compose a registry into a CompositionIR.
 */
export function composeNodeTree(registry: RegistryView, options: ComposeOptions = {}): CompositionIR {
    const layouts: LayoutRefs = {sampler: SHARED_SAMPLER_LAYOUT}
    const rttBoundaries: {key: string; root: Expr}[] = []
    const textures: TextureBinding[] = []
    const externalTextures: ExternalTextureBinding[] = []
    const computeSteps: ComputeSpec[] = []
    const composedNodeIds = new Set<string>()
    const onBeforeRender: ((p: unknown) => void)[] = []
    const onAfterRender: ((p: unknown) => void)[] = []
    const onResize: ((p: unknown) => void)[] = []
    const onCleanup: (() => void)[] = []
    let rttCounter = 0
    let computeTexCounter = 0 // keys compute-written textures (kind:'compute')
    let externalTexCounter = 0 // keys external (video/webcam) textures
    let mediaTexCounter = 0 // keys shader-owned media textures (kind:'media')
    // Keys whose texture format is float32 (r/rg/rgba32float — 'unfilterable-float' sample type).
    // Their group-1 layout entries MUST be declared `sampleType: 'unfilterable-float'`: the TypeGPU
    // default is 'float' (filterable), and binding a float32 view against a 'float' entry is only
    // valid on devices with the OPTIONAL `float32-filterable` feature — present on desktop/Apple,
    // absent on most Android GPUs, where createBindGroup then fails and the shader renders nothing.
    // Safe because float32 compute textures (the volumetric SDF field) are only ever read with
    // `textureLoad` via `.accessor()`, never `.sample()` with a filtering sampler.
    const unfilterableKeys = new Set<string>()

    // ── uniform / _sys accessors ──────────────────────────────────────────────────────
    const uniformLayout = registry.store.layout
    if (uniformLayout) uniformLayout.$idx(BIND_GROUPS.uniforms)
    layouts.uniform = uniformLayout

    // Emit a fully-formed uniform accessor string, registering the `uni` layout external.
    const uniAccessorRaw = (fullPath: string): Expr =>
        new Expr((ctx) => {
            if (uniformLayout) ctx.external(uniformLayout, 'uni')
            return fullPath
        })
    // Synthetic-field accessor (`_opacity`, `_bbox_*`, `_xform_*`, `_map_*`, `_smoothed_*`) —
    // hand-built by the composer (these live under the same `n_<id>` struct as the props). Prop
    // accessors use `store.gpuAccessor`.
    const uniAccessor = (accessorPath: string): Expr => uniAccessorRaw(`uni.$.uniforms.${accessorPath}`)
    // Prop accessor via the store (never hand-built — honors the gpuAccessor contract).
    const propAccessor = (handle: {accessorPath: string}): Expr =>
        uniAccessorRaw(registry.store.gpuAccessor(handle, 'uni'))

    const sysCtx: KitCtx = {
        uv: new Expr(() => 'uv'),
        time: uniAccessor('_sys.time'),
        viewportSize: uniAccessor('_sys.viewportSize'),
        logicalViewportSize: uniAccessor('_sys.logicalViewportSize'),
        aspect: uniAccessor('_sys.aspect'),
        pointer: uniAccessor('_sys.pointer'),
    }

    // ── convertToTexture — registers an RTT boundary, returns a KitTexture ──────────────
    // sample/accessor emit RAW `tex.$.<key>` / `samp.$.<sampler>` text; the per-pass texture
    // + sampler layouts are built AFTER emit (from the keys the body actually uses), so each
    // pass binds ONLY what it reads — never its own RTT output (avoids the read/write hazard).
    const makeKitTexture = (key: string): KitTexture => ({
        key,
        sample: (uv: Expr, sampler: SharedSamplerName = 'linearClamp') =>
            new Expr((ctx) => `textureSample(tex.$.${key}, samp.$.${sampler}, ${uv._emit(ctx)})`),
        sampleLevel: (uv: Expr, sampler: SharedSamplerName = 'linearClamp') =>
            new Expr((ctx) => `textureSampleLevel(tex.$.${key}, samp.$.${sampler}, ${uv._emit(ctx)}, 0.0)`),
        accessor: () => new Expr(() => `tex.$.${key}`),
        dimensions: () => new Expr(() => `vec2f(textureDimensions(tex.$.${key}))`),
    })

    // External (video/webcam) textures are a distinct WGSL type (`texture_external`)
    // with their own accessor prefix (`ext.$.<key>`, group 3) and MUST be sampled with
    // `textureSampleBaseClampToEdge` — plain `textureSample` is invalid on an external texture. The
    // per-pass `externalLayout` is built by `buildFragmentEntry` from the `ext.$.<key>` matches the
    // body emits, exactly as the texture layout is built from `tex.$.<key>`.
    const makeExternalKitTexture = (key: string): KitTexture => ({
        key,
        sample: (uv: Expr, sampler: SharedSamplerName = 'linearClamp') =>
            new Expr((ctx) => `textureSampleBaseClampToEdge(ext.$.${key}, samp.$.${sampler}, ${uv._emit(ctx)})`),
        // texture_external has no textureSampleLevel; its only sampling builtin takes no
        // derivatives, so the uniform-flow-safe variant IS the plain sample.
        sampleLevel: (uv: Expr, sampler: SharedSamplerName = 'linearClamp') =>
            new Expr((ctx) => `textureSampleBaseClampToEdge(ext.$.${key}, samp.$.${sampler}, ${uv._emit(ctx)})`),
        accessor: () => new Expr(() => `ext.$.${key}`),
        dimensions: () => new Expr(() => `vec2f(textureDimensions(ext.$.${key}))`),
    })

    const convertToTexture = (e: Expr): KitTexture => {
        const key = `rtt_${rttCounter++}`
        rttBoundaries.push({key, root: e})
        textures.push({key, kind: 'rtt'})
        return makeKitTexture(key)
    }

    // d3kit: the map SOURCE RTT for a (node, prop), memoised so the fragment override AND a compute
    // hook's `getMapInfo` share ONE boundary (one RTT pass rendering the source) — the compute needs
    // the source rendered anyway, so the fragment's (often dead-for-Blur) override reuses the same
    // texture rather than registering a second identical pass. Returns null when the source can't
    // resolve (no custom id / unknown source), so map-driven props without a live source degrade
    // gracefully (the fragment keeps the base value; a compute consumer falls back to its static path).
    const mapSourceCache = new Map<string, KitTexture | null>()
    const mapSourceTexture = (node: RegistryNode, prop: string, source: string | undefined): KitTexture | null => {
        const cacheKey = `${node.id}:${prop}`
        const cached = mapSourceCache.get(cacheKey)
        if (cached !== undefined) return cached
        const sourceId = source ? registry.resolveCustomId(source) : null
        const tex = sourceId ? convertToTexture(composeNode(sourceId, new Set([node.id]))) : null
        mapSourceCache.set(cacheKey, tex)
        return tex
    }

    // ── per-node accessors + driver overrides ─────────────────────────────────────────
    // Driver overrides are resolved EAGERLY (during the walk) so the RTT boundaries their
    // map sources need are registered before the passes are built; only the local-var
    // EMISSION is deferred to the fragment where the node is used (memoised per ctx).
    const overridesByNode = new Map<string, Record<string, Expr>>()
    const getOverrides = (node: RegistryNode): Record<string, Expr> => {
        const cached = overridesByNode.get(node.id)
        if (cached) return cached
        const out: Record<string, Expr> = {}
        const maps = node.metadata.maps
        if (maps) {
            // Custom per-prop sample UVs (DotGrid/Grid mapSampleUVs). Ask the shader for the UV
            // each map-driven prop's SOURCE should be sampled at (cell-center for tiled patterns).
            // Computed from BASE prop accessors (never the driver-overridden local —
            // `uniformExprs`→`propsAccessor`→`getOverrides` would recurse). The `ctx` (uv/viewport)
            // mirrors what a fragment builder gets. Any prop absent from the result falls back to
            // `sysCtx.uv`.
            let customSampleUVs: Record<string, Expr> = {}
            const sampleUVsFn = node.definition.mapSampleUVs
            if (sampleUVsFn) {
                const baseUniforms: Record<string, Expr> = {}
                for (const [prop, h] of Object.entries(node.handles)) {
                    if (h.cpu) continue
                    baseUniforms[prop] = propAccessor(h)
                }
                customSampleUVs = sampleUVsFn({uniforms: baseUniforms, ctx: sysCtx}) ?? {}
            }
            for (const [prop, driver] of Object.entries(maps)) {
                if (!(prop in node.handles)) continue
                const overExpr = resolveDriver(driver, node, prop, customSampleUVs[prop])
                if (overExpr) out[prop] = overExpr
            }
        }
        overridesByNode.set(node.id, out)
        return out
    }

    const propsAccessor = (node: RegistryNode): Expr => {
        const key = nodeKey(node.id)
        const maps = node.metadata.maps
        if (!maps || Object.keys(maps).length === 0) {
            return uniAccessor(key)
        }
        const overrides = getOverrides(node) // eager: registers map-source RTT boundaries now
        if (Object.keys(overrides).length === 0) return uniAccessor(key)
        // Emit a local copy + per-prop overwrites once per fragment.
        return new Expr((ctx) =>
            ctx.memo(`props:${node.id}`, () => {
                if (uniformLayout) ctx.external(uniformLayout, 'uni')
                const local = ctx.freshLocal(`p_${key}`)
                ctx.statement(`var ${local} = uni.$.uniforms.${key};`)
                for (const [prop, overExpr] of Object.entries(overrides)) {
                    ctx.statement(`${local}.${prop} = ${overExpr._emit(ctx)};`)
                }
                return local
            }),
        )
    }

    // Per-prop CPU values (post-transform) for compile-time branching in fragment/uvRemap
    // builders. Sourced from the node's handle mirrors — the same values the structural hash keys
    // compileTime props on, so a change recomposes and the builder re-runs. Twirl's `edges` (and
    // every other compileTime-branched distortion) needs this.
    const propValuesFor = (node: RegistryNode): Record<string, unknown> => {
        const out: Record<string, unknown> = {}
        for (const [prop, h] of Object.entries(node.handles)) out[prop] = h.value
        return out
    }

    const uniformExprs = (node: RegistryNode): Record<string, Expr> => {
        const out: Record<string, Expr> = {}
        const key = nodeKey(node.id)
        const propsE = propsAccessor(node)
        void key
        for (const [prop, h] of Object.entries(node.handles)) {
            if (h.cpu) continue
            const maps = node.metadata.maps
            if (maps && prop in maps) {
                // Field of the driver-overridden local.
                out[prop] = propsE.member(prop)
            } else {
                out[prop] = propAccessor(h)
            }
        }
        // extraFields live under the SAME `n_<id>` struct (registered by the renderer's
        // buildFieldInits, like `_animTime`), so they aren't in `node.handles` (props only). Expose
        // each as a GPU accessor here so the builder reads it via `uniforms.<name>`. `propsE.member`
        // reads the driver-override local when maps are active (the whole struct is copied, so the
        // extra field carries through) — consistent with the `_animTime` `props.member(...)` read.
        const extraFields = node.definition.extraFields
        if (extraFields) {
            for (const name of Object.keys(extraFields)) out[name] = propsE.member(name)
        }
        return out
    }

    // Resolve a prop driver to an override Expr (map = sample+remap; mouse/auto = synthetic field).
    // `sampleUV` is the shader's custom per-prop sample UV from mapSampleUVs, else undefined.
    const resolveDriver = (driver: unknown, node: RegistryNode, prop: string, sampleUV?: Expr): Expr | null => {
        const cfg = driver as {type?: string; source?: string; channel?: string}
        const key = nodeKey(node.id)
        if (cfg.type === 'map') {
            // Shared with the compute hook's getMapInfo (one source RTT per node+prop).
            const srcTex = mapSourceTexture(node, prop, cfg.source)
            if (!srcTex) return null
            // Sample the source at the shader's custom UV (cell-center) when provided, else at
            // the fragment UV.
            const sampled = srcTex.sample(sampleUV ?? sysCtx.uv)
            const raw = extractChannel(sampled, cfg.channel ?? 'luminance')
            // remap uniforms are synthetic per-prop fields:
            // uni.$.uniforms.n_x._map_<prop>_{inputMin,inputMax,outputMin,outputMax,curve}
            const iMin = uniAccessor(`${key}._map_${prop}_inputMin`)
            const iMax = uniAccessor(`${key}._map_${prop}_inputMax`)
            const oMin = uniAccessor(`${key}._map_${prop}_outputMin`)
            const oMax = uniAccessor(`${key}._map_${prop}_outputMax`)
            const curve = uniAccessor(`${key}._map_${prop}_curve`)
            return remapExpr(raw, iMin, iMax, oMin, oMax, curve)
        }
        if (cfg.type === 'mouse-position' || cfg.type === 'mouse' || cfg.type === 'auto-animate') {
            // Smoothed value is a synthetic per-prop field updated CPU-side each frame.
            return uniAccessor(`${key}._smoothed_${prop}`)
        }
        return null
    }

    // ── mask (applyMaskIfNeeded) ────────────────────────────────────────────────────────
    const applyMaskIfNeeded = (
        result: Expr,
        node: RegistryNode,
        processedMasks: Set<string>,
        distortedUV?: Expr,
    ): Expr => {
        const maskCfg = node.metadata.mask
        if (!maskCfg?.source) return result
        const maskNodeId = registry.resolveCustomId(maskCfg.source)
        if (!maskNodeId) return result
        if (processedMasks.has(maskNodeId)) {
            console.error("[gpu] circular dependency in shader masks.")
            return result
        }
        const next = new Set(processedMasks)
        next.add(node.id)
        let maskExpr = composeNode(maskNodeId, next)
        if (distortedUV) {
            const maskTex = convertToTexture(maskExpr)
            maskExpr = maskTex.sample(distortedUV)
        }
        return applyMaskFn(result, maskExpr, (maskCfg.type ?? 'alpha') as MaskType)
    }

    // ── bbox / transform helpers ────────────────────────────────────────────────────────
    const opacityExpr = (node: RegistryNode): Expr => uniAccessor(`${nodeKey(node.id)}._opacity`)

    // An identity full-frame box renders exactly like no box, so it must not activate the box
    // path — otherwise a `boundingBox={{origin:'center'}}` on a Group costs a whole RTT + resample
    // for nothing. MUST stay in lockstep with gpu/index.ts `bboxActive` (which allocates the
    // `_bbox_*` uniforms every emitter here reads) and with the structural hash below.
    const hasBox = (node: RegistryNode): boolean =>
        !!node.metadata.boundingBox &&
        !!node.definition.boundingBoxDeclaration &&
        !isIdentityBoundingBox(node.metadata.boundingBox)
    const cutoutActive = (node: RegistryNode): boolean =>
        (node.handles.cutout?.value as number) === 1
    const requiresChild = (node: RegistryNode): boolean => !!node.definition.requiresChild
    const blendWithChildren = (node: RegistryNode): boolean => !!node.definition.blendWithChildren

    // usesBoxClipWindow — the box is a pure clip WINDOW, not a
    // switch-to-composite-over-child trigger.
    const usesBoxClipWindow = (node: RegistryNode): boolean =>
        hasBox(node) &&
        !node.definition.boundingBoxDeclaration?.boxResamplesContent &&
        requiresChild(node) &&
        node.metadata.blendMode === 'normal' &&
        !cutoutActive(node) &&
        !blendWithChildren(node)

    // Half a device pixel, expressed in the clip mask's square space (x aspect-corrected, y in
    // canvas-height UV) — so it is isotropic and just `0.5 / viewportHeightPx`. Feeds the clip
    // mask's coverage ramp so a box edge resolves to fractional alpha over exactly one pixel
    // instead of a binary step (which stair-steps hard once a distortion magnifies it).
    const halfPixelSq = (): Expr =>
        new Expr((ctx) => `(0.5 / max(${sysCtx.viewportSize.member('y')._emit(ctx)}, 1.0))`)

    // The rectangular clip mask node (1 inside box, 0 outside, antialiased across the edge).
    // Synthetic bbox uniforms live under the node struct (`_bbox_*`). Uses the kit clip-mask fn.
    // `uv` defaults to the screen UV; a caller inside an analytic distortion fold passes the
    // FOLDED coordinate so the clip travels with the distorted content.
    const clipMaskExpr = (node: RegistryNode, uv: Expr = sysCtx.uv): Expr => {
        const key = nodeKey(node.id)
        const bu = (f: string) => uniAccessor(`${key}._bbox_${f}`)
        return call(applyRectangularClipMaskFn, 'bboxClipMask', [
            uv,
            bu('centerX'),
            bu('centerY'),
            bu('halfWidth'),
            bu('halfHeight'),
            bu('cornerRadius'),
            bu('rotation'),
            bu('aspectRatio'),
            halfPixelSq(),
        ])
    }

    // applyBoundingBoxTransformation — clip path (pure alpha mask) + resample path (Group).
    const applyBoxTransform = (result: Expr, node: RegistryNode): Expr => {
        if (!hasBox(node)) return result
        // boxResamplesContent (Group) — treat the box as an image-like viewport over the
        // node's composited result. RTT the composite, sample it through the box→content UV map
        // (the same mapping resize-fit generators use), then clip to the box rect. convertToTexture
        // yields PREMULTIPLIED data → unpremultiply the sample before the straight-alpha clip.
        if (node.definition.boundingBoxDeclaration?.boxResamplesContent) {
            const key = nodeKey(node.id)
            const bu = (f: string) => uniAccessor(`${key}._bbox_${f}`)
            const tex = convertToTexture(result)
            const contentUV = call(boundingBoxToGeneratorUVContextFn, 'bboxGenUVContext', [
                sysCtx.uv, bu('centerX'), bu('centerY'), bu('halfWidth'), bu('halfHeight'), bu('rotation'), bu('aspectRatio'),
            ])
            const sampled = call(unpremultiplyAlphaFn, 'unpremultiplyAlpha', [tex.sample(contentUV)])
            return vec4(sampled.member('rgb'), sampled.member('a').mul(clipMaskExpr(node)))
        }
        const mask = clipMaskExpr(node)
        return vec4(result.member('rgb'), result.member('a').mul(mask))
    }

    // applyNodeTransform — RTT + UV transform + edge handling. Active
    // only when transform values differ from defaults (or the RTT-active flag latched).
    const applyNodeTransform = (result: Expr, node: RegistryNode): Expr => {
        const tf = node.metadata.transform
        if (!tf || !needsTransformation(tf)) return result
        const key = nodeKey(node.id)
        const tex = convertToTexture(result)
        const tu = (f: string) => uniAccessor(`${key}._xform_${f}`)
        const transformedUV = call(applyUVTransformFn, 'applyUVTransform', [
            sysCtx.uv,
            tu('offsetX'),
            tu('offsetY'),
            tu('rotation'),
            tu('scale'),
            tu('anchorX'),
            tu('anchorY'),
            tu('aspectRatio'),
        ])
        // Edge handling for the RTT transform path — a compile-time branch on the layer
        // transform's `edges` mode (a change recomposes; gpu/index.ts flags it):
        // stretch(0) samples straight (the linearClamp sampler stretches the edge pixel);
        // transparent(1) alpha-cuts outside [0,1]; mirror(2)/wrap(3) re-sample the RTT texture at
        // the reflected/tiled UV so a scaled-down layer tiles the plane instead of stretching.
        return applyEdgeHandlingExpr(transformedUV, (uv) => tex.sample(uv), TRANSFORM_EDGE_MODES[tf.edges] ?? 0)
    }

    // applyNodeBboxOrTransform — nested-children path tail.
    const applyNodeBboxOrTransform = (result: Expr, composedInput: Expr | undefined, node: RegistryNode): Expr => {
        if (hasBox(node)) {
            if (composedInput !== undefined && usesBoxClipWindow(node)) {
                return mixExpr(composedInput, result, clipMaskExpr(node))
            }
            return applyBoxTransform(result, node)
        }
        return applyNodeTransform(result, node)
    }

    // applyChildFragmentBox — leaf-filter path.
    const applyChildFragmentBox = (
        fragmentResult: Expr,
        node: RegistryNode,
    ): {transformedFragment: Expr; clipMaskNode: Expr | undefined} => {
        if (usesBoxClipWindow(node)) {
            return {transformedFragment: fragmentResult, clipMaskNode: clipMaskExpr(node)}
        }
        const transformedFragment = hasBox(node)
            ? applyBoxTransform(fragmentResult, node)
            : applyNodeTransform(fragmentResult, node)
        return {transformedFragment, clipMaskNode: undefined}
    }

    // composeReplaceFilter.
    const composeReplaceFilter = (
        accumulated: Expr | undefined,
        childResult: Expr,
        node: RegistryNode,
        clipMaskNode: Expr | undefined,
    ): Expr => {
        const opacity = opacityExpr(node)
        const noBox = blendWithChildren(node)
            ? accumulated !== undefined
                ? applyBlend(accumulated, childResult, 'normal', opacity)
                : mixExpr(ZERO, childResult, opacity)
            : mixExpr(accumulated ?? ZERO, childResult, opacity)
        if (clipMaskNode === undefined) return noBox
        return mixExpr(accumulated ?? ZERO, noBox, clipMaskNode)
    }

    // ── compute init (initNodeCompute) ──────────────────────────────────────────────────
    const initNodeCompute = (node: RegistryNode, params: GpuFragmentParams): Record<string, KitTexture | GpuStorageBuffer> | undefined => {
        if (!node.definition.compute) return undefined
        try {
            const result = node.definition.compute(params)
            if (result) {
                // Carry bindInputs so passManager can (re)bind the pass's RTT inputs.
                computeSteps.push({nodeId: node.id, getComputeNodes: result.getComputeNodes, bindInputs: result.bindInputs})
                return result.outputs
            }
        } catch (e) {
            debugError(`[gpu] error initializing compute for node ${node.id}:`, e)
        }
        return undefined
    }

    // ── fragment params builder ─────────────────────────────────────────────────────────
    // childBoundsParams / ownBoundsParams wiring. Only `wantsBoundsParams` shaders (Repeater)
    // consume these; {} for everything else, so it spreads harmlessly into any params object.
    // ownBoundsParams = this node's own active-bbox `_bbox_*` fields. childBoundsParams = the
    // FIRST visible direct child (render order) whose bounds resolve: an active-bbox child
    // forwards its live `_bbox_*` handles (auto-updates as the box is dragged); a shape child
    // (computeBounds / propBindings) uses this node's own CPU-driven `_childBounds_*` fields
    // (gpu/index.ts updateChildBounds fills them each frame). Rotation is 0 for the shape path
    // (the axis-aligned screen rect). No resolvable child → no childBoundsParams (full-canvas src).
    const boundsFromBboxFields = (n: RegistryNode): GpuBoundsParams => {
        const k = nodeKey(n.id)
        const bu = (f: string) => uniAccessor(`${k}._bbox_${f}`)
        return {centerX: bu('centerX'), centerY: bu('centerY'), halfWidth: bu('halfWidth'), halfHeight: bu('halfHeight'), rotation: bu('rotation')}
    }
    const childHasShapeBounds = (n: RegistryNode): boolean => {
        const dcl = n.definition.boundingBoxDeclaration
        return !!dcl && (!!dcl.computeBounds || !!dcl.propBindings)
    }
    const nodeBoundsParams = (node: RegistryNode): Partial<GpuFragmentParams> => {
        if (!node.definition.wantsBoundsParams) return {}
        const out: Partial<GpuFragmentParams> = {}
        if (hasBox(node)) out.ownBoundsParams = boundsFromBboxFields(node)
        const children = [...registry.getChildren(node.id)].sort(
            (a, b) => a.metadata.renderOrder - b.metadata.renderOrder,
        )
        for (const child of children) {
            if (child.metadata.visible === false) continue
            if (hasBox(child)) {
                out.childBoundsParams = boundsFromBboxFields(child)
                break
            }
            if (childHasShapeBounds(child)) {
                const k = nodeKey(node.id)
                const cb = (f: string) => uniAccessor(`${k}._childBounds_${f}`)
                out.childBoundsParams = {
                    centerX: cb('centerX'),
                    centerY: cb('centerY'),
                    halfWidth: cb('halfWidth'),
                    halfHeight: cb('halfHeight'),
                    rotation: floatE(0),
                }
                break
            }
        }
        return out
    }

    let usesCustomWgsl = false
    const makeParams = (node: RegistryNode, childNode: Expr | undefined, extra: Partial<GpuFragmentParams> = {}): GpuFragmentParams => {
        const params: GpuFragmentParams = {
            noteCustomWgsl: () => {
                usesCustomWgsl = true
            },
            props: propsAccessor(node),
            uniforms: uniformExprs(node),
            propValues: propValuesFor(node), // compile-time prop reads
            childNode,
            ctx: sysCtx,
            logicalViewportSize: sysCtx.logicalViewportSize,
            convertToTexture,
            declareTexture: (tex: KitTexture) => {
                if (!textures.some((t) => t.key === tex.key)) {
                    textures.push({key: tex.key, kind: 'media', resource: (tex as {_resource?: unknown})._resource})
                }
            },
            // Register an external (video/webcam) texture the shader samples. Records a
            // per-frame ExternalTextureBinding {key, getSource} on the IR; the pass manager re-imports
            // the source + rebuilds the group-3 bind group every frame (skipping the pass when
            // getSource returns null). Returns the KitTexture for the shader's sample/dimensions math.
            registerExternalTexture: (getSource: () => unknown): KitTexture => {
                const key = `video_${externalTexCounter++}`
                externalTextures.push({key, getSource})
                return makeExternalKitTexture(key)
            },
            // Register a compute-written texture (storage+sampled) as a sampleable
            // KitTexture. kind:'compute' → passManager binds `texture` via mediaResources.
            registerComputeTexture: (texture: unknown): KitTexture => {
                const key = `compute_${computeTexCounter++}`
                textures.push({key, kind: 'compute', resource: {texture}})
                // Float32 formats need an 'unfilterable-float' group-1 layout entry (see
                // `unfilterableKeys`). Format read off the TgpuTexture's props.
                const format = (texture as {props?: {format?: string}} | null)?.props?.format
                if (format === 'r32float' || format === 'rg32float' || format === 'rgba32float') {
                    unfilterableKeys.add(key)
                }
                return makeKitTexture(key)
            },
            // Create a shader-owned media texture via the renderer's TextureManager (injected).
            createMediaTexture: (opts) =>
                (options.createMediaTexture as ((o: unknown) => GpuMediaTexture) | undefined)?.(opts) ??
                (() => {
                    throw new Error('[gpu] createMediaTexture unavailable (no TextureManager injected)')
                })(),
            // Raw-float data texture (SVG-SDF fields), routed to the renderer's TextureManager.
            createDataTexture: (opts) =>
                (options.createDataTexture as ((o: unknown) => GpuMediaTexture) | undefined)?.(opts) ??
                (() => {
                    throw new Error('[gpu] createDataTexture unavailable (no TextureManager injected)')
                })(),
            // Register a media texture the shader body samples. The resource exposes a `texture`
            // GETTER over the shader's live getter, so a swap (reassigning the shader's `current`)
            // changes `resource.texture` identity → passManager rebuilds the pass's group-1 bind group.
            registerMediaTexture: (getTexture: () => unknown): KitTexture => {
                const key = `media_${mediaTexCounter++}`
                const resource = {
                    get texture() {
                        return getTexture()
                    },
                }
                textures.push({key, kind: 'media', resource})
                return makeKitTexture(key)
            },
            // Live post-transform CPU value of a prop, re-read each call (per-frame
            // compute updates). When the prop carries a mouse-position/mouse/auto-animate DRIVER its
            // static handle never moves (drivers route to metadata.maps, not the uniform), so prefer
            // the renderer's per-frame driver-resolved value — the same `_smoothed_<prop>` the fragment
            // consumes. Falls back to the live handle (no driver / GPU-free tests / static maps).
            getCpuValue: (prop: string): unknown => {
                if ((node.metadata.maps?.[prop] || prop.startsWith('_animTime')) && options.resolveDriverCpuValue) {
                    const resolved = options.resolveDriverCpuValue(node.id, prop)
                    if (resolved !== undefined) return resolved
                }
                return node.handles[prop]?.value
            },
            // Compute-map interplay. Expose the map SOURCE (as the SHARED late-bound RTT
            // KitTexture — bound in the compute node's bindInputs by resolving `.key`), the channel,
            // and a live remap-window reader for a map-driven prop. Null when the prop has no map
            // driver. The window resolves via the renderer-injected `resolveMapWindow` (post
            // dynamic-bound values, matching the fragment `_map_<prop>_*` uniforms), falling back to
            // the raw driver-config numbers when no resolver is injected (static maps / tests).
            getMapInfo: (prop: string): GpuMapInfo | null => {
                const driver = node.metadata.maps?.[prop] as
                    | {type?: string; source?: string; channel?: string; inputMin?: number; inputMax?: number; outputMin?: number; outputMax?: number; curve?: number}
                    | undefined
                if (!driver || driver.type !== 'map') return null
                const sourceTexture = mapSourceTexture(node, prop, driver.source)
                if (!sourceTexture) return null
                return {
                    sourceTexture,
                    channel: driver.channel ?? 'luminance',
                    window: () =>
                        options.resolveMapWindow?.(node.id, prop) ?? {
                            inputMin: typeof driver.inputMin === 'number' ? driver.inputMin : 0,
                            inputMax: typeof driver.inputMax === 'number' ? driver.inputMax : 1,
                            outputMin: typeof driver.outputMin === 'number' ? driver.outputMin : 0,
                            outputMax: typeof driver.outputMax === 'number' ? driver.outputMax : 1,
                            curve: typeof driver.curve === 'number' ? driver.curve : 0,
                        },
                }
            },
            // Layer-typed prop (ui type 'layer'): resolve the selected layer's custom id to its
            // shared RTT texture via the map-source cache/boundary. Empty / unresolvable id → null
            // (the shader falls back to its no-source path). The prop is compileTime, so the id is
            // in the structural hash and a change recomposes (re-running this resolve).
            getLayerTexture: (prop: string): KitTexture | null => {
                const source = node.handles[prop]?.value
                if (typeof source !== 'string' || source === '') return null
                return mapSourceTexture(node, prop, source)
            },
            // Per-frame writer for extraFields (Blob light-dir). Routes to the renderer's live
            // handle via the injected callback; the builder reads the same field via uniforms.<name>.
            setExtraField: (name: string, value: number | number[]): void =>
                options.writeExtraField?.(node.id, name, value),
            onCleanup: (cb) => onCleanup.push(cb),
            onBeforeRender: (cb) => onBeforeRender.push(cb as (p: unknown) => void),
            onAfterRender: (cb) => onAfterRender.push(cb as (p: unknown) => void),
            onResize: (cb) => onResize.push(cb as (p: unknown) => void),
            canvas: options.canvas as HTMLCanvasElement,
            dimensions: options.dimensions ?? {width: 1, height: 1},
            gpu: options.gpu as {device: GPUDevice; root: import('typegpu').TgpuRoot},
            domCanvas: node.domCanvas, // per-node DOM-capture host (HTMLInCanvas)
            ...nodeBoundsParams(node), // childBoundsParams / ownBoundsParams (Repeater)
            ...extra,
        }
        return params
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // composeNodeTree recursion — root + non-root sibling loops
    // ══════════════════════════════════════════════════════════════════════════════════
    function composeNode(id: string, processedMasks: Set<string> = new Set()): Expr {
        const node = registry.getNode(id)
        if (!node) return ZERO
        const children = registry.getChildren(id)

        // ── UV-context fast path (providesUVContextViaCompute chain) ──
        if (children.length > 0 && node.definition.providesUVContextViaCompute) {
            const uvResult = tryComposeViaUVContext(node, children, processedMasks)
            if (uvResult !== null) return uvResult
        }
        // ── Analytic push-down (N pointwise children) then the nested single-child chain ──
        if (children.length > 0 && node.definition.uvRemap) {
            const pushed = tryComposeViaAnalyticPushdown(node, children)
            if (pushed !== null) return pushed
            const nested = tryComposeViaAnalyticUVRemapNested(node, children, processedMasks)
            if (nested !== null) return nested
        }

        if (children.length === 0) {
            composedNodeIds.add(id)
            const params = makeParams(node, undefined, {
                bboxMask: bboxMaskForNode(node),
                ...genResizeFitParams(node),
            })
            params.computeOutputs = initNodeCompute(node, params)
            const result = node.definition.fragment(params)
            const transformed = hasBox(node) ? applyBoxTransform(result, node) : applyNodeTransform(result, node)
            return applyMaskIfNeeded(transformed, node, processedMasks)
        }

        const sorted = [...children].sort((a, b) => a.metadata.renderOrder - b.metadata.renderOrder)

        // ── Analytic inline-generator fast path ──
        const analytic = tryComposeViaAnalyticUVRemap(sorted, processedMasks)
        if (analytic !== null) {
            const params = makeParams(node, analytic, {bboxMask: bboxMaskForNode(node)})
            params.computeOutputs = initNodeCompute(node, params)
            const result = node.definition.fragment(params)
            const transformed = hasBox(node) ? applyBoxTransform(result, node) : applyNodeTransform(result, node)
            return applyMaskIfNeeded(transformed, node, processedMasks)
        }

        // Sibling loop (root & non-root are identical here).
        const composed = composeSiblings(node, sorted, processedMasks)

        composedNodeIds.add(id)
        const params = makeParams(node, composed, {bboxMask: bboxMaskForNode(node)})
        params.computeOutputs = initNodeCompute(node, params)
        const result = node.definition.fragment(params)
        const transformed = applyNodeBboxOrTransform(result, composed, node)
        return applyMaskIfNeeded(transformed, node, processedMasks)
    }

    // The incremental sibling accumulation (root/non-root loops, unified).
    function composeSiblings(parent: RegistryNode, sorted: RegistryNode[], processedMasks: Set<string>): Expr | undefined {
        let composed: Expr | undefined = undefined
        for (let i = 0; i < sorted.length; i++) {
            const child = sorted[i]
            if (child.metadata.visible === false) continue

            // Analytic composite terminus: a run of uvRemap distortions over composed content.
            if (composed !== undefined && analyticRunEligible(child)) {
                const runResult = tryFoldAnalyticRunComposite(sorted, i, composed, processedMasks)
                if (runResult) {
                    composed = runResult.node
                    i = runResult.endIndex
                    continue
                }
            }

            composedNodeIds.add(child.id)
            const childHasChildren = registry.getChildren(child.id).length > 0
            let childResult: Expr
            let clipMaskNode: Expr | undefined

            if (childHasChildren) {
                childResult = composeNode(child.id, processedMasks)
            } else {
                // RTT boundary for a filter that consumes the accumulated siblings.
                let effectiveChild = composed
                if (child.definition.requiresRTT && composed !== undefined) {
                    effectiveChild = applyBlend(ZERO, composed, 'normal', floatE(1))
                }
                // An acceptsOptionalChild GENERATOR consumes ONLY an explicitly nested child (the
                // childHasChildren branch above) — never the preceding-sibling fallback. As a leaf in
                // the stack it must render standalone, so it gets no child here (it still composites
                // over the accumulated siblings normally below).
                if (child.definition.acceptsOptionalChild) {
                    effectiveChild = undefined
                }
                const params = makeParams(child, effectiveChild, {
                    bboxMask: bboxMaskForNode(child),
                    ...genResizeFitParams(child),
                })
                params.computeOutputs = initNodeCompute(child, params)
                const fragmentResult = child.definition.fragment(params)
                const childBox = applyChildFragmentBox(fragmentResult, child)
                clipMaskNode = childBox.clipMaskNode
                childResult = applyMaskIfNeeded(childBox.transformedFragment, child, processedMasks)
            }

            const shouldReplace =
                child.definition.requiresChild && child.metadata.blendMode === 'normal' && !childHasChildren

            if (shouldReplace) {
                composed = composeReplaceFilter(composed, childResult, child, clipMaskNode)
            } else {
                const blendMode = (child.metadata.blendMode ?? 'normal') as BlendMode
                const effectiveOpacity = child.metadata.opacity ?? 1.0
                if (composed === undefined) {
                    const needsRTT = parent.definition.requiresRTT || effectiveOpacity !== 1.0
                    composed = needsRTT ? applyBlend(WHITE0, childResult, blendMode, opacityExpr(child)) : childResult
                } else {
                    composed = applyBlend(composed, childResult, blendMode, opacityExpr(child))
                }
            }
        }
        return composed
    }

    // ── genResizeFitParams — generator resize-fit ──
    // `uv` defaults to the screen UV; the analytic push-down passes the FOLDED coordinate so the
    // box→content mapping composes on top of the distortion instead of fighting it.
    function genResizeFitParams(node: RegistryNode, uv: Expr = sysCtx.uv): Partial<GpuFragmentParams> {
        if (!hasBox(node)) return {}
        if (!node.definition.boundingBoxDeclaration?.supportsResizeFit) return {}
        const key = nodeKey(node.id)
        const bu = (f: string) => uniAccessor(`${key}._bbox_${f}`)
        const uvContext = call(boundingBoxToGeneratorUVContextFn, 'bboxGenUVContext', [
            uv, bu('centerX'), bu('centerY'), bu('halfWidth'), bu('halfHeight'), bu('rotation'), bu('aspectRatio'),
        ])
        const effViewport = vec4(
            bu('halfWidth').mul(2).mul(sysCtx.viewportSize.member('x')),
            bu('halfHeight').mul(2).mul(sysCtx.viewportSize.member('y')),
            0,
            0,
        ).member('xy')
        return {uvContext, effectiveViewportSize: effViewport}
    }

    // ── bboxMaskForNode ──
    function bboxMaskForNode(node: RegistryNode): Expr | undefined {
        if (!hasBox(node) || !node.definition.requiresRTT) return undefined
        if (node.definition.boundingBoxDeclaration?.supportsResizeFit) return undefined
        return clipMaskExpr(node)
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // Analytic UV-remap fast paths — eligibility + folding
    // ══════════════════════════════════════════════════════════════════════════════════
    const analyticHasFallbackTriggers = (n: RegistryNode): boolean =>
        !!n.metadata.mask ||
        (n.metadata.opacity !== undefined && n.metadata.opacity < 1) ||
        (n.metadata.blendMode ?? 'normal') !== 'normal' ||
        needsTransformation(n.metadata.transform) ||
        hasBox(n)

    const analyticRunEligible = (n: RegistryNode): boolean =>
        !!n.definition.uvRemap && registry.getChildren(n.id).length === 0 && !analyticHasFallbackTriggers(n)

    // remapAt — run ONE node's uvRemap over an incoming (uv, mask). The single place a
    // distortion's analytic contribution is evaluated; the run fold and the push-down both use it.
    function remapAt(dNode: RegistryNode, uv: Expr, mask: Expr): {uv: Expr; mask: Expr} {
        composedNodeIds.add(dNode.id)
        const params = makeParams(dNode, undefined)
        const computeOutputs = initNodeCompute(dNode, params)
        return dNode.definition.uvRemap!({
            uv,
            mask,
            uniforms: uniformExprs(dNode),
            propValues: propValuesFor(dNode), // compile-time prop reads
            props: propsAccessor(dNode),
            aspect: sysCtx.aspect,
            onBeforeRender: (cb) => onBeforeRender.push(cb as (p: unknown) => void),
            computeOutputs,
        })
    }

    // foldAnalyticRun — fold a run into one combined UV over the base UV.
    // (uvRemap distortions never sample children, so no mask composition happens here.)
    function foldAnalyticRun(run: RegistryNode[]): {uv: Expr; mask: Expr} {
        let uv: Expr = sysCtx.uv
        let mask: Expr = floatE(1)
        for (let i = run.length - 1; i >= 0; i--) {
            const remapped = remapAt(run[i], uv, mask)
            uv = remapped.uv
            mask = remapped.mask
        }
        return {uv, mask}
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // Analytic UV PUSH-DOWN — a distortion over SEVERAL pointwise children
    //
    // A UV remap commutes with pointwise compositing: `blend(cᵢ)(f(uv))` is the same image as
    // `blend(cᵢ(f(uv)))`. So instead of rasterising the composed children to an RTT and
    // re-sampling that at the remapped coordinate (two resamples, capped at the RTT's
    // resolution), each child can be evaluated AT the remapped coordinate directly — text samples
    // its own supersampled glyph atlas, an image samples its own source pixels. No intermediate
    // raster, no RTT pass, and quality no longer degrades with magnification.
    //
    // The chain path below (`tryComposeViaAnalyticUVRemapNested`) already does this for a single
    // child; this handles N children, and additionally allows per-child opacity / blend mode /
    // clip box — all of which are pointwise and so are equally safe to evaluate at the folded
    // coordinate. Anything that is NOT pointwise in screen space (a nested filter that resamples,
    // a mask source, a layer transform, a Group box that resamples its content) is ineligible and
    // falls through to the RTT path.
    // ══════════════════════════════════════════════════════════════════════════════════

    /** Fan-out/depth ceiling for one push-down, so a pathological tree can't emit a huge fold. */
    const PUSHDOWN_MAX_NODES = 24
    /** Shared "no coverage loss" mask for the `uvRemap` contract, which requires a real Expr. */
    const fullCoverage = floatE(1)

    /**
     * Can this node's whole subtree be evaluated at a substituted coordinate? PURE — it must not
     * compose anything, because a partial compose would leave RTT boundaries / compute steps /
     * media textures registered for a path we then abandon.
     *
     * Eligible: a childless generator that opted into `acceptsUVContext` (the flag IS the shader's
     * promise that it positions itself from `params.uvContext`), or a distortion whose own children
     * are all eligible (its remap folds into ours). Per-child opacity / blendMode / clip-or-resize
     * box are fine. A `requiresChild` leaf is NOT: as a leaf it consumes its preceding siblings,
     * which is not a pointwise operation.
     */
    const pushdownEligible = (n: RegistryNode, budget: {left: number}): boolean => {
        if (budget.left-- <= 0) return false
        if (n.metadata.mask) return false
        if (needsTransformation(n.metadata.transform)) return false
        const children = registry.getChildren(n.id).filter((c) => c.metadata.visible !== false)
        if (children.length === 0) {
            // A box that RESAMPLES its content (Group) is an RTT round-trip, not a pointwise op;
            // a clip / resize-fit box is fine — composeAtUV evaluates it at the folded coordinate.
            if (hasBox(n) && n.definition.boundingBoxDeclaration?.boxResamplesContent) return false
            return !!n.definition.acceptsUVContext && !n.definition.requiresChild
        }
        if (!n.definition.uvRemap) return false
        // An INTERMEDIATE distortion carrying a box would need its clip applied to the composite
        // it wraps, which composeAtUV's fold branch does not do — leave those to the RTT path
        // rather than silently dropping the box.
        if (hasBox(n)) return false
        return children.every((c) => pushdownEligible(c, budget))
    }

    /**
     * Run one distortion's remap, tracking whether it actually changed coverage. Only the
     * `transparent` edge mode does; every other mode hands the mask straight back (identical
     * object), so `undefined` — "full coverage" — survives the fold and no `* 1.0` is emitted.
     */
    function remapCoverage(dNode: RegistryNode, uv: Expr, mask: Expr | undefined): {uv: Expr; mask: Expr | undefined} {
        const incoming = mask ?? fullCoverage
        const out = remapAt(dNode, uv, incoming)
        return {uv: out.uv, mask: out.mask === incoming ? mask : out.mask}
    }

    /** `a * b`, where `undefined` means "full coverage" on either side. */
    const combineCoverage = (a: Expr | undefined, b: Expr | undefined): Expr | undefined =>
        a === undefined ? b : b === undefined ? a : a.mul(b)

    /**
     * Compose an eligible subtree at `uv`, carrying `mask` (the coverage a distortion's
     * `transparent` edge mode drops; `undefined` = fully covered). Returns null only if
     * eligibility was mis-predicted — callers must have run `pushdownEligible` first.
     */
    function composeAtUV(node: RegistryNode, uv: Expr, mask: Expr | undefined): Expr | null {
        const children = registry.getChildren(node.id).filter((c) => c.metadata.visible !== false)

        // A nested distortion: fold its remap into the coordinate and keep descending.
        if (children.length > 0) {
            if (!node.definition.uvRemap) return null
            const next = remapCoverage(node, uv, mask)
            return composeChildrenAtUV(children, asLocal(next.uv, 'foldUV'), next.mask)
        }
        if (!node.definition.acceptsUVContext) return null

        composedNodeIds.add(node.id)
        // A resize-fit box maps the (already folded) coordinate into box-local space; every other
        // node reads the folded coordinate directly.
        const resizeFit = genResizeFitParams(node, uv)
        const params = makeParams(node, undefined, Object.keys(resizeFit).length > 0 ? resizeFit : {uvContext: uv})
        params.computeOutputs = initNodeCompute(node, params)
        const fragmentResult = node.definition.fragment(params)
        // A clip box is a pointwise alpha mask — evaluated at the folded coordinate so the clip
        // travels with the content it is clipping.
        const coverage = combineCoverage(mask, hasBox(node) ? clipMaskExpr(node, uv) : undefined)
        if (coverage === undefined) return fragmentResult
        // Hoisted: splitting a result into `.rgb` and `.a` reads it TWICE, and the push-down nests
        // that split once per tree level — inline, a generator body would be re-emitted 2^depth
        // times. A local keeps every level linear.
        const result = asLocal(fragmentResult, 'gen')
        return vec4(result.member('rgb'), result.member('a').mul(coverage))
    }

    /** Blend a sibling set, each child evaluated at `uv`, then apply the run's coverage `mask`. */
    function composeChildrenAtUV(children: RegistryNode[], uv: Expr, mask: Expr | undefined): Expr | null {
        const sorted = [...children].sort((a, b) => a.metadata.renderOrder - b.metadata.renderOrder)
        let composed: Expr | undefined
        for (const child of sorted) {
            const childResult = composeAtUV(child, uv, undefined)
            if (childResult === null) return null
            const blendMode = (child.metadata.blendMode ?? 'normal') as BlendMode
            const effectiveOpacity = child.metadata.opacity ?? 1.0
            if (composed === undefined) {
                // Matches composeSiblings: a first child that is plain normal/1.0 needs no blend
                // base at all; anything else blends over the transparent WHITE0 base.
                composed =
                    blendMode === 'normal' && effectiveOpacity === 1.0
                        ? childResult
                        : applyBlend(WHITE0, childResult, blendMode, opacityExpr(child))
            } else {
                composed = applyBlend(composed, childResult, blendMode, opacityExpr(child))
            }
        }
        if (composed === undefined) return null
        if (mask === undefined) return composed
        const sibs = asLocal(composed, 'sibs') // see the hoist note in composeAtUV
        return vec4(sibs.member('rgb'), sibs.member('a').mul(mask))
    }

    // tryComposeViaAnalyticPushdown — a uvRemap parent over N pointwise children.
    function tryComposeViaAnalyticPushdown(parent: RegistryNode, children: RegistryNode[]): Expr | null {
        if (!parent.definition.uvRemap) return null
        if (analyticHasFallbackTriggers(parent)) return null
        const visible = children.filter((c) => c.metadata.visible !== false)
        if (visible.length === 0) return null
        const budget = {left: PUSHDOWN_MAX_NODES}
        if (!visible.every((c) => pushdownEligible(c, budget))) return null

        composedNodeIds.add(parent.id)
        const {uv, mask} = remapCoverage(parent, sysCtx.uv, undefined)
        // The parent carries no mask / transform / box (analyticHasFallbackTriggers ruled those
        // out), so the composite needs no tail here.
        return composeChildrenAtUV(visible, asLocal(uv, 'foldUV'), mask)
    }

    // tryComposeViaAnalyticUVRemap — inline-generator terminus.
    // (No mask threading: the run's distortions + inline generator carry no mask triggers by
    //  eligibility, so `_processedMasks` is accepted for call-site symmetry but unused.)
    function tryComposeViaAnalyticUVRemap(sorted: RegistryNode[], _processedMasks: Set<string>): Expr | null {
        const visible = sorted.filter((c) => c.metadata.visible !== false)
        if (!visible.some((c) => !!c.definition.uvRemap)) return null
        if (visible.length < 2) return null
        let runStart = visible.length
        for (let i = visible.length - 1; i >= 0; i--) {
            if (!analyticRunEligible(visible[i])) break
            runStart = i
        }
        const run = visible.slice(runStart)
        const contentBelow = visible.slice(0, runStart)
        if (run.length === 0) return null
        if (contentBelow.length !== 1) return null
        const generator = contentBelow[0]
        if (!generator.definition.acceptsUVContext) return null
        if (registry.getChildren(generator.id).length > 0) return null
        if (analyticHasFallbackTriggers(generator)) return null

        composedNodeIds.add(generator.id)
        for (const dNode of run) composedNodeIds.add(dNode.id)

        const {uv, mask} = foldAnalyticRun(run)
        const params = makeParams(generator, undefined, {uvContext: uv})
        params.computeOutputs = initNodeCompute(generator, params)
        const genResult = generator.definition.fragment(params)
        return vec4(genResult.member('rgb'), genResult.member('a').mul(mask))
    }

    // tryFoldAnalyticRunComposite — composite terminus (one photo).
    function tryFoldAnalyticRunComposite(
        sorted: RegistryNode[],
        startIdx: number,
        contentBelow: Expr,
        _processedMasks: Set<string>,
    ): {node: Expr; endIndex: number} | null {
        if (!analyticRunEligible(sorted[startIdx])) return null
        const run: RegistryNode[] = []
        let endIndex = startIdx
        for (let j = startIdx; j < sorted.length; j++) {
            const n = sorted[j]
            if (n.metadata.visible === false) continue
            if (!analyticRunEligible(n)) break
            run.push(n)
            endIndex = j
        }
        if (run.length === 0) return null
        for (const dNode of run) composedNodeIds.add(dNode.id)
        const {uv, mask} = foldAnalyticRun(run)
        const rttInput = applyBlend(ZERO, contentBelow, 'normal', floatE(1))
        const tex = convertToTexture(rttInput)
        const sampled = tex.sample(uv)
        return {node: vec4(sampled.member('rgb'), sampled.member('a').mul(mask)), endIndex}
    }

    // tryComposeViaAnalyticUVRemapNested — nested single-child chain.
    function tryComposeViaAnalyticUVRemapNested(parent: RegistryNode, children: RegistryNode[], processedMasks: Set<string>): Expr | null {
        if (!parent.definition.uvRemap) return null
        if (analyticHasFallbackTriggers(parent)) return null
        const MAX = 8
        const chain: RegistryNode[] = []
        let cursor = parent
        let hopChildren = children
        let terminusGenerator: RegistryNode | null = null
        let terminusComposite: RegistryNode | null = null
        for (let depth = 0; depth < MAX; depth++) {
            chain.push(cursor)
            const visible = hopChildren.filter((c) => c.metadata.visible !== false)
            if (visible.length !== 1) return null
            const next = visible[0]
            const nextChildren = registry.getChildren(next.id)
            if (next.definition.acceptsUVContext && nextChildren.length === 0 && !analyticHasFallbackTriggers(next)) {
                terminusGenerator = next
                break
            }
            if (next.definition.uvRemap && !analyticHasFallbackTriggers(next) && nextChildren.length === 1) {
                cursor = next
                hopChildren = nextChildren
                continue
            }
            terminusComposite = next
            break
        }
        if (!terminusGenerator && !terminusComposite) return null
        for (const c of chain) composedNodeIds.add(c.id)
        const run = [...chain].reverse()
        const {uv, mask} = foldAnalyticRun(run)

        let result: Expr
        if (terminusGenerator) {
            composedNodeIds.add(terminusGenerator.id)
            const params = makeParams(terminusGenerator, undefined, {uvContext: uv})
            params.computeOutputs = initNodeCompute(terminusGenerator, params)
            const gen = terminusGenerator.definition.fragment(params)
            result = vec4(gen.member('rgb'), gen.member('a').mul(mask))
        } else {
            const below = composeNode(terminusComposite!.id, processedMasks)
            // The terminus child's OWN opacity + blend mode. `composeNode` does not apply either —
            // they belong to the sibling loop, which this path bypasses, so without this a
            // distortion's single child ignores its opacity slider entirely. Blended over WHITE0,
            // the same transparent-WHITE base composeSiblings hands a first child (a transparent
            // BLACK base would collapse multiply-like modes to black); `base.a == 0` means the
            // result is still premultiplied, which is what the RTT stores.
            const terminusBlend = (terminusComposite!.metadata.blendMode ?? 'normal') as BlendMode
            const rttInput = applyBlend(WHITE0, below, terminusBlend, opacityExpr(terminusComposite!))
            const tex = convertToTexture(rttInput)
            const sampled = tex.sample(uv)
            result = vec4(sampled.member('rgb'), sampled.member('a').mul(mask))
        }
        const transformed = hasBox(parent) ? applyBoxTransform(result, parent) : applyNodeTransform(result, parent)
        return applyMaskIfNeeded(transformed, parent, processedMasks)
    }

    // tryComposeViaUVContext — providesUVContextViaCompute chain.
    function tryComposeViaUVContext(parent: RegistryNode, children: RegistryNode[], processedMasks: Set<string>): Expr | null {
        const hasTriggers = (n: RegistryNode): boolean =>
            !!n.metadata.mask ||
            (n.metadata.opacity !== undefined && n.metadata.opacity < 1) ||
            (n.metadata.blendMode ?? 'normal') !== 'normal' ||
            needsTransformation(n.metadata.transform) ||
            hasBox(n)
        if (!parent.definition.providesUVContextViaCompute) return null
        if (hasTriggers(parent)) return null

        const chain: RegistryNode[] = []
        let cursor = parent
        let terminus: RegistryNode | null = null
        let hopChildren = children
        const MAX = 8
        for (let depth = 0; depth < MAX; depth++) {
            chain.push(cursor)
            const visible = hopChildren.filter((c) => c.metadata.visible !== false)
            if (visible.length !== 1) return null
            const next = visible[0]
            if (hasTriggers(next)) return null
            const nextHasChildren = registry.getChildren(next.id).length > 0
            if (next.definition.acceptsUVContext && !nextHasChildren) {
                terminus = next
                break
            }
            if (next.definition.providesUVContextViaCompute) {
                cursor = next
                hopChildren = registry.getChildren(next.id)
                continue
            }
            return null
        }
        if (!terminus) return null

        composedNodeIds.add(terminus.id)
        for (const link of chain) composedNodeIds.add(link.id)

        // Dispatch each link's compute; thread upstream uv map. Each link must expose a
        // uvMaskMap compute output (else ineligible → RTT path).
        const linkParams: GpuFragmentParams[] = []
        let upstream: KitTexture | GpuStorageBuffer | undefined
        for (let i = 0; i < chain.length; i++) {
            const link = chain[i]
            const params = makeParams(link, undefined, {
                computeOutputs: upstream ? {_upstreamUVMap: upstream} : undefined,
            })
            const outputs = initNodeCompute(link, params)
            if (!outputs || !outputs.uvMaskMap) return null
            params.computeOutputs = outputs
            linkParams.push(params)
            upstream = outputs.uvMaskMap
        }
        // Sample the innermost uv map at the base UV → the terminus generator's uvContext.
        const finalUVMap = upstream as KitTexture
        const finalSample = finalUVMap.sample(sysCtx.uv)
        const lookedUpUV = vec4(finalSample.member('r'), finalSample.member('g'), 0, 0).member('xy')

        const termParams = makeParams(terminus, undefined, {uvContext: lookedUpUV})
        termParams.computeOutputs = initNodeCompute(terminus, termParams)
        let composed = terminus.definition.fragment(termParams)
        for (let i = chain.length - 1; i >= 0; i--) {
            linkParams[i].childNode = composed
            // The link's child (the downstream generator) is already evaluated at this link's
            // looked-up UV, so the link fragment composites it directly (Surface3D reads this flag to
            // pick the composite path over its RTT-resample fallback).
            linkParams[i].uvPropagationActive = true
            composed = chain[i].definition.fragment(linkParams[i])
        }
        const transformed = hasBox(parent) ? applyBoxTransform(composed, parent) : applyNodeTransform(composed, parent)
        return applyMaskIfNeeded(transformed, parent, processedMasks)
    }

    // ── run the walk ────────────────────────────────────────────────────────────────────
    const rootExpr = registry.rootId ? composeNode(registry.rootId) : ZERO

    // ── serialise RTT passes (creation order == leaves→root) ────────────────────────────
    const rttPasses: RttPassSpec[] = rttBoundaries.map((b) => {
        const {body, externals} = serialiseFragment(b.root, (composed) => composed, options.flipY ?? true, layouts.uniform)
        return {textureKey: b.key, fragment: buildFragmentEntry(body, externals, `rtt_${b.key}`, layouts.sampler, unfilterableKeys)}
    })

    // ── serialise the final pass (tonemap + sRGB OETF tail) ─────────────────────────────
    const toneMode = options.toneMapping ?? 'linear'
    const finalSer = serialiseFragment(
        rootExpr,
        (composed, ctx) => finalTail(composed, ctx, toneMode, options.premultiplyAlpha ?? false),
        options.flipY ?? true,
        layouts.uniform,
    )
    const finalPass = buildFragmentEntry(finalSer.body, finalSer.externals, 'final', layouts.sampler, unfilterableKeys)

    return {
        computeSteps,
        rttPasses,
        finalPass,
        textures,
        externalTextures,
        layouts: {
            uniforms: layouts.uniform,
            samplers: layouts.sampler,
        },
        onBeforeRender,
        onAfterRender,
        onResize,
        onCleanup,
        composedNodeIds,
        usesCustomWgsl,
    }
}

// ── final-pass tail: composed vec4 → tonemap(rgb) → linearToSrgb → (premultiply?) ─────────
function finalTail(composed: string, ctx: EmitContext, mode: ToneMappingMode, premultiply: boolean): string {
    const toneFn = ctx.external(tonemapFns[mode] ?? tonemapFns.linear, `tone_${mode}`)
    const oetf = ctx.external(linearToSrgb, 'linearToSrgb')
    const local = ctx.freshLocal('composed')
    ctx.statement(`var ${local} = ${composed};`)
    const rgb = `${oetf}(${toneFn}(${local}.rgb))`
    return premultiply ? `vec4f(${rgb} * ${local}.a, ${local}.a)` : `vec4f(${rgb}, ${local}.a)`
}

// ── build a fragmentFn entry from a body + externals ──────────────────────────────────────
// Scans the emitted body for `tex.$.<key>` / `ext.$.<key>` / `samp.$.…` and builds the
// per-pass texture + external layouts from EXACTLY the keys this pass reads (so an RTT pass
// never binds its own output texture). Adds those layouts to the externals so `tgpu.resolve`
// links the raw accessors to concrete bindings.
function buildFragmentEntry(
    body: string,
    externals: Record<string, unknown>,
    name: string,
    samplerLayout: TgpuBindGroupLayout,
    unfilterableKeys: ReadonlySet<string> = new Set(),
): FragmentSpec {
    const reads = uniqueMatches(body, /\btex\.\$\.(\w+)/g)
    const externalReads = uniqueMatches(body, /\bext\.\$\.(\w+)/g)
    const usesSamplers = /\bsamp\.\$\./.test(body)

    let textureLayout: TgpuBindGroupLayout | undefined
    if (reads.length > 0) {
        const entries: Record<string, {texture: ReturnType<typeof d.texture2d>; sampleType?: GPUTextureSampleType}> = {}
        // Float32 textures (volumetric SDF field) must be declared 'unfilterable-float':
        // TypeGPU's default 'float' entry only validates against a float32 view when the device
        // has the optional `float32-filterable` feature — absent on most Android GPUs.
        for (const key of reads) {
            entries[key] = unfilterableKeys.has(key)
                ? {texture: d.texture2d(d.f32), sampleType: 'unfilterable-float'}
                : {texture: d.texture2d(d.f32)}
        }
        textureLayout = tgpu.bindGroupLayout(entries).$idx(BIND_GROUPS.textures) as TgpuBindGroupLayout
        externals.tex = textureLayout
    }
    let externalLayout: TgpuBindGroupLayout | undefined
    if (externalReads.length > 0) {
        const entries: Record<string, {externalTexture: ReturnType<typeof d.textureExternal>}> = {}
        for (const key of externalReads) entries[key] = {externalTexture: d.textureExternal()}
        // The external layout must take the LOWEST FREE bind-group index, not a fixed
        // 3. TypeGPU builds `usedBindGroupLayouts` as a SPARSE array indexed by each layout's `.index`;
        // a hole (e.g. samplers at 2 with no texture at 1 — the VideoTexture case) makes
        // `new Set(sparseArray)` yield an `undefined` that `forEach` skips, so it is reported as a
        // phantom "<unnamed>" missing bind group at draw. Group 0 is already anchored (the `_sys.time`
        // touch), and a texture sample always pulls in the sampler, so the only possible hole is
        // group 1 (textures) when a pass samples an external texture but no regular texture. Filling
        // the lowest free index keeps the array dense.
        const occupied = new Set<number>([BIND_GROUPS.uniforms])
        if (reads.length > 0) occupied.add(BIND_GROUPS.textures)
        if (usesSamplers) occupied.add(BIND_GROUPS.samplers)
        let externalIdx = 0
        while (occupied.has(externalIdx)) externalIdx++
        externalLayout = tgpu.bindGroupLayout(entries).$idx(externalIdx) as TgpuBindGroupLayout
        externals.ext = externalLayout
    }
    if (usesSamplers) externals.samp = samplerLayout

    const entry = tgpu
        .fragmentFn({in: {uv: d.vec2f}, out: d.vec4f})(body)
        .$uses(externals)
        .$name(name) as unknown as TgpuFragmentFn
    return {body, externals, entry, reads, externalReads, usesSamplers, textureLayout, externalLayout}
}

function uniqueMatches(text: string, re: RegExp): string[] {
    const out = new Set<string>()
    for (const m of text.matchAll(re)) out.add(m[1])
    return [...out]
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Helpers: channel extraction + remap + placeholder kit fns for bbox/transform
// ═══════════════════════════════════════════════════════════════════════════════════════

/** Extract the driving scalar from a sampled color (resolveMapForProp channels). */
function extractChannel(sampled: Expr, channel: string): Expr {
    switch (channel) {
        case 'alpha':
            return sampled.member('a')
        case 'alphaInverted':
            return new Expr((ctx) => `(1.0 - ${sampled._emit(ctx)}.a)`)
        case 'luminanceInverted':
            return new Expr((ctx) => `(1.0 - dot(${sampled._emit(ctx)}.rgb, vec3f(0.2126, 0.7152, 0.0722)))`)
        case 'luminance':
        default:
            return new Expr((ctx) => `dot(${sampled._emit(ctx)}.rgb, vec3f(0.2126, 0.7152, 0.0722))`)
    }
}

/** map remap: mix(oMin, oMax, pow(clamp((raw-iMin)/max(iMax-iMin,0.0001),0,1), pow(2, -curve*2))). */
function remapExpr(raw: Expr, iMin: Expr, iMax: Expr, oMin: Expr, oMax: Expr, curve: Expr): Expr {
    return new Expr((ctx) => {
        const r = raw._emit(ctx)
        const im = iMin._emit(ctx)
        const ix = iMax._emit(ctx)
        const om = oMin._emit(ctx)
        const ox = oMax._emit(ctx)
        const cv = curve._emit(ctx)
        const norm = `clamp((${r} - ${im}) / max(${ix} - ${im}, 0.0001), 0.0, 1.0)`
        const eased = `pow(${norm}, pow(2.0, -${cv} * 2.0))`
        return `mix(${om}, ${ox}, ${eased})`
    })
}

// The bbox / transform kit fns are ports living in kit/uvTransform.ts (applyRectangularClipMask,
// applyUVTransform, boundingBoxToGeneratorUVContext). Imported lazily as externals.
import {
    applyRectangularClipMask as applyRectangularClipMaskFn,
    applyUVTransform as applyUVTransformFn,
    boundingBoxToGeneratorUVContext as boundingBoxToGeneratorUVContextFn,
} from './kit/uvTransform'
// Expr-level edge handling for the RTT layer-transform path (mirror/wrap re-sample, transparent
// alpha-cut, stretch straight-sample). The string→mode map (stretch=0, transparent=1, mirror=2,
// wrap=3) is inlined here rather than imported from transformations.ts.
import {applyEdgeHandlingExpr} from './kit/edges'

const TRANSFORM_EDGE_MODES: Record<string, number> = {stretch: 0, transparent: 1, mirror: 2, wrap: 3}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Structural hash inputs — collected here (node semantics) for pipelineCache
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Collect the exact recompile-trigger set, in registry order. pipelineCache digests this into
 * the cache key. Any change here that alters emitted WGSL MUST change the hash (that is the
 * invariant the cache tests assert).
 *
 * The walk is pre-order, and pre-order ALONE does not identify a tree — `parent > [A > [B], C]`
 * and `parent > [A > [B, C]]` visit the same four nodes in the same sequence, with the same
 * renderOrder on each. That is a real editing move (drag a layer into a filter, below its existing
 * child): every line would be byte-identical, the cache would serve the pre-move composition, and
 * the dragged layer would keep rendering as if it were still outside. So each line names its own
 * PARENT — with that, the sequence plus the parent links pin the tree exactly.
 */
export function collectStructuralHashInputs(registry: RegistryView, options: ComposeOptions = {}): string[] {
    const parts: string[] = []
    parts.push(`tone:${options.toneMapping ?? 'linear'}`)
    parts.push(`flipY:${options.flipY ?? true}`)
    parts.push(`premul:${options.premultiplyAlpha ?? false}`)

    const visit = (id: string): void => {
        const node = registry.getNode(id)
        if (!node) return
        const m = node.metadata
        const def = node.definition
        const transformActive = needsTransformation(m.transform)
        // Mirrors the composer's `hasBox` (identity boxes emit nothing) so crossing the
        // identity boundary — dragging a full-frame box off full-frame — recompiles.
        const bboxActive = !!m.boundingBox && !!def.boundingBoxDeclaration && !isIdentityBoundingBox(m.boundingBox)
        const opacityBucket = (m.opacity ?? 1) === 0 ? 'zero' : 'nonzero'
        const maskCfg = m.mask ? `${m.mask.source}:${m.mask.type}` : 'none'
        // compile-time-relevant prop values: props marked compileTime. (Array-length / colorStops
        // count triggers are compileTimeWhen, not compileTime, and are fingerprinted separately in
        // the renderer's extraHashInputs — the base hash has no synthetic-field access.)
        const compileTimeProps: string[] = []
        for (const [prop, cfg] of Object.entries(def.props ?? {})) {
            const pc = cfg as {compileTime?: boolean}
            if (pc.compileTime) {
                const h = node.handles[prop]
                compileTimeProps.push(`${prop}=${h ? String(h.value) : '?'}`)
            }
        }
        parts.push(
            [
                `id:${node.id}`,
                // Parentage — see the pre-order note on this function. Without it, re-parenting a
                // layer into a filter below that filter's existing child is invisible to the cache.
                `parent:${node.parentId ?? ''}`,
                `name:${node.componentName}`,
                `rev:${def.revision ?? ''}`,
                `blend:${m.blendMode ?? 'normal'}`,
                `mask:${maskCfg}`,
                `order:${m.renderOrder ?? 0}`,
                `visible:${m.visible !== false}`,
                `opacity:${opacityBucket}`,
                // When a layer transform is active its `edges` mode is a compile-time branch in
                // applyNodeTransform (mirror/wrap re-sample, transparent alpha-cut, stretch straight),
                // so it MUST vary the hash — otherwise the cache serves a stale pipeline on an
                // edges-only change. Omitted when inactive (no branch emitted → mode irrelevant).
                `xform:${transformActive}${transformActive ? `:${m.transform?.edges ?? 'transparent'}` : ''}`,
                `bbox:${bboxActive}:${bboxActive ? (def.boundingBoxDeclaration?.boxResamplesContent ? 'resample' : def.boundingBoxDeclaration?.supportsResizeFit ? 'resize' : 'clip') : ''}`,
                `rtt:${!!def.requiresRTT}`,
                `maps:${m.maps ? Object.keys(m.maps).sort().join(',') : ''}`,
                `ct:${compileTimeProps.sort().join('|')}`,
            ].join('|'),
        )
        for (const child of registry.getChildren(id).sort((a, b) => a.metadata.renderOrder - b.metadata.renderOrder)) {
            visit(child.id)
        }
    }
    if (registry.rootId) visit(registry.rootId)
    return parts
}
