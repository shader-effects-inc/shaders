/**
 * `shaderRendererGPU()` — the WebGPU (TypeGPU) renderer. This is the runtime SHELL: the node
 * registry, registration queue, RAF-batched recompose, the CPU driver layer, the global event
 * manager, sizing/DPR, visibility, the frame loop, screenshots/synthetic export, recording
 * resolution, device loss, cleanup, and performance stats, over the building blocks:
 *
 *   composeNodeTree (composer.ts) → CompositionIR
 *   createUniformStore (uniformStore.ts) → packed per-node uniform struct + FieldHandles
 *   createPassManager (passManager.ts) → RTT lifecycle + ordered GPU passes
 *   createPipelineCache (pipelineCache.ts) → structural-hash cache + swap-when-ready
 *   createTextureManager / createComputeDispatcher / acquireRoot (root.ts)
 *
 * WebGPU-only behavior:
 *   - There are no WebGL paths. `initialize({context})` rejects with a descriptive error.
 *   - WE own the clock (see frame.ts deriveElapsedTime); shader `time` is CPU-accumulated /
 *     shared-origin-derived.
 *   - DPR is a factor WE own in the size math (`pixelRatio`); there is no `setPixelRatio`.
 *   - The composer applies a layer transform strictly when `needsTransformation(transform)`
 *     (part of the structural hash), so a transform returning exactly to default recomposes.
 *     Swap-when-ready hides any flash.
 *
 * SSR-safe: no window/navigator access at import time.
 */
import type {TgpuRoot} from 'typegpu'
import * as d from 'typegpu/data'

import {acquireRoot, configureCanvasContext, type RootContext} from './root'
import {debugError, getGpuUnusableReason, isGpuUnavailableError, markGpuUnusable, type GpuFailureReason, authorError} from './support'
import {createUniformStore, updateFieldValue, FieldHandle, ArrayFieldHandle, type UniformStore, type NodeHandles, type FieldInit} from './uniformStore'
import {SystemUniforms} from './kit/coords'
import {getAnimatedTimeState} from './kit/time'
import {colorStopsTransform, listPropTransform, transformColorSpace, colorToRGBA, setColorSpaceModeGpu} from './transforms'
import {resolveListItems, packList, listFieldName, listCountName, listDriverPath, parseListDriverPath, isPositionDriver, listHasDrivers} from '../utilities/listProps'
import {packStops, packConvertedStops, type ColorStop} from './kit/colorStops'
import {composeNodeTree, collectStructuralHashInputs, type CompositionIR, type ComposeOptions} from './composer'
import {createPipelineCache, structuralHash, type PipelineCache} from './pipelineCache'
import {createPassManager, type PassManager} from './passManager'
import {createTextureManager, type TextureManager, type RenderTexture} from './textures'
import {createComputeDispatcher, type ComputeDispatcher} from './compute'
import {needsTransformation, boundingBoxToUVParams, screenUVToBoxLocal} from './kit/uvTransform'
import {
    applySpring,
    applyEasing,
    clampToTextureCap as clampToTextureCapPure,
    deriveElapsedTime,
    frameGate,
    awaitGpuIdle,
    createFrameLoop,
    runFrameSequence,
    type FrameLoop,
} from './frame'
import type {
    GpuShaderDefinition,
    RegistryView,
    RegistryNode,
    UniformHandleView,
} from './contract'

import {
    buildDimensionalPlan,
    buildExtraSizeProps,
    boxHalfExtentsUV,
    isDimensionalProp,
    resolveDimensionalProp,
    setShaderCanvasDimensions,
    type SizeConversion,
} from '../utilities/dimensionalProps'
import {maxPixelRatioForDevice} from '../utilities/device'
import {isIdentityBoundingBox} from '../utilities/boundingBox'
import {getSdfContentBounds, isSdfContentBoundsScanned} from '../utilities/sdfBounds'
import {computeTreeLayout, type LayoutNodeView} from '../utilities/layout'
import {onNaturalSizeChange} from '../utilities/naturalSize'
import {PerformanceTracker, type PerformanceStats} from '../performanceTracker'
import type {
    NodeMetadata,
    PropDriver,
    MapConfig,
    MouseMapConfig,
    MousePositionConfig,
    AutoAnimateConfig,
    BoundingBoxOrigin,
    ToneMappingMode,
} from '../types'

// Re-export the root helpers (skeleton kept them public; callers may still want them).
export {acquireRoot, configureCanvasContext, getRootForDevice, destroyDefaultGpuDevice} from './root'
export type {AcquireRootOptions, RootContext} from './root'

// Availability probing + the quiet-by-default diagnostics channel.
export {isWebGPUSupported, getWebGPUSupport, setShadersDebug, isShadersDebug, GpuUnavailableError, isGpuUnavailableError} from './support'
export type {GpuFailureReason, WebGPUSupportInfo} from './support'

// ═══════════════════════════════════════════════════════════════════════════════════════
// Failure thresholds
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Uncaptured VALIDATION errors tolerated before we declare the device unusable and go
 * transparent. A healthy renderer emits exactly zero of these, so the bar is set high
 * enough that a transient swap-chain/resize hiccup can never trip it — it exists purely to
 * stop a partially-implemented WebGPU backend from spinning the GPU forever behind a blank
 * canvas. Out-of-memory and internal errors are fatal on the FIRST occurrence.
 */
const MAX_UNCAPTURED_VALIDATION_ERRORS = 32

/**
 * Consecutive frames that may throw before we give up. One-off exceptions (a texture that
 * wasn't ready yet) recover on the next frame and reset the counter; a genuinely broken
 * pipeline throws every frame and hits this within ~80ms.
 */
const MAX_CONSECUTIVE_RENDER_ERRORS = 5

// ═══════════════════════════════════════════════════════════════════════════════════════
// Registration input
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * One prop's registration input. Carries the plain current `value` plus the CPU `transform`
 * and the compile-time flags. The uniform store applies the transform when seeding/patching the
 * packed struct field (see uniformStore.FieldHandle). The framework templates produce this
 * shape.
 */
export interface GpuUniformInput {
    /** Current value (raw, or dimensionally-resolved for shape props). The store transforms it. */
    value: unknown
    /** CPU transform (transformColor / transformPosition / transformAngle / …). */
    transform?: (value: unknown) => unknown
    /** Recompiles the composition when this prop changes (baked into the WGSL at build). */
    compileTime?: boolean
    /** Recompiles only when the predicate fires (e.g. blur crossing 0). */
    compileTimeWhen?: (previousValue: unknown, newValue: unknown) => boolean
    /** @internal last value a compileTimeWhen recompose was keyed to. */
    _lastCompiledValue?: unknown
    /** @internal raw (px / DimensionalValue) value for resize re-resolution of shape props. */
    _rawDimensional?: unknown
    /** String / JSON / origin / select props — CPU-only, never enter the packed struct. */
    cpu?: boolean
    /** @internal list props: the RAW items (drivers included) the renderer re-resolves per frame. */
    _rawList?: unknown
    /** @internal list props: the item spec (fields / maxItems) the renderer packs with. */
    _listSpec?: import('../utilities/listProps').ListPropSpec
    /** Explicit WGSL schema override (else inferred from the transformed value). */
    schema?: import('typegpu/data').AnyWgslData
}

/** A node's registration input, keyed by prop name. */
export type GpuUniformsMap = Record<string, GpuUniformInput>

// ═══════════════════════════════════════════════════════════════════════════════════════
// Initialize options (WebGPU-only)
// ═══════════════════════════════════════════════════════════════════════════════════════

export interface GpuInitializeOptions {
    canvas: HTMLCanvasElement
    /** Element to observe for resize. Defaults to `canvas.parentElement`. */
    resizeTarget?: HTMLElement
    enablePerformanceTracking?: boolean
    colorSpace?: 'p3-linear' | 'srgb'
    toneMapping?: ToneMappingMode
    /** Inject a shared device (Projects multi-tile) — one root per device. */
    gpu?: {device: GPUDevice; adapter?: GPUAdapter}
    forceFullFrameRate?: boolean
    /** When false, skips ResizeObserver + IntersectionObserver — use resize()/start()/stop(). */
    observeElement?: boolean
    /** Shared wall-clock time origin (performance.now ms) for synced multi-renderer animation. */
    timeOrigin?: number
    /**
     * REJECTED. This renderer is WebGPU-only and throws a descriptive error if a context is
     * passed. Present only to give a clear error, not for use.
     */
    context?: unknown
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Internal node model
// ═══════════════════════════════════════════════════════════════════════════════════════

interface MouseDriverState {
    currentX: number
    currentY: number
    velocityX: number
    velocityY: number
    /**
     * Set once the state has seen a REAL pointer position. A state created before any pointer
     * event seeds from the off-screen park (-10) — on the first real event it must SNAP to the
     * cursor, not spring across the canvas (content visibly flying in from off-screen).
     */
    pointerSnapped?: boolean
    /** Live pre-flip box-local point for UI readback (mouse-position). */
    liveX?: number
    liveY?: number
    /** Live resolved scalar for UI readback (mouse-map range). */
    liveScalar?: number
}

/** Resolved map/mouse remap bounds (plain numbers), refreshed on metadata + resize. */
interface MapValues {
    inputMin: number
    inputMax: number
    outputMin: number
    outputMax: number
    curve: number
}

interface GpuNodeInfo {
    id: string
    componentName: string
    parentId: string | null
    definition: GpuShaderDefinition
    metadata: NodeMetadata
    uniforms: GpuUniformsMap
    domCanvas?: HTMLCanvasElement

    // Declarative flags copied from the definition (fast reads on the hot path).
    requiresRTT: boolean
    requiresChild: boolean
    blendWithChildren: boolean
    acceptsUVContext: boolean
    providesUVContextViaCompute: boolean
    usesPointer: boolean
    hasUvRemap: boolean
    boundingBoxDeclaration?: GpuShaderDefinition['boundingBoxDeclaration']
    wantsBoundsParams: boolean
    extraDimProps?: Map<string, SizeConversion>

    // Persistent CPU driver state (survives recompose).
    mouseDriverState: Record<string, MouseDriverState>
    autoAnimateState: Record<string, {}>
    mapValues: Record<string, MapValues>

    /** Live per-field handles from the CURRENTLY BOUND composition's store (props + synthetic). */
    _liveHandles: NodeHandles | null
}

interface QueuedRegistration {
    id: string
    definition: GpuShaderDefinition
    parentId: string | null
    metadata: NodeMetadata | null
    uniforms: GpuUniformsMap
    domCanvas?: HTMLCanvasElement
}

/** A fully-built composition cached in the pipeline cache (own store + own passManager). */
interface BuiltComposition {
    hash: string
    store: UniformStore
    ir: CompositionIR
    passManager: PassManager
    handlesById: Map<string, NodeHandles>
    /** True on the first getOrBuild that created it (skip re-seed); cleared after binding. */
    fresh: boolean
    /**
     * Frames left in the validation window. While a custom WGSL body is in play, the first
     * frames of a fresh composition render inside a validation error scope; a captured error
     * marks the composition `broken` (nothing draws until the structure changes) instead of
     * counting toward the fatal uncaptured-error limit — an author's typo must not stop the
     * renderer for good.
     */
    validationFrames: number
    /** The GPU validation message that broke this composition, when it did. */
    broken?: string
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Global window-event manager
// ═══════════════════════════════════════════════════════════════════════════════════════

interface PointerHandlers {
    onMouseMove: (e: MouseEvent) => void
    onTouchMove: (e: TouchEvent) => void
    onMouseDown: (e: MouseEvent) => void
    onTouchStart: (e: TouchEvent) => void
    onMouseUp: () => void
    onTouchEnd: () => void
}

const globalEventManager = (() => {
    let instanceCount = 0
    const callbacks = new Set<PointerHandlers>()
    const fan = <K extends keyof PointerHandlers>(key: K, arg?: unknown): void => {
        callbacks.forEach((cb) => {
            try {
                ;(cb[key] as (a?: unknown) => void)(arg)
            } catch (error) {
                debugError('[Shaders] Error in pointer handler:', error)
            }
        })
    }
    const onMove = (e: MouseEvent) => fan('onMouseMove', e)
    const onTouch = (e: TouchEvent) => fan('onTouchMove', e)
    const onDown = (e: MouseEvent) => fan('onMouseDown', e)
    const onStart = (e: TouchEvent) => fan('onTouchStart', e)
    const onUp = () => fan('onMouseUp')
    const onEnd = () => fan('onTouchEnd')
    // Press state is window-level like move (a canvas is routinely covered by overlaid content —
    // hero copy, editor hit layers — that would swallow a canvas-attached mousedown; each renderer
    // bounds-checks against its own canvas rect). down/up use capture so an overlay's
    // stopPropagation can't block the press or, worse, strand pointerActive after release.
    return {
        register(handlers: PointerHandlers): () => void {
            callbacks.add(handlers)
            instanceCount++
            if (instanceCount === 1 && typeof window !== 'undefined') {
                window.addEventListener('mousemove', onMove)
                window.addEventListener('touchmove', onTouch)
                window.addEventListener('mousedown', onDown, true)
                window.addEventListener('touchstart', onStart, true)
                window.addEventListener('mouseup', onUp, true)
                window.addEventListener('touchend', onEnd, true)
            }
            return () => {
                callbacks.delete(handlers)
                instanceCount--
                if (instanceCount === 0 && typeof window !== 'undefined') {
                    window.removeEventListener('mousemove', onMove)
                    window.removeEventListener('touchmove', onTouch)
                    window.removeEventListener('mousedown', onDown, true)
                    window.removeEventListener('touchstart', onStart, true)
                    window.removeEventListener('mouseup', onUp, true)
                    window.removeEventListener('touchend', onEnd, true)
                }
            }
        },
    }
})()

/**
 * True when a shape JSON config has any sub-prop driven by a mouse config (resolved CPU-side
 * in the SDF sampler, not via metadata.maps) — so the renderer knows to attach pointer
 * listeners.
 */
function shapeConfigUsesMouse(raw: unknown): boolean {
    if (!raw) return false
    let cfg: unknown = raw
    if (typeof raw === 'string') {
        try {
            cfg = JSON.parse(raw)
        } catch {
            return false
        }
    }
    if (!cfg || typeof cfg !== 'object') return false
    for (const key in cfg as Record<string, unknown>) {
        const v = (cfg as Record<string, unknown>)[key]
        if (v && typeof v === 'object' && (v as {type?: string}).type === 'mouse') return true
    }
    return false
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// shaderRendererGPU
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Construct a WebGPU renderer instance.
 */
export function shaderRendererGPU() {
    // ── Device / root / GPU services ────────────────────────────────────────────────────
    let rootCtx: RootContext | null = null
    let root: TgpuRoot | null = null
    let context: GPUCanvasContext | null = null
    let textureManager: TextureManager | null = null
    let dispatcher: ComputeDispatcher | null = null

    // ── Canvas / lifecycle ────────────────────────────────────────────────────────────────
    let canvasElement: HTMLCanvasElement | null = null
    let isInitialized = false
    let isInitializing = false
    let isRendererReady = false
    let initializationAbortController: AbortController | null = null

    // ── Registry ─────────────────────────────────────────────────────────────────────────
    const nodes = new Map<string, GpuNodeInfo>()
    let rootId: string | null = null
    const parentToChildren = new Map<string | null, Set<string>>()
    let pendingRegistrationQueue: QueuedRegistration[] = []
    /**
     * Uniform / metadata updates that arrived for a node still sitting in the registration
     * queue. Hosts mount their whole tree before `initialize()` resolves a device, so a prop
     * that changes in that window (a `visible` flag flipped from a mount effect, a media
     * query, a store hydration) would otherwise hit `nodes.get(id) === undefined` and be
     * dropped for good — the queue would then flush the mount-time snapshot and the node
     * would sit at a value the host has already moved on from. Replayed in arrival order
     * right after the flush, so every update runs its normal side effects against a live node.
     */
    let pendingNodeUpdates: Array<
        | {id: string; kind: 'uniform'; name: string; value: unknown}
        | {id: string; kind: 'metadata'; metadata: Partial<NodeMetadata>}
    > = []

    // ── Composition state ───────────────────────────────────────────────────────────────
    const pipelineCache: PipelineCache<BuiltComposition> = createPipelineCache<BuiltComposition>({
        maxSize: 4,
        dispose: (bc) => disposeComposition(bc),
    })
    const liveCompositions = new Set<BuiltComposition>()
    let boundComposition: BuiltComposition | null = null
    let desiredHash: string | null = null
    let structuralDirty = true

    // ── Sizing / DPR (WE own the pixel ratio now) ───────────────────────────────────────
    let currentWidth = 0 // CSS-px backing-buffer size before DPR
    let currentHeight = 0
    let logicalWidth = 0 // unclamped authored-frame size (px-unit resolution basis)
    let logicalHeight = 0
    let pixelRatio = 1 // the DPR factor we back the canvas at (setResolutionScale / recording adjust it)
    let cachedMaxTextureDim = 8192 // WebGPU spec minimum until refined post-init
    let lastRequestedWidth = 0
    let lastRequestedHeight = 0
    let hasInitialDimensions = false
    const pxResWidth = () => logicalWidth || currentWidth || 1
    const pxResHeight = () => logicalHeight || currentHeight || 1

    // ── Animation / visibility / throttle ───────────────────────────────────────────────
    let shouldAnimate = true
    let isVisible = false
    let forceFullFrameRate = false
    let lastRenderTime = 0
    // Host-imposed on-screen frame interval cap (ms), 0 = default 60 FPS.
    // See setFrameRateCap — a multi-tile canvas demotes tiny tiles with this.
    let frameIntervalCap = 0
    let globalElapsedTime = 0
    let sharedTimeOrigin: number | null = null
    let pendingRenderRAF: number | null = null

    // ── Pointer ─────────────────────────────────────────────────────────────────────────
    // Starts far outside normalized UV space (0..1 is "on canvas") rather than dead-center.
    // Hosts that never forward real pointer events (Framer's static preview, thumbnails,
    // headless renders) would otherwise make every cursor-driven effect look like the mouse
    // is permanently parked in the middle — visibly wrong, and not actually "interactive" in
    // those contexts. Off-screen reads as "no cursor yet" to any distance/falloff-based
    // effect until a real pointermove/touchmove lands.
    let pointerX = -10
    let pointerY = -10
    let pointerActive = false
    /**
     * False until the FIRST real pointer/touch event lands on this canvas. Every consumer of the
     * parked position needs to know the difference between "the cursor is here" and "no cursor has
     * ever been seen": velocity trackers must not measure a delta across the park→real transition
     * (a giant impulse/streak), and mouse-driver springs must snap to the first real position
     * instead of animating content in from off-screen.
     */
    let pointerSeen = false
    let cachedCanvasRect: DOMRect | null = null
    let mouseListenersNeeded = false
    let globalEventUnregister: (() => void) | null = null

    // ── Observers / handlers ────────────────────────────────────────────────────────────
    let resizeObserver: ResizeObserver | null = null
    let intersectionObserver: IntersectionObserver | null = null
    let unloadHandler: (() => void) | null = null
    let windowResizeHandler: (() => void) | null = null
    let pendingResize: {width: number; height: number} | null = null
    let isResizeScheduled = false

    // ── Options ─────────────────────────────────────────────────────────────────────────
    let rendererColorSpace: 'p3-linear' | 'srgb' = 'p3-linear'
    let rendererToneMapping: ToneMappingMode = 'linear'
    let enablePerformanceTracking = false
    let setColorSpaceModeFn: ((mode: 'p3-linear' | 'srgb') => void) | null = null

    // ── onReady / device-loss callbacks ─────────────────────────────────────────────────
    let onReadyCallback: (() => void) | null = null
    let hasEmittedReady = false
    let firstRenderPending = false
    let onDeviceLostCallback: ((reason: string) => void) | null = null
    let deviceLostFired = false

    // ── Fatal-failure state (WebGPU unavailable / device unusable) ──────────────────────
    /**
     * Set once, permanently, the first time this renderer decides it cannot draw. Every
     * entry point checks it: `initialize` refuses to retry into the same wall, the frame
     * loop is stopped, and the canvas is left transparent. Deliberately NOT cleared by
     * `cleanup()` — a renderer that failed for environmental reasons will fail again, and
     * hosts that re-init on a prop change must not restart the failure cycle.
     */
    let failureReason: GpuFailureReason | null = null
    /** Survives `cleanup()` so a host still hears about a failure raised during teardown. */
    let onUnavailableCallback: ((reason: GpuFailureReason, error?: unknown) => void) | null = null
    let hasNotifiedUnavailable = false
    /** The device we attached the `uncapturederror` listener to (detached on cleanup). */
    let uncapturedErrorDevice: GPUDevice | null = null
    let uncapturedErrorHandler: ((event: Event) => void) | null = null
    let uncapturedValidationErrors = 0
    let consecutiveRenderErrors = 0

    // ── Performance ─────────────────────────────────────────────────────────────────────
    const performanceTracker = new PerformanceTracker()

    // ── Reusable per-frame params (avoids per-frame allocation) ─────────────────────────
    const frameParams = {
        deltaTime: 0,
        pointer: {x: -10, y: -10, seen: false},
        pointerActive: false,
        dimensions: {width: 0, height: 0},
    }

    // ── Frame loop (RAF; throttle decision lives in renderFrameInternal) ────────────────
    const frameLoop: FrameLoop = createFrameLoop(() => renderFrame())

    // ────────────────────────────────────────────────────────────────────────────────────
    // Registry helpers
    // ────────────────────────────────────────────────────────────────────────────────────

    const sanitizeId = (id: string): string => id.replace(/[^a-zA-Z0-9_]/g, '_')

    const findChildNodes = (parentId: string): GpuNodeInfo[] => {
        const childIds = parentToChildren.get(parentId)
        if (!childIds) return []
        const out: GpuNodeInfo[] = []
        for (const cid of childIds) {
            const info = nodes.get(cid)
            if (info) out.push(info)
        }
        return out
    }

    const findNodeByCustomId = (customId: string): string | null => {
        const sanitized = sanitizeId(customId)
        for (const [nodeId, info] of nodes) {
            if (info.metadata.id === sanitized) return nodeId
        }
        return null
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Dimensional (px / origin) resolution
    // ────────────────────────────────────────────────────────────────────────────────────

    /** Per-axis half-extent for non-centre origin resolution. */
    const computeDimExtent = (
        node: GpuNodeInfo,
        origin: BoundingBoxOrigin,
    ): {hwx: number; hhy: number} | undefined => {
        if (origin === 'center') return undefined
        const decl = node.boundingBoxDeclaration
        if (!decl) return undefined
        const snap: Record<string, unknown> = {}
        for (const [k, u] of Object.entries(node.uniforms)) {
            snap[k] = u._rawDimensional !== undefined ? u._rawDimensional : u.value
        }
        subscribeSdfBoundsReresolve(node, snap.shapeSdfUrl)
        return boxHalfExtentsUV(decl, snap, pxResWidth(), pxResHeight())
    }

    /**
     * Custom-SVG shape effects anchor against CONTENT-TIGHT bounds that arrive from an async scan of
     * the SDF (`getSdfContentBounds`). Until it lands, `shapeContentExtent` falls back to the full
     * field — so a non-centre origin resolves the position against the wrong half-extent and the
     * shape sits off its anchor. Nothing re-resolved it before: the scan filled its cache and the
     * node stayed put until some unrelated prop patch happened to re-run the resolve (a no-op HMR
     * edit "fixed" it). Subscribe once per node+url so the node re-anchors the frame the scan lands.
     */
    const sdfBoundsSubscribed = new WeakMap<GpuNodeInfo, Set<string>>()
    const subscribeSdfBoundsReresolve = (node: GpuNodeInfo, url: unknown): void => {
        if (typeof url !== 'string' || !url || isSdfContentBoundsScanned(url)) return
        const seen = sdfBoundsSubscribed.get(node) ?? new Set<string>()
        sdfBoundsSubscribed.set(node, seen)
        if (seen.has(url)) return
        seen.add(url)
        getSdfContentBounds(url, () => {
            seen.delete(url)
            // The node may have been removed (or re-registered as a new object) while the scan ran.
            if (nodes.get(node.id) !== node) return
            reresolveDimensionalUniforms(node)
            // Layout overrides in-flow children's positions — rerun it so the re-resolved authored
            // value doesn't stick where the group placed the node (mirrors updateUniformValue).
            if (anyLayoutGroups()) markLayoutDirty()
            requestRender()
        })
    }

    /** Re-resolve a node's dimensional props against the live canvas size. */
    const reresolveDimensionalUniforms = (node: GpuNodeInfo): void => {
        const decl = node.boundingBoxDeclaration
        const extra = node.extraDimProps
        const plan = buildDimensionalPlan(decl, extra)
        if (!plan) return
        const origin = (node.uniforms['origin']?.value as BoundingBoxOrigin) ?? 'center'
        const he = computeDimExtent(node, origin)
        const names = [...plan.positionProps, ...plan.sizeProps.keys()]
        for (const name of names) {
            const u = node.uniforms[name]
            if (!u || u._rawDimensional === undefined) continue
            const resolved = resolveDimensionalProp(decl, name, u._rawDimensional, origin, pxResWidth(), pxResHeight(), extra, he)
            u.value = resolved
            writePropHandle(node, name, resolved)
        }
    }

    /** Resolve a driver output bound (number or px DimensionalValue) to UV. */
    const resolveDynamicBound = (node: GpuNodeInfo, propName: string, bound: unknown): number => {
        if (typeof bound === 'number') return bound
        const origin = (node.uniforms['origin']?.value as BoundingBoxOrigin) ?? 'center'
        const resolved = resolveDimensionalProp(node.boundingBoxDeclaration, propName, bound, origin, pxResWidth(), pxResHeight(), node.extraDimProps)
        return typeof resolved === 'number' ? resolved : ((bound as {value?: number})?.value ?? 0)
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // CPU value → synthetic-field resolution (transform / bbox / driver params)
    // ────────────────────────────────────────────────────────────────────────────────────

    const aspectRatio = (): number => {
        const {width, height} = bufferSize()
        return width / Math.max(1, height)
    }

    const transformFieldValues = (node: GpuNodeInfo): Record<string, number> => {
        const t = node.metadata.transform
        return {
            offsetX: t?.offsetX ?? 0,
            offsetY: t?.offsetY ?? 0,
            rotation: t?.rotation ?? 0,
            scale: t?.scale ?? 1,
            anchorX: t?.anchorX ?? 0.5,
            anchorY: t?.anchorY ?? 0.5,
            aspectRatio: aspectRatio(),
        }
    }

    const bboxFieldValues = (node: GpuNodeInfo): Record<string, number> | null => {
        const bbox = node.metadata.boundingBox
        if (!bbox) return null
        const p = boundingBoxToUVParams(bbox, pxResWidth(), pxResHeight())
        return {
            centerX: p.centerX,
            centerY: p.centerY,
            halfWidth: p.halfWidthUV,
            halfHeight: p.halfHeightUV,
            cornerRadius: p.cornerRadiusUV,
            rotation: p.rotation,
            aspectRatio: aspectRatio(),
        }
    }

    // Identity full-frame boxes are not active: the composer emits no `_bbox_*` reader for them,
    // so allocating/driving the fields would be dead weight. Kept in lockstep with the composer's
    // `hasBox` + `collectStructuralHashInputs` — all three share `isIdentityBoundingBox`.
    const bboxActive = (node: GpuNodeInfo): boolean =>
        !!node.metadata.boundingBox &&
        !!node.boundingBoxDeclaration &&
        !isIdentityBoundingBox(node.metadata.boundingBox)

    // CPU-resolve a node's shape bounds (canvas-UV, Y-down centre). Half-extents via
    // boxHalfExtentsUV; centre from the position propBinding's live value; child rotation ignored
    // (axis-aligned screen rect). Returns null when the decl carries no usable size. `.value` is
    // RAW (pre-transform), so `cy = value.y` is the un-flipped centre directly.
    const computeNodeBoundsUV = (
        node: GpuNodeInfo,
    ): {centerX: number; centerY: number; halfWidth: number; halfHeight: number} | null => {
        const decl = node.boundingBoxDeclaration
        if (!decl || !node.uniforms) return null
        if (!decl.computeBounds && !decl.propBindings) return null
        const snap: Record<string, unknown> = {}
        for (const [k, u] of Object.entries(node.uniforms)) {
            snap[k] = u._rawDimensional !== undefined ? u._rawDimensional : u.value
        }
        const he = boxHalfExtentsUV(decl, snap, pxResWidth(), pxResHeight())
        if (he.hwx <= 0 && he.hhy <= 0) return null
        let cx = 0.5
        let cy = 0.5
        const posProp = decl.propBindings?.x?.prop ?? decl.propBindings?.y?.prop
        const posValue = posProp ? node.uniforms[posProp]?.value : undefined
        if (posValue && typeof (posValue as {x?: unknown}).x === 'number' && typeof (posValue as {y?: unknown}).y === 'number') {
            cx = (posValue as {x: number}).x
            cy = (posValue as {y: number}).y
        }
        // Driven position: the raw uniform never moves — a mouse-position driver renders through
        // the `_smoothed_<prop>` field, so reading `.value` alone freezes the rect at the authored
        // centre while the shape follows the pointer (Repeater then culls it to that stale rect).
        // metadata.maps is the single canonical driver representation (every host strips PropDriver
        // values out of uniforms into maps). Read the live CPU-resolved centre: liveX/liveY are the
        // final screen-UV point (per-axis pins + reach/origin/invert applied, y-down like the raw
        // value; the box-local remap can't apply here — an active-bbox child never reaches this
        // resolver). Before the first driver tick fall back per-axis: pinned axes to their config
        // number, tracked axes to the seeded smoothed pointer.
        if (posProp) {
            const driver = node.metadata.maps?.[posProp]
            if (driver?.type === 'mouse-position') {
                const s = node.mouseDriverState[posProp]
                if (s) {
                    cx = s.liveX ?? (typeof driver.x === 'number' ? driver.x : s.currentX)
                    cy = s.liveY ?? (typeof driver.y === 'number' ? driver.y : s.currentY)
                }
            }
        }
        return {centerX: cx, centerY: cy, halfWidth: he.hwx, halfHeight: he.hhy}
    }

    // Shape-bounds values for a wantsBoundsParams node's chosen child, or null. Mirrors the
    // composer's childBoundsParams selection: the FIRST visible direct child (render order). An
    // active-bbox child is handled by the composer forwarding its own _bbox_* fields (returns null —
    // don't drive); a shape child (computeBounds/propBindings) is CPU-resolved into the parent's
    // _childBounds_* fields. No resolvable child → null (full-canvas fallback holds).
    const childBoundsValues = (
        node: GpuNodeInfo,
    ): {centerX: number; centerY: number; halfWidth: number; halfHeight: number} | null => {
        if (!node.definition.wantsBoundsParams) return null
        const children = findChildNodes(node.id).sort((a, b) => a.metadata.renderOrder - b.metadata.renderOrder)
        for (const child of children) {
            if (child.metadata.visible === false) continue
            if (bboxActive(child)) return null // composer forwards the child's live _bbox_* fields
            const b = computeNodeBoundsUV(child)
            if (b) return b
        }
        return null
    }

    // Refresh each wantsBoundsParams node's _childBounds_* fields from its shape child each frame.
    const updateChildBounds = (): void => {
        for (const node of nodes.values()) {
            if (!node.definition.wantsBoundsParams) continue
            const b = childBoundsValues(node)
            if (!b) continue
            writeSyntheticScalar(node, '_childBounds_centerX', b.centerX)
            writeSyntheticScalar(node, '_childBounds_centerY', b.centerY)
            writeSyntheticScalar(node, '_childBounds_halfWidth', b.halfWidth)
            writeSyntheticScalar(node, '_childBounds_halfHeight', b.halfHeight)
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Group layout (flex-like column/row)
    // ────────────────────────────────────────────────────────────────────────────────────
    //
    // The walk + arithmetic live in utilities/layout.ts (pure, tested); this wiring only builds
    // the node views, applies the computed writes, and decides WHEN to re-run:
    //   - resize (after dimensional re-resolution — layout overrides win)
    //   - any prop / metadata change while a layout group exists
    //   - a media natural size landing (an image finishing its load can change a stack)
    // Writes go through the native channel: u.value + live handle for position props (the user's
    // authored value stays in _rawDimensional, so disabling layout restores it on the next
    // re-resolution), and metadata.boundingBox for box-measured children.
    let layoutDirty = false
    const markLayoutDirty = (): void => {
        layoutDirty = true
        requestRender()
    }
    const anyLayoutGroups = (): boolean => {
        for (const node of nodes.values()) {
            const m = node.metadata.flow?.mode
            if (m === 'column' || m === 'row') return true
        }
        return false
    }

    const buildLayoutView = (node: GpuNodeInfo): LayoutNodeView => {
        const snap: Record<string, unknown> = {}
        for (const [k, u] of Object.entries(node.uniforms)) {
            snap[k] = u._rawDimensional !== undefined ? u._rawDimensional : u.value
        }
        const posProp = node.boundingBoxDeclaration?.propBindings?.x?.prop
            ?? node.boundingBoxDeclaration?.propBindings?.y?.prop
        const posValue = posProp ? node.uniforms[posProp]?.value : undefined
        const positionDriven = !!(posProp && (
            (posValue as {type?: string} | undefined)?.type === 'mouse-position' ||
            node.metadata.maps?.[posProp]
        ))
        return {
            id: node.id,
            componentName: node.componentName,
            visible: node.metadata.visible !== false,
            absolute: node.metadata.absolute === true,
            requiresChild: node.requiresChild,
            layout: node.metadata.flow,
            decl: node.boundingBoxDeclaration,
            props: snap,
            boundingBox: node.metadata.boundingBox,
            positionDriven,
            children: findChildNodes(node.id)
                .sort((a, b) => a.metadata.renderOrder - b.metadata.renderOrder)
                .map(buildLayoutView),
        }
    }

    const runLayoutPass = (): void => {
        if (!anyLayoutGroups()) {
            layoutDirty = false
            return
        }
        const cw = pxResWidth()
        const ch = pxResHeight()
        // Not sized yet (first frames report 0×0 / 1×1 while the host measures the canvas):
        // KEEP the dirty flag so the pass retries next frame — clearing it here was the
        // "children stay at authored positions until a prop is touched" fresh-load bug.
        if (cw <= 1 || ch <= 1) return
        layoutDirty = false
        // Only the root node (parentId === null) starts a walk — its view already contains
        // every descendant, so ALSO pushing parentId === rootId nodes traversed and measured
        // each top-level subtree twice.
        const roots: LayoutNodeView[] = []
        for (const node of nodes.values()) {
            if (node.parentId === null) roots.push(buildLayoutView(node))
        }
        const writes = computeTreeLayout(roots, cw, ch)
        for (const w of writes) {
            const node = nodes.get(w.id)
            if (!node) continue
            if (w.kind === 'position') {
                // Absolute uv render centre, written as the resolved value (raw-space {x,y} — the
                // prop transform applies at the handle). _rawDimensional is deliberately NOT
                // touched: it keeps the user's authored position for when layout turns off.
                const u = node.uniforms[w.prop]
                if (!u) continue
                const prev = u.value as {x?: unknown; y?: unknown} | undefined
                const next = (prev && typeof prev === 'object') ? {...prev, x: w.value.x, y: w.value.y} : w.value
                u.value = next
                writePropHandle(node, w.prop, next)
            } else {
                // Box-measured child: drive its box in px (top-left). First activation of a box
                // crosses the identity boundary → structural; afterwards it's runtime-only.
                const wasActive = bboxActive(node)
                node.metadata.boundingBox = {
                    x: {value: w.xPx, unit: 'px'},
                    y: {value: w.yPx, unit: 'px'},
                    width: {value: w.widthPx, unit: 'px'},
                    height: {value: w.heightPx, unit: 'px'},
                    origin: 'top-left',
                    rotation: node.metadata.boundingBox?.rotation ?? 0,
                    ...(node.metadata.boundingBox?.cornerRadius ? {cornerRadius: node.metadata.boundingBox.cornerRadius} : {}),
                    ...(node.metadata.boundingBox?.lockAspect !== undefined ? {lockAspect: node.metadata.boundingBox.lockAspect} : {}),
                }
                if (!wasActive && bboxActive(node)) {
                    markStructuralDirty()
                } else {
                    const bb = bboxFieldValues(node)
                    if (bb) for (const [k, v] of Object.entries(bb)) writeSyntheticScalar(node, `_bbox_${k}`, v)
                }
            }
        }
    }

    const runLayoutPassIfDirty = (): void => {
        if (layoutDirty) runLayoutPass()
    }

    // A media natural size landing may resize a stacked image → re-layout.
    let unsubscribeNaturalSize: (() => void) | null = null

    // A webfont finishing its load changes Text measurement without any prop changing —
    // re-layout so stacked text snaps to the real font's metrics (same trigger the editor
    // overlay uses via its fontsTick).
    const onFontsLoadedLayout = (): void => {
        if (anyLayoutGroups()) markLayoutDirty()
    }

    // cleanup() detaches both listeners, and hosts re-enter initialize() on the same instance
    // (toneMapping/colorSpace changes) — so attachment must be re-runnable, not factory-scoped.
    // Idempotent: the subscription is guarded and addEventListener dedupes an identical handler.
    const attachLayoutListeners = (): void => {
        if (!unsubscribeNaturalSize) {
            unsubscribeNaturalSize = onNaturalSizeChange(() => {
                if (anyLayoutGroups()) markLayoutDirty()
            })
        }
        if (typeof document !== 'undefined' && document.fonts?.addEventListener) {
            document.fonts.addEventListener('loadingdone', onFontsLoadedLayout)
        }
    }
    attachLayoutListeners()

    const transformActive = (node: GpuNodeInfo): boolean => needsTransformation(node.metadata.transform)

    /** Ensure resolved map/mouse/auto output bounds exist for a driven prop. */
    const ensureMapValues = (node: GpuNodeInfo, prop: string, driver: PropDriver): MapValues => {
        const mv: MapValues = node.mapValues[prop] ?? {inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 1, curve: 0}
        if (driver.type === 'map') {
            const c = driver as MapConfig
            mv.inputMin = c.inputMin
            mv.inputMax = c.inputMax
            mv.outputMin = resolveDynamicBound(node, prop, c.outputMin)
            mv.outputMax = resolveDynamicBound(node, prop, c.outputMax)
            mv.curve = c.curve ?? 0
        } else if (driver.type === 'mouse') {
            const c = driver as MouseMapConfig
            mv.outputMin = resolveDynamicBound(node, prop, c.outputMin)
            mv.outputMax = resolveDynamicBound(node, prop, c.outputMax)
            mv.curve = c.curve ?? 0
        } else if (driver.type === 'auto-animate') {
            const c = driver as AutoAnimateConfig
            mv.outputMin = resolveDynamicBound(node, prop, c.outputMin)
            mv.outputMax = resolveDynamicBound(node, prop, c.outputMax)
        }
        node.mapValues[prop] = mv
        return mv
    }

    const ensureMouseState = (node: GpuNodeInfo, prop: string, driver: MousePositionConfig | MouseMapConfig): MouseDriverState => {
        let s = node.mouseDriverState[prop]
        if (s) return s
        if (driver.type === 'mouse-position') {
            const trackX = driver.x === undefined || driver.x === 'mouse'
            const trackY = driver.y === undefined || driver.y === 'mouse'
            s = {
                currentX: trackX ? pointerX : (driver.x as number),
                currentY: trackY ? pointerY : (driver.y as number),
                velocityX: 0,
                velocityY: 0,
                pointerSnapped: pointerSeen,
            }
        } else {
            const initial = Math.max(0.001, driver.axis === 'x' ? pointerX : pointerY)
            s = {currentX: initial, currentY: 0, velocityX: 0, velocityY: 0, pointerSnapped: pointerSeen}
        }
        node.mouseDriverState[prop] = s
        return s
    }

    /**
     * (Re)populate a node's persistent CPU driver state (mouseDriverState / autoAnimateState /
     * mapValues) from its metadata.maps. `buildFieldInits` does this on every cache-MISS build,
     * but a re-registered node — the design editor remounts every component on a structural edit,
     * so each node is a brand-new object with empty driver state — can bind straight onto a CACHED
     * composition (undo/redo, removing a just-added layer), skipping buildFieldInits entirely.
     * Without this, `updateMouseDrivers` has nothing to iterate and every mouse / mouse-position /
     * auto-animate driver silently freezes until the next never-seen-hash build.
     */
    const ensureDriverState = (node: GpuNodeInfo): void => {
        const maps = node.metadata.maps
        if (!maps) return
        for (const [prop, driver] of Object.entries(maps)) {
            if (!(prop in node.uniforms)) continue
            if (driver.type === 'map') {
                ensureMapValues(node, prop, driver)
            } else if (driver.type === 'mouse-position') {
                ensureMouseState(node, prop, driver)
            } else if (driver.type === 'mouse') {
                ensureMouseState(node, prop, driver)
                ensureMapValues(node, prop, driver)
            } else if (driver.type === 'auto-animate') {
                if (!node.autoAnimateState[prop]) node.autoAnimateState[prop] = {}
                ensureMapValues(node, prop, driver)
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Field-init assembly for a node (props + synthetic `_`-fields)
    // ────────────────────────────────────────────────────────────────────────────────────

    /**
     * Stable WeakMap key for a node's animated-time accumulator — the node's speed
     * GpuUniformInput (stable per registration, survives recompose so the animation continues).
     * Falls back to the node itself if the declared speed prop is absent. Both `buildFieldInits`
     * (seed) and `updateAnimatedTime` (drive) use this identical key so they share one accumulator.
     */
    const animatedTimeKey = (node: GpuNodeInfo): object => {
        const at = node.definition.animatedTime
        const speedU = at ? node.uniforms[at.speed] : undefined
        return (speedU as object | undefined) ?? node
    }

    // Stable accumulator key for an EXTRA animated-time clock — the clock's own speed
    // GpuUniformInput (distinct from the primary's, so the clocks accumulate independently).
    const extraAnimatedTimeKey = (node: GpuNodeInfo, speedProp: string): object =>
        (node.uniforms[speedProp] as object | undefined) ?? node

    const buildFieldInits = (node: GpuNodeInfo): FieldInit[] => {
        const inits: FieldInit[] = []

        // Props (the shader's own uniforms).
        for (const [propName, u] of Object.entries(node.uniforms)) {
            inits.push({name: propName, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema})
        }

        // _opacity — always.
        inits.push({name: '_opacity', schema: d.f32, initial: node.metadata.opacity ?? 1})

        // _animTime — per-node accumulated time, when declared.
        // Seed from the LIVE accumulator (not 0) so a recompose resumes at the accumulated value
        // instead of flashing back to the start; the frame driver advances it from here.
        if (node.definition.animatedTime) {
            inits.push({name: '_animTime', schema: d.f32, initial: getAnimatedTimeState(animatedTimeKey(node)).value})
        }

        // Extra animated-time clocks (FlowField's evolution speed) — one field/accumulator each.
        if (node.definition.extraAnimatedTimes) {
            for (const [key, speedProp] of Object.entries(node.definition.extraAnimatedTimes)) {
                inits.push({name: `_animTime_${key}`, schema: d.f32, initial: getAnimatedTimeState(extraAnimatedTimeKey(node, speedProp)).value})
            }
        }

        // extraFields — per-node derived uniforms (Blob's normalized light direction) the
        // shader writes each frame via setExtraField. Registered on the node struct like _animTime,
        // seeded from the declared initial. Not a prop → never in the structural hash, so a per-frame
        // change patches in place without recomposing. A recompose re-seeds the initial; the shader's
        // onBeforeRender runs before the first flush and overwrites it, so no stale-value flash.
        const extraFields = node.definition.extraFields
        if (extraFields) {
            for (const [name, f] of Object.entries(extraFields)) inits.push({name, schema: f.schema, initial: f.initial})
        }

        // _xform_* — when a layer transform is active (matches the composer + structural hash).
        if (transformActive(node)) {
            const tf = transformFieldValues(node)
            for (const [k, v] of Object.entries(tf)) inits.push({name: `_xform_${k}`, schema: d.f32, initial: v})
        }

        // _bbox_* — when a bounding box is active.
        if (bboxActive(node)) {
            const bb = bboxFieldValues(node)
            if (bb) for (const [k, v] of Object.entries(bb)) inits.push({name: `_bbox_${k}`, schema: d.f32, initial: v})
        }

        // _childBounds_* — CPU-resolved shape bounds of a wantsBoundsParams node's (Repeater's)
        // first visible shape child, driven each frame by updateChildBounds. Allocated for every
        // wantsBoundsParams node; the composer reads them only for the shape-child path (an
        // active-bbox child forwards its own _bbox_* fields instead). Full-canvas fallback initial
        // values until the driver fills them from the child's live props.
        if (node.definition.wantsBoundsParams) {
            const cb = childBoundsValues(node) ?? {centerX: 0.5, centerY: 0.5, halfWidth: 0.5, halfHeight: 0.5}
            for (const [k, v] of Object.entries(cb)) inits.push({name: `_childBounds_${k}`, schema: d.f32, initial: v})
        }

        // Driver fields — ensureDriverState populates the persistent CPU state these read.
        const maps = node.metadata.maps
        if (maps) {
            ensureDriverState(node)
            for (const [prop, driver] of Object.entries(maps)) {
                if (!(prop in node.uniforms)) continue
                if (driver.type === 'map') {
                    const mv = node.mapValues[prop]
                    inits.push({name: `_map_${prop}_inputMin`, schema: d.f32, initial: mv.inputMin})
                    inits.push({name: `_map_${prop}_inputMax`, schema: d.f32, initial: mv.inputMax})
                    inits.push({name: `_map_${prop}_outputMin`, schema: d.f32, initial: mv.outputMin})
                    inits.push({name: `_map_${prop}_outputMax`, schema: d.f32, initial: mv.outputMax})
                    inits.push({name: `_map_${prop}_curve`, schema: d.f32, initial: mv.curve})
                } else if (driver.type === 'mouse-position') {
                    const s = node.mouseDriverState[prop]
                    inits.push({name: `_smoothed_${prop}`, schema: d.vec2f, initial: [s.currentX, 1 - s.currentY]})
                } else if (driver.type === 'mouse') {
                    const mv = node.mapValues[prop]
                    inits.push({name: `_smoothed_${prop}`, schema: d.f32, initial: mv.outputMin})
                } else if (driver.type === 'auto-animate') {
                    const mv = node.mapValues[prop]
                    inits.push({name: `_smoothed_${prop}`, schema: d.f32, initial: mv.outputMin})
                }
            }
        }
        return inits
    }

    // Write a prop handle in place (applies its transform), if the composition is bound.
    const writePropHandle = (node: GpuNodeInfo, propName: string, value: unknown): void => {
        const h = node._liveHandles?.[propName]
        if (!h) return
        setColorSpaceModeFn?.(rendererColorSpace)
        updateFieldValue(h, value)
    }

    const writeSyntheticScalar = (node: GpuNodeInfo, field: string, value: number): void => {
        const h = node._liveHandles?.[field]
        if (h instanceof FieldHandle) h.value = value
    }

    // Write an extraFields field's live handle (scalar OR vec components — FieldHandle.value
    // accepts a number or a number[]; an `arrayOf(...)` field is an ArrayFieldHandle and takes the
    // whole flat array). Patches the packed buffer for the next flush. No-op when unbound or the
    // name isn't a registered field. Backs GpuFragmentParams.setExtraField.
    const writeExtraFieldValue = (node: GpuNodeInfo, field: string, value: number | number[]): void => {
        const h = node._liveHandles?.[field]
        if (h instanceof FieldHandle) h.value = value
        else if (h instanceof ArrayFieldHandle && Array.isArray(value)) h.array = value
    }

    // A colorStops prop expands (in the bridge) into a cpu `stops` mirror + four packed struct
    // fields (colorsArray/positionsArray/convertedColorsArray/stopCount). Keep those arrays in
    // sync: re-pack all four from the node's CURRENT stops at its CURRENT colorSpace mode,
    // updating BOTH the `.value` mirror (so a recompose re-seeds fresh) and the live handle (so a
    // same-count edit patches in place). Fixed field names — a shader has at most one colorStops
    // prop (the bridge invariant). No-ops for a node without colorStops.

    /** The node's active colorSpace mode (the mode convertedColorsArray must be packed at). */
    const nodeColorSpaceMode = (node: GpuNodeInfo): number => {
        for (const u of Object.values(node.uniforms)) {
            if (u.transform === (transformColorSpace as unknown)) {
                const raw = u.value
                if (typeof raw === 'string') return transformColorSpace(raw)
                if (typeof raw === 'number') return raw
            }
        }
        return 0
    }

    const repackColorStops = (node: GpuNodeInfo): void => {
        if (!node.uniforms.convertedColorsArray) return // not a colorStops shader
        let stops: ColorStop[] | null = null
        for (const u of Object.values(node.uniforms)) {
            if (u.transform === (colorStopsTransform as unknown)) {
                stops = (u.value ?? null) as ColorStop[] | null
                break
            }
        }
        const packed = packStops(stops, colorToRGBA)
        const converted = packConvertedStops(packed.colors, packed.stopCount, nodeColorSpaceMode(node))
        const write = (name: string, value: unknown): void => {
            const u = node.uniforms[name]
            if (u) u.value = value
            writePropHandle(node, name, value)
        }
        write('colorsArray', packed.colors)
        write('positionsArray', packed.positions)
        write('convertedColorsArray', converted)
        write('stopCount', packed.stopCount)
    }

    // A list prop (the generic array-prop mechanism, `utilities/listProps`) keeps its RAW items on
    // `_rawList`; the mirror handle holds the RESOLVED items (what `getCpuValue` returns) and the
    // per-field lane arrays + count hold the packed uniforms. Re-pack all of them from the raw
    // items. Position fields may carry a mouse-position driver: its state is keyed by the field
    // PATH (`lights.0.position`) and substituted live; states for elements no longer driven are
    // dropped so the pointer stops steering a light that was pinned or removed.
    const repackListProp = (node: GpuNodeInfo, propName: string): void => {
        const u = node.uniforms[propName]
        const spec = u?._listSpec
        if (!u || !spec) return
        const raw = u._rawList
        const driven = new Set<string>()
        const items = resolveListItems(raw, spec, {
            color: colorToRGBA,
            drivenPosition: (index, field, driver) => {
                const key = listDriverPath(propName, index, field)
                driven.add(key)
                const st = ensureMouseState(node, key, driver)
                return st.liveX !== undefined && st.liveY !== undefined ? {x: st.liveX, y: 1 - st.liveY} : undefined
            },
        })
        for (const key of Object.keys(node.mouseDriverState)) {
            const path = parseListDriverPath(key)
            if (path && path.prop === propName && !driven.has(key)) delete node.mouseDriverState[key]
        }
        u.value = items
        writePropHandle(node, propName, items)
        const packed = packList(items, spec)
        const write = (name: string, value: unknown): void => {
            const f = node.uniforms[name]
            if (f) f.value = value
            writePropHandle(node, name, value)
        }
        for (const [field, lanes] of Object.entries(packed.fields)) write(listFieldName(propName, field), lanes)
        write(listCountName(propName), packed.count)
    }

    /** The driver behind a `mouseDriverState` key: a top-level prop's from metadata.maps, or a list element's from its raw items. */
    const driverForKey = (node: GpuNodeInfo, key: string): PropDriver | undefined => {
        const top = node.metadata.maps?.[key] as PropDriver | undefined
        if (top) return top
        const path = parseListDriverPath(key)
        if (!path) return undefined
        const raw = node.uniforms[path.prop]?._rawList
        const item = Array.isArray(raw) ? (raw[path.index] as Record<string, unknown> | undefined) : undefined
        const value = item?.[path.field]
        return isPositionDriver(value) ? (value as PropDriver) : undefined
    }

    /** Does any list prop on the node carry a driven position (needs pointer listeners + per-frame re-pack)? */
    const listsUseMouse = (node: GpuNodeInfo): boolean =>
        Object.values(node.uniforms).some((u) => u._listSpec && listHasDrivers(u._rawList, u._listSpec))

    /** Re-drive a node's synthetic fields from current metadata/state (used on bind + resize). */
    const writeNodeSynthetic = (node: GpuNodeInfo): void => {
        writeSyntheticScalar(node, '_opacity', node.metadata.opacity ?? 1)
        // A reused buffer (cache-hit rebind) holds the hash's LAST _animTime; refresh from the
        // live accumulator so a swap-back resumes at the current animated value.
        if (node.definition.animatedTime) {
            writeSyntheticScalar(node, '_animTime', getAnimatedTimeState(animatedTimeKey(node)).value)
        }
        // Refresh each extra animated-time clock from its live accumulator (cache-hit rebind).
        if (node.definition.extraAnimatedTimes) {
            for (const [key, speedProp] of Object.entries(node.definition.extraAnimatedTimes)) {
                writeSyntheticScalar(node, `_animTime_${key}`, getAnimatedTimeState(extraAnimatedTimeKey(node, speedProp)).value)
            }
        }
        if (transformActive(node)) {
            for (const [k, v] of Object.entries(transformFieldValues(node))) writeSyntheticScalar(node, `_xform_${k}`, v)
        }
        if (bboxActive(node)) {
            const bb = bboxFieldValues(node)
            if (bb) for (const [k, v] of Object.entries(bb)) writeSyntheticScalar(node, `_bbox_${k}`, v)
        }
        const maps = node.metadata.maps
        if (maps) {
            for (const [prop, driver] of Object.entries(maps)) {
                if (driver.type === 'map' && node.mapValues[prop]) {
                    const mv = node.mapValues[prop]
                    writeSyntheticScalar(node, `_map_${prop}_inputMin`, mv.inputMin)
                    writeSyntheticScalar(node, `_map_${prop}_inputMax`, mv.inputMax)
                    writeSyntheticScalar(node, `_map_${prop}_outputMin`, mv.outputMin)
                    writeSyntheticScalar(node, `_map_${prop}_outputMax`, mv.outputMax)
                    writeSyntheticScalar(node, `_map_${prop}_curve`, mv.curve)
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // RegistryView adapters (hash-only + full-composition)
    // ────────────────────────────────────────────────────────────────────────────────────

    const propHandleView = (node: GpuNodeInfo, handles: NodeHandles): Record<string, UniformHandleView> => {
        const out: Record<string, UniformHandleView> = {}
        for (const propName of Object.keys(node.uniforms)) {
            const h = handles[propName]
            if (h) out[propName] = h as unknown as UniformHandleView
        }
        return out
    }

    // Source-of-truth handle view for hashing (no store needed).
    const sotHandleView = (node: GpuNodeInfo): Record<string, UniformHandleView> => {
        const out: Record<string, UniformHandleView> = {}
        for (const [propName, u] of Object.entries(node.uniforms)) {
            out[propName] = {accessorPath: '', cpu: !!u.cpu, value: u.value}
        }
        return out
    }

    const makeRegistryNode = (node: GpuNodeInfo, handles: Record<string, UniformHandleView>): RegistryNode => ({
        id: node.id,
        customId: node.metadata.id,
        componentName: node.componentName,
        parentId: node.parentId,
        definition: node.definition,
        metadata: node.metadata,
        handles,
        domCanvas: node.domCanvas,
    })

    const buildRegistryView = (
        store: UniformStore | null,
        handlesById: Map<string, NodeHandles> | null,
    ): RegistryView => {
        const cache = new Map<string, RegistryNode>()
        const build = (node: GpuNodeInfo): RegistryNode => {
            const handles = store && handlesById ? propHandleView(node, handlesById.get(node.id) ?? {}) : sotHandleView(node)
            return makeRegistryNode(node, handles)
        }
        const getNode = (id: string): RegistryNode | undefined => {
            const hit = cache.get(id)
            if (hit) return hit
            const node = nodes.get(id)
            if (!node) return undefined
            const rn = build(node)
            cache.set(id, rn)
            return rn
        }
        return {
            rootId,
            getNode,
            getChildren: (pid) => findChildNodes(pid).map((n) => getNode(n.id)!),
            resolveCustomId: (cid) => findNodeByCustomId(cid),
            store: {
                gpuAccessor: (h, layoutVar) => (store ? store.gpuAccessor(h as never, layoutVar) : ''),
                layout: store?.layout,
            },
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Structural hash (base inputs + driver/analytic-opacity extension)
    // ────────────────────────────────────────────────────────────────────────────────────

    const composeOptions = (): ComposeOptions => ({
        toneMapping: rendererToneMapping as ComposeOptions['toneMapping'],
        flipY: false,
        // The canvas is configured alphaMode:'premultiplied' (root.ts), so the final pass MUST
        // emit premultiplied RGB (rgb*a) — root.ts's own comment states this. With this off, a
        // half-transparent shader composited over a page/background renders wrong (opaque content
        // is unaffected: a=1 → rgb*1).
        premultiplyAlpha: true,
        canvas: canvasElement ?? undefined,
        dimensions: bufferSize(),
        gpu: root ? {device: root.device, root} : undefined,
        // Back GpuFragmentParams.setExtraField — the shader's onBeforeRender writes a derived
        // uniform (Blob light-dir) on the node's live handle. Harmless during hash-only calls.
        writeExtraField: (nodeId, name, value) => {
            const node = nodes.get(nodeId)
            if (node) writeExtraFieldValue(node, name, value)
        },
        // Back GpuFragmentParams.createMediaTexture — a media shader (ImageTexture/Text/
        // HTMLInCanvas) creates its upload target through the renderer's TextureManager (so it's
        // counted + destroyed with the renderer). Read `textureManager` at call time (set in initialize).
        createMediaTexture: (opts) => {
            if (!textureManager) throw new Error('[gpu] createMediaTexture before init')
            return textureManager.createMediaTexture(opts) as never
        },
        // Back GpuFragmentParams.createDataTexture — raw-float (r16/r32) SVG-SDF field
        // textures, through the same TextureManager (counted + destroyed with the renderer).
        createDataTexture: (opts) => {
            if (!textureManager) throw new Error('[gpu] createDataTexture before init')
            return textureManager.createDataTexture(opts) as never
        },
        // Back GpuFragmentParams.getMapInfo(prop).window() — the LIVE resolved remap window
        // of a map-driven prop (post dynamic-bound resolution), the same values writeNodeSynthetic
        // pushes into the `_map_<prop>_*` uniforms the fragment path reads. A compute hook (Blur's
        // variable-blur fill kernel) reads these each frame so its per-pixel radius matches the
        // fragment remap. Returns undefined for a non-map / unknown prop → the composer falls back
        // to the raw driver-config numbers.
        resolveMapWindow: (nodeId, prop) => {
            const node = nodes.get(nodeId)
            const driver = node?.metadata.maps?.[prop]
            if (!node || !driver || driver.type !== 'map') return undefined
            const mv = ensureMapValues(node, prop, driver)
            return {inputMin: mv.inputMin, inputMax: mv.inputMax, outputMin: mv.outputMin, outputMax: mv.outputMax, curve: mv.curve}
        },
        // Back GpuFragmentParams.getCpuValue for a DRIVEN prop read by a compute hook — the driver
        // resolves CPU-side each frame (updateMouseDrivers / updateAutoAnimate) but writes only the
        // `_smoothed_<prop>` synthetic field, which a maps-routed prop's static handle never reflects.
        // Return the same resolved value the fragment sees so the compute sim (Smoke/SmokeFill emitter,
        // ProgressiveBlur/TiltShift centre) tracks the pointer instead of freezing at the default. A
        // mouse-position vec2 → post-flip {x, 1 - y} (transformPosition convention, == the field);
        // mouse/auto scalar → the `_smoothed_<prop>` handle. `map` drivers return undefined (compute
        // samples the source per-pixel via getMapInfo, not a single CPU value).
        resolveDriverCpuValue: (nodeId, prop) => {
            const node = nodes.get(nodeId)
            if (!node) return undefined
            // Synthetic animated-time fields (`_animTime`, `_animTime_<key>`): the accumulator the
            // GPU reads THIS frame (updateAnimatedTime runs before onBeforeRender), so a shader's
            // per-frame CPU work (scatteredAnchors) shares the fragment's clock exactly.
            if (prop === '_animTime' || prop.startsWith('_animTime_')) {
                const h = node._liveHandles?.[prop]
                return h instanceof FieldHandle ? h.value : undefined
            }
            const driver = node.metadata.maps?.[prop]
            if (!driver) return undefined
            if (driver.type === 'mouse-position') {
                const s = node.mouseDriverState[prop]
                if (!s || s.liveX === undefined || s.liveY === undefined) return undefined
                return {x: s.liveX, y: 1 - s.liveY}
            }
            if (driver.type === 'mouse' || driver.type === 'auto-animate') {
                const h = node._liveHandles?.[`_smoothed_${prop}`]
                return h instanceof FieldHandle ? h.value : undefined
            }
            return undefined
        },
    })

    /**
     * Extra hash inputs the base `collectStructuralHashInputs` does not distinguish but which
     * change the emitted WGSL / field set, so same-hash ⟹ same-struct ⟹ same-IR holds
     * (required by the store-keyed-by-hash cache):
     *   - analytic-opacity: a uvRemap / acceptsUVContext node crossing opacity<1 flips analytic
     *     eligibility (base hash only buckets opacity as zero/nonzero).
     *   - driver fingerprint: the STRUCTURAL bits of each driver (type/source/channel/axis) —
     *     the same bits updateNodeMetadata treats as a structural change.
     *   - colorStops count: a gradient's `stopCount > 1` two-color↔multi-stop selection is a
     *     COMPILE-TIME branch in the fragment (the JS `if (stopCount > 1)`), and the `stops` prop
     *     uses compileTimeWhen to request a recompose when that effective count changes — but the
     *     base hash omits the count, so getOrBuild would hand back the stale composition (the
     *     gradient stuck on whichever path it first built: two-color when materialised from null,
     *     multi-stop otherwise). Fingerprint the SAME effective count colorStopsPropConfig's
     *     compileTimeWhen keys on (`n > 1 ? n : 0`) so a count change actually rebuilds. Same-count
     *     recolors keep an identical fingerprint → still an in-place array patch, no rebuild.
     *   - compileTimeWhen bucket: a prop whose `compileTimeWhen(prev,next)` predicate gates a
     *     COMPILE-TIME branch — the frosted-blur / aberration pass PRESENCE (Glass, FlutedGlass),
     *     the sharpen/glow/hueShift bypass at 0, Saturation's identity at 1, GridDistortion's
     *     clamped cell count, Surface3D's lighting/cursor thresholds. The predicate fires (and
     *     `markStructuralDirty`) on a boundary crossing, but the base composer hash can't see the
     *     value (it is renderer-only state), so getOrBuild would serve the stale pipeline — the
     *     exact colorStops/transform-edges class. `_lastCompiledValue` is the value each recompose
     *     was keyed to (seeded at register, updated iff the predicate fires), i.e. a bucket-stable
     *     representative of the compile-relevant projection: a boundary crossing changes it
     *     (rebuild), a same-bucket edit leaves it untouched (fast in-place path). colorStops' `stops`
     *     prop is also compileTimeWhen, so this covers it too (redundant with the count above, and
     *     equally count-keyed since `_lastCompiledValue` only moves on a count change).
     */
    const extraHashInputs = (): string[] => {
        const parts: string[] = []
        for (const [id, node] of nodes) {
            const m = node.metadata
            // opacity<1 is a COMPILE-TIME branch for every node, not just the analytic ones: the
            // sibling loop skips the blend entirely for a first child at opacity 1 (`needsRTT`), and
            // analytic eligibility flips on the same boundary. The base hash only buckets opacity as
            // zero/nonzero, so without this a layer whose opacity leaves 1 would keep the
            // composition that never reads `_opacity` — the slider would do nothing. Same-bucket
            // edits (0.9 → 0.5) keep an identical fingerprint and stay a pure uniform patch.
            parts.push(`aopac:${id}:${(m.opacity ?? 1) < 1}`)
            // `stopCount` is the synthetic field expandColorStops adds only for a colorStops shader,
            // so its presence marks one; its value is the exact compile-time branch input.
            const sc = node.uniforms.stopCount?.value
            if (typeof sc === 'number') parts.push(`stops:${id}:${sc > 1 ? sc : 0}`)
            // compileTimeWhen props: fold in the value the current compile bucket was keyed to.
            for (const [prop, u] of Object.entries(node.uniforms)) {
                if (!u.compileTimeWhen) continue
                const v = u._lastCompiledValue
                // Only NUMBER/BOOLEAN threshold predicates fold their crossing value in directly
                // (the blur / aberration / sharpness / saturation / gridSize / lighting numerics).
                // Everything else keys its compile branch on a CANONICAL projection fingerprinted
                // separately — never the raw value — because the raw value can churn without a
                // bucket change:
                //   - Object-valued predicates (colorStops `stops`) → effective count, above; a
                //     same-count recolor and a 1-stop-vs-null pair MUST hash identically.
                //   - String-valued predicates (a color/enum/url prop that gained a compileTimeWhen)
                //     → the raw string. An animated `rgba(…)`/`hsl(…)` value mutates frame-by-frame,
                //     so folding it verbatim changes the pipeline-cache key every frame → swap-when-
                //     ready churn → blank canvas. A string predicate must fingerprint a canonical
                //     projection here (like `stops` does) if one is ever added.
                if (typeof v !== 'number' && typeof v !== 'boolean') continue
                parts.push(`ctw:${id}:${prop}:${String(v)}`)
            }
            if (m.maps) {
                const fp = Object.entries(m.maps)
                    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
                    .map(([p, dr]) => {
                        const cfg = dr as {type: string; source?: string; channel?: string; axis?: string}
                        return `${p}=${cfg.type}:${cfg.source ?? ''}:${cfg.channel ?? ''}:${cfg.axis ?? ''}`
                    })
                    .join(',')
                parts.push(`mapfp:${id}:${fp}`)
            }
        }
        return parts
    }

    const computeStructuralHash = (): string | null => {
        if (!rootId) return null
        const reg = buildRegistryView(null, null)
        const base = collectStructuralHashInputs(reg, composeOptions())
        return structuralHash([...base, ...extraHashInputs()])
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Composition build / bind / dispose
    // ────────────────────────────────────────────────────────────────────────────────────

    const bufferSize = (): {width: number; height: number} => ({
        width: Math.max(1, Math.round(currentWidth * pixelRatio)),
        height: Math.max(1, Math.round(currentHeight * pixelRatio)),
    })

    const mediaResourcesFromIr = (ir: CompositionIR): Record<string, RenderTexture | {texture: unknown}> => {
        const out: Record<string, RenderTexture | {texture: unknown}> = {}
        for (const t of ir.textures) {
            if (t.kind !== 'rtt' && t.resource) out[t.key] = t.resource as {texture: unknown}
        }
        return out
    }

    const buildComposition = (hash: string): BuiltComposition => {
        if (!root || !textureManager || !dispatcher) throw new Error('[gpu] buildComposition before init')
        const store = createUniformStore(root, {systemSchema: SystemUniforms as never})
        const handlesById = new Map<string, NodeHandles>()
        for (const [id, node] of nodes) {
            handlesById.set(id, store.defineNode(id, buildFieldInits(node)))
        }
        store.defineSystem()
        store.finalize()

        const registry = buildRegistryView(store, handlesById)
        const ir = composeNodeTree(registry, composeOptions())
        const pm = createPassManager(root, {textureManager, dispatcher})
        // A composition carrying user-authored WGSL builds its pipelines inside a validation
        // scope: a body that fails to compile marks THIS composition broken (nothing draws until
        // the structure changes) instead of counting toward the fatal uncaptured-error limit.
        // The first frames render inside a scope too (see `render`), since a pipeline may only
        // be realised on first use.
        const bc: BuiltComposition = {hash, store, ir, passManager: pm, handlesById, fresh: true, validationFrames: 3}
        const device = ir.usesCustomWgsl ? root.device : undefined
        if (device) device.pushErrorScope('validation')
        try {
            pm.setComposition(ir, store.bindGroup, bufferSize(), mediaResourcesFromIr(ir))
        } finally {
            // Pop in `finally`: a synchronous throw above must not leave the scope open (it would
            // swallow every later validation error on the device and unbalance the scope stack).
            if (device) {
                device.popErrorScope().then((error) => {
                    if (error && !bc.broken) markBroken(bc, error.message)
                }).catch(() => {
                    /* device lost mid-build — the loss handler owns recovery */
                })
            }
        }
        liveCompositions.add(bc)
        return bc
    }

    /** Record a validation failure against one composition and tell the author once. */
    const markBroken = (bc: BuiltComposition, message: string): void => {
        bc.broken = message
        authorError(
            '[gpu] a shader in this composition failed GPU validation — it will not draw until it changes. ' +
            'If you are writing a wgsl`…` body, the message below points at the line:\n' + message,
        )
    }

    const disposeComposition = (bc: BuiltComposition): void => {
        // Run shader onCleanup before tearing down the pass manager. Some
        // compute shaders (Blur) allocate their own storage textures (intermediate/output buffers)
        // outside the pass manager's RTT set; their onCleanup destroys them. Without draining these
        // here they leak on every recompose. (Twirl et al. register none — safe no-op.)
        try {
            for (const cb of bc.ir.onCleanup) cb()
        } catch (e) {
            debugError('[gpu] composition onCleanup error:', e)
        }
        try {
            bc.passManager.dispose()
        } catch (e) {
            debugError('[gpu] passManager dispose error:', e)
        }
        try {
            bc.store.destroy()
        } catch (e) {
            debugError('[gpu] store destroy error:', e)
        }
        liveCompositions.delete(bc)
        if (boundComposition === bc) boundComposition = null
    }

    /** Point every node's live handles at `bc`'s store; re-seed the buffer on a cache hit. */
    const bindComposition = (bc: BuiltComposition): void => {
        // Re-point every current node at this composition's store handles, tracking whether any
        // node still needed pointing. A re-registered node — undo/redo bumps `structureVersion`,
        // so PresetRenderer remounts and each shader node is removeNode()'d then registerNode()'d
        // again with the SAME id (and, for a value-only change, the SAME structural hash) — is a
        // brand-new node object carrying `_liveHandles: null`, so it is NOT bound to the (unchanged)
        // current composition and the reused buffer still holds the pre-remount values. Detecting
        // that here means binding the same composition still refreshes the new node.
        let repointed = false
        for (const [id, handles] of bc.handlesById) {
            const node = nodes.get(id)
            if (node && node._liveHandles !== handles) {
                node._liveHandles = handles
                repointed = true
            }
        }
        // Steady state — same composition, every node already bound — nothing to refresh.
        if (bc === boundComposition && !repointed) return
        if (!bc.fresh) {
            // Cache hit (swap-back to another built composition) OR a re-registration onto the
            // already-bound composition (undo/redo remount): the reused buffer holds this hash's
            // LAST values; re-seed from the live source of truth so the render shows current state.
            // Re-registered nodes also carry EMPTY driver state (buildFieldInits never ran for
            // them on this path) — ensure it before writeNodeSynthetic so map windows refresh and
            // updateMouseDrivers has state to iterate (else dynamic props freeze until the next
            // cache-miss build).
            for (const node of nodes.values()) {
                for (const [propName, u] of Object.entries(node.uniforms)) writePropHandle(node, propName, u.value)
                ensureDriverState(node)
                writeNodeSynthetic(node)
            }
            bc.store.writeAll()
        }
        bc.fresh = false
        boundComposition = bc
        // onReady fires once, after the first composed frame renders.
        if (!hasEmittedReady && bc.ir.composedNodeIds.size > 0) firstRenderPending = true
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Recompose scheduling
    // ────────────────────────────────────────────────────────────────────────────────────

    const markStructuralDirty = (): void => {
        structuralDirty = true
        // Any structural rebuild re-seeds uniforms from node values — re-run layout right after
        // so laid-out positions survive registration order / remount races (the pass patches the
        // freshly bound handles on the next frame's updateDrivers).
        if (anyLayoutGroups()) layoutDirty = true
        requestRender()
    }

    /** Ensure a frame runs soon. No-op while the animation loop is running (it will render). */
    const requestRender = (): void => {
        if (failureReason) return
        if (frameLoop.running) return
        if (pendingRenderRAF !== null) return
        if (typeof requestAnimationFrame !== 'function') return
        pendingRenderRAF = requestAnimationFrame(() => {
            pendingRenderRAF = null
            renderFrame()
        })
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Registration / update / removal
    // ────────────────────────────────────────────────────────────────────────────────────

    const registerNode = (
        id: string,
        fragmentNodeFunc: GpuShaderDefinition['fragment'] | null,
        parentId: string | null,
        metadata: NodeMetadata | null,
        uniforms: GpuUniformsMap = {},
        componentDefinition?: GpuShaderDefinition,
        domCanvas?: HTMLCanvasElement,
    ): void => {
        // A permanently-failed renderer accepts nothing: hosts keep registering children
        // (mount, reset signals, prop changes) long after we've given up, and queuing them
        // would just hold definitions and uniform objects alive for the life of the page.
        if (failureReason) return

        // Removal (component unmount) — a null fragment is the removal signal.
        if (fragmentNodeFunc === null) {
            const queueIndex = pendingRegistrationQueue.findIndex((item) => item.id === id)
            if (queueIndex !== -1) {
                pendingRegistrationQueue.splice(queueIndex, 1)
                pendingNodeUpdates = pendingNodeUpdates.filter((u) => u.id !== id)
                return
            }
            removeNode(id)
            return
        }
        if (!componentDefinition) {
            console.error(`[gpu] registerNode("${id}") requires a componentDefinition`)
            return
        }

        // Queue if the renderer is not ready (registration race).
        if (!isRendererReady) {
            const existing = pendingRegistrationQueue.findIndex((item) => item.id === id)
            const entry: QueuedRegistration = {id, definition: componentDefinition, parentId, metadata, uniforms, domCanvas}
            if (existing !== -1) {
                pendingRegistrationQueue[existing] = entry
                // The replacement snapshot is newer than anything parked for this id —
                // replaying older updates over it would resurrect stale values.
                pendingNodeUpdates = pendingNodeUpdates.filter((u) => u.id !== id)
            } else {
                pendingRegistrationQueue.push(entry)
            }
            return
        }

        // Default render order = max sibling order + 1.
        let defaultRenderOrder = 0
        if (parentId !== null) {
            const siblings = parentToChildren.get(parentId)
            if (siblings && siblings.size > 0) {
                let maxOrder = 0
                for (const siblingId of siblings) {
                    const sibling = nodes.get(siblingId)
                    if (sibling && sibling.metadata.renderOrder > maxOrder) maxOrder = sibling.metadata.renderOrder
                }
                defaultRenderOrder = maxOrder + 1
            }
        }

        const nodeMetadata: NodeMetadata = {
            blendMode: metadata?.blendMode || 'normal',
            opacity: metadata?.opacity as number | undefined,
            visible: metadata?.visible === false ? false : true,
            id: metadata?.id ? sanitizeId(metadata.id) : metadata?.id,
            mask: metadata?.mask,
            maps: metadata?.maps,
            renderOrder: metadata?.renderOrder !== undefined ? metadata.renderOrder : defaultRenderOrder,
            transform: metadata?.transform,
            boundingBox: metadata?.boundingBox,
            // Layout fields MUST land at registration, not only via later metadata updates —
            // component watches don't fire on mount, so after a remount (visibility toggle,
            // undo, structureVersion bump) the renderer would otherwise see a flow group whose
            // children's `absolute` flags arrive in arbitrary reactive order. Losing that race
            // laid out an escaped child as in-flow and poisoned its position uniform.
            flow: metadata?.flow,
            absolute: metadata?.absolute,
        }

        // Resolve dimensional (px / origin) props to UV up front, stashing the raw value.
        const decl = componentDefinition.boundingBoxDeclaration
        const extraDimProps = componentDefinition.props ? buildExtraSizeProps(componentDefinition.props) : undefined
        const node: GpuNodeInfo = {
            id,
            componentName: componentDefinition.name || 'Unknown',
            parentId,
            definition: componentDefinition,
            metadata: nodeMetadata,
            uniforms,
            domCanvas,
            requiresRTT: !!componentDefinition.requiresRTT,
            requiresChild: !!componentDefinition.requiresChild,
            blendWithChildren: !!componentDefinition.blendWithChildren,
            acceptsUVContext: !!componentDefinition.acceptsUVContext,
            providesUVContextViaCompute: !!componentDefinition.providesUVContextViaCompute,
            usesPointer: !!componentDefinition.usesPointer,
            hasUvRemap: !!componentDefinition.uvRemap,
            boundingBoxDeclaration: decl,
            wantsBoundsParams: !!componentDefinition.wantsBoundsParams,
            extraDimProps,
            mouseDriverState: {},
            autoAnimateState: {},
            mapValues: {},
            _liveHandles: null,
        }

        if (decl || extraDimProps) {
            const plan = buildDimensionalPlan(decl, extraDimProps)
            if (plan) {
                const origin = (uniforms['origin']?.value as BoundingBoxOrigin) ?? 'center'
                const he = computeDimExtent(node, origin)
                for (const name of [...plan.positionProps, ...plan.sizeProps.keys()]) {
                    const u = uniforms[name]
                    if (!u) continue
                    // RE-registration (renderer reset: toneMapping change, device loss) passes the
                    // SAME uniforms map whose .value is already RESOLVED. Preserve the original raw
                    // and resolve from IT — resolving a resolved value re-applies the origin
                    // half-extent offset, walking the layer toward its anchor on every recompile.
                    if (u._rawDimensional === undefined) u._rawDimensional = u.value
                    u.value = resolveDimensionalProp(decl, name, u._rawDimensional, origin, pxResWidth(), pxResHeight(), extraDimProps, he)
                }
            }
        }

        // Seed compileTimeWhen baselines from the initial value so the first predicate call
        // compares against a real value, not `undefined`.
        for (const u of Object.values(uniforms)) {
            if (u.compileTimeWhen && u._lastCompiledValue === undefined) u._lastCompiledValue = u.value
        }

        nodes.set(id, node)
        if (parentId === null) rootId = id

        const childrenSet = parentToChildren.get(parentId) || new Set<string>()
        childrenSet.add(id)
        parentToChildren.set(parentId, childrenSet)

        updateMouseListenerRegistration()
        markStructuralDirty()
        if (anyLayoutGroups()) markLayoutDirty()
    }

    /** Park an update aimed at a node that is still queued; unknown ids are ignored. */
    const deferUntilRegistered = (update: (typeof pendingNodeUpdates)[number]): void => {
        if (pendingRegistrationQueue.some((q) => q.id === update.id)) pendingNodeUpdates.push(update)
    }

    const updateUniformValue = (nodeId: string, uniformName: string, value: unknown): void => {
        const node = nodes.get(nodeId)
        if (!node) {
            deferUntilRegistered({id: nodeId, kind: 'uniform', name: uniformName, value})
            return
        }
        const u = node.uniforms[uniformName]
        if (!u) return

        // Shape dimensional props: stash raw, resolve px → UV before applying the transform.
        const decl = node.boundingBoxDeclaration
        const extra = node.extraDimProps
        if ((decl || extra) && isDimensionalProp(decl, uniformName, extra)) {
            u._rawDimensional = value
            const origin = (node.uniforms['origin']?.value as BoundingBoxOrigin) ?? 'center'
            const he = computeDimExtent(node, origin)
            value = resolveDimensionalProp(decl, uniformName, value, origin, pxResWidth(), pxResHeight(), extra, he)
        }

        // A list prop edit: keep the raw items, re-resolve + re-pack (a driver toggled inside an item
        // registers / drops its pointer state here), refresh the listener need. Count is a runtime
        // uniform, so adding or removing an item never recompiles.
        if (u.transform === (listPropTransform as unknown)) {
            u._rawList = value
            repackListProp(node, uniformName)
            updateMouseListenerRegistration()
            requestRender()
            return
        }

        u.value = value
        writePropHandle(node, uniformName, value)

        // A colorStops edit re-packs the four expansion fields at the active colorSpace mode.
        // Runs BEFORE the compileTimeWhen recompose check below: a same-count edit patches the live
        // arrays in place (then falls through to requestRender); a count-change edit patches AND
        // recomposes (the recompose re-seeds from the fresh `.value` mirrors this just wrote).
        // A colorSpace change is compileTime (recomposes below) but convertedColorsArray was
        // seeded at the OLD mode — re-pack it at the NEW mode now, BEFORE markStructuralDirty, so
        // the recompose re-seeds fresh converted colors.
        if (u.transform === (colorStopsTransform as unknown) || u.transform === (transformColorSpace as unknown)) {
            repackColorStops(node)
        }

        // Origin / size-affecting re-resolution.
        if (uniformName === 'origin' && decl) {
            reresolveDimensionalUniforms(node)
        } else if (decl) {
            const origin = (node.uniforms['origin']?.value as BoundingBoxOrigin) ?? 'center'
            if (origin !== 'center') {
                const plan = buildDimensionalPlan(decl, extra)
                const affectsExtent = !!decl.computeBounds || (plan?.sizeProps.has(uniformName) ?? false)
                const isPositionProp = plan?.positionProps.has(uniformName) ?? false
                if (affectsExtent && !isPositionProp) reresolveDimensionalUniforms(node)
            }
        }

        // A shape sub-prop may have toggled a mouse driver inside the `shape` JSON.
        if (uniformName === 'shape') updateMouseListenerRegistration()

        // A prop edit may change a stacked child's measured size (fontSize, radius, text…) —
        // re-run layout when any layout group exists. Cheap guard; the pass itself is deferred
        // to the next frame.
        if (anyLayoutGroups()) markLayoutDirty()

        // Compile-time props recompile the composition.
        if (u.compileTime) {
            markStructuralDirty()
            return
        }
        if (u.compileTimeWhen) {
            const prev = u._lastCompiledValue
            if (u.compileTimeWhen(prev, value)) {
                u._lastCompiledValue = value
                markStructuralDirty()
                return
            }
        }
        requestRender()
    }

    const updateNodeMetadata = (nodeId: string, metadata: Partial<NodeMetadata>): void => {
        const node = nodes.get(nodeId)
        if (!node) {
            deferUntilRegistered({id: nodeId, kind: 'metadata', metadata})
            return
        }
        let needsRecompose = false

        if (metadata.blendMode !== undefined && node.metadata.blendMode !== metadata.blendMode) {
            node.metadata.blendMode = metadata.blendMode
            needsRecompose = true
        }
        if (metadata.opacity !== undefined && node.metadata.opacity !== metadata.opacity) {
            const oldOpacity = node.metadata.opacity ?? 1
            const newOpacity = metadata.opacity
            node.metadata.opacity = newOpacity
            writeSyntheticScalar(node, '_opacity', newOpacity)
            // Crossing opacity=1 changes the emitted glue for ANY node: analytic UV-remap
            // eligibility flips on it, and composeSiblings emits no blend at all for a first child
            // sitting at exactly 1 (so `_opacity` is never read). Recompose on the crossing —
            // matched by the per-node `aopac` hash input, and cheap because both sides stay in the
            // LRU while a user drags the slider back and forth.
            const crossedBoundary = oldOpacity >= 1 !== newOpacity >= 1
            if (crossedBoundary) needsRecompose = true
        }
        if (metadata.renderOrder !== undefined && node.metadata.renderOrder !== metadata.renderOrder) {
            node.metadata.renderOrder = metadata.renderOrder
            needsRecompose = true
        }
        if ('visible' in metadata) {
            const newVisible = metadata.visible === false ? false : true
            if (node.metadata.visible !== newVisible) {
                node.metadata.visible = newVisible
                needsRecompose = true
            }
        }
        if (metadata.id !== undefined) {
            const sanitized = sanitizeId(metadata.id)
            if (node.metadata.id !== sanitized) {
                node.metadata.id = sanitized
                needsRecompose = true
            }
        }
        if ('mask' in metadata && (node.metadata.mask?.source !== metadata.mask?.source || node.metadata.mask?.type !== metadata.mask?.type)) {
            node.metadata.mask = metadata.mask
            needsRecompose = true
        }
        if ('maps' in metadata && JSON.stringify(node.metadata.maps) !== JSON.stringify(metadata.maps)) {
            const oldMaps = node.metadata.maps
            const newMaps = metadata.maps
            node.metadata.maps = newMaps
            updateMouseListenerRegistration()
            const isStructuralChange = ((): boolean => {
                if (!oldMaps || !newMaps) return true
                const oldKeys = Object.keys(oldMaps)
                const newKeys = Object.keys(newMaps)
                if (oldKeys.length !== newKeys.length || !oldKeys.every((k) => k in newMaps)) return true
                for (const key of newKeys) {
                    const oldCfg = oldMaps[key]
                    const newCfg = newMaps[key]
                    if (oldCfg.type !== newCfg.type) return true
                    if (newCfg.type === 'map') {
                        const oc = oldCfg as MapConfig
                        const nc = newCfg as MapConfig
                        if (oc.source !== nc.source || oc.channel !== nc.channel) return true
                    } else if (newCfg.type === 'mouse') {
                        if ((oldCfg as MouseMapConfig).axis !== (newCfg as MouseMapConfig).axis) return true
                    }
                }
                return false
            })()
            if (isStructuralChange) {
                // Clear stale driver state for props whose type changed.
                if (oldMaps && newMaps) {
                    for (const key of Object.keys(oldMaps)) {
                        if (key in newMaps && oldMaps[key].type !== newMaps[key].type) {
                            delete node.mouseDriverState[key]
                            delete node.autoAnimateState[key]
                            delete node.mapValues[key]
                        }
                    }
                }
                needsRecompose = true
            } else if (newMaps) {
                // Non-structural: refresh resolved bounds in place (+ GPU fields for map drivers).
                for (const [key, driver] of Object.entries(newMaps)) {
                    const mv = ensureMapValues(node, key, driver)
                    if (driver.type === 'map') {
                        writeSyntheticScalar(node, `_map_${key}_inputMin`, mv.inputMin)
                        writeSyntheticScalar(node, `_map_${key}_inputMax`, mv.inputMax)
                        writeSyntheticScalar(node, `_map_${key}_outputMin`, mv.outputMin)
                        writeSyntheticScalar(node, `_map_${key}_outputMax`, mv.outputMax)
                        writeSyntheticScalar(node, `_map_${key}_curve`, mv.curve)
                    }
                }
            }
        }
        if (metadata.transform !== undefined) {
            const prev = node.metadata.transform
            const merged = {
                offsetX: metadata.transform.offsetX ?? prev?.offsetX ?? 0,
                offsetY: metadata.transform.offsetY ?? prev?.offsetY ?? 0,
                rotation: metadata.transform.rotation ?? prev?.rotation ?? 0,
                scale: metadata.transform.scale ?? prev?.scale ?? 1,
                anchorX: metadata.transform.anchorX ?? prev?.anchorX ?? 0.5,
                anchorY: metadata.transform.anchorY ?? prev?.anchorY ?? 0.5,
                edges: metadata.transform.edges ?? prev?.edges ?? 'transparent',
            }
            const wasActive = transformActive(node)
            node.metadata.transform = merged
            const nowActive = needsTransformation(merged)
            // Activation / deactivation (or an edge-mode change) changes the emitted glue.
            if (wasActive !== nowActive || (prev && prev.edges !== merged.edges)) needsRecompose = true
            else if (nowActive) for (const [k, v] of Object.entries(transformFieldValues(node))) writeSyntheticScalar(node, `_xform_${k}`, v)
        }
        if (metadata.boundingBox !== undefined && metadata.boundingBox !== null && node.boundingBoxDeclaration !== undefined) {
            const wasActive = bboxActive(node)
            node.metadata.boundingBox = metadata.boundingBox
            const nowActive = bboxActive(node)
            // Activation OR deactivation changes the emitted glue (an identity full-frame box emits
            // nothing at all), so either crossing recomposes; a live box that stays active just
            // patches its uniforms. Mirrors the transform branch above.
            if (wasActive !== nowActive) {
                needsRecompose = true
            } else if (nowActive) {
                const bb = bboxFieldValues(node)
                if (bb) for (const [k, v] of Object.entries(bb)) writeSyntheticScalar(node, `_bbox_${k}`, v)
            }
        } else if (metadata.boundingBox === null) {
            if (node.metadata.boundingBox !== undefined) {
                node.metadata.boundingBox = undefined
                needsRecompose = true
            }
        }

        // Layout config / escape hatch — runtime-only (positions are ordinary uniforms), no
        // recompose. Turning layout OFF re-resolves descendants' dimensional props so their
        // authored (or editor-baked) positions win back from _rawDimensional.
        if ('flow' in metadata && JSON.stringify(node.metadata.flow) !== JSON.stringify(metadata.flow)) {
            const wasActive = node.metadata.flow?.mode === 'column' || node.metadata.flow?.mode === 'row'
            node.metadata.flow = metadata.flow ?? undefined
            const nowActive = node.metadata.flow?.mode === 'column' || node.metadata.flow?.mode === 'row'
            if (wasActive && !nowActive) {
                const restore = (id: string): void => {
                    for (const child of findChildNodes(id)) {
                        reresolveDimensionalUniforms(child)
                        restore(child.id)
                    }
                }
                restore(nodeId)
            }
            markLayoutDirty()
        }
        if ('absolute' in metadata && node.metadata.absolute !== metadata.absolute) {
            node.metadata.absolute = metadata.absolute
            // An escaped child re-resolves back to its own authored position.
            if (metadata.absolute) reresolveDimensionalUniforms(node)
            markLayoutDirty()
        }
        // visible/renderOrder/boundingBox changes can reshape a stack.
        if (anyLayoutGroups()) markLayoutDirty()

        if (needsRecompose) markStructuralDirty()
        else requestRender()
    }

    const removeNode = (id: string): void => {
        const node = nodes.get(id)
        if (!node) return
        // Cleanup callbacks live on the composition IR; nothing per-node to run here.
        for (const child of findChildNodes(id)) removeNode(child.id)
        if (rootId === id) rootId = null
        const siblingSet = parentToChildren.get(node.parentId)
        if (siblingSet) {
            siblingSet.delete(id)
            if (siblingSet.size === 0) parentToChildren.delete(node.parentId)
        }
        nodes.delete(id)
        updateMouseListenerRegistration()
        markStructuralDirty()
        if (anyLayoutGroups()) markLayoutDirty()
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // CPU driver layer (updateMouseDrivers)
    // ────────────────────────────────────────────────────────────────────────────────────

    const updateMouseDrivers = (): void => {
        const deltaTime = frameParams.deltaTime
        globalElapsedTime = deriveElapsedTime(sharedTimeOrigin, globalElapsedTime, deltaTime, performance.now())

        for (const node of nodes.values()) {
            const maps = node.metadata.maps ?? {}

            for (const propName in node.mouseDriverState) {
                const state = node.mouseDriverState[propName]
                const driver = driverForKey(node, propName)
                if (!driver) continue

                if (driver.type === 'mouse-position') {
                    const smoothing = driver.smoothing ?? 0
                    const momentum = driver.momentum ?? 0
                    const trackX = driver.x === undefined || driver.x === 'mouse'
                    const trackY = driver.y === undefined || driver.y === 'mouse'

                    // First real pointer event after seeding from the park: snap, don't spring.
                    if (!state.pointerSnapped && pointerSeen) {
                        state.pointerSnapped = true
                        if (trackX) { state.currentX = pointerX; state.velocityX = 0 }
                        if (trackY) { state.currentY = pointerY; state.velocityY = 0 }
                    }

                    let finalX: number
                    let finalY: number
                    if (trackX) {
                        const [nx, vx] = applySpring(state.currentX, state.velocityX, pointerX, smoothing, momentum, deltaTime)
                        state.currentX = nx
                        state.velocityX = vx
                        finalX = nx
                    } else {
                        finalX = driver.x as number
                        state.velocityX = 0
                    }
                    if (trackY) {
                        const [ny, vy] = applySpring(state.currentY, state.velocityY, pointerY, smoothing, momentum, deltaTime)
                        state.currentY = ny
                        state.velocityY = vy
                        finalY = ny
                    } else {
                        finalY = driver.y as number
                        state.velocityY = 0
                    }

                    const reach = driver.reach ?? 1
                    const originX = driver.originX ?? 0.5
                    const originY = driver.originY ?? 0.5
                    const xSign = driver.invertX ? -1 : 1
                    const ySign = driver.invertY ? -1 : 1
                    const scaledX = originX + xSign * (finalX - originX) * reach
                    const scaledY = originY + ySign * (finalY - originY) * reach

                    // Resize-fit generators read this as BOX-LOCAL; map through the box like the GPU.
                    let localX = scaledX
                    let localY = scaledY
                    const bb = node.metadata.boundingBox
                    if (bb && node.boundingBoxDeclaration?.supportsResizeFit && bboxActive(node)) {
                        const p = boundingBoxToUVParams(bb, pxResWidth(), pxResHeight())
                        const bl = screenUVToBoxLocal(scaledX, scaledY, {
                            centerX: p.centerX,
                            centerY: p.centerY,
                            halfWidthUV: p.halfWidthUV,
                            halfHeightUV: p.halfHeightUV,
                            rotationDeg: p.rotation,
                            aspectRatio: aspectRatio(),
                        })
                        localX = bl.x
                        localY = bl.y
                    }
                    state.liveX = localX
                    state.liveY = localY
                    const h = node._liveHandles?.[`_smoothed_${propName}`]
                    if (h instanceof FieldHandle) h.setComponents(localX, 1 - localY)
                } else if (driver.type === 'mouse') {
                    const smoothing = driver.smoothing ?? 0
                    const momentum = driver.momentum ?? 0
                    const target = driver.axis === 'x' ? pointerX : pointerY
                    // First real pointer event after seeding from the park: snap, don't spring.
                    if (!state.pointerSnapped && pointerSeen) {
                        state.pointerSnapped = true
                        state.currentX = Math.max(0.001, target)
                        state.velocityX = 0
                    }
                    const [nx, vx] = applySpring(state.currentX, state.velocityX, target, smoothing, momentum, deltaTime)
                    state.currentX = nx
                    state.velocityX = vx
                    const smoothed01 = Math.max(0.001, nx)
                    // The composer reads _smoothed_<prop> as the FINAL value, so remap on CPU.
                    const mv = node.mapValues[propName]
                    const resolved = mv ? mv.outputMin + (mv.outputMax - mv.outputMin) * Math.pow(smoothed01, Math.pow(2, -mv.curve * 2)) : smoothed01
                    state.liveScalar = resolved
                    writeSyntheticScalar(node, `_smoothed_${propName}`, resolved)
                }
            }

            // Driven list elements: re-resolve + re-pack from the states the loop above advanced.
            for (const [prop, u] of Object.entries(node.uniforms)) {
                if (u._listSpec && listHasDrivers(u._rawList, u._listSpec)) repackListProp(node, prop)
            }

            for (const propName in node.autoAnimateState) {
                const driver = maps[propName] as PropDriver | undefined
                if (!driver || driver.type !== 'auto-animate') continue
                const globalT = globalElapsedTime * (driver.speed ?? 1.0) * 0.2
                let phase: number
                const easing = driver.easing ?? driver.waveform ?? 'sine'
                if (driver.mode === 'loop') {
                    const t = ((globalT % 1) + 1) % 1
                    phase = applyEasing(t, easing)
                } else {
                    const t01 = ((globalT % 1) + 1) % 1
                    const triangleT = t01 < 0.5 ? t01 * 2 : (1 - t01) * 2
                    phase = applyEasing(triangleT, easing)
                }
                const mv = node.mapValues[propName]
                const outputMin = mv ? mv.outputMin : resolveDynamicBound(node, propName, driver.outputMin)
                const outputMax = mv ? mv.outputMax : resolveDynamicBound(node, propName, driver.outputMax)
                const newValue = outputMin + phase * (outputMax - outputMin)

                // Replicate compileTimeWhen (auto-animate bypasses updateUniformValue).
                const u = node.uniforms[propName]
                if (u?.compileTimeWhen) {
                    const prev = u._lastCompiledValue
                    if (u.compileTimeWhen(prev, newValue)) {
                        u._lastCompiledValue = newValue
                        markStructuralDirty()
                    }
                }
                writeSyntheticScalar(node, `_smoothed_${propName}`, newValue)
            }
        }
    }

    /**
     * Advance every animated-time node's CPU accumulator and write its `_animTime` field.
     * Uses this frame's deltaTime; speed=0 pauses (no rewind) because `advance` adds `dt * speed`.
     * Runs alongside `updateMouseDrivers` (step 1) so the value is in place before the flush.
     */
    const updateAnimatedTime = (): void => {
        const dt = frameParams.deltaTime
        for (const node of nodes.values()) {
            const at = node.definition.animatedTime
            if (at) {
                const speedU = node.uniforms[at.speed]
                const speed = speedU ? Number(speedU.value) || 0 : 0
                const state = getAnimatedTimeState(animatedTimeKey(node))
                state.advance(dt, speed)
                writeSyntheticScalar(node, '_animTime', state.value)
            }
            // Advance each extra clock on its own accumulator (FlowField's evolution speed).
            const extras = node.definition.extraAnimatedTimes
            if (extras) {
                for (const [key, speedProp] of Object.entries(extras)) {
                    const su = node.uniforms[speedProp]
                    const sp = su ? Number(su.value) || 0 : 0
                    const st = getAnimatedTimeState(extraAnimatedTimeKey(node, speedProp))
                    st.advance(dt, sp)
                    writeSyntheticScalar(node, `_animTime_${key}`, st.value)
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Sizing (DPR is our own factor)
    // ────────────────────────────────────────────────────────────────────────────────────

    const clampToTextureCap = (w: number, h: number): {width: number; height: number} =>
        clampToTextureCapPure(w, h, {
            maxTextureDim: cachedMaxTextureDim,
            pixelRatio,
            viewportWidth: typeof window !== 'undefined' ? window.innerWidth : w,
            viewportHeight: typeof window !== 'undefined' ? window.innerHeight : h,
        })

    const basePixelRatio = (): number => {
        const deviceRatio = (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1
        return Math.min(deviceRatio, maxPixelRatioForDevice())
    }

    /** Apply a concrete backing-buffer size to the canvas, RTT textures and every node. */
    const applyRendererDimensions = (width: number, height: number, rawWidth: number, rawHeight: number): void => {
        // `windowResizeHandler` deliberately defeats updateRendererDimensions' request dedupe so the
        // viewport-based texture cap gets re-evaluated on every window resize (it can go stale with no
        // container resize). Most of the time the cap lands on the same numbers — a canvas of fixed
        // size inside a resizable window — so nothing here could possibly change. Bail before the
        // expensive tail (texture realloc, dimensional re-resolve, layout, repaint, render).
        const unchanged =
            hasInitialDimensions &&
            width === currentWidth &&
            height === currentHeight &&
            rawWidth === logicalWidth &&
            rawHeight === logicalHeight
        if (unchanged) {
            const {width: curBufW, height: curBufH} = bufferSize()
            if (!canvasElement || (canvasElement.width === curBufW && canvasElement.height === curBufH)) return
        }

        currentWidth = width
        currentHeight = height
        logicalWidth = rawWidth
        logicalHeight = rawHeight
        setShaderCanvasDimensions(rawWidth, rawHeight)

        if (!hasInitialDimensions) {
            hasInitialDimensions = true
            if (nodes.size > 0) markStructuralDirty()
        }

        // Size the canvas backing buffer (WE own DPR now).
        const {width: bufW, height: bufH} = bufferSize()
        if (canvasElement) {
            canvasElement.width = bufW
            canvasElement.height = bufH
        }
        // Resize every live composition's RTT textures to the new backing buffer.
        let anyTextureRecreated = false
        for (const c of liveCompositions) {
            if (c.passManager.resize(bufW, bufH)) anyTextureRecreated = true
        }

        // Drain shader onResize callbacks. Blur's compute keeps its input-canvas
        // dimensions as uniforms (fixed-resolution buffers never realloc); setInputDimensions
        // pulls the new backing size through.
        for (const c of liveCompositions) {
            for (const cb of c.ir.onResize) {
                try {
                    cb({width: bufW, height: bufH})
                } catch (e) {
                    debugError('[gpu] composition onResize error:', e)
                }
            }
        }

        // Refresh aspect-dependent + dimensional per-node uniforms.
        for (const node of nodes.values()) {
            if (transformActive(node)) writeSyntheticScalar(node, '_xform_aspectRatio', aspectRatio())
            if (bboxActive(node)) {
                const bb = bboxFieldValues(node)
                if (bb) for (const [k, v] of Object.entries(bb)) writeSyntheticScalar(node, `_bbox_${k}`, v)
            }
            reresolveDimensionalUniforms(node)
        }
        // Layout runs AFTER dimensional re-resolution: it overrides in-flow children's positions,
        // so the re-resolved authored values must land first (they win only when layout is off).
        runLayoutPass()

        // Warm-up repaint: passManager.resize just destroyed + recreated every RTT texture
        // (zero-initialized), and the frame below runs compute BEFORE the RTT passes — a compute
        // step that reads a child RTT (Glass frosted blur, kit/blur consumers) would blur black
        // and the effect would go dark for one frame. Repaint the RTT passes now so that compute
        // reads real content. RTT-only: no compute dispatch (simulations don't double-step) and
        // no time advance. Flush first so the repaint sees the new viewport uniforms (staged
        // above + system) rather than last frame's. Skipped when `resize` reallocated nothing: the
        // textures still hold last frame's content, so there is no zeroed texture to refill.
        if (anyTextureRecreated) {
            for (const c of liveCompositions) {
                writeSystemUniforms(c.store)
                c.store.flush()
                c.passManager.repaintRtt()
            }
        }

        lastRenderTime = 0
        renderFrame()
    }

    /** RAF-throttled resize entry point. */
    const updateRendererDimensions = (width: number, height: number): void => {
        if (width <= 0 || height <= 0) return
        if (width === lastRequestedWidth && height === lastRequestedHeight) return
        lastRequestedWidth = width
        lastRequestedHeight = height
        pendingResize = {width, height}
        if (!isResizeScheduled && typeof requestAnimationFrame === 'function') {
            isResizeScheduled = true
            requestAnimationFrame(() => {
                isResizeScheduled = false
                if (!pendingResize) return
                const raw = pendingResize
                pendingResize = null
                const {width: cw, height: ch} = clampToTextureCap(raw.width, raw.height)
                applyRendererDimensions(cw, ch, raw.width, raw.height)
            })
        }
    }

    const handleResize = (entries: ResizeObserverEntry[]): void => {
        const entry = entries[0]
        if (!entry) return
        let width: number, height: number
        if (entry.contentBoxSize) {
            const boxSize = Array.isArray(entry.contentBoxSize) ? entry.contentBoxSize[0] : entry.contentBoxSize
            if (boxSize) {
                width = Math.round(boxSize.inlineSize)
                height = Math.round(boxSize.blockSize)
            } else {
                width = Math.round(entry.contentRect.width)
                height = Math.round(entry.contentRect.height)
            }
        } else {
            width = Math.round(entry.contentRect.width)
            height = Math.round(entry.contentRect.height)
        }
        if (width === 0 || height === 0) return
        updateRendererDimensions(width, height)
    }

    const handleVisibilityChange = (entries: IntersectionObserverEntry[]): void => {
        const entry = entries[0]
        if (!entry) return
        isVisible = entry.isIntersecting
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Render
    // ────────────────────────────────────────────────────────────────────────────────────

    const canRender = (): boolean =>
        isInitialized && !!root && !!context && currentWidth > 0 && currentHeight > 0

    const writeSystemUniforms = (store: UniformStore): void => {
        const sys = store.systemHandles
        const {width, height} = bufferSize()
        const bpr = basePixelRatio()
        const set = (name: string, value: number | number[]): void => {
            const h = sys[name]
            if (h instanceof FieldHandle) h.value = value
        }
        set('time', globalElapsedTime)
        set('viewportSize', [width, height])
        set('logicalViewportSize', [pxResWidth() * bpr, pxResHeight() * bpr])
        set('aspect', width / Math.max(1, height))
        set('pointer', [pointerX, pointerY])
        set('pointerActive', pointerActive ? 1 : 0)
    }

    const runCallbacks = (callbacks: ((p: unknown) => void)[]): void => {
        for (const cb of callbacks) {
            try {
                cb(frameParams)
            } catch (e) {
                debugError('[gpu] render callback error:', e)
            }
        }
    }

    const renderFrameInternal = (overrideDeltaTime?: number): void => {
        if (failureReason || !canRender()) return

        const now = performance.now()
        let delta: number
        if (overrideDeltaTime !== undefined) {
            delta = overrideDeltaTime // synthetic: bypass throttle + clamp
        } else {
            const gate = frameGate(now, {lastRenderTime, isVisible, forceFullFrameRate, minInterval: frameIntervalCap})
            if (!gate.render) return
            delta = gate.deltaTime
        }
        lastRenderTime = now

        if (enablePerformanceTracking) performance.mark('shader-cpu-start')

        // Per-frame params — read synchronously by driver + shader callbacks below.
        // Shader CPU callbacks (PixelThrow, ChromaFlow, Shatter, ...) do raw
        // distance/velocity math on frameParams.pointer without checking
        // pointerActive — feeding them the -10 off-canvas sentinel produces a
        // spurious velocity/distance spike the instant the pointer leaves the
        // canvas. Default to normalized center instead; pointerActive is
        // still exposed separately for callbacks that do want to distinguish
        // "no cursor yet".
        // Once a real pointer position has landed, HOLD it even while pointerActive is false —
        // mouseup fires window-wide and flips pointerActive off on any click anywhere on the
        // page, and snapping to center there made every velocity-based sim (PixelThrow,
        // InkFlow) burst toward 0.5/0.5 on off-canvas clicks. Center is only the "no cursor
        // ever" default (sentinel still parked at -10).
        const hasRealPointer = pointerX > -5
        frameParams.deltaTime = delta
        frameParams.pointer.x = hasRealPointer ? pointerX : 0.5
        frameParams.pointer.y = hasRealPointer ? pointerY : 0.5
        frameParams.pointer.seen = pointerSeen
        frameParams.pointerActive = pointerActive
        frameParams.dimensions.width = currentWidth
        frameParams.dimensions.height = currentHeight

        let built: BuiltComposition | null = null
        let renderComp: BuiltComposition | null = null

        // Frame order: drivers/time → [ensure composition] → onBeforeRender →
        // flush → render → markReady → onAfterRender.
        //
        // The whole sequence is guarded: composition BUILD (WGSL resolve, pipeline creation,
        // RTT allocation) happens inside `ensureComposition`, and on a driver that can't
        // honour our shaders that throws every single frame. Unguarded it escapes the RAF
        // callback as an uncaught error — one console entry per frame, forever.
        try {
            runFrameSequence({
                updateDrivers: () => {
                    updateMouseDrivers()
                    updateAnimatedTime()
                    updateChildBounds() // Repeater child-shape bounds refresh
                    runLayoutPassIfDirty() // group layout re-arrange (deferred from prop/metadata edits)
                },
                ensureComposition: () => {
                    // Recompute the structural hash only when a structural change is pending.
                    if (structuralDirty || desiredHash === null) {
                        desiredHash = computeStructuralHash()
                        structuralDirty = false
                    }
                    if (desiredHash === null) return false // no root composed yet
                    built = pipelineCache.getOrBuild(desiredHash, () => buildComposition(desiredHash!))
                    bindComposition(built)
                    return !!boundComposition
                },
                beforeRender: () => runCallbacks(boundComposition!.ir.onBeforeRender),
                flush: () => {
                    writeSystemUniforms(boundComposition!.store)
                    boundComposition!.store.flush()
                },
                render: () => {
                    if (enablePerformanceTracking) performance.mark('shader-gpu-start')
                    renderComp = pipelineCache.renderValue ?? built
                    // A composition a custom WGSL body broke draws nothing (the last good frame
                    // stays up) until the structure — a fixed body, a new revision — changes.
                    if (renderComp?.broken) return false
                    const comp = renderComp
                    const scopeDevice = comp && comp.validationFrames > 0 && comp.ir.usesCustomWgsl ? root?.device : undefined
                    if (scopeDevice) scopeDevice.pushErrorScope('validation')
                    try {
                        // afterCompute: re-flush field patches written during compute-node collection
                        // (setExtraField) so they reach the GPU before this frame's passes encode —
                        // see passManager.render's comment (one-frame _vf* domain mismatch otherwise).
                        comp?.passManager.render(context, frameParams, () => boundComposition?.store.flush())
                    } finally {
                        // Pop in `finally` so a throwing frame cannot leave the scope open.
                        if (scopeDevice && comp) {
                            comp.validationFrames--
                            scopeDevice.popErrorScope().then((error) => {
                                if (error && !comp.broken) markBroken(comp, error.message)
                            }).catch(() => {
                                /* device lost mid-frame — the loss handler owns recovery */
                            })
                        }
                    }
                    if (enablePerformanceTracking) {
                        performance.mark('shader-gpu-end')
                        try {
                            performance.measure('shader-gpu-time', 'shader-gpu-start', 'shader-gpu-end')
                            const gpuMeasure = performance.getEntriesByName('shader-gpu-time')[0]
                            if (gpuMeasure) performanceTracker.recordGpuTime(gpuMeasure.duration)
                            performance.clearMeasures('shader-gpu-time')
                            performance.clearMarks('shader-gpu-start')
                            performance.clearMarks('shader-gpu-end')
                        } catch {
                            /* ignore measurement errors */
                        }
                    }
                },
                markReady: () => {
                    pipelineCache.markReady(desiredHash!)
                    if (firstRenderPending && !hasEmittedReady) {
                        hasEmittedReady = true
                        firstRenderPending = false
                        if (onReadyCallback) queueMicrotask(onReadyCallback)
                    }
                },
                afterRender: () => runCallbacks((renderComp ?? boundComposition!).ir.onAfterRender),
            })
            consecutiveRenderErrors = 0
        } catch (error) {
            noteRenderFailure(error)
            return
        }

        if (enablePerformanceTracking) {
            performance.mark('shader-cpu-end')
            try {
                performance.measure('shader-cpu-time', 'shader-cpu-start', 'shader-cpu-end')
                const cpuMeasure = performance.getEntriesByName('shader-cpu-time')[0]
                if (cpuMeasure) performanceTracker.recordCpuTime(cpuMeasure.duration)
                performance.clearMarks('shader-cpu-start')
                performance.clearMarks('shader-cpu-end')
                performance.clearMeasures('shader-cpu-time')
            } catch {
                /* ignore measurement errors */
            }
        }
        performanceTracker.recordFrame(performance.now() - now)
    }

    const renderFrame = (): void => renderFrameInternal(undefined)

    /**
     * Force one frame now and wait for it, bypassing the frame-rate throttle. Used by the
     * capture paths (screenshot, thumbnail, recording setup) that must read pixels straight
     * after.
     *
     * `waitForGpu: false` skips the `onSubmittedWorkDone()` fence — see the note on
     * {@link renderSyntheticFrame} for exactly when that is safe, and for why it matters
     * (the fence costs ~104ms per call on Firefox 152 vs ~1ms on Chromium).
     */
    const renderAndWait = async (options?: {waitForGpu?: boolean}): Promise<void> => {
        // A permanently-failed renderer resolves instead of rejecting: capture/export callers
        // then read a transparent canvas (the truthful result) rather than having to handle —
        // and usually log — an exception on every browser that can't run WebGPU.
        if (failureReason) return
        if (!canRender()) throw new Error('Renderer is not ready')
        const wasAnimating = frameLoop.running
        if (wasAnimating) stopAnimation()
        lastRenderTime = 0
        renderFrame()
        if (root && options?.waitForGpu !== false) await awaitGpuIdle(root)
        if (wasAnimating) startAnimation()
    }

    /**
     * Advance the clock by exactly `deltaSeconds` and draw one frame, independent of
     * wall-clock time. The deterministic path behind frame-locked video export.
     *
     * `waitForGpu` (default true) awaits `queue.onSubmittedWorkDone()` before resolving, so
     * the caller is guaranteed the frame is finished. That fence is cheap on Chromium
     * (~0.3–1.3ms) but costs **~104ms per call on Firefox 152**, which turns a 600-frame
     * export from ~12s into ~60s.
     *
     * Pass `waitForGpu: false` ONLY when the very next thing you do is a `toBlob()` /
     * `convertToBlob()` readback of the same canvas. That readback has to produce real pixels,
     * so it synchronises with the queue itself and the explicit fence is redundant — which is
     * precisely why the capture paths use `toBlob` rather than `drawImage`/`createImageBitmap`
     * on the canvas (those read a presented-or-recycled frame and were the source of the old
     * intermittent black/corrupt export frames).
     *
     * Do NOT skip the fence if you intend to read the canvas some other way, hand the device
     * to another renderer, or just assume the frame has landed.
     */
    const renderSyntheticFrame = async (deltaSeconds: number, options?: {waitForGpu?: boolean}): Promise<void> => {
        renderFrameInternal(deltaSeconds)
        if (root && options?.waitForGpu !== false) await awaitGpuIdle(root)
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Animation control
    // ────────────────────────────────────────────────────────────────────────────────────

    const startAnimation = (): void => {
        // A failed renderer must never spin a RAF loop. Hosts call this from their own
        // visibility observers, which keep firing long after we've given up.
        if (failureReason || frameLoop.running || !shouldAnimate) return
        performanceTracker.setRendering(true)
        frameLoop.start()
    }

    const stopAnimation = (): void => {
        if (!frameLoop.running) return
        frameLoop.stop()
        performanceTracker.setRendering(false)
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Pointer / event listeners
    // ────────────────────────────────────────────────────────────────────────────────────

    const updatePointerCoords = (event: MouseEvent | TouchEvent): void => {
        if (!canvasElement) return
        cachedCanvasRect = canvasElement.getBoundingClientRect()
        let clientX: number
        let clientY: number
        if ('touches' in event) {
            if (event.touches.length === 0) return
            clientX = event.touches[0].clientX
            clientY = event.touches[0].clientY
        } else {
            clientX = event.clientX
            clientY = event.clientY
        }
        pointerX = (clientX - cachedCanvasRect.left) / cachedCanvasRect.width
        pointerY = (clientY - cachedCanvasRect.top) / cachedCanvasRect.height
        pointerSeen = true
    }

    const onPointerMove = (event: MouseEvent): void => {
        if (!isInitialized || !canvasElement) return
        updatePointerCoords(event)
        // Hover alone (no press) still means the cursor is genuinely over the
        // canvas — frameParams.pointer's off-canvas fallback is keyed off this
        // flag, so hover-driven effects need it true here too, not just on
        // press. Coords are already current from the line above.
        pointerActive = true
        // A mouse-driven prop must repaint even when the RAF loop is stopped (render-on-demand:
        // paused/off-screen host that still wants pointer response). No-op while animating —
        // requestRender early-returns if the frame loop is running or a frame is already pending.
        requestRender()
    }
    const onTouchMove = (event: TouchEvent): void => {
        if (!isInitialized || !canvasElement) return
        updatePointerCoords(event)
        pointerActive = true
        requestRender()
    }
    const onPointerUp = (): void => {
        if (!isInitialized) return
        pointerActive = false
        requestRender()
    }
    const onTouchEnd = (): void => {
        if (!isInitialized) return
        pointerActive = false
        requestRender()
    }
    // Down events arrive window-level (capture) — a press only counts when it lands within THIS
    // canvas's rect, so overlaid content (editor hit layers, hero copy) can't swallow the press
    // yet presses on unrelated UI don't grab the shader.
    const pressWithinCanvas = (clientX: number, clientY: number): boolean => {
        if (!isInitialized || !canvasElement) return false
        const rect = canvasElement.getBoundingClientRect()
        if (rect.width === 0 || rect.height === 0) return false
        return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom
    }
    const onPointerDown = (event: MouseEvent): void => {
        if (event.button !== 0) return // primary only — a context-menu press may never deliver its mouseup
        if (!pressWithinCanvas(event.clientX, event.clientY)) return
        pointerActive = true
        updatePointerCoords(event)
        requestRender()
    }
    const onTouchStart = (event: TouchEvent): void => {
        if (event.touches.length === 0) return
        if (!pressWithinCanvas(event.touches[0].clientX, event.touches[0].clientY)) return
        pointerActive = true
        updatePointerCoords(event)
        requestRender()
    }

    const updateMouseListenerRegistration = (): void => {
        let needsMouse = false
        for (const node of nodes.values()) {
            if (node.usesPointer) {
                needsMouse = true
                break
            }
            if (shapeConfigUsesMouse(node.uniforms?.shape?.value) || listsUseMouse(node)) {
                needsMouse = true
                break
            }
            if (!node.metadata.maps) continue
            for (const driver of Object.values(node.metadata.maps)) {
                if (driver && typeof driver === 'object' && 'type' in driver) {
                    const t = (driver as {type: string}).type
                    if (t === 'mouse' || t === 'mouse-position') {
                        needsMouse = true
                        break
                    }
                }
            }
            if (needsMouse) break
        }

        if (needsMouse && !mouseListenersNeeded) {
            mouseListenersNeeded = true
            if (!globalEventUnregister && canvasElement) {
                globalEventUnregister = globalEventManager.register({
                    onMouseMove: onPointerMove,
                    onTouchMove,
                    onMouseDown: onPointerDown,
                    onTouchStart,
                    onMouseUp: onPointerUp,
                    onTouchEnd,
                })
            }
        } else if (!needsMouse && mouseListenersNeeded) {
            mouseListenersNeeded = false
            if (globalEventUnregister) {
                globalEventUnregister()
                globalEventUnregister = null
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Registration queue processing
    // ────────────────────────────────────────────────────────────────────────────────────

    const processQueuedRegistrations = (): void => {
        if (pendingRegistrationQueue.length === 0) return
        const queue = [...pendingRegistrationQueue]
        pendingRegistrationQueue = []
        for (const q of queue) {
            registerNode(q.id, q.definition.fragment, q.parentId, q.metadata, q.uniforms, q.definition, q.domCanvas)
        }
        // Catch the nodes up on anything that changed between their registration and now.
        const updates = pendingNodeUpdates
        pendingNodeUpdates = []
        for (const u of updates) {
            if (u.kind === 'uniform') updateUniformValue(u.id, u.name, u.value)
            else updateNodeMetadata(u.id, u.metadata)
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Fatal failure → transparent canvas, silent console
    // ────────────────────────────────────────────────────────────────────────────────────

    /**
     * Give up permanently: stop the frame loop, release every GPU resource, and tell the
     * host exactly once. Nothing is written to the console unless debugging is switched on —
     * a decorative shader that can't run must not shout about it.
     *
     * A renderer that failed before it ever drew leaves a transparent canvas, which is the
     * whole contract on an unsupported browser. One that dies after drawing keeps whatever
     * frame the browser is already presenting (see the note in `cleanup`).
     *
     * Idempotent: the first reason wins, later ones are ignored.
     */
    const fail = (reason: GpuFailureReason, error?: unknown): void => {
        if (failureReason) return
        failureReason = reason
        // Some reasons are facts about the whole page, not just this renderer — publish them
        // so sibling shaders don't each repeat the same failing request.
        markGpuUnusable(reason)
        debugError(`[gpu] renderer stopped (${reason}):`, error)
        try {
            cleanup()
        } catch (e) {
            debugError('[gpu] cleanup during failure threw:', e)
        }
        if (!hasNotifiedUnavailable && onUnavailableCallback) {
            hasNotifiedUnavailable = true
            const cb = onUnavailableCallback
            queueMicrotask(() => {
                try {
                    cb(reason, error)
                } catch (e) {
                    debugError('[gpu] onUnavailable callback threw:', e)
                }
            })
        }
    }

    /**
     * Count a frame that threw. Isolated hiccups (a media texture that wasn't decoded yet)
     * clear on the next successful frame; a pipeline the driver can't run throws every
     * frame, so the counter trips and we shut down instead of logging once per frame
     * forever. Only the FIRST error of a run is debug-logged — a per-frame console flood is
     * itself a way to freeze a tab.
     */
    const noteRenderFailure = (error: unknown): void => {
        // A GpuUnavailableError from the build path is a DEFINITE verdict, not a flaky frame
        // (e.g. "this composition needs more uniform bytes than the device allows"). Retrying it
        // four more times can only produce the same answer, so short-circuit and keep the
        // specific reason instead of flattening it to 'render-failed'.
        if (isGpuUnavailableError(error)) {
            fail(error.reason, error)
            return
        }
        consecutiveRenderErrors++
        if (consecutiveRenderErrors === 1) authorError('[gpu] render error:', error)
        if (consecutiveRenderErrors >= MAX_CONSECUTIVE_RENDER_ERRORS) fail('render-failed', error)
    }

    /**
     * Listen for device-level errors that escaped every error scope.
     *
     * `preventDefault()` is the load-bearing part: per the WebGPU spec a user agent reports
     * an uncaptured error to the console only when the event was NOT canceled. Cancelling it
     * is what stops browsers printing `Uncaptured WebGPU error: …` on pages that embed a
     * shader they can't run.
     *
     * Severity policy:
     *   - out-of-memory → fatal immediately. The GPU has already refused an allocation;
     *     continuing to submit frames is exactly the path that hard-freezes a tab.
     *   - internal      → fatal immediately. The device is in an undefined state.
     *   - validation    → tolerated up to {@link MAX_UNCAPTURED_VALIDATION_ERRORS}, because a
     *     healthy renderer produces none and we don't want a single transient one to blank a
     *     working page.
     */
    const attachUncapturedErrorHandler = (device: GPUDevice): void => {
        if (!device || typeof device.addEventListener !== 'function') return
        detachUncapturedErrorHandler()
        uncapturedErrorDevice = device
        uncapturedErrorHandler = (event: Event): void => {
            try {
                event.preventDefault()
            } catch {
                /* non-cancelable in some polyfills — nothing else we can do */
            }
            const gpuError = (event as {error?: unknown}).error
            authorError('[gpu] uncaptured device error (a custom WGSL body that fails to compile reports here — read the message for the line):', gpuError)
            if (typeof GPUOutOfMemoryError !== 'undefined' && gpuError instanceof GPUOutOfMemoryError) {
                fail('out-of-memory', gpuError)
                return
            }
            if (typeof GPUInternalError !== 'undefined' && gpuError instanceof GPUInternalError) {
                fail('gpu-error', gpuError)
                return
            }
            uncapturedValidationErrors++
            if (uncapturedValidationErrors >= MAX_UNCAPTURED_VALIDATION_ERRORS) fail('gpu-error', gpuError)
        }
        device.addEventListener('uncapturederror', uncapturedErrorHandler)
    }

    const detachUncapturedErrorHandler = (): void => {
        if (uncapturedErrorDevice && uncapturedErrorHandler) {
            try {
                uncapturedErrorDevice.removeEventListener('uncapturederror', uncapturedErrorHandler)
            } catch {
                /* ignore */
            }
        }
        uncapturedErrorDevice = null
        uncapturedErrorHandler = null
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Device loss
    // ────────────────────────────────────────────────────────────────────────────────────

    const fireDeviceLost = (reason: string): void => {
        if (deviceLostFired) return
        deviceLostFired = true
        try {
            stopAnimation()
        } catch {
            /* noop */
        }
        if (onDeviceLostCallback) {
            const cb = onDeviceLostCallback
            queueMicrotask(() => {
                try {
                    cb(reason)
                } catch (e) {
                    debugError('[gpu] onDeviceLost callback threw:', e)
                }
            })
            return
        }
        // Nobody is going to rebuild us (the framework hosts don't wire recovery), so don't
        // sit on a dead device holding resources and a stale frame — release everything and
        // go transparent. `fail` also notifies any onUnavailable subscriber.
        fail('device-lost', reason)
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Initialize / cleanup
    // ────────────────────────────────────────────────────────────────────────────────────

    const initialize = async (options: GpuInitializeOptions): Promise<void> => {
        if (options.context !== undefined) {
            throw new Error(
                '[gpu] initialize({ context }) is not supported — this renderer is WebGPU-only. Remove the WebGL `context` option.',
            )
        }
        // This renderer has already decided it can't draw here. Hosts re-enter initialize on
        // colorSpace/toneMapping changes and when a canvas scrolls back into view; retrying
        // would re-run the same failing device request on every one of those.
        if (failureReason) return
        // Another renderer on this page already proved WebGPU is a dead end (no adapter, the
        // GPU refusing allocations, a backend that can't run our shaders). Starting anyway
        // would repeat a request we know fails and, on the memory failures, deepen the very
        // pressure that froze things — so adopt the verdict without touching the GPU.
        const pageWide = getGpuUnusableReason()
        if (pageWide) {
            fail(pageWide)
            return
        }
        if (isInitialized || isInitializing) return

        const {
            canvas,
            resizeTarget,
            enablePerformanceTracking: enableTracking = false,
            colorSpace = 'p3-linear',
            toneMapping = 'linear',
            gpu,
            forceFullFrameRate: forceFullFps = false,
            observeElement: shouldObserveElement = true,
            timeOrigin: initialTimeOrigin,
        } = options

        if (initialTimeOrigin != null) sharedTimeOrigin = initialTimeOrigin
        enablePerformanceTracking = enableTracking
        forceFullFrameRate = forceFullFps
        rendererColorSpace = colorSpace
        rendererToneMapping = toneMapping
        // Re-arm the layout triggers a prior cleanup() detached (toneMapping cycles re-init
        // this same instance) — without this, image loads and webfonts stop re-flowing groups.
        attachLayoutListeners()

        isInitializing = true
        initializationAbortController = new AbortController()
        const abort = initializationAbortController

        try {
            // GPU-authoritative color-space setter (gpu/transforms.ts).
            setColorSpaceModeFn = setColorSpaceModeGpu
            setColorSpaceModeFn(colorSpace)
            if (abort.signal.aborted) return

            canvasElement = canvas

            // Acquire the root (adopting an injected device, or requesting our own).
            rootCtx = await acquireRoot({device: gpu?.device, adapter: gpu?.adapter, onDeviceLost: (info) => fireDeviceLost(info.message || info.reason || 'unknown')})
            if (abort.signal.aborted) return
            root = rootCtx.root

            // Catch device errors that escape every error scope BEFORE any GPU work is
            // submitted, so the very first bad allocation is intercepted (and kept out of
            // the console) rather than reported by the browser.
            attachUncapturedErrorHandler(root.device)

            // DPR is our own factor (no setPixelRatio).
            pixelRatio = basePixelRatio()

            // Refine the real max texture dimension.
            const lim = root.device.limits?.maxTextureDimension2D
            if (typeof lim === 'number' && lim > 0) cachedMaxTextureDim = lim

            // GPU services.
            textureManager = createTextureManager(root)
            dispatcher = createComputeDispatcher(root)

            // Page unload + window resize (viewport-based cap can go stale without a container resize).
            if (typeof window !== 'undefined') {
                unloadHandler = () => cleanup()
                window.addEventListener('beforeunload', unloadHandler)
                windowResizeHandler = () => {
                    if (lastRequestedWidth > 0 && lastRequestedHeight > 0) {
                        const rawW = lastRequestedWidth
                        const rawH = lastRequestedHeight
                        lastRequestedWidth = 0
                        lastRequestedHeight = 0
                        updateRendererDimensions(rawW, rawH)
                    }
                }
                window.addEventListener('resize', windowResizeHandler, {passive: true})
            }

            // Initial dimensions (immediate, not RAF-throttled — so the first frame has a size).
            const rect = canvas.getBoundingClientRect()
            cachedCanvasRect = rect
            isVisible =
                !shouldObserveElement ||
                (rect.width > 0 &&
                    rect.height > 0 &&
                    rect.top < window.innerHeight &&
                    rect.bottom > 0 &&
                    rect.left < window.innerWidth &&
                    rect.right > 0)
            shouldAnimate = true

            const rawWidth = rect.width > 0 ? rect.width : canvas.width
            const rawHeight = rect.height > 0 ? rect.height : canvas.height
            const roundedRawWidth = Math.round(rawWidth)
            const roundedRawHeight = Math.round(rawHeight)
            if (roundedRawWidth > 0 && roundedRawHeight > 0) {
                lastRequestedWidth = roundedRawWidth
                lastRequestedHeight = roundedRawHeight
                const {width: cw, height: ch} = clampToTextureCap(roundedRawWidth, roundedRawHeight)
                currentWidth = cw
                currentHeight = ch
                logicalWidth = roundedRawWidth
                logicalHeight = roundedRawHeight
                hasInitialDimensions = true
                setShaderCanvasDimensions(roundedRawWidth, roundedRawHeight)
                const {width: bufW, height: bufH} = bufferSize()
                canvas.width = bufW
                canvas.height = bufH
            } else {
                hasInitialDimensions = false
            }

            // Configure the canvas context (once — it auto-tracks canvas size).
            context = configureCanvasContext(root, canvas)

            if (shouldObserveElement && typeof ResizeObserver !== 'undefined') {
                const elementToObserve = resizeTarget ?? canvas.parentElement
                if (elementToObserve) {
                    resizeObserver = new ResizeObserver(handleResize)
                    resizeObserver.observe(elementToObserve)
                } else {
                    console.warn('[gpu] no element available for resize observation — use resize() manually')
                }
                if (typeof IntersectionObserver !== 'undefined') {
                    // 128px rootMargin matches the framework host's own visibility observer
                    // (Shader.vue). Without it the two disagree in a ~128px band around the
                    // viewport edge: the host keeps the RAF loop running while THIS observer reports
                    // not-intersecting, dropping us into the 1-FPS off-screen throttle — drivers then
                    // look "frozen"/laggy on a canvas the user still sees. Aligning the margin closes
                    // that window; a canvas truly far off-screen still throttles as intended.
                    intersectionObserver = new IntersectionObserver(handleVisibilityChange, {threshold: 0, rootMargin: '128px'})
                    intersectionObserver.observe(canvas)
                }
            }

            if (abort.signal.aborted) return

            isInitialized = true
            isRendererReady = true
            processQueuedRegistrations()

            if (currentWidth > 0 && currentHeight > 0) renderFrame()
            if (shouldAnimate) startAnimation()
        } catch (error) {
            // Every start-up failure is terminal for THIS renderer. Attribute it (so hosts can
            // tell "no WebGPU in this browser" from "something broke"), release whatever was
            // half-built, and leave the canvas transparent — silently.
            const reason: GpuFailureReason = isGpuUnavailableError(error) ? error.reason : 'init-failed'
            isInitializing = false
            if (initializationAbortController === abort) initializationAbortController = null
            fail(reason, error)
            return
        } finally {
            isInitializing = false
            if (initializationAbortController === abort) initializationAbortController = null
        }
    }

    const cleanup = (): void => {
        if (initializationAbortController) initializationAbortController.abort()
        if (resizeObserver) {
            resizeObserver.disconnect()
            resizeObserver = null
        }
        if (intersectionObserver) {
            intersectionObserver.disconnect()
            intersectionObserver = null
        }
        if (globalEventUnregister) {
            globalEventUnregister()
            globalEventUnregister = null
        }
        if (unloadHandler && typeof window !== 'undefined') {
            window.removeEventListener('beforeunload', unloadHandler)
            unloadHandler = null
        }
        if (windowResizeHandler && typeof window !== 'undefined') {
            window.removeEventListener('resize', windowResizeHandler)
            windowResizeHandler = null
        }
        if (pendingRenderRAF !== null && typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(pendingRenderRAF)
            pendingRenderRAF = null
        }

        stopAnimation()
        detachUncapturedErrorHandler()
        unsubscribeNaturalSize?.()
        unsubscribeNaturalSize = null
        if (typeof document !== 'undefined' && document.fonts?.removeEventListener) {
            document.fonts.removeEventListener('loadingdone', onFontsLoadedLayout)
        }
        // Drop our device-loss subscription. The default device is shared page-wide and
        // outlives us, so leaving it attached would leak a callback per mount and later fire
        // it against this torn-down renderer.
        try {
            rootCtx?.release?.()
        } catch {
            /* ignore */
        }

        // Dispose every built composition (RTT textures + uniform buffers).
        pipelineCache.clear()
        for (const c of [...liveCompositions]) disposeComposition(c)
        boundComposition = null

        // Destroy shared textures/samplers.
        try {
            textureManager?.destroy()
        } catch {
            /* ignore */
        }

        // Deliberately NOT unconfiguring the canvas context here. A canvas that never drew
        // is already transparent (the fall-back-to-nothing case), and unconfiguring is the
        // one thing that would ALSO wipe a good frame — either mid-session, when a device
        // dies after rendering fine for a while, or as a visible flash across the
        // cleanup→re-initialize cycle a toneMapping change performs.

        // Only destroy the device/root if WE own it EXCLUSIVELY. In practice never today —
        // an injected device belongs to the caller, and the default device is shared by the
        // whole page (outliving components is what preserves the browser's pipeline cache
        // across route changes). Kept so the invariant is enforced rather than assumed.
        try {
            if (rootCtx?.createdDevice) root?.destroy()
        } catch {
            /* ignore */
        }

        nodes.clear()
        rootId = null
        parentToChildren.clear()
        pendingRegistrationQueue = []
        pendingNodeUpdates = []
        isRendererReady = false
        pendingResize = null
        isResizeScheduled = false

        canvasElement = null
        context = null
        textureManager = null
        dispatcher = null
        root = null
        rootCtx = null
        isInitialized = false
        isInitializing = false
        hasInitialDimensions = false
        initializationAbortController = null
        currentWidth = 0
        currentHeight = 0
        logicalWidth = 0
        logicalHeight = 0
        desiredHash = null
        structuralDirty = true
        isVisible = false
        shouldAnimate = true
        pointerX = -10
        pointerY = -10
        pointerActive = false
        pointerSeen = false
        mouseListenersNeeded = false // listeners were just unregistered — re-init must re-attach
        onReadyCallback = null
        hasEmittedReady = false
        firstRenderPending = false
        onDeviceLostCallback = null
        deviceLostFired = false
        uncapturedValidationErrors = 0
        consecutiveRenderErrors = 0
        // NOTE: `failureReason` / `onUnavailableCallback` deliberately survive cleanup — see
        // their declarations. A renderer that failed for environmental reasons must stay
        // failed, and the host must still hear about a failure raised during teardown.
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Performance stats
    // ────────────────────────────────────────────────────────────────────────────────────

    const getPerformanceStats = (): PerformanceStats => {
        const nodeCount = Math.max(0, nodes.size - 1) // exclude the invisible root
        const rttNodeCount = boundComposition ? boundComposition.ir.rttPasses.length : 0
        performanceTracker.updateNodeCounts(nodeCount, rttNodeCount)
        // Feed the info shape performanceTracker.getStats reads:
        //   drawCalls = pass count, shaderPrograms = live pipeline count, textureCount = tracker.
        const rendererInfo = {
            render: {calls: boundComposition ? boundComposition.passManager.passCount : 0},
            programs: {length: pipelineCache.size},
            memory: {textures: textureManager ? textureManager.textureCount : 0},
        }
        return performanceTracker.getStats(rendererInfo)
    }

    // ────────────────────────────────────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────────────────────────────────────

    return {
        initialize,
        cleanup,
        registerNode,
        removeNode,
        updateUniformValue,
        updateNodeMetadata,
        isInitialized: () => isInitialized,
        resize: (width: number, height: number): void => {
            if (!root) return
            updateRendererDimensions(width, height)
        },
        startAnimation,
        stopAnimation,
        renderAndWait,
        renderSyntheticFrame,
        getPerformanceStats,

        getLiveDriverValue: (nodeId: string, propName: string): {x: number; y: number} | null => {
            const state = nodes.get(nodeId)?.mouseDriverState?.[propName]
            if (!state || state.liveX === undefined || state.liveY === undefined) return null
            return {x: state.liveX, y: state.liveY}
        },
        getLiveScalarValue: (nodeId: string, propName: string): number | null => {
            const node = nodes.get(nodeId)
            if (!node) return null
            const driver = node.metadata.maps?.[propName]
            if (driver?.type === 'auto-animate') {
                const h = node._liveHandles?.[`_smoothed_${propName}`]
                const v = h instanceof FieldHandle ? (h.value as number) : undefined
                return typeof v === 'number' ? v : null
            }
            if (driver?.type === 'mouse') {
                const s = node.mouseDriverState[propName]
                return s?.liveScalar ?? null
            }
            return null
        },

        getNodeRegistry: () => ({nodes: new Map(nodes) as ReadonlyMap<string, GpuNodeInfo>}),

        /** Always 'webgpu' (type kept wide for source-compat with callers expecting a union). */
        getRendererType: (): 'webgpu' | 'webgl' | null => 'webgpu',

        /** The renderer context object: `{ device, adapter?, root }`. */
        getInternalRenderer: (): {device: GPUDevice; adapter?: GPUAdapter; root: TgpuRoot} | null =>
            root ? {device: root.device, adapter: rootCtx?.adapter, root} : null,

        setForceFullFrameRate: (enabled: boolean): boolean => {
            const previous = forceFullFrameRate
            forceFullFrameRate = enabled
            return previous
        },

        beginRecordingResolution: (targetWidth: number, targetHeight: number, ratio: number): (() => void) => {
            if (!root || targetWidth <= 0 || targetHeight <= 0) return () => {}
            const prevRatio = pixelRatio
            const prevWidth = currentWidth
            const prevHeight = currentHeight
            const prevLogicalWidth = logicalWidth
            const prevLogicalHeight = logicalHeight
            const prevLastRequestedWidth = lastRequestedWidth
            const prevLastRequestedHeight = lastRequestedHeight

            pixelRatio = ratio
            applyRendererDimensions(targetWidth, targetHeight, targetWidth, targetHeight)

            return () => {
                if (!root) return
                pixelRatio = prevRatio
                applyRendererDimensions(prevWidth, prevHeight, prevLogicalWidth, prevLogicalHeight)
                lastRequestedWidth = prevLastRequestedWidth
                lastRequestedHeight = prevLastRequestedHeight
            }
        },

        setResolutionScale: (scale: number): void => {
            if (!root) return
            const clampedScale = Math.max(0.05, Math.min(scale, 1))
            const targetRatio = Math.max(0.01, basePixelRatio() * clampedScale)
            if (Math.abs(pixelRatio - targetRatio) < 0.001) return
            pixelRatio = targetRatio
            const logicalW = logicalWidth || currentWidth
            const logicalH = logicalHeight || currentHeight
            if (logicalW <= 0 || logicalH <= 0) return
            const {width, height} = clampToTextureCap(logicalW, logicalH)
            applyRendererDimensions(width, height, logicalW, logicalH)
        },

        /**
         * Cap this renderer's ON-screen frame rate (`null`/0 restores the default
         * 60 FPS). The lever a multi-tile canvas pulls per tile: a frame that is
         * 80px on screen doesn't earn 60 renders a second. The RAF loop keeps
         * ticking; skipped frames cost one gate check and nothing else (no driver
         * update, no flush, no submit). Time stays correct across skips — delta
         * accumulates, and a shared time origin is wall-clock anyway. Caps are
         * clamped to 10 FPS minimum: below that the frame interval exceeds the
         * gate's 0.1s delta clamp and accumulated animation time would lose
         * elapsed wall-clock time on every rendered frame.
         */
        setFrameRateCap: (fps: number | null): void => {
            frameIntervalCap = fps && fps > 0 ? 1000 / Math.max(fps, 10) - 1 : 0
        },

        setTimeOrigin: (origin: number | null): void => {
            sharedTimeOrigin = origin
        },
        setOnReady: (callback: (() => void) | null): void => {
            onReadyCallback = callback
            // A late subscriber (after the first composed frame already rendered) fires next tick.
            if (callback && hasEmittedReady) queueMicrotask(callback)
        },
        setOnDeviceLost: (callback: ((reason: string) => void) | null): void => {
            onDeviceLostCallback = callback
        },

        /**
         * Subscribe to "this renderer can never draw" — WebGPU missing, no adapter, device
         * refused/lost, or the GPU proving unusable at runtime. Fires at most once. A late
         * subscriber (registered after the failure) is still notified, so hosts can attach
         * this after `initialize()` has already resolved.
         *
         * By the time it fires every GPU resource has been released and the frame loop is
         * stopped. This is the hook for swapping in a static fallback.
         */
        setOnUnavailable: (callback: ((reason: GpuFailureReason, error?: unknown) => void) | null): void => {
            onUnavailableCallback = callback
            if (callback && failureReason && !hasNotifiedUnavailable) {
                hasNotifiedUnavailable = true
                const reason = failureReason
                queueMicrotask(() => {
                    try {
                        callback(reason)
                    } catch (e) {
                        debugError('[gpu] onUnavailable callback threw:', e)
                    }
                })
            }
        },

        /** Why this renderer gave up, or `null` while it is healthy. */
        getFailureReason: (): GpuFailureReason | null => failureReason,

        // Testing-only — not for production use.
        __testing: {
            needsTransformation,
            findChildNodes,
            /** The per-node FieldInit set the store registers (props + synthetic `_`-fields). */
            buildFieldInits: (id: string): FieldInit[] | null => {
                const node = nodes.get(id)
                return node ? buildFieldInits(node) : null
            },
            /** Re-seed a node's persistent CPU driver state (the cache-hit rebind path). */
            ensureDriverState: (id: string): void => {
                const node = nodes.get(id)
                if (node) ensureDriverState(node)
            },
            /** Advance the CPU mouse/auto drivers one step without a device (driver test). */
            stepMouseDrivers: (deltaTime: number): void => {
                frameParams.deltaTime = deltaTime
                updateMouseDrivers()
            },
            /** Advance + read a node's `_animTime` accumulator without a device (driver test). */
            stepAnimatedTime: (deltaTime: number): void => {
                frameParams.deltaTime = deltaTime
                updateAnimatedTime()
            },
            getAnimatedTimeValue: (id: string): number | null => {
                const node = nodes.get(id)
                if (!node?.definition.animatedTime) return null
                return getAnimatedTimeState(animatedTimeKey(node)).value
            },
            // extraFields write/read on the node's LIVE handle (device-bound). writeExtraField
            // mirrors what params.setExtraField does at frame time; readExtraField observes it.
            writeExtraField: (id: string, name: string, value: number | number[]): void => {
                const node = nodes.get(id)
                if (node) writeExtraFieldValue(node, name, value)
            },
            readExtraField: (id: string, name: string): unknown => {
                const h = nodes.get(id)?._liveHandles?.[name]
                return h instanceof FieldHandle ? h.value : undefined
            },
            getNodeRegistry: () => ({nodes, rootId, parentToChildren}),
            getParentToChildren: () => parentToChildren,
            getPendingRegistrations: () => [...pendingRegistrationQueue],
            /** Passive read of the live frame-timing/pointer state (RAF-loop diagnosis). */
            getFrameDiagnostics: () => ({
                deltaTime: frameParams.deltaTime,
                pointerX,
                pointerY,
                pointerActive,
                globalElapsedTime,
                lastRenderTime,
                isVisible,
                forceFullFrameRate,
                sharedTimeOrigin,
                animating: frameLoop.running,
                mouseListenersNeeded,
            }),
            computeStructuralHash,
            isStructuralDirty: () => structuralDirty,
            clearStructuralDirty: () => {
                structuralDirty = false
            },
            /** Force the renderer "ready" WITHOUT a device, for registration/scheduling tests. */
            setTestReady: (dims?: {width: number; height: number}): void => {
                isInitialized = true
                isRendererReady = true
                if (dims) {
                    currentWidth = dims.width
                    currentHeight = dims.height
                    logicalWidth = dims.width
                    logicalHeight = dims.height
                    hasInitialDimensions = true
                }
                processQueuedRegistrations()
            },
        },
    }
}
