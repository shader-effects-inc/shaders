/**
 * `CompositionIR` → ordered GPU passes. Owns the RTT textures (create/resize/destroy,
 * Set-tracked) and executes a composition each frame: compute dispatch → RTT passes
 * (leaves→root) → final pass to the canvas context view.
 *
 * Bind groups (composer-pinned): uniforms = group 0 (shared, from the uniform store),
 * textures = group 1 (PER PASS — only the keys that pass samples, so an RTT pass never binds
 * its own output → no read/write hazard), samplers = group 2 (shared kit layout), external
 * textures = group 3 (per-frame re-import). Every registered RTT boundary is rendered
 * explicitly.
 *
 * GPU-free testable: all device interaction goes through injected `root` / `textureManager` /
 * `dispatcher` deps, which the tests mock (the way uniformStore/compute tests mock the device).
 */
import {common, type TgpuRoot, type TgpuBindGroup, type TgpuRenderPipeline} from 'typegpu'
import type {CompositionIR, FragmentSpec} from './composer'
import {SHARED_SAMPLER_LAYOUT} from './composer'
import type {TextureManager, RenderTexture, ExternalTextureBinding} from './textures'
import type {ComputeDispatcher} from './compute'
import {labelledTimer, type GpuPassTimer} from './gpuTimer'

/** The last pass draws every layer that isn't rendered to a texture of its own, so their cost is in it. */
const FINAL_PASS_LABEL = 'final composite (every layer drawn straight to the frame)'
import {debugError, debugWarn} from './support'

/** A color-attachment target: the canvas context view, or an RTT texture by key. */
type PassTarget = {kind: 'canvas'} | {kind: 'rtt'; key: string}

/** One executable pass: its pipeline + the bind groups it needs + its render target. */
interface ExecutablePass {
    fragment: FragmentSpec
    /** Names the pass in a GPU profile. */
    label?: string
    pipeline: TgpuRenderPipeline
    textureBindGroup?: TgpuBindGroup
    externalBinding?: ExternalTextureBinding
    target: PassTarget
    /** Latched so a "not-ready → skip" transition logs at most once per composition. */
    externalSkipLogged?: boolean
    /**
     * Media/compute read keys this pass samples (a subset of `fragment.reads` present in
     * `mediaResources`). A media shader swaps its backing texture at runtime (image loaded at native
     * size, text re-rastered, DOM resized), so before drawing we compare each key's current backing
     * texture to `mediaSnapshot` and rebuild `textureBindGroup` when one changed. Compute textures are
     * stable so their check is a harmless no-op.
     */
    mediaReads?: string[]
    /** The backing texture identity per media read the current `textureBindGroup` was built with. */
    mediaSnapshot?: Map<string, unknown>
}

export interface PassManagerOptions {
    /** Injected TextureManager (textures.ts). */
    textureManager: TextureManager
    /** Injected compute dispatcher (compute.ts). */
    dispatcher: ComputeDispatcher
    /** Canvas color format for the final pass target. Default the preferred canvas format. */
    canvasFormat?: GPUTextureFormat
    /** RTT target format. Default `rgba16float`. */
    rttFormat?: GPUTextureFormat
    /**
     * Device sampled-texture-per-stage ceiling (assert + log, never throw). Defaults to the
     * live `maxSampledTexturesPerShaderStage` limit, falling back to 16 — the WebGPU core
     * MINIMUM — when the device doesn't report one.
     */
    maxSampledTextures?: number
    /** Vertex entry for the fullscreen pass. Default `common.fullScreenTriangle`. */
    fullScreenTriangle?: unknown
}

export interface PassManager {
    /**
     * Build the executable passes for a composition: allocate RTT textures, create pipelines,
     * build per-pass bind groups. `uniformBindGroup` is the store's group-0 bind group;
     * `mediaResources` maps shader-owned media/compute texture keys → their bound resource.
     */
    setComposition(
        ir: CompositionIR,
        uniformBindGroup: TgpuBindGroup | undefined,
        size: {width: number; height: number},
        mediaResources?: Record<string, RenderTexture | {texture: unknown}>,
    ): void
    /** Execute the current composition: compute → RTT passes → final pass to `canvasView`. */
    render(canvasView: unknown, frameParams: unknown, afterCompute?: () => void, timer?: GpuPassTimer): void
    /**
     * Re-execute only the RTT passes (no compute, no final pass). `resize` recreates every RTT
     * texture zero-initialized, and `render` runs compute FIRST — so a compute step that reads a
     * child RTT (Glass frosted blur, kit/blur consumers) would read black for one frame after a
     * resize. The renderer calls this right after `resize` to repopulate the textures before the
     * next full frame samples them.
     */
    repaintRtt(): void
    /**
     * Resize all RTT textures to the new backing-buffer size. Returns true if any texture was
     * actually destroyed + recreated — the caller uses that to decide whether the warm-up
     * `repaintRtt` is needed at all.
     */
    resize(width: number, height: number): boolean
    /** Live RTT texture count (feeds performanceTracker.textureCount). */
    readonly textureCount: number
    /** Passes-per-frame (RTT + final) — feeds performanceTracker.drawCalls. */
    readonly passCount: number
    /** Destroy all RTT textures + drop pass state. */
    dispose(): void
}

const LABEL = 'passManager'

export function createPassManager(root: TgpuRoot, options: PassManagerOptions): PassManager {
    const {textureManager, dispatcher} = options
    const canvasFormat = options.canvasFormat ?? preferredCanvasFormat()
    const rttFormat = options.rttFormat ?? 'rgba16float'
    // Read the REAL device ceiling rather than assuming the spec minimum: hardware varies from
    // 16 (baseline) to 64+, so a hardcoded 16 both cried wolf on capable GPUs and told us
    // nothing useful about the one device that matters. 16 stays the fallback because it's the
    // core minimum — the safe assumption when a device reports nothing.
    const deviceMaxSampled = root.device?.limits?.maxSampledTexturesPerShaderStage
    const maxSampled = options.maxSampledTextures ?? (typeof deviceMaxSampled === 'number' && deviceMaxSampled > 0 ? deviceMaxSampled : 16)
    const vertex = options.fullScreenTriangle ?? common.fullScreenTriangle

    // RTT textures for the CURRENT composition, keyed + Set-tracked for lifecycle.
    const rttTextures = new Map<string, RenderTexture>()
    const liveTextures = new Set<RenderTexture>()

    let current: {
        ir: CompositionIR
        uniformBindGroup?: TgpuBindGroup
        samplerBindGroup?: TgpuBindGroup
        computeSteps: CompositionIR['computeSteps']
        rttPasses: ExecutablePass[]
        finalPass: ExecutablePass
        size: {width: number; height: number}
        /** Shader-owned media/compute resources — read live for the swap check (see execPass). */
        mediaResources?: Record<string, RenderTexture | {texture: unknown}>
    } | null = null

    // ── build the shared sampler bind group (group 2) ───────────────────────────────────
    function buildSamplerBindGroup(): TgpuBindGroup | undefined {
        const s = textureManager.samplers
        try {
            return root.createBindGroup(SHARED_SAMPLER_LAYOUT as never, {
                linearClamp: s.linearClamp,
                nearestClamp: s.nearestClamp,
                linearRepeat: s.linearRepeat,
                nearestRepeat: s.nearestRepeat,
            } as never) as TgpuBindGroup
        } catch (e) {
            debugError(`[${LABEL}] failed to build sampler bind group:`, e)
            return undefined
        }
    }

    // ── allocate / reuse RTT textures for a composition ─────────────────────────────────
    function allocateRttTextures(ir: CompositionIR, size: {width: number; height: number}): void {
        const needed = new Set<string>()
        for (const t of ir.textures) {
            if (t.kind !== 'rtt') continue
            needed.add(t.key)
            if (!rttTextures.has(t.key)) {
                const rt = textureManager.createRenderTexture({
                    width: size.width,
                    height: size.height,
                    format: rttFormat,
                    label: `rtt:${t.key}`,
                })
                rttTextures.set(t.key, rt)
                liveTextures.add(rt)
            } else {
                rttTextures.get(t.key)!.resize(size.width, size.height)
            }
        }
        // Drop RTT textures the new composition no longer uses.
        for (const [key, rt] of [...rttTextures]) {
            if (!needed.has(key)) {
                rt.destroy()
                liveTextures.delete(rt)
                rttTextures.delete(key)
            }
        }
    }

    // ── per-pass texture bind group (group 1) ───────────────────────────────────────────
    function buildTextureBindGroup(
        fragment: FragmentSpec,
        mediaResources?: Record<string, RenderTexture | {texture: unknown}>,
    ): TgpuBindGroup | undefined {
        if (!fragment.textureLayout || fragment.reads.length === 0) return undefined
        const entries: Record<string, unknown> = {}
        for (const key of fragment.reads) {
            const rt = rttTextures.get(key)
            const media = mediaResources?.[key]
            const resource = rt ?? media
            if (!resource) {
                debugError(`[${LABEL}] pass "${fragment.entry}" reads unknown texture "${key}"`)
                return undefined
            }
            entries[key] = (resource as {texture: unknown}).texture
        }
        // Assert the sampled-texture ceiling (log-only).
        if (fragment.reads.length > maxSampled) {
            debugWarn(
                `[${LABEL}] pass samples ${fragment.reads.length} textures (> device limit ~${maxSampled}); composition may fail on this device.`,
            )
        }
        try {
            return root.createBindGroup(fragment.textureLayout as never, entries as never) as TgpuBindGroup
        } catch (e) {
            debugError(`[${LABEL}] failed to build texture bind group:`, e)
            return undefined
        }
    }

    function buildPipeline(fragment: FragmentSpec, targetFormat: GPUTextureFormat): TgpuRenderPipeline {
        return root.createRenderPipeline({
            vertex: vertex as never,
            fragment: fragment.entry as never,
            targets: {format: targetFormat} as never,
        } as never) as unknown as TgpuRenderPipeline
    }

    // ── per-pass external-texture binding (group 3) ─────────────────────────────────────
    // A pass that samples an external (video/webcam) texture rebuilds its group-3
    // bind group EVERY frame — external textures expire after the frame/task, so `.update()`
    // re-imports the current source via `root.device.importExternalTexture` and returns a fresh
    // bind group (or null when no source is ready). Single-external passes (the common case —
    // VideoTexture yields one) go through `textureManager.createExternalTextureBinding` (textures.ts);
    // a pass that inlines several external textures imports them all into one group-3 group
    // here (e.g. WebcamTexture).
    function buildExternalBinding(
        fragment: FragmentSpec,
        externalSources: Map<string, () => unknown>,
    ): ExternalTextureBinding | undefined {
        if (!fragment.externalLayout || fragment.externalReads.length === 0) return undefined
        const keys = fragment.externalReads
        if (keys.length === 1) {
            const getSource = externalSources.get(keys[0])
            if (!getSource) {
                debugError(`[${LABEL}] external read "${keys[0]}" has no registered source`)
                return undefined
            }
            return textureManager.createExternalTextureBinding({
                layout: fragment.externalLayout,
                entryKey: keys[0],
                getSource: getSource as () => HTMLVideoElement | VideoFrame | null | undefined,
            })
        }
        // Multi-external pass: import every source into one bind group; null if ANY is unavailable.
        let current: TgpuBindGroup | null = null
        return {
            update() {
                const entries: Record<string, unknown> = {}
                for (const key of keys) {
                    const source = externalSources.get(key)?.()
                    if (!source) {
                        current = null
                        return null
                    }
                    // A source can transiently lose its backing resource — treat an import throw
                    // exactly like "not ready" (skip the pass this frame).
                    try {
                        entries[key] = root.device.importExternalTexture({source: source as HTMLVideoElement})
                    } catch {
                        current = null
                        return null
                    }
                }
                current = root.createBindGroup(fragment.externalLayout as never, entries as never) as TgpuBindGroup
                return current
            },
            get current() {
                return current
            },
            destroy() {
                current = null
            },
        }
    }

    function buildExecutablePass(
        fragment: FragmentSpec,
        target: PassTarget,
        mediaResources: Record<string, RenderTexture | {texture: unknown}> | undefined,
        externalSources: Map<string, () => unknown>,
    ): ExecutablePass {
        const targetFormat = target.kind === 'canvas' ? canvasFormat : rttFormat
        // Which of this pass's reads are shader-owned media/compute textures (the swappable set),
        // and the backing texture each is bound to right now — so execPass can rebuild on a swap.
        const mediaReads = fragment.reads.filter((k) => mediaResources?.[k])
        const mediaSnapshot = new Map<string, unknown>(
            mediaReads.map((k) => [k, (mediaResources![k] as {texture: unknown}).texture]),
        )
        return {
            fragment,
            pipeline: buildPipeline(fragment, targetFormat),
            textureBindGroup: buildTextureBindGroup(fragment, mediaResources),
            externalBinding: buildExternalBinding(fragment, externalSources),
            target,
            mediaReads,
            mediaSnapshot,
        }
    }

    function setComposition(
        ir: CompositionIR,
        uniformBindGroup: TgpuBindGroup | undefined,
        size: {width: number; height: number},
        mediaResources?: Record<string, RenderTexture | {texture: unknown}>,
    ): void {
        allocateRttTextures(ir, size)
        // A compute pass may read a composition RTT boundary as its input (Blur's
        // child). Those textures exist only now (just allocated), so let each compute node build
        // its input bind group here — resolving a texture key to the RTT texture (or a media/compute
        // texture). Runs on every recompose (setComposition), matching how fragment texture bind
        // groups are (re)built. See contract `GpuComputeNode.bindInputs`.
        for (const spec of ir.computeSteps) {
            spec.bindInputs?.((key) => rttTextures.get(key) ?? (mediaResources?.[key] as {texture: unknown} | undefined))
        }
        const samplerBindGroup = buildSamplerBindGroup()
        // Key each external (video/webcam) source by its texture key so a pass that
        // reads `ext.$.<key>` can re-import it every frame.
        const externalSources = new Map<string, () => unknown>()
        for (const e of ir.externalTextures) externalSources.set(e.key, e.getSource)
        const rttPasses = ir.rttPasses.map((p) => ({
            ...buildExecutablePass(p.fragment, {kind: 'rtt', key: p.textureKey}, mediaResources, externalSources),
            label: p.label,
        }))
        const finalPass = buildExecutablePass(ir.finalPass, {kind: 'canvas'}, mediaResources, externalSources)
        current = {ir, uniformBindGroup, samplerBindGroup, computeSteps: ir.computeSteps, rttPasses, finalPass, size, mediaResources}
    }

    // ── execute one pass ────────────────────────────────────────────────────────────────
    function execPass(pass: ExecutablePass, view: unknown, timer?: GpuPassTimer, label?: string): void {
        let p = pass.pipeline
        if (current?.uniformBindGroup) p = p.with(current.uniformBindGroup)
        // Rebuild the group-1 bind group if a media shader swapped its backing texture since it
        // was built (ImageTexture load / Text re-raster / HTMLInCanvas resize). Cheap — a bind-group
        // rebuild, no pipeline rebuild, no recompose. Detected by backing-texture identity change.
        if (pass.mediaReads && pass.mediaReads.length > 0 && current?.mediaResources) {
            let swapped = false
            for (const key of pass.mediaReads) {
                const tex = (current.mediaResources[key] as {texture: unknown}).texture
                if (tex !== pass.mediaSnapshot!.get(key)) swapped = true
            }
            if (swapped) {
                pass.textureBindGroup = buildTextureBindGroup(pass.fragment, current.mediaResources)
                for (const key of pass.mediaReads) {
                    pass.mediaSnapshot!.set(key, (current.mediaResources[key] as {texture: unknown}).texture)
                }
            }
        }
        if (pass.textureBindGroup) p = p.with(pass.textureBindGroup)
        if (pass.fragment.usesSamplers && current?.samplerBindGroup) p = p.with(current.samplerBindGroup)
        // A pass that samples an external texture needs its group-3 binding rebuilt
        // this frame. If no source is ready (getSource → null: video still loading / paused), SKIP
        // the draw — the pipeline layout requires group 3, so drawing without it is invalid. Logged
        // once per composition to avoid per-frame spam; the pass renders normally once a frame lands.
        if (pass.fragment.externalReads.length > 0) {
            const bg = pass.externalBinding?.update() ?? null
            if (!bg) {
                if (!pass.externalSkipLogged) {
                    debugWarn(`[${LABEL}] external texture not ready — skipping pass (${pass.fragment.externalReads.join(', ')})`)
                    pass.externalSkipLogged = true
                }
                return
            }
            p = p.with(bg)
        }
        const writes = timer?.next(label)
        if (writes) p = p.withTimestampWrites(writes)
        p.withColorAttachment({view} as never).draw(3)
    }

    function render(canvasView: unknown, frameParams: unknown, afterCompute?: () => void, timer?: GpuPassTimer): void {
        if (!current) return
        // 1. Compute passes (registry order).
        for (const step of current.computeSteps) {
            const steps = step.getComputeNodes(frameParams)
            if (steps && steps.length) dispatcher.dispatch(steps, labelledTimer(timer, step.label ?? 'compute'))
        }
        // Uniform patches written DURING compute collection (a compute node's setExtraField —
        // e.g. the volumetric field's _vf* sample-domain params on an activeRes bucket crossing)
        // land after the frame's store flush, so without a second flush here the fragment samples
        // the freshly re-marched field with LAST frame's domain params — a one-frame wrong-scale
        // flash on every bucket crossing (scale drags cross buckets constantly). The flush is
        // dirty-coalesced, so this is a no-op on frames where collection patched nothing.
        afterCompute?.()
        // 2. RTT passes (leaves→root — the IR order is a valid topological order).
        repaintRtt(timer)
        // 3. Final pass → canvas.
        execPass(current.finalPass, canvasView, timer, FINAL_PASS_LABEL)
    }

    function repaintRtt(timer?: GpuPassTimer): void {
        if (!current) return
        for (const pass of current.rttPasses) {
            const rt = rttTextures.get((pass.target as {key: string}).key)
            if (!rt) continue
            execPass(pass, renderView(rt), timer, `${pass.label ?? 'layer'} · render to texture`)
        }
    }

    function renderView(rt: RenderTexture): unknown {
        // A raw render view over the RTT texture (RENDER_ATTACHMENT usage was flagged at
        // creation). Kept behind this helper so tests can mock `root.unwrap`.
        return root.unwrap(rt.texture).createView()
    }

    // Rebuild every bind group that references an RTT texture after a resize.
    // `buildTextureBindGroup` captures the RTT/media texture identities at build time; `rt.resize`
    // below destroys + recreates the backing GPU texture (a NEW identity), so a pass's cached
    // group-1 bind group — and any compute step's input bind group — now points at a DESTROYED
    // texture. Reusing it in the next submit is exactly the "Destroyed texture used in a submit"
    // validation error, and because WebGPU drops the whole command buffer on validation failure the
    // frame goes blank. Mirror setComposition's post-allocate rebuild so future submits only ever
    // bind live textures. Bind-group rebuild only: pipelines (format/entry unchanged) and external
    // bindings (rebuilt every frame in execPass) don't need it.
    function rebindPassTextures(pass: ExecutablePass): void {
        if (!pass.fragment.textureLayout || pass.fragment.reads.length === 0) return
        pass.textureBindGroup = buildTextureBindGroup(pass.fragment, current?.mediaResources)
        if (pass.mediaReads && pass.mediaSnapshot && current?.mediaResources) {
            for (const key of pass.mediaReads) {
                pass.mediaSnapshot.set(key, (current.mediaResources[key] as {texture: unknown}).texture)
            }
        }
    }

    function resize(width: number, height: number): boolean {
        let recreated = false
        for (const rt of rttTextures.values()) {
            if (rt.resize(width, height)) recreated = true
        }
        if (!current) return recreated
        current.size = {width, height}
        // Nothing was destroyed + recreated, so no cached bind group can be stale.
        if (!recreated) return false
        // A compute pass may read a composition RTT boundary as input (Blur's child); re-resolve
        // those input bind groups against the recreated textures — same call setComposition makes.
        for (const spec of current.computeSteps) {
            spec.bindInputs?.((key) => rttTextures.get(key) ?? (current!.mediaResources?.[key] as {texture: unknown} | undefined))
        }
        for (const pass of current.rttPasses) rebindPassTextures(pass)
        rebindPassTextures(current.finalPass)
        return true
    }

    return {
        setComposition,
        render,
        repaintRtt,
        resize,
        get textureCount() {
            return liveTextures.size
        },
        get passCount() {
            return current ? current.rttPasses.length + 1 : 0
        },
        dispose() {
            for (const rt of [...liveTextures]) rt.destroy()
            liveTextures.clear()
            rttTextures.clear()
            current = null
        },
    }
}

function preferredCanvasFormat(): GPUTextureFormat {
    if (typeof navigator !== 'undefined' && navigator.gpu) return navigator.gpu.getPreferredCanvasFormat()
    return 'bgra8unorm'
}
