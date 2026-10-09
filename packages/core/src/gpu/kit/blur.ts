/**
 * GPU compute-based separable Gaussian blur.
 *
 * Compute resolution is decoupled from canvas resolution: storage textures allocate once at
 * a fixed compute size and never reallocate on resize. The fragment shader bilinear-samples
 * the result at canvas resolution. Resize-time work is one small uniform write (input canvas
 * dimensions) — no pipeline rebuild, no texture realloc, no main-thread freeze regardless of
 * resize speed or scale.
 *
 * This mirrors how real-time bloom shaders in games operate (typically at 0.5× canvas
 * resolution) — the blur smooths over any resolution loss in its own output. Sharp content
 * stays at canvas resolution because the fragment shader composites the canvas-resolution
 * source with the compute-resolution blur.
 *
 * Shared utility used by Glass, Blur, ProgressiveBlur, TiltShift, ChannelBlur, and Glow.
 *
 * ── Implementation notes ──────────────────────────────────────────────────────────────────
 * • 2D guarded dispatch. Dispatch a 2D `(computeWidth, computeHeight)` grid ([16,16] workgroups)
 *   so `(cx, cy)` arrive as clean u32 thread coordinates. A flat `computeWidth*computeHeight`
 *   count with `(cx, cy)` reconstructed via `idx % cw` / `idx / cw` is unsafe because TGSL always
 *   transpiles `/` to FLOAT division. No manual `If(cy < ch)` guard — the guarded pipeline
 *   provides the bounds check.
 * • Weight arrays are READONLY STORAGE buffers, not uniform. WGSL forbids uniform arrays with
 *   element stride < 16 bytes (`array<f32, N>` has stride 4); TypeGPU emits `var<uniform>
 *   array<f32,N>` as-is, which the device rejects at pipeline creation. Storage arrays are
 *   tightly packed and runtime-indexable.
 * • The truncation/renormalization math (`buildTruncatedWeights`) is a pure, exported helper so
 *   it can be golden-tested on the CPU; the per-frame `updateWeights` closure calls it.
 * • Input texture reaches the kernel as a bind-group TEXTURE layout entry (modern
 *   `{ texture: d.texture2d(d.f32) }` form — the legacy `{ texture: 'float' }` string form
 *   collapses the layout's `Entries` generic to the default and types `layout.$.*` as `unknown`),
 *   read via `std.textureLoad`, NOT a slot or fn argument. The layout is a pure schema so the
 *   kernels resolve GPU-free; only the concrete texture view is runtime.
 * • STORAGE-texture reads take NO mip-level arg: `std.textureLoad(storageTex, coords)` (2 args),
 *   unlike SAMPLED-texture reads `std.textureLoad(sampledTex, coords, level)` (3 args). Passing a
 *   level to a storage read emits invalid WGSL (`textureLoad(texture_storage_2d, coords, 0)`) that
 *   a device rejects.
 */
import tgpu, {type TgpuRoot, type TgpuTexture, type TgpuTextureView, type TgpuBindGroupLayout, type TgpuFn} from 'typegpu'
import * as d from 'typegpu/data'
import * as std from 'typegpu/std'
import {createGuardedCompute, type ComputeStep, type KitComputePipeline} from '../compute'
import {call, ZERO} from '../composer'
import type {Expr, GpuComputeNode, GpuFragmentParams, GpuMapInfo, GpuMapWindow, KitTexture} from '../contract'
import {unpremultiplyAlpha} from './blend'
import * as constants from './constants'

/** Degrees → radians (module const; `Math.*` inside a `'use gpu'` body is unproven — kit convention). */
const DEG_TO_RAD = constants.DEG_TO_RAD

const DEFAULT_HALF_KERNEL = 24

/** Default compute resolution. ~655k pixels — comparable to 0.5× of a 1080p canvas.
 *  Memory per RGBA32F texture: ~10 MB (~5 MB at rgba16float). Heavy presets (Glow with 4
 *  storage textures) total ~30 MB regardless of canvas size. Callers can override for
 *  memory-tight scenarios or higher quality. */
export const DEFAULT_COMPUTE_WIDTH = 1024
export const DEFAULT_COMPUTE_HEIGHT = 640

const PASSTHROUGH_THRESHOLD = 0.5

/** Storage format of the intermediate (H-pass output) — read only via textureLoad. 16-bit float:
 *  the sources being blurred are rgba16float RTTs already, so a 32-bit intermediate held no extra
 *  information and doubled the read/write bandwidth of the heaviest pass (the V pass reads it
 *  49× per pixel). fp16 still holds HDR bloom values far above 1.0. */
const INTERMEDIATE_FORMAT = 'rgba16float' as const
/** Storage format of the final output — also `sampled` so the fragment can bilinear-sample it.
 *  rgba32float is NOT filterable without float32-filterable; rgba16float is filterable by
 *  default and still holds HDR values well above 1.0. */
const OUTPUT_FORMAT = 'rgba16float' as const
/** Storage format of the variable-blur radius map (caller-written). */
export const BLUR_MAP_FORMAT = 'rgba32float' as const

/**
 * Bindable source-texture forms accepted by the blur (the RTT being blurred). A `TgpuTexture`
 * must carry `sampled` usage; callers may also pass a KitTexture's underlying view.
 */
export type BlurInputTexture = TgpuTextureView | TgpuTexture | GPUTextureView

/**
 * Full symmetric Gaussian weights over the whole kernel (no truncation), normalized to 1.
 * Used by the variable-radius blur, whose per-pixel tap spread is scaled at sample time so the
 * weights describe SHAPE only.
 */
export function buildFixedWeights(halfKernel: number): number[] {
    const sigma = halfKernel / 3
    const size = halfKernel * 2 + 1
    const weights: number[] = []
    let sum = 0
    for (let i = -halfKernel; i <= halfKernel; i++) {
        const w = Math.exp(-(i * i) / (2 * sigma * sigma))
        weights.push(w)
        sum += w
    }
    for (let i = 0; i < size; i++) weights[i] /= sum
    return weights
}

/**
 * Truncated + renormalized Gaussian for one pass. Taps beyond ~3σ contribute nothing visible,
 * so the active half-kernel is `clamp(ceil(3σ), 1, halfKernel)`; weights outside it are zero and
 * the remaining support is renormalized to sum to 1. Returns the full-length weight array (zeros
 * padded outside the active window) plus the active half so the kernel loop walks only that far.
 * Pulled out of the per-frame `updateWeights` closure so it can be golden-tested on the CPU.
 */
export function buildTruncatedWeights(halfKernel: number, sigma: number): {weights: number[]; activeHalf: number} {
    const size = halfKernel * 2 + 1
    const activeHalf = Math.max(1, Math.min(halfKernel, Math.ceil(sigma * 3)))
    const weights = new Array<number>(size).fill(0)
    let total = 0
    for (let i = -halfKernel; i <= halfKernel; i++) {
        if (Math.abs(i) <= activeHalf) {
            const w = Math.exp(-(i * i) / (2 * sigma * sigma))
            weights[i + halfKernel] = w
            total += w
        }
    }
    for (let i = 0; i < size; i++) weights[i] /= total
    return {weights, activeHalf}
}

export interface GaussianBlurComputeResult {
    /**
     * Ordered compute steps: [horizontal pass, vertical pass]. Dispatch via the compute
     * dispatcher. The horizontal pass reads the input texture; when the input is bound late
     * (via `setInputTexture` — see blur-run note below) this array's H entry is swapped in place,
     * so a caller that holds this array reference always dispatches the currently-bound H pass.
     */
    computeSteps: ComputeStep[]
    /** rgba16float, storage + sampled. The buffer consumers bilinear-sample at canvas resolution. */
    outputTexture: TgpuTexture
    updateRadius: (pixelRadius: number) => void
    /** Fixed compute resolution — exposed so callers know the buffer's pixel size. */
    computeWidth: number
    computeHeight: number
    /** Update input-canvas dimensions on resize. Doesn't trigger any rebuild. */
    setInputDimensions: (width: number, height: number) => void
    /**
     * blur-run: (re)bind the source texture the H pass reads. Needed when the input is an RTT
     * boundary the composition renders — the pass manager allocates that texture AFTER
     * composition, so the consumer defers the input and binds it here once it exists (and rebinds
     * on recompose). Rebuilds only the H bind group; the V pass is input-independent.
     */
    setInputTexture: (input: BlurInputTexture) => void
}

export interface VariableBlurComputeResult {
    computeSteps: ComputeStep[]
    outputTexture: TgpuTexture
    /**
     * rgba32float, storage. Stores the desired blur radius (input/canvas pixels) per compute
     * pixel in its .r channel. The CALLER fills it in their own compute pass by binding it to a
     * `{ storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only') }` layout entry.
     */
    blurMapTexture: TgpuTexture
    computeWidth: number
    computeHeight: number
    setInputDimensions: (width: number, height: number) => void
    /** blur-run: (re)bind the source texture the H pass reads (see GaussianBlurComputeResult). */
    setInputTexture: (input: BlurInputTexture) => void
    /**
     * Per-frame detail dials: `jitter` (0–1) dithers the tap comb; `dispersion` (signed) is the
     * per-channel radius spread, read only by a graph built with `chromatic: true`. Patches the
     * uniforms only when a value changes.
     */
    setDetail: (jitter: number, dispersion: number) => void
}

// ── H-pass uniform params (fixed blur): active half-kernel + input canvas dims. ───────────
function fixedHParams() {
    return d.struct({activeHalf: d.i32, inputWidth: d.f32, inputHeight: d.f32})
}
// ── V-pass uniform params (fixed blur): active half-kernel only (works in compute space). ──
function fixedVParams() {
    return d.struct({activeHalf: d.i32})
}

/**
 * GPU-free construction of the fixed-blur node graph: bind-group layouts + kernel fns. No
 * device is touched, so the kernels can be `tgpu.resolve`d for WGSL snapshots. `createGaussian
 * BlurCompute` calls this and then allocates the resources + pipelines.
 */
export function buildFixedBlurGraph(halfKernel: number, computeWidth: number, computeHeight: number) {
    const KERNEL_SIZE = halfKernel * 2 + 1
    const HALF_KERNEL = halfKernel
    const HParams = fixedHParams()
    const VParams = fixedVParams()

    const hLayout = tgpu.bindGroupLayout({
        input: {texture: d.texture2d(d.f32)},
        intermediate: {storageTexture: d.textureStorage2d(INTERMEDIATE_FORMAT, 'write-only')},
        weights: {storage: d.arrayOf(d.f32, KERNEL_SIZE), access: 'readonly'},
        params: {uniform: HParams},
    })
    const vLayout = tgpu.bindGroupLayout({
        src: {storageTexture: d.textureStorage2d(INTERMEDIATE_FORMAT, 'read-only')},
        output: {storageTexture: d.textureStorage2d(OUTPUT_FORMAT, 'write-only')},
        weights: {storage: d.arrayOf(d.f32, KERNEL_SIZE), access: 'readonly'},
        params: {uniform: VParams},
    })

    // ── Horizontal pass ──────────────────────────────────────────────────────────────────
    // Maps this compute pixel's center into input coordinate space (uniform-backed input dims,
    // so resize pulls through without recompile), then walks the active half-kernel along X in
    // input pixels, writing the intermediate at the compute coord.
    const kernelH = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = hLayout.$.params
        const inputCx = (d.f32(cx) + 0.5) * p.inputWidth / computeWidth - 0.5
        const inputCy = (d.f32(cy) + 0.5) * p.inputHeight / computeHeight - 0.5
        const inputWm1 = d.i32(p.inputWidth - 1)
        const inputHm1 = d.i32(p.inputHeight - 1)
        const inputCxI = d.i32(std.round(inputCx))
        const inputCyI = std.clamp(d.i32(std.round(inputCy)), 0, inputHm1)
        const ah = p.activeHalf
        const count = ah * 2 + 1
        let sum = d.vec4f(0, 0, 0, 0)
        for (let i = 0; i < count; i++) {
            const off = i - ah
            const sampleX = std.clamp(inputCxI + off, 0, inputWm1)
            const texel = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(sampleX), d.u32(inputCyI)), 0)
            sum = sum.add(texel.mul(hLayout.$.weights[off + HALF_KERNEL]))
        }
        std.textureStore(hLayout.$.intermediate, d.vec2u(cx, cy), sum)
    }).$name('gaussianBlurH')

    // ── Vertical pass ────────────────────────────────────────────────────────────────────
    // Reads the intermediate at compute coord; each tap walks 1 compute-Y pixel. The V weights'
    // sigma is pre-scaled by 1/scaleY on the CPU (see updateWeights) so the visible blur is
    // symmetric with the H pass. computeHeight-1 is a compile-time clamp bound.
    const kernelV = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const av = vLayout.$.params.activeHalf
        const count = av * 2 + 1
        const cyi = d.i32(cy)
        let sum = d.vec4f(0, 0, 0, 0)
        for (let i = 0; i < count; i++) {
            const off = i - av
            const sampleY = std.clamp(cyi + off, 0, computeHeight - 1)
            const texel = std.textureLoad(vLayout.$.src, d.vec2u(cx, d.u32(sampleY)))
            sum = sum.add(texel.mul(vLayout.$.weights[off + HALF_KERNEL]))
        }
        std.textureStore(vLayout.$.output, d.vec2u(cx, cy), sum)
    }).$name('gaussianBlurV')

    return {hLayout, vLayout, kernelH, kernelV, HParams, VParams, KERNEL_SIZE}
}

/**
 * 2-pass separable Gaussian blur backed by compute shaders, fixed-resolution output. The
 * fragment shader samples the result at canvas-resolution screenUV (bilinear handles up/downscale).
 *
 * @param root - the TypeGPU root (device).
 * @param inputTexture - source texture view to blur (typically a `convertToTexture` RTT).
 * @param inputWidth/inputHeight - source dimensions (canvas size).
 * @param onCleanup - cleanup callback registration.
 * @param halfKernel - half-kernel width (default 24 → 49 taps per pass).
 * @param computeWidth/computeHeight - fixed internal compute resolution (never reallocated).
 */
export function createGaussianBlurCompute(
    root: TgpuRoot,
    inputTexture: BlurInputTexture | null,
    inputWidth: number,
    inputHeight: number,
    onCleanup: (cb: () => void) => void,
    halfKernel: number = DEFAULT_HALF_KERNEL,
    computeWidth: number = DEFAULT_COMPUTE_WIDTH,
    computeHeight: number = DEFAULT_COMPUTE_HEIGHT,
): GaussianBlurComputeResult {
    const HALF_KERNEL = halfKernel
    const graph = buildFixedBlurGraph(halfKernel, computeWidth, computeHeight)
    const {hLayout, vLayout, kernelH, kernelV, KERNEL_SIZE} = graph

    // Storage textures at fixed compute resolution. Never reallocated on resize.
    const intermediateTex = root.createTexture({size: [computeWidth, computeHeight], format: INTERMEDIATE_FORMAT}).$usage('storage')
    const outputTex = root.createTexture({size: [computeWidth, computeHeight], format: OUTPUT_FORMAT}).$usage('storage', 'sampled')

    // Two weight arrays — H at input-pixel tap spacing, V at compute-pixel spacing (sigma scaled
    // by 1/scaleY). Readonly storage (see header note on the uniform-array WGSL restriction).
    const weightsHBuf = root.createBuffer(d.arrayOf(d.f32, KERNEL_SIZE)).$usage('storage')
    const weightsVBuf = root.createBuffer(d.arrayOf(d.f32, KERNEL_SIZE)).$usage('storage')
    const paramsH = root.createUniform(graph.HParams)
    const paramsV = root.createUniform(graph.VParams)

    onCleanup(() => {
        intermediateTex.destroy()
        outputTex.destroy()
    })

    const vBindGroup = root.createBindGroup(vLayout, {
        src: intermediateTex,
        output: outputTex,
        weights: weightsVBuf,
        params: paramsV.buffer,
    })

    // The V pass is input-independent → bound now. The H pass reads the source texture, which may
    // be an RTT boundary allocated AFTER composition (blur-run late input, see setInputTexture):
    // build its guarded pipeline once, then swap in a bound handle when the input is (re)bound.
    const hPipeline = createGuardedCompute(root, (cx: number, cy: number) => {
        'use gpu'
        kernelH(cx, cy)
    }, {size: [computeWidth, computeHeight]})
    const vPass = createGuardedCompute(root, (cx: number, cy: number) => {
        'use gpu'
        kernelV(cx, cy)
    }, {size: [computeWidth, computeHeight], bindGroup: vBindGroup})

    const computeSteps: ComputeStep[] = [hPipeline, vPass]
    function setInputTexture(input: BlurInputTexture): void {
        const hBindGroup = root.createBindGroup(hLayout, {
            input: input as never,
            intermediate: intermediateTex,
            weights: weightsHBuf,
            params: paramsH.buffer,
        })
        computeSteps[0] = hPipeline.with(hBindGroup)
    }
    if (inputTexture != null) setInputTexture(inputTexture)

    // scaleY converts blur radius (input pixels) → compute-Y pixels for the V pass.
    let scaleY = inputHeight / computeHeight
    let curInputWidth = inputWidth
    let curInputHeight = inputHeight
    let lastRadius = -1

    // Per-frame weight update. Caller passes pixelRadius in INPUT pixels; the H pass sigma is in
    // input pixels and the V pass sigma is inputSigma / scaleY so the visible blur is symmetric.
    function updateWeights() {
        const pixelRadius = lastRadius < 0 ? 4 : lastRadius
        const sigmaH = Math.max(pixelRadius * 0.5, 0.001)
        const sigmaV = Math.max(sigmaH / scaleY, 0.001)
        const h = buildTruncatedWeights(HALF_KERNEL, sigmaH)
        const v = buildTruncatedWeights(HALF_KERNEL, sigmaV)
        weightsHBuf.write(h.weights)
        weightsVBuf.write(v.weights)
        paramsH.write({activeHalf: h.activeHalf, inputWidth: curInputWidth, inputHeight: curInputHeight})
        paramsV.write({activeHalf: v.activeHalf})
    }

    function updateRadius(pixelRadius: number) {
        if (Math.abs(pixelRadius - lastRadius) < 0.01) return
        lastRadius = pixelRadius
        updateWeights()
    }

    updateRadius(4)

    return {
        computeSteps,
        outputTexture: outputTex,
        updateRadius,
        computeWidth,
        computeHeight,
        setInputTexture,
        setInputDimensions: (newW: number, newH: number) => {
            curInputWidth = newW
            curInputHeight = newH
            paramsH.patch({inputWidth: newW, inputHeight: newH})
            const newScaleY = newH / computeHeight
            if (Math.abs(newScaleY - scaleY) > 0.001) {
                scaleY = newScaleY
                // Recompute V weights to keep the blur symmetric as scaleY changes.
                updateWeights()
            }
        },
    }
}

// ── Variable-blur params. H needs input dims; V needs scaleY (both uniform-backed). Both carry
// the detail dials: `jitter` (0–1, per-pixel tap-comb phase) and `dispersion` (signed, the
// per-channel radius spread the chromatic variant reads). ────
function variableHParams() {
    return d.struct({inputWidth: d.f32, inputHeight: d.f32, jitter: d.f32, dispersion: d.f32})
}
function variableVParams() {
    return d.struct({scaleY: d.f32, jitter: d.f32, dispersion: d.f32})
}

export interface VariableBlurGraphOptions {
    /**
     * Blur red, green and blue by different radii (`radius × (1 ± dispersion)`) for a defocus
     * colour fringe. A build-time variant — three loads per tap instead of one — so callers only
     * enable it while the dispersion is non-zero.
     */
    chromatic?: boolean
}

/**
 * GPU-free construction of the variable-blur node graph. The tap comb is dithered by
 * interleaved gradient noise (Jimenez 2014) scaled by the runtime `jitter` uniform — at 0 the
 * offsets are exactly the undithered comb. `options.chromatic` selects the three-radius kernel
 * bodies (comptime), so enabling it changes the emitted WGSL.
 */
export function buildVariableBlurGraph(
    halfKernel: number,
    computeWidth: number,
    computeHeight: number,
    options: VariableBlurGraphOptions = {},
) {
    const KERNEL_SIZE = halfKernel * 2 + 1
    const HALF_KERNEL = halfKernel
    const invHalf = 1.0 / HALF_KERNEL
    const chromatic = options.chromatic === true
    const HParams = variableHParams()
    const VParams = variableVParams()

    const hLayout = tgpu.bindGroupLayout({
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'read-only')},
        input: {texture: d.texture2d(d.f32)},
        intermediate: {storageTexture: d.textureStorage2d(INTERMEDIATE_FORMAT, 'write-only')},
        weights: {storage: d.arrayOf(d.f32, KERNEL_SIZE), access: 'readonly'},
        params: {uniform: HParams},
    })
    const vLayout = tgpu.bindGroupLayout({
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'read-only')},
        src: {storageTexture: d.textureStorage2d(INTERMEDIATE_FORMAT, 'read-only')},
        output: {storageTexture: d.textureStorage2d(OUTPUT_FORMAT, 'write-only')},
        weights: {storage: d.arrayOf(d.f32, KERNEL_SIZE), access: 'readonly'},
        params: {uniform: VParams},
    })

    // ── Horizontal pass ──────────────────────────────────────────────────────────────────
    // Per-pixel blur radius from the map; tap offsets in input pixels = (i + jitter)/halfKernel ×
    // radius. Below the passthrough threshold the pixel is a sharp single-tap copy.
    const kernelHPlain = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = hLayout.$.params
        const blurRadius = std.textureLoad(hLayout.$.blurMap, d.vec2u(cx, cy)).r
        const inputCx = (d.f32(cx) + 0.5) * p.inputWidth / computeWidth - 0.5
        const inputCy = (d.f32(cy) + 0.5) * p.inputHeight / computeHeight - 0.5
        const inputWm1 = d.i32(p.inputWidth - 1)
        const inputHm1 = d.i32(p.inputHeight - 1)
        const inputCyI = std.clamp(d.i32(std.round(inputCy)), 0, inputHm1)
        if (blurRadius < PASSTHROUGH_THRESHOLD) {
            const cxClamp = std.clamp(d.i32(std.round(inputCx)), 0, inputWm1)
            const texel = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(cxClamp), d.u32(inputCyI)), 0)
            std.textureStore(hLayout.$.intermediate, d.vec2u(cx, cy), texel)
        } else {
            // Per-pixel comb shift in [-0.5, 0.5) tap spacings × jitter (interleaved gradient
            // noise, Jimenez 2014) — decorrelates the fixed tap comb's moiré into fine noise.
            const jitterH = (std.fract(52.9829189 * std.fract(d.f32(cx) * 0.06711056 + d.f32(cy) * 0.00583715)) - 0.5) * p.jitter
            let sum = d.vec4f(0, 0, 0, 0)
            for (let i = -HALF_KERNEL; i <= HALF_KERNEL; i++) {
                const offset = (d.f32(i) + jitterH) * invHalf * blurRadius
                const sampleX = std.clamp(d.i32(std.round(inputCx + offset)), 0, inputWm1)
                const texel = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(sampleX), d.u32(inputCyI)), 0)
                sum = sum.add(texel.mul(hLayout.$.weights[i + HALF_KERNEL]))
            }
            std.textureStore(hLayout.$.intermediate, d.vec2u(cx, cy), sum)
        }
    }).$name('variableBlurH')

    // Chromatic variant: red taps spread by (1 + dispersion), blue by (1 − dispersion), green
    // (and alpha) by the plain radius — three loads per tap, recombined per channel.
    const kernelHChromatic = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = hLayout.$.params
        const blurRadius = std.textureLoad(hLayout.$.blurMap, d.vec2u(cx, cy)).r
        const inputCx = (d.f32(cx) + 0.5) * p.inputWidth / computeWidth - 0.5
        const inputCy = (d.f32(cy) + 0.5) * p.inputHeight / computeHeight - 0.5
        const inputWm1 = d.i32(p.inputWidth - 1)
        const inputHm1 = d.i32(p.inputHeight - 1)
        const inputCyI = std.clamp(d.i32(std.round(inputCy)), 0, inputHm1)
        if (blurRadius < PASSTHROUGH_THRESHOLD) {
            const cxClamp = std.clamp(d.i32(std.round(inputCx)), 0, inputWm1)
            const texel = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(cxClamp), d.u32(inputCyI)), 0)
            std.textureStore(hLayout.$.intermediate, d.vec2u(cx, cy), texel)
        } else {
            const jitterH = (std.fract(52.9829189 * std.fract(d.f32(cx) * 0.06711056 + d.f32(cy) * 0.00583715)) - 0.5) * p.jitter
            const radiusR = blurRadius * (1.0 + p.dispersion)
            const radiusB = blurRadius * (1.0 - p.dispersion)
            let sum = d.vec4f(0, 0, 0, 0)
            for (let i = -HALF_KERNEL; i <= HALF_KERNEL; i++) {
                const t = (d.f32(i) + jitterH) * invHalf
                const xG = std.clamp(d.i32(std.round(inputCx + t * blurRadius)), 0, inputWm1)
                const xR = std.clamp(d.i32(std.round(inputCx + t * radiusR)), 0, inputWm1)
                const xB = std.clamp(d.i32(std.round(inputCx + t * radiusB)), 0, inputWm1)
                const texG = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(xG), d.u32(inputCyI)), 0)
                const texR = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(xR), d.u32(inputCyI)), 0)
                const texB = std.textureLoad(hLayout.$.input, d.vec2u(d.u32(xB), d.u32(inputCyI)), 0)
                const texel = d.vec4f(texR.x, texG.y, texB.z, texG.w)
                sum = sum.add(texel.mul(hLayout.$.weights[i + HALF_KERNEL]))
            }
            std.textureStore(hLayout.$.intermediate, d.vec2u(cx, cy), sum)
        }
    }).$name('variableBlurHChromatic')

    // ── Vertical pass ────────────────────────────────────────────────────────────────────
    // Tap offsets in compute-Y pixels (= radiusInput / scaleY). Different noise phase than H so
    // the two combs don't correlate.
    const kernelVPlain = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const blurRadius = std.textureLoad(vLayout.$.blurMap, d.vec2u(cx, cy)).r
        if (blurRadius < PASSTHROUGH_THRESHOLD) {
            const texel = std.textureLoad(vLayout.$.src, d.vec2u(cx, cy))
            std.textureStore(vLayout.$.output, d.vec2u(cx, cy), texel)
        } else {
            const yi = d.i32(cy)
            const radiusComputeY = blurRadius / vLayout.$.params.scaleY
            const jitterV = (std.fract(52.9829189 * std.fract(d.f32(cx) * 0.00583715 + d.f32(cy) * 0.06711056)) - 0.5) * vLayout.$.params.jitter
            let sum = d.vec4f(0, 0, 0, 0)
            for (let i = -HALF_KERNEL; i <= HALF_KERNEL; i++) {
                const offset = (d.f32(i) + jitterV) * invHalf * radiusComputeY
                const sampleY = std.clamp(yi + d.i32(std.round(offset)), 0, computeHeight - 1)
                const texel = std.textureLoad(vLayout.$.src, d.vec2u(cx, d.u32(sampleY)))
                sum = sum.add(texel.mul(vLayout.$.weights[i + HALF_KERNEL]))
            }
            std.textureStore(vLayout.$.output, d.vec2u(cx, cy), sum)
        }
    }).$name('variableBlurV')

    const kernelVChromatic = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const blurRadius = std.textureLoad(vLayout.$.blurMap, d.vec2u(cx, cy)).r
        if (blurRadius < PASSTHROUGH_THRESHOLD) {
            const texel = std.textureLoad(vLayout.$.src, d.vec2u(cx, cy))
            std.textureStore(vLayout.$.output, d.vec2u(cx, cy), texel)
        } else {
            const yi = d.i32(cy)
            const p = vLayout.$.params
            const radiusComputeY = blurRadius / p.scaleY
            const radiusR = radiusComputeY * (1.0 + p.dispersion)
            const radiusB = radiusComputeY * (1.0 - p.dispersion)
            const jitterV = (std.fract(52.9829189 * std.fract(d.f32(cx) * 0.00583715 + d.f32(cy) * 0.06711056)) - 0.5) * p.jitter
            let sum = d.vec4f(0, 0, 0, 0)
            for (let i = -HALF_KERNEL; i <= HALF_KERNEL; i++) {
                const t = (d.f32(i) + jitterV) * invHalf
                const yG = std.clamp(yi + d.i32(std.round(t * radiusComputeY)), 0, computeHeight - 1)
                const yR = std.clamp(yi + d.i32(std.round(t * radiusR)), 0, computeHeight - 1)
                const yB = std.clamp(yi + d.i32(std.round(t * radiusB)), 0, computeHeight - 1)
                const texG = std.textureLoad(vLayout.$.src, d.vec2u(cx, d.u32(yG)))
                const texR = std.textureLoad(vLayout.$.src, d.vec2u(cx, d.u32(yR)))
                const texB = std.textureLoad(vLayout.$.src, d.vec2u(cx, d.u32(yB)))
                const texel = d.vec4f(texR.x, texG.y, texB.z, texG.w)
                sum = sum.add(texel.mul(vLayout.$.weights[i + HALF_KERNEL]))
            }
            std.textureStore(vLayout.$.output, d.vec2u(cx, cy), sum)
        }
    }).$name('variableBlurVChromatic')

    const kernelH = chromatic ? kernelHChromatic : kernelHPlain
    const kernelV = chromatic ? kernelVChromatic : kernelVPlain
    return {hLayout, vLayout, kernelH, kernelV, HParams, VParams, KERNEL_SIZE}
}

/**
 * Variable-radius (per-pixel) Gaussian blur with fixed-resolution output. The blur map texture is
 * at compute resolution and stores the desired blur radius (input/canvas pixels) per compute
 * pixel. The CALLER fills the map in their own compute pass by binding `blurMapTexture` to a
 * write-only storage-texture entry (the `blurMapWriteNode` replacement).
 *
 * @param options.chromatic - the three-radius (per-channel) kernel variant; baked into the WGSL.
 * @param options.jitter - initial tap-comb jitter (0–1, runtime; `setDetail` changes it per frame).
 */
export function createVariableGaussianBlurCompute(
    root: TgpuRoot,
    inputTexture: BlurInputTexture | null,
    inputWidth: number,
    inputHeight: number,
    onCleanup: (cb: () => void) => void,
    halfKernel: number = DEFAULT_HALF_KERNEL,
    computeWidth: number = DEFAULT_COMPUTE_WIDTH,
    computeHeight: number = DEFAULT_COMPUTE_HEIGHT,
    options?: {chromatic?: boolean; jitter?: number},
): VariableBlurComputeResult {
    const HALF_KERNEL = halfKernel
    const graph = buildVariableBlurGraph(halfKernel, computeWidth, computeHeight, {chromatic: options?.chromatic})
    const {hLayout, vLayout, kernelH, kernelV, KERNEL_SIZE} = graph

    // All storage textures at fixed compute resolution.
    const blurMapTex = root.createTexture({size: [computeWidth, computeHeight], format: BLUR_MAP_FORMAT}).$usage('storage')
    const intermediateTex = root.createTexture({size: [computeWidth, computeHeight], format: INTERMEDIATE_FORMAT}).$usage('storage')
    const outputTex = root.createTexture({size: [computeWidth, computeHeight], format: OUTPUT_FORMAT}).$usage('storage', 'sampled')

    // Bell-curve weights (shape only — tap spread is scaled by per-pixel radius). Written once.
    const weightsBuf = root.createBuffer(d.arrayOf(d.f32, KERNEL_SIZE)).$usage('storage')
    weightsBuf.write(buildFixedWeights(HALF_KERNEL))

    const paramsH = root.createUniform(graph.HParams)
    const paramsV = root.createUniform(graph.VParams)
    const detail = {jitter: options?.jitter ?? 0, dispersion: 0}
    paramsH.write({inputWidth, inputHeight, ...detail})
    paramsV.write({scaleY: inputHeight / computeHeight, ...detail})

    onCleanup(() => {
        blurMapTex.destroy()
        intermediateTex.destroy()
        outputTex.destroy()
    })

    const vBindGroup = root.createBindGroup(vLayout, {
        blurMap: blurMapTex,
        src: intermediateTex,
        output: outputTex,
        weights: weightsBuf,
        params: paramsV.buffer,
    })

    // H pass reads the source texture (blur-run late input, see setInputTexture); V pass does not.
    const hPipeline: KitComputePipeline = createGuardedCompute(root, (cx: number, cy: number) => {
        'use gpu'
        kernelH(cx, cy)
    }, {size: [computeWidth, computeHeight]})
    const vPass: KitComputePipeline = createGuardedCompute(root, (cx: number, cy: number) => {
        'use gpu'
        kernelV(cx, cy)
    }, {size: [computeWidth, computeHeight], bindGroup: vBindGroup})

    const computeSteps: ComputeStep[] = [hPipeline, vPass]
    function setInputTexture(input: BlurInputTexture): void {
        const hBindGroup = root.createBindGroup(hLayout, {
            blurMap: blurMapTex,
            input: input as never,
            intermediate: intermediateTex,
            weights: weightsBuf,
            params: paramsH.buffer,
        })
        computeSteps[0] = hPipeline.with(hBindGroup)
    }
    if (inputTexture != null) setInputTexture(inputTexture)

    return {
        computeSteps,
        outputTexture: outputTex,
        blurMapTexture: blurMapTex,
        computeWidth,
        computeHeight,
        setInputTexture,
        setInputDimensions: (newW: number, newH: number) => {
            paramsH.patch({inputWidth: newW, inputHeight: newH})
            paramsV.patch({scaleY: newH / computeHeight})
        },
        setDetail: (jitter: number, dispersion: number) => {
            if (jitter === detail.jitter && dispersion === detail.dispersion) return
            detail.jitter = jitter
            detail.dispersion = dispersion
            paramsH.patch({jitter, dispersion})
            paramsV.patch({jitter, dispersion})
        },
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// FRAGMENT-PATH TAP GATHER
//
// The three motion blurs (Angular, Linear, Zoom) are one shader with an interchangeable tap
// coordinate: RTT the child, sample it N times along some path, weight each tap by a Gaussian,
// sum. Only the coordinate differs, so only the coordinate stays per-shader.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Normalized Gaussian weights for `count` taps spread over ±1 of a Gaussian with std-dev `sigma`,
 * i.e. tap `i` sits at `t = (i/(count-1) - 0.5) * 2`. Computed on the CPU at build time so the
 * emitted WGSL carries literal weights instead of `exp()` per tap per pixel.
 *
 * σ² is rounded to 15 significant digits before the divide. `0.8 * 0.8` is `0.6400000000000001` in
 * float64, and these weights land in the WGSL as FULL-PRECISION literals — the un-rounded product
 * would move every consumer's snapshot for a one-ULP difference. 15 digits is lossless for any
 * decimal σ an author writes, and reproduces the hand-written `/ 0.64` tables bit-for-bit.
 */
export function gaussianTapWeights(count: number, sigma: number): number[] {
    const variance = Number((sigma * sigma).toPrecision(15))
    const last = count - 1
    const raw = Array.from({length: count}, (_, i) => {
        const t = (i / last - 0.5) * 2
        return Math.exp((-0.5 * t * t) / variance)
    })
    const sum = raw.reduce((a, b) => a + b)
    return raw.map((w) => w / sum)
}

export interface UnrolledTapGatherConfig {
    /** One weight per tap; its length IS the tap count. Typically {@link gaussianTapWeights}. */
    weights: number[]
    /** Sample coordinate for tap `i` (0-based). */
    tapCoord: (tapIndex: number) => Expr
    /** Sample the source at a coordinate — usually `(uv) => texture.sample(uv)`. */
    sample: (coord: Expr) => Expr
    /**
     * Reduce each sample to one channel before weighting (`'a'`, `'r'`, …). Omit to gather the
     * whole vec4. Used by the alpha-silhouette blurs (DropShadow) that only carry one channel.
     */
    component?: string
}

/**
 * Build the weighted sum of N unrolled taps: `Σ sample(tapCoord(i)) * weights[i]`.
 *
 * WHY UNROLLED. `textureSample` is illegal under non-uniform control flow, so a runtime loop over
 * taps is not available in a fragment shader. The builder emits a flat expression tree instead —
 * one `textureSample` per tap, no control flow at all. The tree is left-folded in tap order
 * (`((t0 + t1) + t2) …`) because float addition is not associative: any other grouping is a
 * different result.
 *
 * The caller owns the alpha convention. When the source is a premultiplied RTT (the normal case
 * for a filter) the result is premultiplied — finish with `blend.unpremultiplyAlpha`, or let
 * `defineRttFilter` append that tail.
 */
export function unrolledTapGather(config: UnrolledTapGatherConfig): Expr {
    const {weights, tapCoord, sample, component} = config
    let total: Expr | undefined
    for (let i = 0; i < weights.length; i++) {
        const raw = sample(tapCoord(i))
        const tap = component ? raw.member(component) : raw
        const weighted = tap.mul(weights[i])
        total = total ? total.add(weighted) : weighted
    }
    return total as Expr
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// MAP-DRIVEN RADIUS PLUMBING
//
// Every blur whose radius prop accepts a `map` driver has to reproduce the FRAGMENT path's map
// resolve inside a compute kernel: sample the map-source RTT at the corresponding canvas pixel,
// reduce it to a scalar via the chosen channel, and push it through the same remap window the
// `_map_<prop>_*` uniforms describe. Six kernels wrote that byte-for-byte (Blur, ProgressiveBlur,
// TiltShift, Glow, BokehBlur, and Blur's own fill graph). It lives here now.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * The four map-source channels a blur radius can be driven by — the same set the fragment path's
 * `resolveMapForProp` supports. A BUILD-TIME constant at every consumer: the channel selects one
 * of the memoized fns below, so each channel emits its own specialised WGSL.
 */
export type BlurMapChannel = 'luminance' | 'luminanceInverted' | 'alpha' | 'alphaInverted'

/**
 * Struct fields for the map-source canvas dimensions. Spread into a fill kernel's params struct so
 * the kernel can map its compute pixel onto the source's canvas pixel. Spread ORDER is
 * load-bearing (it is the uniform struct's member order) — these go before the remap window, which
 * is the order all six kernels already used.
 */
export const MAP_SOURCE_DIM_FIELDS = {inputWidth: d.f32, inputHeight: d.f32} as const

/** Struct fields for the remap window — the CPU side of these is {@link GpuMapWindow}. */
export const REMAP_WINDOW_FIELDS = {
    inputMin: d.f32,
    inputMax: d.f32,
    outputMin: d.f32,
    outputMax: d.f32,
    curve: d.f32,
} as const

const mapSourceScalarCache = new Map<BlurMapChannel, TgpuFn<(sample: d.Vec4f) => d.F32>>()

/**
 * The driving scalar a map source contributes at one texel, for `channel`. Memoized and `$name`d
 * per channel (C3): a shader picks one at graph-build time and calls it from its kernel body.
 *
 * Deliberately `dot(rgb, vec3(0.2126, 0.7152, 0.0722))` rather than `tone.luma709`, which spells
 * the same weights as a multiply-add chain. Same standard, different instruction sequence — and
 * these kernels are gated byte-identical.
 */
export function mapSourceScalar(channel: BlurMapChannel): TgpuFn<(sample: d.Vec4f) => d.F32> {
    const cached = mapSourceScalarCache.get(channel)
    if (cached) return cached
    const fn = tgpu.fn([d.vec4f], d.f32)((sample) => {
        'use gpu'
        return channel === 'alpha'
            ? sample.w
            : channel === 'alphaInverted'
              ? 1.0 - sample.w
              : channel === 'luminanceInverted'
                ? 1.0 - std.dot(sample.xyz, d.vec3f(0.2126, 0.7152, 0.0722))
                : std.dot(sample.xyz, d.vec3f(0.2126, 0.7152, 0.0722))
    }).$name(`mapSourceScalar_${channel}`)
    mapSourceScalarCache.set(channel, fn)
    return fn
}

/**
 * The map remap window, identical to the fragment path's: normalise `raw` into the input window,
 * apply the curve as an exponent (`2^(-curve·2)`, so 0 is linear and the sign flips the bend), and
 * mix across the output window. The input range is guarded against a zero-width window.
 *
 * Six scalar args rather than a struct (C1's threshold is "about five") because every consumer
 * reads these straight off its own uniform struct — a struct param here would either force a
 * per-kernel uniform-layout change or an inline struct construction, and these kernels are gated
 * byte-identical below the call.
 */
export const applyRemapWindow = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (raw, inputMin, inputMax, outputMin, outputMax, curve) => {
        'use gpu'
        const inputRange = std.max(inputMax - inputMin, 0.0001)
        const normalised = std.clamp((raw - inputMin) / inputRange, 0.0, 1.0)
        const exponent = std.pow(2.0, -curve * 2.0)
        const eased = std.pow(normalised, exponent)
        return std.mix(outputMin, outputMax, eased)
    },
).$name('applyRemapWindow')

/** Spread a live {@link GpuMapWindow} into a fill kernel's params write. */
export function remapWindowValues(window: GpuMapWindow): GpuMapWindow {
    return {
        inputMin: window.inputMin,
        inputMax: window.inputMax,
        outputMin: window.outputMin,
        outputMax: window.outputMax,
        curve: window.curve,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// BLOOM PRE-PASS (bright extract + radius fill)
//
// A bloom is: reduce the child to the pixels brighter than a threshold, blur THAT, add it back.
// The extraction runs at compute resolution and doubles as the radius-map fill, so one kernel
// feeds both the variable blur's input and its per-pixel radius. Glow and FilmStock's halation
// are the same pre-pass with different names on it.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Bright-pixel extraction for a bloom, at one compute pixel.
 *
 * Two things are happening. First, the canvas-pixel FOOTPRINT of this compute pixel is
 * area-averaged over a 4×4 stratified grid rather than point-sampled: that is energy-preserving, so
 * a hairline or a dither dot contributes in proportion to its coverage instead of shimmering on and
 * off as the grids slide past each other. Second, the threshold has a quadratic soft knee (the
 * Unity / Call-of-Duty bloom curve) so brightness ramps INTO the bloom around the threshold instead
 * of popping, and the mask is luminance-normalised so the extracted color keeps its hue.
 *
 * Returns the extracted (premultiplied) color to store in the bright buffer. The texture arrives
 * as an fn argument per C6.
 */
export const bloomExtractSoftKnee = tgpu.fn(
    [d.texture2d(d.f32), d.u32, d.u32, d.vec2f, d.vec2f, d.f32],
    d.vec4f,
)((childTexture, cx, cy, inputSize, computeSize, threshold) => {
    'use gpu'
    const iw = inputSize.x
    const ih = inputSize.y
    const inputWm1 = d.i32(iw - 1)
    const inputHm1 = d.i32(ih - 1)

    let sample = d.vec4f(0, 0, 0, 0)
    for (let ty = 0; ty < 4; ty++) {
        for (let tx = 0; tx < 4; tx++) {
            const su = (d.f32(cx) + (d.f32(tx) + 0.5) / 4.0) / computeSize.x
            const sv = (d.f32(cy) + (d.f32(ty) + 0.5) / 4.0) / computeSize.y
            const sx = std.clamp(d.i32(su * iw), 0, inputWm1)
            const sy = std.clamp(d.i32(sv * ih), 0, inputHm1)
            sample = sample.add(std.textureLoad(childTexture, d.vec2u(d.u32(sx), d.u32(sy)), 0))
        }
    }
    sample = sample.mul(0.0625)

    const knee = threshold * 0.5 + 1e-4
    const lum = std.dot(sample.xyz, d.vec3f(0.2126, 0.7152, 0.0722))
    const soft = std.clamp(lum - threshold + knee, 0.0, knee * 2.0)
    const softCurve = (soft * soft) / (knee * 4.0)
    const mask = std.max(softCurve, lum - threshold) / std.max(lum, 1e-4)
    return d.vec4f(sample.xyz.mul(mask), sample.w * mask)
}).$name('bloomExtractSoftKnee')

/**
 * A fill graph: the bind-group layout, the kernel that writes the radius map (and, for a bloom, the
 * bright buffer), and the params struct. Built GPU-free so the kernel `tgpu.resolve`s for snapshots.
 */
export interface BlurFillGraph {
    layout: TgpuBindGroupLayout
    kernel: (cx: number, cy: number) => void
    Params: d.AnyWgslStruct
}

/** Bright-buffer storage format for a bloom pre-pass. Filterable (so the blur's H pass can bind it
 *  SAMPLED), a valid write-only storage format (so the pre-pass can `textureStore` into it), and it
 *  holds HDR values above 1.0 — the one format that satisfies all three roles. */
const BRIGHT_FORMAT = 'rgba16float' as const

/**
 * The bloom pre-pass graph with a UNIFORM blur radius: bright-extract the canvas-res child RTT into
 * `brightMap` and write `params.size` into every texel of `blurMap`.
 *
 * `kernelName` is the emitted WGSL fn name — pass the consumer's existing name so its snapshot
 * doesn't move.
 */
export function buildBloomExtractGraph(computeWidth: number, computeHeight: number, kernelName: string) {
    const Params = d.struct({threshold: d.f32, size: d.f32, ...MAP_SOURCE_DIM_FIELDS})
    const layout = tgpu.bindGroupLayout({
        childTexture: {texture: d.texture2d(d.f32)},
        brightMap: {storageTexture: d.textureStorage2d(BRIGHT_FORMAT, 'write-only')},
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const bright = bloomExtractSoftKnee(
            layout.$.childTexture, cx, cy,
            d.vec2f(p.inputWidth, p.inputHeight), d.vec2f(computeWidth, computeHeight), p.threshold,
        )
        std.textureStore(layout.$.brightMap, d.vec2u(cx, cy), bright)
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(p.size, 0.0, 0.0, 1.0))
    }).$name(kernelName)

    return {layout, kernel, Params}
}

/**
 * The bloom pre-pass graph with a MAP-DRIVEN blur radius. Bright extraction is identical to
 * {@link buildBloomExtractGraph}; the radius comes from a nearest tap on the map SOURCE (a
 * canvas-resolution RTT, bound late) pushed through the shared remap window. No `× 0.36` scaling —
 * a bloom's size prop is already in pixels.
 */
export function buildBloomExtractMapGraph(
    computeWidth: number,
    computeHeight: number,
    kernelName: string,
    channel: BlurMapChannel,
) {
    const Params = d.struct({threshold: d.f32, ...MAP_SOURCE_DIM_FIELDS, ...REMAP_WINDOW_FIELDS})
    const layout = tgpu.bindGroupLayout({
        childTexture: {texture: d.texture2d(d.f32)},
        source: {texture: d.texture2d(d.f32)},
        brightMap: {storageTexture: d.textureStorage2d(BRIGHT_FORMAT, 'write-only')},
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })
    const channelScalar = mapSourceScalar(channel)

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const iw = p.inputWidth
        const ih = p.inputHeight
        const bright = bloomExtractSoftKnee(
            layout.$.childTexture, cx, cy,
            d.vec2f(iw, ih), d.vec2f(computeWidth, computeHeight), p.threshold,
        )
        std.textureStore(layout.$.brightMap, d.vec2u(cx, cy), bright)

        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const srcX = std.clamp(d.i32(std.round(u * iw)), 0, d.i32(iw - 1))
        const srcY = std.clamp(d.i32(std.round(v * ih)), 0, d.i32(ih - 1))
        const mapSample = std.textureLoad(layout.$.source, d.vec2u(d.u32(srcX), d.u32(srcY)), 0)
        const radius = applyRemapWindow(
            channelScalar(mapSample), p.inputMin, p.inputMax, p.outputMin, p.outputMax, p.curve,
        )
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name(kernelName)

    return {layout, kernel, Params}
}

/**
 * Compute resolution for a bloom: cap the LONGER edge at the default long edge and derive the other
 * from the canvas aspect. A fixed 1024×640 grid stretches the glow kernel on any other aspect —
 * the blur works in compute pixels, so non-square compute pixels smear it directionally.
 */
export function aspectAwareComputeRes(canvasWidth: number, canvasHeight: number): {width: number; height: number} {
    const LONG_EDGE = Math.max(DEFAULT_COMPUTE_WIDTH, DEFAULT_COMPUTE_HEIGHT)
    const aspect = canvasHeight > 0 ? canvasWidth / canvasHeight : 1
    return {
        width: Math.max(8, aspect >= 1 ? LONG_EDGE : Math.round(LONG_EDGE * aspect)),
        height: Math.max(8, aspect >= 1 ? Math.round(LONG_EDGE / aspect) : LONG_EDGE),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// COMPUTE-HOOK LIFECYCLE
//
// Every compute-backed blur wires the same six things: RTT the child, allocate the blur, publish
// its output as a compute texture, follow resize, bind the child's RTT late (the pass manager
// allocates it AFTER composition), and write per-frame params before dispatching. Only the params
// and the radius policy are per-shader. These wrappers own the rest.
//
// All three return the compute node directly, so a shader's `compute` hook is a one-liner plus its
// own bypasses. They return `null` for "no child" and "no device" — the fragment's
// compute-unavailable fallback covers both (see `composeBlurredOverSharp`).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/** What a `compute` hook returns. */
type BlurComputeNode = NonNullable<ReturnType<GpuComputeNode>>

/** Key the blurred buffer is published under in `outputs` unless overridden. */
const DEFAULT_BLURRED_KEY = 'blurredTexture'

function canvasPixels(dimensions: {width: number; height: number}) {
    return {width: Math.max(1, Math.round(dimensions.width)), height: Math.max(1, Math.round(dimensions.height))}
}

export interface FixedBlurComputeConfig {
    /** Per-frame Gaussian radius in INPUT (canvas) pixels. Read live so mouse/auto drivers pull through. */
    radius: () => number
    halfKernel?: number
    outputKey?: string
}

/**
 * A uniform-radius separable Gaussian over the composed child. Blur's static path and ChannelBlur.
 */
export function withFixedBlurCompute(params: GpuFragmentParams, config: FixedBlurComputeConfig): BlurComputeNode | null {
    const {childNode, gpu, convertToTexture, registerComputeTexture, onCleanup, onResize, dimensions} = params
    if (!childNode) return null
    const root = gpu?.root
    if (!root) return null
    const outputKey = config.outputKey ?? DEFAULT_BLURRED_KEY

    // The composed child, RTT'd once: the blur's input AND the sharp layer the fragment composites.
    const childTexture = convertToTexture(childNode)
    const canvas = canvasPixels(dimensions)
    const gaussian = createGaussianBlurCompute(root, null, canvas.width, canvas.height, onCleanup, config.halfKernel)
    const blurredTexture = registerComputeTexture(gaussian.outputTexture)

    onResize(({width, height}) => {
        const next = canvasPixels({width, height})
        gaussian.setInputDimensions(next.width, next.height)
    })

    return {
        outputs: {childTexture, [outputKey]: blurredTexture},
        bindInputs: (resolve) => {
            const src = resolve(childTexture.key)
            if (src) gaussian.setInputTexture(src.texture as never)
        },
        getComputeNodes: () => {
            gaussian.updateRadius(config.radius())
            return gaussian.computeSteps
        },
    }
}

export interface VariableBlurComputeConfig {
    /**
     * Build the radius-map fill graph at the blur's compute resolution. Its layout MUST bind
     * `blurMap` (write-only, {@link BLUR_MAP_FORMAT}) and `params`; with `source` set it must also
     * bind a `source` sampled texture.
     */
    buildFill: (computeWidth: number, computeHeight: number) => BlurFillGraph
    /**
     * Per-frame values for the fill params. `dims` is the live canvas size in device pixels;
     * `window` is the live remap window, present only on the map-driven path.
     */
    fillValues: (dims: {width: number; height: number}, window?: GpuMapWindow) => Record<string, number>
    /** Map-driven radius: pass `getMapInfo(prop)`. Its source RTT is bound late, like the child. */
    source?: GpuMapInfo
    halfKernel?: number
    /** Build the three-radius (per-channel) kernel variant. Enable only while the dispersion is non-zero. */
    chromatic?: boolean
    /** Per-frame detail dials (see `VariableBlurComputeResult.setDetail`). */
    detail?: () => {jitter: number; dispersion: number}
    outputKey?: string
}

/**
 * A per-pixel-radius Gaussian whose radius map is filled by a caller-supplied kernel each frame.
 * ProgressiveBlur, TiltShift, and Blur's map-driven path.
 */
export function withVariableBlurCompute(params: GpuFragmentParams, config: VariableBlurComputeConfig): BlurComputeNode | null {
    const {childNode, gpu, convertToTexture, registerComputeTexture, onCleanup, onResize, dimensions} = params
    if (!childNode) return null
    const root = gpu?.root
    if (!root) return null
    const outputKey = config.outputKey ?? DEFAULT_BLURRED_KEY
    const source = config.source

    const childTexture = convertToTexture(childNode)
    let canvas = canvasPixels(dimensions)
    const variable = createVariableGaussianBlurCompute(
        root, null, canvas.width, canvas.height, onCleanup, config.halfKernel, undefined, undefined, {chromatic: config.chromatic},
    )
    const blurredTexture = registerComputeTexture(variable.outputTexture)

    const fill = config.buildFill(variable.computeWidth, variable.computeHeight)
    // The fill graph's params struct is opaque here (each shader declares its own geometry fields),
    // so the uniform and its per-frame write are typed through `never` — the shader's `fillValues`
    // return type is what actually keeps the two in step.
    const fillParams = root.createUniform(fill.Params as never)
    // kit types blurMapTexture as a plain TgpuTexture (no StorageFlag in its type) — hence the cast.
    const blurMap = variable.blurMapTexture as never
    // Static fill: nothing late to wait for, so the bind group exists now. Map-driven fill: the
    // source RTT is allocated after composition, so the pipeline stays unbound until `bindInputs`.
    const fillStepOpts = {size: [variable.computeWidth, variable.computeHeight] as [number, number]}
    let fillStep = source
        ? createGuardedCompute(root, (cx: number, cy: number) => {
            'use gpu'
            fill.kernel(cx, cy)
        }, fillStepOpts)
        : createGuardedCompute(root, (cx: number, cy: number) => {
            'use gpu'
            fill.kernel(cx, cy)
        }, {...fillStepOpts, bindGroup: root.createBindGroup(fill.layout as never, {blurMap, params: fillParams.buffer} as never)})

    onResize(({width, height}) => {
        canvas = canvasPixels({width, height})
        variable.setInputDimensions(canvas.width, canvas.height)
    })

    return {
        outputs: {childTexture, [outputKey]: blurredTexture},
        bindInputs: (resolve) => {
            const src = resolve(childTexture.key)
            if (src) variable.setInputTexture(src.texture as never)
            if (!source) return
            const mapSrc = resolve(source.sourceTexture.key)
            if (mapSrc) {
                const bg = root.createBindGroup(fill.layout as never, {
                    source: mapSrc.texture, blurMap, params: fillParams.buffer,
                } as never)
                fillStep = fillStep.with(bg)
            }
        },
        getComputeNodes: () => {
            fillParams.write(config.fillValues(canvas, source?.window()) as never)
            if (config.detail) {
                const dials = config.detail()
                variable.setDetail(dials.jitter, dials.dispersion)
            }
            // Fill the radius map first; the variable H/V passes then read it per pixel.
            return [fillStep, ...variable.computeSteps]
        },
    }
}

export interface BloomComputeConfig {
    /**
     * Build the extract graph at the bloom's compute resolution — normally a one-line delegate to
     * {@link buildBloomExtractGraph} carrying the shader's own kernel name (so its snapshot is
     * stable and its GPU-free resolve test has something to import).
     */
    buildExtract: (computeWidth: number, computeHeight: number) => BlurFillGraph
    /** The map-driven twin, via {@link buildBloomExtractMapGraph}. Required when `mapInfo` is given. */
    buildExtractMap?: (computeWidth: number, computeHeight: number, channel: BlurMapChannel) => BlurFillGraph
    /** Map-driven radius: pass `getMapInfo(sizeProp)`, or null for the uniform path. */
    mapInfo?: GpuMapInfo | null
    /** Per-frame soft-knee threshold in [0, 1]. */
    threshold: () => number
    /** Per-frame uniform bloom radius in canvas pixels (uniform path only). */
    radius: () => number
    /** Key the blurred bright buffer is published under (Glow: default; FilmStock: `halationTexture`). */
    outputKey?: string
}

/**
 * A bloom: bright-extract the child at aspect-aware compute resolution, then blur the extract.
 *
 * The blur's input dimensions ARE its compute dimensions here (scaleY 1), so the H pass maps 1:1 and
 * the blur never needs `setInputDimensions` — only the extract kernel's canvas dims change on
 * resize. Tap jitter is on: the bloom radius is large and a fixed tap comb beats against periodic
 * content as halo moiré.
 */
export function withBloomCompute(params: GpuFragmentParams, config: BloomComputeConfig): BlurComputeNode | null {
    const {childNode, gpu, convertToTexture, registerComputeTexture, onCleanup, onResize, dimensions} = params
    if (!childNode) return null
    const root = gpu?.root
    if (!root) return null
    const outputKey = config.outputKey ?? DEFAULT_BLURRED_KEY
    const mapInfo = config.mapInfo ?? null

    const childTexture = convertToTexture(childNode)
    let canvas = canvasPixels(dimensions)
    const res = aspectAwareComputeRes(canvas.width, canvas.height)

    // Bright buffer at compute res: written by the extract (storage), sampled by the blur.
    const brightBuffer = root.createTexture({size: [res.width, res.height], format: BRIGHT_FORMAT}).$usage('storage', 'sampled')
    onCleanup(() => brightBuffer.destroy())

    const variable = createVariableGaussianBlurCompute(
        root, brightBuffer, res.width, res.height, onCleanup, undefined, res.width, res.height, {jitter: 1},
    )
    const blurredTexture = registerComputeTexture(variable.outputTexture)

    // Only the extract's child-RTT coord scaling changes on resize; the compute buffers are fixed.
    onResize(({width, height}) => {
        canvas = canvasPixels({width, height})
    })

    const graph = mapInfo && config.buildExtractMap
        ? config.buildExtractMap(res.width, res.height, mapInfo.channel as BlurMapChannel)
        : config.buildExtract(res.width, res.height)
    // Opaque params struct (the two variants differ) — see the note in `withVariableBlurCompute`.
    const prepassParams = root.createUniform(graph.Params as never)
    let prepassStep = createGuardedCompute(root, (cx: number, cy: number) => {
        'use gpu'
        graph.kernel(cx, cy)
    }, {size: [res.width, res.height]})

    return {
        outputs: {childTexture, [outputKey]: blurredTexture},
        // The child RTT (and the size-map source) are allocated after composition.
        bindInputs: (resolve) => {
            const src = resolve(childTexture.key)
            if (!src) return
            const mapSrc = mapInfo ? resolve(mapInfo.sourceTexture.key) : undefined
            if (mapInfo && !mapSrc) return
            const entries: Record<string, unknown> = {
                childTexture: src.texture,
                brightMap: brightBuffer,
                // kit types blurMapTexture as a plain TgpuTexture (no StorageFlag in its type).
                blurMap: variable.blurMapTexture,
                params: prepassParams.buffer,
            }
            if (mapSrc) entries.source = mapSrc.texture
            prepassStep = prepassStep.with(root.createBindGroup(graph.layout as never, entries as never))
        },
        getComputeNodes: () => {
            const threshold = config.threshold()
            const values: Record<string, number> = mapInfo
                ? {threshold, inputWidth: canvas.width, inputHeight: canvas.height, ...remapWindowValues(mapInfo.window())}
                : {threshold, size: config.radius(), inputWidth: canvas.width, inputHeight: canvas.height}
            prepassParams.write(values as never)
            return [prepassStep, ...variable.computeSteps]
        },
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// FRAGMENT COMPOSITE
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * The fragment tail every compute-backed blur shares: composite the canvas-resolution SHARP child
 * against the compute-resolution BLURRED buffer, and unpremultiply on the way back into the
 * straight-alpha blend pipeline (both textures hold premultiplied data — the child RTT always does).
 *
 * `compose` receives `(blurred, sharp)` samples at `ctx.uv` and returns the premultiplied result:
 * `vec4(blurred.rgb, sharp.a)` for a plain blur, a mix by the shader's own blur-amount field for a
 * selective one, a bloom add for a glow.
 *
 * When compute did not run — no device, or a shader's own bypass returned `null` from its `compute`
 * hook — this falls back to a sharp passthrough of the child RTT. That fallback is what makes the
 * GPU-free resolve gates work, so every consumer needs it and none should write it again.
 */
export function composeBlurredOverSharp(
    params: GpuFragmentParams,
    compose: (blurred: Expr, sharp: Expr) => Expr,
    options?: {blurredKey?: string; sharpKey?: string},
): Expr {
    const {childNode, computeOutputs, ctx, convertToTexture} = params
    if (!childNode) return ZERO
    const blurred = computeOutputs?.[options?.blurredKey ?? DEFAULT_BLURRED_KEY] as KitTexture | undefined
    const sharp = computeOutputs?.[options?.sharpKey ?? 'childTexture'] as KitTexture | undefined
    if (!blurred || !sharp) {
        const tex = convertToTexture(childNode)
        return call(unpremultiplyAlpha, 'unpremultiplyAlpha', [tex.sample(ctx.uv)])
    }
    const sharpSample = sharp.sample(ctx.uv)
    const blurredSample = blurred.sample(ctx.uv)
    return call(unpremultiplyAlpha, 'unpremultiplyAlpha', [compose(blurredSample, sharpSample)])
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// PER-EFFECT FILL GRAPHS + COMPOSE FNS (behind std/effects/blurs' compute-backed nouns)
//
// Each compute-backed blur contributes two shader-specific pieces on top of the shared lifecycle
// wrappers above: the fill kernel that writes its radius map (its geometry ramp / map resolve) and
// the fragment compose fn that mixes the blurred buffer against the sharp child. They live here so
// the shader files carry no GPU code.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Blur intensity (UI 0–200) → Gaussian pixel radius. The separable Gaussian derives its per-pass
 * sigma from this radius (sigmaH = radius × 0.5). The factor is `intensity × 0.36` — shared by
 * Blur, ProgressiveBlur, and TiltShift (their intensity props are all 0–100/0–200 UI scales over
 * the same pixel-radius mapping).
 */
export const INTENSITY_TO_RADIUS = 0.36
export const intensityToRadius = (intensity: number): number => intensity * INTENSITY_TO_RADIUS

/**
 * How far a full `dispersion` of ±1 spreads the per-channel radii of the chromatic variable blur:
 * red at `radius × (1 + 0.5)`, blue at `radius × (1 − 0.5)`.
 */
export const VARIABLE_BLUR_DISPERSION_SPREAD = 0.5

/**
 * ChannelBlur's per-channel UI intensity (0–100) → Gaussian pixel radius via the `× 0.1` factor. A
 * single fixed Gaussian runs at the MAX per-channel radius; the fragment then mixes each channel
 * between the sharp source and that one blurred buffer by `channelRadius / maxRadius` (an
 * approximation that avoids 6 dispatches / 6 storage textures).
 */
export const CHANNEL_INTENSITY_TO_RADIUS = 0.1
export const channelIntensityToRadius = (intensity: number): number => intensity * CHANNEL_INTENSITY_TO_RADIUS

// ── Blur (map-driven fill): canvas dims + the map remap window. ─────────────────────────────
function fillMapParams() {
    return d.struct({...MAP_SOURCE_DIM_FIELDS, ...REMAP_WINDOW_FIELDS})
}

/**
 * Blur's map-driven fill graph: per compute pixel, sample the map-source RTT at the corresponding
 * canvas pixel, extract the driving scalar from a channel, run the SAME remap `resolveMapForProp`
 * uses in the fragment path, scale to a pixel radius (× 0.36), and write it into the variable
 * blur's radius map. GPU-free (resolvable without a device). `channel` is a build-time constant —
 * the channel branch is comptime-folded, so each channel emits its own specialised WGSL.
 */
export function buildFillBlurMapGraph(computeWidth: number, computeHeight: number, channel: BlurMapChannel) {
    const Params = fillMapParams()
    const layout = tgpu.bindGroupLayout({
        source: {texture: d.texture2d(d.f32)},
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const channelScalar = mapSourceScalar(channel)

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        // Map this compute pixel's center to the corresponding canvas pixel of the map source.
        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const srcX = std.clamp(d.i32(std.round(u * p.inputWidth)), 0, d.i32(p.inputWidth - 1))
        const srcY = std.clamp(d.i32(std.round(v * p.inputHeight)), 0, d.i32(p.inputHeight - 1))
        const sample = std.textureLoad(layout.$.source, d.vec2u(d.u32(srcX), d.u32(srcY)), 0)

        // The shared channel + remap-window path — the same resolve `resolveMapForProp` runs in the
        // fragment, so a map-driven blur matches its static twin at the same effective intensity.
        const remapped = applyRemapWindow(
            channelScalar(sample), p.inputMin, p.inputMax, p.outputMin, p.outputMax, p.curve,
        )
        // Scale to pixel radius (mirrors the static path: intensity × 0.36).
        const radius = remapped * 0.36
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('fillBlurMap')

    return {layout, kernel, Params}
}

/**
 * ChannelBlur's fragment composite: per-channel `mix(sharp, blurred, channelRadius / maxRadius)`.
 * The blur ran at `maxRadius = max(r, g, b, 0.01)`, so the max-intensity channel takes the full
 * blur, a zero channel stays exactly sharp, and intermediate channels linearly interpolate. Alpha
 * comes from the sharp child.
 */
export const channelBlurCompose = tgpu.fn([d.vec4f, d.vec4f, d.f32, d.f32, d.f32], d.vec4f)(
    (sharp, blurred, redIntensity, greenIntensity, blueIntensity) => {
        'use gpu'
        const redRadius = redIntensity * 0.1
        const greenRadius = greenIntensity * 0.1
        const blueRadius = blueIntensity * 0.1
        const maxRadius = std.max(std.max(std.max(redRadius, greenRadius), blueRadius), 0.01)
        const rMix = redRadius / maxRadius
        const gMix = greenRadius / maxRadius
        const bMix = blueRadius / maxRadius
        const finalR = std.mix(sharp.x, blurred.x, rMix)
        const finalG = std.mix(sharp.y, blurred.y, gMix)
        const finalB = std.mix(sharp.z, blurred.z, bMix)
        return d.vec4f(finalR, finalG, finalB, sharp.w)
    },
).$name('channelBlurCompose')

/**
 * ProgressiveBlur's directional blur amount at a UV: project the vector from the center onto the
 * (aspect-corrected) blur direction, clamp to the forward half-plane, and ramp it over `falloff`.
 * `center` is the TRANSFORMED position (transformPosition stores `(x, 1 - y)`), so `1.0 - centerY`
 * recovers the authored y. Pure float; used by the fill kernel to build the per-pixel radius map.
 */
export const progressiveBlurAmount = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.vec2f, d.f32], d.f32)(
    (angleDeg, centerX, centerY, falloff, uv, aspect) => {
        'use gpu'
        const angleRad = angleDeg * DEG_TO_RAD
        const direction = d.vec2f(std.cos(angleRad) / aspect, std.sin(angleRad))
        const centerPos = d.vec2f(centerX, 1.0 - centerY)
        const centeredUV = uv.sub(centerPos)
        const directionalDist = std.max(0.0, std.dot(centeredUV, direction))
        return std.smoothstep(0.0, falloff, directionalDist)
    },
).$name('progressiveBlurAmount')

// ── ProgressiveBlur fill params: geometry (angle/center/falloff) + aspect + the max radius. ──
function progressiveFillParams() {
    return d.struct({
        angle: d.f32,
        centerX: d.f32,
        centerY: d.f32,
        falloff: d.f32,
        aspect: d.f32,
        maxRadius: d.f32,
    })
}

// ── Map-driven variant: geometry (minus the single maxRadius) + the map-source canvas dims + the
// remap window. The per-pixel maxRadius is sampled from the map source and remapped, then scaled
// by the directional blur amount. ──
function progressiveFillMapParams() {
    return d.struct({
        angle: d.f32,
        centerX: d.f32,
        centerY: d.f32,
        falloff: d.f32,
        aspect: d.f32,
        ...MAP_SOURCE_DIM_FIELDS,
        ...REMAP_WINDOW_FIELDS,
    })
}

/**
 * ProgressiveBlur's fill graph: the kernel writes `blurAmount × maxRadius` (input pixels) into the
 * variable blur's radius map. GPU-free.
 */
export function buildProgressiveBlurFillGraph(computeWidth: number, computeHeight: number) {
    const Params = progressiveFillParams()
    const layout = tgpu.bindGroupLayout({
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        // UV is naturally normalized — independent of canvas size (`(cx + 0.5) / computeWidth`).
        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const blurAmount = progressiveBlurAmount(p.angle, p.centerX, p.centerY, p.falloff, d.vec2f(u, v), p.aspect)
        const radius = blurAmount * p.maxRadius
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('progressiveBlurFillBlurMap')

    return {layout, kernel, Params}
}

/**
 * ProgressiveBlur's MAP-DRIVEN fill graph. The per-pixel max radius is sampled from the map SOURCE
 * (canvas-resolution RTT), remapped through the same window `resolveMapForProp` uses in the
 * fragment path, scaled `× 0.36`, then multiplied by the directional blur amount. Bind-group adds
 * the `source` texture (bound late — its RTT is allocated after composition).
 */
export function buildProgressiveBlurFillMapGraph(computeWidth: number, computeHeight: number, channel: BlurMapChannel) {
    const Params = progressiveFillMapParams()
    const layout = tgpu.bindGroupLayout({
        source: {texture: d.texture2d(d.f32)},
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const channelScalar = mapSourceScalar(channel)

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const blurAmount = progressiveBlurAmount(p.angle, p.centerX, p.centerY, p.falloff, d.vec2f(u, v), p.aspect)

        // Map this compute pixel's center to the corresponding canvas pixel of the map source, then
        // through the shared channel + remap-window path (identical to the fragment's map resolve).
        const srcX = std.clamp(d.i32(std.round(u * p.inputWidth)), 0, d.i32(p.inputWidth - 1))
        const srcY = std.clamp(d.i32(std.round(v * p.inputHeight)), 0, d.i32(p.inputHeight - 1))
        const sample = std.textureLoad(layout.$.source, d.vec2u(d.u32(srcX), d.u32(srcY)), 0)
        const remapped = applyRemapWindow(
            channelScalar(sample), p.inputMin, p.inputMax, p.outputMin, p.outputMax, p.curve,
        )
        const radius = blurAmount * (remapped * INTENSITY_TO_RADIUS)
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('progressiveBlurFillBlurMapVariable')

    return {layout, kernel, Params}
}

/**
 * TiltShift's blur amount at a UV: perpendicular distance from the (aspect-corrected) focus line,
 * ramped from the sharp band (`focusWidth = width × 0.5`) out over `falloff`. `center` is the
 * TRANSFORMED position (transformPosition stores `(x, 1 - y)`), so `1.0 - center.y` recovers the
 * authored y. Pure float. Shared by BOTH the fill kernel (to build the per-pixel radius map) AND
 * the fragment (to keep in-focus pixels crisp), so the two can never drift.
 */
export const tiltShiftBlurAmount = tgpu.fn([d.f32, d.vec2f, d.f32, d.f32, d.vec2f, d.f32], d.f32)(
    (angleDeg, center, width, falloff, uv, aspect) => {
        'use gpu'
        const angleRad = angleDeg * DEG_TO_RAD
        const perpVector = d.vec2f(std.sin(angleRad) * -1.0, std.cos(angleRad))
        const centerPos = d.vec2f(center.x, 1.0 - center.y)
        const centeredUV = uv.sub(centerPos)
        const aspectCorrectedUV = d.vec2f(centeredUV.x * aspect, centeredUV.y)
        const distFromLine = std.abs(std.dot(aspectCorrectedUV, perpVector))
        const focusWidth = width * 0.5
        return std.smoothstep(focusWidth, focusWidth + falloff, distFromLine)
    },
).$name('tiltShiftBlurAmount')

// ── TiltShift fill params: geometry (angle/center/width/falloff) + aspect + the max radius. ──
function tiltShiftFillParams() {
    return d.struct({
        angle: d.f32,
        centerX: d.f32,
        centerY: d.f32,
        width: d.f32,
        falloff: d.f32,
        aspect: d.f32,
        maxRadius: d.f32,
    })
}

// ── Map-driven variant: geometry (minus the single maxRadius) + the map-source canvas dims + the
// remap window. ──
function tiltShiftFillMapParams() {
    return d.struct({
        angle: d.f32,
        centerX: d.f32,
        centerY: d.f32,
        width: d.f32,
        falloff: d.f32,
        aspect: d.f32,
        ...MAP_SOURCE_DIM_FIELDS,
        ...REMAP_WINDOW_FIELDS,
    })
}

/**
 * TiltShift's fill graph: the kernel writes `blurAmount × maxRadius` (input pixels) into the
 * variable blur's radius map. GPU-free.
 */
export function buildTiltShiftFillGraph(computeWidth: number, computeHeight: number) {
    const Params = tiltShiftFillParams()
    const layout = tgpu.bindGroupLayout({
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        // UV is naturally normalized — independent of canvas size.
        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const blurAmount = tiltShiftBlurAmount(p.angle, d.vec2f(p.centerX, p.centerY), p.width, p.falloff, d.vec2f(u, v), p.aspect)
        const radius = blurAmount * p.maxRadius
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('tiltShiftFillBlurMap')

    return {layout, kernel, Params}
}

/**
 * TiltShift's MAP-DRIVEN fill graph. The per-pixel max radius is sampled from the map SOURCE,
 * remapped through the fragment path's window, scaled `× 0.36`, then multiplied by the tilt-shift
 * blur amount. Bind-group adds the `source` texture (bound late).
 */
export function buildTiltShiftFillMapGraph(computeWidth: number, computeHeight: number, channel: BlurMapChannel) {
    const Params = tiltShiftFillMapParams()
    const layout = tgpu.bindGroupLayout({
        source: {texture: d.texture2d(d.f32)},
        blurMap: {storageTexture: d.textureStorage2d(BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const channelScalar = mapSourceScalar(channel)

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const blurAmount = tiltShiftBlurAmount(p.angle, d.vec2f(p.centerX, p.centerY), p.width, p.falloff, d.vec2f(u, v), p.aspect)

        // Map this compute pixel's center to the corresponding canvas pixel of the map source, then
        // through the shared channel + remap-window path (identical to the fragment's map resolve).
        const srcX = std.clamp(d.i32(std.round(u * p.inputWidth)), 0, d.i32(p.inputWidth - 1))
        const srcY = std.clamp(d.i32(std.round(v * p.inputHeight)), 0, d.i32(p.inputHeight - 1))
        const sample = std.textureLoad(layout.$.source, d.vec2u(d.u32(srcX), d.u32(srcY)), 0)
        const remapped = applyRemapWindow(
            channelScalar(sample), p.inputMin, p.inputMax, p.outputMin, p.outputMax, p.curve,
        )
        const radius = blurAmount * (remapped * INTENSITY_TO_RADIUS)
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('tiltShiftFillBlurMapVariable')

    return {layout, kernel, Params}
}

/**
 * Glow's bloom composite: original color plus the intensity-scaled blurred bloom, with the glow
 * aura extending into transparent areas (composite the blurred coverage OVER the child's alpha so
 * the halo isn't clipped to `original.a`). Pure float.
 */
export const glowCompose = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.vec4f)(
    (original, bloom, intensity) => {
        'use gpu'
        const finalColor = original.xyz.add(bloom.xyz.mul(intensity))
        const glowAlpha = std.clamp(bloom.w * intensity, 0.0, 1.0)
        const finalAlpha = original.w + glowAlpha * (1.0 - original.w)
        return d.vec4f(finalColor, finalAlpha)
    },
).$name('glowCompose')

/** Glow's combined bright-extract + blur-map fill pre-pass (uniform static size). GPU-free. */
export const buildGlowPrepassGraph = (computeWidth: number, computeHeight: number) =>
    buildBloomExtractGraph(computeWidth, computeHeight, 'glowExtractAndFill')

/** Glow's MAP-DRIVEN pre-pass: bright extraction is identical; the blur radius per pixel is sampled
 *  from the size-map SOURCE and remapped through the same window the fragment path uses. No
 *  `× 0.36` — Glow's size is already in pixels. */
export const buildGlowPrepassMapGraph = (computeWidth: number, computeHeight: number, channel: BlurMapChannel) =>
    buildBloomExtractMapGraph(computeWidth, computeHeight, 'glowExtractAndFillVariable', channel)


// ═══════════════════════════════════════════════════════════════════════════════════════
// Aperture-table (bokeh) gather. Single-pass scatter-as-gather defocus over a CPU-precomputed
// aperture tap table: tap positions are generated inside the chosen aperture shape (unit-radius
// space) and uploaded as a small uniform; the kernels only rotate + scale them. That makes the
// aperture shape a RUNTIME value (switching shapes rewrites 1.6KB — no recompile) and supports
// shapes a per-angle radius function never could (heart, ring).
// ═══════════════════════════════════════════════════════════════════════════════════════
// ── Fold-time constants (Math.* is unproven inside a `'use gpu'` body — kit convention). ──────────
/** Vogel-spiral golden angle: distributes N taps evenly over a disc with no clumping. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))
const BOKEH_TWO_PI = Math.PI * 2

/** Radius UI (0–100) → gather radius in input/canvas pixels. Max ~80px disc — creamy but bounded so
 *  the fixed tap comb stays dense enough to keep discs clean. */
export const BOKEH_RADIUS_SCALE = 0.8
export const bokehRadiusToPixels = (radius: number): number => radius * BOKEH_RADIUS_SCALE

/** Soft-knee width for the highlight boost mask (luminance units above threshold). */
const HIGHLIGHT_KNEE = 0.35
/** Rim brightening: real bokeh discs are brighter at the edge ("soap-bubble" aperture). Baked
 *  constants (not props) — a subtle, always-on lens characteristic. Folded into each tap's
 *  precomputed rim weight (`1 + EDGE_BOOST · edgeProximity^EDGE_POWER`), so only taps near the
 *  aperture boundary lift. */
const EDGE_BOOST = 0.6
const EDGE_POWER = 8.0
/** SDF band (unit-aperture units) inside the boundary over which the rim boost ramps in — the
 *  rejection-sampled shapes' analogue of the Vogel path's radial `rNorm^EDGE_POWER` falloff. */
const RIM_BAND = 0.25
/** Below this pixel radius the gather is a sharp single-tap copy (nothing to defocus). */
const PASSTHROUGH_RADIUS = 0.5

/** How far the cat-eye clipping circle shifts at the frame corner with `catEye` 1 (unit-aperture units; 2 would clip everything). */
const BOKEH_CATEYE_SHIFT = 1.4

/** Gathered-buffer format. rgba16float is storage-writable (the kernel `textureStore`s into it),
 *  sampled/filterable (the fragment bilinear-samples it at canvas resolution) AND holds HDR values
 *  above 1.0 — bright highlight discs keep their energy. Mirrors Glow's BRIGHT_FORMAT choice. */
const BOKEH_OUTPUT_FORMAT = 'rgba16float' as const

/** Fixed gather tap count. Bokeh isn't separable (a shaped disc can't split into two 1D passes),
 *  so taps cost linearly — but the gather runs at the kit blurs' capped compute resolution and the
 *  taps carry NO per-tap trig (positions are CPU-precomputed), so 100 taps lands in the same cost
 *  ballpark as the kit's fixed 2×49-tap separable Gaussian. One tier, no quality knob. */
export const BOKEH_TAP_COUNT = 100

// ═══════════════════════════════════════════════════════════════════════════════════════
// CPU tap-table generation — the aperture lives here, not in the kernel.
//
// Tap positions are precomputed on the CPU inside the chosen aperture shape (unit-radius space)
// and uploaded as a small uniform table; the kernel only rotates + scales them. That makes the
// aperture shape a RUNTIME prop (switching shapes rewrites 1.6KB — no recompile) and supports
// shapes a per-angle radius function never could (heart, ring). Shape math mirrors kit/sdf.ts.
// ═══════════════════════════════════════════════════════════════════════════════════════

export interface BokehTap {
    x: number
    y: number
    rim: number
}

export const BOKEH_SHAPES = ['blades', 'circle', 'star', 'heart', 'flower', 'cross', 'ring'] as const
export type BokehShape = (typeof BOKEH_SHAPES)[number]

/** Star/flower point count from the blades slider — below 3 points the shapes degenerate, so fall
 *  back to the classic 5-point look. */
const resolvePoints = (bladeCount: number): number => {
    const n = Math.round(bladeCount)
    return n >= 3 ? n : 5
}

/** Polygon aperture radius along direction `theta` for a regular `sides`-gon inscribed in the unit
 *  circle (vertices at r=1); `sides < 3` → circle. CPU mirror of the old GPU helper. */
export const bokehPolygonRadius = (theta: number, sides: number): number => {
    if (sides < 3) return 1
    const a = BOKEH_TWO_PI / sides
    const m = theta - a * Math.floor(theta / a + 0.5)
    return Math.cos(a * 0.5) / Math.cos(m)
}

// ── CPU mirrors of kit/sdf.ts shapes (unit aperture, y-down screen convention). ──────────────────
const SQRT2_OVER_4 = Math.SQRT2 / 4

/** Heart (iq) fit inside the unit circle, upright in y-down coords (lobes up, point down). */
const sdfHeart = (dx: number, dy: number): number => {
    const S = 1 / 0.75
    const px = Math.abs(dx) / S
    const py = -dy / S + 0.6
    if (px + py > 1) {
        const ax = px - 0.25
        const ay = py - 0.75
        return (Math.hypot(ax, ay) - SQRT2_OVER_4) * S
    }
    const b1y = py - 1
    const dot1 = px * px + b1y * b1y
    const m = Math.max(px + py, 0) * 0.5
    const b2x = px - m
    const b2y = py - m
    return Math.sqrt(Math.min(dot1, b2x * b2x + b2y * b2y)) * Math.sign(px - py) * S
}

/** Exact segment-based star polygon, outer radius 1. Tips at angle 0 / multiples of the sector. */
const sdfStar = (dx: number, dy: number, sides: number, innerRatio: number): number => {
    const len = Math.hypot(dx, dy)
    const angle = Math.atan2(dy, dx)
    const sector = BOKEH_TWO_PI / sides
    const idx = Math.floor(angle / sector + 0.5)
    const bn = angle - idx * sector
    const fpx = Math.abs(len * Math.sin(bn))
    const fpy = len * Math.cos(bn)
    const an = Math.PI / sides
    const ex = innerRatio * Math.sin(an)
    const ey = innerRatio * Math.cos(an) - 1
    const qx = fpx
    const qy = fpy - 1
    const t = Math.min(Math.max((qx * ex + qy * ey) / (ex * ex + ey * ey), 0), 1)
    const nx = fpx - ex * t
    const ny = fpy - (1 + ey * t)
    return Math.hypot(nx, ny) * Math.sign(ex * qy - ey * qx)
}

/** N-petalled flower (triangular radial wave), outer radius 1. Tips at angle 0 / sector multiples. */
const sdfFlower = (dx: number, dy: number, sides: number, innerRatio: number): number => {
    const angle = Math.atan2(dy, dx)
    const len = Math.hypot(dx, dy)
    const tAngle = (angle * sides) / BOKEH_TWO_PI
    const tFrac = tAngle - Math.floor(tAngle)
    const t = Math.abs(tFrac * 2 - 1)
    return len - (innerRatio + (1 - innerRatio) * t)
}

/** Plus-sign cross sized so rounding keeps the extent within the unit circle. */
const CROSS_SIZE = 0.92
const CROSS_THICKNESS = 0.3
const CROSS_ROUNDING = 0.08
const sdfCross = (dx: number, dy: number): number => {
    const px = Math.abs(dx)
    const py = Math.abs(dy)
    const dHoriz = Math.max(px - CROSS_SIZE, py - CROSS_THICKNESS)
    const dVert = Math.max(py - CROSS_SIZE, px - CROSS_THICKNESS)
    return Math.min(dHoriz, dVert) - CROSS_ROUNDING
}

/** Annulus — donut bokeh (mirror-lens look). Outer edge exactly 1. */
const RING_CENTERLINE = 0.72
const RING_HALF_WIDTH = 0.28
const sdfRing = (dx: number, dy: number): number => Math.abs(Math.hypot(dx, dy) - RING_CENTERLINE) - RING_HALF_WIDTH

/** Halton low-discrepancy sequence — deterministic, evenly-spread rejection candidates. */
const halton = (index: number, base: number): number => {
    let f = 1
    let r = 0
    let i = index
    while (i > 0) {
        f /= base
        r += f * (i % base)
        i = Math.floor(i / base)
    }
    return r
}

/** Star/flower tips point up in y-down screen coords: rotate the candidate so screen-up maps to
 *  the SDF's tip direction (angle 0). */
const toTipUp = (x: number, y: number): [number, number] => [-y, x]

/**
 * Aperture shape configs as DATA: the CPU SDF mirror each rejection-sampled shape samples against
 * (unit aperture, `points` from the blades slider where the shape uses it). `blades`/`circle` are
 * NOT here — they take the exact Vogel-spiral + polygon-radius path below, which is also the
 * fallback for unknown values.
 */
const APERTURE_SDF_SHAPES: Record<string, (x: number, y: number, points: number) => number> = {
    star: (x, y, points) => {
        const [rx, ry] = toTipUp(x, y)
        return sdfStar(rx, ry, points, 0.5)
    },
    flower: (x, y, points) => {
        const [rx, ry] = toTipUp(x, y)
        return sdfFlower(rx, ry, points, 0.55)
    },
    heart: (x, y) => sdfHeart(x, y),
    cross: (x, y) => sdfCross(x, y),
    ring: (x, y) => sdfRing(x, y),
}

/**
 * Precompute `tapCount` gather offsets inside the chosen aperture shape (unit-radius space,
 * unrotated) plus each tap's rim-brightening weight. Circle/blades use the exact Vogel-spiral +
 * polygon-radius math the kernel used to run per tap (pixel-identical look); the other shapes
 * rejection-sample a Halton sequence against the CPU SDF mirrors. Offsets are NEGATED so the
 * VISIBLE disc (a bright point spreads to pixels at `point − offset`) matches the shape's screen
 * orientation — hearts upright, star tips up. Deterministic: same inputs → same table.
 */
export function generateBokehTaps(shape: string, bladeCount: number, tapCount: number): BokehTap[] {
    const taps: BokehTap[] = []

    const shapeSdf = APERTURE_SDF_SHAPES[shape]
    if (!shapeSdf) {
        // 'blades' / 'circle' (and any unknown value): Vogel spiral warped by the polygon radius.
        const sides = shape === 'circle' ? 0 : bladeCount
        for (let i = 0; i < tapCount; i++) {
            const t = (i + 0.5) / tapCount
            const rNorm = Math.sqrt(t)
            const theta = i * GOLDEN_ANGLE
            const r = rNorm * bokehPolygonRadius(theta, sides)
            taps.push({
                x: -Math.cos(theta) * r,
                y: -Math.sin(theta) * r,
                rim: 1 + EDGE_BOOST * Math.pow(rNorm, EDGE_POWER),
            })
        }
        return taps
    }

    const points = resolvePoints(bladeCount)
    const sdf = (x: number, y: number): number => shapeSdf(x, y, points)

    // Rejection-sample the Halton(2,3) sequence over the unit square; every accepted point carries
    // its SDF-derived edge proximity as a precomputed rim weight. All shapes fit the unit circle and
    // cover ≥ ~25% of the square, so the candidate cap is never the limiting factor in practice.
    const MAX_CANDIDATES = 20000
    for (let idx = 1; taps.length < tapCount && idx <= MAX_CANDIDATES; idx++) {
        const x = halton(idx, 2) * 2 - 1
        const y = halton(idx, 3) * 2 - 1
        const dist = sdf(x, y)
        if (dist > 0) continue
        const prox = Math.min(Math.max(1 + dist / RIM_BAND, 0), 1)
        taps.push({x: -x, y: -y, rim: 1 + EDGE_BOOST * prox * prox})
    }
    // Degenerate safety nets: cycle-pad a short table; a fully-empty one falls back to a circle.
    if (taps.length === 0) return generateBokehTaps('circle', 0, tapCount)
    for (let i = 0; taps.length < tapCount; i++) taps.push(taps[i % taps.length])
    // Some shape fits overshoot the unit disc a touch (heart scale, cross corner rounding) —
    // normalize back so `radius` means the same disc size for every aperture.
    const maxLen = taps.reduce((max, tap) => Math.max(max, Math.hypot(tap.x, tap.y)), 0)
    if (maxLen > 1) {
        const inv = 1 / maxLen
        for (const tap of taps) {
            tap.x *= inv
            tap.y *= inv
        }
    }
    return taps
}

/** Map-driven channel (the driving scalar the radius-map source contributes per pixel). Same four
 *  channels the fragment map path supports; a build-time constant so the branch is comptime-folded
 *  (Blur's `BlurMapChannel` precedent). */
export type BokehMapChannel = BlurMapChannel

// ── Uniform-radius kernel params: the gather radius (input px) + look controls + input canvas dims.
// The aperture rotation ships as a CPU-computed cos/sin pair — the kernel just rotates taps. ──
function bokehParams() {
    return d.struct({
        radius: d.f32,
        highlightGain: d.f32,
        highlightThreshold: d.f32,
        rotCos: d.f32,
        rotSin: d.f32,
        chromaticFringe: d.f32,
        catEye: d.f32,
        inputWidth: d.f32,
        inputHeight: d.f32,
    })
}

// ── Map-driven kernel params: look controls + input canvas dims + the remap window (the per-pixel
// radius is sampled from the map SOURCE and remapped, then scaled × RADIUS_SCALE). ──
function bokehMapParams() {
    return d.struct({
        highlightGain: d.f32,
        highlightThreshold: d.f32,
        rotCos: d.f32,
        rotSin: d.f32,
        chromaticFringe: d.f32,
        catEye: d.f32,
        ...MAP_SOURCE_DIM_FIELDS,
        ...REMAP_WINDOW_FIELDS,
    })
}

/**
 * Highlight-weighting stage (the bokeh trick): weight bright taps up so they dominate the gather
 * and form discs instead of washing out. Soft-knee smoothstep above `threshold`; `rim` carries the
 * tap's precomputed rim-brightening lift. Separable from the gather (pure per-tap math), shared by
 * the uniform-radius and map-driven kernels.
 */
export const bokehHighlightWeight = tgpu.fn([d.vec4f, d.f32, d.f32, d.f32], d.f32)(
    (base, threshold, gain, rim) => {
        'use gpu'
        const lum = std.dot(base.xyz, d.vec3f(0.2126, 0.7152, 0.0722))
        const hi = std.smoothstep(threshold, threshold + HIGHLIGHT_KNEE, lum)
        return (1.0 + gain * hi * hi) * rim
    },
).$name('bokehHighlightWeight')

/**
 * GPU-free construction of the single-pass bokeh gather node graph (bind-group layout + kernel fn).
 * Bokeh is NOT separable, so unlike the kit's 2-pass Gaussian this scatters-as-gather over the
 * precomputed aperture tap table in ONE pass: for each output pixel it walks `tapCount` taps (unit
 * aperture offsets, rotated by the uniform cos/sin and scaled by the gather radius), boosts bright
 * taps via the highlight-weighting stage, brightens the rim via each tap's precomputed weight, and
 * optionally fringes the R/B channels for a lens look. Mirrors kit/blur's `buildFixedBlurGraph`
 * style (GPU-free, 2D dispatch — no manual bounds guard, SAMPLED `textureLoad` with a mip level
 * for the input, STORAGE `textureStore` for the output).
 *
 * The tap loop itself is an ATOMIC gather core — irreducible: one gather integral whose taps
 * accumulate into shared color/alpha/weight sums, with the chromatic-fringe taps bound to this
 * graph's layout (conditional textureLoads can't leave the kernel) — so the loop does not
 * decompose past the weighting stage above.
 */
export function buildBokehGraph(computeWidth: number, computeHeight: number, tapCount: number) {
    const TAPS = tapCount
    const Params = bokehParams()
    const TapArray = d.arrayOf(d.vec4f, TAPS)
    const layout = tgpu.bindGroupLayout({
        input: {texture: d.texture2d(d.f32)},
        output: {storageTexture: d.textureStorage2d(BOKEH_OUTPUT_FORMAT, 'write-only')},
        params: {uniform: Params},
        taps: {uniform: TapArray},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const radius = p.radius
        // Map this compute pixel's center into input-pixel space (uniform-backed dims → resize pulls
        // through without recompile). Working in input pixels keeps discs perfectly round.
        const inputCx = (d.f32(cx) + 0.5) * p.inputWidth / computeWidth - 0.5
        const inputCy = (d.f32(cy) + 0.5) * p.inputHeight / computeHeight - 0.5
        const inputWm1 = d.i32(p.inputWidth - 1)
        const inputHm1 = d.i32(p.inputHeight - 1)

        if (radius < PASSTHROUGH_RADIUS) {
            const sx = std.clamp(d.i32(std.round(inputCx)), 0, inputWm1)
            const sy = std.clamp(d.i32(std.round(inputCy)), 0, inputHm1)
            const texel = std.textureLoad(layout.$.input, d.vec2u(d.u32(sx), d.u32(sy)), 0)
            std.textureStore(layout.$.output, d.vec2u(cx, cy), texel)
        } else {
            const gain = p.highlightGain
            const threshold = p.highlightThreshold
            const fringe = p.chromaticFringe
            const rotC = p.rotCos
            const rotS = p.rotSin
            // Optical vignetting (cat-eye): a second unit circle, shifted toward the frame edge by
            // `catEye × distance from centre`, clips the aperture into a lens shape that grows
            // toward the corners. At catEye 0 the shift is zero and every tap stays inside it.
            const aspect = p.inputWidth / p.inputHeight
            const vu = ((d.f32(cx) + 0.5) / computeWidth - 0.5) * aspect
            const vv = (d.f32(cy) + 0.5) / computeHeight - 0.5
            const vLen = std.sqrt(vu * vu + vv * vv)
            const vMax = std.sqrt(0.25 * aspect * aspect + 0.25)
            const shift = p.catEye * (vLen / vMax) * BOKEH_CATEYE_SHIFT / std.max(vLen, 0.0001)
            const barrelShift = d.vec2f(vu * shift, vv * shift)
            let accum = d.vec3f(0, 0, 0)
            let alphaAccum = d.f32(0)
            let wSum = d.f32(0)
            for (let i = 0; i < TAPS; i++) {
                // Precomputed unit-aperture tap: xy = offset, z = rim weight. Rotate + scale only.
                const tap = layout.$.taps[i]
                const ox = tap.x * rotC - tap.y * rotS
                const oy = tap.x * rotS + tap.y * rotC
                const off = d.vec2f(ox * radius, oy * radius)

                const sx = std.clamp(d.i32(std.round(inputCx + off.x)), 0, inputWm1)
                const sy = std.clamp(d.i32(std.round(inputCy + off.y)), 0, inputHm1)
                const base = std.textureLoad(layout.$.input, d.vec2u(d.u32(sx), d.u32(sy)), 0)

                // Chromatic fringe: sample R slightly outward, B slightly inward (lens dispersion).
                // Uniform branch (same fringe for all threads) → coherent, and textureLoad has no
                // derivative requirement, so the conditional loads are legal in compute.
                let colR = base.x
                let colB = base.z
                if (fringe > 0.001) {
                    const offR = off.mul(1.0 + fringe * 0.5)
                    const rx = std.clamp(d.i32(std.round(inputCx + offR.x)), 0, inputWm1)
                    const ry = std.clamp(d.i32(std.round(inputCy + offR.y)), 0, inputHm1)
                    colR = std.textureLoad(layout.$.input, d.vec2u(d.u32(rx), d.u32(ry)), 0).x
                    const offB = off.mul(1.0 - fringe * 0.5)
                    const bx = std.clamp(d.i32(std.round(inputCx + offB.x)), 0, inputWm1)
                    const by = std.clamp(d.i32(std.round(inputCy + offB.y)), 0, inputHm1)
                    colB = std.textureLoad(layout.$.input, d.vec2u(d.u32(bx), d.u32(by)), 0).z
                }
                const col = d.vec3f(colR, base.y, colB)

                // Highlight boost — the shared weighting stage (tap.z carries the rim lift).
                const barrel = 1.0 - std.smoothstep(1.02, 1.12, std.length(d.vec2f(ox, oy).add(barrelShift)))
                const w = bokehHighlightWeight(base, threshold, gain, tap.z) * barrel
                accum = accum.add(col.mul(w))
                alphaAccum = alphaAccum + base.w * w
                wSum = wSum + w
            }
            const invW = 1.0 / std.max(wSum, 0.0001)
            std.textureStore(layout.$.output, d.vec2u(cx, cy), d.vec4f(accum.mul(invW), alphaAccum * invW))
        }
    }).$name('bokehGather')

    return {layout, kernel, Params, TapArray}
}

/**
 * GPU-free construction of the MAP-DRIVEN gather graph. Identical gather to `buildBokehGraph`, but the
 * per-pixel radius is sampled from the radius-map SOURCE (canvas-resolution RTT), remapped through the
 * same window `resolveMapForProp` uses in the fragment path, then scaled × RADIUS_SCALE. The `source`
 * texture is bound late (its RTT is allocated after composition). Mirrors Glow's map-prepass remap math.
 */
export function buildBokehMapGraph(computeWidth: number, computeHeight: number, tapCount: number, channel: BokehMapChannel) {
    const TAPS = tapCount
    const Params = bokehMapParams()
    const TapArray = d.arrayOf(d.vec4f, TAPS)
    const channelScalar = mapSourceScalar(channel)
    const layout = tgpu.bindGroupLayout({
        input: {texture: d.texture2d(d.f32)},
        source: {texture: d.texture2d(d.f32)},
        output: {storageTexture: d.textureStorage2d(BOKEH_OUTPUT_FORMAT, 'write-only')},
        params: {uniform: Params},
        taps: {uniform: TapArray},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const inputCx = (d.f32(cx) + 0.5) * p.inputWidth / computeWidth - 0.5
        const inputCy = (d.f32(cy) + 0.5) * p.inputHeight / computeHeight - 0.5
        const inputWm1 = d.i32(p.inputWidth - 1)
        const inputHm1 = d.i32(p.inputHeight - 1)

        // Per-pixel radius from the map source (nearest tap at the corresponding canvas pixel).
        const u = (d.f32(cx) + 0.5) / computeWidth
        const v = (d.f32(cy) + 0.5) / computeHeight
        const srcX = std.clamp(d.i32(std.round(u * p.inputWidth)), 0, inputWm1)
        const srcY = std.clamp(d.i32(std.round(v * p.inputHeight)), 0, inputHm1)
        const mapSample = std.textureLoad(layout.$.source, d.vec2u(d.u32(srcX), d.u32(srcY)), 0)
        // The shared channel + remap-window path — the same resolve the fragment map path runs.
        const remapped = applyRemapWindow(
            channelScalar(mapSample), p.inputMin, p.inputMax, p.outputMin, p.outputMax, p.curve,
        )
        const radius = remapped * BOKEH_RADIUS_SCALE

        if (radius < PASSTHROUGH_RADIUS) {
            const sx = std.clamp(d.i32(std.round(inputCx)), 0, inputWm1)
            const sy = std.clamp(d.i32(std.round(inputCy)), 0, inputHm1)
            const texel = std.textureLoad(layout.$.input, d.vec2u(d.u32(sx), d.u32(sy)), 0)
            std.textureStore(layout.$.output, d.vec2u(cx, cy), texel)
        } else {
            const gain = p.highlightGain
            const threshold = p.highlightThreshold
            const fringe = p.chromaticFringe
            const rotC = p.rotCos
            const rotS = p.rotSin
            // Optical vignetting (cat-eye): a second unit circle, shifted toward the frame edge by
            // `catEye × distance from centre`, clips the aperture into a lens shape that grows
            // toward the corners. At catEye 0 the shift is zero and every tap stays inside it.
            const aspect = p.inputWidth / p.inputHeight
            const vu = ((d.f32(cx) + 0.5) / computeWidth - 0.5) * aspect
            const vv = (d.f32(cy) + 0.5) / computeHeight - 0.5
            const vLen = std.sqrt(vu * vu + vv * vv)
            const vMax = std.sqrt(0.25 * aspect * aspect + 0.25)
            const shift = p.catEye * (vLen / vMax) * BOKEH_CATEYE_SHIFT / std.max(vLen, 0.0001)
            const barrelShift = d.vec2f(vu * shift, vv * shift)
            let accum = d.vec3f(0, 0, 0)
            let alphaAccum = d.f32(0)
            let wSum = d.f32(0)
            for (let i = 0; i < TAPS; i++) {
                const tap = layout.$.taps[i]
                const ox = tap.x * rotC - tap.y * rotS
                const oy = tap.x * rotS + tap.y * rotC
                const off = d.vec2f(ox * radius, oy * radius)

                const sx = std.clamp(d.i32(std.round(inputCx + off.x)), 0, inputWm1)
                const sy = std.clamp(d.i32(std.round(inputCy + off.y)), 0, inputHm1)
                const base = std.textureLoad(layout.$.input, d.vec2u(d.u32(sx), d.u32(sy)), 0)

                let colR = base.x
                let colB = base.z
                if (fringe > 0.001) {
                    const offR = off.mul(1.0 + fringe * 0.5)
                    const rx = std.clamp(d.i32(std.round(inputCx + offR.x)), 0, inputWm1)
                    const ry = std.clamp(d.i32(std.round(inputCy + offR.y)), 0, inputHm1)
                    colR = std.textureLoad(layout.$.input, d.vec2u(d.u32(rx), d.u32(ry)), 0).x
                    const offB = off.mul(1.0 - fringe * 0.5)
                    const bx = std.clamp(d.i32(std.round(inputCx + offB.x)), 0, inputWm1)
                    const by = std.clamp(d.i32(std.round(inputCy + offB.y)), 0, inputHm1)
                    colB = std.textureLoad(layout.$.input, d.vec2u(d.u32(bx), d.u32(by)), 0).z
                }
                const col = d.vec3f(colR, base.y, colB)

                const barrel = 1.0 - std.smoothstep(1.02, 1.12, std.length(d.vec2f(ox, oy).add(barrelShift)))
                const w = bokehHighlightWeight(base, threshold, gain, tap.z) * barrel
                accum = accum.add(col.mul(w))
                alphaAccum = alphaAccum + base.w * w
                wSum = wSum + w
            }
            const invW = 1.0 / std.max(wSum, 0.0001)
            std.textureStore(layout.$.output, d.vec2u(cx, cy), d.vec4f(accum.mul(invW), alphaAccum * invW))
        }
    }).$name('bokehGatherVariable')

    return {layout, kernel, Params, TapArray}
}

