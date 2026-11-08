/**
 * Grid-kernel builders — the neighbour-reading stencil vocabulary for fixed-grid simulations.
 *
 * Where `scaffolds/fluids` owns the Stable-Fluids solver chain, this module owns the OTHER grid
 * kernel families the sims are built from: a reaction-diffusion stencil step (Gray-Scott), a
 * scatter seed, a state-publish copy, an odd-even transposition sort pass, a feedback
 * advect+diffuse+decay+stamp trail step, a macroblock hold/refresh advection (the corrupted-codec
 * decode rule), and a DCT block-quantize pass (JPEG).
 *
 * Each builder creates its own bind-group layouts and params struct and bakes kernels against
 * them, so a consuming shader stays purely declarative: it passes the LOOK (grid size, densities,
 * shift ranges, variant flags — the constants that ARE the effect) and receives a kernel set plus
 * the layouts to bind its state to. Every option is STRUCTURAL (baked into the WGSL — recompose to
 * change); the params-struct members are the runtime surface.
 *
 * Kernel `$name`s take the consumer's `namePrefix` (the fluids convention): names land verbatim in
 * the emitted WGSL, so this is what keeps per-shader snapshots stable and keeps two grid sims in
 * one tree from colliding on WGSL identifiers.
 */
import {tgpu, d, std} from '../kit'
import * as noise from '../kit/noise'
import * as blend from '../kit/blend'
import * as colorMixing from '../kit/colorMixing'
import {createGuardedCompute, type KitComputePipeline} from '../compute'
import type {TgpuRoot} from 'typegpu'

/** A 2D-dispatched cell kernel over a grid. */
export type GridKernel = (cx: number, cy: number) => void

/**
 * One kernel as a guarded compute pass. The general form of `createFluidKernelPass`: `size` is
 * optional (a dynamically-dispatched pass passes none and drives `dispatchThreads` itself), and
 * non-square grids are allowed.
 */
export function createGridKernelPass(
    root: TgpuRoot,
    kernel: GridKernel,
    opts: {size?: [number, number]; bindGroup?: unknown} = {},
): KitComputePipeline {
    return createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; kernel(cx, cy)}, {
        ...(opts.size ? {size: opts.size} : {}),
        ...(opts.bindGroup ? {bindGroup: opts.bindGroup as never} : {}),
    })
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Gray-Scott reaction-diffusion stencil step
// ════════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Per-cell drive from a sampled texture (the "style map"): the texture's luminance — shaped by
 * contrast/threshold/invert and weighted by alpha coverage — shifts the reaction's operating point
 * per cell. Ranges clamp the shifted point inside the caller's stable band.
 */
export interface ReactionDriveOptions {
    /** Contrast slider 0..1 → luma gain 1..(1+contrastMax). */
    contrastMax: number
    /** Max feed shift the drive can apply. */
    dFeed: number
    /** Max kill shift the drive can apply. */
    dKill: number
    feedMin: number
    feedMax: number
    killMin: number
    killMax: number
}

export interface GrayScottOptions {
    /** Grid resolution per axis (the kernel is dispatched 2D over n×n). */
    n: number
    namePrefix: string
    /** Spatially bias feed/kill from a sampled drive texture (adds `driveTex` to the layout). */
    drive?: ReactionDriveOptions
}

const grayScottParamsSchema = () => d.struct({
    feed: d.f32, kill: d.f32, dv: d.f32,
    brushRadSq: d.f32, brushStrength: d.f32,
    cursorX: d.f32, cursorY: d.f32, cursorActive: d.f32,
    driveInfluence: d.f32, driveContrast: d.f32, driveThreshold: d.f32, driveInvert: d.f32,
    driveMapX: d.f32, driveMapY: d.f32,
    _pad0: d.f32, _pad1: d.f32,
}).$name('GrayScottParams')

type GrayScottParamsSchema = ReturnType<typeof grayScottParamsSchema>

const makeGrayScottPlainLayout = (n: number, Params: GrayScottParamsSchema) => tgpu.bindGroupLayout({
    readBuf: {storage: d.arrayOf(d.vec2f, n * n), access: 'readonly'},
    writeBuf: {storage: d.arrayOf(d.vec2f, n * n), access: 'mutable'},
    params: {uniform: Params},
})
const makeGrayScottDriveLayout = (n: number, Params: GrayScottParamsSchema) => tgpu.bindGroupLayout({
    readBuf: {storage: d.arrayOf(d.vec2f, n * n), access: 'readonly'},
    writeBuf: {storage: d.arrayOf(d.vec2f, n * n), access: 'mutable'},
    params: {uniform: Params},
    driveTex: {texture: d.texture2d(d.f32)},
})
export type GrayScottPlainLayout = ReturnType<typeof makeGrayScottPlainLayout>
export type GrayScottDriveLayout = ReturnType<typeof makeGrayScottDriveLayout>

/**
 * The Gray-Scott reaction + brush-injection step over a vec2f (U,V) cell state.
 *
 * 9-point weighted Laplacian (0.2 orthogonal, 0.05 diagonal, −1 centre) with reflective NO-FLUX
 * boundaries via clamp-to-edge neighbour fetch. Du folds to 1 (the stability ceiling for this
 * explicit scheme), dt folds to 1 (evolution speed is the per-frame iteration count); Dv arrives
 * as the `dv` uniform. A soft Gaussian brush ADDS V around the cursor while `cursorActive > 0`
 * (a uniform → coherent branch, no divergence). With `drive`, the drive texture's shaped luminance
 * shifts feed up / kill down along the density axis per cell.
 *
 * The uniform struct is written IN FULL each frame (16 f32 — trailing pads keep it at a multiple
 * of 16 bytes). The drive members are dead-but-present in the driveless variant so one write path
 * serves both.
 */
export function buildGrayScottStep(opts: GrayScottOptions & {drive: ReactionDriveOptions}): {layout: GrayScottDriveLayout; Params: GrayScottParamsSchema; kernel: GridKernel}
export function buildGrayScottStep(opts: GrayScottOptions & {drive?: undefined}): {layout: GrayScottPlainLayout; Params: GrayScottParamsSchema; kernel: GridKernel}
export function buildGrayScottStep(opts: GrayScottOptions): {layout: GrayScottPlainLayout | GrayScottDriveLayout; Params: GrayScottParamsSchema; kernel: GridKernel} {
    const {n, namePrefix, drive} = opts
    const Params = grayScottParamsSchema()
    const suffix = drive ? 'Drive' : ''
    // Both variants share the kernel body; only the layout (an extra texture entry — a WebGPU
    // pipeline layout cannot make one optional) and the rates fn differ.
    const layout = drive ? makeGrayScottDriveLayout(n, Params) : makeGrayScottPlainLayout(n, Params)
    const driveLayout = drive ? (layout as GrayScottDriveLayout) : null

    // Neighbour fetch with clamp-to-edge addressing → a reflective no-flux boundary at the grid
    // edges (a boundary cell's off-grid neighbour clamps to itself → zero gradient).
    const nbr = tgpu.fn([d.i32, d.i32, d.i32, d.i32], d.vec2f)((ii, ji, di, dj) => {
        'use gpu'
        const row = std.clamp(ii + di, 0, n - 1)
        const col = std.clamp(ji + dj, 0, n - 1)
        return layout.$.readBuf[d.u32(row * n + col)]
    }).$name(`${namePrefix}Nbr${suffix}`)

    // Sample the drive texture for grid cell (cx,cy) — mapped through the caller's display fit
    // (driveMapX/Y = the inverse of the display transform), shaped by contrast/threshold/invert
    // into a signed [-1,1] drive weighted by alpha coverage. The texture is premultiplied (an
    // RTT), so unpremultiply before reading luminance.
    const driveMap = drive && driveLayout ? tgpu.fn([d.u32, d.u32], d.f32)((cx, cy) => {
        'use gpu'
        const p = driveLayout.$.params
        const dims = std.textureDimensions(driveLayout.$.driveTex)
        const fw = d.f32(dims.x)
        const fh = d.f32(dims.y)
        const fx = (d.f32(cx) + 0.5) / d.f32(n)
        const fy = (d.f32(cy) + 0.5) / d.f32(n)
        const cu = std.clamp((fx - 0.5) * p.driveMapX + 0.5, d.f32(0), d.f32(1))
        const cv = std.clamp((fy - 0.5) * p.driveMapY + 0.5, d.f32(0), d.f32(1))
        const px = d.u32(std.min(cu * fw, fw - 1.0))
        const py = d.u32(std.min(cv * fh, fh - 1.0))
        const texel = std.textureLoad(driveLayout.$.driveTex, d.vec2u(px, py), 0)
        const cov = std.clamp(texel.w, d.f32(0), d.f32(1))
        const rgb = texel.xyz.div(std.max(texel.w, d.f32(0.001)))
        const luma = std.dot(rgb, d.vec3f(0.2126, 0.7152, 0.0722))
        const gain = d.f32(1) + p.driveContrast * d.f32(drive.contrastMax)
        const shaped = std.clamp((luma - p.driveThreshold) * gain + 0.5, d.f32(0), d.f32(1))
        let signed = (shaped - 0.5) * 2.0
        signed = std.select(signed, signed * -1.0, p.driveInvert > 0.5)
        return signed * cov
    }).$name(`${namePrefix}DriveMap`) : null

    // Per-cell (feed, kill) operating point. The two variants cannot be one runtime branch: a
    // captured `null` fn has no WGSL form and the transpiler resolves identifiers on dead paths,
    // so the variation is carried by a fn that is always valid.
    const buildDriveRates = (drv: ReactionDriveOptions, dm: NonNullable<typeof driveMap>) =>
        tgpu.fn([d.f32, d.f32, d.u32, d.u32], d.vec2f)((feed, kill, cx, cy) => {
            'use gpu'
            const mod = dm(cx, cy) * layout.$.params.driveInfluence
            return d.vec2f(
                std.clamp(feed + mod * d.f32(drv.dFeed), d.f32(drv.feedMin), d.f32(drv.feedMax)),
                std.clamp(kill - mod * d.f32(drv.dKill), d.f32(drv.killMin), d.f32(drv.killMax)),
            )
        }).$name(`${namePrefix}Rates${suffix}`)
    const rates = drive && driveMap
        ? buildDriveRates(drive, driveMap)
        : tgpu.fn([d.f32, d.f32, d.u32, d.u32], d.vec2f)((feed, kill) => {
            'use gpu'
            return d.vec2f(feed, kill)
        }).$name(`${namePrefix}Rates`)

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const idx = cy * d.u32(n) + cx
        const self = layout.$.readBuf[idx]

        const ii = d.i32(cy)
        const ji = d.i32(cx)
        const L = nbr(ii, ji, 0, -1)
        const R = nbr(ii, ji, 0, 1)
        const T = nbr(ii, ji, -1, 0)
        const B = nbr(ii, ji, 1, 0)
        const TL = nbr(ii, ji, -1, -1)
        const TR = nbr(ii, ji, -1, 1)
        const BL = nbr(ii, ji, 1, -1)
        const BR = nbr(ii, ji, 1, 1)

        // 9-point weighted Laplacian (centre −1).
        const lapU = (L.x + R.x + T.x + B.x) * 0.2 + (TL.x + TR.x + BL.x + BR.x) * 0.05 - self.x
        const lapV = (L.y + R.y + T.y + B.y) * 0.2 + (TL.y + TR.y + BL.y + BR.y) * 0.05 - self.y

        const fk = rates(p.feed, p.kill, cx, cy)
        const feedCell = fk.x
        const killCell = fk.y

        const U = self.x
        const V = self.y
        const uvv = U * V * V
        // Du = 1 (folded), dt = 1 (folded).
        let newU = U + (lapU - uvv + feedCell * (1.0 - U))
        let newV = V + (p.dv * lapV + uvv - (feedCell + killCell) * V)

        // Pointer injection: a soft Gaussian brush ADDS V while the pointer is moving, in
        // grid-cell space. cursorActive is a uniform, so this is a coherent branch that avoids
        // the per-cell exp() on every frame the user isn't dragging.
        if (p.cursorActive > 0.0) {
            const cellX = d.f32(cx) + 0.5
            const cellY = d.f32(cy) + 0.5
            const dxp = cellX - p.cursorX
            const dyp = cellY - p.cursorY
            const distSq = dxp * dxp + dyp * dyp
            const inf = std.exp((distSq / std.max(p.brushRadSq, d.f32(1.0))) * -1.0) * p.cursorActive
            newV = std.min(newV + inf * p.brushStrength, d.f32(1.0))
        }

        newU = std.clamp(newU, 0.0, 1.0)
        newV = std.clamp(newV, 0.0, 1.0)
        layout.$.writeBuf[idx] = d.vec2f(newU, newV)
    }).$name(`${namePrefix}React${suffix}`)

    return {layout, Params, kernel}
}

/**
 * Scatter seed for a vec2f (U,V) cell state: U=1 everywhere, a hash-scattered V=1 seed at
 * `density`. Integer bitcast hash — cell coords reach large-grid extents, where the classic
 * sin-fract hash streaks on iOS Metal.
 */
export function buildScatterSeedKernel(opts: {n: number; density: number; namePrefix: string}) {
    const {n, density, namePrefix} = opts
    const layout = tgpu.bindGroupLayout({
        stateBuf: {storage: d.arrayOf(d.vec2f, n * n), access: 'mutable'},
    })
    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const h = noise.hash12(d.vec2f(d.f32(cx), d.f32(cy)))
        const vSeed = std.select(d.f32(0.0), d.f32(1.0), h < density)
        layout.$.stateBuf[idx] = d.vec2f(1.0, vSeed)
    }).$name(`${namePrefix}Seed`)
    return {layout, kernel}
}

/** Publish a vec2f cell state into a sampled texture (`.rg` = the pair) the fragment reads. */
export function buildPairPublishKernel(opts: {n: number; format: 'rgba16float' | 'rgba32float'; namePrefix: string}) {
    const {n, format, namePrefix} = opts
    const layout = tgpu.bindGroupLayout({
        srcBuf: {storage: d.arrayOf(d.vec2f, n * n), access: 'readonly'},
        outTex: {storageTexture: d.textureStorage2d(format, 'write-only')},
    })
    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const s = layout.$.srcBuf[idx]
        std.textureStore(layout.$.outTex, d.vec2u(cx, cy), d.vec4f(s.x, s.y, 0.0, 0.0))
    }).$name(`${namePrefix}Output`)
    return {layout, kernel}
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Odd-even transposition displacement sort
// ════════════════════════════════════════════════════════════════════════════════════════════════

const brushWeightCache = new Map<number, ReturnType<typeof buildAspectBrushWeight>>()

function buildAspectBrushWeight(n: number) {
    return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)(
        (cxf, cyf, mouseX, mouseY, radius, falloff, aspect) => {
            'use gpu'
            const dx = cxf / n - mouseX
            const dy = cyf / n - mouseY
            const ddx = std.select(dx, dx * aspect, aspect >= 1.0)
            const ddy = std.select(dy / aspect, dy, aspect >= 1.0)
            const dist = std.sqrt(ddx * ddx + ddy * ddy)
            const inner = radius * (1.0 - falloff)
            return 1.0 - std.smoothstep(inner, radius, dist)
        },
    ).$name('aspectBrushWeight')
}

/**
 * Aspect-corrected radial brush weight over an n-cell grid: 1 at the cursor centre → 0 past the
 * radius, with `falloff` softening the edge. Memoized per grid size (one WGSL fn per n).
 */
export function makeAspectBrushWeight(n: number) {
    const hit = brushWeightCache.get(n)
    if (hit) return hit
    const built = buildAspectBrushWeight(n)
    brushWeightCache.set(n, built)
    return built
}

export interface OddEvenSortOptions {
    /** Grid resolution per axis. */
    n: number
    format: 'rgba16float' | 'rgba32float'
    namePrefix: string
    /** Sort along the vertical axis (columns) instead of horizontal (rows). */
    vertical: boolean
    /** Sort order along the axis: +1 ascending key, −1 descending. */
    direction: 1 | -1
}

/**
 * The odd-even transposition sort kernel set over a persistent per-cell DISPLACEMENT map.
 *
 * State is a per-cell f32 offset from identity along the sort axis (so `decay` can melt cells back
 * home); the sort key is cached once per frame by the `key` prepass (input luminance, unpremultiplied)
 * and lerped at fractional coordinates (luma is linear in RGB → lerp-of-luma == luma-of-lerp). Each
 * compare-swap pass alternates parity; both members of a pair derive the same exchange decision from
 * the same two cached keys → lock-free consistent swap. Swaps are gated by an aspect-corrected brush
 * around the cursor, dithered per cell for a soft edge. The publish kernel writes the normalised
 * source coordinate along the axis into the state texture the fragment remaps with.
 */
export function buildOddEvenSortSet(opts: OddEvenSortOptions) {
    const {n, format, namePrefix, vertical, direction} = opts
    const count = n * n
    const nF = n
    const maxC = n - 1

    const LumaParams = d.struct({inputWidth: d.f32, inputHeight: d.f32}).$name('SortKeyParams')
    const SwapParams = d.struct({
        mouseX: d.f32, mouseY: d.f32, radius: d.f32, falloff: d.f32, decay: d.f32, aspect: d.f32, seed: d.f32,
    }).$name('SortSwapParams')

    const lumaLayout = tgpu.bindGroupLayout({
        input: {texture: d.texture2d(d.f32)},
        lumaBuf: {storage: d.arrayOf(d.f32, count), access: 'mutable'},
        params: {uniform: LumaParams},
    })
    const swapLayout = tgpu.bindGroupLayout({
        readBuf: {storage: d.arrayOf(d.f32, count), access: 'readonly'},
        writeBuf: {storage: d.arrayOf(d.f32, count), access: 'mutable'},
        lumaBuf: {storage: d.arrayOf(d.f32, count), access: 'readonly'},
        params: {uniform: SwapParams},
    })
    const outputLayout = tgpu.bindGroupLayout({
        srcBuf: {storage: d.arrayOf(d.f32, count), access: 'readonly'},
        stateTex: {storageTexture: d.textureStorage2d(format, 'write-only')},
    })

    const brushWeight = makeAspectBrushWeight(n)

    // Key prepass: cache each cell's input luma once per frame (the swap passes read the cache).
    const key = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = lumaLayout.$.params
        const idx = cy * d.u32(n) + cx
        const u = (d.f32(cx) + 0.5) / nF
        const v = (d.f32(cy) + 0.5) / nF
        const ix = std.clamp(d.i32(std.round(u * p.inputWidth)), 0, d.i32(p.inputWidth - 1.0))
        const iy = std.clamp(d.i32(std.round(v * p.inputHeight)), 0, d.i32(p.inputHeight - 1.0))
        const texel = std.textureLoad(lumaLayout.$.input, d.vec2u(d.u32(ix), d.u32(iy)), 0)
        const rgb = blend.unpremultiplyAlpha(texel).xyz
        lumaLayout.$.lumaBuf[idx] = std.dot(rgb, d.vec3f(0.299, 0.587, 0.114))
    }).$name(`${namePrefix}Luma`)

    // Compare-swap kernel: `parity`, `vertical` and `direction` are compile-time constants baked
    // into the WGSL.
    const makeSwap = (parity: number) => tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = swapLayout.$.params
        const idx = cy * d.u32(n) + cx
        const xf = d.f32(cx)
        const yf = d.f32(cy)

        const lineC = vertical ? cy : cx
        const fixedC = vertical ? cx : cy
        const lineCf = d.f32(lineC)
        const lineCi = d.i32(lineC)

        const myOff = swapLayout.$.readBuf[idx]
        const myCol = std.clamp(lineCf + myOff, 0.0, maxC)

        const isLeft = (lineC + d.u32(parity)) % d.u32(2) === d.u32(0)
        const partnerLineI = std.select(lineCi - 1, lineCi + 1, isLeft)
        const valid = partnerLineI >= 0 && partnerLineI < n
        const pClamped = std.clamp(partnerLineI, 0, n - 1)
        const pClampedU = d.u32(pClamped)

        const partnerX = vertical ? fixedC : pClampedU
        const partnerY = vertical ? pClampedU : fixedC
        const partnerIdx = partnerY * d.u32(n) + partnerX
        const partnerOff = swapLayout.$.readBuf[partnerIdx]
        const partnerCol = std.clamp(d.f32(pClamped) + partnerOff, 0.0, maxC)

        // Sort key at a (fractional) line coord: lerp between the two adjacent cached cells.
        const myC0 = std.clamp(std.floor(myCol), 0.0, maxC)
        const myC1 = std.min(myC0 + 1.0, maxC)
        const myIdx0 = vertical ? d.u32(myC0) * d.u32(n) + fixedC : fixedC * d.u32(n) + d.u32(myC0)
        const myIdx1 = vertical ? d.u32(myC1) * d.u32(n) + fixedC : fixedC * d.u32(n) + d.u32(myC1)
        const myKey = std.mix(swapLayout.$.lumaBuf[myIdx0], swapLayout.$.lumaBuf[myIdx1], myCol - myC0)

        const pC0 = std.clamp(std.floor(partnerCol), 0.0, maxC)
        const pC1 = std.min(pC0 + 1.0, maxC)
        const pIdx0 = vertical ? d.u32(pC0) * d.u32(n) + fixedC : fixedC * d.u32(n) + d.u32(pC0)
        const pIdx1 = vertical ? d.u32(pC1) * d.u32(n) + fixedC : fixedC * d.u32(n) + d.u32(pC1)
        const partnerKey = std.mix(swapLayout.$.lumaBuf[pIdx0], swapLayout.$.lumaBuf[pIdx1], partnerCol - pC0)

        // Soft brush gate, dithered by a per-cell random for a soft edge.
        const wMy = brushWeight(xf, yf, p.mouseX, p.mouseY, p.radius, p.falloff, p.aspect)
        const wP = brushWeight(d.f32(partnerX), d.f32(partnerY), p.mouseX, p.mouseY, p.radius, p.falloff, p.aspect)
        const rnd = noise.hash13(d.vec3f(xf, yf, p.seed))
        const gate = valid && std.min(wMy, wP) > rnd

        // Take the partner's column if this cell holds the wrong-ranked pixel.
        const keyDiff = (myKey - partnerKey) * direction
        const takePartner = std.select(keyDiff < 0.0, keyDiff > 0.0, isLeft)
        const newCol = std.select(myCol, partnerCol, gate && takePartner)

        // Decay every cell back toward home; store the offset from identity.
        const decayed = std.mix(newCol, lineCf, p.decay)
        swapLayout.$.writeBuf[idx] = decayed - lineCf
    }).$name(`${namePrefix}Swap`)

    // Publish: per cell, the normalised source coordinate along the sort axis.
    const output = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(n) + cx
        const lineC = vertical ? cy : cx
        const col = std.clamp(d.f32(lineC) + outputLayout.$.srcBuf[idx], 0.0, maxC)
        const u = (col + 0.5) / nF
        std.textureStore(outputLayout.$.stateTex, d.vec2u(cx, cy), d.vec4f(u, 0.0, 0.0, 1.0))
    }).$name(`${namePrefix}Output`)

    return {lumaLayout, LumaParams, swapLayout, SwapParams, outputLayout, key, swap0: makeSwap(0), swap1: makeSwap(1), output}
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Feedback trail step (advect → diffuse+decay → motion gate → stamp)
// ════════════════════════════════════════════════════════════════════════════════════════════════

/**
 * Frame-difference motion mask: how much a pixel changed since last frame, smoothstepped over
 * [threshold, 2·threshold] into a 0–1 stamp weight. Premultiplied RGB luma difference + alpha
 * difference, so both color motion (opaque footage) and coverage motion (moving shapes) trigger.
 */
export const frameDiffMask = tgpu.fn([d.vec4f, d.vec4f, d.f32], d.f32)((live, prev, threshold) => {
    'use gpu'
    const dRgb = std.abs(live.xyz.sub(prev.xyz))
    const dA = std.abs(live.w - prev.w)
    const diff = std.dot(dRgb, d.vec3f(0.299, 0.587, 0.114)) + dA
    return std.smoothstep(threshold, threshold * 2.0 + 0.001, diff)
}).$name('frameDiffMask')

/** One turn in radians (pre-folded — no Math.* inside 'use gpu' bodies). */
const HUE_TURN_RAD = 6.28318530718

/** Rotate an RGB color's hue by `amount` turns (via HSL). */
export const hueRotateTurns = tgpu.fn([d.vec3f, d.f32], d.vec3f)((rgb, amount) => {
    'use gpu'
    // rgbToHsl's hue is in RADIANS: rotate by amount × 2π and wrap into [0, 2π) (hslToRgb also
    // wraps, but keep the value tidy). The old `fract(h + amount)` treated radians as turns and
    // squashed every hue into [0, 1) rad (red→orange only).
    const hsl = colorMixing.rgbToHsl(rgb)
    const turned = hsl.x / HUE_TURN_RAD + amount
    const rotated = d.vec3f(std.fract(turned) * HUE_TURN_RAD, hsl.y, hsl.z)
    return colorMixing.hslToRgb(rotated)
}).$name('hueRotateTurns')

export interface FeedbackTrailOptions {
    /** State resolution per axis. */
    res: number
    format: 'rgba16float' | 'rgba32float'
    namePrefix: string
    /** What sheds the trail: frame-difference motion, or the live frame's alpha coverage. */
    source: 'motion' | 'alpha'
    /** Rotate the decayed trail's hue a touch each frame (compounds → a spectrum down the trail). */
    hueCycle: boolean
    /** Tent-blur half-offset in UV at diffusion 1 (small — it compounds every frame). */
    diffuseHalfUv: number
    /** Drift prop (−1..1) → UV per second before the dt scale. */
    driftScale: number
}

/**
 * One feedback-trail state step: advect the previous accumulation (zoom about centre + drift),
 * soften it with a four-tap bilinear tent, fade it by `persistence`, optionally hue-cycle it, then
 * stamp the live frame back on top (source-over alpha; screen/add RGB variants for light-painting).
 *
 * Motion source gates the stamp by the frame-difference mask and writes the display copy BEFORE
 * the stamp (ghosts only — the fragment lays them over the live frame); it ping-pongs a prev-live
 * copy through `prevLive`/`prevLiveOut`. Alpha source stamps unconditionally, publishes the
 * stamped state, and never touches the prev-live pair (bind 1×1 stubs).
 *
 * Params ABI (runtime surface): persistence, diffusion, driftX, driftY, dt, zoom, hueRate,
 * blendMode (0 normal · 1 screen · 2 add), motionThreshold.
 */
export function buildFeedbackTrailStep(opts: FeedbackTrailOptions) {
    const {res, format, namePrefix, source, hueCycle, diffuseHalfUv, driftScale} = opts

    const Params = d.struct({
        persistence: d.f32,
        diffusion: d.f32,
        driftX: d.f32, driftY: d.f32, dt: d.f32,
        zoom: d.f32,
        hueRate: d.f32,
        blendMode: d.f32,
        motionThreshold: d.f32,
    }).$name('FeedbackTrailParams')

    const layout = tgpu.bindGroupLayout({
        src: {texture: d.texture2d(d.f32)},
        prev: {texture: d.texture2d(d.f32)},
        next: {storageTexture: d.textureStorage2d(format, 'write-only')},
        display: {storageTexture: d.textureStorage2d(format, 'write-only')},
        prevLive: {texture: d.texture2d(d.f32)},
        prevLiveOut: {storageTexture: d.textureStorage2d(format, 'write-only')},
        // A filtering sampler (clamp-to-edge): `textureSampleLevel` is stage-agnostic, so the
        // compute kernel gets hardware bilinear instead of 4 texel loads per tap.
        samp: {sampler: 'filtering'},
        params: {uniform: Params},
    })

    const sampleHistory = tgpu.fn([d.f32, d.f32], d.vec4f)((uvx, uvy) => {
        'use gpu'
        return std.textureSampleLevel(layout.$.prev, layout.$.samp, d.vec2f(uvx, uvy), d.f32(0))
    }).$name(`${namePrefix}SampleHistory`)

    const sampleLive = tgpu.fn([d.f32, d.f32], d.vec4f)((uvx, uvy) => {
        'use gpu'
        return std.textureSampleLevel(layout.$.src, layout.$.samp, d.vec2f(uvx, uvy), d.f32(0))
    }).$name(`${namePrefix}SampleLive`)

    /** This texel's centre in normalized state UV. */
    const texelUv = tgpu.fn([d.u32, d.u32], d.vec2f)((cx, cy) => {
        'use gpu'
        const r = d.f32(res)
        return d.vec2f((d.f32(cx) + 0.5) / r, (d.f32(cy) + 0.5) / r)
    }).$name(`${namePrefix}TexelUv`)

    /**
     * Feedback UV: zoom around the centre then translate by the time-scaled drift. `zoom` arrives
     * already raised to the dt-scaled power on the CPU, so both axes are framerate-independent.
     */
    const feedbackUv = tgpu.fn([d.vec2f], d.vec2f)((uv) => {
        'use gpu'
        const P = layout.$.params
        const zoom = std.max(P.zoom, 0.01)
        return d.vec2f(
            (uv.x - 0.5) / zoom + 0.5 - P.driftX * P.dt * driftScale,
            (uv.y - 0.5) / zoom + 0.5 - P.driftY * P.dt * driftScale,
        )
    }).$name(`${namePrefix}FeedbackUv`)

    /**
     * Advect + soften + fade: four bilinear taps at the tent's diagonal quarter-offsets, averaged
     * (each hardware tap already blends 2×2 texels, so four diagonals cover a 3×3 footprint at 4
     * fetches). At diffusion 0 all taps collapse to the centre → identity.
     */
    const decayed = tgpu.fn([d.f32, d.f32], d.vec4f)((bx, by) => {
        'use gpu'
        const P = layout.$.params
        const h = P.diffusion * diffuseHalfUv
        const sum = sampleHistory(bx - h, by - h)
            .add(sampleHistory(bx + h, by - h))
            .add(sampleHistory(bx - h, by + h))
            .add(sampleHistory(bx + h, by + h))
        return sum.mul(P.persistence * 0.25)
    }).$name(`${namePrefix}Decayed`)

    // Hue-cycling variant: a SEPARATE fn (not a runtime branch) so the HSL round-trip is absent
    // from the WGSL entirely unless selected. Unpremultiply → rotate → repremultiply.
    const decayedTinted = tgpu.fn([d.f32, d.f32], d.vec4f)((bx, by) => {
        'use gpu'
        const dec = decayed(bx, by)
        const a = std.max(dec.w, 0.0001)
        const rotated = hueRotateTurns(dec.xyz.div(a), layout.$.params.hueRate)
        return d.vec4f(rotated.mul(dec.w), dec.w)
    }).$name(`${namePrefix}DecayedTinted`)

    const trailFn = hueCycle ? decayedTinted : decayed

    /**
     * Stamp the live frame (gated by `m`) into the trail, premultiplied. Alpha is plain source-over
     * for every mode; only the RGB combine differs (screen / add build glowing trails).
     */
    const stamp = tgpu.fn([d.vec4f, d.f32, d.vec4f], d.vec4f)((live, m, trail) => {
        'use gpu'
        const P = layout.$.params
        const s = d.vec4f(live.xyz.mul(m), live.w * m)
        const outA = s.w + trail.w * (1.0 - s.w)
        const rgbNormal = s.xyz.add(trail.xyz.mul(1.0 - s.w))
        const rgbScreen = s.xyz.add(trail.xyz).sub(s.xyz.mul(trail.xyz))
        const rgbAdd = s.xyz.add(trail.xyz)
        const rgbSN = std.select(rgbNormal, rgbScreen, P.blendMode > 0.5)
        return d.vec4f(std.select(rgbSN, rgbAdd, P.blendMode > 1.5), outA)
    }).$name(`${namePrefix}Stamp`)

    const kernel = source === 'motion'
        ? tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const uv = texelUv(cx, cy)
            const b = feedbackUv(uv)
            const trail = trailFn(b.x, b.y)
            const live = sampleLive(uv.x, uv.y)
            const pl = std.textureLoad(layout.$.prevLive, d.vec2u(cx, cy), 0)
            const m = frameDiffMask(live, pl, layout.$.params.motionThreshold)
            std.textureStore(layout.$.next, d.vec2u(cx, cy), stamp(live, m, trail))
            std.textureStore(layout.$.display, d.vec2u(cx, cy), trail)
            std.textureStore(layout.$.prevLiveOut, d.vec2u(cx, cy), live)
        }).$name(`${namePrefix}Step`)
        : tgpu.fn([d.u32, d.u32])((cx, cy) => {
            'use gpu'
            const uv = texelUv(cx, cy)
            const b = feedbackUv(uv)
            const trail = trailFn(b.x, b.y)
            const live = sampleLive(uv.x, uv.y)
            const outC = stamp(live, d.f32(1), trail)
            std.textureStore(layout.$.next, d.vec2u(cx, cy), outC)
            std.textureStore(layout.$.display, d.vec2u(cx, cy), outC)
        }).$name(`${namePrefix}StepAlpha`)

    return {layout, Params, kernel}
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Macroblock hold/refresh advection (the corrupted-codec decode rule)
// ════════════════════════════════════════════════════════════════════════════════════════════════

export interface MacroblockAdvectOptions {
    /** State resolution per axis. */
    res: number
    format: 'rgba16float' | 'rgba32float'
    namePrefix: string
    /** Coarse-lattice block scale multiplier and the hash fraction picked coarse. */
    blockMix?: {coarseScale: number; coarsePick: number}
    /** Held-pixel generation loss: quantize levels, quantize blend amount, per-frame energy decay. */
    generationLoss?: {levels: number; amount: number; decay: number}
}

/**
 * One decode step of a macroblock hold/refresh advection — what a video decoder does when P-frame
 * motion vectors keep arriving after the I-frame was dropped. Each state texel picks its
 * macroblock (two mixed scales re-rolled on a slow lattice clock), advances the block's OWN
 * hash-phased epoch clock (no global heartbeat — blocks lock, drift and recover independently),
 * decides held vs fresh against a drifting product-wave corruption field, then either advects the
 * previous state along the block's flow-field motion vector (bilinear, with subtle generation
 * loss + residual re-key back toward the live source) or re-keys from the live input (bilinear).
 * Off-canvas advection sources re-key (a clamped source would conveyor-belt the border color).
 *
 * The block hash chain is a ladder of shared intermediates read by every later stage — it does not
 * split into smaller parts without recomputing rungs. Params ABI: time, dt, seed, intensity,
 * blockSize, drift, churn.
 */
export function buildMacroblockAdvectKernel(opts: MacroblockAdvectOptions) {
    const {res, format, namePrefix} = opts
    const coarseScale = opts.blockMix?.coarseScale ?? 2.6
    const coarsePick = opts.blockMix?.coarsePick ?? 0.35
    const quantLevels = opts.generationLoss?.levels ?? 18
    const quantAmount = opts.generationLoss?.amount ?? 0.12
    const energyDecay = opts.generationLoss?.decay ?? 0.9985

    const Params = d.struct({
        time: d.f32, dt: d.f32, seed: d.f32,
        intensity: d.f32, blockSize: d.f32, drift: d.f32, churn: d.f32,
    }).$name('MacroblockAdvectParams')

    const layout = tgpu.bindGroupLayout({
        src: {texture: d.texture2d(d.f32)},
        prev: {texture: d.texture2d(d.f32)},
        next: {storageTexture: d.textureStorage2d(format, 'write-only')},
        display: {storageTexture: d.textureStorage2d(format, 'write-only')},
        params: {uniform: Params},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const P = layout.$.params
        const resF = d.f32(res)
        const px = d.f32(cx)
        const py = d.f32(cy)
        const fx = (px + 0.5) / resF
        const fy = (py + 0.5) / resF

        // Per-block re-roll rate: every block's clock is phase-offset by its own hash AND its
        // period is jittered, so re-rolls never synchronize across the screen.
        const churnRate = 0.15 + P.churn * 0.8

        // Macroblock: a coarse lattice hash picks fine or coarse blocks per region (mixed-scale
        // macroblocks read as real codec partitioning). The partition re-rolls very slowly.
        const bsF = std.max(P.blockSize, 8.0)
        const coarse = bsF * coarseScale
        const cbx = std.floor(px / coarse)
        const cby = std.floor(py / coarse)
        const cPhase = noise.hash11(cbx * 91.17 + cby * 41.23 + P.seed * 3.7)
        const cEpoch = std.floor(P.time * churnRate * 0.25 + cPhase)
        const scalePick = noise.hash11(cbx * 12.9898 + cby * 78.233 + cEpoch * 37.71 + P.seed * 3.7)
        const bs = std.select(bsF, coarse, scalePick < coarsePick)
        const bx = std.floor(px / bs)
        const by = std.floor(py / bs)

        // The block's own epoch clock: phase AND rate jittered per block.
        const bPhase = noise.hash11(bx * 127.1 + by * 311.7 + P.seed * 17.9)
        const bRate = churnRate * (0.7 + 0.6 * noise.hash11(bx * 53.7 + by * 97.3 + P.seed * 5.3))
        const bEpoch = std.floor(P.time * bRate + bPhase)
        const bh = bx * 127.1 + by * 311.7 + bEpoch * 74.7 + P.seed * 17.9
        const h1 = noise.hash11(bh + 0.13)
        const h2 = noise.hash11(bh + 7.77)
        const h3 = noise.hash11(bh + 23.19)

        // Corruption comes in EPISODES: localized hotspots of a product-wave field drift and pulse
        // across the canvas; the per-block hash roughens only the zone borders, and the hold is an
        // ANALOG strength (soft smoothstep) so zone edges feather.
        const bcx = (bx + 0.5) * bs / resF
        const bcy = (by + 0.5) * bs / resF
        const field = 0.5
            + 0.25 * std.sin(bcx * 2.7 + P.time * 0.31 + P.seed) * std.sin(bcy * 2.1 - P.time * 0.23 + P.seed * 0.4)
            + 0.25 * std.sin((bcx + bcy) * 1.6 + P.time * 0.17 + P.seed * 0.7)
        const thresh = 1.0 - P.intensity * 0.55
        const hold = std.smoothstep(thresh - 0.09, thresh + 0.09, field + (h1 - 0.5) * 0.3)

        // Motion vectors from a smooth flow field (+ per-block jitter): neighboring blocks smear
        // coherently, and the field slowly reorients.
        const flowAng = std.sin(bcx * 1.9 + P.seed * 3.1) * 1.7
            + std.cos(bcy * 2.3 - P.seed * 2.7) * 1.3
            + P.time * 0.15
        const ang = flowAng + (h2 - 0.5) * 1.1
        const mag = (0.5 + 0.5 * h3) * P.drift * P.dt * 10.0 * (bs / resF)
        // Off-canvas advection source → treat as a dropped reference and re-key instead.
        const suRaw = fx - std.cos(ang) * mag
        const svRaw = fy - std.sin(ang) * mag
        const inBounds = std.select(d.f32(0), d.f32(1), suRaw >= 0.0)
            * std.select(d.f32(0), d.f32(1), suRaw <= 1.0)
            * std.select(d.f32(0), d.f32(1), svRaw >= 0.0)
            * std.select(d.f32(0), d.f32(1), svRaw <= 1.0)
        const su = std.clamp(suRaw, 0.0, 1.0) * resF - 0.5
        const sv = std.clamp(svRaw, 0.0, 1.0) * resF - 0.5

        // Bilinear advection tap from the previous state.
        const sx0 = std.floor(su)
        const sy0 = std.floor(sv)
        const sfx = su - sx0
        const sfy = sv - sy0
        const maxT = resF - 1.0
        const ix0 = d.u32(std.clamp(sx0, d.f32(0), maxT))
        const ix1 = d.u32(std.clamp(sx0 + 1.0, d.f32(0), maxT))
        const iy0 = d.u32(std.clamp(sy0, d.f32(0), maxT))
        const iy1 = d.u32(std.clamp(sy0 + 1.0, d.f32(0), maxT))
        const p00 = std.textureLoad(layout.$.prev, d.vec2u(ix0, iy0), 0)
        const p10 = std.textureLoad(layout.$.prev, d.vec2u(ix1, iy0), 0)
        const p01 = std.textureLoad(layout.$.prev, d.vec2u(ix0, iy1), 0)
        const p11 = std.textureLoad(layout.$.prev, d.vec2u(ix1, iy1), 0)
        const prevC = std.mix(std.mix(p00, p10, sfx), std.mix(p01, p11, sfx), sfy)

        // Generation loss: quantize toward a coarse palette + slow energy decay.
        const quant = std.floor(prevC.xyz.mul(quantLevels)).add(d.vec3f(0.5, 0.5, 0.5)).div(quantLevels)
        const heldRgb = std.mix(prevC.xyz, quant, quantAmount).mul(energyDecay)
        const held = d.vec4f(heldRgb, prevC.w)

        // Residual re-key: held pixels bleed live content back in every frame (τ ≈ 0.3–0.8s by
        // intensity), so the smears stay anchored and the loop self-heals.
        const rekey = std.clamp(P.dt * (1.2 + (1.0 - P.intensity) * 2.0), 0.0, 0.2)

        // Fresh: bilinear re-key from the live input (size queried from the texture itself — CPU
        // dimensions can disagree with the actual RTT allocation).
        const srcDims = std.textureDimensions(layout.$.src)
        const srcW = d.f32(srcDims.x)
        const srcH = d.f32(srcDims.y)
        const cu = fx * srcW - 0.5
        const cv = fy * srcH - 0.5
        const cx0 = std.floor(cu)
        const cy0 = std.floor(cv)
        const cfx = cu - cx0
        const cfy = cv - cy0
        const jx0 = d.u32(std.clamp(cx0, d.f32(0), srcW - 1.0))
        const jx1 = d.u32(std.clamp(cx0 + 1.0, d.f32(0), srcW - 1.0))
        const jy0 = d.u32(std.clamp(cy0, d.f32(0), srcH - 1.0))
        const jy1 = d.u32(std.clamp(cy0 + 1.0, d.f32(0), srcH - 1.0))
        const c00 = std.textureLoad(layout.$.src, d.vec2u(jx0, jy0), 0)
        const c10 = std.textureLoad(layout.$.src, d.vec2u(jx1, jy0), 0)
        const c01 = std.textureLoad(layout.$.src, d.vec2u(jx0, jy1), 0)
        const c11 = std.textureLoad(layout.$.src, d.vec2u(jx1, jy1), 0)
        const fresh = std.mix(std.mix(c00, c10, cfx), std.mix(c01, c11, cfx), cfy)

        const heldAnchored = std.mix(held, fresh, rekey)
        const outC = std.mix(fresh, heldAnchored, hold * inBounds)
        std.textureStore(layout.$.next, d.vec2u(cx, cy), outC)
        std.textureStore(layout.$.display, d.vec2u(cx, cy), outC)
    }).$name(`${namePrefix}Step`)

    return {layout, Params, kernel}
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// DCT block quantize (JPEG)
// ════════════════════════════════════════════════════════════════════════════════════════════════

// Canonical JPEG Annex K / IJG quantization tables (indexed [v*8 + u]).
const LUMA_QUANT = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]
const CHROMA_QUANT = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]

// DCT cosine basis LUT: COS_LUT[k*8 + u] = cos((2k+1)·u·π/16). Flattened for a runtime-indexable
// tgpu.const (a TGSL body cannot index a captured JS array). Pre-folded at module init since
// `Math.*` inside a `'use gpu'` body is unproven.
const COS_LUT_FLAT: number[] = []
for (let k = 0; k < 8; k++) {
    for (let u = 0; u < 8; u++) COS_LUT_FLAT[k * 8 + u] = Math.cos(((2 * k + 1) * u * Math.PI) / 16)
}
const SQRT1_2 = Math.SQRT1_2
const PI_OVER_16 = Math.PI / 16

const COS_LUT = tgpu.const(d.arrayOf(d.f32, 64), COS_LUT_FLAT)
const LUMA_QUANT_C = tgpu.const(d.arrayOf(d.f32, 64), LUMA_QUANT)
const CHROMA_QUANT_C = tgpu.const(d.arrayOf(d.f32, 64), CHROMA_QUANT)

const jpegBlockCache = new Map<number, ReturnType<typeof buildJpegBlockRgb>>()

function buildJpegBlockRgb(nLuma: number) {
    return tgpu.fn([d.arrayOf(d.vec3f, 64), d.f32, d.f32, d.f32], d.vec3f)(
        (block, localX, localY, quality) => {
            'use gpu'
            const qualityC = std.clamp(quality, 1.0, 100.0)
            // qualityC < 50 ? 5000/qualityC : 200 - qualityC*2 (std.select arg order: false, true, cond)
            const qScale = std.select(200.0 - qualityC * 2.0, 5000.0 / qualityC, qualityC < 50.0)

            // Level-shifted YCbCr of the 8×8 block.
            const Yv = d.arrayOf(d.f32, 64)()
            const Cbv = d.arrayOf(d.f32, 64)()
            const Crv = d.arrayOf(d.f32, 64)()
            for (let i = 0; i < 64; i++) {
                const c = block[i]
                Yv[i] = c.x * 0.299 + c.y * 0.587 + c.z * 0.114 - 0.5
                Cbv[i] = c.x * -0.168736 + c.y * -0.331264 + c.z * 0.5
                Crv[i] = c.x * 0.5 + c.y * -0.418688 + c.z * -0.081312
            }

            // ── Luma: separable forward DCT → quantize → inverse DCT at (localX, localY). ──
            const rowCoef = d.arrayOf(d.f32, 8 * nLuma)()
            for (let y = 0; y < 8; y++) {
                for (let u = 0; u < nLuma; u++) {
                    let sum = d.f32(0.0)
                    for (let x = 0; x < 8; x++) sum = sum + Yv[y * 8 + x] * COS_LUT.$[x * 8 + u]
                    rowCoef[y * nLuma + u] = sum
                }
            }
            let acc = d.f32(0.0)
            for (let v = 0; v < nLuma; v++) {
                const av = std.select(1.0, SQRT1_2, v === 0)
                const cy = std.cos((localY * 2.0 + 1.0) * (d.f32(v) * PI_OVER_16))
                for (let u = 0; u < nLuma; u++) {
                    const au = std.select(1.0, SQRT1_2, u === 0)
                    let colSum = d.f32(0.0)
                    for (let y = 0; y < 8; y++) colSum = colSum + rowCoef[y * nLuma + u] * COS_LUT.$[y * 8 + v]
                    const G = colSum * (0.25 * au * av)
                    // qStep = floor((qScale·base + 50)/100) clamped [1,255] / 255
                    const q = std.clamp(std.floor((qScale * LUMA_QUANT_C.$[v * 8 + u] + 50.0) / 100.0), 1.0, 255.0) / 255.0
                    const Gq = std.floor(G / q + 0.5) * q
                    const cxv = std.cos((localX * 2.0 + 1.0) * (d.f32(u) * PI_OVER_16))
                    acc = acc + Gq * (au * av) * cxv * cy
                }
            }
            const Y = acc * 0.25 + 0.5

            // ── Chroma: DC only (flat color per block). ──
            let sumCb = d.f32(0.0)
            let sumCr = d.f32(0.0)
            for (let i = 0; i < 64; i++) {
                sumCb = sumCb + Cbv[i]
                sumCr = sumCr + Crv[i]
            }
            const qChroma = std.clamp(std.floor((qScale * CHROMA_QUANT_C.$[0] + 50.0) / 100.0), 1.0, 255.0) / 255.0
            const GdcCb = sumCb / 8.0
            const GdcCr = sumCr / 8.0
            const Cb = (std.floor(GdcCb / qChroma + 0.5) * qChroma) / 8.0
            const Cr = (std.floor(GdcCr / qChroma + 0.5) * qChroma) / 8.0

            // YCbCr → RGB (Cb/Cr centred at 0).
            const r = Y + Cr * 1.402
            const g = Y - Cb * 0.344136 - Cr * 0.714136
            const b = Y + Cb * 1.772
            return std.clamp(d.vec3f(r, g, b), d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0))
        },
    ).$name('jpegBlockRgb')
}

/**
 * The JPEG block color for one cell: forward separable DCT (rows→columns) over a 64-sample RGB
 * block → IJG-table quantize at `quality` (the IJG curve `qScale = q<50 ? 5000/q : 200−2q`) →
 * inverse DCT at `(localX, localY)`, keeping `nLuma` low-frequency luma coefficients; chroma is
 * DC-only (a flat color per block). Pure — the caller samples the block. Memoized per nLuma.
 */
export function makeJpegBlockRgb(nLuma: number) {
    const hit = jpegBlockCache.get(nLuma)
    if (hit) return hit
    const built = buildJpegBlockRgb(nLuma)
    jpegBlockCache.set(nLuma, built)
    return built
}

export interface BlockQuantizeOptions {
    /** Low-frequency luma coefficients kept per channel. */
    nLuma: number
    /** Device pixels per cell (the 8×8 block spans 8·scaleN px, sampled at stride scaleN). */
    scaleN: number
    format: 'rgba16float' | 'rgba32float'
    namePrefix: string
}

/**
 * The block-quantize pass: one thread per scaleN×scaleN device-pixel cell samples its 8×8 block
 * from the input texture (point `textureLoad`, unpremultiplied), runs the DCT block color, and
 * writes it into the cell texture. Dispatch the active cell grid dynamically; the guarded pipeline
 * bounds-checks. Params ABI: inputWidth, inputHeight, quality.
 */
export function buildBlockQuantizeGraph(opts: BlockQuantizeOptions) {
    const {nLuma, scaleN, namePrefix, format} = opts
    const blockPx = 8 * scaleN
    const Params = d.struct({inputWidth: d.f32, inputHeight: d.f32, quality: d.f32}).$name('BlockQuantizeParams')
    const layout = tgpu.bindGroupLayout({
        input: {texture: d.texture2d(d.f32)},
        cellTex: {storageTexture: d.textureStorage2d(format, 'write-only')},
        params: {uniform: Params},
    })
    const blockRgb = makeJpegBlockRgb(nLuma)

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        const cellXf = d.f32(cx)
        const cellYf = d.f32(cy)
        // local = (cellX%8, cellY%8); blockOrigin = (floor(cellX/8)·blockPx, floor(cellY/8)·blockPx).
        const blockCol = std.floor(cellXf / 8.0)
        const blockRow = std.floor(cellYf / 8.0)
        const localX = cellXf - blockCol * 8.0
        const localY = cellYf - blockRow * 8.0
        const originX = blockCol * blockPx
        const originY = blockRow * blockPx

        // Sample the 8×8 block: textureLoad point-samples the nearest texel (compute has no
        // filtering sampler bound). Unpremultiply each texel (the input is a premultiplied RTT).
        const block = d.arrayOf(d.vec3f, 64)()
        const wm1 = d.i32(p.inputWidth - 1.0)
        const hm1 = d.i32(p.inputHeight - 1.0)
        for (let y = 0; y < 8; y++) {
            for (let x = 0; x < 8; x++) {
                const px = originX + d.f32(x) * scaleN + scaleN * 0.5
                const py = originY + d.f32(y) * scaleN + scaleN * 0.5
                const ix = std.clamp(d.i32(px), 0, wm1)
                const iy = std.clamp(d.i32(py), 0, hm1)
                const texel = std.textureLoad(layout.$.input, d.vec2u(d.u32(ix), d.u32(iy)), 0)
                block[y * 8 + x] = blend.unpremultiplyAlpha(texel).xyz
            }
        }

        const rgb = blockRgb(block, localX, localY, p.quality)
        std.textureStore(layout.$.cellTex, d.vec2u(cx, cy), d.vec4f(rgb, 1.0))
    }).$name(`${namePrefix}Cell`)

    return {layout, kernel, Params}
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// Flow-field display
// ════════════════════════════════════════════════════════════════════════════════════════════════

/**
 * color a packed flow field by direction: five smoothing-cross taps (r,g = flow vector, b =
 * density) → a blend of the four per-direction colors weighted by the smoothed flow direction,
 * faded to `base` where flow is weak and to transparent where density is low.
 */
export const flowDirectionColor = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.vec4f], d.vec4f)(
    (s0, s1, s2, s3, s4, base, up, down, left, right) => {
        'use gpu'
        const smoothedLiquid = (s0.z + s1.z + s2.z + s3.z + s4.z) * 0.2
        const liquidIntensity = std.smoothstep(0.0, 0.1, smoothedLiquid)
        const flow = s0.xy.add(s1.xy).add(s2.xy).add(s3.xy).add(s4.xy).mul(0.2)
        const flowMagnitude = std.length(flow)
        const hasFlow = std.smoothstep(0.01, 0.1, flowMagnitude)
        const normalizedX = flow.x / (flowMagnitude + 0.001)
        const normalizedY = flow.y / (flowMagnitude + 0.001)
        const rightAmount = std.smoothstep(0.0, 0.7, std.max(normalizedX, 0.0))
        const leftAmount = std.smoothstep(0.0, 0.7, std.max(normalizedX * -1.0, 0.0))
        const upAmount = std.smoothstep(0.0, 0.7, std.max(normalizedY, 0.0))
        const downAmount = std.smoothstep(0.0, 0.7, std.max(normalizedY * -1.0, 0.0))
        const horizontalColor = left.mul(leftAmount).add(right.mul(rightAmount))
        const verticalColor = down.mul(downAmount).add(up.mul(upAmount))
        const horizontalWeight = leftAmount + rightAmount
        const verticalWeight = upAmount + downAmount
        const totalWeight = horizontalWeight + verticalWeight + 0.001
        const directionalColor = horizontalColor.mul(horizontalWeight / totalWeight).add(verticalColor.mul(verticalWeight / totalWeight))
        const finalColor = std.mix(base, directionalColor, d.vec4f(hasFlow))
        return finalColor.mul(liquidIntensity)
    }).$name('flowDirectionColor')
