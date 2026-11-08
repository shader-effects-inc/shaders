import type {GpuShaderDefinition, GpuFragmentParams, GpuComputeNode, KitTexture, Expr} from "@coreroot/gpu/porters"
import {call, vec4, ZERO, buildBlockQuantizeGraph, makeJpegBlockRgb, createGridKernelPass} from "@coreroot/gpu/porters"
import {defineShader, p as prop, type PropRef} from "@coreroot/std"
import {add, clamp, div, floor, local, mul, vec2} from "@coreroot/std/math"
import {blend} from "@coreroot/gpu/kit"

// How many low-frequency coefficients to keep per luma channel (high-quality-loss JPEG zeroes the
// high frequencies anyway). Fixed block geometry: 32px blocks sampled on an 8×8 grid (stride 4 px).
const N_LUMA = 6
const SCALE_N = 4
const STRIDE = SCALE_N
// Cell-grid ceiling: 1024×1024 cells covers canvases up to 4096×4096 device px.
const MAX_CELLS = 1024
const CELL_FORMAT = 'rgba16float' as const

export interface ComponentProps {
    quality: number
}

/**
 * The per-cell compute graph: the kit's DCT block-quantize pass (JPEG Annex K tables, IJG quality
 * curve, `N_LUMA` low-frequency luma coefficients, DC-only chroma) at this effect's block
 * geometry. One thread per STRIDE×STRIDE device-pixel cell samples the child RTT's 8×8 block and
 * writes the block color into the cell texture.
 */
export const buildCompressionGraph = () => buildBlockQuantizeGraph({
    nLuma: N_LUMA, scaleN: SCALE_N, format: CELL_FORMAT, namePrefix: 'compression',
})

/** The DCT block-color core (the shape the port gate resolves + CPU-goldens). */
export const jpegBlockRgb = makeJpegBlockRgb(N_LUMA)

// ═══════════════════════════════════════════════════════════════════════════════════════
// The jpegCompress noun — its contract:
//   - quality mapping: the IJG curve inside the kit's DCT core, driven by the live `quality`
//     scalar (map-driven per-pixel quality is not supported — the block color uses the uniform);
//   - cell grid: one compute thread per STRIDE×STRIDE (4×4) device-pixel cell, written into a
//     fixed MAX_CELLS² rgba16float cell texture (covers canvases up to 4096² device px);
//   - the 1-frame-lag contract: compute runs before the RTT passes, so the cell RGB reads the
//     PREVIOUS frame's child RTT;
//   - the post stage below (crisp cell fetch + full-res alpha).
// ═══════════════════════════════════════════════════════════════════════════════════════

/** Post stage — cellFetch composite: the precomputed block color (one nearest cell fetch) with
 *  the full-res sharp alpha (crisp edges). Cell fetch UV: floor(pPix/STRIDE) clamped to the grid,
 *  → the cell texel centre. */
function cellFetchComposite(cellTex: KitTexture, sharp: KitTexture, params: GpuFragmentParams): Expr {
    const {ctx} = params
    const center = call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sharp.sample(ctx.uv)])
    const pPix = local(floor(mul(ctx.uv, ctx.viewportSize)), 'pPix')
    const cellX = clamp(floor(div(pPix.member('x'), STRIDE)), 0, MAX_CELLS - 1)
    const cellY = clamp(floor(div(pPix.member('y'), STRIDE)), 0, MAX_CELLS - 1)
    const cellUV = vec2(div(add(cellX, 0.5), MAX_CELLS), div(add(cellY, 0.5), MAX_CELLS))
    const cellSample = cellTex.sample(cellUV, 'nearestClamp')
    return vec4(cellSample.member('rgb'), center.member('a'))
}

/** The two halves the noun contributes — spread it into the definition. */
function jpegCompress(opts: {quality: PropRef}): {compute: GpuComputeNode; gpu: {fragment: (params: GpuFragmentParams) => Expr}} {
    return {
        // One thread per cell: sample the child RTT's 8×8 block, run the DCT core, write the block
        // color into the cell texture.
        compute: (params: GpuFragmentParams) => {
            const {childNode, gpu, convertToTexture, registerComputeTexture, getCpuValue, onCleanup, onResize, dimensions} = params
            if (!childNode) return null
            const root = gpu?.root
            if (!root) return null // GPU-free (no device): fragment falls back to sharp passthrough.

            const childTexture = convertToTexture(childNode)
            const cellTex = root.createTexture({size: [MAX_CELLS, MAX_CELLS], format: CELL_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => cellTex.destroy())
            const jpegCellTexture = registerComputeTexture(cellTex)

            const graph = buildCompressionGraph()
            const compParams = root.createUniform(graph.Params)

            // `dimensions` is already the device-pixel backing size (the renderer owns DPR).
            let curW = Math.max(1, Math.round(dimensions.width))
            let curH = Math.max(1, Math.round(dimensions.height))
            const cellsFor = (w: number, h: number): [number, number] => [
                Math.min(Math.ceil(w / STRIDE), MAX_CELLS),
                Math.min(Math.ceil(h / STRIDE), MAX_CELLS),
            ]
            let [cellsX, cellsY] = cellsFor(curW, curH)

            // No pre-bound size (the active cell grid varies per frame → dispatched via a thunk); the
            // bind group (input RTT + cell texture + params) is bound LATE once the child RTT exists.
            const pipeline = createGridKernelPass(root, graph.kernel)
            let bound = pipeline
            let inputBound = false

            onResize(({width, height}) => {
                curW = Math.max(1, Math.round(width))
                curH = Math.max(1, Math.round(height))
                ;[cellsX, cellsY] = cellsFor(curW, curH)
            })

            return {
                outputs: {childTexture, jpegCellTexture},
                bindInputs: (resolve) => {
                    const src = resolve(childTexture.key)
                    if (src) {
                        const bg = root.createBindGroup(graph.layout, {
                            input: src.texture as never,
                            cellTex: cellTex as never,
                            params: compParams.buffer,
                        })
                        bound = pipeline.with(bg)
                        inputBound = true
                    }
                },
                getComputeNodes: () => {
                    const q = getCpuValue(opts.quality.name)
                    compParams.write({inputWidth: curW, inputHeight: curH, quality: typeof q === 'number' ? q : 12} as never)
                    // Dispatch exactly the active cell grid (2D). The guarded pipeline bounds-checks.
                    return [() => { if (inputBound) bound.dispatchThreads(cellsX, cellsY) }]
                },
            }
        },

        gpu: {
            fragment: (params: GpuFragmentParams): Expr => {
                const {childNode, ctx, computeOutputs, convertToTexture} = params
                if (!childNode) return ZERO

                const cellTex = computeOutputs?.jpegCellTexture as KitTexture | undefined
                const sharp = computeOutputs?.childTexture as KitTexture | undefined
                if (!cellTex || !sharp) {
                    // Compute unavailable (no GPU device): sharp passthrough, unpremultiplied.
                    const tex = convertToTexture(childNode)
                    return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [tex.sample(ctx.uv)])
                }
                return cellFetchComposite(cellTex, sharp, params)
            },
        },
    }
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "CompressionArtifacts",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Simulates lossy JPEG compression — 8×8 DCT block quantization, blockiness, ringing and color bleed",
    requiresRTT: true,
    requiresChild: true,
    props: {
        quality: {
            default: 12,
            description: "JPEG quality (1 = heavily destroyed, 100 = near lossless)",
            ui: { type: ['range', 'map'], min: 1, max: 100, step: 1, label: 'Quality', group: 'Effect' }
        }
    },

    // WebGPU compute: evaluate the ~3500-op DCT kernel ONCE per 4×4 device-pixel cell (the block
    // color is constant across a cell — `local` only changes per cell), then the fragment is one
    // cell fetch + the full-res alpha sample. See the noun's contract above.
    ...jpegCompress({quality: prop('quality')}),
})

export default componentDefinition
