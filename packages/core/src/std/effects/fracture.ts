/**
 * Broken glass: the canvas splits into shards around random sites, cracks open where shards
 * meet, and the layer inside bends across them. `voronoiRegionField` is the compute hook that
 * finds, for every pixel, its shard and the next-nearest one. The crack words run in the
 * fragment: `crackGeom` measures the crack at a pixel, `crackRefractUV` bends one color
 * channel across it, and `crackShardCompose` joins the three channels with a little light on
 * the tilted shards.
 *
 * The crack words are GPU functions. Invoke one with `call(fn, 'name', [args])`. Each shard's
 * position and displacement reach the fragment through a one-row data texture you fill;
 * `cellLaneUVFor` addresses it per shard.
 */
// Maintainer: Voronoi-fracture vocabulary — a nearest/second-nearest region field precomputed on
// the GPU (re-dispatched only on seed change), per-cell data-lane addressing, and the crack
// render parts (crack geometry from the two nearest sites, per-channel refracted taps,
// tilted-shard composite). The CPU shard physics stays with the consumer (Shatter).
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import {createGuardedCompute, createStateBuffer} from '../../gpu/porters'
import {tgpu, d, std} from '../../gpu/kit/index'

const VORONOI_FORMAT = 'rgba16float' as const

interface VoronoiGraph {
    layout: ReturnType<typeof makeVoronoiGraph>['layout']
    kernel: ReturnType<typeof makeVoronoiGraph>['kernel']
}

function makeVoronoiGraph(count: number, size: number) {
    const layout = tgpu.bindGroupLayout({
        cellPos: {storage: d.arrayOf(d.f32, count * 2), access: 'readonly'},
        voronoiTex: {storageTexture: d.textureStorage2d(VORONOI_FORMAT, 'write-only')},
    })
    /** Nearest + second-nearest cell per pixel. The running (d1, d2, idx1, idx2) selection is one
     *  serial scan whose intermediates every branch shares — an atomic core. */
    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const u = (d.f32(cx) + 0.5) / size
        const v = (d.f32(cy) + 0.5) / size
        let d1 = d.f32(99999.0)
        let d2 = d.f32(99999.0)
        let idx1 = d.f32(0.0)
        let idx2 = d.f32(0.0)
        for (let i = 0; i < count; i++) {
            const cellX = layout.$.cellPos[i * 2]
            const cellY = layout.$.cellPos[i * 2 + 1]
            const ddx = u - cellX
            const ddy = v - cellY
            const distSq = ddx * ddx + ddy * ddy
            if (distSq < d1) {
                d2 = d1
                idx2 = idx1
                d1 = distSq
                idx1 = d.f32(i)
            } else {
                if (distSq < d2) {
                    d2 = distSq
                    idx2 = d.f32(i)
                }
            }
        }
        std.textureStore(layout.$.voronoiTex, d.vec2u(cx, cy), d.vec4f(idx1, idx2, 0.0, 0.0))
    }).$name('voronoiNearest2')
    return {layout, kernel}
}

const voronoiGraphCache = new Map<string, VoronoiGraph>()

/** @internal The cached per-(count, size) nearest-two compute graph behind `voronoiRegionField`. */
// Both `count` and `size` are baked into the kernel as compile-time constants, so the graph is
// memoised per pair.
export function voronoiNearest2Graph(count: number, size: number): VoronoiGraph {
    const key = `${count}|${size}`
    let g = voronoiGraphCache.get(key)
    if (!g) {
        g = makeVoronoiGraph(count, size)
        voronoiGraphCache.set(key, g)
    }
    return g
}

// UV of cell `idx`'s `field` slot in a 1-tall data texture of `laneWidth` texels (4 per cell).
const cellLaneUVCache = new Map<number, ReturnType<typeof makeCellLaneUV>>()
function makeCellLaneUV(laneWidth: number) {
    return tgpu.fn([d.f32, d.f32], d.vec2f)((idx, field) => {
        'use gpu'
        return d.vec2f((idx * 4.0 + field + 0.5) / laneWidth, 0.5)
    })
}
/**
 * Where one shard's value sits in a one-row data texture.
 *
 * Lay the texture out as four values per shard, so `laneWidth` is the shard count times four.
 * The returned GPU function takes a shard index and a value index (0 to 3) and gives the uv to
 * sample with `'nearestClamp'`. Built once per width.
 *
 * @example
 * ```ts
 * const cellLaneUV = cellLaneUVFor(16 * 4)
 * const dataAt = (idx: Expr, field: number) => dataKit.sample(call(cellLaneUV, 'cellLaneUV', [idx, floatE(field)]), 'nearestClamp').member('r')
 * ```
 * @see voronoiRegionField, crackGeom
 */
export function cellLaneUVFor(laneWidth: number) {
    let fn = cellLaneUVCache.get(laneWidth)
    if (!fn) {
        fn = makeCellLaneUV(laneWidth)
        cellLaneUVCache.set(laneWidth, fn)
    }
    return fn
}

/** @internal The record `crackGeom` returns. */
// Crack geometry for one pixel (nearest/second cell positions + the nearest cell's displacement).
export const CrackGeom = d.struct({
    crackIntensity: d.f32,
    edgeNormal: d.vec2f,
    displacedUV: d.vec2f,
    disp: d.vec2f,
})

/**
 * How much crack there is at a pixel, and which way the shard under it has moved.
 *
 * Takes the pixel's uv, the nearest and second-nearest shard positions (x and y, in uv), the
 * nearest shard's displacement (x and y, in uv) and `crackWidth` (about 0.5 to 5). Returns a
 * record with `crackIntensity` (0 to 1, and 0 until the shard has actually moved),
 * `edgeNormal` (the direction across the crack), `displacedUV` (where to sample the layer
 * inside) and `disp` (the displacement again, for `crackShardCompose`).
 *
 * @example
 * ```ts
 * const geom = call(crackGeom, 'crackGeom', [ctx.uv, dataAt(nearest, 0), dataAt(nearest, 1), dataAt(second, 0), dataAt(second, 1), dataAt(nearest, 2), dataAt(nearest, 3), uniforms.crackWidth])
 * ```
 * @see crackRefractUV, crackShardCompose, cellLaneUVFor
 */
// Maintainer: crack = 1 − smoothstep(0, crackWidth·0.005, d2 − d1), gated by
// smoothstep(0, 0.01, |disp|) so undisplaced shards show no seam; edgeNormal is the nearest-site
// direction rotated 90°.
export const crackGeom = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    CrackGeom,
)((uv, nPosX, nPosY, sPosX, sPosY, dispX, dispY, crackWidth) => {
    'use gpu'
    const nearestPos = d.vec2f(nPosX, nPosY)
    const secondPos = d.vec2f(sPosX, sPosY)
    const disp = d.vec2f(dispX, dispY)
    const fromNearest = uv.sub(nearestPos)
    const minDist1 = std.length(fromNearest)
    const minDist2 = std.length(uv.sub(secondPos))
    const displacementMag = std.length(disp)
    const edgeDiff = minDist2 - minDist1
    const crackThreshold = crackWidth * 0.005
    const baseCrackIntensity = 1.0 - std.smoothstep(0.0, crackThreshold, edgeDiff)
    const displacementFactor = std.smoothstep(0.0, 0.01, displacementMag)
    const crackIntensity = baseCrackIntensity * displacementFactor
    const toNearest = fromNearest.div(std.max(minDist1, 1e-5))
    const edgeNormal = d.vec2f(toNearest.y * -1.0, toNearest.x)
    const displacedUV = uv.sub(disp)
    return CrackGeom({crackIntensity, edgeNormal, displacedUV, disp})
})

/**
 * Where to sample one color channel so the crack bends the light and splits it like a prism.
 *
 * `channel` is 1 for red, 0 for green and -1 for blue; the three samples land slightly apart.
 * `refractionStrength` (0 to 10) bends the sample across the crack and `chromaticSplit` (0 to 5)
 * spreads the channels. Both fade with `crackIntensity`, so unbroken glass is untouched.
 *
 * @example
 * ```ts
 * const red = child.sample(call(crackRefractUV, 'crackRefractUV', [geom.member('displacedUV'), geom.member('edgeNormal'), geom.member('crackIntensity'), uniforms.refractionStrength, uniforms.chromaticSplit, floatE(1)]))
 * ```
 * @see crackGeom, crackShardCompose
 */
// Maintainer: offset = edgeNormal · (refractionStrength · 0.01 + channel · chromaticSplit · 0.005),
// scaled by crackIntensity.
export const crackRefractUV = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (displacedUV, edgeNormal, crackIntensity, refractionStrength, chromaticSplit, channel) => {
        'use gpu'
        const refractionOffset = edgeNormal.mul(refractionStrength * 0.01)
        const chromaticOffset = chromaticSplit * 0.005
        const chromaShift = edgeNormal.mul(channel * chromaticOffset)
        const offset = refractionOffset.add(chromaShift)
        return displacedUV.add(offset.mul(crackIntensity))
    },
)

// normalize(vec2(0.3, 0.6)), pre-folded so the shard light direction is a pair of literals.
const SHARD_LIGHT_DIR_X = 0.3 / Math.hypot(0.3, 0.6)
const SHARD_LIGHT_DIR_Y = 0.6 / Math.hypot(0.3, 0.6)

/**
 * The finished glass: the three refracted channels blended into the crack, with a hint of light on tilted shards.
 *
 * Pass the plain sample at `displacedUV`, the red, green and blue samples from
 * `crackRefractUV`, then `crackIntensity` and `disp` from `crackGeom`, and `shardLighting`
 * (0 to 0.5). The samples come from the layer's texture, so unpremultiply the result before
 * returning it.
 *
 * @example
 * ```ts
 * const glass = call(crackShardCompose, 'crackShardCompose', [plain, red, green, blue, geom.member('crackIntensity'), geom.member('disp'), uniforms.shardLighting])
 * ```
 * @see crackRefractUV, crackGeom
 */
// Maintainer: refracted rgb = (r.x, g.y, b.z) mixed over the plain color by crackIntensity, then a
// tilt-lighting factor from disp against a fixed light direction, fading in over |disp| 0..0.02.
// Alpha is the plain sample's. Output is still premultiplied.
export const crackShardCompose = tgpu.fn([d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.f32, d.vec2f, d.f32], d.vec4f)(
    (normalColor, rFinal, gFinal, bFinal, crackIntensity, disp, shardLighting) => {
        'use gpu'
        const refractedRGB = d.vec3f(rFinal.x, gFinal.y, bFinal.z)
        const shadedRGB = normalColor.xyz.mul(1.0 - crackIntensity).add(refractedRGB.mul(crackIntensity))
        const displacementLength = std.length(disp)
        const tiltX = disp.x / (displacementLength + 0.001)
        const tiltY = disp.y / (displacementLength + 0.001)
        const normalDot = tiltX * SHARD_LIGHT_DIR_X + tiltY * SHARD_LIGHT_DIR_Y
        const lightingFactor = 1.0 + normalDot * shardLighting
        const lightingIntensity = std.smoothstep(0.0, 0.02, displacementLength)
        const finalLighting = 1.0 + (lightingFactor - 1.0) * lightingIntensity
        const finalRGB = shadedRGB.mul(finalLighting)
        return d.vec4f(finalRGB, normalColor.w)
    },
)

/**
 * Splits the canvas into shards around random sites and records, per pixel, its shard and the next-nearest one.
 *
 * A compute hook for the `compute:` field. `count` is the number of shards, `size` the field's
 * resolution in pixels, `sites(seed)` your function returning the shard positions as
 * `[x, y, x, y, …]` in uv, and `seedProp` the prop that re-rolls them. The field only
 * recomputes when the seed changes. In the fragment, sample `computeOutputs[output]` with
 * `'nearestClamp'`: `.r` is the nearest shard's index, `.g` the second-nearest.
 *
 * @example
 * ```ts
 * compute: voronoiRegionField({count: 16, size: 1024, sites: shardSites, seedProp: 'seed', seedDefault: 2, output: 'voronoiField'})
 * ```
 * @tip Fill your shard data texture from the same `sites` function, or the cracks drift off their shards.
 * @see cellLaneUVFor, crackGeom
 */
// Maintainer: nearest + second-nearest cell ID per pixel, precomputed on the GPU and re-dispatched
// only when the seed prop changes (avoiding a many-megaop CPU stall). Cell positions live in a
// small storage buffer regenerated CPU-side by `sites(seed)` on seed change; between changes the
// field is static and the frame program dispatches nothing. Returns null without a device (the
// fragment falls back to passthrough).
export function voronoiRegionField(opts: {
    count: number
    size: number
    sites: (seed: number) => Float32Array
    seedProp: string
    seedDefault: number
    output: string
}): GpuComputeNode {
    return (params: GpuFragmentParams) => {
        const {gpu, registerComputeTexture, getCpuValue, onCleanup} = params
        const root = gpu?.root
        if (!root) return null // GPU-free (no device): fragment falls back to passthrough.

        const {layout, kernel} = voronoiNearest2Graph(opts.count, opts.size)
        const cellPos = createStateBuffer(root, d.f32, opts.count * 2)
        const voronoiTex = root.createTexture({size: [opts.size, opts.size], format: VORONOI_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => voronoiTex.destroy())
        const field = registerComputeTexture(voronoiTex)

        const bindGroup = root.createBindGroup(layout, {cellPos, voronoiTex})
        const pipeline = createGuardedCompute(root, (cx: number, cy: number) => {
            'use gpu'
            kernel(cx, cy)
        }, {size: [opts.size, opts.size], bindGroup})

        const generateCells = (seed: number): void => {
            cellPos.write(opts.sites(seed) as never)
        }

        let currentSeed = (getCpuValue(opts.seedProp) as number) ?? opts.seedDefault
        generateCells(currentSeed)
        let needsDispatch = true

        return {
            outputs: {[opts.output]: field},
            getComputeNodes: () => {
                const newSeed = (getCpuValue(opts.seedProp) as number) ?? opts.seedDefault
                if (newSeed !== currentSeed) {
                    currentSeed = newSeed
                    generateCells(currentSeed)
                    needsDispatch = true
                }
                if (needsDispatch) {
                    needsDispatch = false
                    return [pipeline]
                }
                return null // the field is static between seed changes.
            },
        }
    }
}
