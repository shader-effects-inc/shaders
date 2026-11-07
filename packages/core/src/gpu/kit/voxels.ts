/**
 * kit/voxels — the VOXELIZED shape-field pre-march.
 *
 * The volumetric spine (kit/sdf3d) sphere-traces a smooth SDF into a 2D field texture. This module
 * is its sibling for a QUANTISED solid: the same compute pre-march (same setups, same
 * `MarchParams`, same field texture, same dirty-key/active-res machinery) but the march is a
 * grid DDA (Amanatides–Woo) over voxel cells whose occupancy is the SDF sampled at the cell centre.
 * Any shape — flat analytic, flat SVG, SVG extrude, analytic 3D — goes through it: flat shapes are
 * lifted into a slab by `createAnalytic2dExtrudeSetup` / the SVG extrusion, so a flat circle is a
 * voxel disc and a 3D sphere is a voxel ball.
 *
 * Three dispatches, each with its own dirty key:
 *   1. GRID BAKE — one thread per four cells; the only place the shape SDF runs. Each cell stores
 *      0 (solid) or 1 + its clearance in cells (the empty-space leap the march may take). Keyed on
 *      the shape's geometry (+ rotation/camera only for a screen-aligned grid), so rotating a model
 *      or moving the light never re-bakes.
 *   2. SHADOW MAP — an orthographic depth render of the grid from the light's direction (one DDA
 *      per map texel, independent of screen size). Keyed on the grid + the light direction in grid
 *      space, so a light orbit re-renders only this.
 *   3. MARCH — per field texel: DDA to the first solid cell, sub-shape hit (cube / sphere / rounded),
 *      Minecraft-style smooth vertex AO from the 8 face neighbours (byte reads). No light dependence.
 *
 * The field texel is a G-BUFFER, not a distance field:
 *   .r = signed coverage (−texel on a hit texel, +texel + slack on a miss — the bilinear/bicubic
 *        samplers reconstruct a one-texel anti-aliased silhouette; far misses grow so the
 *        `insideShape` early-exit still skips them)
 *   .g = faceId·65536 + ao·256  (faceId 0..5 = axis·2 + positive, ao quantised 0..255)
 *   .b = cell index ix + iy·256 + iz·65536 (grid capped below 256 cells per axis → exact f32)
 *   .a = view-space hit depth (tEnter)
 * `.g/.b/.a` are DISCRETE — read them with the unfiltered `texelSampler`, never bilinear. The
 * fragment (std/paint/voxels) reconstructs the hit point from `.a` + the published view→grid
 * matrix, derives the exact normal / face coords / per-voxel identity from the cell, and takes the
 * shadow from the map with {@link voxelShadowLookup} (contact-hardening PCSS) — live, per pixel.
 */
import {tgpu, d, std} from './index'
import type {TgpuTexture} from 'typegpu'
import {createGuardedCompute, createStateBuffer} from '../compute'
import {isMobileGpuViewport} from '../../utilities/device'
import {resolveShapeType} from '../../utilities/shapeEffectBounds'
import type {GpuFragmentParams, KitTexture, GpuComputeStep} from '../contract'
import {parseShapeConfigValue} from './sdf'
import {
    RotStruct, rotateVec3, rotateVecCpu, sdRoundBox,
    createAnalytic3dSdfSetup, createSvg3dSdfSetup, createAnalytic2dExtrudeSetup,
    buildVolumetricFieldKernel, createVolumetricFieldCompute, resolveActiveFieldRes,
    is3dShapeType, isSvg3dShapeType,
    type BakedSdf, type VolumetricFieldLayout, type VolumetricFieldSetup, type ShapeFrameParams,
} from './sdf3d'
import {isSvgShape} from '../../utilities/svgShape'

export type VoxelStyle = 'cube' | 'sphere' | 'rounded'
export type VoxelGridSpace = 'shape' | 'view'

/** Cells per axis cap (device-tiered: the occupancy buffer is W·N² u32 words, 7 MB / 2 MB at the
 *  caps). Indices pack into one f32 as ix + iy·256 + iz·65536, so the cap must stay below 256. */
export const VOXEL_GRID_MAX = isMobileGpuViewport() ? 128 : 192
/** u32 words per grid row: four u8 cell values per word. */
const OCC_WORDS_PER_ROW = Math.ceil(VOXEL_GRID_MAX / 4)
/** Fixed occupancy-buffer allocation (words). */
export const OCC_BUFFER_WORDS = OCC_WORDS_PER_ROW * VOXEL_GRID_MAX * VOXEL_GRID_MAX
/** Shadow-map resolution (texels per side over the bounding diameter). ~30 texels per default
 *  voxel edge on desktop — a hard shadow's edge lands well inside one voxel face. */
export const VOXEL_SHADOW_RES = isMobileGpuViewport() ? 512 : 1024
export const VOXEL_SHADOW_FORMAT = 'r32float' as const

const DEG = Math.PI / 180
/** Half the diagonal of a unit cube — how far a voxel can puff past the smooth surface. */
const HALF_DIAGONAL = 0.8660254
/** Clearance margin (cells) baked into each empty cell's value: a jump of `sdf/v − fill − MARGIN`
 *  cells from anywhere inside the cell cannot reach an occupied cell (two half-diagonals: the ray
 *  point → this centre, and the farthest centre a jump can touch). */
const CLEARANCE_MARGIN = 1.74
/** Golden angle — the Vogel disc's tap spiral. */
const GOLDEN_ANGLE = 2.39996323
/** Directional-light softness → tan(half-angle) at full softness (~15°). */
const SOFT_TAN = 0.27
/** Depth stored for a shadow-map texel whose ray hits nothing. */
const SHADOW_FAR = 1e9

/**
 * The voxel pre-march uniform (bound with the occupancy grid + shadow map). FULL-WRITE ONLY (the
 * `cam` member is Decorated — see MarchParams).
 *   cam        — camera pitch/yaw as a ZXY RotStruct, applied to the view ray BEFORE the shape rot
 *   lightDir   — unit light direction in GRID space; lightT1/T2 its tangent basis (the shadow-map frame)
 *   voxelSize  — cell edge (field UV) · fill — occupancy threshold in cells (sdf(centre) < fill·size)
 *   voxelScale — sub-shape size within its cell (1 = touching) · bevel — rounded style radius fraction
 *   gridOrigin — grid min corner (all axes; = −rBound) · gridN / gridW — cells per axis, words per row
 */
export const VoxelMarchParams = d.struct({
    cam: d.align(16, d.size(32, RotStruct)),
    lightDir: d.vec3f, voxelSize: d.f32,
    lightT1: d.vec3f, fill: d.f32,
    lightT2: d.vec3f, voxelScale: d.f32,
    bevel: d.f32, gridOrigin: d.f32, gridN: d.f32, gridW: d.f32,
})

/**
 * The march + shadow-map bind group: the voxel params, the baked OCCUPANCY GRID (read-only) and
 * the SHADOW MAP (the shadow pass writes it; the march pipeline simply never touches it). Each grid
 * cell is a u8 packed four-per-u32 along x: 0 = solid · k ≥ 1 = empty with (k − 1) cells of
 * guaranteed clearance (the empty-space skip distance). Baked once per geometry change by
 * `buildVoxelBakeFn`, so the march, the AO taps and the shadow rays never evaluate the shape SDF.
 */
export function makeVoxelLayout() {
    return tgpu.bindGroupLayout({
        vox: {uniform: VoxelMarchParams},
        occ: {storage: d.arrayOf(d.u32, OCC_BUFFER_WORDS), access: 'readonly'},
        shadowMap: {storageTexture: d.textureStorage2d(VOXEL_SHADOW_FORMAT, 'write-only')},
    })
}
export type VoxelLayout = ReturnType<typeof makeVoxelLayout>

/** The bake's bind group over the SAME uniform + buffer, with the grid writable. */
export function makeVoxelBakeLayout() {
    return tgpu.bindGroupLayout({
        vox: {uniform: VoxelMarchParams},
        occ: {storage: d.arrayOf(d.u32, OCC_BUFFER_WORDS), access: 'mutable'},
    })
}
export type VoxelBakeLayout = ReturnType<typeof makeVoxelBakeLayout>

export interface VoxelFieldOptions {
    style: VoxelStyle
    /** `'shape'`: the grid rotates with the solid (a voxel MODEL). `'view'`: the grid is fixed to
     *  the screen and the solid moves through it (a 3D pixelation). */
    gridSpace: VoxelGridSpace
}

/**
 * Build the occupancy BAKE kernel: one thread per u32 word (four x-adjacent cells), dispatched
 * `(gridW, gridN, gridN)`. Each cell's value is 0 when the shape SDF at its centre is below the fill
 * threshold, else 1 + the cells of clearance the SDF guarantees around it (see CLEARANCE_MARGIN).
 * `sdfFn` is the setup's baked shape SDF; `marchLayout` is the setup's layout (the SDF reads its
 * params off it, and `'view'` grids read `rot`). The only place the shape SDF runs.
 */
export function buildVoxelBakeFn(
    sdfFn: BakedSdf,
    marchLayout: VolumetricFieldLayout,
    bakeLayout: VoxelBakeLayout,
    gridSpace: VoxelGridSpace,
) {
    const gridToShape = tgpu.fn([d.vec3f], d.vec3f)((pG) => {
        'use gpu'
        if (gridSpace === 'view') return rotateVec3(rotateVec3(pG, bakeLayout.$.vox.cam), marchLayout.$.params.rot)
        else return d.vec3f(pG.x, pG.y, pG.z)
    })

    return tgpu.fn([d.u32, d.u32, d.u32])((wx, iy, iz) => {
        'use gpu'
        const vx = bakeLayout.$.vox
        const n = d.u32(vx.gridN)
        const w = d.u32(vx.gridW)
        if (wx < w && iy < n && iz < n) {
            const v = vx.voxelSize
            const o = vx.gridOrigin
            let word = d.u32(0)
            let scale = d.u32(1)
            for (let k = 0; k < 4; k++) {
                const ix = wx * d.u32(4) + d.u32(k)
                let val = d.u32(1)
                if (ix < n) {
                    const c = d.vec3f((d.f32(ix) + 0.5) * v + o, (d.f32(iy) + 0.5) * v + o, (d.f32(iz) + 0.5) * v + o)
                    const dd = sdfFn(gridToShape(c))
                    const clear = std.clamp(dd / v - vx.fill - CLEARANCE_MARGIN, 0.0, 254.0)
                    val = std.select(d.u32(1) + d.u32(std.floor(clear)), d.u32(0), dd < vx.fill * v)
                }
                word = word + val * scale
                scale = scale * d.u32(256)
            }
            bakeLayout.$.occ[wx + (iy + iz * n) * w] = word
        }
    }).$name('voxelGridBake')
}

/**
 * The grid TRAVERSAL vocabulary over a `VoxelLayout` — shared by the march and the shadow-map
 * pass: baked cell reads, the sub-shape ray test and the DDA with empty-space leaps. `style` is a
 * build-time string; only its branch is emitted.
 */
export function buildVoxelTraversal(voxLayout: VoxelLayout, style: VoxelStyle) {
    /** The baked cell value (see `makeVoxelLayout`): 0 solid · k ≥ 1 empty with k − 1 clearance.
     *  Cells outside the grid read as empty with no clearance. */
    const cellValue = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((cx, cy, cz) => {
        'use gpu'
        const vx = voxLayout.$.vox
        const n = vx.gridN
        let out = d.f32(1)
        if (cx >= 0.0 && cx < n && cy >= 0.0 && cy < n && cz >= 0.0 && cz < n) {
            const wx = std.floor(cx * 0.25)
            const sub = cx - wx * 4.0
            const idx = d.u32(wx) + (d.u32(cy) + d.u32(cz) * d.u32(n)) * d.u32(vx.gridW)
            const word = voxLayout.$.occ[idx]
            out = d.f32(std.extractBits(word, d.u32(sub) * d.u32(8), d.u32(8)))
        }
        return out
    })

    const cellCentre = tgpu.fn([d.f32, d.f32, d.f32], d.vec3f)((cx, cy, cz) => {
        'use gpu'
        const vx = voxLayout.$.vox
        return d.vec3f(
            (cx + 0.5) * vx.voxelSize + vx.gridOrigin,
            (cy + 0.5) * vx.voxelSize + vx.gridOrigin,
            (cz + 0.5) * vx.voxelSize + vx.gridOrigin,
        )
    })

    /** 1 when the cell is solid (baked value 0). */
    const occupied = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((cx, cy, cz) => {
        'use gpu'
        return std.select(d.f32(0), d.f32(1), cellValue(cx, cy, cz) < 0.5)
    })

    /** Ray vs the sub-shape inside an occupied cell (a box shrunk by voxelScale, a sphere, or a
     *  rounded box sphere-traced within its slab bracket). `tNow` = the ray's entry into the cell. */
    const SubHit = d.struct({hit: d.f32, t: d.f32})
    const subShapeHit = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.f32], SubHit)((ro, rd, c, tNow) => {
        'use gpu'
        const vx = voxLayout.$.vox
        const h = vx.voxelSize * 0.5 * vx.voxelScale
        let hit = d.f32(0)
        let t = tNow
        if (style === 'sphere') {
            const oc = ro.sub(c)
            const b = std.dot(oc, rd)
            const cc = std.dot(oc, oc) - h * h
            const disc = b * b - cc
            if (disc >= 0.0) {
                const ts = b * -1.0 - std.sqrt(disc)
                if (ts >= tNow - 0.0001) {
                    hit = d.f32(1)
                    t = ts
                }
            }
        } else {
            // Slab test against the shrunk box. Signed inverse directions are magnitude-clamped so
            // an axis-parallel ray yields ±huge brackets that resolve correctly instead of NaN.
            const sx = std.select(d.f32(-1), d.f32(1), rd.x >= 0.0)
            const sy = std.select(d.f32(-1), d.f32(1), rd.y >= 0.0)
            const sz = std.select(d.f32(-1), d.f32(1), rd.z >= 0.0)
            const ix = sx / std.max(std.abs(rd.x), 0.000001)
            const iy = sy / std.max(std.abs(rd.y), 0.000001)
            const iz = sz / std.max(std.abs(rd.z), 0.000001)
            const t1x = (c.x - h - ro.x) * ix
            const t2x = (c.x + h - ro.x) * ix
            const t1y = (c.y - h - ro.y) * iy
            const t2y = (c.y + h - ro.y) * iy
            const t1z = (c.z - h - ro.z) * iz
            const t2z = (c.z + h - ro.z) * iz
            const tIn = std.max(std.min(t1x, t2x), std.max(std.min(t1y, t2y), std.min(t1z, t2z)))
            const tOut = std.min(std.max(t1x, t2x), std.min(std.max(t1y, t2y), std.max(t1z, t2z)))
            if (tIn <= tOut) {
                if (tOut >= tNow - 0.0001) {
                    if (style === 'rounded') {
                        const r = std.clamp(vx.bevel, d.f32(0), d.f32(1)) * h
                        let s = std.max(tIn, tNow - 0.0001)
                        for (let i = 0; i < 12; i++) {
                            const p = ro.add(rd.mul(s)).sub(c)
                            const dd = sdRoundBox(p, h, h, h, r)
                            if (dd < h * 0.003) {
                                hit = d.f32(1)
                                break
                            }
                            s = s + dd
                            if (s > tOut) { break }
                        }
                        t = s
                    } else {
                        hit = d.f32(1)
                        t = std.max(tIn, tNow)
                    }
                }
            }
        }
        return SubHit({hit, t})
    })

    /**
     * The grid DDA: walk cells from `tStart` to `tEnd`, testing occupancy at each cell and the
     * sub-shape on solid cells, leaping across baked clearance. `skip` is a cell to ignore (pass a
     * far sentinel for none). Returns the hit cell + refined t; `minD` is the least clearance seen
     * (world units) — a safe lower bound on a miss's distance to any solid.
     */
    const DdaHit = d.struct({hit: d.f32, t: d.f32, cx: d.f32, cy: d.f32, cz: d.f32, minD: d.f32})
    const voxelDda = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.vec3f, d.i32], DdaHit)((ro, rd, tStart, tEnd, skip, maxSteps) => {
        'use gpu'
        const vx = voxLayout.$.vox
        const v = vx.voxelSize
        const o = vx.gridOrigin
        const p0 = ro.add(rd.mul(tStart))
        let cx = std.floor((p0.x - o) / v)
        let cy = std.floor((p0.y - o) / v)
        let cz = std.floor((p0.z - o) / v)
        const sx = std.select(d.f32(-1), d.f32(1), rd.x >= 0.0)
        const sy = std.select(d.f32(-1), d.f32(1), rd.y >= 0.0)
        const sz = std.select(d.f32(-1), d.f32(1), rd.z >= 0.0)
        const ax = 1.0 / std.max(std.abs(rd.x), 0.000001)
        const ay = 1.0 / std.max(std.abs(rd.y), 0.000001)
        const az = 1.0 / std.max(std.abs(rd.z), 0.000001)
        const dtx = v * ax
        const dty = v * ay
        const dtz = v * az
        // Distance along the ray to the first boundary on each axis.
        const bx = (cx + std.max(sx, d.f32(0))) * v + o
        const by = (cy + std.max(sy, d.f32(0))) * v + o
        const bz = (cz + std.max(sz, d.f32(0))) * v + o
        let tmx = tStart + (bx - p0.x) * sx * ax
        let tmy = tStart + (by - p0.y) * sy * ay
        let tmz = tStart + (bz - p0.z) * sz * az

        let t = tStart
        let hit = d.f32(0)
        let tHit = tEnd
        let minClear = d.f32(255)
        for (let i = 0; i < maxSteps; i++) {
            const same = std.step(std.abs(cx - skip.x) + std.abs(cy - skip.y) + std.abs(cz - skip.z), 0.5)
            const val = cellValue(cx, cy, cz)
            const clear = std.max(val - 1.0, 0.0)
            minClear = std.min(minClear, clear)
            if (val < 0.5) {
                if (same < 0.5) {
                    const sh = subShapeHit(ro, rd, cellCentre(cx, cy, cz), t)
                    if (sh.hit > 0.5) {
                        hit = d.f32(1)
                        tHit = sh.t
                        break
                    }
                }
            }
            if (clear >= 1.0) {
                // Empty-space skip: the baked clearance guarantees no solid cell within `clear`
                // cells of anywhere in this one — leap and re-seed the DDA from the landing cell.
                t = t + clear * v
                const pj = ro.add(rd.mul(t))
                cx = std.floor((pj.x - o) / v)
                cy = std.floor((pj.y - o) / v)
                cz = std.floor((pj.z - o) / v)
                tmx = t + (((cx + std.max(sx, d.f32(0))) * v + o) - pj.x) * sx * ax
                tmy = t + (((cy + std.max(sy, d.f32(0))) * v + o) - pj.y) * sy * ay
                tmz = t + (((cz + std.max(sz, d.f32(0))) * v + o) - pj.z) * sz * az
            } else {
                if (tmx < tmy) {
                    if (tmx < tmz) {
                        t = tmx
                        cx = cx + sx
                        tmx = tmx + dtx
                    } else {
                        t = tmz
                        cz = cz + sz
                        tmz = tmz + dtz
                    }
                } else {
                    if (tmy < tmz) {
                        t = tmy
                        cy = cy + sy
                        tmy = tmy + dty
                    } else {
                        t = tmz
                        cz = cz + sz
                        tmz = tmz + dtz
                    }
                }
            }
            if (t > tEnd) { break }
        }
        return DdaHit({hit, t: tHit, cx, cy, cz, minD: minClear * v})
    })

    return {cellValue, cellCentre, occupied, subShapeHit, voxelDda}
}

/** The DDA step ceiling for a grid: a ray crosses at most ~3·N cells; clamped per device. */
const stepCap = () => (isMobileGpuViewport() ? 288 : 720)

/**
 * Build the SHADOW-MAP kernel: dispatched `(VOXEL_SHADOW_RES, VOXEL_SHADOW_RES)`, each texel an
 * orthographic ray from the light side of the bounding sphere along −lightDir, storing the distance
 * to the first solid (SHADOW_FAR on a miss). The map frame is `lightT1/lightT2` over ±rBound.
 */
export function buildVoxelShadowMapFn(voxLayout: VoxelLayout, style: VoxelStyle) {
    const {voxelDda} = buildVoxelTraversal(voxLayout, style)
    const RES = VOXEL_SHADOW_RES
    const STEP_CAP = stepCap()

    return tgpu.fn([d.u32, d.u32])((i, j) => {
        'use gpu'
        const vx = voxLayout.$.vox
        const rB = vx.gridOrigin * -1.0
        const u = ((d.f32(i) + 0.5) / d.f32(RES) * 2.0 - 1.0) * rB
        const w = ((d.f32(j) + 0.5) / d.f32(RES) * 2.0 - 1.0) * rB
        const L = vx.lightDir
        const p0 = vx.lightT1.mul(u).add(vx.lightT2.mul(w)).add(L.mul(rB))
        const rd = L.mul(-1.0)
        let depth = d.f32(SHADOW_FAR)
        const perp2 = u * u + w * w
        if (perp2 < rB * rB) {
            const half = std.sqrt(rB * rB - perp2)
            const maxSteps = d.i32(std.clamp(rB * 6.0 / vx.voxelSize + 8.0, d.f32(8), d.f32(STEP_CAP)))
            const none = d.vec3f(-9999.0, -9999.0, -9999.0)
            const hh = voxelDda(p0, rd, rB - half, rB + half + 0.01, none, maxSteps)
            if (hh.hit > 0.5) { depth = hh.t }
        }
        std.textureStore(voxLayout.$.shadowMap, d.vec2u(i, j), d.vec4f(depth, 0.0, 0.0, 0.0))
    }).$name('voxelShadowMap')
}

/**
 * Bilinear shadow-map comparison at map texel coords (mx, my): the fraction of the 2×2 footprint
 * whose stored depth is nearer the light than `ref` (1 = shadowed).
 */
const shadowCompare = tgpu.fn([d.texture2d(d.f32), d.f32, d.f32, d.f32], d.f32)((tex, mx, my, ref) => {
    'use gpu'
    const maxT = d.f32(VOXEL_SHADOW_RES - 1)
    const x0 = std.floor(mx)
    const y0 = std.floor(my)
    const fx = mx - x0
    const fy = my - y0
    const cx0 = d.u32(std.clamp(x0, d.f32(0), maxT))
    const cx1 = d.u32(std.clamp(x0 + 1.0, d.f32(0), maxT))
    const cy0 = d.u32(std.clamp(y0, d.f32(0), maxT))
    const cy1 = d.u32(std.clamp(y0 + 1.0, d.f32(0), maxT))
    const s00 = std.select(d.f32(0), d.f32(1), std.textureLoad(tex, d.vec2u(cx0, cy0), 0).r < ref)
    const s10 = std.select(d.f32(0), d.f32(1), std.textureLoad(tex, d.vec2u(cx1, cy0), 0).r < ref)
    const s01 = std.select(d.f32(0), d.f32(1), std.textureLoad(tex, d.vec2u(cx0, cy1), 0).r < ref)
    const s11 = std.select(d.f32(0), d.f32(1), std.textureLoad(tex, d.vec2u(cx1, cy1), 0).r < ref)
    return std.mix(std.mix(s00, s10, fx), std.mix(s01, s11, fx), fy)
}).$name('voxelShadowCompare')

/**
 * Contact-hardening soft shadow (PCSS) from the voxel shadow map, for a receiver point `p` in grid
 * space (already offset off its surface along the normal). `L`/`T1`/`T2` are the map frame,
 * `rBound` its half extent, `softness` 0..1 (0 = one bilinear-compared tap), `voxel` the cell edge
 * (depth bias scale). Returns the shadowed fraction, 1 = fully in shadow.
 */
export const voxelShadowLookup = tgpu.fn(
    [d.texture2d(d.f32), d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.f32, d.f32, d.f32],
    d.f32,
)((tex, p, L, T1, T2, rBound, softness, voxel) => {
    'use gpu'
    const RES = d.f32(VOXEL_SHADOW_RES)
    const texelWorld = rBound * 2.0 / RES
    const u = std.dot(p, T1)
    const w = std.dot(p, T2)
    const dL = rBound - std.dot(p, L)
    const mx = (u / rBound * 0.5 + 0.5) * RES - 0.5
    const my = (w / rBound * 0.5 + 0.5) * RES - 0.5
    const ref = dL - (voxel * 0.05 + texelWorld * 2.0)
    let shadow = d.f32(0)
    if (softness < 0.001) {
        shadow = shadowCompare(tex, mx, my, ref)
    } else {
        // Blocker search over the widest possible penumbra, then PCF over the penumbra the average
        // blocker distance implies (nearer occluders → sharper contact shadow).
        const k = softness * SOFT_TAN
        const rSearch = std.max(std.min(rBound * k, rBound * 0.25) / texelWorld, 1.0)
        let sum = d.f32(0)
        let count = d.f32(0)
        for (let s = 0; s < 8; s++) {
            const r = std.sqrt((d.f32(s) + 0.5) / 8.0) * rSearch
            const th = d.f32(s) * GOLDEN_ANGLE
            const sx = std.clamp(mx + std.cos(th) * r, d.f32(0), RES - 1.0)
            const sy = std.clamp(my + std.sin(th) * r, d.f32(0), RES - 1.0)
            const dep = std.textureLoad(tex, d.vec2u(d.u32(sx), d.u32(sy)), 0).r
            if (dep < ref) {
                sum = sum + dep
                count = count + 1.0
            }
        }
        if (count > 0.5) {
            const avgBlocker = sum / count
            const penumbra = std.clamp((dL - avgBlocker) * k / texelWorld, 1.0, rSearch)
            let acc = d.f32(0)
            for (let s = 0; s < 12; s++) {
                const r = std.sqrt((d.f32(s) + 0.5) / 12.0) * penumbra
                const th = d.f32(s) * GOLDEN_ANGLE + 0.7
                acc = acc + shadowCompare(tex, mx + std.cos(th) * r, my + std.sin(th) * r, ref)
            }
            shadow = acc / 12.0
        }
    }
    return shadow
}).$name('voxelShadowLookup')

// ─── Analytic per-pixel intersections (the fragment re-rasterises the G-buffer's candidate cells) ──
//
// The field texture tells the fragment WHICH voxels lie under a pixel; these resolve WHERE the
// pixel's own ray (and its sub-pixel rays) meets them, exactly — so the surface and every edge are
// pixel-exact regardless of how many field texels the shape spans. All return the entry distance
// `t` along the ray, or −1 on a miss.

/** Ray vs the axis-aligned box of half size `h` centred at `c`. */
export const voxelRayBox = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.f32], d.f32)((ro, rd, c, h) => {
    'use gpu'
    const sx = std.select(d.f32(-1), d.f32(1), rd.x >= 0.0)
    const sy = std.select(d.f32(-1), d.f32(1), rd.y >= 0.0)
    const sz = std.select(d.f32(-1), d.f32(1), rd.z >= 0.0)
    const ix = sx / std.max(std.abs(rd.x), 0.000001)
    const iy = sy / std.max(std.abs(rd.y), 0.000001)
    const iz = sz / std.max(std.abs(rd.z), 0.000001)
    const t1x = (c.x - h - ro.x) * ix
    const t2x = (c.x + h - ro.x) * ix
    const t1y = (c.y - h - ro.y) * iy
    const t2y = (c.y + h - ro.y) * iy
    const t1z = (c.z - h - ro.z) * iz
    const t2z = (c.z + h - ro.z) * iz
    const tIn = std.max(std.min(t1x, t2x), std.max(std.min(t1y, t2y), std.min(t1z, t2z)))
    const tOut = std.min(std.max(t1x, t2x), std.min(std.max(t1y, t2y), std.max(t1z, t2z)))
    const ok = std.select(d.f32(0), d.f32(1), tIn <= tOut) * std.select(d.f32(0), d.f32(1), tOut >= 0.0)
    return std.select(d.f32(-1), std.max(tIn, d.f32(0)), ok > 0.5)
}).$name('voxelRayBox')

/** Ray vs the sphere of radius `h` centred at `c`. */
export const voxelRaySphere = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.f32], d.f32)((ro, rd, c, h) => {
    'use gpu'
    const oc = ro.sub(c)
    const b = std.dot(oc, rd)
    const cc = std.dot(oc, oc) - h * h
    const disc = b * b - cc
    const t = b * -1.0 - std.sqrt(std.max(disc, 0.0))
    const ok = std.select(d.f32(0), d.f32(1), disc >= 0.0) * std.select(d.f32(0), d.f32(1), t >= 0.0)
    return std.select(d.f32(-1), t, ok > 0.5)
}).$name('voxelRaySphere')

/** Ray vs the rounded box (half `h`, corner radius `r`) centred at `c`: the box entry refined by a
 *  short sphere trace inside its slab bracket. */
export const voxelRayRoundedBox = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.f32, d.f32], d.f32)((ro, rd, c, h, r) => {
    'use gpu'
    const tBox = voxelRayBox(ro, rd, c, h)
    let t = d.f32(-1)
    if (tBox >= 0.0) {
        let s = tBox
        let hit = d.f32(0)
        for (let i = 0; i < 8; i++) {
            const p = ro.add(rd.mul(s)).sub(c)
            const dd = sdRoundBox(p, h, h, h, r)
            if (dd < h * 0.004) {
                hit = d.f32(1)
                break
            }
            s = s + dd
            if (s > tBox + h * 3.5) { break }
        }
        t = std.select(d.f32(-1), s, hit > 0.5)
    }
    return t
}).$name('voxelRayRoundedBox')

/**
 * Build the voxel march fn — plug-compatible with `buildVolumetricFieldKernel` (`(fuv, rot, rBound)
 * → vec4`). Walks the BAKED occupancy grid (`voxLayout.occ`) — no shape SDF and no light here;
 * `marchLayout` is the setup's layout (the kernel reads the field domain off it). `style` and
 * `gridSpace` are build-time strings — only the selected branches are emitted.
 */
export function buildVoxelFieldFn(
    marchLayout: VolumetricFieldLayout,
    voxLayout: VoxelLayout,
    opts: VoxelFieldOptions,
) {
    const {style, gridSpace} = opts
    const STEP_CAP = stepCap()
    const {cellCentre, occupied, voxelDda} = buildVoxelTraversal(voxLayout, style)

    /** View-space vector → grid space: camera then shape rotation when the grid rides the shape,
     *  identity when it rides the screen. (A fn, not a `let` reassignment — TGSL forbids assigning a
     *  vector reference to a `let`.) */
    const viewToGrid = tgpu.fn([d.vec3f, RotStruct], d.vec3f)((v, rot) => {
        'use gpu'
        if (gridSpace === 'shape') return rotateVec3(rotateVec3(v, voxLayout.$.vox.cam), rot)
        else return d.vec3f(v.x, v.y, v.z)
    })

    /** Classic voxel vertex AO: two edge neighbours + the corner → 0 (fully occluded) … 1 (open). */
    const vertexAo = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((s1, s2, k) => {
        'use gpu'
        const open = (3.0 - (s1 + s2 + k)) / 3.0
        return std.select(open, d.f32(0), s1 + s2 > 1.5)
    })

    const march = tgpu.fn([d.vec2f, RotStruct, d.f32], d.vec4f)((fuv, rot, rBound) => {
        'use gpu'
        const vx = voxLayout.$.vox
        const v = vx.voxelSize
        // Field UV → view ray (y up, +z into the scene); the camera tilts it, then the shape rotation
        // brings it into shape-local space when the grid rides the shape.
        const ro = viewToGrid(d.vec3f(fuv.x - 0.5, 0.5 - fuv.y, -1.2), rot)
        const rd = viewToGrid(d.vec3f(0.0, 0.0, 1.0), rot)

        const b = std.dot(ro, rd)
        const perp2 = std.max(std.dot(ro, ro) - b * b, 0.0)
        const halfSpan = std.sqrt(std.max(rBound * rBound - perp2, 0.0))
        const tStart = std.max(b * -1.0 - halfSpan, 0.0)
        const tEnd = b * -1.0 + halfSpan + 0.01
        const mp = marchLayout.$.params
        // One field texel in UV — the larger domain span, as the shared texelSpan convention.
        const texel = std.max(mp.spanX, mp.spanY) / d.f32(mp.activeRes)

        // Miss defaults: outside the bounding sphere the coverage field is the distance to it.
        let fieldVal = std.sqrt(perp2) - rBound + texel
        let payload = d.f32(0)
        let cellPacked = d.f32(0)
        let depth = tEnd + 0.4

        if (perp2 < rBound * rBound) {
            const maxSteps = d.i32(std.clamp(rBound * 6.0 / v + 8.0, d.f32(8), d.f32(STEP_CAP)))
            const none = d.vec3f(-9999.0, -9999.0, -9999.0)
            const hh = voxelDda(ro, rd, tStart, tEnd, none, maxSteps)
            if (hh.hit > 0.5) {
                const hp = ro.add(rd.mul(hh.t))
                const c = cellCentre(hh.cx, hh.cy, hh.cz)
                const l = hp.sub(c)
                // Dominant local axis = the face hit (exact for cubes; the AO frame for the curved
                // styles).
                const alx = std.abs(l.x)
                const aly = std.abs(l.y)
                const alz = std.abs(l.z)
                let axis = d.f32(0)
                if (aly > alx) {
                    if (aly >= alz) { axis = d.f32(1) } else { axis = d.f32(2) }
                } else {
                    if (alz > alx) { axis = d.f32(2) }
                }
                const la = std.select(std.select(l.x, l.y, axis > 0.5), l.z, axis > 1.5)
                const sgn = std.select(d.f32(-1), d.f32(1), la >= 0.0)
                const isX = std.select(d.f32(0), d.f32(1), axis < 0.5)
                const isZ = std.select(d.f32(0), d.f32(1), axis > 1.5)
                const isY = 1.0 - isX - isZ
                const nx = sgn * isX
                const ny = sgn * isY
                const nz = sgn * isZ
                // Face tangent axes: b = e_y on an x face, else e_x · c = e_y on a z face, else e_z.
                const bxv = 1.0 - isX
                const byv = isX
                const eyv = isZ
                const ezv = 1.0 - isZ
                const lb = l.x * bxv + l.y * byv
                const lc = l.y * eyv + l.z * ezv
                const h = v * 0.5 * vx.voxelScale
                const u = std.clamp(lb / (2.0 * h) + 0.5, 0.0, 1.0)
                const w = std.clamp(lc / (2.0 * h) + 0.5, 0.0, 1.0)

                // ── Smooth vertex AO from the 8 neighbours in the layer in front of the face ──
                const fx = hh.cx + nx
                const fy = hh.cy + ny
                const fz = hh.cz + nz
                const sB1 = occupied(fx - bxv, fy - byv, fz)
                const sB2 = occupied(fx + bxv, fy + byv, fz)
                const sC1 = occupied(fx, fy - eyv, fz - ezv)
                const sC2 = occupied(fx, fy + eyv, fz + ezv)
                const k11 = occupied(fx - bxv, fy - byv - eyv, fz - ezv)
                const k21 = occupied(fx + bxv, fy + byv - eyv, fz - ezv)
                const k12 = occupied(fx - bxv, fy - byv + eyv, fz + ezv)
                const k22 = occupied(fx + bxv, fy + byv + eyv, fz + ezv)
                const ao0 = std.mix(vertexAo(sB1, sC1, k11), vertexAo(sB2, sC1, k21), u)
                const ao1 = std.mix(vertexAo(sB1, sC2, k12), vertexAo(sB2, sC2, k22), u)
                const ao = std.mix(ao0, ao1, w)

                const faceId = axis * 2.0 + std.max(sgn, d.f32(0))
                fieldVal = texel * -1.0
                payload = faceId * 65536.0 + std.floor(ao * 255.0 + 0.5) * 256.0
                cellPacked = hh.cx + hh.cy * 256.0 + hh.cz * 65536.0
                depth = hh.t
            } else {
                fieldVal = texel + std.max(hh.minD - v, 0.0)
            }
        }
        return d.vec4f(fieldVal, payload, cellPacked, depth)
    }).$name('voxelFieldMarch')

    return march
}

// ─── The compute node ───────────────────────────────────────────────────────────────────────────

/** The live values the voxel pre-march reads each frame (a consumer maps its props onto these). */
export interface VoxelMarchValues {
    /** Cell edge in field UV. */
    voxelSize: number
    /** Occupancy threshold in cells: sdf(centre) < fill·size. 0 = centre inside; +0.5 fatter. */
    fill: number
    /** Sub-shape size within its cell (1 = touching neighbours, <1 = gaps). */
    voxelScale: number
    /** Rounded-style corner radius as a fraction of the half size. */
    bevel: number
    /** Camera pitch / yaw in degrees (yaw includes any turntable spin). */
    pitch: number
    yaw: number
    /** Key light azimuth (degrees, the fleet's screen-space `lightAngle` convention) / elevation
     *  above the screen plane (degrees). */
    lightAngle: number
    lightElevation: number
    /** Slab thickness given to FLAT shapes (field UV). */
    depth: number
}

export interface VoxelFieldSpec extends VoxelFieldOptions {
    getShapeConfig: () => unknown
    getValues: () => VoxelMarchValues
}

/** The extraFields a voxel consumer's fragment reads: the view→grid rotation (columns), the live
 *  voxel size, the grid origin, and the shadow-map frame (light direction + tangents, grid space)
 *  — everything needed to reconstruct a hit from the G-buffer and look up its shadow. */
export const VOXEL_FIELD_EXTRA_FIELDS: Record<string, {schema: typeof d.vec3f | typeof d.f32; initial: number | number[]}> = {
    _vxMx: {schema: d.vec3f, initial: [1, 0, 0]},
    _vxMy: {schema: d.vec3f, initial: [0, 1, 0]},
    _vxMz: {schema: d.vec3f, initial: [0, 0, 1]},
    _vxVoxel: {schema: d.f32, initial: 0.03},
    _vxGridOrigin: {schema: d.f32, initial: -0.6},
    _vxL: {schema: d.vec3f, initial: [0, 0, -1]},
    _vxT1: {schema: d.vec3f, initial: [1, 0, 0]},
    _vxT2: {schema: d.vec3f, initial: [0, 1, 0]},
}

/** Smallest cell edge that keeps the grid within the packing cap for a bounding radius. */
function clampVoxel(size: number, rBound: number): number {
    const minCell = (2 * (rBound + 0.002)) / VOXEL_GRID_MAX
    return Math.max(Number.isFinite(size) ? size : 0.03, minCell, 0.002)
}

type Vec3 = {x: number; y: number; z: number}
const cross = (a: Vec3, b: Vec3): Vec3 => ({x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x})
const normalize = (a: Vec3): Vec3 => {
    const l = Math.hypot(a.x, a.y, a.z) || 1
    return {x: a.x / l, y: a.y / l, z: a.z / l}
}
const clampNum = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v))

/**
 * The voxel counterpart of `createVolumetricFieldComputeNode`: routes EVERY shape type to a
 * volumetric setup (3D analytic · SVG extrude · flat SVG lifted by `depth` · flat analytic lifted by
 * `depth`), bakes its occupancy grid, renders the shadow map and marches the field — each on its own
 * dirty key — and publishes the `_vf*` SampleParams plus the `_vx*` reconstruction fields. Outputs
 * the field texture AND the shadow map (`voxelShadowMap`). Returns null without a device.
 */
export function createVoxelFieldComputeNode(
    params: GpuFragmentParams,
    spec: VoxelFieldSpec,
): {
    outputs: {volumetricFieldTexture: KitTexture; voxelShadowMap: KitTexture}
    getComputeNodes: (frameParams?: unknown) => GpuComputeStep[] | null
} | null {
    const root = params.gpu?.root
    if (!root) return null

    const cfg = parseShapeConfigValue(spec.getShapeConfig())
    const shapeSdfUrl = (params.getCpuValue('shapeSdfUrl') as string) || ''
    const shapeType = (params.getCpuValue('shapeType') as string) || (cfg.type as string) || ''

    let values = spec.getValues()
    let lastRBound = 0.6
    let voxel = clampVoxel(values.voxelSize, lastRBound)
    // Voxels puff past the smooth surface by up to half a cell diagonal: the setups grow their
    // bounding sphere + field domain by it so no cube is clipped at the silhouette.
    const extraPad = () => voxel * HALF_DIAGONAL + 0.01

    let setup: VolumetricFieldSetup
    if (isSvgShape(shapeSdfUrl, cfg)) {
        const getCfg = isSvg3dShapeType(shapeType)
            ? spec.getShapeConfig
            : () => ({
                ...parseShapeConfigValue(spec.getShapeConfig()),
                type: 'svgExtrude3D', depth: values.depth, bevel: 0, rotX: 0, rotY: 0, rotZ: 0,
            })
        setup = createSvg3dSdfSetup(params, shapeSdfUrl, getCfg, {extraPad})
    } else if (is3dShapeType(shapeType)) {
        setup = createAnalytic3dSdfSetup(root, shapeType, cfg, spec.getShapeConfig, {extraPad})
    } else {
        setup = createAnalytic2dExtrudeSetup(root, resolveShapeType(shapeType, cfg), spec.getShapeConfig, () => values.depth * 0.5, {extraPad})
    }

    // Shared resources: the voxel uniform, the occupancy grid, the shadow map.
    const voxLayout = makeVoxelLayout()
    const bakeLayout = makeVoxelBakeLayout()
    const voxUniform = root.createUniform(VoxelMarchParams)
    const occBuffer = createStateBuffer(root, d.u32, OCC_BUFFER_WORDS)
    const shadowTexture: TgpuTexture = root
        .createTexture({size: [VOXEL_SHADOW_RES, VOXEL_SHADOW_RES], format: VOXEL_SHADOW_FORMAT})
        .$usage('storage', 'sampled')
    params.onCleanup(() => {
        occBuffer.destroy()
        shadowTexture.destroy()
    })
    const voxBindGroup = root.createBindGroup(voxLayout, {vox: voxUniform.buffer, occ: occBuffer, shadowMap: shadowTexture} as never)
    const bakeBindGroup = root.createBindGroup(bakeLayout, {vox: voxUniform.buffer, occ: occBuffer} as never)

    // 3. The march (the volumetric spine's kernel/compute over our march fn).
    const march = buildVoxelFieldFn(setup.layout, voxLayout, {style: spec.style, gridSpace: spec.gridSpace})
    const kernel = buildVolumetricFieldKernel(setup.layout, march)
    const compute = createVolumetricFieldCompute(root, setup.layout, kernel, params.onCleanup, {
        paramsUniform: setup.marchParamsBuffer,
        sdfSourceTexture: setup.sdfSourceTexture,
        svgOutlineBuffers: setup.svgOutlineBuffers,
        extraBindGroups: [voxBindGroup],
    })
    const fieldTexture = params.registerComputeTexture(compute.fieldTexture)
    const shadowMap = params.registerComputeTexture(shadowTexture)

    // 1. The grid bake: the shape SDF over every cell. Its bind group over the setup layout mirrors
    //    the march's (the SDF reads its params there).
    const bakeFn = buildVoxelBakeFn(setup.sdfFn, setup.layout, bakeLayout, spec.gridSpace)
    const setupBindGroup = root.createBindGroup(setup.layout, {
        field: compute.fieldTexture,
        params: setup.marchParamsBuffer.buffer,
        ...(setup.sdfSourceTexture ? {sdfSource: setup.sdfSourceTexture} : {}),
        ...(setup.svgOutlineBuffers ?? {}),
    } as never)
    const bake = createGuardedCompute(root, (wx: number, iy: number, iz: number) => {
        'use gpu'
        bakeFn(wx, iy, iz)
    }, {bindGroup: setupBindGroup}).with(bakeBindGroup as never)

    // 2. The shadow map: one DDA per texel from the light, over the baked grid only.
    const shadowFn = buildVoxelShadowMapFn(voxLayout, spec.style)
    const shadowPass = createGuardedCompute(root, (i: number, j: number) => {
        'use gpu'
        shadowFn(i, j)
    }, {bindGroup: voxBindGroup, size: [VOXEL_SHADOW_RES, VOXEL_SHADOW_RES]})

    let activeRes = compute.maxRes
    let lastMarchKey = ''
    let lastBakeKey = ''
    let lastShadowKey = ''
    const marchEvery = isMobileGpuViewport() ? 2 : 1
    let changedFrames = 0

    const getComputeNodes = (frameParams?: unknown): GpuComputeStep[] | null => {
        values = spec.getValues()
        voxel = clampVoxel(values.voxelSize, lastRBound)
        setup.update(frameParams as ShapeFrameParams | undefined)
        const fp = setup.getFootprint()
        lastRBound = fp.rBound
        voxel = clampVoxel(voxel, fp.rBound)
        const scale = (params.getCpuValue('scale') as number) ?? 1
        activeRes = resolveActiveFieldRes(fp.spanX, fp.spanY, scale, params.dimensions.height, compute.maxRes, activeRes)

        // View → grid: camera (pitch about x, yaw about y) then the shape rotation.
        const pitch = values.pitch * DEG
        const yaw = values.yaw * DEG
        const cam = {cx: Math.cos(pitch), sx: Math.sin(pitch), cy: Math.cos(yaw), sy: Math.sin(yaw), cz: 1, sz: 0}
        const rot = setup.getRotation()
        const R = (x: number, y: number, z: number): Vec3 => {
            const a = rotateVecCpu(x, y, z, cam)
            return rotateVecCpu(a.x, a.y, a.z, rot)
        }
        const Mx = R(1, 0, 0)
        const My = R(0, 1, 0)
        const Mz = R(0, 0, 1)

        // Key light: the fleet's material convention (x right, y DOWN, z into the scene) → view space
        // (y up) → grid space. T1/T2 complete the shadow-map frame.
        const az = values.lightAngle * DEG
        const el = clampNum(values.lightElevation, 1, 89) * DEG
        const LV: Vec3 = {x: Math.cos(az) * Math.cos(el), y: -Math.sin(az) * Math.cos(el), z: -Math.sin(el)}
        const LG = spec.gridSpace === 'shape' ? R(LV.x, LV.y, LV.z) : LV
        const helper: Vec3 = Math.abs(LG.x) < 0.9 ? {x: 1, y: 0, z: 0} : {x: 0, y: 1, z: 0}
        const T1 = normalize(cross(LG, helper))
        const T2 = cross(LG, T1)

        const gridOrigin = -fp.rBound
        const gridN = Math.min(VOXEL_GRID_MAX, Math.ceil((2 * fp.rBound) / voxel))
        const gridW = Math.ceil(gridN / 4)
        const fill = clampNum(values.fill, -0.5, 0.5)
        const voxelScale = clampNum(values.voxelScale, 0.2, 1)
        const bevel = clampNum(values.bevel, 0, 1)
        voxUniform.write({
            cam,
            lightDir: d.vec3f(LG.x, LG.y, LG.z), voxelSize: voxel,
            lightT1: d.vec3f(T1.x, T1.y, T1.z), fill,
            lightT2: d.vec3f(T2.x, T2.y, T2.z), voxelScale,
            bevel, gridOrigin, gridN, gridW,
        })

        // Three dirty keys. The grid depends on geometry + cell size + fill (+ rotation/camera only
        // when the grid rides the screen); the shadow map adds the light frame and the sub-shape; the
        // march adds the view (rotation/camera), the active resolution and the sub-shape — no light.
        const geometryKey = setup.getGeometryKey ? setup.getGeometryKey() : setup.getStateKey()
        const gridKey = spec.gridSpace === 'view'
            ? `${setup.getStateKey()},${values.pitch},${values.yaw},${voxel},${fill},${gridN}`
            : `${geometryKey},${voxel},${fill},${gridN}`
        const subShapeKey = `${voxelScale},${bevel}`
        const lightKey = `${LG.x},${LG.y},${LG.z}`
        const bakeKey = gridKey
        const shadowKey = `${gridKey},${subShapeKey},${lightKey}`
        const marchKey = `${gridKey},${subShapeKey},${setup.getStateKey()},${values.pitch},${values.yaw},${activeRes}`
        const rebake = bakeKey !== lastBakeKey
        const reshadow = shadowKey !== lastShadowKey
        const remarch = marchKey !== lastMarchKey
        if (!rebake && !reshadow && !remarch) return null
        if (remarch) {
            // Mobile re-march throttle: an animated shape changes every frame; march every Nth.
            changedFrames++
            if (marchEvery > 1 && changedFrames % marchEvery !== 0 && !rebake) return null
        }
        lastBakeKey = bakeKey
        lastShadowKey = shadowKey
        const steps: GpuComputeStep[] = []
        if (rebake) steps.push(() => bake.dispatchThreads(gridW, gridN, gridN))
        if (reshadow) {
            params.setExtraField('_vxL', [LG.x, LG.y, LG.z])
            params.setExtraField('_vxT1', [T1.x, T1.y, T1.z])
            params.setExtraField('_vxT2', [T2.x, T2.y, T2.z])
            steps.push(shadowPass)
        }
        if (remarch) {
            lastMarchKey = marchKey
            setup.setActiveRes(activeRes)
            params.setExtraField('_vfOriginX', fp.originX)
            params.setExtraField('_vfOriginY', fp.originY)
            params.setExtraField('_vfSpanX', fp.spanX)
            params.setExtraField('_vfSpanY', fp.spanY)
            params.setExtraField('_vfActiveRes', activeRes)
            params.setExtraField('_vfRBound', fp.rBound)
            params.setExtraField('_vxMx', [Mx.x, Mx.y, Mx.z])
            params.setExtraField('_vxMy', [My.x, My.y, My.z])
            params.setExtraField('_vxMz', [Mz.x, Mz.y, Mz.z])
            params.setExtraField('_vxVoxel', voxel)
            params.setExtraField('_vxGridOrigin', gridOrigin)
            steps.push(() => compute.dispatch(activeRes))
        }
        return steps
    }

    return {
        outputs: {volumetricFieldTexture: fieldTexture, voxelShadowMap: shadowMap},
        getComputeNodes,
    }
}
