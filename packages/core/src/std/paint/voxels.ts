/**
 * std/paint/voxels — any shape rebuilt from blocks.
 *
 * `voxelSurface` turns a shape (flat, SVG or 3D) into a solid of cubes, spheres or rounded
 * cubes and hands your `surface:` one block at a time with everything a material needs already
 * worked out: its normal, its baked ambient occlusion, the shadow other blocks cast on it, its
 * cell position and a stable per-block random number. You write only the look, a color and a
 * light response over that frame with the `materials` words, and return `vec4(rgb, coverage)`.
 * Spread the result into a shape-effect definition next to the standard shape props.
 */
// Maintainer notes. This is the shape-effect spine (shapedSurface) with the smooth march
// swapped for the voxel pre-march (kit/voxels).
//
// PIXEL-EXACT GEOMETRY: the field texture is only used to find WHICH voxels lie under a pixel
// (the cells of the four nearest texels). The fragment then intersects the pixel's OWN ray with
// those candidate cells analytically (kit `voxelRay*`) — so sphere surfaces, face edges and
// silhouettes are exact whatever the field resolution — shades each candidate once at its
// centre-ray hit, and resolves visibility with eight sub-pixel rays (an 8× rotated-grid
// supersample of the edges). The shadow lookup runs once per pixel at the nearest hit. The
// `insideShape` guard wraps it all here.
//
// The geometry knobs (`voxelSize`, `fill`, `voxelScale`, `bevel`, camera, light direction,
// `depth`) are PropRefs the compute half reads live (`shadowSoftness` is fragment-side); `style`
// and `gridSpace` are compile-time selects (the kernel emits only the chosen branches). Their
// change re-marches the field (like rotating a 3D shape does); color/material props never do.
// The fragment also reads `uniforms.scale` and `uniforms.shadows` (the shadow lookup is skipped
// when `shadows` is 0), so a consumer must declare both props.
import type {Expr, GpuFragmentParams, GpuComputeNode, KitTexture} from '../../gpu/contract'
import {call, floatE, ZERO} from '../../gpu/composer'
import {noise, voxels as voxelKit} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {uniformOf} from '../invoke'
import {
    shapedSurface, surfaceField, viewRay, guarded, insideShape,
    type ShapedSurfaceEffect, type SurfaceFrame, type SurfaceField,
} from './materials'
import {
    abs, add, clamp, div, dot, floor, ge, gt, local, lt, max, min, mul, neg, normalize, select, sign, sub, vec2, vec3, vec4,
} from '../math'

/** What each block is: `'cube'`, `'sphere'` or `'rounded'` (a cube with rounded corners). */
export type VoxelStyle = voxelKit.VoxelStyle
/** Where the grid sits: `'shape'` turns with the shape like a built model, `'view'` stays fixed to the canvas. */
export type VoxelGridSpace = voxelKit.VoxelGridSpace

/** Sub-pixel ray offsets (pixel units): the 8-sample rotated-grid pattern. */
const SUBSAMPLES: Array<[number, number]> = [
    [1 / 16, -3 / 16], [-1 / 16, 3 / 16], [5 / 16, 1 / 16], [-3 / 16, -5 / 16],
    [-5 / 16, 5 / 16], [-7 / 16, -1 / 16], [3 / 16, 7 / 16], [7 / 16, -7 / 16],
]
const FAR = 1e9

/** What `voxelSurface` needs. Every field but `surface` is a prop ref, `p('name')`. */
export interface VoxelSurfaceSpec {
    /** The block style, `'cube' | 'sphere' | 'rounded'`. Mark the prop `compileTime: true`. */
    style: PropRef
    /** Where the grid sits, `'shape' | 'view'`. Mark the prop `compileTime: true`. */
    gridSpace: PropRef
    /** The edge length of one block, relative to the shape. 0.035 is about a tenth of the default sphere. */
    voxelSize: PropRef
    /** How eagerly cells along the surface fill in, −0.5 to 0.5. Negative carves the model thinner, positive puffs it out. */
    fill: PropRef
    /** The size of each block inside its cell, 0.3–1. 1 touches its neighbours, less opens gaps. */
    voxelScale: PropRef
    /** Corner rounding as a fraction of the block size, 0–1. A hairline chamfer on cubes, the corner radius of rounded cubes. */
    bevel: PropRef
    /** The soft-edge width of cast shadows, 0–1. 0 is razor sharp. */
    shadowSoftness: PropRef
    /** Optional camera tilt, in degrees. */
    pitch?: PropRef
    /** Optional camera turn, in degrees. */
    yaw?: PropRef
    /** Optional turntable speed. Declare `animatedTime: {speed: 'spin'}` too. A speed of 1 is one turn every ten seconds. */
    spin?: PropRef
    /** The direction the key light comes from, in degrees around the canvas. Shadows fall away from it. */
    lightAngle: PropRef
    /** How high the key light sits above the canvas, in degrees. Low for long shadows, high for flat top light. */
    lightElevation: PropRef
    /** The thickness given to flat shapes so they become a slab of blocks, relative to the shape. 3D shapes ignore it. */
    depth: PropRef
    /** Your material: the color of one block. Return `vec4(rgb, vox.coverage)`. */
    surface: (vox: VoxelFrame, frame: SurfaceFrame, params: GpuFragmentParams) => Expr
}
// Maintainer notes on the spec. `shadowSoftness` is fragment-side (drivable). `pitch`/`yaw`
// apply to the view ray before the shape rotation. Mouse and auto drivers on the light props
// are honoured by the shadow map too (`getCpuValue` resolves them); a texture MAP driver only
// reaches the fragment lighting, the shadow direction then stays at the prop's static value.
// The spine resolves visibility and coverage itself from the analytic hits, so only the
// surface's `.rgb` is consumed.

/** One block, as `voxelSurface` hands it to your `surface`. */
export interface VoxelFrame {
    /** The shape field basics: `sdf`, `pxH` (one pixel in uv) and `aspect`. */
    field: SurfaceField
    /** 1 when this is a real block, 0 for an empty slot. Return it as your alpha. */
    coverage: Expr
    /** The block's unit normal at this pixel. x right, y down, z negative toward the viewer. */
    normal: Expr
    /** How open the block's surroundings are, 1 in the open and lower in creases. Multiply ambient light by it. */
    ao: Expr
    /** How much shadow other blocks cast here, 0 in the light and 1 fully shadowed. */
    shadow: Expr
    /** The block's grid position as three whole numbers in a vec3. */
    cell: Expr
    /** A random 0–1 number that stays the same for this block every frame. Use it for per-block color variation. */
    cellHash: Expr
    /** How high the block sits, 0 at the bottom of the model and 1 at the top. */
    heightT: Expr
    /** How deep the block sits, 0 nearest the viewer and 1 farthest. */
    depthT: Expr
    /** How far this pixel is from the centre of its face toward the edge, 0–1. Seams live near 1. */
    edge: Expr
    /** The view ray into the scene, in the same convention as `normal`. */
    view: Expr
    /** The block style in force, known when the shader compiles. */
    style: VoxelStyle
}
// Maintainer notes on the frame. `shadow` is the contact-hardening shadow-map lookup, shared
// across the pixel's candidates (evaluated at the nearest hit). `ao` is the baked smooth vertex
// AO. `normal` is exact for faces, spheres and the rounded-cube fillet. `heightT`/`depthT` are
// normalised across the bounding volume.

const styleOf = (v: unknown): VoxelStyle => (v === 'sphere' || v === 'rounded' ? v : 'cube')
const gridSpaceOf = (v: unknown): VoxelGridSpace => (v === 'view' ? 'view' : 'shape')

/**
 * Rebuild the shape from blocks and shade each block with your `surface`.
 *
 * Returns `{extraFields, compute, gpu}` to spread into a shape-effect definition. Your props
 * must include the standard shape props (`center`, `scale`, `rotation`, `shape`, `shapeSdfUrl`,
 * `shapeType`), a `shadows` amount (0–1) and everything the spec names. Your `surface` gets a
 * `VoxelFrame` per block and returns `vec4(rgb, vox.coverage)`. Geometry props re-build the
 * model when they change. Color and light props never do.
 *
 * @example
 * ```ts
 * ...voxelSurface({style: p('voxelShape'), gridSpace: p('gridSpace'), voxelSize: p('voxelSize'), fill: p('fill'), voxelScale: p('voxelScale'), bevel: p('bevel'), shadowSoftness: p('shadowSoftness'), lightAngle: p('lightAngle'), lightElevation: p('lightElevation'), depth: p('depth'), surface: (vox, frame, params) => math.vec4(shade(vox), vox.coverage)})
 * ```
 * @tip Light with `materials.lambert(vox.normal, L, {wrap: 0.12})` times `1 − vox.shadow · shadows` and the blocks read as solid.
 * @see shapedSurface, lambert
 */
export function voxelSurface(spec: VoxelSurfaceSpec): ShapedSurfaceEffect {
    // The voxel spine noun: shapedSurface with the centre-only fast tap; the pre-march is the
    // kit's `createVoxelFieldComputeNode`, which also publishes the shadow map.
    const spine = shapedSurface({
        stencil: 'centre',
        centreTap: 'fast',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const style = styleOf(params.propValues[spec.style.name])
            const gridSpace = gridSpaceOf(params.propValues[spec.gridSpace.name])
            const field = surfaceField(frame, params, {scale: u.scale})
            const view = viewRay(params, field)

            // ── Frames: view → grid, the pixel's ray, a pixel's footprint in grid space ──
            const Mx = u._vxMx
            const My = u._vxMy
            const Mz = u._vxMz
            const toGrid = (p: Expr): Expr => gridSpace === 'shape'
                ? add(add(mul(Mx, p.member('x')), mul(My, p.member('y'))), mul(Mz, p.member('z')))
                : p
            const toView = (n: Expr): Expr => gridSpace === 'shape'
                ? vec3(dot(Mx, n), dot(My, n), dot(Mz, n))
                : n
            const suv = frame.sdfUV!
            const roG = local(toGrid(vec3(sub(suv.member('x'), 0.5), sub(0.5, suv.member('y')), -1.2)), 'vxRo')
            const rdG = local(toGrid(vec3(0, 0, 1)), 'vxRd')
            // One device pixel in field units (the field is isotropic; sdfUV = screen / scale).
            const pxUV = local(div(field.pxH, max(u.scale, 0.001)), 'vxPx')
            const ex = local(toGrid(vec3(pxUV, 0, 0)), 'vxEx')
            const ey = local(toGrid(vec3(0, neg(pxUV), 0)), 'vxEy')
            const v = local(u._vxVoxel, 'vxSize')
            const o = local(u._vxGridOrigin, 'vxOrigin')
            const half = local(mul(mul(v, 0.5), uniformOf(spec.voxelScale, params)), 'vxHalf')
            const rEff = local(max(mul(uniformOf(spec.bevel, params), half), mul(half, 0.04)), 'vxREff')
            const span = local(mul(o, -2), 'vxSpan')

            /** Analytic ray vs a candidate cell's sub-shape → t, or −1 on a miss. */
            const rayHit = (ro: Expr, centre: Expr): Expr => {
                if (style === 'sphere') return call(voxelKit.voxelRaySphere, 'voxelRaySphere', [ro, rdG, centre, half])
                if (style === 'rounded') return call(voxelKit.voxelRayRoundedBox, 'voxelRayRoundedBox', [ro, rdG, centre, half, mul(uniformOf(spec.bevel, params), half)])
                return call(voxelKit.voxelRayBox, 'voxelRayBox', [ro, rdG, centre, half])
            }

            // ── The four field texels under this pixel → candidate voxels ──
            const res = local(u._vfActiveRes, 'vxRes')
            const ox = u._vfOriginX
            const oy = u._vfOriginY
            const sx = u._vfSpanX
            const sy = u._vfSpanY
            const maxT = sub(res, 1)
            const px = sub(mul(div(sub(suv.member('x'), ox), sx), res), 0.5)
            const py = sub(mul(div(sub(suv.member('y'), oy), sy), res), 0.5)
            const x0 = local(clamp(floor(px), 0, maxT), 'vxX0')
            const y0 = local(clamp(floor(py), 0, maxT), 'vxY0')
            const x1 = min(add(x0, 1), maxT)
            const y1 = min(add(y0, 1), maxT)
            const texelUV = (tx: Expr, ty: Expr): Expr => vec2(
                add(ox, mul(div(add(tx, 0.5), res), sx)),
                add(oy, mul(div(add(ty, 0.5), res), sy)),
            )
            const corners: Array<{tx: Expr; ty: Expr; tag: string}> = [
                {tx: x0, ty: y0, tag: '00'}, {tx: x1, ty: y0, tag: '10'},
                {tx: x0, ty: y1, tag: '01'}, {tx: x1, ty: y1, tag: '11'},
            ]
            const cands = corners.map((c) => {
                const h = (name: string) => `${name}${c.tag}`
                const uvC = local(texelUV(c.tx, c.ty), h('vxUv'))
                const tex = local(frame.texelSampler(uvC), h('vxTex'))
                const cov = local(select(lt(tex.member('x'), 0), 1, 0), h('cov'))
                const g = tex.member('y')
                const faceId = floor(div(g, 65536))
                const ao = local(div(floor(div(sub(g, mul(faceId, 65536)), 256)), 255), h('ao'))
                const bCh = tex.member('z')
                const cz = local(floor(div(bCh, 65536)), h('cz'))
                const bRest = sub(bCh, mul(cz, 65536))
                const cy = local(floor(div(bRest, 256)), h('cy'))
                const cx = local(sub(bRest, mul(cy, 256)), h('cx'))
                const cell = local(vec3(cx, cy, cz), h('cell'))
                const centre = local(add(mul(add(cell, 0.5), v), o), h('centre'))
                // The texel's own hit (fallback when the pixel-centre ray misses this candidate).
                const texDepth = tex.member('w')
                const texHp = local(toGrid(vec3(sub(uvC.member('x'), 0.5), sub(0.5, uvC.member('y')), sub(texDepth, 1.2))), h('texHp'))
                // Pixel-centre ray vs this candidate.
                const tHit = local(select(gt(cov, 0.5), rayHit(roG, centre), -1), h('tC'))
                const valid = gt(tHit, -0.5)
                const hp = local(select(valid, add(roG, mul(rdG, tHit)), texHp), h('hp'))
                const depth = local(select(valid, tHit, texDepth), h('depth'))
                const l = local(sub(hp, centre), h('l'))
                // Face axis = dominant local axis (exact on cubes; the rounding frame on the rest).
                const al = local(abs(l), h('al'))
                const mx = max(al.member('x'), max(al.member('y'), al.member('z')))
                const isX = local(select(ge(al.member('x'), mx), 1, 0), h('isX'))
                const isY = local(mul(select(ge(al.member('y'), mx), 1, 0), sub(1, isX)), h('isY'))
                const isZ = sub(sub(1, isX), isY)
                const faceN = local(mul(sign(l), vec3(isX, isY, isZ)), h('faceN'))
                const tangentMask = local(sub(1, abs(faceN)), h('tMask'))
                let nGraw: Expr
                if (style === 'sphere') {
                    nGraw = normalize(l)
                } else {
                    // Rounded-box normal: exact face axis + tangential terms within the rounding radius
                    // of an edge (a hairline chamfer on hard cubes, the real fillet on rounded).
                    const q = mul(mul(sign(l), max(sub(al, sub(half, rEff)), vec3(0, 0, 0))), tangentMask)
                    nGraw = normalize(add(mul(faceN, rEff), q))
                }
                const nG = local(nGraw, h('nG'))
                const nV = local(toView(nG), h('nV'))
                const normal = local(vec3(nV.member('x'), neg(nV.member('y')), nV.member('z')), h('n'))
                const lt2 = mul(al, tangentMask)
                const edge = local(clamp(div(max(max(lt2.member('x'), lt2.member('y')), lt2.member('z')), half), 0, 1), h('edge'))
                const cellHash = local(call(noise.hash13, 'hash13', [add(cell, vec3(0.37, 0.61, 0.83))]), h('hash'))
                const heightT = local(clamp(div(mul(add(cy, 0.5), v), span), 0, 1), h('heightT'))
                const depthT = local(clamp(div(sub(depth, add(1.2, o)), span), 0, 1), h('depthT'))
                const tSort = local(select(valid, tHit, FAR), h('tSort'))
                return {tag: c.tag, cov, cell, centre, hp, nG, tSort, frame: {field, coverage: cov, normal, ao, cell, cellHash, heightT, depthT, edge, view, style}}
            })
            // Same-cell flags: the four texels usually name ONE voxel; a duplicate reuses the first
            // candidate's shading and, when all four agree on a solid cell, the pixel is interior to it.
            const sameAs0 = (k: number): Expr => local(mul(select(lt(abs(sub(cands[k].cell, cands[0].cell)).member('x'), 0.5), 1, 0),
                mul(select(lt(abs(sub(cands[k].cell, cands[0].cell)).member('y'), 0.5), 1, 0), select(lt(abs(sub(cands[k].cell, cands[0].cell)).member('z'), 0.5), 1, 0))), `vxSame${k}`)
            const same = [floatE(1), sameAs0(1), sameAs0(2), sameAs0(3)]
            const allOne = local(mul(mul(mul(same[1], same[2]), same[3]), mul(mul(cands[0].cov, cands[1].cov), mul(cands[2].cov, cands[3].cov))), 'vxAllOne')

            // ── Shadow once per pixel, at the nearest centre-ray hit (fallback: the first texel) ──
            const shadowTex = params.computeOutputs?.voxelShadowMap as KitTexture | undefined
            let shadow: Expr = floatE(0)
            if (shadowTex) {
                const pick = (a: {tSort: Expr; hp: Expr; nG: Expr}, b: {tSort: Expr; hp: Expr; nG: Expr}) => {
                    const aWins = lt(a.tSort, b.tSort)
                    return {tSort: select(aWins, a.tSort, b.tSort), hp: select(aWins, a.hp, b.hp), nG: select(aWins, a.nG, b.nG)}
                }
                const best = pick(pick(cands[0], cands[1]), pick(cands[2], cands[3]))
                const hpS = local(best.hp, 'vxHpShadow')
                const nS = local(best.nG, 'vxNShadow')
                const lookup = call(voxelKit.voxelShadowLookup, 'voxelShadowLookup', [
                    shadowTex.accessor(),
                    add(hpS, mul(nS, mul(v, 0.12))),
                    u._vxL, u._vxT1, u._vxT2,
                    neg(o),
                    uniformOf(spec.shadowSoftness, params),
                    v,
                ])
                // The lookup only runs when the material can show it (`shadows` > 0) AND the surface
                // faces the light: a face turned away receives no direct light, so it is simply
                // shadowed (fallback 1) — no rays needed. Below −0.12 the wrapped Lambert is 0 anyway.
                const facing = gt(dot(nS, u._vxL), -0.12)
                const wanted = gt(mul(select(gt(u.shadows, 0.001), 1, 0), select(facing, 1, 0)), 0.5)
                shadow = guarded(wanted, lookup, floatE(1), 'shadow', [hpS, nS])
            }

            // ── Shade each candidate once — a duplicate of an earlier cell reuses its shading ──
            const shadeDeps = cands.flatMap((c) => [c.cov, c.centre, c.tSort, c.frame.normal, c.frame.ao, c.frame.cellHash, c.frame.heightT, c.frame.depthT, c.frame.edge])
            const shade0 = local(spec.surface({...cands[0].frame, shadow}, frame, params).member('rgb'), 'vxShade00')
            const shaded = cands.map((c, k) => k === 0
                ? shade0
                : guarded(lt(same[k], 0.5), spec.surface({...c.frame, shadow}, frame, params).member('rgb'), shade0, `vxShade${c.tag}`, k === 1 ? shadeDeps : []))

            // ── Visibility: eight sub-pixel rays, nearest candidate wins, misses stay transparent.
            //    Skipped when all four texels are the same solid voxel: every sub-sample would land on
            //    the same color, so the pixel is that candidate at full coverage. ──
            let alpha: Expr = floatE(0)
            let rgb: Expr = vec3(0, 0, 0)
            SUBSAMPLES.forEach(([dx, dy], si) => {
                const ro = local(add(add(roG, mul(ex, dx)), mul(ey, dy)), `vxRo${si}`)
                const ts = cands.map((c) => {
                    const t = local(select(gt(c.cov, 0.5), rayHit(ro, c.centre), -1), `vxTr${si}_${c.tag}`)
                    return local(select(gt(t, -0.5), t, FAR), `vxT${si}_${c.tag}`)
                })
                const t23 = min(ts[2], ts[3])
                const t123 = min(ts[1], t23)
                const best = min(ts[0], t123)
                const col = select(lt(ts[0], t123), shaded[0],
                    select(lt(ts[1], t23), shaded[1],
                        select(lt(ts[2], ts[3]), shaded[2], shaded[3])))
                const hit = local(select(lt(best, FAR * 0.5), 1, 0), `vxHit${si}`)
                alpha = add(alpha, hit)
                rgb = add(rgb, mul(col, hit))
            })
            const resolved = vec4(div(rgb, max(alpha, 0.0001)), div(alpha, SUBSAMPLES.length))
            const interior = vec4(shade0, 1)
            const out = guarded(lt(allOne, 0.5), resolved, interior, 'vxOut', [shade0, ...shaded.slice(1)])
            return guarded(insideShape(field.sdf, field.pxH), out, ZERO, 'voxels')
        },
    })

    const compute: GpuComputeNode = (params) => {
        const num = (ref: PropRef | undefined, fallback: number): number => {
            if (!ref) return fallback
            const raw = params.getCpuValue(ref.name)
            return typeof raw === 'number' && Number.isFinite(raw) ? raw : fallback
        }
        const clock = (): number => {
            const t = params.getCpuValue('_animTime')
            return typeof t === 'number' && Number.isFinite(t) ? t : 0
        }
        return voxelKit.createVoxelFieldComputeNode(params, {
            style: styleOf(params.propValues[spec.style.name]),
            gridSpace: gridSpaceOf(params.propValues[spec.gridSpace.name]),
            getShapeConfig: () => params.getCpuValue('shape'),
            getValues: () => ({
                voxelSize: num(spec.voxelSize, 0.03),
                fill: num(spec.fill, 0),
                voxelScale: num(spec.voxelScale, 1),
                bevel: num(spec.bevel, 0),
                pitch: num(spec.pitch, 0),
                yaw: num(spec.yaw, 0) + (spec.spin ? clock() * 36 : 0),
                lightAngle: num(spec.lightAngle, 225),
                lightElevation: num(spec.lightElevation, 45),
                depth: num(spec.depth, 0.12),
            }),
        }) as ReturnType<GpuComputeNode>
    }

    return {
        extraFields: {...spine.extraFields, ...voxelKit.VOXEL_FIELD_EXTRA_FIELDS},
        compute,
        gpu: spine.gpu,
    }
}
