/**
 * std/paint/radiance — light a shape gives off or blocks.
 *
 * Words for real light spilling from a shape's edges into its surroundings, with the shape's
 * body casting shadows. You work inside a shape-effect material (the `surface:` of
 * `shapedSurface`). `irradianceField` gathers the light once into a texture and re-gathers only
 * when a light or the shape moves, so a still scene costs one texture read per pixel.
 * `shadowVisibility` asks whether a light reaches a point. `gatherIrradiance` is the same
 * gather done per pixel, for when you need it live.
 */
// Maintainer notes. Light that a SHAPE gives off or blocks cannot be read from the nearest
// boundary point alone: that partitions the plane along the shape's medial axis (hard seams),
// ignores everything the body occludes (light inside a ring's hole), and stamps an invented
// distance profile along whole edges. These words do the transport instead:
//
//  - gatherIrradiance — a cone gather: the circle around a pixel is tiled by `cones` cones,
//    each sphere-traced against the field. A cone does not hit or miss — it tracks its closest
//    approach to the boundary and reports the FRACTION of its width the shape covers (the
//    distance-field cone-tracing estimate), so the gather is a smooth integral rather than a
//    binary Monte-Carlo sample: seamless, physically thinning with the emitter's angular size,
//    occluded where the body stands in the way, and nearly noise-free at 16 cones. Each cone
//    contributes the caller's `emission` at its closest boundary point, weighted by coverage.
//    The per-pixel `jitter` rotates the fan a fraction of a cone to dissolve the residual
//    cone-boundary structure into fine grain.
//  - shadowVisibility — a shadow ray from a boundary point toward a light: 1 when the segment
//    is clear, 0 when the body blocks it, with the same cone penumbra estimate.
//
// Both are Expr-STATEMENT machinery (real WGSL loops, like `guarded`'s branch): the loop bodies
// host every `local()` the callbacks hoist, so anything shared with the enclosing recipe must be
// pre-emitted through `deps` (the `guarded` rule). The field closure must be flow-safe (explicit
// LOD / textureLoad samplers — the shape-effect spine's samplers all are).
import type {EmitContext, GpuComputeStep} from '../../gpu/contract'
import {Expr} from '../../gpu/contract'
import {d} from '../../gpu/kit/index'
import {div, local, max} from '../math'
import {formatFloat} from '../../gpu/contract'
import {PI, TAU} from '../../gpu/kit/constants'

let radianceCounter = 0

const raw = (name: string): Expr => new Expr(() => name)

const num = (v: Expr | number, ctx: EmitContext): string => (typeof v === 'number' ? formatFloat(v) : v._emit(ctx))

/** A child emit context whose statements land in `into` (the loop / branch body). */
function scopedInto(ctx: EmitContext, into: string[]): EmitContext {
    return {
        external: (v, h) => ctx.external(v, h),
        statement: (w) => into.push(w),
        freshLocal: (h) => ctx.freshLocal(h),
        memo: (k, f) => ctx.memo(k, f),
    }
}

/** What `gatherIrradiance` needs. */
export interface GatherSpec {
    /** The pixel, as a 2D point in the shape field's coordinates. */
    origin: Expr
    /** The shape as a signed distance: given a point, how far it is from the edge (negative inside). */
    field: (at: Expr) => Expr
    /** How many directions to look in. 16 is nearly noise-free. Fixed when the shader compiles. */
    cones: number
    /** The most steps each direction may take toward the shape. */
    steps: number
    /** The distance beyond which light no longer counts, in field units. */
    reach: Expr | number
    /** Where each direction starts looking. The pixel's own distance to the shape is the natural start. */
    start: Expr | number
    /** How close counts as touching the edge, about one device pixel in field units. */
    surfaceEps: Expr | number
    /** A per-pixel 0–1 value that turns the fan of directions so its structure becomes fine grain. */
    jitter: Expr | number
    /** How bright the edge is where a direction touches it, as a single number. */
    emission: (hit: {point: Expr; normal: Expr; distance: Expr; coverage: Expr}) => Expr
    /** Values from outside that `emission` also uses. List them here or they are out of scope. */
    deps?: Expr[]
    hint?: string
}

/**
 * The light arriving at a pixel from a glowing shape edge, with the shape's body blocking it.
 *
 * Looks out from the pixel in `cones` directions, finds where each touches the shape, and
 * averages the `emission` there. Edges facing away or hidden behind the body contribute
 * nothing. Returns one number: the brightness at this pixel. It runs per pixel per frame, so
 * prefer `irradianceField` unless the light must be live.
 *
 * @example
 * ```ts
 * const light = gatherIrradiance({origin: frame.sdfUV!, field: far.at, cones: 16, steps: 24, reach: 3, start: far.sdf, surfaceEps: px, jitter: hash, emission: ({normal}) => math.max(math.dot(normal, lightDir), 0), deps: [lightDir]})
 * ```
 * @tip Any value computed outside `emission` and read inside it must be listed in `deps`.
 * @see irradianceField, shadowVisibility
 */
export function gatherIrradiance(spec: GatherSpec): Expr {
    const id = radianceCounter++
    const hint = spec.hint ?? 'irradiance'
    // tan of the cone half-angle: the cone's half-width per unit distance.
    const coneWidth = Math.tan(PI / spec.cones)
    return new Expr((ctx) =>
        ctx.memo(`gather:${id}`, () => {
            for (const dep of spec.deps ?? []) dep._emit(ctx)
            const acc = ctx.freshLocal(hint)
            const o = ctx.freshLocal('coneOrigin')
            const eps = ctx.freshLocal('coneEps')
            const reach = ctx.freshLocal('coneReach')
            const start = ctx.freshLocal('coneStart')
            const jitter = ctx.freshLocal('coneJitter')
            const dir = ctx.freshLocal('coneDir')
            const t = ctx.freshLocal('coneT')
            const pos = ctx.freshLocal('conePos')
            const dd = ctx.freshLocal('coneD')
            const best = ctx.freshLocal('coneBest')
            const bestT = ctx.freshLocal('coneBestT')
            const bestD = ctx.freshLocal('coneBestD')
            const cover = ctx.freshLocal('coneCover')
            const i = ctx.freshLocal('cone')
            const s = ctx.freshLocal('step')
            const nrm = ctx.freshLocal('hitN')
            const hitP = ctx.freshLocal('hitP')
            const clear = ctx.freshLocal('coneClear')

            ctx.statement(`let ${o} = ${spec.origin._emit(ctx)};`)
            ctx.statement(`let ${eps} = ${num(spec.surfaceEps, ctx)};`)
            ctx.statement(`let ${reach} = ${num(spec.reach, ctx)};`)
            ctx.statement(`let ${start} = ${num(spec.start, ctx)};`)
            ctx.statement(`let ${jitter} = ${num(spec.jitter, ctx)};`)
            ctx.statement(`var ${acc} = 0.0;`)

            // The march: everything the field closure emits lands inside the inner loop.
            const marchStmts: string[] = []
            const fieldAtPos = spec.field(raw(pos))._emit(scopedInto(ctx, marchStmts))

            // The closest approach: outward normal by forward differences off the recorded
            // distance, the boundary point behind it, then the caller's emission — its hoists
            // land inside the coverage branch.
            const hitStmts: string[] = []
            const hitCtx = scopedInto(ctx, hitStmts)
            const fx = spec.field(raw(`(${pos} + vec2f(${eps}, 0.0))`))._emit(hitCtx)
            const fy = spec.field(raw(`(${pos} + vec2f(0.0, ${eps}))`))._emit(hitCtx)
            hitStmts.push(`let ${nrm} = normalize(vec2f(${fx} - ${bestD}, ${fy} - ${bestD}) + vec2f(0.0, 0.000001));`)
            hitStmts.push(`let ${hitP} = ${pos} - ${nrm} * max(${bestD}, 0.0);`)
            const emission = spec.emission({point: raw(hitP), normal: raw(nrm), distance: raw(bestT), coverage: raw(cover)})._emit(hitCtx)

            ctx.statement([
                `for (var ${i} = 0u; ${i} < ${spec.cones}u; ${i}++) {`,
                `  let ${dir} = vec2f(cos((f32(${i}) + ${jitter}) * ${formatFloat(TAU / spec.cones)}), sin((f32(${i}) + ${jitter}) * ${formatFloat(TAU / spec.cones)}));`,
                `  var ${t} = ${start};`,
                `  var ${pos} = ${o};`,
                `  var ${best} = 1.0;`,
                `  var ${bestT} = ${start};`,
                `  var ${bestD} = 0.0;`,
                `  for (var ${s} = 0u; ${s} < ${spec.steps}u; ${s}++) {`,
                `    ${pos} = ${o} + ${dir} * ${t};`,
                ...marchStmts.map((w) => `    ${w}`),
                `    let ${dd} = ${fieldAtPos};`,
                // How much of the cone's width (t · tan half-angle) the boundary leaves clear.
                `    let ${clear} = ${dd} / (${t} * ${formatFloat(coneWidth)});`,
                `    if (${clear} < ${best}) { ${best} = ${clear}; ${bestT} = ${t}; ${bestD} = ${dd}; }`,
                `    if (${dd} < ${eps}) { break; }`,
                `    ${t} += max(${dd}, ${eps});`,
                `    if (${t} > ${reach}) { break; }`,
                `  }`,
                `  let ${cover} = 1.0 - clamp(${best}, 0.0, 1.0);`,
                `  if (${cover} > 0.002) {`,
                `    ${pos} = ${o} + ${dir} * ${bestT};`,
                ...hitStmts.map((w) => `    ${w}`),
                `    ${acc} += ${cover} * (${emission});`,
                `  }`,
                `}`,
            ].join('\n'))
            return `(${acc} * ${formatFloat(1 / spec.cones)})`
        }),
    )
}

/** What `shadowVisibility` needs. */
export interface ShadowSpec {
    /** The point being lit, moved a couple of pixels off the shape's edge along its normal. */
    from: Expr
    /** Where the light is, in the same coordinates. */
    toward: Expr
    /** The shape as a signed distance: given a point, how far it is from the edge (negative inside). */
    field: (at: Expr) => Expr
    /** The most steps the ray may take. 12 is typical. */
    steps: number
    /** How close counts as hitting the shape, about one device pixel in field units. */
    surfaceEps: Expr | number
    /** How wide the soft edge of the shadow is, 0–1. 0 is a hard shadow. */
    softness?: Expr | number
    hint?: string
}

/**
 * How much of a light reaches a point, 1 when nothing is in the way and 0 when the shape blocks it.
 *
 * Casts a ray from `from` toward the light and stops at the shape. Values between 0 and 1
 * are the soft edge of the shadow, wider with `softness`. Multiply a light's contribution by
 * it.
 *
 * @example
 * ```ts
 * const lit = math.mul(strength, shadowVisibility({from: math.add(edge.point, math.mul(edge.normal, math.mul(px, 2))), toward: light.uv, field: far.at, steps: 12, surfaceEps: px, softness: u.shadowSoftness}))
 * ```
 * @tip Start `from` two pixels off the edge along its normal or the point shadows itself.
 * @see gatherIrradiance, irradianceField
 */
export function shadowVisibility(spec: ShadowSpec): Expr {
    const id = radianceCounter++
    const hint = spec.hint ?? 'visibility'
    return new Expr((ctx) =>
        ctx.memo(`shadow:${id}`, () => {
            const vis = ctx.freshLocal(hint)
            const from = ctx.freshLocal('shFrom')
            const seg = ctx.freshLocal('shSeg')
            const len = ctx.freshLocal('shLen')
            const dir = ctx.freshLocal('shDir')
            const eps = ctx.freshLocal('shEps')
            const soft = ctx.freshLocal('shSoft')
            const t = ctx.freshLocal('shT')
            const pos = ctx.freshLocal('shPos')
            const dd = ctx.freshLocal('shD')
            const k = ctx.freshLocal('shStep')

            ctx.statement(`let ${from} = ${spec.from._emit(ctx)};`)
            ctx.statement(`let ${seg} = ${spec.toward._emit(ctx)} - ${from};`)
            ctx.statement(`let ${len} = length(${seg});`)
            ctx.statement(`let ${dir} = ${seg} / max(${len}, 0.0001);`)
            ctx.statement(`let ${eps} = ${num(spec.surfaceEps, ctx)};`)
            ctx.statement(`let ${soft} = ${num(spec.softness ?? 0, ctx)};`)
            ctx.statement(`var ${vis} = 1.0;`)

            const marchStmts: string[] = []
            const fieldAtPos = spec.field(raw(pos))._emit(scopedInto(ctx, marchStmts))

            ctx.statement([
                `{`,
                `  var ${t} = ${eps} * 2.0;`,
                `  for (var ${k} = 0u; ${k} < ${spec.steps}u; ${k}++) {`,
                `    if (${t} >= ${len}) { break; }`,
                `    let ${pos} = ${from} + ${dir} * ${t};`,
                ...marchStmts.map((w) => `    ${w}`),
                `    let ${dd} = ${fieldAtPos};`,
                // Distance-field penumbra: how narrowly the ray misses, relative to how far it
                // has travelled (the classic soft-shadow estimate); a hit is full shadow.
                `    ${vis} = min(${vis}, clamp(${dd} / max(${t} * ${soft}, ${eps}), 0.0, 1.0));`,
                `    if (${vis} <= 0.0) { break; }`,
                `    ${t} += max(${dd}, ${eps});`,
                `  }`,
                `}`,
            ].join('\n'))
            return vis
        }),
    )
}

// ─── The compute half: irradiance gathered once into a texture ─────────────────────────────────
//
// The fragment gather above re-integrates the light field per device pixel per frame. Irradiance
// is a smooth field, so `irradianceField` gathers it in a compute pass at a modest fixed resolution
// (the DataMosh fixed-buffer precedent) — and ONLY when something feeding it changed (the
// CursorRipples idle-skip precedent): a static scene costs one texture sample per pixel. The gather
// is a stratified Monte-Carlo estimate with interleaved-gradient sample offsets, progressively
// accumulated while the scene is still, and denoised by a small Gaussian (the kit blur compute, at
// field resolution) — so the field is band-free and grain-free before the bilinear read. The shape
// arrives through the standard three field sources; 3D shapes ride the volumetric pre-march, which
// this node owns (a consumer spreads the spine and overrides `compute` with this one, as Glass
// does with its frosted prepass).

import type {GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {sdf, sdf3d} from '../../gpu/kit/index'
import {
    IrradianceParams, irradianceAnalyticLayout, irradianceSvgLayout, irradianceFieldLayout,
    analyticField, svgField, volumetricField, buildIrradianceKernel, createIrradiancePass, IRRADIANCE_FORMAT, MAX_LIGHTS,
    AccumulateParams, accumulateLayout, accumulateKernel, copyLayout, copyKernel,
    SilhouetteParams, silhouetteSeedLayout, silhouetteStepLayout, silhouetteSeedKernel, silhouetteStepKernel,
} from '../../gpu/scaffolds/radiance'
import {resolveShapeType} from '../../utilities/shapeEffectBounds'
import {createGaussianBlurCompute} from '../../gpu/kit/blur'

// Denoise: a Gaussian of this radius (field texels) over the Monte-Carlo gather — each output texel
// averages ~50 texels' independent ray sets.
const DENOISE_RADIUS = 3
const DENOISE_HALF_KERNEL = 6
// Frames after creation during which the gather always re-runs (startup settling).
const WARMUP_FRAMES = 3

const {createSdfDataTexture, analyticSubPropValues, parseShapeConfigValue} = sdf
const {createVolumetricFieldComputeNode} = sdf3d

/** The kit's `sdfSpaceUV` on the CPU: a stored-convention position → placement (sdfUV) space. */
function toPlacement(
    center: {x: number; y: number}, scale: number, rotationDeg: number, aspect: number,
    stored: {x: number; y: number},
): {x: number; y: number} {
    const dxAc = (stored.x - center.x) * aspect
    const dy = (1 - stored.y) - (1 - center.y)
    const r = rotationDeg * Math.PI / 180
    const c = Math.cos(r)
    const s = Math.sin(r)
    const sc = Math.max(scale, 0.001)
    return {x: (dxAc * c + dy * s) / sc + 0.5, y: (dy * c - dxAc * s) / sc + 0.5}
}

/** A resolved light list (`getCpuValue` of a list prop) → the fixed-capacity ABI lanes + count,
 *  positions mapped to placement space once here (not per hit per light on the GPU). */
function lightLanes(
    items: unknown,
    fields: {position: string; color: string; intensity: string},
    placement: {center: {x: number; y: number}; scale: number; rotation: number; aspect: number},
): {pos: [number, number, number, number][]; color: [number, number, number, number][]; count: number} {
    const list = (Array.isArray(items) ? items : []).slice(0, MAX_LIGHTS) as Record<string, unknown>[]
    const pos: [number, number, number, number][] = []
    const color: [number, number, number, number][] = []
    for (let i = 0; i < MAX_LIGHTS; i++) {
        const item = list[i]
        const stored = (item?.[fields.position] ?? {x: 0.5, y: 0.5}) as {x: number; y: number}
        const p = toPlacement(placement.center, placement.scale, placement.rotation, placement.aspect, stored)
        const c = (item?.[fields.color] ?? [1, 1, 1, 1]) as [number, number, number, number]
        const intensity = typeof item?.[fields.intensity] === 'number' ? (item[fields.intensity] as number) : 0
        pos.push(item ? [p.x, p.y, 0, 0] : [0, 0, 0, 0])
        color.push(item ? [c[0], c[1], c[2], intensity] : [0, 0, 0, 0])
    }
    return {pos, color, count: list.length}
}

/** What `irradianceField` needs. Every string is the name of one of your props. */
export interface IrradianceFieldSpec {
    /**
     * How many rays each texel casts per frame: `burst` on the first frame after something
     * changes, `motion` while changes keep coming, then `refine` for each of `refineFrames`
     * frames once the scene is still. After that nothing runs until the next change.
     */
    rays: {burst: number; motion: number; refine: number; refineFrames: number}
    /** The most steps a ray may take toward the shape (default 32). */
    steps?: number
    /** The most steps a shadow ray may take. 0 turns shadows off entirely (default 12). */
    shadowSteps?: number
    /** The size of the square light texture, in texels per side. 512 is plenty; it is a smooth field. */
    resolution: number
    /**
     * Your list prop of lights (made with `listPropConfig`) and the names of its item fields: a
     * `position`, a `color` and a `number` for intensity. A light's position may be driven by the mouse.
     */
    lights: {prop: string; position: string; color: string; intensity: string}
    /** The prop holding how far from the shape light is gathered, relative to the shape. */
    reach: string
    /** The prop holding how far light wraps around edges facing away from it, 0–1. */
    wrap: string
    /** The prop holding each light's range: the distance at which it falls to half strength. */
    lightRange: string
    /** The prop holding the shadow's soft-edge width, 0–1. */
    shadowSoftness: string
    /** The name the texture is published under for other compute passes (default 'irradianceTexture'). */
    outputKey?: string
    namePrefix?: string
}

/**
 * The light spilling from a shape's edges, gathered once into a texture and read per pixel.
 *
 * Returns two halves. Put `compute` on your definition's `compute:` field. In your material,
 * call `sample(params)` for the rgb light at the pixel. It returns `undefined` when there is no
 * GPU to gather on, so write `sample(params) ?? splat3(0)`. It reads your standard shape props
 * (`center`, `scale`, `rotation`, `shape`, `shapeSdfUrl`, `shapeType`) plus the props named in
 * the spec. The texture only re-gathers when a light or the shape changes.
 *
 * @example
 * ```ts
 * const spill = irradianceField(GATHER).sample(params) ?? math.splat3(0)
 * ```
 * @tip Build the spec once as a constant and use it for both `compute:` and `sample`.
 * @see gatherIrradiance, shadowVisibility
 */
export function irradianceField(spec: IrradianceFieldSpec): {
    compute: GpuComputeNode
    sample: (params: GpuFragmentParams) => Expr | undefined
} {
    const outputKey = spec.outputKey ?? 'irradianceTexture'
    const namePrefix = spec.namePrefix ?? 'irradiance'
    const kernelOpts = {steps: spec.steps ?? 32, shadowSteps: spec.shadowSteps ?? 12, namePrefix}
    const res = spec.resolution

    const compute: GpuComputeNode = (params) => {
        const {gpu, getCpuValue, registerComputeTexture, onCleanup, onResize} = params
        const root = gpu?.root
        if (!root) return null

        const shapeSdfUrl = (getCpuValue('shapeSdfUrl') as string) || ''
        const shapeType = resolveShapeType((getCpuValue('shapeType') as string) || '', getCpuValue('shape'))
        // Volumetric shapes: this node owns the pre-march (null for flat shapes).
        const vf = createVolumetricFieldComputeNode(params, () => getCpuValue('shape'), 'none', 'span')

        // Textures: this frame's gather → the running mean (write + its previous-frame copy) → denoised.
        const mkTex = () => {
            const t = root.createTexture({size: [res, res], format: IRRADIANCE_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => t.destroy())
            return t
        }
        const outTex = mkTex()
        const meanTex = mkTex()
        const prevTex = mkTex()
        const denoise = createGaussianBlurCompute(root, meanTex, res, res, onCleanup, DENOISE_HALF_KERNEL, res, res)
        denoise.updateRadius(DENOISE_RADIUS)
        const outputTexture = registerComputeTexture(denoise.outputTexture)
        const paramsU = root.createUniform(IrradianceParams)
        const accumU = root.createUniform(AccumulateParams)
        const accumulate = createIrradiancePass(root, accumulateKernel, {
            res, bindGroup: root.createBindGroup(accumulateLayout, {sample: outTex, prev: prevTex, out: meanTex, params: accumU.buffer}),
        })
        const keepMean = createIrradiancePass(root, copyKernel, {
            res, bindGroup: root.createBindGroup(copyLayout, {src: meanTex, out: prevTex}),
        })

        let pass
        let fieldVersion: () => number = () => 0
        // Volumetric shapes: the gather marches against the SILHOUETTE distance field, jump-flooded
        // from the marched field over its domain whenever the field re-marches (see the scaffold).
        let silhouette: {steps: (domain: {originX: number; originY: number; spanX: number; spanY: number; activeRes: number; rBound: number}) => GpuComputeStep[]} | null = null
        if (vf) {
            const jfaA = mkTex()
            const jfaB = mkTex()
            const silU = root.createUniform(SilhouetteParams)
            const seedPass = createIrradiancePass(root, silhouetteSeedKernel, {
                res, bindGroup: root.createBindGroup(silhouetteSeedLayout, {params: silU.buffer, fieldTex: vf.fieldTexture as never, out: jfaA}),
            })
            const stepAB = createIrradiancePass(root, silhouetteStepKernel, {res, bindGroup: root.createBindGroup(silhouetteStepLayout, {params: silU.buffer, src: jfaA, out: jfaB})})
            const stepBA = createIrradiancePass(root, silhouetteStepKernel, {res, bindGroup: root.createBindGroup(silhouetteStepLayout, {params: silU.buffer, src: jfaB, out: jfaA})})
            // Halving steps down to 1 texel; the final texture's parity follows from the pass count.
            const jumps: number[] = []
            for (let step = Math.ceil(res / 2); step >= 1; step = Math.floor(step / 2)) { jumps.push(step); if (step === 1) break }
            const finalTex = jumps.length % 2 === 1 ? jfaB : jfaA
            silhouette = {
                steps: (domain) => {
                    const nodes: GpuComputeStep[] = []
                    const base = {
                        vfOriginX: domain.originX, vfOriginY: domain.originY, vfSpanX: domain.spanX, vfSpanY: domain.spanY,
                        vfActiveRes: domain.activeRes, vfRBound: domain.rBound, res,
                    }
                    nodes.push(() => silU.write({...base, step: 0} as never), seedPass)
                    jumps.forEach((step, i) => {
                        nodes.push(() => silU.write({...base, step} as never), i % 2 === 0 ? stepAB : stepBA)
                    })
                    return nodes
                },
            }
            const samp = root.createSampler({magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'})
            const kernel = buildIrradianceKernel(irradianceFieldLayout, volumetricField, kernelOpts)
            const bg = root.createBindGroup(irradianceFieldLayout, {params: paramsU.buffer, outTex, silhouetteTex: finalTex, samp})
            pass = createIrradiancePass(root, kernel, {res, bindGroup: bg})
        } else if (shapeSdfUrl) {
            const sdfTex = createSdfDataTexture(params, shapeSdfUrl)
            // The SDF .bin loads asynchronously (a placeholder until it lands): its version must feed
            // the dirty key, or the gather idles on the placeholder until an unrelated edit.
            fieldVersion = () => sdfTex.getVersion()
            const samp = root.createSampler({magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'})
            const kernel = buildIrradianceKernel(irradianceSvgLayout, svgField, kernelOpts)
            const bg = root.createBindGroup(irradianceSvgLayout, {params: paramsU.buffer, outTex, sdfSource: sdfTex.texture.texture as never, samp})
            pass = createIrradiancePass(root, kernel, {res, bindGroup: bg})
        } else {
            const kernel = buildIrradianceKernel(irradianceAnalyticLayout, analyticField(shapeType), kernelOpts)
            const bg = root.createBindGroup(irradianceAnalyticLayout, {params: paramsU.buffer, outTex})
            pass = createIrradiancePass(root, kernel, {res, bindGroup: bg})
        }

        let width = Math.max(1, params.dimensions.width)
        let height = Math.max(1, params.dimensions.height)
        onResize(({width: w, height: h}) => { width = Math.max(1, w); height = Math.max(1, h) })

        const num = (key: string, fallback: number): number => {
            const v = getCpuValue(key)
            return typeof v === 'number' ? v : fallback
        }
        const pos = (key: string): {x: number; y: number} => {
            const v = getCpuValue(key) as {x?: number; y?: number} | undefined
            return {x: typeof v?.x === 'number' ? v.x : 0.5, y: typeof v?.y === 'number' ? v.y : 0.5}
        }

        let lastKey = ''
        let silhouetteReady = false
        let frame = 0
        // Re-gather unconditionally for the first few frames: inputs that settle asynchronously at
        // startup (canvas size, driver state, late-bound textures) would otherwise be frozen in.
        let warmup = WARMUP_FRAMES
        let samples = 0
        let changedLastFrame = false

        return {
            outputs: {[outputKey]: outputTexture, ...(vf?.outputs ?? {})},
            getComputeNodes: (frameParams: unknown) => {
                // The pre-march first (null when the shape is static), then read its domain.
                const pre = vf?.getComputeNodes(frameParams) ?? null
                let silhouettePasses: GpuComputeStep[] = []
                if (silhouette && (pre || !silhouetteReady)) {
                    const dom = vf!.getFieldDomain()
                    silhouettePasses = silhouette.steps(dom)
                    silhouetteReady = true
                }

                const center = pos('center')
                const scale = num('scale', 1)
                const rotation = num('rotation', 0)
                const lights = lightLanes(getCpuValue(spec.lights.prop), spec.lights, {center, scale, rotation, aspect: width / height})
                const sub = analyticSubPropValues(parseShapeConfigValue(getCpuValue('shape')))
                const dom = vf?.getFieldDomain()
                const values = {
                    center: [center.x, center.y] as [number, number],
                    lightPos: lights.pos,
                    lightColor: lights.color,
                    lightCount: lights.count,
                    scale,
                    rotation,
                    aspect: width / height,
                    reach: num(spec.reach, 3),
                    wrap: num(spec.wrap, 0),
                    lightRange: num(spec.lightRange, 1),
                    shadowSoftness: num(spec.shadowSoftness, 0),
                    // Hit threshold + minimum step at FIELD-texel scale (not device pixels): the field
                    // cannot resolve finer, and coarser steps keep grazing rays inside the step budget.
                    eps: (1 / res) / Math.max(scale, 0.001),
                    res,
                    saRadius: sub.radius, saSides: sub.sides, saRounding: sub.rounding, saInnerRatio: sub.innerRatio,
                    saRotation: sub.rotation, saHeight: sub.height, saOffset: sub.offset, saAperture: sub.aperture,
                    vfOriginX: dom?.originX ?? 0, vfOriginY: dom?.originY ?? 0,
                    vfSpanX: dom?.spanX ?? 1, vfSpanY: dom?.spanY ?? 1,
                    vfActiveRes: dom?.activeRes ?? 1, vfRBound: dom?.rBound ?? 0.6,
                    fieldVersion: fieldVersion(),
                }
                // A change (or a pre-march) restarts the progressive estimate with a burst of rays (a
                // full burst if the scene was still, the lighter motion budget if changes keep coming);
                // a still scene then refines with small gathers for `refineFrames`, then idles: the
                // textures already hold the converged answer.
                const key = JSON.stringify(values)
                const warming = warmup > 0
                if (warming) warmup--
                const changed = key !== lastKey || Boolean(pre) || silhouettePasses.length > 0 || warming
                let rays: number
                if (changed) {
                    lastKey = key
                    frame = 0
                    samples = 0
                    rays = changedLastFrame ? spec.rays.motion : spec.rays.burst
                } else if (frame > spec.rays.refineFrames) {
                    changedLastFrame = false
                    return null
                } else {
                    rays = spec.rays.refine
                }
                changedLastFrame = changed
                const sampleIndex = frame++
                // Sample-weighted running mean: this frame's estimate carries `rays` samples.
                const weight = rays / (samples + rays)
                samples += rays
                const nodes: GpuComputeStep[] = pre ? [...pre, ...silhouettePasses] : [...silhouettePasses]
                nodes.push(() => {
                    // `fieldVersion` is dirty-key-only, not an ABI field.
                    const {fieldVersion: _version, ...abi} = values
                    paramsU.write({
                        ...abi,
                        frame: sampleIndex,
                        rays,
                        center: d.vec2f(values.center[0], values.center[1]),
                        lightPos: values.lightPos.map((l) => d.vec4f(l[0], l[1], l[2], l[3])),
                        lightColor: values.lightColor.map((l) => d.vec4f(l[0], l[1], l[2], l[3])),
                    } as never)
                    accumU.write({weight} as never)
                })
                nodes.push(pass, accumulate, keepMean, ...denoise.computeSteps)
                return nodes
            },
        }
    }

    const sample = (params: GpuFragmentParams): Expr | undefined => {
        const tex = params.computeOutputs?.[outputKey] as KitTexture | undefined
        if (!tex) return undefined
        // Premultiplied by the outside mask (see the kernel): E = rgb / a, guarded inside the body.
        const s = local(tex.sampleLevel(params.ctx.uv), 'irrTex')
        return div(s.member('rgb'), max(s.member('a'), 0.001))
    }

    return {compute, sample}
}
