/**
 * Real-time SVG → signed distance field, generated on the GPU from the SVG itself.
 *
 * No `.bin` round trip: the browser rasterises the SVG (full SVG fidelity — fill rules, strokes,
 * masks, text, transforms — exactly as the old CPU generator did), the canvas is copied straight
 * to the GPU (`copyExternalImageToTexture`, no CPU readback), and four compute stages build the
 * field:
 *
 *   1. seed     — every boundary texel stores the sub-pixel point where the anti-aliased edge
 *                 crosses it (Gustavson & Strand's edtaa estimate from coverage + edge direction,
 *                 correct on diagonals where the old `|0.5 − α|` seed wobbled) — anti-aliased
 *                 texels only, hard-edge texels only where no neighbour is anti-aliased;
 *   2. flood    — jump flooding propagates each texel's nearest edge point (log₂ size passes + one
 *                 refinement pass), in a compact rg32float seed buffer;
 *   3. resolve  — distance to the edge's tangent line at that point (point distance at corners),
 *                 signed by coverage, in field units (texels / size);
 *   4. finalize — the shape effects' texture: `.r` distance, `.g/.b` the forward-difference slope
 *                 the gradient sampler bakes (EPS 0.01 in field UV, unchanged from the CPU bake),
 *                 `.a` 1. The same stage upsamples a legacy `.bin` when the SVG can't be used, so
 *                 fallback is just another input.
 *
 * Measured (spike, Apple GPU, warm): 2048² in ~2.5 ms GPU + ~2 ms browser raster. Against an exact
 * circle: 0.02 texel mean distance error, 1.4° rms normal error (the `.bin` method: 0.07 / 4.3°),
 * at ~4× the old 512² resolution. Conventions match the old generator:
 * letterboxed with 5% padding, uniform scale, centred; outside positive; y-down texel rows.
 */
// Maintainer notes: written against the raw GPUDevice (the TypeGPU root's) because these are
// one-shot setup passes over textures the root owns — no per-frame graph, no bind-group layouts
// shared with shaders. The output is a TypeGPU texture (sampled + storage), so every consumer
// binds it exactly like the old data texture. The temporaries are destroyed once the work is done.
import type {TgpuRoot} from 'typegpu'

/** Field padding around the SVG, as a fraction of the field (matches the old generator). */
const PADDING = 0.05
/** Forward-difference step of the baked slope, in field UV (the effects' FD EPS). */
const SLOPE_EPS_UV = 0.01
const WG = 8

const FILL = /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  textureStore(dst, vec2i(id.xy), vec4f(0.5, 0.0, 0.0, 1.0)); // outside: invisible until ready
}`

const SEED = /* wgsl */ `
@group(0) @binding(0) var coverage: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rg32float, write>;
fn alphaAt(p: vec2i, size: vec2i) -> f32 { return textureLoad(coverage, clamp(p, vec2i(0), size - 1), 0).a; }
// Distance from a pixel centre to an anti-aliased edge, from its coverage and the edge direction.
fn edgeDistance(gIn: vec2f, a: f32) -> f32 {
  if (gIn.x == 0.0 || gIn.y == 0.0) { return 0.5 - a; }
  var g = abs(normalize(gIn));
  if (g.x < g.y) { g = g.yx; }
  let a1 = 0.5 * g.y / g.x;
  if (a < a1) { return 0.5 * (g.x + g.y) - sqrt(2.0 * g.x * g.y * a); }
  if (a < 1.0 - a1) { return (0.5 - a) * g.x; }
  return -0.5 * (g.x + g.y) + sqrt(2.0 * g.x * g.y * (1.0 - a));
}
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  let p = vec2i(id.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let a = alphaAt(p, size);
  let k = 1.41421356;
  let gx = -alphaAt(p + vec2i(-1, -1), size) - k * alphaAt(p + vec2i(-1, 0), size) - alphaAt(p + vec2i(-1, 1), size)
          + alphaAt(p + vec2i(1, -1), size) + k * alphaAt(p + vec2i(1, 0), size) + alphaAt(p + vec2i(1, 1), size);
  let gy = -alphaAt(p + vec2i(-1, -1), size) - k * alphaAt(p + vec2i(0, -1), size) - alphaAt(p + vec2i(1, -1), size)
          + alphaAt(p + vec2i(-1, 1), size) + k * alphaAt(p + vec2i(0, 1), size) + alphaAt(p + vec2i(1, 1), size);
  // Seed only texels the anti-aliased edge actually crosses (0 < α < 1): their coverage pins the
  // edge to a fraction of a texel. A fully covered or empty texel next to the 0.5 crossing says
  // nothing about where the edge is (edtaa would put it half a texel off, scattered irregularly
  // along every curve — the bands glass magnified); those seed only on a HARD edge, where no
  // neighbour is anti-aliased and the boundary lies between the texel centres.
  let inside = a >= 0.5;
  let aR = alphaAt(p + vec2i(1, 0), size);
  let aL = alphaAt(p + vec2i(-1, 0), size);
  let aD = alphaAt(p + vec2i(0, 1), size);
  let aU = alphaAt(p + vec2i(0, -1), size);
  let crosses = (aR >= 0.5) != inside || (aL >= 0.5) != inside || (aD >= 0.5) != inside || (aU >= 0.5) != inside;
  let partial = a > 0.0 && a < 1.0;
  let neighbourPartial = (aR > 0.0 && aR < 1.0) || (aL > 0.0 && aL < 1.0) || (aD > 0.0 && aD < 1.0) || (aU > 0.0 && aU < 1.0);
  var seed = vec2f(-1e9); // no edge here
  if (partial || (crosses && !neighbourPartial)) {
    let g = vec2f(gx, gy);
    let centre = vec2f(p) + 0.5;
    seed = centre;
    if (length(g) > 1e-6) { seed = centre + normalize(g) * edgeDistance(g, clamp(a, 0.0, 1.0)); }
  }
  textureStore(dst, p, vec4f(seed, 0.0, 0.0));
}`

const FLOOD = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> step: vec4i;
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  let p = vec2i(id.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let centre = vec2f(p) + 0.5;
  var best = textureLoad(src, p, 0).xy;
  var bestD = distance(centre, best);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let q = p + vec2i(i, j) * step.x;
      if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) { continue; }
      let s = textureLoad(src, q, 0).xy;
      let d = distance(centre, s);
      if (d < bestD) { bestD = d; best = s; }
    }
  }
  textureStore(dst, p, vec4f(best, 0.0, 0.0));
}`

// Distance to the edge's TANGENT LINE at the nearest seed, not to the seed point. The seeds are a
// dotted line along the edge (one point per edge texel), so point distance overestimates between
// dots and the iso-lines scallop at the dot spacing — uneven slopes along every curve. The edge
// normal at the seed (coverage gradient, recomputed here from the texel the point lies in) makes the
// field exact on smooth curves; where the nearest seed is well off to the side (a convex corner's
// fan), it blends back to point distance, which is right there.
const RESOLVE = /* wgsl */ `
@group(0) @binding(0) var seeds: texture_2d<f32>;
@group(0) @binding(1) var coverage: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
fn alphaAt(p: vec2i, size: vec2i) -> f32 { return textureLoad(coverage, clamp(p, vec2i(0), size - 1), 0).a; }
fn edgeNormal(q: vec2i, size: vec2i) -> vec2f {
  let k = 1.41421356;
  let gx = -alphaAt(q + vec2i(-1, -1), size) - k * alphaAt(q + vec2i(-1, 0), size) - alphaAt(q + vec2i(-1, 1), size)
          + alphaAt(q + vec2i(1, -1), size) + k * alphaAt(q + vec2i(1, 0), size) + alphaAt(q + vec2i(1, 1), size);
  let gy = -alphaAt(q + vec2i(-1, -1), size) - k * alphaAt(q + vec2i(0, -1), size) - alphaAt(q + vec2i(1, -1), size)
          + alphaAt(q + vec2i(-1, 1), size) + k * alphaAt(q + vec2i(0, 1), size) + alphaAt(q + vec2i(1, 1), size);
  return vec2f(gx, gy);
}
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  let p = vec2i(id.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let seed = textureLoad(seeds, p, 0).xy;
  let v = vec2f(p) + 0.5 - seed;
  let dPoint = length(v);
  var dTexels = dPoint;
  let g = edgeNormal(vec2i(floor(seed)), size);
  if (seed.x > -1e8 && length(g) > 1e-6) {
    let dLine = abs(dot(v, normalize(g)));
    let side = sqrt(max(dPoint * dPoint - dLine * dLine, 0.0));
    dTexels = mix(dLine, dPoint, smoothstep(1.0, 2.5, side));
  }
  let d = dTexels / f32(size.x);
  let inside = alphaAt(p, size) >= 0.5;
  textureStore(dst, p, vec4f(select(d, -d, inside), 0.0, 0.0, 1.0));
}`

// Bilinear reads of any-size distance field (clamp-to-edge, texel centres at integer + 0.5), so the
// same stage finalises a generated field (same size) and upsamples a legacy 512² `.bin`.
const FINALIZE = /* wgsl */ `
@group(0) @binding(0) var dist: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
fn at(uv: vec2f) -> f32 {
  let size = vec2i(textureDimensions(dist));
  let t = uv * vec2f(size) - 0.5;
  let i = vec2i(floor(t));
  let f = t - floor(t);
  let lo = vec2i(0);
  let hi = size - 1;
  let a = textureLoad(dist, clamp(i, lo, hi), 0).x;
  let b = textureLoad(dist, clamp(i + vec2i(1, 0), lo, hi), 0).x;
  let c = textureLoad(dist, clamp(i + vec2i(0, 1), lo, hi), 0).x;
  let d = textureLoad(dist, clamp(i + vec2i(1, 1), lo, hi), 0).x;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let f0 = at(uv);
  let gx = (at(uv + vec2f(${SLOPE_EPS_UV}, 0.0)) - f0) / ${SLOPE_EPS_UV};
  let gy = (at(uv + vec2f(0.0, ${SLOPE_EPS_UV})) - f0) / ${SLOPE_EPS_UV};
  textureStore(dst, vec2i(id.xy), vec4f(f0, gx, gy, 1.0));
}`

interface Pipelines {fill: GPUComputePipeline; seed: GPUComputePipeline; flood: GPUComputePipeline; resolve: GPUComputePipeline; finalize: GPUComputePipeline}
const pipelineCache = new WeakMap<GPUDevice, Pipelines>()
function pipelinesFor(device: GPUDevice): Pipelines {
    let p = pipelineCache.get(device)
    if (!p) {
        const make = (code: string, label: string) => device.createComputePipeline({
            label: `svg-sdf-${label}`, layout: 'auto', compute: {module: device.createShaderModule({code, label: `svg-sdf-${label}`}), entryPoint: 'main'},
        })
        p = {fill: make(FILL, 'fill'), seed: make(SEED, 'seed'), flood: make(FLOOD, 'flood'), resolve: make(RESOLVE, 'resolve'), finalize: make(FINALIZE, 'finalize')}
        pipelineCache.set(device, p)
    }
    return p
}

/** A generated (or fallback-filled) SVG field texture and when it last changed. */
export interface SvgSdfField {
    /** The TypeGPU texture (`rgba16float`, sampled + storage): `.r` distance, `.g/.b` slope, `.a` 1. */
    texture: unknown
    /** The field's resolution (square). */
    size: number
    /** Bumps each time the texture's content changes (placeholder → field). */
    getVersion: () => number
    /** Resolves with how the field was filled: from the SVG, from the fallback `.bin`, or not at all. */
    ready: Promise<'svg' | 'bin' | 'none'>
    destroy: () => void
}

/** Rasterise an SVG URL into a letterboxed square canvas (the old generator's placement). */
async function rasterise(svgUrl: string, size: number): Promise<OffscreenCanvas | HTMLCanvasElement> {
    if (typeof Image === 'undefined') throw new Error('no DOM image decoder')
    const response = await fetch(svgUrl, {mode: 'cors'})
    if (!response.ok) throw new Error(`SVG fetch failed: HTTP ${response.status}`)
    const objectUrl = URL.createObjectURL(await response.blob())
    try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
            const el = new Image()
            el.onload = () => resolve(el)
            el.onerror = () => reject(new Error('SVG failed to decode'))
            el.src = objectUrl
        })
        const canvas: OffscreenCanvas | HTMLCanvasElement = typeof OffscreenCanvas !== 'undefined'
            ? new OffscreenCanvas(size, size)
            : Object.assign(document.createElement('canvas'), {width: size, height: size})
        // `willReadFrequently` keeps this canvas on the CPU rasteriser, which anti-aliases with true
        // fractional coverage. A GPU-accelerated canvas multisamples instead: ~4 coverage levels
        // (0 / ½ / ¾ / 1), so the edge estimate steps by up to half a texel in a staircase along
        // every straight edge — on an extrusion each step lands in its own screen rows (streaks).
        const ctx = canvas.getContext('2d', {willReadFrequently: true}) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null
        if (!ctx) throw new Error('no 2D canvas context')
        const pad = size * PADDING
        const content = size - 2 * pad
        const w0 = img.naturalWidth || size
        const h0 = img.naturalHeight || size
        const s = content / Math.max(w0, h0)
        ctx.clearRect(0, 0, size, size)
        ctx.drawImage(img, (size - w0 * s) / 2, (size - h0 * s) / 2, w0 * s, h0 * s)
        return canvas
    } finally {
        URL.revokeObjectURL(objectUrl)
    }
}

/**
 * Build an SVG's signed distance field on the GPU, falling back to a legacy `.bin` if the SVG
 * can't be fetched or decoded. Returns at once with a placeholder (all outside); `getVersion`
 * bumps when the field lands, so consumers that cache work on the field (a 3D re-march) refresh.
 */
export function generateSvgSdfField(
    root: TgpuRoot,
    sources: {svgUrl?: string; sdfUrl?: string},
    size: number,
    loadBin: (url: string, into: Float32Array) => Promise<void>,
    binSize: number,
): SvgSdfField {
    const device = root.device
    const pipes = pipelinesFor(device)
    const texture = (root as unknown as {createTexture: (o: object) => {$usage: (...u: string[]) => unknown}})
        .createTexture({size: [size, size], format: 'rgba16float'})
        .$usage('sampled', 'storage')
    const gpuTexture = (root as unknown as {unwrap: (t: unknown) => GPUTexture}).unwrap(texture)
    let version = 0
    let destroyed = false
    const groups = Math.ceil(size / WG)

    const bind = (pipeline: GPUComputePipeline, resources: GPUBindingResource[]) =>
        device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({binding, resource}))})
    const submit = (record: (pass: GPUComputePassEncoder) => void) => {
        const enc = device.createCommandEncoder({label: 'svg-sdf'})
        const pass = enc.beginComputePass({label: 'svg-sdf'})
        record(pass)
        pass.end()
        device.queue.submit([enc.finish()])
    }

    // Placeholder: all outside, so nothing draws until the field is ready.
    submit((pass) => {
        pass.setPipeline(pipes.fill)
        pass.setBindGroup(0, bind(pipes.fill, [gpuTexture.createView()]))
        pass.dispatchWorkgroups(groups, groups)
    })

    const fromSvg = async (svgUrl: string) => {
        const canvas = await rasterise(svgUrl, size)
        if (destroyed) return
        const U = GPUTextureUsage
        const coverage = device.createTexture({label: 'svg-sdf-coverage', size: [size, size], format: 'rgba8unorm', usage: U.TEXTURE_BINDING | U.COPY_DST | U.RENDER_ATTACHMENT})
        const seedA = device.createTexture({label: 'svg-sdf-seeds-a', size: [size, size], format: 'rg32float', usage: U.TEXTURE_BINDING | U.STORAGE_BINDING})
        const seedB = device.createTexture({label: 'svg-sdf-seeds-b', size: [size, size], format: 'rg32float', usage: U.TEXTURE_BINDING | U.STORAGE_BINDING})
        const dist = device.createTexture({label: 'svg-sdf-distance', size: [size, size], format: 'r32float', usage: U.TEXTURE_BINDING | U.STORAGE_BINDING})
        device.queue.copyExternalImageToTexture({source: canvas}, {texture: coverage}, [size, size])
        const steps: number[] = []
        for (let s = size >> 1; s >= 1; s >>= 1) steps.push(s)
        steps.push(1) // one refinement pass (JFA+1)
        const stepBuffers = steps.map((s) => {
            const b = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
            device.queue.writeBuffer(b, 0, new Int32Array([s, 0, 0, 0]))
            return b
        })
        submit((pass) => {
            pass.setPipeline(pipes.seed)
            pass.setBindGroup(0, bind(pipes.seed, [coverage.createView(), seedA.createView()]))
            pass.dispatchWorkgroups(groups, groups)
            let src = seedA
            let dst = seedB
            pass.setPipeline(pipes.flood)
            stepBuffers.forEach((buffer) => {
                pass.setBindGroup(0, bind(pipes.flood, [src.createView(), dst.createView(), {buffer}]))
                pass.dispatchWorkgroups(groups, groups)
                const t = src
                src = dst
                dst = t
            })
            pass.setPipeline(pipes.resolve)
            pass.setBindGroup(0, bind(pipes.resolve, [src.createView(), coverage.createView(), dist.createView()]))
            pass.dispatchWorkgroups(groups, groups)
            pass.setPipeline(pipes.finalize)
            pass.setBindGroup(0, bind(pipes.finalize, [dist.createView(), gpuTexture.createView()]))
            pass.dispatchWorkgroups(groups, groups)
        })
        await device.queue.onSubmittedWorkDone()
        for (const t of [coverage, seedA, seedB, dist]) t.destroy()
        for (const b of stepBuffers) b.destroy()
    }

    const fromBin = async (sdfUrl: string) => {
        const data = new Float32Array(binSize * binSize).fill(0.5)
        await loadBin(sdfUrl, data)
        if (destroyed) return
        const dist = device.createTexture({label: 'svg-sdf-bin', size: [binSize, binSize], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST})
        device.queue.writeTexture({texture: dist}, data, {bytesPerRow: binSize * 4}, [binSize, binSize])
        submit((pass) => {
            pass.setPipeline(pipes.finalize)
            pass.setBindGroup(0, bind(pipes.finalize, [dist.createView(), gpuTexture.createView()]))
            pass.dispatchWorkgroups(groups, groups)
        })
        await device.queue.onSubmittedWorkDone()
        dist.destroy()
    }

    const ready = (async (): Promise<'svg' | 'bin' | 'none'> => {
        if (sources.svgUrl) {
            try {
                await fromSvg(sources.svgUrl)
                if (destroyed) return 'none'
                version++
                return 'svg'
            } catch {
                // Unreachable or undecodable SVG: fall through to the legacy field.
            }
        }
        if (sources.sdfUrl) {
            try {
                await fromBin(sources.sdfUrl)
                if (destroyed) return 'none'
                version++
                return 'bin'
            } catch {
                // Nothing usable: the placeholder (all outside) stays.
            }
        }
        return 'none'
    })()

    return {
        texture,
        size,
        getVersion: () => version,
        ready,
        destroy: () => {
            destroyed = true
            gpuTexture.destroy()
        },
    }
}
