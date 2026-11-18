/**
 * 3D model (`mesh3D`) → the shape effects' volumetric field, by rasterizing — no SDF.
 *
 * Shape effects never read a 3D distance function: they read the volumetric field texture
 * (`sdf3d.ts`), per texel `.r` = distance to the silhouette outside / −chord/2 inside, `.g/.b` =
 * surface pattern coords, `.a` = view depth of the first hit. Analytic and SVG-extruded shapes
 * fill it by raymarching an SDF in compute. A model fills the SAME texture by drawing itself:
 *
 *   1. front pass — nearest surface (depth `less`): writes (t, pattern.u, pattern.v, 1)
 *   2. back pass  — farthest surface (depth `greater`): writes t
 *   3. jump flood — from the covered texels, the distance outside the silhouette (field UV)
 *   4. resolve    — inside: −(back − front)/2, pattern, front depth; outside: the flood distance
 *                   and the march's miss depth
 *
 * `t` is the march's ray parameter (view z + 1.2, the camera plane at z = −1.2), so a model's field
 * is indistinguishable from a marched one to every consumer. The chord is first entry → last exit
 * (the march's 'span' mode); 'firstLobe' would need depth peeling and is treated as 'span'.
 *
 * Smooth shading: consumers derive normals from the depth channel, so a flat-triangle depth would
 * show every facet. The fragment depth is Phong-tessellated instead — each pixel's point is pulled
 * toward the tangent planes of its triangle's three vertex normals — which curves the surface
 * between vertices while authored hard edges (split normals) stay hard. Models without normals
 * get crease-aware smooth normals on load (60°).
 *
 * The passes re-run only when the shape state changes (rotation, size, domain, resolution, load),
 * like the march. Nothing is stored besides the .glb: everything is derived on load.
 */
import type {TgpuRoot} from 'typegpu'
import {parseGlb, type MeshGeometry} from '../../utilities/glb'
import {MESH3D_DEFAULTS, MESH3D_MAX_SIZE} from '../../utilities/meshShape'
import {resolveShapeSubProp, type ShapeSubPropFrame} from '../../utilities/sdf3d'
import {rawTimestampWrites, type GpuPassTimer} from '../gpuTimer'
import {authorError} from '../support'

export type MeshPatternMode = 'none' | 'raw' | 'triplanar'

// ─── Load + prepare (CPU, once per URL) ─────────────────────────────────────────────────────

/** A model ready for the GPU: centred, scaled to a unit bounding sphere, z flipped into the
 *  field's frame (so the model's +Z front faces the camera), with one packed normal per corner. */
export interface PreparedMesh {
    /** xyz per vertex, |p| ≤ 1. */
    positions: Float32Array
    /** Three vertex indices per triangle. */
    indices: Uint32Array
    /** One octahedral-packed unit normal (2× snorm16) per triangle corner. */
    cornerNormals: Uint32Array
    /** Half extents of the (unit-sphere) bounding box. */
    halfExtents: [number, number, number]
    triangles: number
}

const CREASE_COS = Math.cos(60 * Math.PI / 180)

function octEncode(x: number, y: number, z: number): number {
    const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1
    let u = x / l1
    let v = y / l1
    if (z < 0) {
        const ou = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1)
        const ov = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1)
        u = ou
        v = ov
    }
    const pu = Math.round(Math.max(-1, Math.min(1, u)) * 32767) & 0xffff
    const pv = Math.round(Math.max(-1, Math.min(1, v)) * 32767) & 0xffff
    return (pu | (pv << 16)) >>> 0
}

/** Weld vertices that share a position (models split vertices at UV seams and hard edges). */
function weldVertices(pos: Float32Array, eps: number): Int32Array {
    const n = pos.length / 3
    let cap = 1
    while (cap < n * 2) cap <<= 1
    const mask = cap - 1
    const table = new Int32Array(cap).fill(-1)
    const q = new Int32Array(n * 3)
    const weld = new Int32Array(n)
    const inv = 1 / eps
    for (let v = 0; v < n; v++) {
        const qx = Math.round(pos[v * 3] * inv)
        const qy = Math.round(pos[v * 3 + 1] * inv)
        const qz = Math.round(pos[v * 3 + 2] * inv)
        q[v * 3] = qx; q[v * 3 + 1] = qy; q[v * 3 + 2] = qz
        let h = (Math.imul(qx, 73856093) ^ Math.imul(qy, 19349663) ^ Math.imul(qz, 83492791)) & mask
        for (;;) {
            const rep = table[h]
            if (rep < 0) { table[h] = v; weld[v] = v; break }
            if (q[rep * 3] === qx && q[rep * 3 + 1] === qy && q[rep * 3 + 2] === qz) { weld[v] = rep; break }
            h = (h + 1) & mask
        }
    }
    return weld
}

/** Centre, normalise, flip z, and pack one normal per corner (authored, else crease-smoothed). */
export function prepareMesh(geometry: MeshGeometry): PreparedMesh {
    const src = geometry.positions
    const n = src.length / 3
    const tris = geometry.indices.length / 3

    const min = [Infinity, Infinity, Infinity]
    const max = [-Infinity, -Infinity, -Infinity]
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const v = src[i * 3 + c]
            if (v < min[c]) min[c] = v
            if (v > max[c]) max[c] = v
        }
    }
    const cx = (min[0] + max[0]) / 2, cy = (min[1] + max[1]) / 2, cz = (min[2] + max[2]) / 2
    let r2 = 0
    for (let i = 0; i < n; i++) {
        const dx = src[i * 3] - cx, dy = src[i * 3 + 1] - cy, dz = src[i * 3 + 2] - cz
        const d2 = dx * dx + dy * dy + dz * dz
        if (d2 > r2) r2 = d2
    }
    const s = r2 > 0 ? 1 / Math.sqrt(r2) : 1

    const positions = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
        positions[i * 3] = (src[i * 3] - cx) * s
        positions[i * 3 + 1] = (src[i * 3 + 1] - cy) * s
        positions[i * 3 + 2] = -(src[i * 3 + 2] - cz) * s
    }
    const indices = geometry.indices

    // Face normals (area-weighted: the raw cross product) in the prepared frame.
    const face = new Float32Array(tris * 3)
    const faceUnit = new Float32Array(tris * 3)
    for (let t = 0; t < tris; t++) {
        const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3
        const e1x = positions[b] - positions[a], e1y = positions[b + 1] - positions[a + 1], e1z = positions[b + 2] - positions[a + 2]
        const e2x = positions[c] - positions[a], e2y = positions[c + 1] - positions[a + 1], e2z = positions[c + 2] - positions[a + 2]
        const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x
        face[t * 3] = nx; face[t * 3 + 1] = ny; face[t * 3 + 2] = nz
        const len = Math.hypot(nx, ny, nz) || 1
        faceUnit[t * 3] = nx / len; faceUnit[t * 3 + 1] = ny / len; faceUnit[t * 3 + 2] = nz / len
    }

    const authored = geometry.normals
    let missing = false
    for (let i = 0; i < n && !missing; i++) if (Number.isNaN(authored[i * 3])) missing = true

    // Incident triangles per welded vertex (CSR), only when some normals must be generated.
    let weld: Int32Array | null = null
    let start: Int32Array | null = null
    let list: Int32Array | null = null
    if (missing) {
        weld = weldVertices(positions, 1e-6)
        const count = new Int32Array(n + 1)
        for (let i = 0; i < indices.length; i++) count[weld[indices[i]] + 1]++
        start = new Int32Array(n + 1)
        for (let v = 0; v < n; v++) start[v + 1] = start[v] + count[v + 1]
        const fill = start.slice(0, n)
        list = new Int32Array(indices.length)
        for (let i = 0; i < indices.length; i++) list[fill[weld[indices[i]]]++] = (i / 3) | 0
    }

    const cornerNormals = new Uint32Array(tris * 3)
    for (let t = 0; t < tris; t++) {
        const fx = faceUnit[t * 3], fy = faceUnit[t * 3 + 1], fz = faceUnit[t * 3 + 2]
        for (let k = 0; k < 3; k++) {
            const v = indices[t * 3 + k]
            const ax = authored[v * 3]
            if (!Number.isNaN(ax)) {
                cornerNormals[t * 3 + k] = octEncode(ax, authored[v * 3 + 1], -authored[v * 3 + 2])
                continue
            }
            // Crease-aware smooth normal: sum the area-weighted normals of the faces around this
            // position that are within 60° of this face (winding-agnostic: flip opposed faces).
            let sx = 0, sy = 0, sz = 0
            const w = weld![v]
            for (let j = start![w]; j < start![w + 1]; j++) {
                const u = list![j]
                const d = fx * faceUnit[u * 3] + fy * faceUnit[u * 3 + 1] + fz * faceUnit[u * 3 + 2]
                if (Math.abs(d) < CREASE_COS) continue
                const sign = d < 0 ? -1 : 1
                sx += face[u * 3] * sign; sy += face[u * 3 + 1] * sign; sz += face[u * 3 + 2] * sign
            }
            const len = Math.hypot(sx, sy, sz)
            cornerNormals[t * 3 + k] = len > 1e-20 ? octEncode(sx / len, sy / len, sz / len) : octEncode(fx, fy, fz || 1)
        }
    }

    return {
        positions,
        indices,
        cornerNormals,
        halfExtents: [(max[0] - min[0]) / 2 * s, (max[1] - min[1]) / 2 * s, (max[2] - min[2]) / 2 * s],
        triangles: tris,
    }
}

const meshCache = new Map<string, Promise<PreparedMesh | null>>()
const MESH_CACHE_MAX = 8
const reportedFailures = new Set<string>()

/** Fetch, parse and prepare a model, once per URL. Resolves null when it can't be loaded. */
export function loadPreparedMesh(url: string): Promise<PreparedMesh | null> {
    let p = meshCache.get(url)
    if (!p) {
        p = (async () => {
            try {
                const response = await fetch(url, {mode: 'cors'})
                if (!response.ok) throw new Error(`HTTP ${response.status}`)
                return prepareMesh(parseGlb(await response.arrayBuffer()))
            } catch (e) {
                meshCache.delete(url) // let a later mount retry (network blips)
                if (!reportedFailures.has(url)) {
                    reportedFailures.add(url)
                    console.warn(`[shaders] Couldn't load the 3D model ${url}: ${e instanceof Error ? e.message : String(e)}`)
                }
                return null
            }
        })()
        if (meshCache.size >= MESH_CACHE_MAX) {
            const oldest = meshCache.keys().next().value
            if (oldest !== undefined) meshCache.delete(oldest)
        }
        meshCache.set(url, p)
    }
    return p
}

// ─── GPU programs (raw WGSL, one set per device) ────────────────────────────────────────────

/** View-from-local rotation rows (w of row 0 = size), the field domain, depth range + curvature. */
const RASTER_UNIFORM_BYTES = 80

/** The march's ray parameter spans t ∈ [1.2 − size, 1.2 + size] ⊂ (0, DEPTH_RANGE). */
const DEPTH_RANGE = 3
/** Phong tessellation blend (0 = flat triangles, 1 = full); 0.75 is the usual choice. */
const CURVATURE = 0.75

const patternWgsl = (mode: MeshPatternMode) => {
    if (mode === 'raw') return 'let pattern = vec2f(v.local.x + v.local.z * 0.7, v.local.y + v.local.z * 0.5);'
    if (mode === 'triplanar') {
        // The march's per-face axis pick, from the local face normal.
        return `let faceN = cross(dpdx(v.local), dpdy(v.local));
  let an = abs(faceN);
  let hp = v.local;
  let b1 = vec2f(hp.y + 2.17, hp.z + 5.31);
  let b2 = vec2f(hp.z + 7.73, hp.x + 1.13);
  let b3 = vec2f(hp.x, hp.y);
  let b23 = select(b3, b2, an.y >= an.z);
  let inner = select(b23, b1, an.x >= an.z);
  let pattern = select(b23, inner, an.x >= an.y);`
    }
    return 'let pattern = vec2f(0.0, 0.0);'
}

const RASTER = (mode: MeshPatternMode) => /* wgsl */ `
struct Raster {
  row0: vec4f,
  row1: vec4f,
  row2: vec4f,
  domain: vec4f,
  misc: vec4f,
}
@group(0) @binding(0) var<uniform> R: Raster;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> normals: array<u32>;

struct V {
  @builtin(position) clip: vec4f,
  @location(0) bary: vec3f,
  @location(1) @interpolate(flat) p0: vec3f,
  @location(2) @interpolate(flat) p1: vec3f,
  @location(3) @interpolate(flat) p2: vec3f,
  @location(4) @interpolate(flat) n0: vec3f,
  @location(5) @interpolate(flat) n1: vec3f,
  @location(6) @interpolate(flat) n2: vec3f,
  @location(7) local: vec3f,
}

fn vertexAt(i: u32) -> vec3f {
  return vec3f(positions[3u * i], positions[3u * i + 1u], positions[3u * i + 2u]) * R.row0.w;
}

fn toView(p: vec3f) -> vec3f {
  return vec3f(dot(R.row0.xyz, p), dot(R.row1.xyz, p), dot(R.row2.xyz, p));
}

fn octDecode(e: u32) -> vec3f {
  let f = unpack2x16snorm(e);
  var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  let t = max(-n.z, 0.0);
  n.x = n.x + select(t, -t, n.x >= 0.0);
  n.y = n.y + select(t, -t, n.y >= 0.0);
  return normalize(n);
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
  let tri = vi / 3u;
  let corner = vi - tri * 3u;
  let l0 = vertexAt(indices[3u * tri]);
  let l1 = vertexAt(indices[3u * tri + 1u]);
  let l2 = vertexAt(indices[3u * tri + 2u]);
  let a = toView(l0);
  let b = toView(l1);
  let c = toView(l2);
  var p = a;
  var l = l0;
  var bary = vec3f(1.0, 0.0, 0.0);
  if (corner == 1u) { p = b; l = l1; bary = vec3f(0.0, 1.0, 0.0); }
  if (corner == 2u) { p = c; l = l2; bary = vec3f(0.0, 0.0, 1.0); }
  // View → field UV (y down) → the active block's clip space; depth = the march's t.
  let fu = p.x + 0.5;
  let fv = 0.5 - p.y;
  let t = p.z + 1.2;
  var o: V;
  o.clip = vec4f((fu - R.domain.x) / R.domain.z * 2.0 - 1.0, 1.0 - (fv - R.domain.y) / R.domain.w * 2.0, t / R.misc.x, 1.0);
  o.bary = bary;
  o.p0 = a;
  o.p1 = b;
  o.p2 = c;
  o.n0 = toView(octDecode(normals[3u * tri]));
  o.n1 = toView(octDecode(normals[3u * tri + 1u]));
  o.n2 = toView(octDecode(normals[3u * tri + 2u]));
  o.local = l;
  return o;
}

// Phong tessellation: the flat point pulled toward each vertex's tangent plane, blended by the
// barycentrics. Continuous across shared edges (only the edge's two vertices contribute there).
fn surfaceT(bary: vec3f, p0: vec3f, p1: vec3f, p2: vec3f, n0: vec3f, n1: vec3f, n2: vec3f) -> f32 {
  let p = p0 * bary.x + p1 * bary.y + p2 * bary.z;
  let q0 = p - dot(p - p0, n0) * n0;
  let q1 = p - dot(p - p1, n1) * n1;
  let q2 = p - dot(p - p2, n2) * n2;
  let q = q0 * bary.x + q1 * bary.y + q2 * bary.z;
  return mix(p.z, q.z, R.misc.y) + 1.2;
}

struct Front {
  @location(0) value: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment fn fsFront(v: V) -> Front {
  ${patternWgsl(mode)}
  let t = surfaceT(v.bary, v.p0, v.p1, v.p2, v.n0, v.n1, v.n2);
  var o: Front;
  o.value = vec4f(t, pattern.x, pattern.y, 1.0);
  o.depth = clamp(t / R.misc.x, 0.0, 1.0);
  return o;
}

struct Back {
  @location(0) value: f32,
  @builtin(frag_depth) depth: f32,
}

@fragment fn fsBack(v: V) -> Back {
  let t = surfaceT(v.bary, v.p0, v.p1, v.p2, v.n0, v.n1, v.n2);
  var o: Back;
  o.value = t;
  o.depth = clamp(t / R.misc.x, 0.0, 1.0);
  return o;
}
`

const PARAMS_WGSL = /* wgsl */ `
struct Params {
  res: u32,
  step: i32,
  texel: vec2f,
  origin: vec2f,
  rBound: f32,
  pad: f32,
}
const NONE: u32 = 0xffffffffu;
fn seedDistance(p: vec2i, s: u32, texel: vec2f) -> f32 {
  let q = vec2i(i32(s & 0xffffu), i32(s >> 16u));
  return length(vec2f(p - q) * texel);
}
`
const PARAMS_BYTES = 32

const SEED = /* wgsl */ `${PARAMS_WGSL}
@group(0) @binding(0) var front: texture_2d<f32>;
@group(0) @binding(1) var seeds: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> P: Params;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.res || id.y >= P.res) { return; }
  let p = vec2i(id.xy);
  let covered = textureLoad(front, p, 0).w > 0.5;
  textureStore(seeds, p, vec4u(select(NONE, id.x | (id.y << 16u), covered), 0u, 0u, 0u));
}
`

const FLOOD = /* wgsl */ `${PARAMS_WGSL}
@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> P: Params;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.res || id.y >= P.res) { return; }
  let p = vec2i(id.xy);
  let res = i32(P.res);
  var best = textureLoad(src, p, 0).x;
  var bestD = 1e9;
  if (best != NONE) { bestD = seedDistance(p, best, P.texel); }
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = p + vec2i(dx, dy) * P.step;
      if (q.x < 0 || q.y < 0 || q.x >= res || q.y >= res) { continue; }
      let s = textureLoad(src, q, 0).x;
      if (s == NONE) { continue; }
      let d = seedDistance(p, s, P.texel);
      if (d < bestD) {
        bestD = d;
        best = s;
      }
    }
  }
  textureStore(dst, p, vec4u(best, 0u, 0u, 0u));
}
`

const RESOLVE = /* wgsl */ `${PARAMS_WGSL}
@group(0) @binding(0) var front: texture_2d<f32>;
@group(0) @binding(1) var back: texture_2d<f32>;
@group(0) @binding(2) var seeds: texture_2d<u32>;
@group(0) @binding(3) var field: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> P: Params;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.res || id.y >= P.res) { return; }
  let p = vec2i(id.xy);
  let f = textureLoad(front, p, 0);
  if (f.w > 0.5) {
    let tBack = textureLoad(back, p, 0).x;
    textureStore(field, p, vec4f(-max(tBack - f.x, 0.0) * 0.5, f.y, f.z, f.x));
    return;
  }
  // Outside: distance to the nearest covered texel, less half a texel (the silhouette runs
  // between texel centres). No coverage at all (not loaded yet) reads as far outside.
  let s = textureLoad(seeds, p, 0).x;
  var dist = 10.0;
  if (s != NONE) { dist = max(seedDistance(p, s, P.texel) - 0.5 * min(P.texel.x, P.texel.y), 0.0); }
  // The march's miss depth: just past where the ray leaves the bounding sphere.
  let rel = P.origin + (vec2f(p) + 0.5) * P.texel - vec2f(0.5, 0.5);
  let viewDepth = 1.2 + sqrt(max(P.rBound * P.rBound - dot(rel, rel), 0.0)) + 0.41;
  textureStore(field, p, vec4f(dist, 0.0, 0.0, viewDepth));
}
`

interface MeshPrograms {
    rasterLayout: GPUBindGroupLayout
    front: GPURenderPipeline
    back: GPURenderPipeline
    seed: GPUComputePipeline
    flood: GPUComputePipeline
    resolve: GPUComputePipeline
}

const programCache = new WeakMap<GPUDevice, Map<MeshPatternMode, MeshPrograms>>()

function programsFor(device: GPUDevice, mode: MeshPatternMode): MeshPrograms {
    let byMode = programCache.get(device)
    if (!byMode) programCache.set(device, (byMode = new Map()))
    let p = byMode.get(mode)
    if (p) return p

    device.pushErrorScope('validation')
    const rasterModule = device.createShaderModule({code: RASTER(mode), label: `mesh-field-raster-${mode}`})
    const rasterLayout = device.createBindGroupLayout({
        label: 'mesh-field-raster',
        entries: [
            {binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {type: 'uniform'}},
            {binding: 1, visibility: GPUShaderStage.VERTEX, buffer: {type: 'read-only-storage'}},
            {binding: 2, visibility: GPUShaderStage.VERTEX, buffer: {type: 'read-only-storage'}},
            {binding: 3, visibility: GPUShaderStage.VERTEX, buffer: {type: 'read-only-storage'}},
        ],
    })
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [rasterLayout]})
    const raster = (entryPoint: 'fsFront' | 'fsBack', format: GPUTextureFormat, depthCompare: GPUCompareFunction) =>
        device.createRenderPipeline({
            label: `mesh-field-${entryPoint}`,
            layout: pipelineLayout,
            vertex: {module: rasterModule, entryPoint: 'vs'},
            fragment: {module: rasterModule, entryPoint, targets: [{format}]},
            // Models aren't reliably closed or consistently wound: draw both sides.
            primitive: {topology: 'triangle-list', cullMode: 'none'},
            depthStencil: {format: 'depth32float', depthWriteEnabled: true, depthCompare},
        })
    const compute = (code: string, label: string) => device.createComputePipeline({
        label: `mesh-field-${label}`,
        layout: 'auto',
        compute: {module: device.createShaderModule({code, label: `mesh-field-${label}`}), entryPoint: 'main'},
    })
    p = {
        rasterLayout,
        front: raster('fsFront', 'rgba32float', 'less'),
        back: raster('fsBack', 'r32float', 'greater'),
        seed: compute(SEED, 'seed'),
        flood: compute(FLOOD, 'flood'),
        resolve: compute(RESOLVE, 'resolve'),
    }
    void device.popErrorScope().then((e) => e && authorError(`[shaders] 3D model field: ${e.message.split('\n')[0]}`))
    byMode.set(mode, p)
    return p
}

// ─── The producer ───────────────────────────────────────────────────────────────────────────

/** The per-frame params the producer's `update` reads (deltaTime → auto clock, pointer → mouse). */
export interface MeshFrameParams {
    deltaTime?: number
    pointer?: {x: number; y: number}
}

export interface MeshFieldProducer {
    /** Resolve the shape config (auto-animate / mouse sub-props) for this frame. */
    update(frame?: MeshFrameParams): void
    /** The aspect-fit field domain + bounding radius for the sampler's `_vf*` uniforms. */
    getFootprint(): {spanX: number; spanY: number; originX: number; originY: number; rBound: number}
    /** Changes whenever the field needs redrawing (rotation, size, domain, model load). */
    getStateKey(): string
    /** Encode the passes that fill the field's top-left `res × res` block. */
    encode(res: number, timer?: GpuPassTimer): void
}

const DEG = Math.PI / 180

function rotateCpu(vx: number, vy: number, vz: number, cx: number, sx: number, cy: number, sy: number, cz: number, sz: number) {
    // Mirror of sdf3d's rotateVec3 (Y, then X, then Z).
    const x1 = vx * cy + vz * sy
    const z1 = -vx * sy + vz * cy
    const y2 = vy * cx - z1 * sx
    const z2 = vy * sx + z1 * cx
    return [x1 * cz - y2 * sz, x1 * sz + y2 * cz, z2] as const
}

/**
 * Create the GPU resources that draw `meshUrl` into `fieldTexture` (the volumetric field,
 * rgba32float storage + sampled, `maxRes²`). The model loads in the background; until then the
 * field reads as empty (all outside). `onCleanup` releases everything with the node.
 */
export function createMeshFieldProducer(opts: {
    root: TgpuRoot
    fieldTexture: GPUTexture
    maxRes: number
    meshUrl: string
    getShapeConfig: () => unknown
    patternMode: MeshPatternMode
    onCleanup: (cb: () => void) => void
}): MeshFieldProducer {
    const {root, fieldTexture, maxRes, meshUrl, getShapeConfig, patternMode, onCleanup} = opts
    const device = root.device
    const programs = programsFor(device, patternMode)
    const U = GPUTextureUsage

    const front = device.createTexture({label: 'mesh-field-front', size: [maxRes, maxRes], format: 'rgba32float', usage: U.RENDER_ATTACHMENT | U.TEXTURE_BINDING})
    const back = device.createTexture({label: 'mesh-field-back', size: [maxRes, maxRes], format: 'r32float', usage: U.RENDER_ATTACHMENT | U.TEXTURE_BINDING})
    const depth = device.createTexture({label: 'mesh-field-depth', size: [maxRes, maxRes], format: 'depth32float', usage: U.RENDER_ATTACHMENT})
    const seedA = device.createTexture({label: 'mesh-field-seeds-a', size: [maxRes, maxRes], format: 'r32uint', usage: U.TEXTURE_BINDING | U.STORAGE_BINDING})
    const seedB = device.createTexture({label: 'mesh-field-seeds-b', size: [maxRes, maxRes], format: 'r32uint', usage: U.TEXTURE_BINDING | U.STORAGE_BINDING})
    const rasterUniform = device.createBuffer({label: 'mesh-field-raster', size: RASTER_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
    // One params buffer per flood step (each pass in the submit reads its own); [0] = seed/resolve.
    let maxSteps = 1
    for (let s = 1; s < maxRes; s <<= 1) maxSteps++
    const paramBuffers = Array.from({length: maxSteps + 2}, () =>
        device.createBuffer({label: 'mesh-field-params', size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}))

    const views = {
        front: front.createView(), back: back.createView(), depth: depth.createView(),
        seedA: seedA.createView(), seedB: seedB.createView(), field: fieldTexture.createView(),
    }
    const bind = (pipeline: GPUComputePipeline, resources: GPUBindingResource[]) =>
        device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({binding, resource}))})
    const seedGroup = bind(programs.seed, [views.front, views.seedA, {buffer: paramBuffers[0]}])
    // Flood ping-pong: step i reads A→B when i is even, B→A when odd.
    const floodGroups = paramBuffers.slice(1).map((buffer, i) =>
        bind(programs.flood, [i % 2 === 0 ? views.seedA : views.seedB, i % 2 === 0 ? views.seedB : views.seedA, {buffer}]))
    const resolveGroups = [views.seedA, views.seedB].map((seeds) =>
        bind(programs.resolve, [views.front, views.back, seeds, views.field, {buffer: paramBuffers[0]}]))

    let mesh: {group: GPUBindGroup; vertices: number; buffers: GPUBuffer[]; halfExtents: [number, number, number]} | null = null
    let meshVersion = 0
    let destroyed = false

    onCleanup(() => {
        destroyed = true
        for (const t of [front, back, depth, seedA, seedB]) t.destroy()
        rasterUniform.destroy()
        for (const b of paramBuffers) b.destroy()
        for (const b of mesh?.buffers ?? []) b.destroy()
        mesh = null
    })

    void loadPreparedMesh(meshUrl).then((prepared) => {
        if (!prepared || destroyed) return
        const upload = (data: Float32Array | Uint32Array, label: string) => {
            const buffer = device.createBuffer({label, size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST})
            device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength)
            return buffer
        }
        const buffers = [
            upload(prepared.positions, 'mesh-positions'),
            upload(prepared.indices, 'mesh-indices'),
            upload(prepared.cornerNormals, 'mesh-normals'),
        ]
        const group = device.createBindGroup({
            layout: programs.rasterLayout,
            entries: [
                {binding: 0, resource: {buffer: rasterUniform}},
                {binding: 1, resource: {buffer: buffers[0]}},
                {binding: 2, resource: {buffer: buffers[1]}},
                {binding: 3, resource: {buffer: buffers[2]}},
            ],
        })
        mesh = {group, vertices: prepared.triangles * 3, buffers, halfExtents: prepared.halfExtents}
        meshVersion++
    })

    // ── CPU shape state ──
    let elapsed = 0
    let lastShapeJson = ''
    let lastCfg: Record<string, unknown> = {}
    const springs = new Map<string, {current: number; velocity: number}>()
    let size = MESH3D_DEFAULTS.size
    let rows: [readonly number[], readonly number[], readonly number[]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    let rotKey = ''
    let spanX = 1, spanY = 1, originX = 0, originY = 0, rBound = 0.6

    const update = (frameParams?: MeshFrameParams) => {
        const deltaTime = frameParams?.deltaTime ?? 0
        elapsed += deltaTime
        const raw = getShapeConfig()
        if (raw && typeof raw === 'object') lastCfg = raw as Record<string, unknown>
        else if (typeof raw === 'string' && raw !== lastShapeJson) {
            lastShapeJson = raw
            try { lastCfg = JSON.parse(raw) as Record<string, unknown> } catch { /* keep previous */ }
        }
        const frame: ShapeSubPropFrame = {
            elapsed, deltaTime,
            pointerX: frameParams?.pointer?.x ?? 0.5,
            pointerY: frameParams?.pointer?.y ?? 0.5,
            springs,
        }
        const sub = (k: string) => resolveShapeSubProp(lastCfg[k], MESH3D_DEFAULTS[k] ?? 0, frame, k)
        size = Math.min(MESH3D_MAX_SIZE, Math.max(0.02, sub('size')))
        const rx = sub('rotX') * DEG, ry = sub('rotY') * DEG, rz = sub('rotZ') * DEG
        const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz)
        // view = Rᵀ·local, so view row i = R·eᵢ (the march computes local = R·view).
        rows = [rotateCpu(1, 0, 0, cx, sx, cy, sy, cz, sz), rotateCpu(0, 1, 0, cx, sx, cy, sy, cz, sz), rotateCpu(0, 0, 1, cx, sx, cy, sy, cz, sz)]
        rotKey = `${cx},${sx},${cy},${sy},${cz},${sz}`

        // Aspect-fit domain: the rotated bounding box (the whole bounding sphere until loaded).
        const [hx, hy, hz] = mesh?.halfExtents ?? [1, 1, 1]
        const ext = (r: readonly number[]) => (Math.abs(r[0]) * hx + Math.abs(r[1]) * hy + Math.abs(r[2]) * hz) * size
        const rPadX = ext(rows[0]) + 0.03 + 0.05
        const rPadY = ext(rows[1]) + 0.03 + 0.05
        spanX = rPadX * 2; spanY = rPadY * 2
        originX = 0.5 - rPadX; originY = 0.5 - rPadY
        rBound = size + 0.02
    }
    update()

    const encode = (res: number, timer?: GpuPassTimer) => {
        if (destroyed) return
        const r = rows
        device.queue.writeBuffer(rasterUniform, 0, new Float32Array([
            r[0][0], r[0][1], r[0][2], size,
            r[1][0], r[1][1], r[1][2], 0,
            r[2][0], r[2][1], r[2][2], 0,
            originX, originY, spanX, spanY,
            DEPTH_RANGE, CURVATURE, 0, 0,
        ]))

        // Flood steps: the largest power of two below res, halving to 1, then one more 1 (JFA+1).
        const steps: number[] = []
        let s = 1
        while (s * 2 < res) s *= 2
        for (; s >= 1; s >>= 1) steps.push(s)
        steps.push(1)
        const texelU = spanX / res, texelV = spanY / res
        const writeParams = (buffer: GPUBuffer, step: number) => {
            const data = new ArrayBuffer(PARAMS_BYTES)
            new Uint32Array(data, 0, 1)[0] = res
            new Int32Array(data, 4, 1)[0] = step
            new Float32Array(data, 8, 5).set([texelU, texelV, originX, originY, rBound])
            device.queue.writeBuffer(buffer, 0, data)
        }
        writeParams(paramBuffers[0], 0)
        steps.forEach((step, i) => writeParams(paramBuffers[i + 1], step))

        const enc = device.createCommandEncoder({label: 'mesh-field'})
        const surfacePass = (target: GPUTextureView, pipeline: GPURenderPipeline, depthClear: number, label: string) => {
            const pass = enc.beginRenderPass({
                label,
                colorAttachments: [{view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0]}],
                depthStencilAttachment: {view: views.depth, depthLoadOp: 'clear', depthClearValue: depthClear, depthStoreOp: 'discard'},
                timestampWrites: rawTimestampWrites(root, timer, label) as GPURenderPassTimestampWrites | undefined,
            })
            if (mesh) {
                pass.setViewport(0, 0, res, res, 0, 1)
                pass.setScissorRect(0, 0, res, res)
                pass.setPipeline(pipeline)
                pass.setBindGroup(0, mesh.group)
                pass.draw(mesh.vertices)
            }
            pass.end()
        }
        surfacePass(views.front, programs.front, 1, 'mesh-field-front')
        surfacePass(views.back, programs.back, 0, 'mesh-field-back')

        const groups = Math.ceil(res / 8)
        const pass = enc.beginComputePass({label: 'mesh-field-flood', timestampWrites: rawTimestampWrites(root, timer, 'mesh-field-flood')})
        pass.setPipeline(programs.seed)
        pass.setBindGroup(0, seedGroup)
        pass.dispatchWorkgroups(groups, groups)
        pass.setPipeline(programs.flood)
        steps.forEach((_, i) => {
            pass.setBindGroup(0, floodGroups[i])
            pass.dispatchWorkgroups(groups, groups)
        })
        // After an odd number of steps the result sits in B, after an even number in A.
        pass.setPipeline(programs.resolve)
        pass.setBindGroup(0, resolveGroups[steps.length % 2 === 1 ? 1 : 0])
        pass.dispatchWorkgroups(groups, groups)
        pass.end()
        device.queue.submit([enc.finish()])
    }

    return {
        update,
        getFootprint: () => ({spanX, spanY, originX, originY, rBound}),
        getStateKey: () => `${meshVersion},${size},${rotKey},${spanX},${spanY}`,
        encode,
    }
}
