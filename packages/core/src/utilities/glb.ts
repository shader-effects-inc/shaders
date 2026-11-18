// GLB (binary glTF 2.0) geometry reader and writer — no dependencies.
//
// A 3D model shape (`mesh3D`) only needs geometry: the shader is the material. `parseGlb` walks
// the default scene, applies every node's transform, and merges all triangle primitives into one
// position / normal / index set. Materials, textures, skins, morph targets and animation are
// ignored. `writeGlb` writes that geometry back out as a minimal GLB (the editor's upload step
// uses it to strip a model down to what the engine reads).
//
// Not supported: Draco (`KHR_draco_mesh_compression`) and meshopt (`EXT_meshopt_compression`)
// compressed geometry, and buffers stored in external files. These throw a `MeshParseError` whose
// message says what to do instead.

/** The merged triangle geometry of a model, in the model's own units and axes (glTF: +Y up, +Z front). */
export interface MeshGeometry {
    /** xyz per vertex. */
    positions: Float32Array
    /** Unit xyz per vertex. NaN where the file had no normal for that vertex. */
    normals: Float32Array
    /** Three vertex indices per triangle. */
    indices: Uint32Array
}

/** The most triangles a model shape accepts (keeps a phone's vertex work and memory bounded). */
export const MAX_MESH_TRIANGLES = 2_000_000

export type MeshParseErrorCode =
    | 'NOT_GLB'
    | 'UNSUPPORTED_VERSION'
    | 'COMPRESSED'
    | 'EXTERNAL_BUFFER'
    | 'NO_TRIANGLES'
    | 'TOO_MANY_TRIANGLES'
    | 'MALFORMED'

export class MeshParseError extends Error {
    constructor(message: string, public code: MeshParseErrorCode) {
        super(message)
        this.name = 'MeshParseError'
    }
}

const GLB_MAGIC = 0x46546c67 // 'glTF'
const CHUNK_JSON = 0x4e4f534a // 'JSON'
const CHUNK_BIN = 0x004e4942 // 'BIN\0'

const COMPRESSION_EXTENSIONS: Record<string, string> = {
    KHR_draco_mesh_compression: 'Draco',
    EXT_meshopt_compression: 'meshopt',
    KHR_meshopt_compression: 'meshopt',
}

interface GltfAccessor {
    bufferView?: number
    byteOffset?: number
    componentType: number
    normalized?: boolean
    count: number
    type: string
    sparse?: {
        count: number
        indices: {bufferView: number; byteOffset?: number; componentType: number}
        values: {bufferView: number; byteOffset?: number}
    }
}

interface GltfBufferView {
    buffer: number
    byteOffset?: number
    byteLength: number
    byteStride?: number
    extensions?: Record<string, unknown>
}

interface GltfPrimitive {
    attributes: Record<string, number>
    indices?: number
    mode?: number
    extensions?: Record<string, unknown>
}

interface GltfNode {
    mesh?: number
    children?: number[]
    matrix?: number[]
    translation?: number[]
    rotation?: number[]
    scale?: number[]
}

interface Gltf {
    asset?: {version?: string}
    extensionsUsed?: string[]
    extensionsRequired?: string[]
    buffers?: {byteLength: number; uri?: string}[]
    bufferViews?: GltfBufferView[]
    accessors?: GltfAccessor[]
    meshes?: {primitives: GltfPrimitive[]}[]
    nodes?: GltfNode[]
    scenes?: {nodes?: number[]}[]
    scene?: number
}

const TYPE_SIZE: Record<string, number> = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16}
const COMPONENT_BYTES: Record<number, number> = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}

/** True when the bytes start with the GLB magic (`glTF`). */
export function isGlb(buffer: ArrayBuffer): boolean {
    return buffer.byteLength >= 12 && new DataView(buffer).getUint32(0, true) === GLB_MAGIC
}

function decodeDataUri(uri: string): Uint8Array {
    const comma = uri.indexOf(',')
    if (!uri.startsWith('data:') || comma < 0 || !uri.slice(0, comma).endsWith(';base64')) {
        throw new MeshParseError('This model keeps its geometry in a separate file. Export it as a single .glb.', 'EXTERNAL_BUFFER')
    }
    const bin = atob(uri.slice(comma + 1))
    const out = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
    return out
}

function readComponent(view: DataView, offset: number, componentType: number, normalized: boolean): number {
    switch (componentType) {
        case 5126: return view.getFloat32(offset, true)
        case 5120: { const v = view.getInt8(offset); return normalized ? Math.max(v / 127, -1) : v }
        case 5121: { const v = view.getUint8(offset); return normalized ? v / 255 : v }
        case 5122: { const v = view.getInt16(offset, true); return normalized ? Math.max(v / 32767, -1) : v }
        case 5123: { const v = view.getUint16(offset, true); return normalized ? v / 65535 : v }
        case 5125: return view.getUint32(offset, true)
        default: throw new MeshParseError(`Unknown accessor component type ${componentType}.`, 'MALFORMED')
    }
}

/**
 * Parse a GLB into merged triangle geometry. Throws `MeshParseError` for anything it can't read
 * (not a GLB, compressed geometry, external buffers, no triangles, over `MAX_MESH_TRIANGLES`).
 */
export function parseGlb(buffer: ArrayBuffer): MeshGeometry {
    if (!isGlb(buffer)) throw new MeshParseError('Not a GLB file. Export the model as binary glTF (.glb).', 'NOT_GLB')
    const header = new DataView(buffer)
    const version = header.getUint32(4, true)
    if (version !== 2) throw new MeshParseError(`GLB version ${version} isn't supported. Export as glTF 2.0.`, 'UNSUPPORTED_VERSION')
    const total = Math.min(header.getUint32(8, true), buffer.byteLength)

    let json: Gltf | null = null
    let bin: Uint8Array | null = null
    let offset = 12
    while (offset + 8 <= total) {
        const length = header.getUint32(offset, true)
        const type = header.getUint32(offset + 4, true)
        const start = offset + 8
        if (start + length > total) throw new MeshParseError('The GLB file is truncated.', 'MALFORMED')
        if (type === CHUNK_JSON && !json) {
            try {
                json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, length))) as Gltf
            } catch {
                throw new MeshParseError('The GLB file has an unreadable header.', 'MALFORMED')
            }
        } else if (type === CHUNK_BIN && !bin) {
            bin = new Uint8Array(buffer, start, length)
        }
        offset = start + length + ((4 - (length % 4)) % 4)
    }
    if (!json) throw new MeshParseError('The GLB file has no glTF header.', 'MALFORMED')
    const gltf = json

    for (const ext of [...(gltf.extensionsRequired ?? []), ...(gltf.extensionsUsed ?? [])]) {
        const name = COMPRESSION_EXTENSIONS[ext]
        if (name) {
            throw new MeshParseError(`This model uses ${name} compression, which isn't supported. Export it again without mesh compression.`, 'COMPRESSED')
        }
    }

    const buffers: Uint8Array[] = (gltf.buffers ?? []).map((b, i) => {
        if (b.uri === undefined) {
            if (i !== 0 || !bin) throw new MeshParseError('The GLB file is missing its binary data.', 'MALFORMED')
            return bin
        }
        return decodeDataUri(b.uri)
    })

    const viewBytes = (index: number): {bytes: Uint8Array; stride: number} => {
        const bv = gltf.bufferViews?.[index]
        if (!bv) throw new MeshParseError(`Missing buffer view ${index}.`, 'MALFORMED')
        const buf = buffers[bv.buffer]
        if (!buf) throw new MeshParseError(`Missing buffer ${bv.buffer}.`, 'MALFORMED')
        const start = bv.byteOffset ?? 0
        if (start + bv.byteLength > buf.byteLength) throw new MeshParseError('A buffer view runs past its buffer.', 'MALFORMED')
        return {bytes: buf.subarray(start, start + bv.byteLength), stride: bv.byteStride ?? 0}
    }

    /** Read an accessor as floats (normalized ints → 0–1 / −1–1), `size` components per element. */
    const readFloats = (index: number): {data: Float32Array; size: number} => {
        const acc = gltf.accessors?.[index]
        if (!acc) throw new MeshParseError(`Missing accessor ${index}.`, 'MALFORMED')
        const size = TYPE_SIZE[acc.type]
        const compBytes = COMPONENT_BYTES[acc.componentType]
        if (!size || !compBytes) throw new MeshParseError(`Unsupported accessor type ${acc.type}.`, 'MALFORMED')
        const data = new Float32Array(acc.count * size)
        const normalized = acc.normalized === true
        if (acc.bufferView !== undefined) {
            const {bytes, stride} = viewBytes(acc.bufferView)
            const elemBytes = size * compBytes
            const step = stride || elemBytes
            const base = acc.byteOffset ?? 0
            if (acc.count > 0 && base + step * (acc.count - 1) + elemBytes > bytes.byteLength) {
                throw new MeshParseError('An accessor runs past its buffer view.', 'MALFORMED')
            }
            const absolute = bytes.byteOffset + base
            if (acc.componentType === 5126 && step === elemBytes && absolute % 4 === 0) {
                data.set(new Float32Array(bytes.buffer, absolute, acc.count * size))
            } else {
                const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
                for (let i = 0; i < acc.count; i++) {
                    const o = base + i * step
                    for (let c = 0; c < size; c++) data[i * size + c] = readComponent(view, o + c * compBytes, acc.componentType, normalized)
                }
            }
        }
        if (acc.sparse && acc.sparse.count > 0) {
            const s = acc.sparse
            const idx = viewBytes(s.indices.bufferView)
            const val = viewBytes(s.values.bufferView)
            const idxView = new DataView(idx.bytes.buffer, idx.bytes.byteOffset, idx.bytes.byteLength)
            const valView = new DataView(val.bytes.buffer, val.bytes.byteOffset, val.bytes.byteLength)
            const idxBytes = COMPONENT_BYTES[s.indices.componentType] ?? 4
            for (let i = 0; i < s.count; i++) {
                const target = readComponent(idxView, (s.indices.byteOffset ?? 0) + i * idxBytes, s.indices.componentType, false)
                for (let c = 0; c < size; c++) {
                    data[target * size + c] = readComponent(valView, (s.values.byteOffset ?? 0) + (i * size + c) * compBytes, acc.componentType, normalized)
                }
            }
        }
        return {data, size}
    }

    const readIndices = (index: number): Uint32Array => {
        const acc = gltf.accessors?.[index]
        if (!acc || acc.bufferView === undefined) throw new MeshParseError(`Missing index accessor ${index}.`, 'MALFORMED')
        const compBytes = COMPONENT_BYTES[acc.componentType]
        if (acc.componentType !== 5121 && acc.componentType !== 5123 && acc.componentType !== 5125) {
            throw new MeshParseError('Unsupported index type.', 'MALFORMED')
        }
        const {bytes, stride} = viewBytes(acc.bufferView)
        const step = stride || compBytes
        const base = acc.byteOffset ?? 0
        if (acc.count > 0 && base + step * (acc.count - 1) + compBytes > bytes.byteLength) {
            throw new MeshParseError('An index accessor runs past its buffer view.', 'MALFORMED')
        }
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
        const out = new Uint32Array(acc.count)
        for (let i = 0; i < acc.count; i++) out[i] = readComponent(view, base + i * step, acc.componentType, false)
        return out
    }

    // ── Gather (mesh, world matrix) instances from the scene graph ──────────────────────────
    const nodes = gltf.nodes ?? []
    const instances: {mesh: number; matrix: Float64Array}[] = []
    const visit = (nodeIndex: number, parent: Float64Array, depth: number) => {
        const node = nodes[nodeIndex]
        if (!node || depth > 64) return
        const world = mat4Multiply(parent, nodeMatrix(node))
        if (node.mesh !== undefined) instances.push({mesh: node.mesh, matrix: world})
        for (const child of node.children ?? []) visit(child, world, depth + 1)
    }
    const scene = gltf.scenes?.[gltf.scene ?? 0]
    if (scene?.nodes?.length) {
        for (const root of scene.nodes) visit(root, mat4Identity(), 0)
    } else if (nodes.length) {
        // No scene: treat every node nobody parents as a root.
        const childSet = new Set(nodes.flatMap((n) => n.children ?? []))
        nodes.forEach((_, i) => { if (!childSet.has(i)) visit(i, mat4Identity(), 0) })
    } else {
        // No nodes at all: every mesh, untransformed.
        (gltf.meshes ?? []).forEach((_, i) => instances.push({mesh: i, matrix: mat4Identity()}))
    }

    // ── Merge triangle primitives ────────────────────────────────────────────────────────────
    const posChunks: Float32Array[] = []
    const nrmChunks: Float32Array[] = []
    const idxChunks: Uint32Array[] = []
    let vertexCount = 0
    let triangleCount = 0

    for (const inst of instances) {
        const mesh = gltf.meshes?.[inst.mesh]
        if (!mesh) continue
        const m = inst.matrix
        const nm = normalMatrix(m)
        for (const prim of mesh.primitives ?? []) {
            const mode = prim.mode ?? 4
            if (mode !== 4 && mode !== 5 && mode !== 6) continue // points / lines: no surface
            for (const ext of Object.keys(prim.extensions ?? {})) {
                if (COMPRESSION_EXTENSIONS[ext]) {
                    throw new MeshParseError(`This model uses ${COMPRESSION_EXTENSIONS[ext]} compression, which isn't supported. Export it again without mesh compression.`, 'COMPRESSED')
                }
            }
            const posAcc = prim.attributes?.POSITION
            if (posAcc === undefined) continue
            const pos = readFloats(posAcc)
            if (pos.size !== 3) continue
            const count = pos.data.length / 3
            const nrm = prim.attributes.NORMAL !== undefined ? readFloats(prim.attributes.NORMAL) : null

            const outPos = new Float32Array(count * 3)
            const outNrm = new Float32Array(count * 3)
            for (let i = 0; i < count; i++) {
                const x = pos.data[i * 3], y = pos.data[i * 3 + 1], z = pos.data[i * 3 + 2]
                outPos[i * 3] = m[0] * x + m[4] * y + m[8] * z + m[12]
                outPos[i * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13]
                outPos[i * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14]
                if (nrm && nrm.size === 3) {
                    const nx = nrm.data[i * 3], ny = nrm.data[i * 3 + 1], nz = nrm.data[i * 3 + 2]
                    const tx = nm[0] * nx + nm[3] * ny + nm[6] * nz
                    const ty = nm[1] * nx + nm[4] * ny + nm[7] * nz
                    const tz = nm[2] * nx + nm[5] * ny + nm[8] * nz
                    const len = Math.hypot(tx, ty, tz)
                    if (len > 1e-12) {
                        outNrm[i * 3] = tx / len; outNrm[i * 3 + 1] = ty / len; outNrm[i * 3 + 2] = tz / len
                        continue
                    }
                }
                outNrm[i * 3] = NaN; outNrm[i * 3 + 1] = NaN; outNrm[i * 3 + 2] = NaN
            }

            let raw: Uint32Array
            if (prim.indices !== undefined) raw = readIndices(prim.indices)
            else {
                raw = new Uint32Array(count)
                for (let i = 0; i < count; i++) raw[i] = i
            }
            const tris = toTriangleList(raw, mode)
            for (let i = 0; i < tris.length; i++) {
                if (tris[i] >= count) throw new MeshParseError('A triangle points at a vertex that does not exist.', 'MALFORMED')
                tris[i] += vertexCount
            }
            triangleCount += tris.length / 3
            if (triangleCount > MAX_MESH_TRIANGLES) {
                throw new MeshParseError(`This model has more than ${MAX_MESH_TRIANGLES.toLocaleString('en-US')} triangles. Simplify it and try again.`, 'TOO_MANY_TRIANGLES')
            }
            posChunks.push(outPos)
            nrmChunks.push(outNrm)
            idxChunks.push(tris)
            vertexCount += count
        }
    }

    if (triangleCount === 0) throw new MeshParseError('This model has no triangles to draw.', 'NO_TRIANGLES')

    return {
        positions: concatFloat(posChunks, vertexCount * 3),
        normals: concatFloat(nrmChunks, vertexCount * 3),
        indices: concatUint(idxChunks, triangleCount * 3),
    }
}

function toTriangleList(raw: Uint32Array, mode: number): Uint32Array {
    if (mode === 4) return raw.slice(0, raw.length - (raw.length % 3))
    const n = Math.max(0, raw.length - 2)
    const out = new Uint32Array(n * 3)
    for (let i = 0; i < n; i++) {
        if (mode === 5) {
            // Strip: alternate the winding so every triangle faces the same way.
            const even = i % 2 === 0
            out[i * 3] = raw[i]
            out[i * 3 + 1] = even ? raw[i + 1] : raw[i + 2]
            out[i * 3 + 2] = even ? raw[i + 2] : raw[i + 1]
        } else {
            out[i * 3] = raw[0]
            out[i * 3 + 1] = raw[i + 1]
            out[i * 3 + 2] = raw[i + 2]
        }
    }
    return out
}

function concatFloat(chunks: Float32Array[], length: number): Float32Array {
    const out = new Float32Array(length)
    let o = 0
    for (const c of chunks) { out.set(c, o); o += c.length }
    return out
}

function concatUint(chunks: Uint32Array[], length: number): Uint32Array {
    const out = new Uint32Array(length)
    let o = 0
    for (const c of chunks) { out.set(c, o); o += c.length }
    return out
}

// ── Column-major 4×4 matrices ──────────────────────────────────────────────────────────────

function mat4Identity(): Float64Array {
    const m = new Float64Array(16)
    m[0] = m[5] = m[10] = m[15] = 1
    return m
}

function mat4Multiply(a: Float64Array, b: Float64Array): Float64Array {
    const out = new Float64Array(16)
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3]
        }
    }
    return out
}

function nodeMatrix(node: GltfNode): Float64Array {
    if (node.matrix && node.matrix.length === 16) return Float64Array.from(node.matrix)
    const [tx, ty, tz] = node.translation ?? [0, 0, 0]
    const [qx, qy, qz, qw] = node.rotation ?? [0, 0, 0, 1]
    const [sx, sy, sz] = node.scale ?? [1, 1, 1]
    const m = new Float64Array(16)
    m[0] = (1 - 2 * (qy * qy + qz * qz)) * sx
    m[1] = (2 * (qx * qy + qz * qw)) * sx
    m[2] = (2 * (qx * qz - qy * qw)) * sx
    m[4] = (2 * (qx * qy - qz * qw)) * sy
    m[5] = (1 - 2 * (qx * qx + qz * qz)) * sy
    m[6] = (2 * (qy * qz + qx * qw)) * sy
    m[8] = (2 * (qx * qz + qy * qw)) * sz
    m[9] = (2 * (qy * qz - qx * qw)) * sz
    m[10] = (1 - 2 * (qx * qx + qy * qy)) * sz
    m[12] = tx; m[13] = ty; m[14] = tz; m[15] = 1
    return m
}

/** The cofactor of the upper 3×3 (∝ inverse-transpose), signed so mirrored nodes keep outward
 *  normals. Column-major 3×3. Directions only — callers renormalize. */
function normalMatrix(m: Float64Array): Float64Array {
    const a = m[0], b = m[4], c = m[8]
    const d = m[1], e = m[5], f = m[9]
    const g = m[2], h = m[6], i = m[10]
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    const s = det < 0 ? -1 : 1
    const out = new Float64Array(9)
    // Row-major cofactor C, stored column-major as M^-T (which equals C / det).
    out[0] = s * (e * i - f * h); out[3] = s * -(d * i - f * g); out[6] = s * (d * h - e * g)
    out[1] = s * -(b * i - c * h); out[4] = s * (a * i - c * g); out[7] = s * -(a * h - b * g)
    out[2] = s * (b * f - c * e); out[5] = s * -(a * f - c * d); out[8] = s * (a * e - b * d)
    return out
}

// ── Writer ──────────────────────────────────────────────────────────────────────────────────

/**
 * Write geometry as a minimal GLB: one mesh, one node, POSITION (+ NORMAL when every vertex has
 * one) and 32-bit indices. Round-trips through `parseGlb`.
 */
export function writeGlb(geometry: MeshGeometry): ArrayBuffer {
    const {positions, normals, indices} = geometry
    const vertexCount = positions.length / 3
    const withNormals = normals.length === positions.length && !normals.some((v) => Number.isNaN(v))

    const min = [Infinity, Infinity, Infinity]
    const max = [-Infinity, -Infinity, -Infinity]
    for (let i = 0; i < vertexCount; i++) {
        for (let c = 0; c < 3; c++) {
            const v = positions[i * 3 + c]
            if (v < min[c]) min[c] = v
            if (v > max[c]) max[c] = v
        }
    }

    const posBytes = positions.byteLength
    const nrmBytes = withNormals ? normals.byteLength : 0
    const idxBytes = indices.byteLength
    const binLength = posBytes + nrmBytes + idxBytes

    const bufferViews: GltfBufferView[] = [{buffer: 0, byteOffset: 0, byteLength: posBytes}]
    const accessors: (GltfAccessor & {min?: number[]; max?: number[]})[] = [
        {bufferView: 0, componentType: 5126, count: vertexCount, type: 'VEC3', min, max},
    ]
    const attributes: Record<string, number> = {POSITION: 0}
    if (withNormals) {
        bufferViews.push({buffer: 0, byteOffset: posBytes, byteLength: nrmBytes})
        accessors.push({bufferView: 1, componentType: 5126, count: vertexCount, type: 'VEC3'})
        attributes.NORMAL = 1
    }
    bufferViews.push({buffer: 0, byteOffset: posBytes + nrmBytes, byteLength: idxBytes})
    accessors.push({bufferView: bufferViews.length - 1, componentType: 5125, count: indices.length, type: 'SCALAR'})

    const gltf = {
        asset: {version: '2.0', generator: 'shaders'},
        buffers: [{byteLength: binLength}],
        bufferViews,
        accessors,
        meshes: [{primitives: [{attributes, indices: accessors.length - 1, mode: 4}]}],
        nodes: [{mesh: 0}],
        scenes: [{nodes: [0]}],
        scene: 0,
    }

    const jsonBytes = new TextEncoder().encode(JSON.stringify(gltf))
    const jsonPadded = jsonBytes.length + ((4 - (jsonBytes.length % 4)) % 4)
    const binPadded = binLength + ((4 - (binLength % 4)) % 4)
    const total = 12 + 8 + jsonPadded + 8 + binPadded
    const out = new ArrayBuffer(total)
    const view = new DataView(out)
    const bytes = new Uint8Array(out)
    view.setUint32(0, GLB_MAGIC, true)
    view.setUint32(4, 2, true)
    view.setUint32(8, total, true)
    view.setUint32(12, jsonPadded, true)
    view.setUint32(16, CHUNK_JSON, true)
    bytes.set(jsonBytes, 20)
    bytes.fill(0x20, 20 + jsonBytes.length, 20 + jsonPadded) // JSON chunk pads with spaces
    const binStart = 20 + jsonPadded
    view.setUint32(binStart, binPadded, true)
    view.setUint32(binStart + 4, CHUNK_BIN, true)
    const data = binStart + 8
    bytes.set(new Uint8Array(positions.buffer, positions.byteOffset, posBytes), data)
    if (withNormals) bytes.set(new Uint8Array(normals.buffer, normals.byteOffset, nrmBytes), data + posBytes)
    bytes.set(new Uint8Array(indices.buffer, indices.byteOffset, idxBytes), data + posBytes + nrmBytes)
    return out
}
