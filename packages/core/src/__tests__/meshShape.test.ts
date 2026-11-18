import {describe, it, expect} from 'vitest'
import {parseGlb, writeGlb, isGlb, MeshParseError, type MeshGeometry} from '../utilities/glb'
import {isMeshShape, meshUrlOfShape, isMesh3dShapeType} from '../utilities/meshShape'
import {isVolumetricShapeType} from '../utilities/sdf3d'
import {shapeContentExtent} from '../utilities/shapeEffectBounds'
import {prepareMesh} from '../gpu/kit/meshField'

/** Wrap a glTF JSON object + binary chunk into a GLB. */
function glb(json: object, bin: Uint8Array): ArrayBuffer {
    const jsonBytes = new TextEncoder().encode(JSON.stringify(json))
    const jsonLen = Math.ceil(jsonBytes.length / 4) * 4
    const binLen = Math.ceil(bin.length / 4) * 4
    const total = 12 + 8 + jsonLen + 8 + binLen
    const out = new ArrayBuffer(total)
    const v = new DataView(out)
    const b = new Uint8Array(out)
    v.setUint32(0, 0x46546c67, true)
    v.setUint32(4, 2, true)
    v.setUint32(8, total, true)
    v.setUint32(12, jsonLen, true)
    v.setUint32(16, 0x4e4f534a, true)
    b.fill(0x20, 20, 20 + jsonLen)
    b.set(jsonBytes, 20)
    v.setUint32(20 + jsonLen, binLen, true)
    v.setUint32(24 + jsonLen, 0x004e4942, true)
    b.set(bin, 28 + jsonLen)
    return out
}

/** A unit cube (24 split vertices, 12 triangles), no normals. */
function cube(): MeshGeometry {
    const faces: [number[], number[], number[]][] = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    ]
    const positions: number[] = []
    const indices: number[] = []
    for (const [n, u, w] of faces) {
        const base = positions.length / 3
        for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
            positions.push(n[0] + u[0] * a + w[0] * b, n[1] + u[1] * a + w[1] * b, n[2] + u[2] * a + w[2] * b)
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3)
    }
    return {positions: new Float32Array(positions), normals: new Float32Array(positions.length).fill(NaN), indices: new Uint32Array(indices)}
}

describe('mesh3D shape routing', () => {
    const model = JSON.stringify({type: 'mesh3D', meshUrl: 'https://example.com/model.glb', size: 0.4})

    it('reads the model URL from the shape JSON or object', () => {
        expect(meshUrlOfShape(model)).toBe('https://example.com/model.glb')
        expect(meshUrlOfShape({type: 'mesh3D', meshUrl: 'a.glb'})).toBe('a.glb')
        expect(meshUrlOfShape(JSON.stringify({type: 'circleSDF'}))).toBe('')
        expect(isMeshShape(model)).toBe(true)
        expect(isMeshShape(JSON.stringify({type: 'mesh3D'}))).toBe(false)
    })

    it('is a volumetric shape type', () => {
        expect(isMesh3dShapeType('mesh3D')).toBe(true)
        expect(isVolumetricShapeType('mesh3D')).toBe(true)
    })

    it('bounds are the bounding-sphere size under any rotation', () => {
        const {hwf, hhf} = shapeContentExtent({shape: model})
        expect(hwf).toBeCloseTo(0.42)
        expect(hhf).toBeCloseTo(0.42)
    })
})

describe('GLB reader', () => {
    it('round-trips geometry through writeGlb', () => {
        const geo = cube()
        geo.normals = new Float32Array(geo.positions.length)
        for (let i = 0; i < geo.normals.length; i += 3) geo.normals[i + 2] = 1
        const bytes = writeGlb(geo)
        expect(isGlb(bytes)).toBe(true)
        const back = parseGlb(bytes)
        expect(Array.from(back.positions)).toEqual(Array.from(geo.positions))
        expect(Array.from(back.normals)).toEqual(Array.from(geo.normals))
        expect(Array.from(back.indices)).toEqual(Array.from(geo.indices))
    })

    it('marks missing normals as NaN when written without them', () => {
        const back = parseGlb(writeGlb(cube()))
        expect(back.normals.every((v) => Number.isNaN(v))).toBe(true)
    })

    it('applies node transforms down the hierarchy and converts strips', () => {
        // One quad as a 4-vertex triangle strip, quantized (normalized u16) positions.
        const pos = new Uint16Array([0, 0, 0, 65535, 0, 0, 0, 65535, 0, 65535, 65535, 0])
        const bin = new Uint8Array(pos.buffer)
        const json = {
            asset: {version: '2.0'},
            extensionsUsed: ['KHR_mesh_quantization'],
            buffers: [{byteLength: bin.byteLength}],
            bufferViews: [{buffer: 0, byteOffset: 0, byteLength: bin.byteLength, byteStride: 6}],
            accessors: [{bufferView: 0, componentType: 5123, normalized: true, count: 4, type: 'VEC3'}],
            meshes: [{primitives: [{attributes: {POSITION: 0}, mode: 5}]}],
            nodes: [
                {children: [1], translation: [10, 0, 0]},
                {mesh: 0, scale: [2, 2, 2], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2]}, // 90° about z
            ],
            scenes: [{nodes: [0]}],
        }
        const geo = parseGlb(glb(json, bin))
        expect(geo.indices.length).toBe(6) // two triangles from a 4-index strip
        // Vertex 1 (1,0,0) → scale 2 → rotate 90° about z → (0,2,0) → +10 x → (10,2,0).
        expect(geo.positions[3]).toBeCloseTo(10)
        expect(geo.positions[4]).toBeCloseTo(2)
        expect(geo.positions[5]).toBeCloseTo(0)
    })

    it('rejects compressed models with a clear error', () => {
        const json = {asset: {version: '2.0'}, extensionsRequired: ['KHR_draco_mesh_compression'], extensionsUsed: ['KHR_draco_mesh_compression']}
        try {
            parseGlb(glb(json, new Uint8Array(4)))
            throw new Error('expected a MeshParseError')
        } catch (e) {
            expect(e).toBeInstanceOf(MeshParseError)
            expect((e as MeshParseError).code).toBe('COMPRESSED')
        }
    })

    it('rejects files that are not GLB, and models with no triangles', () => {
        expect(() => parseGlb(new TextEncoder().encode('{"asset":{}}').buffer as ArrayBuffer)).toThrow(MeshParseError)
        expect(() => parseGlb(glb({asset: {version: '2.0'}}, new Uint8Array(4)))).toThrow(/no triangles/)
    })
})

describe('prepareMesh', () => {
    const decode = (e: number): [number, number, number] => {
        const sn = (v: number) => Math.max(((v << 16) >> 16) / 32767, -1)
        let x = sn(e & 0xffff), y = sn(e >>> 16)
        const z = 1 - Math.abs(x) - Math.abs(y)
        const t = Math.max(-z, 0)
        x += x >= 0 ? -t : t
        y += y >= 0 ? -t : t
        const l = Math.hypot(x, y, z)
        return [x / l, y / l, z / l]
    }

    it('centres the model in a unit bounding sphere and flips z to face the camera', () => {
        const geo = cube()
        for (let i = 0; i < geo.positions.length; i += 3) geo.positions[i] += 5 // off-centre
        const m = prepareMesh(geo)
        let maxR = 0
        for (let i = 0; i < m.positions.length; i += 3) maxR = Math.max(maxR, Math.hypot(m.positions[i], m.positions[i + 1], m.positions[i + 2]))
        expect(maxR).toBeCloseTo(1, 5)
        expect(m.halfExtents[0]).toBeCloseTo(1 / Math.sqrt(3), 5)
        // The +z face (vertices 16..19) ends up at negative z: toward the camera.
        expect(m.positions[16 * 3 + 2]).toBeLessThan(0)
    })

    it('keeps a cube\'s edges hard when generating normals (60° crease)', () => {
        const m = prepareMesh(cube())
        // Every corner of the +x face (first two triangles) gets the face normal, ±x.
        for (let k = 0; k < 6; k++) {
            const [x, y, z] = decode(m.cornerNormals[k])
            expect(Math.abs(x)).toBeCloseTo(1, 3)
            expect(Math.abs(y)).toBeLessThan(1e-3)
            expect(Math.abs(z)).toBeLessThan(1e-3)
        }
    })

    it('smooths across shallow folds', () => {
        // Two triangles sharing an edge, folded 20°: the shared corners average the two faces.
        const a = 20 * Math.PI / 180
        const geo: MeshGeometry = {
            positions: new Float32Array([0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, Math.cos(a), 0, Math.sin(a)]),
            normals: new Float32Array(18).fill(NaN),
            indices: new Uint32Array([0, 1, 2, 3, 5, 4]),
        }
        const m = prepareMesh(geo)
        const shared = decode(m.cornerNormals[0])
        const lone = decode(m.cornerNormals[2])
        // The lone corner is still flat (±z); the shared one tilts by half the fold.
        expect(Math.abs(lone[2])).toBeCloseTo(1, 3)
        expect(Math.abs(shared[2])).toBeCloseTo(Math.cos(a / 2), 2)
    })

    it('uses authored normals as given (z flipped into the field frame)', () => {
        const geo = cube()
        geo.normals = new Float32Array(geo.positions.length)
        for (let i = 0; i < geo.normals.length; i += 3) geo.normals[i + 2] = 1
        const m = prepareMesh(geo)
        expect(decode(m.cornerNormals[0])[2]).toBeCloseTo(-1, 3)
    })
})
