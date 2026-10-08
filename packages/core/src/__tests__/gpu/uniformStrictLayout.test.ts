import {describe, it, expect, vi} from 'vitest'
import {d} from '@coreroot/gpu/kit'
import {createUniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {createGpuUniformsMap} from '@coreroot/gpu/uniformBridge'
import {MarchParams, SampleParams} from '@coreroot/gpu/kit/sdf3d'
import {getAllShaders} from '@coreroot/shaderRegistry'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import type {AnyWgslData, WgslStruct} from 'typegpu/data'

/**
 * Strict-uniform-layout gate.
 *
 * WGSL's uniform address space carries extra layout constraints unless the
 * `uniform_buffer_standard_layout` language extension is exposed (it isn't on Safari, Firefox,
 * or Chromium ≲ 150 — e.g. Electron shells):
 *
 *   1. a struct-typed member must sit at an offset that is a multiple of 16, and the next
 *      member must start ≥ roundUp(16, sizeOf(struct)) bytes after it,
 *   2. an array member must have a 16-byte-multiple element stride and sit at an offset that is
 *      a multiple of roundUp(16, alignOf(element)).
 *
 * TypeGPU emits natural (storage-style) layout by default, so violations surface only on those
 * browsers, as pipeline-creation failures ("struct member offset must be a multiple of 16
 * bytes") — a real-world partner regression that blanked 445/778 catalog presets on
 * Chromium 142. This gate walks every uniform-buffer schema the renderer creates (the packed
 * per-composition struct across ALL registry shaders, plus the standalone compute param
 * structs that nest structs) and re-checks the strict rules from the schema's own
 * alignments/sizes, GPU-free.
 */

type SchemaNode = AnyWgslData & {
    type: string
    inner?: SchemaNode
    propTypes?: Record<string, AnyWgslData>
    elementType?: AnyWgslData
    elementCount?: number
}

const roundUp = (k: number, n: number): number => Math.ceil(n / k) * k

/** Strip `d.align`/`d.size` decoration wrappers (alignmentOf/sizeOf already account for them). */
function unwrap(schema: AnyWgslData): SchemaNode {
    let s = schema as SchemaNode
    while (s.type === 'decorated' && s.inner) s = s.inner
    return s
}

/**
 * Validate the strict (non-`uniform_buffer_standard_layout`) uniform layout rules for a struct
 * schema, recursively. Returns human-readable violations; [] means valid on every WebGPU
 * implementation.
 */
function strictUniformViolations(struct: WgslStruct, path = 'uniforms'): string[] {
    const problems: string[] = []
    const members = Object.entries(struct.propTypes as Record<string, AnyWgslData>)
    let offset = 0
    let prevStruct: {name: string; offset: number; size: number} | null = null

    for (const [name, member] of members) {
        const align = d.alignmentOf(member)
        const size = d.sizeOf(member)
        offset = roundUp(align, offset)
        const here = `${path}.${name}`
        const bare = unwrap(member)

        if (prevStruct) {
            const gap = offset - prevStruct.offset
            if (gap < roundUp(16, prevStruct.size)) {
                problems.push(
                    `${path}.${prevStruct.name}: struct member (size ${prevStruct.size}) is followed by "${name}" after ${gap} bytes — needs ≥ ${roundUp(16, prevStruct.size)}`,
                )
            }
        }
        prevStruct = null

        if (bare.type === 'struct') {
            if (offset % 16 !== 0) {
                problems.push(`${here}: struct member at offset ${offset} — must be a multiple of 16`)
            }
            prevStruct = {name, offset, size}
            problems.push(...strictUniformViolations(bare as unknown as WgslStruct, here))
        } else if (bare.type === 'array' && bare.elementType) {
            const elem = bare.elementType
            const stride = roundUp(d.alignmentOf(elem), d.sizeOf(elem))
            if (stride % 16 !== 0) {
                problems.push(`${here}: array element stride ${stride} — must be a multiple of 16`)
            }
            if (offset % roundUp(16, d.alignmentOf(elem)) !== 0) {
                problems.push(`${here}: array member at offset ${offset} — must be a multiple of ${roundUp(16, d.alignmentOf(elem))}`)
            }
            const bareElem = unwrap(elem)
            if (bareElem.type === 'struct') {
                problems.push(...strictUniformViolations(bareElem as unknown as WgslStruct, `${here}[]`))
            }
        }

        offset += size
    }
    return problems
}

function mockRoot() {
    const buffer = {
        patch: vi.fn(),
        write: vi.fn(),
        destroy: vi.fn(),
        $usage: vi.fn(function (this: unknown) {
            return buffer
        }),
    }
    return {createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({}))} as never
}

function bridgeFieldInits(def: GpuShaderDefinition, id: string): FieldInit[] {
    const props: Record<string, unknown> = {}
    for (const [name, cfg] of Object.entries(def.props ?? {})) props[name] = (cfg as {default: unknown}).default
    const map = createGpuUniformsMap(def as never, props, id)
    const inits: FieldInit[] = []
    for (const [name, u] of Object.entries(map)) {
        inits.push({name, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema})
    }
    // The synthetics the renderer registers on every node struct (see the pattern harness).
    inits.push({name: '_opacity', schema: d.f32, initial: 1})
    if (def.animatedTime) inits.push({name: '_animTime', schema: d.f32, initial: 0})
    for (const k of Object.keys(def.extraAnimatedTimes ?? {})) inits.push({name: `_animTime_${k}`, schema: d.f32, initial: 0})
    for (const [name, cfg] of Object.entries(def.extraFields ?? {})) inits.push({name, schema: cfg.schema, initial: cfg.initial})
    return inits
}

describe('strict uniform layout (uniform_buffer_standard_layout not required)', () => {
    it('the validator itself catches a known-invalid layout (negative control)', () => {
        const bad = d.struct({inner: d.struct({x: d.f32}), after: d.f32})
        expect(strictUniformViolations(bad)).not.toEqual([])
        const good = d.struct({inner: d.align(16, d.size(16, d.struct({x: d.f32}))), after: d.f32})
        expect(strictUniformViolations(good)).toEqual([])
    })

    it('the packed uniform struct across EVERY registry shader satisfies the strict rules', () => {
        const allShaders = getAllShaders()
        expect(allShaders.length).toBeGreaterThan(100)

        const store = createUniformStore(mockRoot(), {systemSchema: SystemUniforms})
        let nodeCount = 0
        for (const entry of allShaders) {
            const def = entry.definition as GpuShaderDefinition
            store.defineNode(`node_${entry.name}_${nodeCount}`, bridgeFieldInits(def, `node_${entry.name}_${nodeCount}`))
            nodeCount++
        }
        const {schema} = store.finalize()

        expect(nodeCount).toBeGreaterThan(100)
        expect(strictUniformViolations(schema)).toEqual([])
    })

    it('standalone compute uniform param structs satisfy the strict rules', () => {
        expect(strictUniformViolations(MarchParams, 'MarchParams')).toEqual([])
        expect(strictUniformViolations(SampleParams, 'SampleParams')).toEqual([])
    })
})

// ─── Real-serializer gate ────────────────────────────────────────────────────────────────
//
// The mock-root tests stub `buffer.patch`/`buffer.write`, so schema shapes that break
// typegpu's ACTUAL serializers sail through CI and explode only in the browser. Case in
// point: wrapping node structs in `d.align(16, …)` produced valid WGSL and passed every
// mocked test, but typegpu's partial-write path (partialIO `collect`) does not unwrap
// Decorated members — flush() patches then wrote NaN over sibling fields and threw
// "Cannot read properties of undefined (reading '0')" on vec members, every frame.
// These tests capture the store's real flush()/writeAll() payloads and run them through
// typegpu's own partial + compiled writers against a plain ArrayBuffer — no GPU needed.
// (Deep path import: typegpu's exports map hides partialIO; path-based resolution works
// and follows the workspace-linked version.)
import {FieldHandle, ArrayFieldHandle} from '@coreroot/gpu/uniformStore'
import {getPatchInstructions} from '../../../node_modules/typegpu/data/partialIO.js'
import {getCompiledWriter} from '../../../node_modules/typegpu/data/compiledIO.js'

describe('packed uniform buffer serializes through typegpu real writers (GPU-free)', () => {
    function buildAllShadersStore() {
        const buffer = {
            patch: vi.fn(),
            write: vi.fn(),
            destroy: vi.fn(),
            $usage: vi.fn(function (this: unknown) {
                return buffer
            }),
        }
        const root = {createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({}))} as never
        const store = createUniformStore(root, {systemSchema: SystemUniforms})
        const allHandles: Array<FieldHandle | ArrayFieldHandle> = []
        let n = 0
        for (const entry of getAllShaders()) {
            const def = entry.definition as GpuShaderDefinition
            const handles = store.defineNode(`node_${entry.name}_${n}`, bridgeFieldInits(def, `node_${entry.name}_${n}`))
            allHandles.push(...Object.values(handles))
            n++
        }
        const {schema} = store.finalize()
        return {store, schema, buffer, allHandles}
    }

    it('writeAll payload survives the compiled full writer (pads written as zeros)', () => {
        const {schema, buffer} = buildAllShadersStore()
        // finalize() already ran writeAll() once — grab its real payload.
        const payload = buffer.write.mock.calls[0][0]
        const bytes = new ArrayBuffer(d.sizeOf(schema))
        expect(() => getCompiledWriter(schema)(new DataView(bytes), 0, payload, true)).not.toThrow()
        // Whole-buffer write with pads-as-zeros must leave no NaN anywhere.
        expect(Array.from(new Float32Array(bytes)).some((v) => Number.isNaN(v))).toBe(false)
    })

    it('an every-field flush() payload survives the real partial writer', () => {
        const {store, schema, buffer, allHandles} = buildAllShadersStore()
        for (const h of allHandles) {
            if (h instanceof ArrayFieldHandle) h.array = h.array
            else if (!h.cpu) h.value = h.value
        }
        store.flush()
        const payload = buffer.patch.mock.calls[0][0]
        expect(() => getPatchInstructions(schema, payload)).not.toThrow()
    })

    it('a single-scalar flush() stays sparse (exactly 4 bytes patched)', () => {
        const {store, schema, buffer, allHandles} = buildAllShadersStore()
        const scalar = allHandles.find((h) => h instanceof FieldHandle && !h.cpu && h._componentCount === 1) as FieldHandle
        scalar.value = 0.123
        store.flush()
        const payload = buffer.patch.mock.calls[0][0]
        const instructions = getPatchInstructions(schema, payload) as Array<{data: Uint8Array}>
        const totalBytes = instructions.reduce((sum, i) => sum + i.data.length, 0)
        // The Decorated-member regression wrote the WHOLE node struct (NaN-ing siblings);
        // a healthy sparse patch touches only the one f32.
        expect(totalBytes).toBe(4)
    })
})
