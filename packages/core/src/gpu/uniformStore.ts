/**
 * Packed-uniform system.
 *
 * A composition builds ONE uniform buffer whose schema is a struct of nested per-node
 * structs plus a `_sys` system struct:
 *
 *   d.struct({ n_<sanitizedId>: <PerNodeStruct>, ..., _sys: SystemUniforms })
 *
 * Every GPU-relevant prop becomes a `FieldHandle` (scalar/vector) or `ArrayFieldHandle`
 * (fixed-size arrays, e.g. colorStops). A FieldHandle is the load-bearing abstraction over one
 * field in the packed struct:
 *
 *   - `.value` (get/set) — CPU mirror. Reads return the last-set value; writes convert via
 *     the prop transform (when the raw path is used) and route into a coalesced patch. This
 *     gives driver code, `updateUniformValue`, dimensional re-resolution, and
 *     `getLiveScalarValue` a single write path.
 *   - `.gpuPath` / `.accessorPath` — how the composer references the field on GPU when it
 *     emits glue WGSL (see the "GPU reference contract" note below).
 *
 * Patches accumulate across a frame and flush once (`flush()`), one `buffer.patch(...)` with
 * a single merged nested-partial object. `writeAll()` is the whole-buffer fallback.
 *
 * NOTE (transforms): the setter accepts every value shape a prop transform can produce — a
 * `{ node, data }` pair, a `Vector4`/`Vector2`-like `{ x, y, z, w }`, a `d.vec2f`/`d.vec4f`
 * instance, a plain array, or a number — by duck-typing, so this module never imports the
 * transform functions.
 */
import tgpu, {type TgpuRoot, type TgpuBindGroupLayout, type TgpuBindGroup} from 'typegpu'
import * as d from 'typegpu/data'
import type {AnyWgslData, WgslStruct, WgslArray} from 'typegpu/data'
import {GpuUnavailableError} from './support'

// ---------------------------------------------------------------------------------------
// GPU reference contract (consumed by the composer)
// ---------------------------------------------------------------------------------------
//
// The uniform buffer lives in bind group 0 under a SINGLE layout entry named `uniforms`
// (`tgpu.bindGroupLayout({ uniforms: { uniform: <combined struct> } })`). A field's location
// inside the combined struct is its `gpuPath` — e.g. `['n_x3', 'speed']` — and its
// `accessorPath` is the dotted form `'n_x3.speed'`.
//
// So the fully-qualified GPU accessor the composer writes into glue WGSL is
//   `<layoutVar>.$.uniforms.<accessorPath>`   e.g. `layout.$.uniforms.n_x3.speed`
// and a per-node driver-override block is
//   `var p = layout.$.uniforms.n_x3;  p.speed = remap(textureSample(...));`
//
// `store.gpuAccessor(handle, layoutVar?)` builds that string. Note the `.uniforms` segment:
// `tgpu.bindGroupLayout` always keys its entries, so the GPU accessor includes the entry key,
// whereas `buffer.patch({ n_x3: { speed } })` does NOT — the buffer's own schema IS the
// combined struct.

/** The bind-group-layout entry key the combined uniform struct is bound under. */
export const UNIFORM_ENTRY_KEY = 'uniforms' as const

/**
 * Name sanitization — uniform names become WGSL struct member identifiers verbatim, so strip
 * anything WGSL-invalid (dots from `Math.random` ids, hyphens in user ids, …). Node keys are
 * additionally prefixed `n_` so a numeric-leading id can't produce an invalid identifier.
 */
export function sanitizeId(id: string): string {
    return id.replace(/[^a-zA-Z0-9_]/g, '_')
}

// WGSL reserved words / keywords that are INVALID as struct-member identifiers.
// A prop named `loop` (VideoTexture) — or `sampler`/`uniform`/`storage`/`type`/… — would sanitize
// to a bare reserved word and make `d.struct({loop: …})` throw. This is the exact set typegpu
// rejects (`typegpu/nameUtils` `bannedTokens`, copied here so we don't depend on an internal path).
const WGSL_RESERVED = new Set<string>([
    'alias', 'break', 'case', 'const', 'const_assert', 'continue', 'continuing', 'default',
    'diagnostic', 'discard', 'else', 'enable', 'false', 'fn', 'for', 'if', 'let', 'loop',
    'override', 'requires', 'return', 'struct', 'switch', 'true', 'var', 'while', 'NULL', 'Self',
    'abstract', 'active', 'alignas', 'alignof', 'as', 'asm', 'asm_fragment', 'async', 'attribute',
    'auto', 'await', 'become', 'cast', 'catch', 'class', 'co_await', 'co_return', 'co_yield',
    'coherent', 'column_major', 'common', 'compile', 'compile_fragment', 'concept', 'const_cast',
    'consteval', 'constexpr', 'constinit', 'crate', 'debugger', 'decltype', 'delete', 'demote',
    'demote_to_helper', 'do', 'dynamic_cast', 'enum', 'explicit', 'export', 'extends', 'extern',
    'external', 'fallthrough', 'filter', 'final', 'finally', 'friend', 'from', 'fxgroup', 'get',
    'goto', 'groupshared', 'highp', 'impl', 'implements', 'import', 'inline', 'instanceof',
    'interface', 'layout', 'lowp', 'macro', 'macro_rules', 'match', 'mediump', 'meta', 'mod',
    'module', 'move', 'mut', 'mutable', 'namespace', 'new', 'nil', 'noexcept', 'noinline',
    'nointerpolation', 'non_coherent', 'noncoherent', 'noperspective', 'null', 'nullptr', 'of',
    'operator', 'package', 'packoffset', 'partition', 'pass', 'patch', 'pixelfragment', 'precise',
    'precision', 'premerge', 'priv', 'protected', 'pub', 'public', 'readonly', 'ref', 'regardless',
    'register', 'reinterpret_cast', 'require', 'resource', 'restrict', 'self', 'set', 'shared',
    'sizeof', 'smooth', 'snorm', 'static', 'static_assert', 'static_cast', 'std', 'subroutine',
    'super', 'target', 'template', 'this', 'thread_local', 'throw', 'trait', 'try', 'type',
    'typedef', 'typeid', 'typename', 'typeof', 'union', 'unless', 'unorm', 'unsafe', 'unsized',
    'use', 'using', 'varying', 'virtual', 'volatile', 'wgsl', 'where', 'with', 'writeonly', 'yield',
    'sampler', 'uniform', 'storage',
])

/**
 * Sanitize a STRUCT-MEMBER name: strip WGSL-invalid chars, then suffix `_` if the result is a
 * reserved word. Node keys already get an `n_` prefix (never reserved), so only per-field names
 * (props + synthetics) route through here — the accessor path derives from the same handle
 * `gpuPath`, so struct field + accessor stay consistent.
 */
export function sanitizeFieldId(id: string): string {
    const s = sanitizeId(id)
    return WGSL_RESERVED.has(s) ? `${s}_` : s
}

/**
 * A stable text form of a uniform schema: member names, types, array lengths and layout
 * decorations, plus the total size. Two stores with the same signature lay their buffers out
 * byte for byte the same, so one can bind through a layout built for the other.
 */
export function schemaSignature(schema: unknown): string {
    const sig = (t: unknown): string => {
        // TypeGPU structs are callable (a schema is also its own constructor): functions count
        if (!t || (typeof t !== 'object' && typeof t !== 'function')) return String(t)
        const s = t as {type?: string; propTypes?: Record<string, unknown>; elementType?: unknown; elementCount?: number; inner?: unknown; attribs?: unknown[]}
        if (s.type === 'struct' && s.propTypes) return `{${Object.entries(s.propTypes).map(([k, v]) => `${k}:${sig(v)}`).join(',')}}`
        if (s.type === 'array') return `array<${sig(s.elementType)},${s.elementCount}>`
        if (s.type === 'decorated') return `@${JSON.stringify(s.attribs ?? [])}${sig(s.inner)}`
        return s.type ?? '?'
    }
    let size = '?'
    try {
        size = String(d.sizeOf(schema as AnyWgslData))
    } catch {
        /* not sizeable: the member text still differs */
    }
    return `${size}:${sig(schema)}`
}

/** The runtime struct key for a node, e.g. `n_abc123`. */
export function nodeKey(rawId: string): string {
    return `n_${sanitizeId(rawId)}`
}

// ---------------------------------------------------------------------------------------
// Schema kind detection + JS-value → schema mapping
// ---------------------------------------------------------------------------------------

type VecKind = 'vec2' | 'vec3' | 'vec4'
type FieldKind = 'scalar' | VecKind

function componentsOf(kind: VecKind): number {
    return kind === 'vec2' ? 2 : kind === 'vec3' ? 3 : 4
}

/** Classify a WGSL data schema into the handle kind this store manages. */
function kindOfSchema(schema: AnyWgslData): FieldKind {
    const t = (schema as {type?: string}).type
    if (t === 'vec2f' || t === 'vec2i' || t === 'vec2u') return 'vec2'
    if (t === 'vec3f' || t === 'vec3i' || t === 'vec3u') return 'vec3'
    if (t === 'vec4f' || t === 'vec4i' || t === 'vec4u') return 'vec4'
    return 'scalar'
}

function isArraySchema(schema: AnyWgslData): schema is WgslArray {
    return (schema as {type?: string}).type === 'array'
}

/**
 * Infer the packed-struct field schema from a JS value (post-transform):
 *   number → f32, boolean → f32 (transforms already encode booleans as ±1),
 *   {x,y}/vec2f → vec2f, {x,y,z}/vec3f → vec3f, color/{x,y,z,w}/vec4f → vec4f,
 *   number[] → arrayOf(f32, length).
 * Strings (urls / shape JSON / origin / compile-time selects) are CPU-only — they never
 * enter the struct; callers mark those `cpu: true` and get a mirror-only FieldHandle.
 *
 * CAUTION (strict uniform layout): `arrayOf(f32, n)` has a 4-byte stride, which is invalid in
 * the uniform address space on browsers without `uniform_buffer_standard_layout` (arrays there
 * need 16-byte strides). No shader hits this inference today — array props (colorStops) pass
 * explicit vec4f/vec3f schemas, whose 16-byte strides are valid everywhere. A new plain
 * `number[]` prop must pack into vec4s (`d.arrayOf(d.vec4f, ceil(n/4))`), not rely on this.
 */
export function inferFieldSchema(value: unknown): AnyWgslData {
    if (typeof value === 'number' || typeof value === 'boolean') return d.f32
    if (value && typeof value === 'object') {
        // Check `.kind` BEFORE Array.isArray — d.vecXf instances are runtime arrays, so the
        // array branch would otherwise swallow them as arrayOf(f32, n).
        const kind = (value as {kind?: string}).kind
        if (typeof kind === 'string') {
            if (kind.startsWith('vec2')) return d.vec2f
            if (kind.startsWith('vec3')) return d.vec3f
            if (kind.startsWith('vec4')) return d.vec4f
        }
        if (Array.isArray(value)) return d.arrayOf(d.f32, value.length || 1)
        const v = value as Record<string, unknown>
        if ('w' in v) return d.vec4f
        if ('z' in v) return d.vec3f
        if ('x' in v && 'y' in v) return d.vec2f
    }
    if (Array.isArray(value)) return d.arrayOf(d.f32, value.length || 1)
    // Fallback (defensive): treat as a scalar rather than crash the composition.
    return d.f32
}

/**
 * Normalize any accepted "vector-ish" input into a plain number[] of length `n`.
 * Accepts: `{ node, data }` (a transform pair — unwraps `.data`), Vector4/Vector2-like
 * `{ x, y, z, w }`, a `d.vecXf` instance, a `VectorFieldView`, a plain array / TypedArray,
 * or a scalar (broadcast to all components, matching `d.vecXf(scalar)`).
 */
function toComponents(input: unknown, n: number): number[] {
    let v: unknown = input
    // transform pair { node, data } → the CPU data half.
    if (v && typeof v === 'object' && 'data' in (v as object) && !('x' in (v as object))) {
        v = (v as {data: unknown}).data
    }
    if (typeof v === 'number') {
        return new Array(n).fill(v)
    }
    if (v instanceof VectorFieldView) {
        return v.toArray().slice(0, n)
    }
    if (Array.isArray(v)) {
        return padTo(v as number[], n)
    }
    if (v && typeof v === 'object') {
        const o = v as Record<string, number> & {length?: number}
        if ('x' in o) {
            const out = [o.x, o.y ?? 0]
            if (n >= 3) out.push(o.z ?? 0)
            if (n >= 4) out.push(o.w ?? 0)
            return out.slice(0, n)
        }
        if (typeof o.length === 'number') {
            // d.vecXf instances are tuple-like (indexable + length); also covers TypedArrays.
            return padTo(Array.from(o as ArrayLike<number>), n)
        }
    }
    throw new Error(`[uniformStore] cannot coerce value to a ${n}-component vector: ${String(input)}`)
}

function padTo(arr: number[], n: number): number[] {
    const out = arr.slice(0, n)
    while (out.length < n) out.push(0)
    return out
}

/** Extract the scalar of a post-transform value ({ node, data } / number / boolean-as-±1). */
function toScalar(input: unknown): number {
    let v: unknown = input
    if (v && typeof v === 'object' && 'data' in (v as object)) v = (v as {data: unknown}).data
    if (typeof v === 'number') return v
    if (typeof v === 'boolean') return v ? 1 : -1
    const n = Number(v)
    return Number.isFinite(n) ? n : 0
}

// ---------------------------------------------------------------------------------------
// Applied-transform result guard
// ---------------------------------------------------------------------------------------
//
// The prop-transform bridge (`gpuTransformFor`) APPLIES every prop transform: color/position
// swap to their GPU equivalents (d.vec*f), and every other transform — shared scalar,
// inline numeric, boolean→float, enum string→number — is called directly to produce the packed
// field value. A pure transform therefore yields a number, boolean, string (→ cpu mirror), a
// vector, or an array. If a shader writes an inline transform that returns anything else
// (a `Color`/`Vector` with no x/y, an arbitrary object), the value silently packs to NaN
// (`inferFieldSchema` falls back to f32, `toScalar` → `Number(obj)` → NaN). This guard surfaces
// that mistake at compose time with a one-time warning per offending transform.
const warnedNonPackableTransforms = new Set<unknown>()

/** True if a (post-transform) value can be packed into a uniform field or held as a cpu mirror. */
function isPackableFieldValue(value: unknown): boolean {
    if (value === null || value === undefined) return true // resolved to a default upstream
    const t = typeof value
    if (t === 'number' || t === 'boolean' || t === 'string') return true
    if (t !== 'object') return false
    if (Array.isArray(value)) return true
    const o = value as Record<string, unknown>
    if (typeof o.kind === 'string' && (o.kind as string).startsWith('vec')) return true // d.vecXf
    if ('x' in o && 'y' in o) return true // Vector-like {x, y[, z[, w]]}
    if (typeof (o as {length?: unknown}).length === 'number') return true // tuple / typed array
    if ('data' in o) return true // { node, data } transform pair (toComponents unwraps .data)
    return false
}

// ---------------------------------------------------------------------------------------
// FieldHandle — one field in the packed struct
// ---------------------------------------------------------------------------------------

/** Minimal surface the store exposes to its handles for coalesced dirty-tracking. */
interface DirtySink {
    markScalar(handle: FieldHandle): void
    markArray(handle: ArrayFieldHandle, index: number | 'all'): void
}

/**
 * A live view over a vector field's CPU mirror, returned by `FieldHandle.value` for vector
 * fields. Its component accessors and `.copy(...)` / `.set(...)` methods route back into the
 * store's coalesced patch — supporting in-place idioms like `uniform.value.copy(data)` and
 * `smoothedUniform.value.set(x, y)`.
 */
export class VectorFieldView {
    constructor(private readonly handle: FieldHandle) {}

    private get n(): number {
        return this.handle._componentCount
    }
    private component(i: number): number {
        return this.handle._mirror[i] ?? 0
    }
    private write(i: number, value: number): void {
        this.handle._mirror[i] = value
        this.handle._enqueue()
    }

    get x(): number { return this.component(0) }
    set x(v: number) { this.write(0, v) }
    get y(): number { return this.component(1) }
    set y(v: number) { this.write(1, v) }
    get z(): number { return this.component(2) }
    set z(v: number) { this.write(2, v) }
    get w(): number { return this.component(3) }
    set w(v: number) { this.write(3, v) }
    get r(): number { return this.component(0) }
    set r(v: number) { this.write(0, v) }
    get g(): number { return this.component(1) }
    set g(v: number) { this.write(1, v) }
    get b(): number { return this.component(2) }
    set b(v: number) { this.write(2, v) }
    get a(): number { return this.component(3) }
    set a(v: number) { this.write(3, v) }

    /** `Vector.copy(src)` — copy another vector-like's components in-place. */
    copy(src: unknown): this {
        const c = toComponents(src, this.n)
        for (let i = 0; i < this.n; i++) this.handle._mirror[i] = c[i]
        this.handle._enqueue()
        return this
    }

    /** `Vector.set(x, y[, z[, w]])`. */
    set(...values: number[]): this {
        for (let i = 0; i < this.n && i < values.length; i++) this.handle._mirror[i] = values[i]
        this.handle._enqueue()
        return this
    }

    toArray(): number[] {
        return this.handle._mirror.slice(0, this.n)
    }
}

/**
 * Backed by one field in the packed struct; keeps a CPU mirror for `.value` reads and routes
 * `.value` writes into the store's coalesced patch.
 */
export class FieldHandle {
    /** Struct-relative path, e.g. `['n_x3', 'speed']`. See the GPU reference contract. */
    readonly gpuPath: readonly [string, string]
    /** Dotted struct-relative accessor, e.g. `'n_x3.speed'`. */
    readonly accessorPath: string
    /** The prop transform (if any). */
    readonly transform?: (value: unknown) => unknown
    /** CPU-only props (strings/JSON) have no struct field — mirror-only, never patched. */
    readonly cpu: boolean
    /** The field's WGSL schema (undefined for cpu-only handles). */
    readonly schema?: AnyWgslData

    /** @internal component count (1 for scalar, 2/3/4 for vectors). */
    readonly _componentCount: number
    /** @internal CPU mirror: index 0 for scalar; [x,y,(z),(w)] for vectors; raw value for cpu-only. */
    _mirror: number[]
    private _raw: unknown
    private readonly _kind: FieldKind
    private readonly _view?: VectorFieldView
    private readonly _sink: DirtySink

    constructor(opts: {
        sink: DirtySink
        gpuPath: [string, string]
        schema?: AnyWgslData
        initial: unknown
        transform?: (value: unknown) => unknown
        cpu?: boolean
    }) {
        this._sink = opts.sink
        this.gpuPath = opts.gpuPath
        this.accessorPath = `${opts.gpuPath[0]}.${opts.gpuPath[1]}`
        this.transform = opts.transform
        this.cpu = opts.cpu ?? false
        this.schema = opts.schema
        this._kind = this.cpu || !opts.schema ? 'scalar' : kindOfSchema(opts.schema)
        this._componentCount = this._kind === 'scalar' ? 1 : componentsOf(this._kind)

        if (this.cpu) {
            this._mirror = []
            this._raw = opts.initial
        } else if (this._kind === 'scalar') {
            this._mirror = [toScalar(opts.initial)]
        } else {
            this._mirror = toComponents(opts.initial, this._componentCount)
            this._view = new VectorFieldView(this)
        }
    }

    /** @internal enqueue this field into the store's coalesced patch. */
    _enqueue(): void {
        if (!this.cpu) this._sink.markScalar(this)
    }

    /** @internal the value written into a `buffer.patch` for this field. */
    _patchValue(): number | number[] {
        return this._kind === 'scalar' ? this._mirror[0] : this._mirror.slice(0, this._componentCount)
    }

    /**
     * CPU mirror read. Scalars return a number; vectors return a live view whose
     * `.x/.y/.z/.w`, `.copy()`, and `.set()` all patch; cpu-only returns the raw stored value.
     */
    get value(): unknown {
        if (this.cpu) return this._raw
        if (this._kind === 'scalar') return this._mirror[0]
        return this._view
    }

    /**
     * CPU mirror write of a FINAL (post-transform) value. Accepts a number, a Vector-like
     * `{x,y,z,w}`, a `d.vecXf` instance, a `VectorFieldView`, a plain array, or (for cpu-only)
     * any raw value. Routes into the coalesced patch. Assigning the same view back
     * (`u.value = u.value`) is a no-op copy that still marks the field dirty, matching the
     * "re-assign to flush" idiom.
     */
    set value(next: unknown) {
        if (this.cpu) {
            this._raw = next
            return
        }
        if (this._kind === 'scalar') {
            this._mirror[0] = toScalar(next)
        } else {
            const c = toComponents(next, this._componentCount)
            for (let i = 0; i < this._componentCount; i++) this._mirror[i] = c[i]
        }
        this._enqueue()
    }

    /**
     * Set from a RAW prop value, applying the prop transform first (the `updateUniformValue`
     * path). CPU-only handles store the raw value untouched.
     */
    setFromRaw(raw: unknown): void {
        if (this.cpu) {
            this._raw = raw
            return
        }
        this.value = this.transform ? this.transform(raw) : raw
    }

    /** Direct component write for vectors (`u.setComponents(x, y)`), avoiding the view. */
    setComponents(...values: number[]): void {
        if (this.cpu || this._kind === 'scalar') {
            if (values.length) this.value = values[0]
            return
        }
        for (let i = 0; i < this._componentCount && i < values.length; i++) this._mirror[i] = values[i]
        this._enqueue()
    }
}

// ---------------------------------------------------------------------------------------
// ArrayFieldHandle — fixed-size array fields (colorStops et al.)
// ---------------------------------------------------------------------------------------

/**
 * Wraps one fixed-size `d.arrayOf(...)` struct field. Whole-array writes (`.array = [...]`)
 * mark the field for a full patch; per-element writes (`.setElement(i, v)`) coalesce into a
 * sparse `patch({ n_x: { colors: { 3: v } } })`.
 *
 * Supports the colorStops flow: `const a = def.colorsArray.array; a[i] = …; def.colorsArray.array = a`
 * (read the live mirror, mutate it, re-assign to flush). The `.array` getter returns the live
 * mirror, and the setter always marks dirty even when the same array reference is re-assigned.
 */
export class ArrayFieldHandle {
    readonly gpuPath: readonly [string, string]
    readonly accessorPath: string
    readonly transform?: (value: unknown) => unknown
    readonly schema: WgslArray
    readonly length: number
    /** Element kind — 'scalar' for `arrayOf(f32,…)` (flat float arrays), else a vec kind. */
    readonly elementKind: FieldKind

    private _mirror: number[]
    private readonly _sink: DirtySink

    constructor(opts: {
        sink: DirtySink
        gpuPath: [string, string]
        schema: WgslArray
        initial?: number[]
        transform?: (value: unknown) => unknown
    }) {
        this._sink = opts.sink
        this.gpuPath = opts.gpuPath
        this.accessorPath = `${opts.gpuPath[0]}.${opts.gpuPath[1]}`
        this.transform = opts.transform
        this.schema = opts.schema
        this.length = opts.schema.elementCount
        this.elementKind = kindOfSchema(opts.schema.elementType as AnyWgslData)
        this._mirror = opts.initial ? opts.initial.slice() : new Array(this.flatLength).fill(0)
    }

    /** Flat mirror length: `elementCount * componentsPerElement`. */
    private get flatLength(): number {
        return this.length * (this.elementKind === 'scalar' ? 1 : componentsOf(this.elementKind))
    }

    /** The live CPU mirror array. Mutate + re-assign via the setter to flush. */
    get array(): number[] {
        return this._mirror
    }
    set array(next: number[]) {
        // Re-assigning the SAME reference must still flush.
        if (next !== this._mirror) this._mirror = next.slice()
        this._sink.markArray(this, 'all')
    }

    /** Coalesced single-element write → sparse patch. */
    setElement(index: number, value: number): void {
        this._mirror[index] = value
        this._sink.markArray(this, index)
    }

    /** Components per element (1 for scalar element arrays). */
    get componentsPerElement(): number {
        return this.elementKind === 'scalar' ? 1 : componentsOf(this.elementKind)
    }

    /**
     * @internal whole-array patch payload. typegpu's dataIO serializes vec-element
     * arrays from PER-ELEMENT tuples, not a flat component array — a flat 32-float
     * array for `arrayOf(vec4f, 8)` writes NaN garbage. The flat number[] mirror is the
     * CPU-side surface; chunk it here.
     */
    _wholeArray(): number[] | number[][] {
        const flat = this._mirror.slice(0, this.flatLength)
        const comps = this.componentsPerElement
        if (comps === 1) return flat
        const out: number[][] = []
        for (let i = 0; i < this.length; i++) out.push(flat.slice(i * comps, (i + 1) * comps))
        return out
    }
    /** @internal single-element patch payload (scalar element arrays; flat index). */
    _elementAt(index: number): number {
        return this._mirror[index]
    }
    /** @internal element tuple at ELEMENT index (vec element arrays). */
    _elementTuple(elemIndex: number): number[] {
        const comps = this.componentsPerElement
        return this._mirror.slice(elemIndex * comps, (elemIndex + 1) * comps)
    }
}

// ---------------------------------------------------------------------------------------
// SystemUniforms (_sys) — default shape; kit/coords.ts can supply the canonical one
// ---------------------------------------------------------------------------------------

/**
 * The `_sys` struct default. `kit/coords.ts` owns the canonical shape and can inject its own
 * via `createUniformStore(root, { systemSchema, systemInitial })`. These fields cover the
 * built-ins the renderer feeds every frame (time, viewport, pointer).
 */
export const SystemUniforms = d.struct({
    time: d.f32,
    deltaTime: d.f32,
    frame: d.f32,
    viewportSize: d.vec2f,
    logicalViewportSize: d.vec2f,
    aspect: d.f32,
    pointer: d.vec2f,
    pointerActive: d.f32,
})

const SYSTEM_DEFAULTS: Record<string, unknown> = {
    time: 0,
    deltaTime: 0,
    frame: 0,
    viewportSize: [1, 1],
    logicalViewportSize: [1, 1],
    aspect: 1,
    pointer: [0.5, 0.5],
    pointerActive: 0,
}

// ---------------------------------------------------------------------------------------
// createUniformStore
// ---------------------------------------------------------------------------------------

/** One field to register on a node (or the system struct). */
export interface FieldInit {
    /** WGSL-safe field name (prop name). */
    name: string
    /**
     * The field's WGSL schema (`d.f32`, `d.vec2f`, `d.vec4f`, `d.arrayOf(...)`, …). Omit to
     * infer from `initial` (see `inferFieldSchema`); pass `cpu: true` for string/JSON props
     * that must not enter the struct.
     */
    schema?: AnyWgslData
    /** Initial FINAL value (already transformed) OR raw value if `transform` is given. */
    initial: unknown
    /** Optional prop transform, retained on the handle for `updateUniformValue`-style writes. */
    transform?: (value: unknown) => unknown
    /** Marks a CPU-only prop (string/JSON/origin) — mirror-only, no struct field. */
    cpu?: boolean
}

export interface FinalizeResult {
    schema: WgslStruct
    layout: TgpuBindGroupLayout
    buffer: {patch: (partial: unknown) => void; write: (data: unknown) => void; destroy?: () => void}
    bindGroup: TgpuBindGroup
    uniformEntryKey: string
}

export interface UniformStoreOptions {
    /** Override the `_sys` struct schema (kit/coords supplies the canonical one). */
    systemSchema?: WgslStruct
    /** Initial values for the system fields (defaults cover the built-in shape). */
    systemInitial?: Record<string, unknown>
}

/** Handles for a single node's fields, keyed by prop name. */
export type NodeHandles = Record<string, FieldHandle | ArrayFieldHandle>

export interface UniformStore {
    /** Register a node's GPU-relevant props; returns its FieldHandles keyed by prop name. */
    /**
     * Register a node's fields. `key` names its struct member (default `nodeKey(rawId)`); the
     * renderer passes a POSITIONAL key (`n_0`, `n_1`, … in tree order) so two compositions with
     * the same structure generate identical WGSL whatever their layer ids (shared compile).
     */
    defineNode(rawId: string, fields: FieldInit[], key?: string): NodeHandles
    /** The struct member a node's fields live under (see defineNode). */
    keyOf(rawId: string): string
    /** Register the `_sys` system fields; returns their handles keyed by field name. */
    defineSystem(fields?: FieldInit[]): NodeHandles
    /**
     * Assemble the combined struct, create the uniform buffer + bind group, and write initial
     * values. Idempotent — returns the cached result on subsequent calls.
     */
    finalize(): FinalizeResult
    /** Coalesced flush — one `buffer.patch(...)` for every field touched since the last flush. */
    flush(): void
    /** Whole-buffer write fallback (`buffer.write(...)`). */
    writeAll(): void
    /** Build the GPU accessor string for glue WGSL, e.g. `layout.$.uniforms.n_x3.speed`. */
    gpuAccessor(handle: FieldHandle | ArrayFieldHandle, layoutVar?: string): string
    /** The combined struct (available after `finalize`). */
    readonly schema?: WgslStruct
    readonly layout?: TgpuBindGroupLayout
    readonly buffer?: FinalizeResult['buffer']
    readonly bindGroup?: TgpuBindGroup
    /** The `_sys` handles (after `defineSystem`). */
    readonly systemHandles: NodeHandles
    destroy(): void
}

/**
 * Create a per-composition uniform store around a TgpuRoot. Register nodes + system fields,
 * then `finalize()` to build the buffer/layout/bind group. Drives the coalesced per-frame
 * patch (`flush()`).
 */
export function createUniformStore(root: TgpuRoot, options: UniformStoreOptions = {}): UniformStore {
    const systemSchema = options.systemSchema ?? SystemUniforms
    const systemInitial = {...SYSTEM_DEFAULTS, ...(options.systemInitial ?? {})}

    // Per-node struct schemas + handle collections, in registration order (== struct member order).
    const nodeStructs = new Map<string, WgslStruct>()
    /** Node id → its struct member name (see defineNode). */
    const keysById = new Map<string, string>()
    const nodeHandles = new Map<string, NodeHandles>()

    let systemHandles: NodeHandles = {}
    let systemStruct: WgslStruct = systemSchema
    let systemDefined = false

    // Coalesced dirty tracking.
    const dirtyScalars = new Set<FieldHandle>()
    const dirtyArrays = new Map<ArrayFieldHandle, Set<number> | 'all'>()

    // Hidden `_pad*` field names added per struct key at finalize (strict uniform layout).
    // They have no handles; fullTree() writes them as zeros so whole-buffer writes stay clean.
    const padFields = new Map<string, string[]>()

    const sink: DirtySink = {
        markScalar(handle) {
            dirtyScalars.add(handle)
        },
        markArray(handle, index) {
            if (index === 'all') {
                dirtyArrays.set(handle, 'all')
                return
            }
            const cur = dirtyArrays.get(handle)
            if (cur === 'all') return
            if (cur) cur.add(index)
            else dirtyArrays.set(handle, new Set([index]))
        },
    }

    let finalized: FinalizeResult | undefined

    function buildHandles(key: string, fields: FieldInit[]): {handles: NodeHandles; struct: WgslStruct} {
        const handles: NodeHandles = {}
        const structProps: Record<string, AnyWgslData> = {}
        for (const f of fields) {
            const name = sanitizeFieldId(f.name)
            // Apply the prop transform to the initial value once (so the first frame renders
            // the transformed value).
            const finalInitial = f.transform ? f.transform(f.initial) : f.initial
            const isCpu = f.cpu ?? typeof finalInitial === 'string'
            // An APPLIED transform must yield a packable value. cpu mirrors legitimately
            // hold strings / JSON / ColorStop[], so skip them; warn once for a genuinely wrong transform.
            if (f.transform && !isCpu && !isPackableFieldValue(finalInitial) && !warnedNonPackableTransforms.has(f.transform)) {
                warnedNonPackableTransforms.add(f.transform)
                console.warn(
                    `[gpu] prop transform for "${f.name}" returned a non-packable value — the uniform field will be NaN. ` +
                        `A prop transform must return a number, boolean, vector, or array (or a string for cpu-only props).`,
                    finalInitial,
                )
            }
            if (isCpu) {
                handles[f.name] = new FieldHandle({
                    sink,
                    gpuPath: [key, name],
                    initial: finalInitial,
                    transform: f.transform,
                    cpu: true,
                })
                continue
            }
            // Unwrap the `{ node, data }` transform pair for schema inference.
            const forInfer =
                finalInitial && typeof finalInitial === 'object' && 'data' in (finalInitial as object)
                    ? (finalInitial as {data: unknown}).data
                    : finalInitial
            const schema = f.schema ?? inferFieldSchema(forInfer)
            structProps[name] = schema
            if (isArraySchema(schema)) {
                const h = new ArrayFieldHandle({
                    sink,
                    gpuPath: [key, name],
                    schema,
                    initial: f.initial as number[] | undefined,
                    transform: f.transform,
                })
                handles[f.name] = h
            } else {
                const h = new FieldHandle({
                    sink,
                    gpuPath: [key, name],
                    schema,
                    initial: finalInitial,
                    transform: f.transform,
                })
                handles[f.name] = h
            }
        }
        return {handles, struct: d.struct(structProps)}
    }

    function assertNotFinalized(): void {
        if (finalized) throw new Error('[uniformStore] cannot register fields after finalize()')
    }

    function fullTree(): Record<string, unknown> {
        const tree: Record<string, unknown> = {}
        for (const [key, handles] of nodeHandles) tree[key] = nodeSubtree(handles, key)
        tree._sys = nodeSubtree(systemHandles, '_sys')
        return tree
    }

    function nodeSubtree(handles: NodeHandles, key?: string): Record<string, unknown> {
        const sub: Record<string, unknown> = {}
        for (const h of Object.values(handles)) {
            if (h instanceof ArrayFieldHandle) {
                sub[h.gpuPath[1]] = h._wholeArray()
            } else if (!h.cpu) {
                sub[h.gpuPath[1]] = h._patchValue()
            }
        }
        if (key) for (const pad of padFields.get(key) ?? []) sub[pad] = 0
        return sub
    }

    const store: UniformStore = {
        get schema() {
            return finalized?.schema
        },
        get layout() {
            return finalized?.layout
        },
        get buffer() {
            return finalized?.buffer
        },
        get bindGroup() {
            return finalized?.bindGroup
        },
        get systemHandles() {
            return systemHandles
        },

        defineNode(rawId, fields, explicitKey) {
            assertNotFinalized()
            const key = explicitKey ?? nodeKey(rawId)
            keysById.set(rawId, key)
            const {handles, struct} = buildHandles(key, fields)
            nodeStructs.set(key, struct)
            nodeHandles.set(key, handles)
            return handles
        },

        keyOf(rawId) {
            return keysById.get(rawId) ?? nodeKey(rawId)
        },

        defineSystem(fields) {
            assertNotFinalized()
            if (fields && fields.length) {
                const built = buildHandles('_sys', fields)
                systemHandles = built.handles
                systemStruct = built.struct
            } else {
                // Default: one handle per SystemUniforms member, seeded from SYSTEM_DEFAULTS.
                const defaults: FieldInit[] = Object.entries(systemSchema.propTypes).map(([name, schema]) => ({
                    name,
                    schema: schema as AnyWgslData,
                    initial: systemInitial[name] ?? 0,
                }))
                const built = buildHandles('_sys', defaults)
                systemHandles = built.handles
                systemStruct = built.struct
            }
            systemDefined = true
            return systemHandles
        },

        finalize() {
            if (finalized) return finalized
            if (!systemDefined) this.defineSystem()

            // Strict WGSL uniform layout: without the `uniform_buffer_standard_layout` language
            // extension (Safari, Firefox, Chromium ≲ 150), a struct-typed member of a uniform
            // struct must sit at a 16-byte-aligned offset, with ≥ roundUp(16, sizeOf) bytes
            // before the next member. TypeGPU's natural (storage-style) packing violates that for
            // any node struct whose size isn't a multiple of 16 — pipeline creation fails with
            // "struct member offset must be a multiple of 16 bytes".
            //
            // The fix pads every node struct's SIZE to a multiple of 16 with hidden `_pad*: f32`
            // members, which makes every natural offset a multiple of 16 and the post-struct gap
            // exactly roundUp(16, sizeOf) — valid everywhere, ≤12 pad bytes per node. Padding is
            // deliberately NOT done via `d.align(16, …)`: typegpu's partial-write path
            // (`buffer.patch` → partialIO `collect`) does not unwrap Decorated members, so a
            // decorated node struct degrades to a "leaf" — flush()'s sparse patches then write
            // NaN over sibling fields and crash on vec members ("Cannot read properties of
            // undefined (reading '0')"). Plain padded structs keep the patch path fully sparse.
            const padTo16 = (key: string, struct: WgslStruct): WgslStruct => {
                // A node whose props are ALL cpu-only yields d.struct({}) — invalid WGSL
                // (structs need ≥1 member) and typegpu's sizeOf/alignmentOf throw on it.
                // Give it a full 16-byte pad block so it is resolvable and keeps the
                // "every member size is a multiple of 16" offset invariant.
                if (Object.keys(struct.propTypes).length === 0) {
                    const names = ['_pad0', '_pad1', '_pad2', '_pad3']
                    padFields.set(key, names)
                    return d.struct(Object.fromEntries(names.map((n) => [n, d.f32])))
                }
                let padded = struct
                const padNames: string[] = []
                while (d.sizeOf(padded) % 16 !== 0) {
                    // Never collide with a real field — overwriting one would silently
                    // corrupt its schema and zero it on every writeAll.
                    let name = `_pad${padNames.length}`
                    while (name in struct.propTypes) name = `_${name}`
                    padNames.push(name)
                    const props = {...struct.propTypes} as Record<string, AnyWgslData>
                    for (const n of padNames) props[n] = d.f32
                    padded = d.struct(props)
                }
                if (padNames.length) padFields.set(key, padNames)
                return padded
            }
            const combinedProps: Record<string, AnyWgslData> = {}
            for (const [key, struct] of nodeStructs) combinedProps[key] = padTo16(key, struct)
            combinedProps._sys = padTo16('_sys', systemStruct)
            const combined = d.struct(combinedProps)

            // EVERY node's props live in this ONE struct, bound as a single uniform binding, so
            // the composition's total prop footprint is capped by maxUniformBufferBindingSize —
            // 65536 bytes on virtually all hardware (it's the core minimum and almost nobody
            // exceeds it). A composition over that ceiling can't be bound at all, and the raw
            // symptom is a pile of validation errors and a blank canvas that names no cause.
            // Check it up front and say exactly what happened instead.
            const combinedSize = d.sizeOf(combined)
            const bindingCap = root.device?.limits?.maxUniformBufferBindingSize
            const cap = typeof bindingCap === 'number' && bindingCap > 0 ? bindingCap : 65536
            if (combinedSize > cap) {
                throw new GpuUnavailableError(
                    'limit-exceeded',
                    `[gpu] this composition needs ${combinedSize} bytes of uniform data across ${nodeStructs.size} nodes, ` +
                        `but the device caps a uniform binding at ${cap} bytes. Reduce the number of layers.`,
                )
            }

            const buffer = root.createBuffer(combined).$usage('uniform')
            const layout = tgpu.bindGroupLayout({[UNIFORM_ENTRY_KEY]: {uniform: combined}})
            const bindGroup = root.createBindGroup(layout as never, {[UNIFORM_ENTRY_KEY]: buffer} as never)

            finalized = {
                schema: combined,
                layout: layout as TgpuBindGroupLayout,
                buffer: buffer as unknown as FinalizeResult['buffer'],
                bindGroup: bindGroup as TgpuBindGroup,
                uniformEntryKey: UNIFORM_ENTRY_KEY,
            }
            // Seed the buffer with initial values, then clear the "everything is dirty" state.
            this.writeAll()
            return finalized
        },

        flush() {
            if (!finalized) return
            if (dirtyScalars.size === 0 && dirtyArrays.size === 0) return
            const patch: Record<string, Record<string, unknown>> = {}
            const bucket = (key: string) => (patch[key] ??= {})

            for (const h of dirtyScalars) {
                bucket(h.gpuPath[0])[h.gpuPath[1]] = h._patchValue()
            }
            for (const [h, idxs] of dirtyArrays) {
                const node = bucket(h.gpuPath[0])
                if (idxs === 'all') {
                    node[h.gpuPath[1]] = h._wholeArray()
                } else if (h.componentsPerElement === 1) {
                    const sparse: Record<number, number> = {}
                    for (const i of idxs) sparse[i] = h._elementAt(i)
                    node[h.gpuPath[1]] = sparse
                } else {
                    // Flat component indices from setElement → element-level sparse
                    // tuples: typegpu patches whole elements, never single components.
                    const comps = h.componentsPerElement
                    const sparse: Record<number, number[]> = {}
                    for (const i of idxs) {
                        const el = Math.floor(i / comps)
                        sparse[el] = h._elementTuple(el)
                    }
                    node[h.gpuPath[1]] = sparse
                }
            }

            finalized.buffer.patch(patch)
            dirtyScalars.clear()
            dirtyArrays.clear()
        },

        writeAll() {
            if (!finalized) return
            finalized.buffer.write(fullTree())
            dirtyScalars.clear()
            dirtyArrays.clear()
        },

        gpuAccessor(handle, layoutVar = 'layout') {
            return `${layoutVar}.$.${UNIFORM_ENTRY_KEY}.${handle.accessorPath}`
        },

        destroy() {
            finalized?.buffer.destroy?.()
            finalized = undefined
            nodeStructs.clear()
            nodeHandles.clear()
            dirtyScalars.clear()
            dirtyArrays.clear()
            padFields.clear()
        },
    }

    return store
}

/**
 * Applies the prop transform and writes in-place (in-place semantics live inside the store).
 * Array (colorStops) handles are updated through their own path (their transform reshapes the
 * arrays), so callers route those separately; here we set the whole array when a plain array is
 * given.
 */
export function updateFieldValue(handle: FieldHandle | ArrayFieldHandle, rawValue: unknown): void {
    if (handle instanceof ArrayFieldHandle) {
        if (Array.isArray(rawValue)) handle.array = rawValue as number[]
        return
    }
    handle.setFromRaw(rawValue)
}
