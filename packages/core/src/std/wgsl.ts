/**
 * `wgsl` — write the per-pixel math yourself, as one WGSL function body.
 *
 * Everything else stays declared on the definition (props, the layer's clock, blend modes,
 * masks, transforms, drivers, export). The body sits in `paint:` or `effect:`, returns a
 * `vec4f` color, and refers to the definition's props and to `uv`, `time`, `aspect`,
 * `viewport`, `pointer` and the child by name. Nothing is declared: the names the body uses
 * are bound for it.
 */
// Maintainer notes.
//
// Only the identifiers the body actually references become function parameters; a name that
// is neither a prop nor a context value is left for WGSL itself to resolve (a builtin, a
// local, or an error the compiler reports). Context bindings per host: every host gets
// `uv: vec2f` (y = 0 at the top; a generator wrapped by a UV-propagating distortion receives
// the distorted UV), `time: f32` (the node's speed-scaled clock when `animatedTime` is
// declared, else `ctx.time`), `aspect: f32`, `viewport: vec2f` (device pixels; the effective
// viewport for generators) and `pointer: vec2f`. Pointwise filters add `child: vec4f`
// (straight alpha); gather filters add `childTexture` + `childSampler` (PREMULTIPLIED) and
// the species unpremultiplies the returned color unless `alpha: 'straight'`. Species is
// inferred from a `childTexture` reference unless declared.
//
// Prop typing mirrors the uniform bridge (`wgslTypeForProp`): colors vec4f (linear RGB +
// alpha), positions vec2f (flipped back to authored y before the body sees them), numbers and
// booleans f32, select props the transformed numeric value; CPU-only props (URL strings, shape
// objects, color-stops arrays, lists) are not bindable.
//
// Lowering: the body becomes the string form of a `tgpu.fn` (typed shell + WGSL text) and is
// invoked through `call(...)` like every kit primitive, so it composes at runtime with no
// transpiler: the words the body sits beside are precompiled, and WGSL text needs none.
import {tgpu, d} from '../gpu/kit/index'
import {call, expr, floatE} from '../gpu/composer'
import {Expr, type GpuFragmentParams} from '../gpu/contract'
import type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import type {RttFilterParams} from '../gpu/scaffolds/rttFilter'
import {animatedTime} from '../gpu/porters'
import {gpuTransformFor} from '../gpu/transforms'
import {inferFieldSchema} from '../gpu/uniformStore'
import {markCustomWgslUsed} from '../gpu/support'
import type {PropConfig} from '../types'
import {transformColor, transformPosition} from '../utilities/transformations'
import {colorStopsTransform} from '../utilities/colorStops'
import {listPropTransform} from '../utilities/listProps'
import {isDimensionalValue} from '../utilities/dimensionalProps'
import {Scalar} from './values'
import {resolveArgIn, type ArgSpec} from './invoke'

// ── Public types ────────────────────────────────────────────────────────────────────────

/** The WGSL types a bound name can have inside a body. */
export type WgslType = 'f32' | 'vec2f' | 'vec3f' | 'vec4f' | 'texture_2d<f32>' | 'sampler'

/**
 * One entry of `inputs`: the value to bind under a name — a prop `p('x')`, a context token
 * `ctx.time`, a `Scalar`, or a number — optionally with its WGSL type spelled out as
 * `{value, type}`. A prop takes its own type; a `Scalar` or a number is `f32`.
 */
export type WgslInput = ArgSpec | {readonly value: ArgSpec; readonly type: WgslType}

/** The long form of a body: `wgsl({body, inputs, alpha, name})`. */
export interface WgslSpec {
    /** The WGSL statements. Must `return` a `vec4f` color. */
    readonly body: string
    /** Extra names to bind, on top of the props and context values bound automatically. */
    readonly inputs?: Record<string, WgslInput>
    /**
     * Gather filters only: what alpha the returned color is in. `'premultiplied'` (default)
     * lets the engine unpremultiply it for you; `'straight'` returns it as-is.
     */
    readonly alpha?: 'premultiplied' | 'straight'
    /** A readable name for the generated WGSL function. Defaults to the definition's name. */
    readonly name?: string
}

/** What `wgsl` returns: a body ready to sit in `paint:` or `effect:`. */
export class WgslBody {
    readonly kind = 'wgsl' as const
    readonly spec: WgslSpec
    /** The names the body refers to (comments stripped, member accesses excluded). */
    readonly identifiers: ReadonlySet<string>
    /** Lowered `tgpu.fn`s, one per distinct parameter signature. */
    private readonly fns = new Map<string, unknown>()

    constructor(spec: WgslSpec) {
        if (typeof spec.body !== 'string' || spec.body.trim() === '') {
            throw new Error('wgsl: the body must be a non-empty WGSL string')
        }
        if (!/\breturn\b/.test(stripComments(spec.body))) {
            throw new Error('wgsl: the body must `return` a vec4f color')
        }
        this.spec = spec
        this.identifiers = scanIdentifiers(spec.body)
    }

    /** True when the body samples `childTexture`, which makes it a gather filter. */
    get samplesChild(): boolean {
        return this.identifiers.has('childTexture')
    }

    /** @internal — the `tgpu.fn` for a parameter signature, created once per body. */
    fnFor(signature: string, make: () => unknown): unknown {
        let fn = this.fns.get(signature)
        if (!fn) {
            fn = make()
            this.fns.set(signature, fn)
        }
        return fn
    }
}

/**
 * True when a value came from `wgsl`.
 *
 * @internal
 */
export function isWgslBody(value: unknown): value is WgslBody {
    return value instanceof WgslBody
}

/**
 * Write the per-pixel math by hand, as one WGSL function body that returns a `vec4f` color.
 *
 * Put it in `paint:` for a generator or `effect:` for a filter. Every name the body uses is
 * bound for you. The definition's props come by name: colors as `vec4f` in linear RGB with
 * alpha, positions as `vec2f` in uv, numbers and booleans as `f32`, select props as their
 * numeric value. The context comes as `uv` (0–1 across the canvas, y down), `time` (seconds;
 * the layer's own clock when `animatedTime` is declared, paused at speed 0), `aspect` (width
 * / height), `viewport` (the frame size in device pixels) and `pointer` (the mouse in uv). In
 * an `effect:` body `child` is the color of the layer inside at this pixel. Sample
 * `childTexture` with `childSampler` instead when you need neighbouring pixels (a blur, a
 * displacement): that gives the filter its own render pass and hands you premultiplied
 * color, which is unpremultiplied for you on the way out. Template values (`${OCTAVES}`) are
 * pasted in as text, so a constant can be a loop bound.
 *
 * Use the `{body, inputs, alpha}` form to bind extra names (`inputs: {t: ctx.time, k: 4}`) or
 * to say a gather body already returns straight alpha (`alpha: 'straight'`).
 *
 * @example
 * ```ts
 * paint: wgsl`
 *   let d = length((uv - center) * vec2f(aspect, 1.0)) / radius;
 *   return vec4f(mix(outer.rgb, inner.rgb, 1.0 - smoothstep(0.0, 1.0, d)), 1.0);
 * `
 * ```
 * @example
 * ```ts
 * // A gather filter: the layer inside, sampled at a rippled coordinate.
 * effect: wgsl`
 *   let wave = sin(length(uv - center) * frequency - time * 4.0) * amplitude;
 *   return textureSample(childTexture, childSampler, uv + wave);
 * `
 * ```
 * @tip The names `uv`, `time`, `aspect`, `viewport`, `pointer`, `child`, `childTexture` and `childSampler` belong to the body; a prop spelled the same is not bound, so rename the prop.
 * @see defineShader, ctx, p
 */
// Three spellings: a tagged template (values inlined as text), a plain string, or a
// `WgslSpec` object. A prop named `viewportSize` or `logicalViewportSize` is also skipped
// (those are context token names), see `CTX_TYPES` / `RESERVED` below.
export function wgsl(spec: WgslSpec): WgslBody
export function wgsl(body: string): WgslBody
export function wgsl(strings: TemplateStringsArray, ...values: unknown[]): WgslBody
export function wgsl(first: WgslSpec | string | TemplateStringsArray, ...values: unknown[]): WgslBody {
    if (typeof first === 'string') return new WgslBody({body: first})
    if (isTemplateStrings(first)) {
        let body = ''
        first.forEach((chunk, i) => {
            body += chunk
            if (i < values.length) body += formatInlineValue(values[i])
        })
        return new WgslBody({body})
    }
    return new WgslBody(first)
}

function isTemplateStrings(value: unknown): value is TemplateStringsArray {
    return Array.isArray(value) && 'raw' in (value as object)
}

/**
 * A template value spliced into the body as text. Numbers keep their JS spelling: an integer
 * stays an abstract-int literal (`3`), which WGSL converts to f32 or i32 from context, so it
 * works as a loop bound AND a float operand; a fraction stays a float literal.
 */
function formatInlineValue(value: unknown): string {
    if (value instanceof WgslBody) throw new Error('wgsl: a body cannot be spliced into another body')
    return String(value)
}

// ── Identifier scan ─────────────────────────────────────────────────────────────────────

function stripComments(source: string): string {
    return source.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ')
}

/**
 * @internal The identifiers a body references. Member accesses (`color.rgb` → `rgb`) are
 * skipped by the negative lookbehind, so a prop named like a swizzle is never bound by
 * accident.
 */
export function scanIdentifiers(source: string): ReadonlySet<string> {
    const out = new Set<string>()
    const re = /(?<![\w.])[A-Za-z_][A-Za-z0-9_]*/g
    for (const m of stripComments(source).matchAll(re)) out.add(m[0])
    return out
}

// ── Prop typing ─────────────────────────────────────────────────────────────────────────

/**
 * @internal The WGSL type a prop binds as, from its config — mirroring the uniform bridge's
 * packing rules so the parameter type always matches the struct field. `null` for a prop
 * that is CPU-only (never a struct field) or expands to arrays (color stops / lists).
 */
export function wgslTypeForProp(config: PropConfig<unknown>): WgslType | null {
    const transform = config.transform as ((value: unknown) => unknown) | undefined
    if (transform === (transformColor as unknown)) return 'vec4f'
    if (transform === (transformPosition as unknown)) return 'vec2f'
    if (transform === (colorStopsTransform as unknown) || transform === (listPropTransform as unknown)) return null
    const def = config.default
    if (isDimensionalValue(def)) return 'f32'
    const gpuTransform = gpuTransformFor(transform as ((value: never) => unknown) | undefined)
    const cpuOnly =
        !gpuTransform &&
        (typeof def === 'string' || (typeof def === 'object' && def !== null && !Array.isArray(def)))
    if (cpuOnly) return null
    let value: unknown = def
    if (gpuTransform) {
        try {
            value = gpuTransform(def)
        } catch {
            value = def
        }
    }
    if (value && typeof value === 'object' && 'data' in (value as object)) value = (value as {data: unknown}).data
    const schema = inferFieldSchema(value) as {type?: string}
    switch (schema.type) {
        case 'f32':
            return 'f32'
        case 'vec2f':
            return 'vec2f'
        case 'vec3f':
            return 'vec3f'
        case 'vec4f':
            return 'vec4f'
        default:
            return null
    }
}

// ── Lowering ────────────────────────────────────────────────────────────────────────────

/** Where a body runs: a generator's `paint:`, or a pointwise or gather filter's `effect:`. */
export type WgslHost = 'generator' | 'pointwise' | 'gather'

interface Binding {
    name: string
    type: WgslType
    value: (params: GpuFragmentParams) => Expr
}

const SCHEMAS: Record<WgslType, unknown> = {
    f32: d.f32,
    vec2f: d.vec2f,
    vec3f: d.vec3f,
    vec4f: d.vec4f,
    'texture_2d<f32>': d.texture2d(d.f32),
    sampler: d.sampler(),
}

/** Names the host owns; a prop spelled the same is not bindable (the context value wins). */
const RESERVED = new Set(['viewport', 'child', 'childTexture', 'childSampler'])

const CTX_TYPES: Record<string, WgslType> = {
    uv: 'vec2f',
    aspect: 'f32',
    time: 'f32',
    viewportSize: 'vec2f',
    logicalViewportSize: 'vec2f',
    pointer: 'vec2f',
}

/** The parameter name a generated function uses for a binding (WGSL-safe). */
function paramName(name: string): string {
    return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : name.replace(/[^A-Za-z0-9_]/g, '_')
}

/**
 * Lower a body for a host. Returns the composition-time builder: it selects the bindings
 * the body references, builds (once per signature) the string-form `tgpu.fn`, and returns
 * the `call(...)` Expr.
 */
export function lowerWgsl(
    body: WgslBody,
    host: WgslHost,
    definition: {name: string; props: Record<string, PropConfig<unknown>>; animatedTime?: {speed: string}},
): (params: GpuFragmentParams) => Expr {
    const hint = paramName(body.spec.name ?? lowerFirst(definition.name) + hostSuffix(host))
    const automatic = automaticBindings(host, definition)
    const explicit = explicitBindings(body.spec.inputs ?? {}, definition.props)

    // Explicit inputs are always passed; automatic ones only when referenced.
    const byName = new Map<string, Binding>()
    for (const b of automatic) if (body.identifiers.has(b.name)) byName.set(b.name, b)
    for (const b of explicit) byName.set(b.name, b)
    const bindings = [...byName.values()]

    const header = `(${bindings.map((b) => `${paramName(b.name)}: ${b.type}`).join(', ')}) -> vec4f`
    const signature = bindings.map((b) => `${b.name}:${b.type}`).join(',')

    return (params: GpuFragmentParams): Expr => {
        markCustomWgslUsed()
        params.noteCustomWgsl?.()
        const fn = body.fnFor(signature, () => {
            const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => (impl: string) => unknown)(
                bindings.map((b) => SCHEMAS[b.type]),
                d.vec4f,
            )
            return shell(`${header} {\n${body.spec.body}\n}`)
        })
        return call(fn, hint, bindings.map((b) => b.value(params)))
    }
}

function lowerFirst(name: string): string {
    return name ? name[0].toLowerCase() + name.slice(1) : 'custom'
}

function hostSuffix(host: WgslHost): string {
    return host === 'generator' ? 'Paint' : 'Filter'
}

function automaticBindings(
    host: WgslHost,
    definition: {props: Record<string, PropConfig<unknown>>; animatedTime?: {speed: string}},
): Binding[] {
    const out: Binding[] = []
    const generator = host === 'generator'
    out.push({name: 'uv', type: 'vec2f', value: (p) => (generator ? (p.uvContext ?? p.ctx.uv) : p.ctx.uv)})
    out.push({
        name: 'time',
        type: 'f32',
        value: (p) => (definition.animatedTime ? animatedTime(p) : p.ctx.time),
    })
    out.push({name: 'aspect', type: 'f32', value: (p) => p.ctx.aspect})
    out.push({
        name: 'viewport',
        type: 'vec2f',
        value: (p) => (generator ? (p.effectiveViewportSize ?? p.ctx.viewportSize) : p.ctx.viewportSize),
    })
    out.push({name: 'pointer', type: 'vec2f', value: (p) => p.ctx.pointer})
    if (host === 'pointwise') {
        out.push({name: 'child', type: 'vec4f', value: (p) => (p as FilterParams).childNode})
    }
    if (host === 'gather') {
        out.push({name: 'childTexture', type: 'texture_2d<f32>', value: (p) => (p as RttFilterParams).texture.accessor()})
        // `samp.$.<name>` is the composer's raw sampler accessor; the entry builder scans the
        // emitted call site for it and binds the shared sampler layout to this pass.
        out.push({name: 'childSampler', type: 'sampler', value: () => expr('samp.$.linearClamp')})
    }
    for (const [name, config] of Object.entries(definition.props)) {
        if (CTX_TYPES[name] || RESERVED.has(name)) continue
        const type = wgslTypeForProp(config)
        if (!type) continue
        // A position prop is STORED as `(x, 1 - y)` for the kit's words (the double-flip
        // convention). The body gets it back in `uv` space — authored y, 0 at the top — so
        // `distance(uv, center)` means what it says.
        const isPosition = config.transform === (transformPosition as unknown)
        out.push({
            name,
            type,
            value: (p) => {
                const accessor = p.uniforms[name]
                if (!accessor) throw new Error(`wgsl: prop '${name}' has no GPU uniform (is it CPU-only?)`)
                return isPosition ? new Expr((ctx) => `vec2f(${accessor.member('x')._emit(ctx)}, 1.0 - ${accessor.member('y')._emit(ctx)})`) : accessor
            },
        })
    }
    return out
}

function explicitBindings(inputs: Record<string, WgslInput>, props: Record<string, PropConfig<unknown>>): Binding[] {
    const out: Binding[] = []
    for (const [name, input] of Object.entries(inputs)) {
        const typed = input !== null && typeof input === 'object' && 'value' in (input as object) && 'type' in (input as object)
        const spec = (typed ? (input as {value: ArgSpec}).value : input) as ArgSpec
        const inferred = inferArgType(spec, props)
        const type = typed ? (input as {type: WgslType}).type : inferred
        // A declared type may only restate what the source IS: a prop's packed type, a context
        // value's type, f32 for a literal or a scalar graph. Anything else would emit a call
        // whose argument and parameter disagree — a WGSL error naming neither.
        if (typed && type !== inferred) {
            throw new Error(`wgsl: input '${name}' is declared ${type} but its source is ${inferred}`)
        }
        out.push({
            name,
            type,
            value: (p) => (typeof spec === 'number' ? floatE(spec) : resolveArgIn(spec, p)),
        })
    }
    return out
}

function inferArgType(spec: ArgSpec, props: Record<string, PropConfig<unknown>>): WgslType {
    if (typeof spec === 'number' || spec instanceof Scalar) return 'f32'
    if (spec.kind === 'ctx') return CTX_TYPES[spec.name] ?? 'f32'
    if (spec.kind === 'prop') {
        const config = props[spec.name]
        if (!config) throw new Error(`wgsl: input binds unknown prop '${spec.name}'`)
        const type = wgslTypeForProp(config)
        if (!type) throw new Error(`wgsl: prop '${spec.name}' is CPU-only and cannot be bound as a WGSL parameter`)
        return type
    }
    return 'f32'
}
