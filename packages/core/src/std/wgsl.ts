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
// booleans f32, select props the transformed numeric value. Images, videos, layers, colour
// stops and lists are bound through wgslResources.ts (textures, callables, lane arrays); the
// `surface` host (a material on a picked shape) adds the shape field. Other CPU-only props
// (free strings, shape objects) are not bindable.
//
// Lowering: the body becomes the string form of a `tgpu.fn` (typed shell + WGSL text) and is
// invoked through `call(...)` like every kit primitive, so it composes at runtime with no
// transpiler: the words the body sits beside are precompiled, and WGSL text needs none.
import {tgpu, d} from '../gpu/kit/index'
import {asLocal, call, expr, floatE} from '../gpu/composer'
import {Expr, type GpuFragmentParams} from '../gpu/contract'
import type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import type {RttFilterParams} from '../gpu/scaffolds/rttFilter'
import {animatedTime} from '../gpu/porters'
import {gpuTransformFor} from '../gpu/transforms'
import {inferFieldSchema} from '../gpu/uniformStore'
import {markCustomWgslUsed} from '../gpu/support'
import type {SurfaceFrame} from './paint/materials'
import type {PortableFieldSampler} from '../gpu/kit/sdf3d'
import {arraySchema, isResourceProp, resourcePlan, sampleStraight, type HostCoords, type ResourceCallable} from './wgslResources'
import type {PropConfig} from '../types'
import {transformColor, transformPosition} from '../utilities/transformations'
import {colorStopsTransform} from '../utilities/colorStops'
import {listPropTransform} from '../utilities/listProps'
import {isDimensionalValue} from '../utilities/dimensionalProps'
import {Scalar} from './values'
import {resolveArgIn, type ArgSpec} from './invoke'
import type {KitTexture} from '../gpu/contract'
import {appendCallArgs, parseWgslDecls, identifiersIn, localsIn, topLevelLocalsIn, structSchemas, type WgslDecls, type WgslFnDecl, type WgslStructDecl} from './wgslSource'
import {inlineConsts, linkLibFns, libStructSchemas, orderFns, stringFn} from './wgslLibrary'
import type {InputPlan} from './inputs/graph'
import {frameTracker} from './inputs/frame'
import {premultipliedKeys, selfOutput} from './inputs/self'

// ── Public types ────────────────────────────────────────────────────────────────────────

/** The WGSL types a bound name can have inside a body. */
export type WgslType = 'f32' | 'vec2f' | 'vec3f' | 'vec4f' | 'texture_2d<f32>' | 'texture_external' | 'sampler' | `array<vec4f, ${number}>` | `array<vec3f, ${number}>`

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

/**
 * Where a body runs: a generator's `paint:`, a pointwise or gather filter's `effect:`, or a
 * shape material's `surface:`.
 */
export type WgslHost = 'generator' | 'pointwise' | 'gather' | 'surface'

interface Binding {
    name: string
    type: WgslType
    value: (params: GpuFragmentParams, frame?: SurfaceFrame) => Expr
}

const SCHEMAS: Record<string, unknown> = {
    f32: d.f32,
    vec2f: d.vec2f,
    vec3f: d.vec3f,
    vec4f: d.vec4f,
    'texture_2d<f32>': d.texture2d(d.f32),
    texture_external: d.textureExternal(),
    sampler: d.sampler(),
}

function schemaFor(type: WgslType): unknown {
    const schema = SCHEMAS[type] ?? arraySchema(type)
    if (!schema) throw new Error(`wgsl: no schema for binding type ${type}`)
    return schema
}

/** Names the host owns; a prop spelled the same is not bindable (the context value wins). */
const RESERVED = new Set(['viewport', 'dt', 'frame', 'pointerVelocity', 'pointerDown', 'idle', 'pointerOver', 'child', 'childTexture', 'childSampler', 'shape', 'shapeDistance', 'shapeSlope', 'shapeUV', 'shapeDepth', 'shapeField', 'shapeAt'])

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

/** `name(` as a call: not a member access, not part of a longer identifier. */
function callPattern(name: string, flags = ''): RegExp {
    return new RegExp(`(?<![\\w.])${name}\\s*\\(`, flags)
}

/** `name` used as a value: an identifier occurrence not followed by `(`. */
function bareUse(name: string): RegExp {
    return new RegExp(`(?<![\\w.])${name}\\b(?!\\s*\\()`)
}

type StringFnShell = (impl: string) => {$uses: (deps: Record<string, unknown>) => unknown}

/**
 * Lower a body for a host. Returns the composition-time builder: it selects the bindings
 * the body references, builds (once per signature) the string-form `tgpu.fn`, and returns
 * the `call(...)` Expr. A `surface` host also takes the shape frame.
 *
 * Props the body calls like functions (`logo(p)`, `ramp(t)`, see wgslResources.ts) are
 * rewritten to calls of library helpers, linked into the body's fn through `$uses`.
 */
export function lowerWgsl(
    body: WgslBody,
    host: WgslHost,
    definition: WgslHostDefinition,
): (params: GpuFragmentParams, frame?: SurfaceFrame) => Expr {
    if (!/\breturn\b/.test(stripComments(body.spec.body))) {
        throw new Error('wgsl: the body must `return` a vec4f color')
    }
    const hint = paramName(body.spec.name ?? lowerFirst(definition.name) + hostSuffix(host))
    const plan = definition.inputs
    const reads = plan?.reads ?? []
    // `field.slope(p)` is not WGSL: rewrite named-output calls to plain names first.
    const fixDots = (t: string): string => reads.filter((r) => r.call.includes('.')).reduce(
        (acc, r) => acc.replace(new RegExp(`(?<![\\w.])${r.call.replace('.', '\\s*\\.\\s*')}\\s*\\(`, 'g'), `${readName(r.call)}(`), t)
    const helperDecls = definition.helpers ? parseWgslDecls(fixDots(definition.helpers)) : null
    const sourceText = fixDots(body.spec.body)
    const {automatic, callables, bareOnly} = automaticBindings(host, definition)
    const explicit = explicitBindings(body.spec.inputs ?? {}, definition.props)
    const allBindings = new Map<string, Binding>()
    for (const b of automatic) allBindings.set(b.name, b)
    for (const b of explicit) allBindings.set(b.name, b)
    const code = stripComments(sourceText)
    const helperCode = helperDecls ? helperDecls.fns.map((f) => stripComments(f.body)).join('\n') : ''
    const called = callables.filter((c) => callPattern(c.name).test(code) || (helperCode && callPattern(c.name).test(helperCode)))
    const calledNames = new Set(called.map((c) => c.name))
    const automaticByName = new Map(automatic.map((b) => [b.name, b]))
    const identifiers = scanIdentifiers(sourceText)
    // A name the body declares at its own top level is the body's: don't hand it in as well
    // (WGSL would refuse the pair as a redeclaration). Old bodies wrote `let dt = …` before `dt`
    // was a built-in, and must keep working. Declarations inside a block shadow legally.
    const owned = topLevelLocalsIn(sourceText)

    // Explicit inputs are always passed; automatic ones only when referenced (a callable-only
    // name is not bound as a value), plus the leading arguments of every prop that is called.
    const byName = new Map<string, Binding>()
    for (const b of automatic) {
        if (!identifiers.has(b.name) || owned.has(b.name)) continue
        if (bareOnly.has(b.name) && calledNames.has(b.name) && !bareUse(b.name).test(code)) continue
        byName.set(b.name, b)
    }
    for (const c of called) for (const lead of c.leading) {
        if (callPattern(c.name).test(code)) byName.set(lead, automaticByName.get(lead)!)
    }
    for (const b of explicit) byName.set(b.name, b)

    const rewriteCalls = (t: string): string => {
        let out = t
        for (const c of called) out = out.replace(callPattern(c.name, 'g'), `${c.alias}(${c.leading.map(paramName).join(', ')}, `)
        return out
    }

    // Helpers: the component's own WGSL functions. Each one that reads a bound name (a prop, a
    // built-in, an input) gets it as an extra trailing parameter, threaded through every call.
    const helpers = helperDecls ? planHelpers(helperDecls, rewriteCalls, allBindings) : null
    let text = rewriteCalls(sourceText)
    if (helpers) {
        for (const [fnName, needs] of helpers.needs) {
            if (needs.length === 0) continue
            text = appendCallArgs(text, fnName, needs)
            for (const n of needs) byName.set(n, allBindings.get(n)!)
        }
        text = inlineConsts(text, helpers.consts)
    }
    const bindings = [...byName.values()]
    // A shape material may sample its shape anywhere: `shapeAt(p)`. The field's inputs depend on
    // the shape path (analytic / SVG / marched 3D), so they are bound per composition below.
    const callsShapeAt = host === 'surface' && callPattern(SHAPE_AT).test(code)
    const libExternals = plan?.lib.length ? linkLibFns(plan.lib) : {}
    const libStructs = plan?.lib.length ? libStructSchemas(plan.lib) : new Map<string, unknown>()
    const builtinsUsed = BUILTIN_FIELDS.filter((f) => byName.has(f))
    const recordsSelf = plan?.inputs.some((i) => i.node.kind === 'previousFrame') ?? false

    return (params: GpuFragmentParams, frame?: SurfaceFrame): Expr => {
        markCustomWgslUsed()
        params.noteCustomWgsl?.()
        if (builtinsUsed.length) writeBuiltins(params, definition, builtinsUsed)
        const linked = called.map((c) => ({alias: c.alias, ...c.helper(params)}))
        let all = bindings
        let bodyText = text
        if (callsShapeAt) {
            const portable = frame?.portable
            if (!portable) throw new Error('wgsl: shapeAt(p) is only available in a shape material')
            const names = portable.args.map((_, i) => `${SHAPE_AT_ARG}${i}`)
            all = [
                ...bindings,
                ...portable.args.map((a, i): Binding => ({name: names[i], type: a.type, value: () => a.value})),
                {name: SHAPE_AT_SCALE, type: 'f32', value: (p) => p.uniforms.scale},
            ]
            bodyText = bodyText.replace(callPattern(SHAPE_AT, 'g'), `${SHAPE_AT_ALIAS}(${[...names, SHAPE_AT_SCALE].join(', ')}, `)
            linked.push({alias: SHAPE_AT_ALIAS, ...shapeAtHelper(portable, names)})
        }
        const header = `(${all.map((b) => `${paramName(b.name)}: ${b.type}`).join(', ')}) -> vec4f`
        const signature = all.map((b) => `${b.name}:${b.type}`).join(',')
        const key = signature + linked.map((l) => `|${l.alias}=${l.key}`).join('')
        const fn = body.fnFor(key, () => {
            const scope = new Map<string, unknown>([...Object.entries(libExternals), ...linked.map((l) => [l.alias, l.fn] as [string, unknown])])
            const helperFns = helpers ? buildHelpers(helpers, scope, libStructs) : {}
            const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => StringFnShell)(
                all.map((b) => schemaFor(b.type)),
                d.vec4f,
            )
            const made = shell(`${header} {\n${bodyText}\n}`)
            const uses: Record<string, unknown> = {
                ...Object.fromEntries(linked.map((l) => [l.alias, l.fn])),
                ...libExternals,
                ...helperFns,
            }
            return Object.keys(uses).length ? made.$uses(uses) : made
        })
        const color = call(fn, hint, all.map((b) => b.value(params, frame)))
        if (!recordsSelf) return color
        // The layer reads its own last frame: render the body to a texture, remember the
        // boundary so the history copy finds it, and show that texture.
        const tex = params.convertToTexture(color)
        selfOutput(params).key = tex.key
        return tex.sample(hostCoords(host).uv(params))
    }
}

/** What a body's lowering needs from its definition. */
export interface WgslHostDefinition {
    name: string
    props: Record<string, PropConfig<unknown>>
    animatedTime?: {speed: string}
    /** The definition's planned `inputs` (textures it reads by name, library functions). */
    inputs?: InputPlan
    /** The definition's `helpers` WGSL. */
    helpers?: string | null
}

/** The name a body calls a named input output by after `field.slope(` is rewritten. */
function readName(call: string): string {
    return call.replace('.', '__')
}

// ── Helpers ─────────────────────────────────────────────────────────────────────────────

interface HelperPlan {
    fns: WgslFnDecl[]
    structs: readonly WgslStructDecl[]
    consts: readonly {name: string; value: string}[]
    /** Bound names each helper needs as extra parameters, in a stable order. */
    needs: Map<string, string[]>
    /** Helper bodies after callable rewriting and call-site threading. */
    bodies: Map<string, string>
    bindings: Map<string, Binding>
}

function planHelpers(decls: WgslDecls, rewriteCalls: (t: string) => string, bindings: Map<string, Binding>): HelperPlan {
    const fns = orderFns(decls.fns)
    const names = new Set(fns.map((f) => f.name))
    for (const f of fns) {
        if (bindings.has(f.name)) throw new Error(`helpers: fn ${f.name} has the same name as a prop or built-in; rename it`)
    }
    const rewritten = new Map(fns.map((f) => [f.name, rewriteCalls(f.body)]))
    const direct = new Map<string, Set<string>>()
    for (const f of fns) {
        // A helper's own names (parameters and locals) shadow a built-in or prop of the same name,
        // so they never make it a parameter (`var uv` inside a helper is its own uv).
        const own = new Set([...f.params.map((p) => p.name), ...localsIn(f.body)])
        const ids = identifiersIn(rewritten.get(f.name)!)
        direct.set(f.name, new Set([...ids].filter((id) => bindings.has(id) && !own.has(id))))
    }
    // A helper that calls another inherits what the callee needs (fns are ordered callee-first).
    const needs = new Map<string, string[]>()
    for (const f of fns) {
        const set = new Set(direct.get(f.name))
        for (const id of identifiersIn(rewritten.get(f.name)!)) {
            if (names.has(id) && id !== f.name) for (const n of needs.get(id) ?? []) set.add(n)
        }
        needs.set(f.name, [...set].sort())
    }
    const bodies = new Map<string, string>()
    for (const f of fns) {
        let b = rewritten.get(f.name)!
        for (const g of fns) if (g !== f) b = appendCallArgs(b, g.name, needs.get(g.name)!)
        bodies.set(f.name, b)
    }
    return {fns, structs: decls.structs, consts: decls.consts, needs, bodies, bindings}
}

function buildHelpers(plan: HelperPlan, scope: Map<string, unknown>, libStructs: Map<string, unknown>): Record<string, unknown> {
    const structs = structSchemas(plan.structs, libStructs)
    const out: Record<string, unknown> = {}
    for (const [name, schema] of structs) if (!libStructs.has(name)) out[name] = schema
    const local = new Map(scope)
    for (const [name, schema] of structs) local.set(name, schema)
    for (const fn of plan.fns) {
        const extra = plan.needs.get(fn.name)!.map((n) => {
            const b = plan.bindings.get(n)!
            return {name: paramName(n), type: b.type, schema: schemaFor(b.type)}
        })
        // The planned body (calls rewritten, built-ins appended) with the helper consts inlined:
        // stringFn's own inlining applies to the raw body, which this replaces.
        const made = stringFn(fn, plan.consts, structs, local, extra, () => inlineConsts(plan.bodies.get(fn.name)!, plan.consts))
        local.set(fn.name, made)
        out[fn.name] = made
    }
    return out
}

// ── Built-ins backed by per-frame fields ─────────────────────────────────────────────────

/** Built-in names a body reads from fields the renderer writes each frame. */
export const BUILTIN_FIELDS = ['dt', 'frame', 'pointerVelocity', 'pointerDown', 'idle', 'pointerOver'] as const
const BUILTIN_TYPES: Record<(typeof BUILTIN_FIELDS)[number], WgslType> = {dt: 'f32', frame: 'f32', pointerVelocity: 'vec2f', pointerDown: 'f32', idle: 'f32', pointerOver: 'f32'}

/** The extra uniform field a built-in is stored in. */
export function builtinField(name: string): string {
    return `_b_${name}`
}

/** The extra fields a definition needs for the built-ins its WGSL reads. */
export function builtinExtraFields(names: readonly string[]): Record<string, {schema: unknown; initial: number | number[]}> {
    const out: Record<string, {schema: unknown; initial: number | number[]}> = {}
    for (const n of names) {
        const type = BUILTIN_TYPES[n as (typeof BUILTIN_FIELDS)[number]]
        if (!type) continue
        out[builtinField(n)] = type === 'vec2f' ? {schema: d.vec2f, initial: [0, 0]} : {schema: d.f32, initial: 0}
    }
    return out
}

const builtinWriters = new WeakSet<object>()
function writeBuiltins(params: GpuFragmentParams, definition: WgslHostDefinition, names: readonly string[]): void {
    if (builtinWriters.has(params)) return
    builtinWriters.add(params)
    const tracker = frameTracker(params, !!definition.animatedTime)
    params.onBeforeRender(() => {
        const v = tracker.values
        for (const n of names) {
            const value = n === 'pointerVelocity' ? v.pointerVelocity : v[n as 'dt' | 'frame' | 'pointerDown' | 'idle' | 'pointerOver']
            params.setExtraField(builtinField(n), value as number | number[])
        }
    })
}

/** The body's call for the shape field anywhere, and the names its lowering reserves. */
const SHAPE_AT = 'shapeAt'
const SHAPE_AT_ALIAS = '_shapeAtFn'
const SHAPE_AT_ARG = '_shapeArg'
const SHAPE_AT_SCALE = '_shapeScale'

/** One `shapeAt` helper per shape path: `(…field inputs, scale, p) -> f32`, the distance at `p` in `shape` units. */
const shapeAtHelpers = new Map<string, unknown>()
function shapeAtHelper(portable: PortableFieldSampler, names: string[]): {key: string; fn: unknown} {
    const key = `shapeAt:${portable.key}`
    let fn = shapeAtHelpers.get(key)
    if (!fn) {
        const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => StringFnShell)(
            [...portable.args.map((a) => schemaFor(a.type)), d.f32, d.vec2f],
            d.f32,
        )
        const paramsText = [...portable.args.map((a, i) => `${names[i]}: ${a.type}`), `${SHAPE_AT_SCALE}: f32`, 'p: vec2f'].join(', ')
        const made = shell(`(${paramsText}) -> f32 {\n  return ${portable.sample(names, 'p')}.x / ${SHAPE_AT_SCALE};\n}`)
        fn = Object.keys(portable.uses).length ? made.$uses(portable.uses) : made
        shapeAtHelpers.set(key, fn)
    }
    return {key, fn}
}

function lowerFirst(name: string): string {
    return name ? name[0].toLowerCase() + name.slice(1) : 'custom'
}

function hostSuffix(host: WgslHost): string {
    return host === 'generator' ? 'Paint' : host === 'surface' ? 'Surface' : 'Filter'
}

/** How the body's `uv` and `viewport` read in a host: a generator follows a parent distortion. */
function hostCoords(host: WgslHost): HostCoords {
    const generator = host === 'generator'
    return {
        uv: (p) => (generator ? (p.uvContext ?? p.ctx.uv) : p.ctx.uv),
        viewport: (p) => (generator ? (p.effectiveViewportSize ?? p.ctx.viewportSize) : p.ctx.viewportSize),
    }
}

/** The shape's screen-aligned slope at this pixel (see the `shapeSlope` binding), hoisted once. */
const surfaceSlopes = new WeakMap<SurfaceFrame, Expr>()
function shapeSlopeAt(p: GpuFragmentParams, frame: SurfaceFrame | undefined): Expr {
    if (!frame?.sdfUV) throw new Error('wgsl: shapeSlope is only available in a surface body')
    let slope = surfaceSlopes.get(frame)
    if (!slope) {
        const at = frame.sdfUV
        const eps = 0.002
        const tap = (dx: number, dy: number): Expr =>
            frame.gradSampler(new Expr((ctx) => `(${at._emit(ctx)} + vec2f(${dx}, ${dy}))`)).member('x')
        const raw = new Expr((ctx) =>
            `(vec2f(${tap(eps, 0)._emit(ctx)} - ${tap(-eps, 0)._emit(ctx)}, ${tap(0, eps)._emit(ctx)} - ${tap(0, -eps)._emit(ctx)}) / ${2 * eps})`)
        const rot = p.uniforms.rotation
        slope = asLocal(rot
            ? new Expr((ctx) => {
                const r = `radians(${rot._emit(ctx)})`
                const v = raw._emit(ctx)
                return `vec2f(cos(${r}) * ${v}.x - sin(${r}) * ${v}.y, sin(${r}) * ${v}.x + cos(${r}) * ${v}.y)`
            })
            : raw, 'shapeSlope')
        surfaceSlopes.set(frame, slope)
    }
    return slope
}

/** The field sample at this pixel, hoisted once per frame (several names read it). */
const surfaceSamples = new WeakMap<SurfaceFrame, Expr>()
function surf0(frame: SurfaceFrame | undefined): Expr {
    if (!frame?.surf0) throw new Error('wgsl: shape values are only available in a surface body')
    let local = surfaceSamples.get(frame)
    if (!local) surfaceSamples.set(frame, (local = asLocal(frame.surf0, 'surf0')))
    return local
}

/** An input's texture this composition, or an empty placeholder when its compute did not run (no GPU). */
function inputTexture(params: GpuFragmentParams, key: string): KitTexture {
    const tex = params.computeOutputs?.[key] as KitTexture | undefined
    if (tex) return tex
    return placeholderFor(params, key)
}
const placeholders = new WeakMap<object, Map<string, KitTexture>>()
function placeholderFor(params: GpuFragmentParams, key: string): KitTexture {
    let m = placeholders.get(params)
    if (!m) placeholders.set(params, (m = new Map()))
    let t = m.get(key)
    if (!t) m.set(key, (t = params.registerMediaTexture(() => null)))
    return t
}

let indexHelper: unknown = null
/** `name(i)` for a slot list (events): texel i of row 0, unfiltered. */
function inputIndexHelper(): {key: string; fn: unknown} {
    if (!indexHelper) {
        const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => StringFnShell)([d.texture2d(d.f32), d.sampler(), d.i32], d.vec4f)
        indexHelper = shell(`(tex: texture_2d<f32>, samp: sampler, i: i32) -> vec4f {
  return textureLoad(tex, vec2i(clamp(i, 0, i32(textureDimensions(tex).x) - 1), 0), 0);
}`).$uses({})
    }
    return {key: 'input:index', fn: indexHelper}
}

let texelHelper: unknown = null
/** `name.out(p)` for an output read texel-exact: the one texel under `p`, never blended (a nearest
 *  point or an index must not mix with its neighbour's). */
function inputTexelHelper(): {key: string; fn: unknown} {
    if (!texelHelper) {
        const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => StringFnShell)([d.texture2d(d.f32), d.sampler(), d.vec2f], d.vec4f)
        texelHelper = shell(`(t: texture_2d<f32>, samp: sampler, p: vec2f) -> vec4f {
  let size = vec2i(textureDimensions(t));
  return textureLoad(t, clamp(vec2i(floor(p * vec2f(size))), vec2i(0), size - vec2i(1)), 0);
}`).$uses({})
    }
    return {key: 'input:texel', fn: texelHelper}
}

let exactHelper: unknown = null
/** `name(p)` for a 32-bit input (a full-precision grid): four exact texels blended by hand, since
 *  a filtering sampler on a float32 texture is not available on every GPU. */
function inputExactHelper(): {key: string; fn: unknown} {
    if (!exactHelper) {
        const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => StringFnShell)([d.texture2d(d.f32), d.sampler(), d.vec2f], d.vec4f)
        exactHelper = shell(`(t: texture_2d<f32>, samp: sampler, p: vec2f) -> vec4f {
  let size = vec2i(textureDimensions(t));
  let q = p * vec2f(size) - 0.5;
  let i = vec2i(floor(q));
  let f = q - floor(q);
  let m = size - vec2i(1);
  let a = textureLoad(t, clamp(i, vec2i(0), m), 0);
  let b = textureLoad(t, clamp(i + vec2i(1, 0), vec2i(0), m), 0);
  let c = textureLoad(t, clamp(i + vec2i(0, 1), vec2i(0), m), 0);
  let e = textureLoad(t, clamp(i + vec2i(1, 1), vec2i(0), m), 0);
  return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
}`).$uses({})
    }
    return {key: 'input:exact', fn: exactHelper}
}

const inputSampleHelpers = new Map<string, unknown>()
/** `name(p)` for an input: a sample at `p`, straight alpha. */
function inputSampleHelper(premultiplied: boolean): {key: string; fn: unknown} {
    const key = premultiplied ? 'input:pm' : 'input'
    let fn = inputSampleHelpers.get(key)
    if (!fn) {
        const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => StringFnShell)([d.texture2d(d.f32), d.sampler(), d.vec2f], d.vec4f)
        fn = shell(premultiplied
            ? `(tex: texture_2d<f32>, samp: sampler, p: vec2f) -> vec4f {
  let c = textureSampleLevel(tex, samp, p, 0.0);
  return select(vec4f(c.rgb / c.a, c.a), vec4f(0.0), c.a <= 0.0001);
}`
            : `(tex: texture_2d<f32>, samp: sampler, p: vec2f) -> vec4f {
  return textureSampleLevel(tex, samp, p, 0.0);
}`).$uses({})
        inputSampleHelpers.set(key, fn)
    }
    return {key, fn}
}

function automaticBindings(
    host: WgslHost,
    definition: WgslHostDefinition,
): {automatic: Binding[]; callables: ResourceCallable[]; bareOnly: Set<string>} {
    const out: Binding[] = []
    const callables: ResourceCallable[] = []
    const bareOnly = new Set<string>()
    const coords = hostCoords(host)
    out.push({name: 'uv', type: 'vec2f', value: coords.uv})
    out.push({
        name: 'time',
        type: 'f32',
        value: (p) => (definition.animatedTime ? animatedTime(p) : p.ctx.time),
    })
    out.push({name: 'aspect', type: 'f32', value: (p) => p.ctx.aspect})
    out.push({name: 'viewport', type: 'vec2f', value: coords.viewport})
    out.push({name: 'pointer', type: 'vec2f', value: (p) => p.ctx.pointer})
    for (const name of BUILTIN_FIELDS) {
        out.push({
            name,
            type: BUILTIN_TYPES[name],
            value: (p) => {
                const accessor = p.uniforms[builtinField(name)]
                if (!accessor) throw new Error(`wgsl: '${name}' needs its frame field; define the shader with defineShader`)
                return accessor
            },
        })
    }
    for (const read of definition.inputs?.reads ?? []) {
        const id = readName(read.call)
        out.push({
            name: `_inTex_${id}`,
            type: 'texture_2d<f32>',
            value: (p) => inputTexture(p, read.key).accessor(),
        })
        callables.push({
            name: id,
            alias: `_inRead_${id}`,
            leading: [`_inTex_${id}`, '_inSamp'],
            helper: (p) => (read.by === 'index' ? inputIndexHelper() : read.by === 'texel' ? inputTexelHelper() : read.unfilterable ? inputExactHelper() : inputSampleHelper(premultipliedKeys(p).has(read.key))),
        })
    }
    if (definition.inputs?.reads.length) out.push({name: '_inSamp', type: 'sampler', value: () => expr('samp.$.linearClamp')})
    if (host === 'pointwise') {
        out.push({name: 'child', type: 'vec4f', value: (p) => (p as FilterParams).childNode})
    }
    if (host === 'gather') {
        out.push({name: 'childTexture', type: 'texture_2d<f32>', value: (p) => (p as RttFilterParams).texture.accessor()})
        // `samp.$.<name>` is the composer's raw sampler accessor; the entry builder scans the
        // emitted call site for it and binds the shared sampler layout to this pass.
        out.push({name: 'childSampler', type: 'sampler', value: () => expr('samp.$.linearClamp')})
        // `child` stays available in a gather filter: the pixel beneath at this uv, straight
        // alpha (a body that samples neighbours usually still wants the pixel itself).
        out.push({name: 'child', type: 'vec4f', value: (p) => sampleStraight((p as RttFilterParams).texture, p.ctx.uv)})
    }
    if (host === 'surface') {
        // The shape's field at this pixel: `.x` the signed distance (negative inside, scaled
        // with the shape like the library's materials), `.w` the depth of a 3D shape.
        const shapeDistance = (p: GpuFragmentParams, frame?: SurfaceFrame): Expr =>
            new Expr((ctx) => `(${surf0(frame).member('x')._emit(ctx)} / ${p.uniforms.scale._emit(ctx)})`)
        out.push({name: 'shapeDistance', type: 'f32', value: shapeDistance})
        // `shape` is the earlier name of `shapeDistance`, kept so saved components still bind.
        out.push({name: 'shape', type: 'f32', value: shapeDistance})
        // Which way the distance grows, screen-aligned: a centred difference of the field across
        // the shape's own square, rotated by the shape's rotation. Normalize it for a normal.
        out.push({name: 'shapeSlope', type: 'vec2f', value: (p, frame) => shapeSlopeAt(p, frame)})
        out.push({name: 'shapeDepth', type: 'f32', value: (_p, frame) => surf0(frame).member('w')})
        out.push({name: 'shapeField', type: 'vec4f', value: (_p, frame) => surf0(frame)})
        out.push({name: 'shapeUV', type: 'vec2f', value: (_p, frame) => frame!.sdfUV!})
        // The layer inside, when the material shades it (premultiplied in the texture; `child` straight).
        out.push({name: 'childTexture', type: 'texture_2d<f32>', value: (_p, frame) => frame!.childTexture!.accessor()})
        out.push({name: 'childSampler', type: 'sampler', value: () => expr('samp.$.linearClamp')})
        out.push({name: 'child', type: 'vec4f', value: (p, frame) => sampleStraight(frame!.childTexture!, p.ctx.uv)})
    }
    for (const [name, config] of Object.entries(definition.props)) {
        if (CTX_TYPES[name] || RESERVED.has(name)) continue
        if (isResourceProp(config)) {
            const plan = resourcePlan(name, config, definition.props, coords)
            if (!plan) continue
            out.push(...plan.bindings)
            callables.push(...plan.callables)
            for (const bare of plan.bareValues) bareOnly.add(bare)
            continue
        }
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
    return {automatic: out, callables, bareOnly}
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

// ── Scalar-slot bodies: a shape's distance, a warp's coordinate ─────────────────────────

/**
 * Lower a body that is a plain function of fixed arguments (a `shape:` body of `p`, a `map:` body
 * of `uv`) to a call builder. Props, `time`, `aspect`, library functions and helpers bind by name,
 * like a pixel body; inputs and the child do not (these bodies run inside the engine's own shape
 * and warp passes).
 */
export function lowerWgslFunction(
    body: WgslBody,
    definition: WgslHostDefinition,
    shape: {name: string; args: readonly {name: string; type: WgslType}[]; ret: 'f32' | 'vec2f'},
): (args: readonly Expr[], value: (name: string) => Expr | null) => Expr {
    if (!/\breturn\b/.test(stripComments(body.spec.body))) {
        throw new Error(`wgsl: the ${shape.name} body must return a ${shape.ret}`)
    }
    const argNames = new Set(shape.args.map((a) => a.name))
    const bindable = new Map<string, Binding>()
    for (const [name, config] of Object.entries(definition.props)) {
        if (argNames.has(name)) continue
        const type = wgslTypeForProp(config)
        if (!type) continue
        bindable.set(name, {name, type, value: () => {
            throw new Error('unreachable')
        }})
    }
    if (!argNames.has('time')) bindable.set('time', {name: 'time', type: 'f32', value: () => expr('0.0')})
    if (!argNames.has('aspect')) bindable.set('aspect', {name: 'aspect', type: 'f32', value: () => expr('1.0')})
    const helperDecls = definition.helpers ? parseWgslDecls(definition.helpers) : null
    const helpers = helperDecls ? planHelpers(helperDecls, (t) => t, bindable) : null
    let text = body.spec.body
    const used = new Set([...scanIdentifiers(text)].filter((id) => bindable.has(id)))
    if (helpers) {
        for (const [fnName, needs] of helpers.needs) {
            if (needs.length === 0) continue
            text = appendCallArgs(text, fnName, needs)
            for (const n of needs) used.add(n)
        }
        text = inlineConsts(text, helpers.consts)
    }
    const bound = [...used].sort()
    const lib = definition.inputs?.lib ?? []
    const libExternals = lib.length ? linkLibFns(lib) : {}
    const libStructs = lib.length ? libStructSchemas(lib) : new Map<string, unknown>()
    const params = [...shape.args, ...bound.map((n) => ({name: n, type: bindable.get(n)!.type}))]
    let fn: unknown = null
    const make = () => {
        const helperFns = helpers ? buildHelpers(helpers, new Map(Object.entries(libExternals)), libStructs) : {}
        const ret = shape.ret === 'f32' ? d.f32 : d.vec2f
        const made = (tgpu.fn as unknown as (a: unknown[], r: unknown) => StringFnShell)(params.map((p) => schemaFor(p.type)), ret)(
            `(${params.map((p) => `${paramName(p.name)}: ${p.type}`).join(', ')}) -> ${shape.ret} {\n${text}\n}`)
        const uses = {...libExternals, ...helperFns}
        return Object.keys(uses).length ? made.$uses(uses) : made
    }
    const hint = paramName(lowerFirst(definition.name) + (shape.ret === 'f32' ? 'Distance' : 'Map'))
    return (args, value) => {
        markCustomWgslUsed()
        fn ??= make()
        const values = bound.map((n) => {
            const v = value(n)
            if (!v) throw new Error(`wgsl: '${n}' cannot be read in a ${shape.name} body`)
            return v
        })
        return call(fn, hint, [...args, ...values])
    }
}
