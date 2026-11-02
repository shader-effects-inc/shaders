/**
 * std/graph — validate, interpret, and document JSON node graphs over the std vocabulary.
 *
 *   validateStdGraph(envelope)   static typing against the catalog — repair-grade errors
 *   defineStdGraph(envelope)     → an ordinary GpuShaderDefinition (a std generator)
 *   renderStdGraphManifest()     the catalog as reference text for whoever writes graphs
 *   resolveStdGraphWgsl(env)     GPU-free compose + tgpu.resolve — the "does it compile" gate
 *
 * The interpreter runs at COMPOSITION time inside the generator's `paint`: each node is
 * resolved to a value (a GPU expression, a field, a palette…) by calling the catalog word,
 * and numeric results are bound to WGSL locals so multi-consumer nodes evaluate once.
 */
import type {Expr, GpuFragmentParams, GpuShaderDefinition, RegistryNode, RegistryView} from '../../gpu/contract'
import {call, floatE} from '../../gpu/composer'
import {composeNodeTree} from '../../gpu/composer'
import {tgpu, d} from '../../gpu/kit/index'
import {markCustomWgslUsed} from '../../gpu/support'
import {createUniformStore, type FieldInit} from '../../gpu/uniformStore'
import {SystemUniforms} from '../../gpu/kit/coords'
import {createGpuUniformsMap} from '../../gpu/uniformBridge'
import type {ComponentProps, NodeMetadata, PropConfig} from '../../types'
import {
    colorSpaceOptions, transformBool01, transformColor, transformColorSpace, transformPosition,
} from '../../utilities/transformations'

import {defineStd} from "../lower"
import type {StdPropConfig} from '../types'
import * as m from '../math'
import {CATALOG, animatedTime, colorSpaceModeOf, isNumVal, num, uniformOf, volumeMarch, type BuildCtx, type InputSpec, type InputType, type NumericType, type OpDef, type StaticType, type Val} from './catalog'
import {STD_GRAPH_LIMITS, StdGraphError, type GraphArg, type GraphNode, type GraphPropKind, type GraphPropSpec, type GraphValueType, type StdGraphEnvelope} from './types'


// ── Shared: references ──────────────────────────────────────────────────────────────────

const SYSTEM_TYPES: Record<string, NumericType> = {uv: 'vec2', aspect: 'f32', viewport: 'vec2', time: 'f32', pointer: 'vec2'}
/** Names bound inside bodies: `p` (fieldOf), `i` (repeat/march), `z`/`t` (march). */
const BODY_BOUND = new Set(['p', 'i', 'z', 't'])
const PROP_TYPES: Record<GraphPropKind, NumericType> = {float: 'f32', angle: 'f32', bool: 'f32', color: 'vec4', position: 'vec2'}
const RESERVED_PROPS = new Set(['colorSpace', 'opacity', 'blendMode', 'visible', 'transform', 'boundingBox', 'maskSource', 'maskType'])
const SWIZZLE = /^[xyzwrgba]{1,4}$/
const SWIZZLE_MIN: Record<string, number> = {x: 1, r: 1, y: 2, g: 2, z: 3, b: 3, w: 4, a: 4}
const WIDTH: Record<NumericType, number> = {f32: 1, vec2: 2, vec3: 3, vec4: 4}
const BY_WIDTH: NumericType[] = ['f32', 'f32', 'vec2', 'vec3', 'vec4']

const isNumeric = (t: string): t is NumericType => t === 'f32' || t === 'vec2' || t === 'vec3' || t === 'vec4'

interface ParsedRef {
    kind: 'node' | 'prop' | 'system'
    name: string
    member?: string
}

function parseRef(s: string): ParsedRef | null {
    const [head, ...rest] = s.split('.')
    if (!head) return null
    const member = rest.length ? rest.join('.') : undefined
    if (head.startsWith('$')) return {kind: 'node', name: head.slice(1), member}
    if (head.startsWith('@')) return {kind: 'prop', name: head.slice(1), member}
    if (head in SYSTEM_TYPES || BODY_BOUND.has(head)) return {kind: 'system', name: head, member}
    return null
}

function swizzleType(base: NumericType, sw: string, where: string): NumericType {
    if (!SWIZZLE.test(sw)) throw new StdGraphError(`"${where}": ".${sw}" is not a swizzle`)
    // WGSL scalars have no components: `.x` of an f32 would compile to an error on the GPU.
    if (base === 'f32') throw new StdGraphError(`"${where}": cannot swizzle an f32 (a single number) — use it without .${sw}`)
    const need = Math.max(...sw.split('').map((c) => SWIZZLE_MIN[c] ?? 9))
    if (need > WIDTH[base]) throw new StdGraphError(`"${where}": .${sw} reads past a ${base}`)
    return BY_WIDTH[sw.length]!
}

/** What a graph reads that its definition must provide: the clock, the colour-space select. */
interface GraphUsage {
    time: boolean
    colorSpace: boolean
}

/** The colour words that mix in the component's colour space (they read its `colorSpace` prop). */
const COLOR_SPACE_OPS = new Set(['pair', 'palette', 'ladder3', 'mixColors'])

// ── Static pass ─────────────────────────────────────────────────────────────────────────

interface StaticScope {
    nodes: Map<string, StaticType>
    parent?: StaticScope
    bound: Record<string, StaticType>
    /** Nodes that failed validation: reading one is not reported again. */
    failed?: Set<string>
}

/** A read of a node (or prop) whose own error is already reported. */
class FollowOnError extends StdGraphError {}

function hasFailed(scope: StaticScope, id: string): boolean {
    for (let s: StaticScope | undefined = scope; s; s = s.parent) {
        if (s.failed?.has(id)) return true
    }
    return false
}

/** `[x, y]` … `[x, y, z, w]` of plain numbers: a vector literal where an input takes a vector. */
function vectorLiteralType(raw: GraphArg): NumericType | null {
    if (!Array.isArray(raw) || raw.length < 2 || raw.length > 4) return null
    if (!raw.every((n) => typeof n === 'number' && Number.isFinite(n))) return null
    return BY_WIDTH[raw.length]!
}

/** Inputs that take structured data (palette stops, schedules) keep arrays as data. */
function takesData(spec: InputSpec): boolean {
    return (Array.isArray(spec.type) ? spec.type : [spec.type]).includes('data')
}

function lookupStatic(scope: StaticScope, id: string): StaticType | undefined {
    for (let s: StaticScope | undefined = scope; s; s = s.parent) {
        const t = s.nodes.get(id)
        if (t) return t
    }
    return undefined
}

function lookupBound(scope: StaticScope, name: string): StaticType | undefined {
    for (let s: StaticScope | undefined = scope; s; s = s.parent) {
        if (name in s.bound) return s.bound[name]
    }
    return undefined
}

function staticArg(arg: GraphArg, scope: StaticScope, props: Record<string, GraphPropSpec>, where: string, usage: GraphUsage): StaticType {
    if (typeof arg === 'number') return {type: 'literal'}
    if (typeof arg === 'boolean') return {type: 'literal'}
    if (arg === null) throw new StdGraphError(`"${where}" is null`)
    if (typeof arg !== 'string') return {type: 'data'}
    const ref = parseRef(arg)
    if (!ref) {
        // A bare identifier that names a node or prop is a forgotten sigil, not an enum.
        const head = arg.split('.')[0]!
        if (lookupStatic(scope, head)) throw new StdGraphError(`"${where}": "${arg}" names a node — write "$${arg}"`)
        if (head in props) throw new StdGraphError(`"${where}": "${arg}" names a prop — write "@${arg}"`)
        return {type: 'enum'}
    }
    let base: StaticType | undefined
    if (ref.kind === 'node') {
        base = lookupStatic(scope, ref.name)
        if (!base && hasFailed(scope, ref.name)) throw new FollowOnError(`"${where}" reads a failed node`)
        if (!base) throw new StdGraphError(`"${where}" references unknown node "$${ref.name}" (nodes must be defined earlier in the list)`)
    } else if (ref.kind === 'prop') {
        const spec = props[ref.name]
        if (!spec) throw new StdGraphError(`"${where}" references undeclared prop "@${ref.name}"`)
        if (!(spec.kind in PROP_TYPES)) throw new FollowOnError(`"${where}" reads an invalid prop`)
        base = {type: PROP_TYPES[spec.kind], ref: true, propKind: spec.kind}
    } else {
        if (ref.name === 'time') usage.time = true
        const bound = lookupBound(scope, ref.name)
        if (bound) base = bound
        else if (ref.name in SYSTEM_TYPES) base = {type: SYSTEM_TYPES[ref.name]!}
        else throw new StdGraphError(`"${where}": "${ref.name}" is only available inside a ${ref.name === 'p' ? 'fieldOf' : ref.name === 'i' ? 'repeat or march' : 'march'} body`)
    }
    if (!ref.member) return base
    if (base.type === 'record') {
        // A field, optionally swizzled: "$key.L", "$key.L.z"
        const [name, ...swizzle] = ref.member.split('.')
        const field = base.fields?.[name!]
        if (!field) throw new StdGraphError(`"${where}": "${arg}" — no field "${name}" (has: ${Object.keys(base.fields ?? {}).join(', ')})`)
        if (!swizzle.length) return field
        if (isNumeric(field.type)) return {type: swizzleType(field.type, swizzle.join('.'), where)}
        throw new StdGraphError(`"${where}": cannot read ".${swizzle.join('.')}" of a ${field.type}`)
    }
    if (isNumeric(base.type)) return {type: swizzleType(base.type, ref.member, where)}
    if (base.type === 'literal') throw new StdGraphError(`"${where}": cannot swizzle a number`)
    throw new StdGraphError(`"${where}": cannot read ".${ref.member}" of a ${base.type}`)
}

/** The input takes vec4 and not vec3 — the case where an rgb should become an opaque rgba. */
function wantsExactly(spec: InputSpec, want: InputType): boolean {
    const types = Array.isArray(spec.type) ? spec.type : [spec.type]
    return types.includes(want) && !types.includes('vec3') && !types.includes('num') && !types.includes('vec')
}

/** `"foo"` where a node or prop named foo exists and the input is not a string slot → `"$foo"` / `"@foo"`. */
function addMissingSigil(raw: string, spec: InputSpec, scope: StaticScope, props: Record<string, GraphPropSpec>): string | null {
    if (parseRef(raw)) return null
    const types = Array.isArray(spec.type) ? spec.type : [spec.type]
    if (types.includes('enum') || types.includes('data') || types.includes('const')) return null
    const [head, ...rest] = raw.split('.')
    if (!head) return null
    const tail = rest.length ? `.${rest.join('.')}` : ''
    if (lookupStatic(scope, head)) return `$${head}${tail}`
    if (head in props) return `@${head}${tail}`
    return null
}

function accepts(spec: InputSpec, t: StaticType, where: string): void {
    const types = Array.isArray(spec.type) ? spec.type : [spec.type]
    const ok = types.some((want: InputType) => {
        switch (want) {
            case 'num': return t.type === 'literal' || isNumeric(t.type)
            case 'f32': return t.type === 'literal' || t.type === 'f32'
            case 'vec': return t.type === 'vec2' || t.type === 'vec3' || t.type === 'vec4'
            case 'vec2': case 'vec3': case 'vec4': return t.type === want
            case 'const': return t.type === 'literal'
            case 'enum': return t.type === 'enum' && (!spec.values || spec.values.includes((t as {value?: string}).value ?? ''))
            case 'data': return t.type === 'data'
            case 'field': case 'palette': case 'schedule': return t.type === want
            case 'prop:float': return !!t.ref && (t.propKind === 'float' || t.propKind === 'angle' || t.propKind === 'bool')
            case 'prop:color': return !!t.ref && t.propKind === 'color'
            case 'prop:position': return !!t.ref && t.propKind === 'position'
        }
    })
    if (!ok) {
        const wantText = types.map((w) => w.startsWith('prop:') ? `a ${w.slice(5)} prop ("@name")` : w).join(' | ')
        const got = t.type === 'literal' ? 'a number' : t.ref ? `a ${t.propKind} prop` : t.type
        throw new StdGraphError(`"${where}" expects ${wantText}, got ${got}${spec.values ? ` (allowed: ${spec.values.join(', ')})` : ''}`)
    }
}

function outputType(op: OpDef, inputs: Record<string, StaticType | undefined>, id: string): StaticType {
    const rule = op.output
    let out: GraphValueType | {record: Record<string, GraphValueType>}
    if (rule === 'broadcast') {
        let width = 1
        let vecType: NumericType | undefined
        for (const [name, t] of Object.entries(inputs)) {
            if (!t || t.type === 'literal' || t.type === 'f32') continue
            if (!isNumeric(t.type)) continue
            if (vecType && vecType !== t.type) throw new StdGraphError(`mixes ${vecType} and ${t.type} (input "${name}") — vector widths must match`, id)
            vecType = t.type
            width = WIDTH[t.type]
        }
        out = BY_WIDTH[width]!
    } else if (typeof rule === 'function') {
        out = rule(inputs)
    } else {
        out = rule
    }
    if (typeof out === 'object') {
        const fields: Record<string, StaticType> = {}
        for (const [k, v] of Object.entries(out.record)) fields[k] = {type: v}
        return {type: 'record', fields}
    }
    return {type: out}
}

interface ValidateInternalOptions {
    handRolled?: Array<{id: string; name: string; type: string; why: string}>
    /** Collect every node's error here and keep going, instead of throwing at the first. */
    errors?: string[]
}

function validateNodes(
    nodes: GraphNode[],
    scope: StaticScope,
    props: Record<string, GraphPropSpec>,
    usage: GraphUsage,
    depth: number,
    options: ValidateInternalOptions = {},
): void {
    const seen = new Set<string>()
    for (const node of nodes) {
        try {
            validateNode(node)
        } catch (err) {
            if (!options.errors) throw err
            // Reported once, where it went wrong: nodes reading this one don't repeat it.
            if (!(err instanceof FollowOnError)) options.errors.push(err instanceof Error ? err.message : String(err))
            if (node && typeof node.id === 'string') (scope.failed ??= new Set()).add(node.id)
        }
    }

    function validateNode(node: GraphNode): void {
        if (!node || typeof node !== 'object') throw new StdGraphError('a node is not an object')
        if (typeof node.id !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(node.id)) {
            throw new StdGraphError(`node id "${String(node.id)}" must be letters/digits/underscore, starting with a letter`)
        }
        if (seen.has(node.id) || lookupStatic(scope, node.id)) throw new StdGraphError(`duplicate id`, node.id)
        if (node.id in SYSTEM_TYPES || BODY_BOUND.has(node.id)) throw new StdGraphError(`id shadows a system value`, node.id)
        seen.add(node.id)
        const op = CATALOG[node.op]
        if (!op) {
            const close = Object.keys(CATALOG).filter((k) => k.toLowerCase().includes(String(node.op).toLowerCase().slice(0, 4))).slice(0, 5)
            throw new StdGraphError(`unknown op "${node.op}"${close.length ? ` — did you mean ${close.join(', ')}?` : ''}`, node.id)
        }
        if (COLOR_SPACE_OPS.has(node.op)) usage.colorSpace = true
        const args = node.args ?? {}
        if (op.control === 'wgsl') {
            const spec = readWgslSpec(node, (arg, where) => {
                const literal = vectorLiteralType(arg)
                if (literal) return literal
                const t = staticArg(arg, scope, props, where, usage)
                if (t.type === 'literal') return 'f32'
                if (isNumeric(t.type)) return t.type
                throw new StdGraphError(`wgsl input at ${where} must be numeric, got ${t.type}`, node.id)
            })
            options.handRolled?.push({id: node.id, name: spec.name, type: spec.returns, why: spec.why})
            scope.nodes.set(node.id, {type: spec.returns})
            return
        }
        for (const key of Object.keys(args)) {
            if (!(key in op.inputs)) throw new StdGraphError(`op ${node.op} has no input "${key}" (inputs: ${Object.keys(op.inputs).join(', ') || 'none'})`, node.id)
        }
        const inputTypes: Record<string, StaticType | undefined> = {}
        for (const [name, spec] of Object.entries(op.inputs)) {
            let raw = args[name]
            if (raw === undefined || raw === null) {
                if (!spec.optional) throw new StdGraphError(`missing required input "${name}" of ${node.op}`, node.id)
                continue
            }
            // A forgotten sigil on a name that can only mean one thing is accepted and written
            // back, so the build pass sees the same reference.
            const sigiled = typeof raw === 'string' ? addMissingSigil(raw, spec, scope, props) : null
            if (sigiled) { raw = sigiled; args[name] = sigiled }
            const literal = takesData(spec) ? null : vectorLiteralType(raw)
            let t: StaticType = literal ? {type: literal} : staticArg(raw, scope, props, `${node.id}.${name}`, usage)
            if (t.type === 'enum' && typeof raw === 'string') (t as {value?: string}).value = raw
            // An rgb handed to a colour slot is promoted to opaque rgba (see buildArgFor).
            if (t.type === 'vec3' && wantsExactly(spec, 'vec4')) t = {type: 'vec4'}
            accepts(spec, t, `${node.id}.${name}`)
            inputTypes[name] = t
        }

        if (op.control) {
            if (depth > 0) throw new StdGraphError(`${node.op} cannot be nested inside another body`, node.id)
            if (!Array.isArray(node.body) || !node.body.length) throw new StdGraphError(`${node.op} needs a non-empty body`, node.id)
            if (node.body.length > STD_GRAPH_LIMITS.maxBody) throw new StdGraphError(`body has ${node.body.length} nodes (max ${STD_GRAPH_LIMITS.maxBody})`, node.id)
            if (typeof node.result !== 'string' || !node.result.startsWith('$')) throw new StdGraphError(`${node.op} needs "result": "$bodyNodeId"`, node.id)
            const bound: Record<string, StaticType> = op.control === 'repeat'
                ? {i: {type: 'literal'}}
                : op.control === 'march'
                    ? {z: {type: 'f32'}, t: {type: 'literal'}, i: {type: 'literal'}}
                    : {p: {type: 'vec2'}}
            const inner: StaticScope = {nodes: new Map(), parent: scope, bound}
            validateNodes(node.body, inner, props, usage, depth + 1, options)
            const resultType = staticArg(node.result, inner, props, `${node.id}.result`, usage)
            const resultNumeric: NumericType | null = resultType.type === 'literal' ? 'f32' : isNumeric(resultType.type) ? resultType.type : null
            if (op.control === 'repeat') {
                const count = args.count
                if (typeof count !== 'number' || count < 1 || count > STD_GRAPH_LIMITS.maxRepeat) throw new StdGraphError(`repeat count must be 1..${STD_GRAPH_LIMITS.maxRepeat}`, node.id)
                if (!resultNumeric) throw new StdGraphError(`repeat result must be numeric, got ${resultType.type}`, node.id)
                scope.nodes.set(node.id, {type: resultNumeric})
            } else if (op.control === 'march') {
                const steps = args.steps
                if (typeof steps !== 'number' || steps < 1 || steps > STD_GRAPH_LIMITS.maxMarchSteps) throw new StdGraphError(`march steps must be 1..${STD_GRAPH_LIMITS.maxMarchSteps}`, node.id)
                if (resultNumeric !== 'vec3' && resultNumeric !== 'f32') throw new StdGraphError(`march result (the emitted light) must be a vec3 or f32, got ${resultType.type}`, node.id)
                if (typeof node.absorb !== 'string' || !node.absorb.startsWith('$')) throw new StdGraphError(`march needs "absorb": "$bodyNodeId" (the step's absorption, f32 or vec3)`, node.id)
                const absorbType = staticArg(node.absorb, inner, props, `${node.id}.absorb`, usage)
                if (absorbType.type !== 'f32' && absorbType.type !== 'vec3' && absorbType.type !== 'literal') throw new StdGraphError(`march absorb must be an f32 or vec3, got ${absorbType.type}`, node.id)
                scope.nodes.set(node.id, {type: 'record', fields: {color: {type: 'vec3'}, transmittance: {type: 'vec3'}}})
            } else {
                if (!resultNumeric) throw new StdGraphError(`fieldOf result must be numeric (f32 height, or a vec3/vec4 colour), got ${resultType.type}`, node.id)
                scope.nodes.set(node.id, {type: 'field', valueType: resultNumeric})
            }
            return
        }
        if (node.body) throw new StdGraphError(`only repeat/fieldOf take a body`, node.id)
        scope.nodes.set(node.id, outputType(op, inputTypes, node.id))
    }
}

/** Check a props manifest (shared with `wgsl` envelopes). Throws StdGraphError naming every problem. */
export function validateGraphProps(props: Record<string, GraphPropSpec>): void {
    const errors = graphPropErrors(props)
    if (errors.length) throw new StdGraphError(summarizeErrors(errors))
}

/** Every problem with a props manifest, one message each. */
function graphPropErrors(props: Record<string, GraphPropSpec>): string[] {
    const errors: string[] = []
    const names = Object.keys(props)
    if (names.length > STD_GRAPH_LIMITS.maxProps) errors.push(`${names.length} props declared (max ${STD_GRAPH_LIMITS.maxProps})`)
    for (const [name, spec] of Object.entries(props)) {
        try {
            checkProp(name, spec)
        } catch (err) {
            errors.push(err instanceof Error ? err.message : String(err))
        }
    }
    return errors
}

/** At most this many problems are listed; the rest are counted. */
const MAX_REPORTED_ERRORS = 10

function summarizeErrors(errors: string[]): string {
    if (errors.length === 1) return errors[0]!
    const listed = errors.slice(0, MAX_REPORTED_ERRORS).map((e) => `- ${e}`)
    if (errors.length > MAX_REPORTED_ERRORS) listed.push(`- … and ${errors.length - MAX_REPORTED_ERRORS} more`)
    return `${errors.length} problems:\n${listed.join('\n')}`
}

function checkProp(name: string, spec: GraphPropSpec): void {
    if (!/^[a-z][A-Za-z0-9]*$/.test(name)) throw new StdGraphError(`prop "${name}" must be camelCase (letters/digits, starting lowercase)`)
    if (RESERVED_PROPS.has(name)) throw new StdGraphError(`prop "${name}" is reserved`)
    if (!spec || !(spec.kind in PROP_TYPES)) throw new StdGraphError(`prop "${name}": kind must be one of ${Object.keys(PROP_TYPES).join(', ')}`)
    if (name === 'speed' && spec.kind !== 'float') throw new StdGraphError(`prop "speed" must be a float (it drives the clock)`)
    const dflt = spec.default
    switch (spec.kind) {
        case 'float': case 'angle':
            if (typeof dflt !== 'number' || !Number.isFinite(dflt)) throw new StdGraphError(`prop "${name}": default must be a number`)
            if (spec.kind === 'float' && (typeof spec.min !== 'number' || typeof spec.max !== 'number')) throw new StdGraphError(`prop "${name}": float props need min and max`)
            if (spec.kind === 'float' && spec.min! >= spec.max!) throw new StdGraphError(`prop "${name}": min must be below max`)
            break
        case 'bool':
            if (typeof dflt !== 'boolean') throw new StdGraphError(`prop "${name}": default must be true/false`)
            break
        case 'color':
            if (typeof dflt !== 'string' || !/^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/.test(dflt)) throw new StdGraphError(`prop "${name}": default must be "#rrggbb"`)
            break
        case 'position': {
            const pos = dflt as {x?: unknown; y?: unknown} | null
            if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number') throw new StdGraphError(`prop "${name}": default must be {x, y}`)
            break
        }
    }
}

export interface StdGraphValidation {
    ok: boolean
    /** Every problem found, as one message (a list when there are several). */
    error?: string
    /** The same problems, one per entry. */
    errors?: string[]
    /** The graph reads the clock, so the definition carries `speed` + animatedTime. */
    usesTime?: boolean
    /** The graph mixes colours in the colour space, so the definition carries `colorSpace`. */
    usesColorSpace?: boolean
    outputType?: 'vec3' | 'vec4'
    /** Hand-rolled WGSL parts the graph declares (name + why), for review and promotion. */
    handRolled?: Array<{id: string; name: string; type: string; why: string}>
}

const WGSL_MAX_CODE = 3000
/** Things a hand-rolled BODY must not do: declare functions/types, touch resources, use attributes. */
const WGSL_FORBIDDEN = /\b(fn|struct|var\s*<|textureSample\w*|textureLoad|textureStore|storage|uniform|workgroup|bind_group|binding|group|override|const_assert|discard|import|enable|requires)\b|@|\$/
const WGSL_TYPES: Record<string, NumericType> = {f32: 'f32', vec2: 'vec2', vec3: 'vec3', vec4: 'vec4'}
/** Names a hand-rolled input can't take: WGSL's keywords and reserved words, and its type names. */
const WGSL_RESERVED = new Set([
    'alias', 'break', 'case', 'const', 'const_assert', 'continue', 'continuing', 'default', 'diagnostic', 'discard',
    'else', 'enable', 'false', 'fn', 'for', 'if', 'let', 'loop', 'override', 'requires', 'return', 'struct', 'switch',
    'true', 'var', 'while',
    'NULL', 'Self', 'abstract', 'active', 'alignas', 'alignof', 'as', 'asm', 'asm_fragment', 'async', 'attribute',
    'auto', 'await', 'become', 'cast', 'catch', 'class', 'co_await', 'co_return', 'co_yield', 'coherent',
    'column_major', 'common', 'compile', 'compile_fragment', 'concept', 'const_cast', 'consteval', 'constexpr',
    'constinit', 'crate', 'debugger', 'decltype', 'delete', 'demote', 'demote_to_helper', 'do', 'dynamic_cast', 'enum',
    'explicit', 'export', 'extends', 'extern', 'external', 'fallthrough', 'filter', 'final', 'finally', 'friend',
    'from', 'fxgroup', 'get', 'goto', 'groupshared', 'highp', 'impl', 'implements', 'import', 'inline', 'instanceof',
    'interface', 'layout', 'lowp', 'macro', 'macro_rules', 'match', 'mediump', 'meta', 'mod', 'module', 'move', 'mut',
    'mutable', 'namespace', 'new', 'nil', 'noexcept', 'noinline', 'nointerpolation', 'non_coherent', 'noncoherent',
    'noperspective', 'null', 'nullptr', 'of', 'operator', 'package', 'packoffset', 'partition', 'pass', 'patch',
    'pixelfragment', 'precise', 'precision', 'premerge', 'priv', 'protected', 'pub', 'public', 'readonly', 'ref',
    'regardless', 'register', 'reinterpret_cast', 'require', 'resource', 'restrict', 'self', 'set', 'shared', 'sizeof',
    'smooth', 'snorm', 'static', 'static_assert', 'static_cast', 'std', 'subroutine', 'super', 'target', 'template',
    'this', 'thread_local', 'throw', 'trait', 'try', 'type', 'typedef', 'typeid', 'typename', 'typeof', 'union',
    'unless', 'unorm', 'unsafe', 'unsized', 'use', 'using', 'varying', 'virtual', 'volatile', 'wgsl', 'where', 'with',
    'writeonly', 'yield',
])
const WGSL_TYPE_NAME = /^(bool|f16|f32|i32|u32|vec[234][fhiu]?|mat[234]x[234][fh]?|array|atomic|ptr|sampler|sampler_comparison|texture_\w+)$/
const WGSL_TYPE_NAMES: Record<NumericType, string> = {f32: 'f32', vec2: 'vec2f', vec3: 'vec3f', vec4: 'vec4f'}
const WGSL_SCHEMAS = {f32: d.f32, vec2: d.vec2f, vec3: d.vec3f, vec4: d.vec4f} as const

interface WgslSpec {
    params: Array<{name: string; type: NumericType; arg: GraphArg}>
    returns: NumericType
    code: string
    name: string
    why: string
}

/** Read and check a `wgsl` node's declaration (shared by the static and build passes). */
function readWgslSpec(node: GraphNode, typeOfArg: (arg: GraphArg, where: string) => NumericType): WgslSpec {
    const args = node.args ?? {}
    const inputs = args.inputs
    if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) throw new StdGraphError('wgsl needs "inputs": {paramName: "$ref" | "@prop" | number}', node.id)
    const returns = WGSL_TYPES[String(args.type)]
    if (!returns) throw new StdGraphError(`wgsl "type" must be one of ${Object.keys(WGSL_TYPES).join(', ')}`, node.id)
    const code = typeof args.code === 'string' ? args.code.trim() : ''
    if (!code) throw new StdGraphError('wgsl needs "code": the function body', node.id)
    if (code.length > WGSL_MAX_CODE) throw new StdGraphError(`wgsl code is ${code.length} chars (max ${WGSL_MAX_CODE}) — split it, or find the word`, node.id)
    if (!/\breturn\b/.test(code)) throw new StdGraphError('wgsl code must return a value', node.id)
    const forbidden = code.match(WGSL_FORBIDDEN)
    if (forbidden) throw new StdGraphError(`wgsl code may not use "${forbidden[0]}" — a hand-rolled part is a pure function of its inputs`, node.id)
    let depth = 0
    for (const ch of code) {
        if (ch === '{') depth++
        else if (ch === '}') depth--
        if (depth < 0) break
    }
    if (depth !== 0) throw new StdGraphError('wgsl code has unbalanced braces', node.id)
    const name = typeof args.name === 'string' ? args.name.trim() : ''
    const why = typeof args.why === 'string' ? args.why.trim() : ''
    if (!name || !why) throw new StdGraphError('wgsl needs "name" (the concept) and "why" (what you tried first) — hand-rolled parts are reviewed for promotion', node.id)
    const params: WgslSpec['params'] = []
    for (const [pname, arg] of Object.entries(inputs as Record<string, GraphArg>)) {
        if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(pname)) throw new StdGraphError(`wgsl input "${pname}" must be an identifier (letters, digits, underscores; starting with a letter)`, node.id)
        if (WGSL_RESERVED.has(pname) || WGSL_TYPE_NAME.test(pname)) throw new StdGraphError(`wgsl input "${pname}" is a WGSL keyword or type name — rename it`, node.id)
        params.push({name: pname, type: typeOfArg(arg, `${node.id}.inputs.${pname}`), arg})
    }
    if (params.length > 12) throw new StdGraphError('wgsl takes at most 12 inputs', node.id)
    return {params, returns, code, name, why}
}

/** `body` / `result` / `absorb` written inside `args` (the syntax card shows args as the only
 *  per-node object) are moved to the node, where the interpreter reads them. Recursive. */
function hoistBodyFields(nodes: GraphNode[] | undefined): void {
    for (const node of nodes ?? []) {
        if (!node || typeof node !== 'object') continue
        const args = node.args as Record<string, unknown> | undefined
        const bag = node as unknown as Record<string, unknown>
        if (args) {
            for (const key of ['body', 'result', 'absorb'] as const) {
                if (key in args && bag[key] === undefined) {
                    bag[key] = args[key]
                    delete args[key]
                }
            }
        }
        if (Array.isArray(node.body)) hoistBodyFields(node.body)
    }
}

/**
 * Coerce the prop-manifest shapes a model plausibly writes into the ones the validator
 * demands, IN PLACE: a position as `[x, y]`, numbers as strings, a float without a range
 * (given one around its default), a bool as 0/1. Anything else is left for validation to
 * name. Cheap forgiveness here saves a whole repair round there.
 */
export function normalizeStdGraph(envelope: {props?: Record<string, GraphPropSpec>; nodes?: GraphNode[]} | null | undefined): void {
    hoistBodyFields(envelope?.nodes)
    for (const spec of Object.values(envelope?.props ?? {})) {
        if (!spec || typeof spec !== 'object') continue
        const raw = spec.default as unknown
        switch (spec.kind) {
            case 'position':
                if (Array.isArray(raw) && raw.length >= 2) spec.default = {x: Number(raw[0]), y: Number(raw[1])}
                break
            case 'float': case 'angle': {
                if (typeof raw === 'string' && raw.trim() !== '' && Number.isFinite(Number(raw))) spec.default = Number(raw)
                const d = spec.default
                if (spec.kind === 'float' && typeof d === 'number') {
                    if (typeof spec.min !== 'number') spec.min = Math.min(0, d)
                    if (typeof spec.max !== 'number') spec.max = d > 0 ? d * 2 : d === 0 ? 1 : 0
                    if (spec.min >= spec.max) spec.max = spec.min + Math.max(1, Math.abs(spec.min))
                }
                break
            }
            case 'bool':
                if (raw === 0 || raw === 1 || raw === 'true' || raw === 'false') spec.default = raw === 1 || raw === 'true'
                break
        }
    }
}

/** Static check of an envelope against the catalog. Never throws. */
export function validateStdGraph(envelope: unknown): StdGraphValidation {
    try {
        const env = envelope as StdGraphEnvelope
        if (!env || typeof env !== 'object') throw new StdGraphError('envelope is not an object')
        if (env.kind !== 'std-graph') throw new StdGraphError(`kind must be "std-graph"`)
        if (typeof env.name !== 'string' || !env.name.trim()) throw new StdGraphError('name is required')
        if (!Array.isArray(env.nodes) || !env.nodes.length) throw new StdGraphError('nodes must be a non-empty list')
        if (env.nodes.length > STD_GRAPH_LIMITS.maxNodes) throw new StdGraphError(`${env.nodes.length} nodes (max ${STD_GRAPH_LIMITS.maxNodes})`)
        hoistBodyFields(env.nodes)
        const props = env.props ?? {}
        // Everything wrong is reported in one pass, so one repair can fix it all.
        const errors = graphPropErrors(props)
        const usage = {time: false, colorSpace: false}
        const handRolled: NonNullable<StdGraphValidation['handRolled']> = []
        const scope: StaticScope = {nodes: new Map(), bound: {}}
        validateNodes(env.nodes, scope, props, usage, 0, {handRolled, errors})
        let outputType: 'vec3' | 'vec4' | undefined
        try {
            if (typeof env.output !== 'string' || !env.output.startsWith('$')) throw new StdGraphError('output must name a node ("$id")')
            const out = staticArg(env.output, scope, props, 'output', usage)
            if (out.type !== 'vec4' && out.type !== 'vec3') throw new StdGraphError(`output must be a vec4 colour (or vec3 rgb), got ${out.type}`)
            outputType = out.type
        } catch (err) {
            if (!(err instanceof FollowOnError)) errors.push(err instanceof Error ? err.message : String(err))
        }
        if (errors.length) return {ok: false, error: summarizeErrors(errors), errors}
        return {ok: true, usesTime: usage.time, usesColorSpace: usage.colorSpace, outputType, handRolled}
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        return {ok: false, error: message, errors: [message]}
    }
}

// ── Build pass ──────────────────────────────────────────────────────────────────────────

interface BuildScope {
    nodes: Map<string, Val>
    parent?: BuildScope
    bound: Record<string, Val>
}

function lookupVal(scope: BuildScope, id: string): Val | undefined {
    for (let s: BuildScope | undefined = scope; s; s = s.parent) {
        const v = s.nodes.get(id)
        if (v) return v
    }
    return undefined
}

function lookupBoundVal(scope: BuildScope, name: string): Val | undefined {
    for (let s: BuildScope | undefined = scope; s; s = s.parent) {
        if (name in s.bound) return s.bound[name]
    }
    return undefined
}

function memberOf(v: Val, member: string, where: string): Val {
    if (v.type === 'record') {
        const [name, ...swizzle] = member.split('.')
        const f = v.fields[name!]
        if (!f) throw new StdGraphError(`"${where}": no field "${name}"`)
        return swizzle.length ? memberOf(f, swizzle.join('.'), where) : f
    }
    if (isNumVal(v)) return {type: swizzleType(v.type, member, where), expr: v.expr.member(member)}
    throw new StdGraphError(`"${where}": cannot read .${member}`)
}

function buildArg(arg: GraphArg, scope: BuildScope, ctx: BuildCtx, props: Record<string, GraphPropSpec>, where: string): Val {
    if (typeof arg === 'number') return {type: 'literal', value: arg}
    if (typeof arg === 'boolean') return {type: 'literal', value: arg ? 1 : 0}
    if (arg === null) throw new StdGraphError(`"${where}" is null`)
    if (typeof arg !== 'string') return {type: 'data', value: arg}
    const ref = parseRef(arg)
    if (!ref) return {type: 'enum', value: arg}
    let base: Val
    if (ref.kind === 'node') {
        const v = lookupVal(scope, ref.name)
        if (!v) throw new StdGraphError(`"${where}": unknown node "$${ref.name}"`)
        base = v
    } else if (ref.kind === 'prop') {
        const spec = props[ref.name]
        if (!spec) throw new StdGraphError(`"${where}": undeclared prop "@${ref.name}"`)
        base = {type: PROP_TYPES[spec.kind], expr: uniformOf({kind: 'prop', name: ref.name}, ctx.params), ref: {kind: 'prop', name: ref.name}, propKind: spec.kind}
    } else {
        const bound = lookupBoundVal(scope, ref.name)
        if (bound) base = bound
        else {
            switch (ref.name) {
                case 'uv': base = {type: 'vec2', expr: ctx.uv}; break
                case 'aspect': base = {type: 'f32', expr: ctx.aspect}; break
                case 'viewport': base = {type: 'vec2', expr: ctx.viewport}; break
                case 'time': base = {type: 'f32', expr: ctx.time()}; break
                case 'pointer': base = {type: 'vec2', expr: ctx.params.ctx.pointer}; break
                default: throw new StdGraphError(`"${where}": "${ref.name}" is not bound here`)
            }
        }
    }
    return ref.member ? memberOf(base, ref.member, where) : base
}

/** A static-type mirror of a build scope chain (values → their types), for typing bodies at build time. */
function staticTypesOf(scope: BuildScope, props: Record<string, GraphPropSpec>): StaticScope {
    void props
    const toStatic = (v: Val): StaticType => {
        if (v.type === 'literal') return {type: 'literal'}
        if (v.type === 'record') {
            const fields: Record<string, StaticType> = {}
            for (const [k, f] of Object.entries(v.fields)) fields[k] = toStatic(f)
            return {type: 'record', fields}
        }
        if (v.type === 'field') return {type: 'field', valueType: v.valueType}
        if (isNumVal(v)) return {type: v.type}
        return {type: v.type}
    }
    const out: StaticScope = {nodes: new Map(), bound: {}}
    for (const [k, v] of Object.entries(scope.bound)) out.bound[k] = toStatic(v)
    for (const [id, v] of scope.nodes) out.nodes.set(id, toStatic(v))
    if (scope.parent) out.parent = staticTypesOf(scope.parent, props)
    return out
}

/** A vector literal (`[x, y]` … `[x, y, z, w]`) as a value, or null when `raw` isn't one. */
function vectorLiteral(raw: GraphArg): Val | null {
    const type = vectorLiteralType(raw)
    if (!type) return null
    const [x, y, z, w] = raw as number[]
    const expr = type === 'vec2' ? m.vec2(x!, y!) : type === 'vec3' ? m.vec3(x!, y!, z!) : m.vec4(x!, y!, z!, w!)
    return {type, expr}
}

/** The GPU expression of a numeric or literal value. */
function exprOf(v: Val, where: string): Expr {
    if (v.type === 'literal') return floatE(v.value)
    if (isNumVal(v)) return v.expr
    throw new StdGraphError(`"${where}" is a ${v.type}, not a number`)
}

function hintFor(id: string): string {
    return `g_${id.replace(/[^A-Za-z0-9_]/g, '_')}`
}

function bindLocal(v: Val, id: string): Val {
    if (isNumVal(v) && !v.ref) return {type: v.type, expr: m.local(v.expr, hintFor(id))}
    return v
}

function buildNodes(nodes: GraphNode[], scope: BuildScope, ctx: BuildCtx, props: Record<string, GraphPropSpec>): void {
    for (const node of nodes) {
        const op = CATALOG[node.op]!
        const args = node.args ?? {}
        const resolved: Record<string, Val | undefined> = {}
        for (const [name, spec] of Object.entries(op.inputs)) {
            const raw = args[name]
            if (raw === undefined || raw === null) continue
            const literal = takesData(spec) ? null : vectorLiteral(raw)
            let v = literal ?? buildArg(raw, scope, ctx, props, `${node.id}.${name}`)
            if (isNumVal(v) && v.type === 'vec3' && wantsExactly(spec, 'vec4')) v = {type: 'vec4', expr: m.vec4(v.expr, 1)}
            resolved[name] = v
        }
        if (op.control === 'repeat') {
            const count = Math.round(args.count as number)
            const reduce = typeof args.reduce === 'string' ? args.reduce : 'add'
            let acc: Val | undefined
            for (let i = 0; i < count; i++) {
                const inner: BuildScope = {nodes: new Map(), parent: scope, bound: {i: {type: 'literal', value: i}}}
                buildNodes(node.body!, inner, ctx, props)
                const r = buildArg(node.result!, inner, ctx, props, `${node.id}.result`)
                const expr = exprOf(r, `${node.id}.result`)
                const type: NumericType = r.type === 'literal' ? 'f32' : (r.type as NumericType)
                if (!acc) acc = {type, expr}
                else {
                    const prev = exprOf(acc, node.id)
                    const next = reduce === 'max' ? m.max(prev, expr) : reduce === 'screen' ? m.sub(m.add(prev, expr), m.mul(prev, expr)) : m.add(prev, expr)
                    acc = {type, expr: next}
                }
            }
            scope.nodes.set(node.id, bindLocal(acc!, node.id))
            continue
        }
        if (op.control === 'fieldOf') {
            const body = node.body!
            const result = node.result!
            let valueType: NumericType = 'f32'
            const field = (params: GpuFragmentParams, coord: Expr): Expr => {
                void params
                const inner: BuildScope = {nodes: new Map(), parent: scope, bound: {p: {type: 'vec2', expr: m.local(coord, 'fp')}}}
                buildNodes(body, inner, ctx, props)
                const r = buildArg(result, inner, ctx, props, `${node.id}.result`)
                if (isNumVal(r)) valueType = r.type
                return exprOf(r, `${node.id}.result`)
            }
            // The static pass already typed the body; mirror it so `sample` knows what it yields.
            const staticScope: StaticScope = {nodes: new Map(), bound: {p: {type: 'vec2'}}}
            try {
                const outer = staticTypesOf(scope, props)
                staticScope.parent = outer
                validateNodes(body, staticScope, props, {time: false, colorSpace: false}, 1)
                const t = staticArg(result, staticScope, props, `${node.id}.result`, {time: false, colorSpace: false})
                if (isNumeric(t.type)) valueType = t.type
            } catch { /* the build pass will throw where it matters */ }
            scope.nodes.set(node.id, {type: 'field', field, valueType})
            continue
        }
        if (op.control === 'wgsl') {
            const inputVal = (arg: GraphArg, where: string): Val => vectorLiteral(arg) ?? buildArg(arg, scope, ctx, props, where)
            const spec = readWgslSpec(node, (arg, where) => {
                const v = inputVal(arg, where)
                return v.type === 'literal' ? 'f32' : (v.type as NumericType)
            })
            const argExprs = spec.params.map((p) => exprOf(inputVal(p.arg, `${node.id}.inputs.${p.name}`), node.id))
            // Hand-rolled WGSL compiles inside the renderer's validation scope, so a broken body
            // marks this composition broken and reports the compiler's message.
            markCustomWgslUsed()
            ctx.params.noteCustomWgsl?.()
            const signature = `(${spec.params.map((p) => `${p.name}: ${WGSL_TYPE_NAMES[p.type]}`).join(', ')}) -> ${WGSL_TYPE_NAMES[spec.returns]}`
            const schemas = spec.params.map((p) => WGSL_SCHEMAS[p.type])
            // The documented string form of tgpu.fn: raw WGSL, no build plugin involved. Named so
            // the emitted program shows what was hand-rolled.
            const fn = (tgpu.fn(schemas as never, WGSL_SCHEMAS[spec.returns]) as unknown as (src: string) => {$name: (n: string) => unknown})(`${signature} {\n${spec.code}\n}`)
                .$name(`handRolled_${spec.name.replace(/[^A-Za-z0-9_]/g, '_')}`)
            scope.nodes.set(node.id, bindLocal({type: spec.returns, expr: call(fn, `handRolled_${spec.name}`, argExprs)}, node.id))
            continue
        }
        if (op.control === 'march') {
            const body = node.body!
            const steps = Math.round(args.steps as number)
            const marched = volumeMarch({
                steps,
                entry: num(resolved.entry, 'entry'),
                length: num(resolved.length, 'length'),
                jitter: resolved.jitter ? num(resolved.jitter) : undefined,
                emissionGain: resolved.emissionGain ? num(resolved.emissionGain) : undefined,
                hint: hintFor(node.id),
            }, (step) => {
                const inner: BuildScope = {nodes: new Map(), parent: scope, bound: {
                    z: {type: 'f32', expr: step.z},
                    t: {type: 'literal', value: step.t},
                    i: {type: 'literal', value: step.i},
                }}
                buildNodes(body, inner, ctx, props)
                const e = buildArg(node.result!, inner, ctx, props, `${node.id}.result`)
                const emit = e.type === 'literal' || (isNumVal(e) && e.type === 'f32') ? m.splat3(exprOf(e, node.id)) : exprOf(e, node.id)
                const absorb = exprOf(buildArg(node.absorb!, inner, ctx, props, `${node.id}.absorb`), `${node.id}.absorb`)
                return {emit, absorb}
            })
            scope.nodes.set(node.id, {type: 'record', fields: {color: {type: 'vec3', expr: marched.color}, transmittance: {type: 'vec3', expr: marched.transmittance}}})
            continue
        }
        let out: Val
        try {
            out = op.build(resolved, ctx)
        } catch (err) {
            if (err instanceof StdGraphError && !err.nodeId) throw new StdGraphError(err.message, node.id)
            throw err
        }
        scope.nodes.set(node.id, bindLocal(out, node.id))
    }
}

/** Interpret the graph against composition params — the generator's `paint`. */
function paintGraph(env: StdGraphEnvelope, params: GpuFragmentParams): Expr {
    const props = env.props ?? {}
    const uv = params.uvContext ?? params.ctx.uv
    const viewport = params.effectiveViewportSize ?? params.ctx.viewportSize
    const aspect = m.local(m.div(viewport.member('x'), m.max(viewport.member('y'), 1e-6)), 'aspect')
    let clock: Expr | undefined
    const ctx: BuildCtx = {
        params, uv, aspect, viewport,
        time: () => (clock ??= m.local(animatedTime(params), 'clock')),
        colorSpaceMode: colorSpaceModeOf(params.propValues.colorSpace),
        resolveRef: (ref) => buildArg(ref, scope, ctx, props, 'layers'),
    }
    const scope: BuildScope = {nodes: new Map(), bound: {}}
    buildNodes(env.nodes, scope, ctx, props)
    const out = buildArg(env.output, scope, ctx, props, 'output')
    if (out.type === 'vec4') return out.expr
    if (out.type === 'vec3') return m.vec4(out.expr, 1)
    throw new StdGraphError(`output is a ${out.type}, not a colour`)
}

// ── Props → std props ───────────────────────────────────────────────────────────────────

/** The prop transforms a manifest can need, by the name `shaders/std` exports them under. */
export const GRAPH_PROP_TRANSFORMS = {transformColor, transformPosition, transformBool01, transformColorSpace} as const
export type GraphPropTransformName = keyof typeof GRAPH_PROP_TRANSFORMS

/** A std prop config as plain data: its transform named rather than held, so it can be written as source. */
export type GraphPropConfigData = Omit<StdPropConfig<unknown>, 'transform'> & {transform?: GraphPropTransformName}

/**
 * Expand a props manifest to std prop configs as data (shared with `wgsl` envelopes and the
 * component source emitter). A component that reads the clock gains a `speed` prop (unless it
 * declares one); `colorSpace` adds the library's colour-space select, which the graph's colour
 * words read.
 */
export function graphPropData(
    props: Record<string, GraphPropSpec>,
    options: {usesTime: boolean; colorSpace: boolean},
): Record<string, GraphPropConfigData> {
    const out: Record<string, GraphPropConfigData> = {}
    for (const [name, spec] of Object.entries(props)) {
        const label = spec.label ?? name.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase())
        const group = spec.group ?? (spec.kind === 'color' ? 'Colors' : 'Effect')
        const base = {description: spec.description ?? label}
        switch (spec.kind) {
            case 'float':
                out[name] = {...base, default: spec.default, ui: {type: ['range', 'map'], min: spec.min, max: spec.max, step: spec.step ?? niceStep(spec.min!, spec.max!), label, group}}
                break
            case 'angle':
                out[name] = {...base, default: spec.default, ui: {type: ['range', 'map'], min: spec.min ?? 0, max: spec.max ?? 360, step: spec.step ?? 1, label, group}}
                break
            case 'bool':
                out[name] = {...base, default: spec.default, transform: 'transformBool01', ui: {type: 'checkbox', label, group}}
                break
            case 'color':
                out[name] = {...base, default: spec.default, transform: 'transformColor', ui: {type: 'color', label, group}}
                break
            case 'position':
                out[name] = {...base, default: spec.default, transform: 'transformPosition', ui: {type: 'position', label, group: spec.group ?? 'Position'}}
                break
        }
    }
    if (options.usesTime && !out.speed) {
        out.speed = {default: 1, description: 'Animation speed', ui: {type: 'range', min: 0, max: 5, step: 0.05, label: 'Speed', group: 'Animation'}}
    }
    if (options.colorSpace) {
        out.colorSpace = {
            default: 'linear', transform: 'transformColorSpace', compileTime: true,
            description: 'Color space for color interpolation',
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'},
        }
    }
    return out
}

/** Prop configs as data → the real thing: each named transform swapped for the function. */
export function propConfigsFromData(data: Record<string, GraphPropConfigData>): Record<string, StdPropConfig<unknown>> {
    const out: Record<string, StdPropConfig<unknown>> = {}
    for (const [name, {transform, ...config}] of Object.entries(data)) {
        out[name] = transform ? {...config, transform: GRAPH_PROP_TRANSFORMS[transform] as (v: unknown) => unknown} : config
    }
    return out
}

/** Lower a props manifest to std prop configs. */
export function graphPropConfigs(
    props: Record<string, GraphPropSpec>,
    options: {usesTime: boolean; colorSpace: boolean},
): Record<string, StdPropConfig<unknown>> {
    return propConfigsFromData(graphPropData(props, options))
}

function niceStep(min: number, max: number): number {
    const span = Math.abs(max - min)
    if (span <= 1) return 0.01
    if (span <= 10) return 0.1
    if (span <= 100) return 1
    return Math.pow(10, Math.floor(Math.log10(span)) - 2)
}

/** A short stable fingerprint of a string (djb2, base36). */
export function fingerprint(text: string): string {
    let h = 5381
    for (let i = 0; i < text.length; i++) h = ((h << 5) + h + text.charCodeAt(i)) >>> 0
    return h.toString(36)
}

/** Lower an envelope to an ordinary generator definition. Throws StdGraphError on an invalid graph. */
export function defineStdGraph(envelope: StdGraphEnvelope): GpuShaderDefinition {
    const check = validateStdGraph(envelope)
    if (!check.ok) throw new StdGraphError(check.error!)
    const usesTime = !!check.usesTime || !!envelope.props?.speed
    return {
        // The renderer's pipeline cache keys on name + structure; an edited graph can replace the
        // body under the same name, so the body's fingerprint has to be part of the key.
        revision: fingerprint(JSON.stringify({p: envelope.props ?? {}, n: envelope.nodes, o: envelope.output})),
        ...defineStd<ComponentProps>({
        name: envelope.name,
        role: 'generator',
        category: 'Custom',
        description: envelope.description ?? envelope.name,
        boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
        acceptsUVContext: true,
        ...(usesTime ? {animatedTime: {speed: 'speed'}} : {}),
        props: graphPropConfigs(envelope.props ?? {}, {usesTime, colorSpace: !!check.usesColorSpace}) as never,
        paint: (params) => paintGraph(envelope, params),
        }),
    }
}

// ── Manifest (reference text) ──────────────────────────────────────────────────────────────

const SECTION_ORDER = ['coordinates', 'geometry', 'masks', 'motion', 'noise', 'fields', 'color', 'light', 'compose', 'figures', 'control', 'algebra', 'vectors', 'escape']
const SECTION_NOTES: Record<string, string> = {
    coordinates: 'Where a pixel is. Start here: `uv` is 0..1 with y = 0 at the TOP; `aspectUV`/`centered` give square units.',
    geometry: 'Signed distance fields (negative inside). Shape them with masks.',
    masks: 'Distances → coverage 0..1.',
    motion: '`time` is the node clock (seconds × the Speed prop the component gains automatically).',
    noise: 'Expression-level noise you sample at coordinates you build. Signed noise: multiply/offset into the range you need.',
    fields: 'Composition-time Field objects (the gradient/noise pipeline). Build, then `sample`/`ramp`/`normalFrom`. A `fieldOf` body turns any expression into a field you can take slopes of.',
    color: 'Colours are vec4 straight-alpha in linear light. Prop colours are vec4 already. Mixing respects the user\'s Color Space prop.',
    light: '2.5D lighting, one convention: normals AND light vectors point toward the viewer (+z). normalFrom/domeNormal give +z normals; keyLight gives a +z light; shine/lambert/grazing/fresnel consume them.',
    compose: 'Stacking and output.',
    figures: 'Complete drawn paints.',
    control: 'The body-carrying nodes: repeat (N things), fieldOf (a reusable function of a coordinate), march (a volumetric ray march).',
    algebra: 'Seasoning — thresholds, ramps, gains over what the words hand you. Not structure.',
    vectors: 'Constructors.',
    escape: 'LAST RESORT. One node type for the part no word covers. It is logged with its name and reason, reviewed for promotion into the vocabulary, and unmeasured for performance until then.',
}

function typeLabel(spec: InputSpec): string {
    const types = Array.isArray(spec.type) ? spec.type : [spec.type]
    return types.map((t) => t.startsWith('prop:') ? `@${t.slice(5)}Prop` : t === 'enum' ? (spec.values?.join('|') ?? 'enum') : t).join('|')
}

function outputLabel(op: OpDef): string {
    const rule = op.output
    if (rule === 'broadcast') return 'same as inputs'
    if (typeof rule === 'function') return 'depends'
    if (typeof rule === 'object') return `{${Object.entries(rule.record).map(([k, v]) => `${k}: ${v}`).join(', ')}}`
    return rule
}

let manifestCache: string | null = null

/** The catalog as reference text: one line per op with its inputs and output, grouped by section. */
export function renderStdGraphManifest(): string {
    if (manifestCache) return manifestCache
    const lines: string[] = []
    for (const section of SECTION_ORDER) {
        const ops = Object.entries(CATALOG).filter(([, op]) => op.section === section)
        if (!ops.length) continue
        lines.push(`## ${section}${SECTION_NOTES[section] ? ` — ${SECTION_NOTES[section]}` : ''}`)
        for (const [name, op] of ops) {
            const sig = Object.entries(op.inputs).map(([k, spec]) => `${k}${spec.optional ? '?' : ''}: ${typeLabel(spec)}`).join(', ')
            const extra = Object.entries(op.inputs).filter(([, s]) => s.doc).map(([k, s]) => `${k} = ${s.doc}`).join('; ')
            lines.push(`- ${name}(${sig}) → ${outputLabel(op)} — ${op.doc}${extra ? `. [${extra}]` : ''}`)
        }
        lines.push('')
    }
    manifestCache = lines.join('\n')
    return manifestCache
}

// ── GPU-free resolve gate ───────────────────────────────────────────────────────────────

const RootContainer: GpuShaderDefinition = {
    name: 'Root',
    props: {} as never,
    fragment: ({childNode}) => childNode ?? (undefined as never),
}

function stubRoot(): never {
    const buffer = {
        patch: () => undefined,
        write: () => undefined,
        destroy: () => undefined,
        $usage() { return buffer },
    }
    return {createBuffer: () => buffer, createBindGroup: () => ({})} as never
}

/**
 * Compose a graph component alone under a root and resolve the final pass to WGSL with no
 * GPU device — the same gate the per-shader vitest suites run. A graph that validates but
 * cannot resolve (a word misused in a way the static types allow) fails here, with the
 * resolver's message, before a browser ever sees it.
 */
export function resolveStdGraphWgsl(envelope: StdGraphEnvelope, propOverrides: Record<string, unknown> = {}): {ok: true; wgsl: string} | {ok: false; error: string} {
    try {
        const def = defineStdGraph(envelope)
        const store = createUniformStore(stubRoot(), {systemSchema: SystemUniforms})
        const defaults: Record<string, unknown> = {}
        for (const [name, cfg] of Object.entries(def.props)) defaults[name] = (cfg as PropConfig<unknown>).default
        const props = {...defaults, ...propOverrides}
        const specs: {id: string; def: GpuShaderDefinition; parentId: string | null; props: Record<string, unknown>}[] = [
            {id: 'root', def: RootContainer, parentId: null, props: {}},
            {id: 'g', def, parentId: 'root', props},
        ]
        const handles: Record<string, unknown> = {}
        for (const s of specs) {
            const map = createGpuUniformsMap(s.def as never, s.props, s.id)
            const inits: FieldInit[] = Object.entries(map).map(([name, u]) => ({name, initial: u.value, transform: u.transform, cpu: u.cpu, schema: u.schema}))
            inits.push({name: '_opacity', schema: d.f32, initial: 1})
            if (s.def.animatedTime) inits.push({name: '_animTime', schema: d.f32, initial: 0})
            handles[s.id] = store.defineNode(s.id, inits)
        }
        store.finalize()
        const nodes = new Map<string, RegistryNode>()
        for (const s of specs) {
            nodes.set(s.id, {
                id: s.id, componentName: s.def.name, parentId: s.parentId, definition: s.def,
                metadata: {blendMode: 'normal', opacity: undefined, renderOrder: 0} as NodeMetadata,
                handles: handles[s.id] as never,
            })
        }
        const registry: RegistryView = {
            rootId: 'root',
            getNode: (id) => nodes.get(id),
            getChildren: (parentId) => (parentId === 'root' ? [nodes.get('g')!] : []),
            resolveCustomId: () => null,
            store,
        }
        const wgsl = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        return {ok: true, wgsl}
    } catch (err) {
        return {ok: false, error: err instanceof Error ? err.message : String(err)}
    }
}
