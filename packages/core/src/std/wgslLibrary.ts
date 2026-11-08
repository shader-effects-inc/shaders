/**
 * The WGSL library: tuned functions a component calls by name from any WGSL body.
 *
 * Import a function from `shaders/std/wgsl`, list it in the definition's `inputs`, and call it by
 * name. Only the functions a component lists are linked, so an app ships only what it uses.
 *
 * @example
 * ```ts
 * import {noise2, aaStep} from 'shaders/std/wgsl'
 *
 * inputs: {noise2, aaStep},
 * paint: wgsl`
 *   let n = noise2(uv * 6.0);
 *   return vec4f(vec3f(aaStep(0.5, n)), 1.0);
 * `,
 * ```
 */
// Maintainer notes. Each library unit is plain WGSL text (structs, one public fn, private helpers
// prefixed `_<name>_`) plus the units it calls. Two consumers: a fragment body links the public fn
// as a string-form `tgpu.fn` (deps through `$uses`, structs as `d.struct`); a compute module
// pastes the dependency-closed text (`libSource`). The tgpu objects are built once per unit and
// shared by every composition, so two components listing `noise2` link the same function.
import {tgpu} from '../gpu/kit/index'
import {
    parseWgslDecls,
    replaceIdentifier,
    identifiersIn,
    schemaForWgslType,
    structSchemas,
    type WgslFnDecl,
} from './wgslSource'

/** One library function: its WGSL, the functions it calls, and its documentation. */
export interface WgslLibFn {
    readonly kind: 'wgslLib'
    /** The name a body calls it by, and the name of the `fn` its `code` declares. */
    readonly name: string
    /** The family it is listed under in the docs (`noise`, `colour`, `sdf`, …). */
    readonly family: string
    /** The signature as documentation, e.g. `noise2(p: vec2f) -> f32` and what the ranges are. */
    readonly signature: string
    /** One or two plain sentences: what it returns and when to reach for it. */
    readonly doc: string
    /** The WGSL: structs it returns, the public fn, private helpers named `_<name>_…`. */
    readonly code: string
    /** Other library functions the code calls. */
    readonly deps: readonly WgslLibFn[]
    /** Struct types the public fn returns or takes, exposed to bodies by name. */
    readonly structs: readonly string[]
}

/** Declare a library function. */
/* @__NO_SIDE_EFFECTS__ */
export function libFn(spec: {
    name: string
    family: string
    signature: string
    doc: string
    code: string
    deps?: readonly WgslLibFn[]
}): WgslLibFn {
    const decls = parseWgslDecls(spec.code)
    if (!decls.fns.some((f) => f.name === spec.name)) {
        throw new Error(`libFn ${spec.name}: the code must declare fn ${spec.name}`)
    }
    for (const f of decls.fns) {
        if (f.name !== spec.name && !f.name.startsWith(`_${spec.name}_`)) {
            throw new Error(`libFn ${spec.name}: private helper ${f.name} must be named _${spec.name}_…`)
        }
    }
    return {
        kind: 'wgslLib',
        name: spec.name,
        family: spec.family,
        signature: spec.signature,
        doc: spec.doc,
        code: spec.code,
        deps: spec.deps ?? [],
        structs: decls.structs.map((s) => s.name),
    }
}

/** True for a value made by `libFn` (a library function listed in `inputs`). */
export function isWgslLibFn(value: unknown): value is WgslLibFn {
    return typeof value === 'object' && value !== null && (value as {kind?: unknown}).kind === 'wgslLib'
}

/** The units a list depends on, deps first, each once. */
export function libClosure(fns: readonly WgslLibFn[]): WgslLibFn[] {
    const out: WgslLibFn[] = []
    const seen = new Set<WgslLibFn>()
    const visit = (fn: WgslLibFn, stack: Set<WgslLibFn>) => {
        if (seen.has(fn)) return
        if (stack.has(fn)) throw new Error(`wgsl library: ${fn.name} depends on itself`)
        stack.add(fn)
        for (const dep of fn.deps) visit(dep, stack)
        stack.delete(fn)
        seen.add(fn)
        out.push(fn)
    }
    for (const fn of fns) visit(fn, new Set())
    return out
}

/** Inline module-scope constants into a block's fn bodies (string fns cannot see module consts). */
/**
 * Replace each helper `const` with its value. Consts may be written in terms of each other in any
 * order (WGSL allows it), so values are resolved against each other first. A scalar or vector
 * const with a declared type keeps it (`const K: f32 = 1;` reads as `f32(1)`).
 */
export function inlineConsts(text: string, consts: readonly {name: string; value: string; type?: string | null}[]): string {
    if (!consts.length) return text
    const values = new Map(consts.map((c) => {
        const typed = c.type && !c.type.startsWith('array') ? `${c.type}(${c.value})` : c.value
        return [c.name, `(${typed})`]
    }))
    // Resolve consts that refer to other consts (a chain at most `consts.length` deep).
    for (let pass = 0; pass < consts.length; pass++) {
        let changed = false
        for (const [name, value] of values) {
            let next = value
            for (const [other, v] of values) if (other !== name) next = replaceIdentifier(next, other, v)
            if (next !== value) {
                values.set(name, next)
                changed = true
            }
        }
        if (!changed) break
    }
    let out = text
    for (const [name, value] of values) out = replaceIdentifier(out, name, value)
    return out
}

/**
 * The plain WGSL for a set of library functions and everything they call, deduplicated, in
 * dependency order. For compute modules, where it is pasted at module scope as is.
 */
export function libSource(fns: readonly WgslLibFn[]): string {
    return libClosure(fns).map((fn) => fn.code.trim()).join('\n\n')
}

// ── Linking into TypeGPU (fragment bodies) ──────────────────────────────────────────────

type StringFnShell = (impl: string) => {$uses: (deps: Record<string, unknown>) => unknown; $name?: (n: string) => unknown}

/** Every linked declaration across all units: name → tgpu fn or struct schema. Built lazily, shared. */
const linkedDecls = new Map<string, unknown>()
/** The unit that owns each linked name, so two units cannot quietly claim the same name. */
const declOwner = new Map<string, WgslLibFn>()
const linkedUnits = new WeakSet<WgslLibFn>()

function linkUnit(unit: WgslLibFn): void {
    if (linkedUnits.has(unit)) return
    for (const dep of unit.deps) linkUnit(dep)
    const decls = parseWgslDecls(unit.code)
    const claim = (name: string) => {
        const owner = declOwner.get(name)
        if (owner && owner !== unit && owner.code !== unit.code) {
            throw new Error(`wgsl library: ${unit.name} and ${owner.name} both declare '${name}'`)
        }
        declOwner.set(name, unit)
    }
    const structs = structSchemas(decls.structs, linkedStructs())
    for (const s of decls.structs) {
        claim(s.name)
        linkedDecls.set(s.name, structs.get(s.name))
    }
    for (const fn of orderFns(decls.fns)) {
        claim(fn.name)
        linkedDecls.set(fn.name, stringFn(fn, decls.consts, structs, linkedDecls))
    }
    linkedUnits.add(unit)
}

function linkedStructs(): Map<string, unknown> {
    const out = new Map<string, unknown>()
    for (const [name, value] of linkedDecls) {
        if (declOwner.get(name)?.structs.includes(name)) out.set(name, value)
    }
    return out
}

/** Functions ordered so each comes after the ones it calls (WGSL has no recursion). */
export function orderFns(fns: readonly WgslFnDecl[]): WgslFnDecl[] {
    const byName = new Map(fns.map((f) => [f.name, f]))
    const out: WgslFnDecl[] = []
    const done = new Set<string>()
    const visit = (fn: WgslFnDecl, stack: Set<string>) => {
        if (done.has(fn.name)) return
        if (stack.has(fn.name)) throw new Error(`wgsl: fn ${fn.name} calls itself (WGSL has no recursion)`)
        stack.add(fn.name)
        for (const id of identifiersIn(fn.body)) {
            const callee = byName.get(id)
            if (callee && callee !== fn) visit(callee, stack)
        }
        stack.delete(fn.name)
        done.add(fn.name)
        out.push(fn)
    }
    for (const fn of fns) visit(fn, new Set())
    return out
}

/**
 * A string-form `tgpu.fn` for one parsed declaration. Calls to other declarations and struct
 * names in its text are linked through `$uses`, from `scope` (name → tgpu fn / struct schema).
 */
export function stringFn(
    fn: WgslFnDecl,
    consts: readonly {name: string; value: string}[],
    structs: ReadonlyMap<string, unknown>,
    scope: ReadonlyMap<string, unknown>,
    extraParams: readonly {name: string; type: string; schema: unknown}[] = [],
    rewriteBody: (body: string) => string = (b) => b,
): unknown {
    const params = [
        ...fn.params.map((p) => ({...p, schema: schemaForWgslType(p.type, structs)})),
        ...extraParams,
    ]
    const ret = fn.ret ? schemaForWgslType(fn.ret, structs) : undefined
    const header = `(${params.map((p) => `${p.name}: ${p.type}`).join(', ')})${fn.ret ? ` -> ${fn.ret}` : ''}`
    const body = rewriteBody(inlineConsts(fn.body, consts))
    const shell = (ret === undefined
        ? (tgpu.fn as unknown as (args: unknown[]) => StringFnShell)(params.map((p) => p.schema))
        : (tgpu.fn as unknown as (args: unknown[], r: unknown) => StringFnShell)(params.map((p) => p.schema), ret))
    const made = shell(`${header} ${body}`)
    const uses: Record<string, unknown> = {}
    const ids = identifiersIn(`${header} ${body}`)
    for (const id of ids) {
        if (id === fn.name) continue
        const linked = scope.get(id) ?? structs.get(id)
        if (linked !== undefined) uses[id] = linked
    }
    const withUses = Object.keys(uses).length ? made.$uses(uses) : made
    return (withUses as {$name?: (n: string) => unknown}).$name?.(fn.name) ?? withUses
}

/**
 * The externals a body needs to call these library functions by name: each public fn (and every
 * struct type the units expose), as tgpu objects keyed by the name the body writes.
 */
export function linkLibFns(fns: readonly WgslLibFn[]): Record<string, unknown> {
    const out: Record<string, unknown> = {}
    for (const unit of libClosure(fns)) {
        linkUnit(unit)
    }
    for (const unit of fns) {
        out[unit.name] = linkedDecls.get(unit.name)
        for (const s of unit.structs) out[s] = linkedDecls.get(s)
    }
    return out
}

/** The struct schemas the listed units expose, for helpers that name them in signatures. */
export function libStructSchemas(fns: readonly WgslLibFn[]): Map<string, unknown> {
    linkLibFns(fns)
    const out = new Map<string, unknown>()
    for (const unit of libClosure(fns)) for (const s of unit.structs) out.set(s, linkedDecls.get(s))
    return out
}
