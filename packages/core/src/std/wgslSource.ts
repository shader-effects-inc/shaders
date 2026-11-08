/**
 * Reading WGSL source as declarations: the structs, functions and constants in a block of WGSL
 * text, and the TypeGPU schema for a WGSL type spelled as text.
 *
 * This is what lets plain WGSL (a component's `helpers`, the WGSL library, an agent's kernel)
 * join the engine's TypeGPU-built shaders: each declaration becomes a string-form `tgpu.fn` or a
 * `d.struct`, linked to the others by name.
 */
// Maintainer notes. A deliberately small reader: it understands the declaration shapes that
// appear at module scope in hand-written WGSL (`struct`, `fn`, `const`, attributes it skips) and
// leaves bodies as text. Brace matching ignores braces inside comments. It is not a validator:
// WGSL errors surface when the composed module compiles, with the engine's error mapping.
import {d} from '../gpu/kit/index'

/** One `struct Name { field: type, … }` declaration. */
export interface WgslStructDecl {
    readonly name: string
    readonly fields: readonly {readonly name: string; readonly type: string}[]
    readonly text: string
}

/** One `fn name(params) -> ret { body }` declaration. */
export interface WgslFnDecl {
    readonly name: string
    readonly params: readonly {readonly name: string; readonly type: string}[]
    /** The return type as written, or null for a function that returns nothing. */
    readonly ret: string | null
    /** The body including its outer braces. */
    readonly body: string
    readonly text: string
}

/** One module-scope `const NAME[: type] = value;` declaration. */
export interface WgslConstDecl {
    readonly name: string
    readonly type: string | null
    readonly value: string
    readonly text: string
}

export interface WgslDecls {
    readonly structs: readonly WgslStructDecl[]
    readonly fns: readonly WgslFnDecl[]
    readonly consts: readonly WgslConstDecl[]
}

/** Thrown when a declaration block cannot be read. Carries a 1-based line number into the block. */
export class WgslSourceError extends Error {
    constructor(message: string, readonly line: number) {
        super(`line ${line}: ${message}`)
        this.name = 'WgslSourceError'
    }
}

/** Replace comments with spaces, keeping every newline so line numbers stay true. */
export function blankComments(source: string): string {
    return source
        .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, ' '))
        .replace(/\/\/[^\n]*/g, (m) => ' '.repeat(m.length))
}

function lineAt(source: string, index: number): number {
    let line = 1
    for (let i = 0; i < index && i < source.length; i++) if (source[i] === '\n') line++
    return line
}

/** Index of the brace closing the one at `open` (which must be `{`), comments already blanked. */
function matchBrace(text: string, open: number): number {
    let depth = 0
    for (let i = open; i < text.length; i++) {
        const ch = text[i]
        if (ch === '{') depth++
        else if (ch === '}') {
            depth--
            if (depth === 0) return i
        }
    }
    return -1
}

/** Split `a: T, b: array<U, 4>` on top-level commas (not those inside `<…>` or `(…)`). */
function splitTopLevel(text: string, sep = ','): string[] {
    const out: string[] = []
    let depth = 0
    let start = 0
    for (let i = 0; i < text.length; i++) {
        const ch = text[i]
        if (ch === '<' || ch === '(' || ch === '[') depth++
        else if (ch === '>' || ch === ')' || ch === ']') depth--
        else if (ch === sep && depth === 0) {
            out.push(text.slice(start, i))
            start = i + 1
        }
    }
    out.push(text.slice(start))
    return out.map((s) => s.trim()).filter((s) => s.length > 0)
}

function parseNameType(entry: string, line: number, what: string): {name: string; type: string} {
    // Drop attributes on a parameter or field (`@location(0) uv: vec2f`, `@builtin(…)`).
    const cleaned = entry.replace(/@\w+(\([^)]*\))?\s*/g, '').trim()
    const m = cleaned.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([\s\S]+)$/)
    if (!m) throw new WgslSourceError(`cannot read ${what} '${entry.trim()}' (expected name: type)`, line)
    return {name: m[1], type: m[2].trim()}
}

/**
 * Read the module-scope declarations in a block of WGSL. Structs, functions and constants are
 * returned in source order. Anything else at module scope (a stray statement, a `var`, an
 * `override`) is an error, because a body or helper block cannot declare resources.
 */
export function parseWgslDecls(source: string): WgslDecls {
    const text = blankComments(source)
    const structs: WgslStructDecl[] = []
    const fns: WgslFnDecl[] = []
    const consts: WgslConstDecl[] = []
    let i = 0
    while (i < text.length) {
        const ws = text.slice(i).match(/^\s+/)
        if (ws) {
            i += ws[0].length
            continue
        }
        const rest = text.slice(i)
        const attr = rest.match(/^@\w+(\([^)]*\))?/)
        if (attr) {
            i += attr[0].length
            continue
        }
        const struct = rest.match(/^struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{/)
        if (struct) {
            const open = i + struct[0].length - 1
            const close = matchBrace(text, open)
            if (close < 0) throw new WgslSourceError(`struct ${struct[1]} has no closing brace`, lineAt(text, i))
            const inner = text.slice(open + 1, close)
            const fields = splitTopLevel(inner).map((f) => parseNameType(f, lineAt(text, open), `field of struct ${struct[1]}`))
            let end = close + 1
            while (end < text.length && /[\s;]/.test(text[end])) {
                if (text[end] === ';') {
                    end++
                    break
                }
                end++
            }
            structs.push({name: struct[1], fields, text: source.slice(i, close + 1)})
            i = end
            continue
        }
        const fn = rest.match(/^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/)
        if (fn) {
            const parenOpen = i + fn[0].length - 1
            let depth = 0
            let parenClose = -1
            for (let k = parenOpen; k < text.length; k++) {
                if (text[k] === '(') depth++
                else if (text[k] === ')') {
                    depth--
                    if (depth === 0) {
                        parenClose = k
                        break
                    }
                }
            }
            if (parenClose < 0) throw new WgslSourceError(`fn ${fn[1]} has an unclosed parameter list`, lineAt(text, i))
            const params = splitTopLevel(text.slice(parenOpen + 1, parenClose)).map((p) =>
                parseNameType(p, lineAt(text, parenOpen), `parameter of fn ${fn[1]}`))
            const open = text.indexOf('{', parenClose)
            if (open < 0) throw new WgslSourceError(`fn ${fn[1]} has no body`, lineAt(text, i))
            const between = text.slice(parenClose + 1, open).trim()
            let ret: string | null = null
            if (between) {
                const r = between.match(/^->\s*(?:@\w+(?:\([^)]*\))?\s*)*([\s\S]+)$/)
                if (!r) throw new WgslSourceError(`fn ${fn[1]}: cannot read the return type '${between}'`, lineAt(text, parenClose))
                ret = r[1].trim()
            }
            const close = matchBrace(text, open)
            if (close < 0) throw new WgslSourceError(`fn ${fn[1]} has no closing brace`, lineAt(text, open))
            fns.push({name: fn[1], params, ret, body: source.slice(open, close + 1), text: source.slice(i, close + 1)})
            i = close + 1
            continue
        }
        const constDecl = rest.match(/^const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([^=]+?))?\s*=/)
        if (constDecl) {
            const semi = text.indexOf(';', i)
            if (semi < 0) throw new WgslSourceError(`const ${constDecl[1]} has no closing ';'`, lineAt(text, i))
            consts.push({
                name: constDecl[1],
                type: constDecl[2]?.trim() ?? null,
                value: source.slice(i + constDecl[0].length, semi).trim(),
                text: source.slice(i, semi + 1),
            })
            i = semi + 1
            continue
        }
        const word = rest.match(/^[^\s]+/)?.[0] ?? rest.slice(0, 20)
        throw new WgslSourceError(`only struct, fn and const declarations can sit here (found '${word}')`, lineAt(text, i))
    }
    return {structs, fns, consts}
}

// ── Types ───────────────────────────────────────────────────────────────────────────────

const SCALARS: Record<string, unknown> = {
    f32: d.f32, i32: d.i32, u32: d.u32, bool: d.bool, f16: d.f16,
    vec2f: d.vec2f, vec3f: d.vec3f, vec4f: d.vec4f,
    vec2i: d.vec2i, vec3i: d.vec3i, vec4i: d.vec4i,
    vec2u: d.vec2u, vec3u: d.vec3u, vec4u: d.vec4u,
    vec2h: d.vec2h, vec3h: d.vec3h, vec4h: d.vec4h,
    mat2x2f: d.mat2x2f, mat3x3f: d.mat3x3f, mat4x4f: d.mat4x4f,
}

const GENERIC_VEC: Record<string, Record<string, string>> = {
    vec2: {f32: 'vec2f', i32: 'vec2i', u32: 'vec2u', f16: 'vec2h'},
    vec3: {f32: 'vec3f', i32: 'vec3i', u32: 'vec3u', f16: 'vec3h'},
    vec4: {f32: 'vec4f', i32: 'vec4i', u32: 'vec4u', f16: 'vec4h'},
}

/**
 * The TypeGPU schema for a WGSL type written as text: scalars, vectors (both `vec3f` and
 * `vec3<f32>` spellings), the square float matrices, fixed-size arrays, and the structs named in
 * `structs`. Throws for anything else (pointers, textures, atomics), which a helper function may
 * not take or return.
 */
export function schemaForWgslType(type: string, structs: ReadonlyMap<string, unknown> = new Map()): unknown {
    const t = type.replace(/\s+/g, '')
    if (SCALARS[t]) return SCALARS[t]
    const generic = t.match(/^(vec[234])<(f32|i32|u32|f16)>$/)
    if (generic) return SCALARS[GENERIC_VEC[generic[1]][generic[2]]]
    const mat = t.match(/^mat([234])x([234])<f32>$/)
    if (mat && mat[1] === mat[2]) return SCALARS[`mat${mat[1]}x${mat[2]}f`]
    const arr = t.match(/^array<(.+),(\d+)u?>$/)
    if (arr) return d.arrayOf(schemaForWgslType(arr[1], structs) as never, Number(arr[2]))
    const named = structs.get(t)
    if (named) return named
    throw new Error(`wgsl: the type '${type}' cannot be passed to or returned from a helper function`)
}

/** TypeGPU struct schemas for parsed struct declarations, in dependency order, keyed by name. */
export function structSchemas(decls: readonly WgslStructDecl[], known: ReadonlyMap<string, unknown> = new Map()): Map<string, unknown> {
    const out = new Map<string, unknown>(known)
    const pending = [...decls]
    let guard = pending.length * pending.length + 1
    while (pending.length > 0 && guard-- > 0) {
        const decl = pending.shift()!
        try {
            const fields: Record<string, unknown> = {}
            for (const f of decl.fields) fields[f.name] = schemaForWgslType(f.type, out)
            out.set(decl.name, (d.struct(fields as never) as unknown as {$name: (n: string) => unknown}).$name(decl.name))
        } catch {
            pending.push(decl) // a field names a struct declared later; retry after it
        }
    }
    if (pending.length > 0) {
        throw new Error(`wgsl: cannot build struct ${pending[0].name} (a field type is unknown or the structs refer to each other in a loop)`)
    }
    return out
}

/** Every identifier in a block of WGSL (comments ignored, member accesses excluded). */
export function identifiersIn(source: string): Set<string> {
    const out = new Set<string>()
    for (const m of blankComments(source).matchAll(/(?<![\w.])[A-Za-z_][A-Za-z0-9_]*/g)) out.add(m[0])
    return out
}

/** Replace every bare use of `name` (not a member access, not part of a longer word). */
export function replaceIdentifier(source: string, name: string, replacement: string): string {
    return source.replace(new RegExp(`(?<![\\w.])${name}(?![\\w])`, 'g'), replacement)
}

/** Insert `extra` as trailing arguments to every call of `name(` in `text`. */
export function appendCallArgs(text: string, name: string, extra: readonly string[]): string {
    if (extra.length === 0) return text
    const re = new RegExp(`(?<![\\w.])${name.replace('.', '\\.')}\\s*\\(`, 'g')
    let out = ''
    let last = 0
    for (const m of text.matchAll(re)) {
        const open = (m.index ?? 0) + m[0].length - 1
        let depth = 0
        let close = -1
        for (let i = open; i < text.length; i++) {
            if (text[i] === '(') depth++
            else if (text[i] === ')') {
                depth--
                if (depth === 0) {
                    close = i
                    break
                }
            }
        }
        if (close < 0) continue
        const inner = text.slice(open + 1, close).trim()
        out += text.slice(last, close) + (inner ? ', ' : '') + extra.join(', ')
        last = close
    }
    return out + text.slice(last)
}

/** Names a body declares itself (`let x`, `var y`, `const z`, `for (var i …)`), comments ignored. */
export function localsIn(body: string): Set<string> {
    const out = new Set<string>()
    for (const m of blankComments(body).matchAll(/\b(?:let|var|const)\s+([A-Za-z_][A-Za-z0-9_]*)/g)) out.add(m[1])
    return out
}

/** Names a body declares at its own top level (outside every `{}` block): the ones that would clash
 *  with a parameter of the same name. A declaration inside a loop or branch shadows legally. */
export function topLevelLocalsIn(body: string): Set<string> {
    const text = blankComments(body)
    const out = new Set<string>()
    let depth = 0
    const decl = /\b(?:let|var|const)\s+([A-Za-z_][A-Za-z0-9_]*)/y
    for (let i = 0; i < text.length; i++) {
        const ch = text[i]
        if (ch === '{') depth++
        else if (ch === '}') depth = Math.max(0, depth - 1)
        else if (depth === 0 && /[lvc]/.test(ch!) && (i === 0 || !/[A-Za-z0-9_]/.test(text[i - 1]!))) {
            decl.lastIndex = i
            const m = decl.exec(text)
            if (m) out.add(m[1]!)
        }
    }
    return out
}
