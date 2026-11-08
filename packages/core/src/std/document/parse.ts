/**
 * Read a component's TypeScript without running it. `parseComponentSource` walks the syntax tree
 * of a `defineShader` file and turns it into the component's IR: plain JSON that names every word
 * it calls, every prop it reads and every WGSL body, and that `evaluateComponentIR` turns back into
 * a working definition. Nothing in the file executes.
 *
 * The accepted file is the agent format: imports from `shaders/std` and `shaders/std/wgsl`,
 * `const` declarations, and one `defineShader({...})` call (the default export or a named one).
 * Values are literals, objects, arrays, calls to std words, `p('name')`, `wgsl` templates and
 * references to earlier consts. Anything else (a function, a loop, a `let`, another import) is
 * refused with the line and column.
 */
// Maintainer notes. The TypeScript compiler is passed in (`ts`), so this module (and the
// `shaders/std/document` entry) never bundles it; the server and CLI already have it. The walker is
// an allowlist over node kinds. IR value encodings: JSON literals and containers as themselves;
// `{$word, args}` a call to an allowed word (namespaced as `simulate.grid`); `{$prop}` for p();
// `{$wgsl, line}` a template (its 1-based line in the file, for error mapping); `{$lib}` a library
// function; `{$value}` an allowed value (`child`, a named transform); `{$const}` a reference to a
// top-level const (kept as a reference so two uses are the same input).
import type * as TS from 'typescript'
import * as library from '../library/index'

/** The IR format version. Bump with a migration when an encoding changes; old versions stay readable. */
export const COMPONENT_IR_VERSION = 1
/** The parser version stamped on every IR, so stored IR can be re-derived when the parser changes. */
export const PARSER_VERSION = '1.0.0'

export type IRValue =
    | null | boolean | number | string
    | IRValue[]
    | {[key: string]: IRValue}

export interface ComponentIR {
    v: number
    parser: string
    /** The `name` the component registers under. */
    name: string
    /** How the definition is exported: `default` or the const's name. */
    exportName: string
    /** Top-level consts the definition refers to, in source order. */
    consts: {name: string; value: IRValue}[]
    /** The object handed to `defineShader`. */
    definition: {[key: string]: IRValue}
}

export interface ParseIssue {
    line: number
    column: number
    message: string
}

export type ParseResult =
    | {ok: true; ir: ComponentIR}
    | {ok: false; errors: ParseIssue[]}

/** Words a file may call, by the name it imports them under (namespaced members spelled out). */
export const CALLABLE_WORDS: ReadonlySet<string> = /* @__PURE__ */ new Set([
    'defineShader', 'wgsl', 'p', 'pointer', 'pointerSpeed',
    'crosses', 'isZero', 'isValue', 'allOf',
    'blur', 'bloom', 'pass', 'downsample', 'directionalBlur', 'previousFrame',
    'struct', 'buffer', 'kernel', 'rasterize', 'events', 'pointerTrail', 'text', 'samplePoints',
    'displaceBy', 'tintToward',
    'simulate.grid', 'simulate.particles', 'simulate.fluid',
    'op.wave', 'op.splat', 'op.gradient', 'op.diffuse', 'op.decay', 'op.reactDiffuse', 'op.automaton',
    'op.heat', 'op.growthFront', 'op.occupancy', 'op.kernel',
])

/** Values a file may name without calling them. */
export const VALUE_WORDS: ReadonlySet<string> = /* @__PURE__ */ new Set([
    'child', 'transformColor', 'transformPosition', 'transformBoolean', 'transformBool01', 'transformEdges',
    'transformAngle', 'transformColorSpace', 'colorStopsTransform', 'listPropTransform',
])

/** Namespaces a file may import and call members of. */
const NAMESPACES: ReadonlySet<string> = /* @__PURE__ */ new Set(['simulate', 'op'])

/** The modules a file may import from. */
export const ALLOWED_MODULES: ReadonlySet<string> = /* @__PURE__ */ new Set(['shaders/std', 'shaders/std/wgsl'])

/** Every library function a file may import from `shaders/std/wgsl`. */
export const LIBRARY_NAMES: ReadonlySet<string> = /* @__PURE__ */ new Set(
    Object.entries(library).filter(([, v]) => typeof v === 'object' && v !== null && (v as {kind?: unknown}).kind === 'wgslLib').map(([k]) => k),
)

class Refusal extends Error {
    constructor(message: string, readonly node: TS.Node) {
        super(message)
    }
}

/**
 * Parse a component file into its IR, or say exactly what was refused and where. Pass the
 * TypeScript compiler module (`import ts from 'typescript'`).
 */
export function parseComponentSource(source: string, ts: typeof TS, fileName = 'component.ts'): ParseResult {
    const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS)
    const errors: ParseIssue[] = []
    const at = (node: TS.Node): {line: number; column: number} => {
        const p = file.getLineAndCharacterOfPosition(node.getStart(file))
        return {line: p.line + 1, column: p.character + 1}
    }
    const refuse = (node: TS.Node, message: string) => {
        errors.push({...at(node), message})
    }
    const diagnostics = (file as unknown as {parseDiagnostics?: TS.Diagnostic[]}).parseDiagnostics ?? []
    for (const d of diagnostics) {
        const p = d.start !== undefined ? file.getLineAndCharacterOfPosition(d.start) : {line: 0, character: 0}
        errors.push({line: p.line + 1, column: p.character + 1, message: ts.flattenDiagnosticMessageText(d.messageText, '\n')})
    }
    if (errors.length) return {ok: false, errors}

    /** Imported name → what it is. */
    const imported = new Map<string, {kind: 'word' | 'value' | 'namespace' | 'lib'; name: string}>()
    const consts = new Map<string, IRValue>()
    const constOrder: string[] = []
    let definition: {[key: string]: IRValue} | null = null
    let exportName = ''
    let definitionNode: TS.Node | null = null

    const unwrap = (e: TS.Expression): TS.Expression => {
        let x = e
        while (ts.isAsExpression(x) || ts.isSatisfiesExpression(x) || ts.isParenthesizedExpression(x) || ts.isNonNullExpression(x)) x = x.expression
        return x
    }

    const calleeName = (e: TS.Expression): string | null => {
        const x = unwrap(e)
        if (ts.isIdentifier(x)) return x.text
        if (ts.isPropertyAccessExpression(x) && ts.isIdentifier(x.expression) && ts.isIdentifier(x.name)) return `${x.expression.text}.${x.name.text}`
        return null
    }

    const foldNumber = (e: TS.Expression): number | null => {
        const x = unwrap(e)
        if (ts.isNumericLiteral(x)) return Number(x.text)
        if (ts.isPrefixUnaryExpression(x) && (x.operator === ts.SyntaxKind.MinusToken || x.operator === ts.SyntaxKind.PlusToken)) {
            const v = foldNumber(x.operand)
            return v === null ? null : x.operator === ts.SyntaxKind.MinusToken ? -v : v
        }
        if (ts.isBinaryExpression(x)) {
            const a = foldNumber(x.left)
            const b = foldNumber(x.right)
            if (a === null || b === null) return null
            switch (x.operatorToken.kind) {
                case ts.SyntaxKind.PlusToken: return a + b
                case ts.SyntaxKind.MinusToken: return a - b
                case ts.SyntaxKind.AsteriskToken: return a * b
                case ts.SyntaxKind.SlashToken: return a / b
                case ts.SyntaxKind.PercentToken: return a % b
                case ts.SyntaxKind.AsteriskAsteriskToken: return a ** b
            }
            return null
        }
        if (ts.isIdentifier(x) && consts.has(x.text) && typeof consts.get(x.text) === 'number') return consts.get(x.text) as number
        if (ts.isPropertyAccessExpression(x) && ts.isIdentifier(x.expression) && x.expression.text === 'Math' && ts.isIdentifier(x.name)) {
            const v = (Math as unknown as Record<string, unknown>)[x.name.text]
            return typeof v === 'number' ? v : null
        }
        return null
    }

    const templateText = (t: TS.TemplateLiteral): string => {
        if (ts.isNoSubstitutionTemplateLiteral(t)) return t.text
        let out = t.head.text
        for (const span of t.templateSpans) {
            const raw = value(span.expression)
            const v = isConstRef(raw) ? consts.get((raw as {$const: string}).$const) : raw
            if (typeof v === 'number') out += String(v)
            else if (typeof v === 'string') out += v
            else throw new Refusal('a ${…} inside WGSL may only be a number or a string (a literal or a const)', span.expression)
            out += span.literal.text
        }
        return out
    }

    const value = (expr: TS.Expression): IRValue => {
        const e = unwrap(expr)
        if (ts.isNumericLiteral(e) || ts.isPrefixUnaryExpression(e) || ts.isBinaryExpression(e)) {
            const n = foldNumber(e)
            if (n === null || !Number.isFinite(n)) throw new Refusal('only arithmetic on numbers is allowed here', e)
            return n
        }
        if (ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) return e.text
        if (ts.isTemplateExpression(e)) return templateText(e)
        if (e.kind === ts.SyntaxKind.TrueKeyword) return true
        if (e.kind === ts.SyntaxKind.FalseKeyword) return false
        if (e.kind === ts.SyntaxKind.NullKeyword) return null
        if (ts.isIdentifier(e) && e.text === 'undefined') return null
        if (ts.isArrayLiteralExpression(e)) {
            const out: IRValue[] = []
            for (const el of e.elements) {
                if (ts.isSpreadElement(el)) {
                    const v = value(el.expression)
                    if (!Array.isArray(v)) throw new Refusal('only an array can be spread into an array', el)
                    out.push(...v)
                } else {
                    out.push(value(el))
                }
            }
            return out
        }
        if (ts.isObjectLiteralExpression(e)) {
            const out: {[key: string]: IRValue} = {}
            for (const prop of e.properties) {
                if (ts.isPropertyAssignment(prop)) {
                    const key = ts.isIdentifier(prop.name) || ts.isStringLiteral(prop.name) || ts.isNumericLiteral(prop.name) ? prop.name.text : null
                    if (key === null) throw new Refusal('object keys must be plain names or strings', prop.name)
                    out[key] = value(prop.initializer)
                } else if (ts.isShorthandPropertyAssignment(prop)) {
                    out[prop.name.text] = value(prop.name)
                } else if (ts.isSpreadAssignment(prop)) {
                    const v = value(prop.expression)
                    const resolved = isConstRef(v) ? consts.get((v as {$const: string}).$const) : v
                    if (!resolved || typeof resolved !== 'object' || Array.isArray(resolved) || isEncoded(resolved)) {
                        throw new Refusal('only a plain object can be spread into an object', prop)
                    }
                    Object.assign(out, resolved)
                } else {
                    throw new Refusal('methods and getters are not allowed: write the value instead', prop)
                }
            }
            return out
        }
        if (ts.isTaggedTemplateExpression(e)) {
            if (calleeName(e.tag) !== 'wgsl' || imported.get('wgsl')?.kind !== 'word') {
                throw new Refusal('the only tagged template allowed is wgsl`…` (imported from shaders/std)', e.tag)
            }
            return {$wgsl: templateText(e.template), line: at(e.template).line}
        }
        if (ts.isIdentifier(e)) {
            if (consts.has(e.text)) {
                return {$const: e.text}
            }
            const imp = imported.get(e.text)
            if (imp?.kind === 'value') return {$value: imp.name}
            if (imp?.kind === 'lib') return {$lib: imp.name}
            if (imp?.kind === 'word') throw new Refusal(`${e.text} is a word: call it, e.g. ${e.text}(…)`, e)
            throw new Refusal(`'${e.text}' is not defined here (only imports from shaders/std and consts declared above are)`, e)
        }
        if (ts.isCallExpression(e)) {
            const name = calleeName(e.expression)
            if (!name) throw new Refusal('only calls to std words are allowed', e.expression)
            const [head, member] = name.split('.')
            const imp = imported.get(head)
            const word = member ? (imp?.kind === 'namespace' ? `${imp.name}.${member}` : null) : imp?.kind === 'word' ? imp.name : null
            if (!word || !CALLABLE_WORDS.has(word)) {
                throw new Refusal(imp ? `${name} is not a word a component can call` : `'${head}' is not imported from shaders/std`, e.expression)
            }
            if (word === 'defineShader') throw new Refusal('defineShader may only be the exported definition', e)
            if (e.typeArguments?.length && word !== 'defineShader') {
                // Type arguments (`struct<…>`) are TypeScript only; they do not change the data.
            }
            if (word === 'wgsl') {
                const arg = e.arguments[0]
                if (!arg) throw new Refusal('wgsl(…) needs a body', e)
                const v = value(arg)
                if (typeof v === 'string') return {$wgsl: v, line: at(arg).line}
                return {$word: 'wgsl', args: [v]}
            }
            return {$word: word, args: e.arguments.map((a) => value(a))}
        }
        if (ts.isPropertyAccessExpression(e)) {
            throw new Refusal('property access is not allowed here; name the value directly', e)
        }
        if (ts.isArrowFunction(e) || ts.isFunctionExpression(e)) {
            throw new Refusal('functions cannot be part of a component: write the logic in WGSL (a body, helpers or a kernel)', e)
        }
        throw new Refusal(`this kind of expression (${ts.SyntaxKind[e.kind]}) is not allowed in a component file`, e)
    }

    for (const st of file.statements) {
        try {
            if (ts.isImportDeclaration(st)) {
                const from = (st.moduleSpecifier as TS.StringLiteral).text
                if (st.importClause?.isTypeOnly) continue
                if (!ALLOWED_MODULES.has(from)) throw new Refusal(`imports may only come from shaders/std or shaders/std/wgsl (not '${from}')`, st.moduleSpecifier)
                const clause = st.importClause
                if (!clause) continue
                if (clause.name) throw new Refusal('use named imports: import {defineShader, wgsl} from \'shaders/std\'', clause.name)
                const bindings = clause.namedBindings
                if (!bindings) continue
                if (ts.isNamespaceImport(bindings)) throw new Refusal('use named imports, not import * as', bindings)
                for (const el of bindings.elements) {
                    if (el.isTypeOnly) continue
                    const original = (el.propertyName ?? el.name).text
                    const local = el.name.text
                    if (from === 'shaders/std/wgsl') {
                        if (!LIBRARY_NAMES.has(original)) throw new Refusal(`'${original}' is not in the WGSL library`, el)
                        imported.set(local, {kind: 'lib', name: original})
                    } else if (NAMESPACES.has(original)) {
                        imported.set(local, {kind: 'namespace', name: original})
                    } else if (CALLABLE_WORDS.has(original)) {
                        imported.set(local, {kind: 'word', name: original})
                    } else if (VALUE_WORDS.has(original)) {
                        imported.set(local, {kind: 'value', name: original})
                    } else {
                        throw new Refusal(`'${original}' cannot be used in a component file (not a std word a component may call)`, el)
                    }
                }
                continue
            }
            if (ts.isInterfaceDeclaration(st) || ts.isTypeAliasDeclaration(st)) continue
            if (ts.isEmptyStatement(st)) continue
            if (ts.isVariableStatement(st)) {
                if (!(st.declarationList.flags & ts.NodeFlags.Const)) throw new Refusal('use const (a component file has no let or var)', st)
                const exported = st.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword) ?? false
                for (const decl of st.declarationList.declarations) {
                    if (!ts.isIdentifier(decl.name)) throw new Refusal('destructuring is not allowed; declare one name per const', decl.name)
                    if (!decl.initializer) throw new Refusal('a const needs a value', decl)
                    const init = unwrap(decl.initializer)
                    if (ts.isCallExpression(init) && calleeName(init.expression) === 'defineShader') {
                        claimDefinition(init, exported ? decl.name.text : null, decl)
                        continue
                    }
                    if (consts.has(decl.name.text) || imported.has(decl.name.text)) throw new Refusal(`'${decl.name.text}' is already declared`, decl.name)
                    consts.set(decl.name.text, value(decl.initializer))
                    constOrder.push(decl.name.text)
                }
                continue
            }
            if (ts.isExportAssignment(st)) {
                const init = unwrap(st.expression)
                if (ts.isCallExpression(init) && calleeName(init.expression) === 'defineShader') {
                    claimDefinition(init, 'default', st)
                    continue
                }
                if (ts.isIdentifier(init) && exportName && exportName !== 'default') {
                    // `export default Halo` after `export const Halo = defineShader(…)`.
                    continue
                }
                throw new Refusal('export default must be the defineShader({...}) call', st)
            }
            if (ts.isExportDeclaration(st)) {
                // `export {Halo}` re-exporting the definition const: allowed, changes nothing.
                continue
            }
            throw new Refusal(`a component file holds imports, consts and one defineShader call (found ${ts.SyntaxKind[st.kind]})`, st)
        } catch (e) {
            if (e instanceof Refusal) refuse(e.node, e.message)
            else throw e
        }
    }

    function claimDefinition(call: TS.CallExpression, name: string | null, node: TS.Node) {
        if (imported.get('defineShader')?.kind !== 'word') throw new Refusal('import defineShader from shaders/std', call.expression)
        if (definitionNode) throw new Refusal('a file defines exactly one component', node)
        definitionNode = node
        const arg = call.arguments[0]
        if (!arg || call.arguments.length !== 1) throw new Refusal('defineShader takes one object', call)
        const v = value(arg)
        if (!v || typeof v !== 'object' || Array.isArray(v) || isEncoded(v)) throw new Refusal('defineShader takes an object literal', arg)
        definition = v as {[key: string]: IRValue}
        exportName = name ?? 'default'
    }

    if (!errors.length && !definition) {
        errors.push({line: 1, column: 1, message: 'no component: export default defineShader({...})'})
    }
    if (errors.length) return {ok: false, errors}
    const def = definition as unknown as {[key: string]: IRValue}
    if (typeof def.name !== 'string' || !def.name) {
        return {ok: false, errors: [{...at(definitionNode!), message: 'the definition needs a name: \'MyComponent\''}]}
    }
    // Keep consts that are referenced (transitively), in order.
    const keep = new Set<string>()
    const visit = (v: IRValue) => {
        if (Array.isArray(v)) v.forEach(visit)
        else if (v && typeof v === 'object') {
            if (isConstRef(v)) {
                const n = (v as {$const: string}).$const
                if (!keep.has(n)) {
                    keep.add(n)
                    visit(consts.get(n)!)
                }
                return
            }
            Object.values(v).forEach(visit)
        }
    }
    visit(def)
    return {
        ok: true,
        ir: {
            v: COMPONENT_IR_VERSION,
            parser: PARSER_VERSION,
            name: def.name,
            exportName,
            consts: constOrder.filter((n) => keep.has(n)).map((n) => ({name: n, value: consts.get(n)!})),
            definition: def,
        },
    }
}

function isConstRef(v: IRValue): boolean {
    return !!v && typeof v === 'object' && !Array.isArray(v) && typeof (v as {$const?: unknown}).$const === 'string'
}

function isEncoded(v: IRValue): boolean {
    return !!v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).some((k) => k.startsWith('$'))
}

/** How `componentSourceForExport` rewrites a file. */
export interface ComponentExportOptions {
    /** The named export the generated shader imports the component as. */
    identifier: string
    /** The `name` the component registers under in the exported app. */
    name: string
    /** Comment lines put at the top of the file. */
    header?: string[]
}

/**
 * A parsed component file ready to drop into an app: its `name` replaced, its definition exported
 * under `identifier` (alongside any export it already had), header comments on top. Everything
 * else is the file exactly as written. Returns null when the file holds no definition to rewrite
 * (run `parseComponentSource` first).
 */
export function componentSourceForExport(source: string, ts: typeof TS, options: ComponentExportOptions): string | null {
    const file = ts.createSourceFile('component.ts', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS)
    const strip = (e: TS.Expression): TS.Expression => {
        let x = e
        while (ts.isAsExpression(x) || ts.isSatisfiesExpression(x) || ts.isParenthesizedExpression(x) || ts.isNonNullExpression(x)) x = x.expression
        return x
    }
    const isDefine = (e: TS.Expression | undefined): e is TS.CallExpression => {
        const x = e && strip(e)
        return !!x && ts.isCallExpression(x) && ts.isIdentifier(x.expression) && x.expression.text === 'defineShader'
    }

    const edits: {start: number; end: number; text: string}[] = []
    let call: TS.CallExpression | null = null
    let constName: string | null = null
    const exported = new Set<string>()
    for (const st of file.statements) {
        if (ts.isVariableStatement(st)) {
            const isExported = st.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword) ?? false
            for (const decl of st.declarationList.declarations) {
                if (!ts.isIdentifier(decl.name)) continue
                if (isExported) exported.add(decl.name.text)
                if (isDefine(decl.initializer)) {
                    call = strip(decl.initializer) as TS.CallExpression
                    constName = decl.name.text
                }
            }
        } else if (ts.isExportAssignment(st) && isDefine(st.expression)) {
            call = strip(st.expression) as TS.CallExpression
            // `export default defineShader(…)` becomes `export const <identifier> = defineShader(…)`
            // plus `export default <identifier>`, so both import styles keep working.
            edits.push({start: st.getStart(file), end: st.expression.getStart(file), text: `export const ${options.identifier} = `})
            edits.push({start: st.getEnd(), end: st.getEnd(), text: `\nexport default ${options.identifier}`})
        } else if (ts.isExportDeclaration(st) && st.exportClause && ts.isNamedExports(st.exportClause)) {
            for (const el of st.exportClause.elements) exported.add(el.name.text)
        }
    }
    if (!call) return null
    const arg = call.arguments[0] && strip(call.arguments[0])
    if (!arg || !ts.isObjectLiteralExpression(arg)) return null
    const nameProp = arg.properties.find((pr): pr is TS.PropertyAssignment =>
        ts.isPropertyAssignment(pr) && (ts.isIdentifier(pr.name) || ts.isStringLiteral(pr.name)) && pr.name.text === 'name')
    if (!nameProp) return null
    edits.push({start: nameProp.initializer.getStart(file), end: nameProp.initializer.getEnd(), text: JSON.stringify(options.name)})
    if (constName && !exported.has(options.identifier)) {
        edits.push({
            start: source.length, end: source.length,
            text: `${source.endsWith('\n') ? '' : '\n'}export {${constName === options.identifier ? constName : `${constName} as ${options.identifier}`}}\n`,
        })
    }

    let out = source
    for (const e of edits.sort((a, b) => b.start - a.start)) out = out.slice(0, e.start) + e.text + out.slice(e.end)
    const header = (options.header ?? []).map((line) => `// ${line}\n`).join('')
    return header + out
}
