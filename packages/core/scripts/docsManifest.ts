/**
 * The docs manifest — ONE generated description of the std vocabulary that every reference
 * surface renders from (shaders.com primitive docs, `llms.txt`, per-category markdown, the
 * future agent skill). Nothing here is hand-maintained per word; the sources, in priority:
 *
 *   1. Signatures and types, read from the TypeScript source with the compiler API.
 *   2. Doc comments on each exported word: the summary (first paragraph), the description
 *      (the rest), and a small tag set — `@category` (override the module's category),
 *      `@example` (a code block), `@see` (related word names), `@tip` (one editorial line),
 *      plus `@param`/`@returns`/`@deprecated`/`@internal` (excluded).
 *   3. Curation, one hand-written markdown file per category under `packages/core/docs/std/`:
 *      the intro, the word ORDER, a "reach for it when" table, and a worked example. Parsed
 *      by heading (see `parseCuration`).
 *
 * Plus two derived facts: "used by" (which library shaders import each word) and the import
 * path a word is reached through on `shaders/std` (a namespace member or a top-level export).
 *
 * `buildDocsManifest` is pure and GPU-free: the core build writes it to `dist/docs-manifest.json`,
 * the coverage test asserts on it, and dotcom's generator calls it against the submodule.
 */
import ts from 'typescript'
import fs from 'fs'
import path from 'path'
import {fileURLToPath} from 'url'

// ── Manifest types (the contract every renderer consumes) ───────────────────────────────

export interface DocsManifest {
    /** Schema version of this file — bump when a renderer would need to change. */
    schemaVersion: 1
    /** The engine version the manifest describes. */
    engineVersion: string
    generatedAt: string
    /** Top-level groups in display order; each lists its category ids in display order. */
    groups: DocsGroup[]
    /** Categories keyed by id. */
    categories: Record<string, DocsCategory>
    /** Words keyed by id (`<category>.<name>` or `<category>.<parent>.<member>`). */
    words: Record<string, DocsWord>
    /** Exported types the words' signatures mention, keyed by `<category>.<Name>`. */
    types: Record<string, DocsType>
    /** Coverage summary, so the test and the docs can show it. */
    coverage: {words: number; documented: number; undocumented: string[]; curatedCategories: string[]}
}

export interface DocsGroup {
    id: string
    title: string
    categories: string[]
}

export interface DocsCategory {
    id: string
    title: string
    group: string
    /** Source module, relative to `packages/core/src`. */
    module: string
    /** How the category is reached from `shaders/std`: a namespace (`paint`) or top-level names. */
    access: {kind: 'namespace'; name: string} | {kind: 'top-level'}
    /** Curated (from the category's markdown file); absent when nobody has written it yet. */
    intro?: string
    /** `use` is markdown: prose with the word names in backticks (each validated). */
    reach?: {when: string; use: string}[]
    example?: string
    curated: boolean
    /** Word ids in display order: curated order first, the rest alphabetical. */
    words: string[]
    types: string[]
}

export interface DocsParam {
    name: string
    type: string
    optional: boolean
    defaultValue?: string
    doc?: string
}

export interface DocsWord {
    id: string
    name: string
    /** `parent.member` for words grouped under an object (`dist.radial`); equals `name` otherwise. */
    qualifiedName: string
    category: string
    module: string
    kind: 'function' | 'constant' | 'class' | 'object'
    /** The signature as an author reads it (`rampOver(field: Field, palette: Palette): Paint`). */
    signature: string
    /** For objects: the member words. */
    members?: string[]
    summary: string
    description: string
    params: DocsParam[]
    returns?: string
    examples: string[]
    tips: string[]
    /** Related word ids (from `@see`, resolved by name where possible). */
    see: string[]
    deprecated?: string
    /** How to import and call it from `shaders/std`. */
    import: {statement: string; usage: string}
    /** Library shaders that use this word (by shader name). */
    usedBy: string[]
    documented: boolean
    source: {file: string; line: number}
}

export interface DocsType {
    id: string
    name: string
    category: string
    module: string
    kind: 'type' | 'interface' | 'class'
    text: string
    summary: string
    description: string
    source: {file: string; line: number}
}

// ── The category map: module → category (the default; `@category` overrides per word) ──

interface CategorySpec {
    id: string
    title: string
    group: string
    module: string
}

/** Groups in display order. */
const GROUPS: DocsGroup[] = [
    {id: 'define', title: 'Defining a shader', categories: ['define', 'values', 'slots', 'invoke', 'lists']},
    {id: 'paint', title: 'Painting', categories: ['paint', 'gradients', 'patterns', 'figures', 'compose', 'media']},
    {id: 'light', title: 'Noise, light & materials', categories: ['noise', 'light', 'materials', 'volume', 'radiance', 'voxels']},
    {id: 'space', title: 'Shapes, frames & warps', categories: ['shape', 'frames', 'warps', 'mask', 'math', 'motion', 'signal']},
    {id: 'effects', title: 'Filters & effects', categories: ['filter', 'effects.color', 'effects.blurs', 'effects.stylize', 'effects.lens', 'effects.reveal', 'effects.instances', 'effects.fracture', 'effects.edgeGlow', 'effects.overlay', 'effects.pointerFields']},
    {id: 'simulation', title: 'Simulation', categories: ['simulate', 'sim.grids', 'sim.fluids', 'sim.feedback', 'sim.agents', 'sim.agentForces', 'sim.agentFrame', 'sim.agentRender', 'sim.shapeFields']},
]

const CATEGORIES: CategorySpec[] = [
    {id: 'define', title: 'defineShader & wgsl', group: 'define', module: 'std/lower.ts'},
    {id: 'values', title: 'Props & values', group: 'define', module: 'std/values.ts'},
    {id: 'slots', title: 'Identity & recompile rules', group: 'define', module: 'std/slots.ts'},
    {id: 'invoke', title: 'Context & invocation', group: 'define', module: 'std/invoke.ts'},
    {id: 'lists', title: 'List props', group: 'define', module: 'std/lists.ts'},
    {id: 'paint', title: 'Fields, palettes & ramps', group: 'paint', module: 'std/paint/fields.ts'},
    {id: 'gradients', title: 'Gradients', group: 'paint', module: 'std/paint/gradients.ts'},
    {id: 'patterns', title: 'Patterns & tiling', group: 'paint', module: 'std/paint/patterns.ts'},
    {id: 'figures', title: 'Figures & strokes', group: 'paint', module: 'std/paint/figures.ts'},
    {id: 'compose', title: 'Layering & output', group: 'paint', module: 'std/paint/compose.ts'},
    {id: 'media', title: 'Media', group: 'paint', module: 'std/paint/media.ts'},
    {id: 'noise', title: 'Noise', group: 'light', module: 'std/paint/noise.ts'},
    {id: 'light', title: 'Light', group: 'light', module: 'std/paint/light.ts'},
    {id: 'materials', title: 'Materials & surfaces', group: 'light', module: 'std/paint/materials.ts'},
    {id: 'volume', title: 'Volumes', group: 'light', module: 'std/paint/volume.ts'},
    {id: 'radiance', title: 'Radiance', group: 'light', module: 'std/paint/radiance.ts'},
    {id: 'voxels', title: 'Voxels', group: 'light', module: 'std/paint/voxels.ts'},
    {id: 'shape', title: 'Shapes (SDF)', group: 'space', module: 'std/shape.ts'},
    {id: 'frames', title: 'Frames & coordinates', group: 'space', module: 'std/frames.ts'},
    {id: 'warps', title: 'Warps', group: 'space', module: 'std/warps.ts'},
    {id: 'mask', title: 'Masks', group: 'space', module: 'std/mask.ts'},
    {id: 'math', title: 'Math', group: 'space', module: 'std/math.ts'},
    {id: 'motion', title: 'Motion & timing', group: 'space', module: 'std/motion.ts'},
    {id: 'signal', title: 'Signals', group: 'space', module: 'std/signal.ts'},
    {id: 'filter', title: 'Filter effects', group: 'effects', module: 'std/filter.ts'},
    {id: 'effects.color', title: 'Color effects', group: 'effects', module: 'std/effects/color.ts'},
    {id: 'effects.blurs', title: 'Blurs & glows', group: 'effects', module: 'std/effects/blurs.ts'},
    {id: 'effects.stylize', title: 'Stylize', group: 'effects', module: 'std/effects/stylize.ts'},
    {id: 'effects.lens', title: 'Lens', group: 'effects', module: 'std/effects/lens.ts'},
    {id: 'effects.reveal', title: 'Reveals & wipes', group: 'effects', module: 'std/effects/reveal.ts'},
    {id: 'effects.instances', title: 'Instances', group: 'effects', module: 'std/effects/instances.ts'},
    {id: 'effects.fracture', title: 'Fracture', group: 'effects', module: 'std/effects/fracture.ts'},
    {id: 'effects.edgeGlow', title: 'Edge glow', group: 'effects', module: 'std/effects/edgeGlow.ts'},
    {id: 'effects.overlay', title: 'Overlays', group: 'effects', module: 'std/effects/overlay.ts'},
    {id: 'effects.pointerFields', title: 'Pointer fields', group: 'effects', module: 'std/effects/pointerFields.ts'},
    {id: 'simulate', title: 'Grid simulations', group: 'simulation', module: 'std/sim.ts'},
    {id: 'sim.grids', title: 'Grid programs', group: 'simulation', module: 'std/sim/grids.ts'},
    {id: 'sim.fluids', title: 'Fluids', group: 'simulation', module: 'std/sim/fluids.ts'},
    {id: 'sim.feedback', title: 'Feedback', group: 'simulation', module: 'std/sim/feedback.ts'},
    {id: 'sim.agents', title: 'Agents', group: 'simulation', module: 'std/sim/agents.ts'},
    {id: 'sim.agentForces', title: 'Agent forces', group: 'simulation', module: 'std/sim/agentForces.ts'},
    {id: 'sim.agentFrame', title: 'Agent frame', group: 'simulation', module: 'std/sim/agentFrame.ts'},
    {id: 'sim.agentRender', title: 'Agent rendering', group: 'simulation', module: 'std/sim/agentRender.ts'},
    {id: 'sim.shapeFields', title: 'Shape fields', group: 'simulation', module: 'std/sim/shapeFields.ts'},
]

/** Modules that contribute words to a category beyond the category's own module. */
const EXTRA_MODULES: Record<string, string[]> = {
    define: ['std/wgsl.ts', 'std/types.ts', 'customShaders.ts'],
}

// ── Public entry ────────────────────────────────────────────────────────────────────────

export interface BuildOptions {
    /** `packages/core` directory. */
    coreDir: string
    /** Engine version to stamp (defaults to the meta package's version when found). */
    engineVersion?: string
    /** Skip the curation files (a build that must not fail on someone else's half-written page). */
    skipCuration?: boolean
}

export function buildDocsManifest(options: BuildOptions): DocsManifest {
    const coreDir = path.resolve(options.coreDir)
    const srcDir = path.join(coreDir, 'src')
    const program = createProgram(coreDir)
    const checker = program.getTypeChecker()

    const access = readIndexAccess(program, srcDir)
    const words: Record<string, DocsWord> = {}
    const types: Record<string, DocsType> = {}
    const categories: Record<string, DocsCategory> = {}

    for (const spec of CATEGORIES) {
        const modules = [spec.module, ...(EXTRA_MODULES[spec.id] ?? [])]
        const category: DocsCategory = {
            id: spec.id,
            title: spec.title,
            group: spec.group,
            module: spec.module,
            access: accessForModule(spec.module, access),
            curated: false,
            words: [],
            types: [],
        }
        for (const mod of modules) {
            const file = program.getSourceFile(path.join(srcDir, mod))
            if (!file) throw new Error(`docs manifest: module ${mod} is not in the program`)
            const exportedNames = exportedNamesOf(mod, access)
            collectModule(file, checker, spec, mod, exportedNames, access, words, types, category)
        }
        // Until a category is curated, its module's header comment is the intro (the maintainer's
        // own one-paragraph description of the vocabulary). Curation replaces it.
        const primary = program.getSourceFile(path.join(srcDir, spec.module))
        const header = primary ? moduleHeader(primary) : undefined
        if (header) category.intro = header
        categories[spec.id] = category
    }

    // `@category` overrides: a word collected under its module's category but declared for
    // another one is listed there instead (collectModule only pushes words whose target is the
    // module's own category). An unknown target is left for the coverage test to report.
    for (const word of Object.values(words)) {
        const target = categories[word.category]
        if (target && !target.words.includes(word.id)) target.words.push(word.id)
    }

    if (!options.skipCuration) applyCuration(coreDir, categories, words)
    applyUsedBy(srcDir, words, access)
    resolveSee(words)

    for (const category of Object.values(categories)) {
        category.words = orderWords(category, words)
        category.types.sort()
    }

    const all = Object.values(words)
    const undocumented = all.filter((w) => !w.documented).map((w) => w.id).sort()
    return {
        schemaVersion: 1,
        engineVersion: options.engineVersion ?? readEngineVersion(coreDir),
        generatedAt: new Date().toISOString(),
        groups: GROUPS,
        categories,
        words,
        types,
        coverage: {
            words: all.length,
            documented: all.length - undocumented.length,
            undocumented,
            curatedCategories: Object.values(categories).filter((c) => c.curated).map((c) => c.id).sort(),
        },
    }
}

// ── Program ─────────────────────────────────────────────────────────────────────────────

function createProgram(coreDir: string): ts.Program {
    const configPath = path.join(coreDir, 'tsconfig.json')
    const parsed = ts.getParsedCommandLineOfConfigFile(configPath, {}, {
        ...ts.sys,
        onUnRecoverableConfigFileDiagnostic: (d) => {
            throw new Error(ts.flattenDiagnosticMessageText(d.messageText, '\n'))
        },
    })
    if (!parsed) throw new Error(`docs manifest: cannot read ${configPath}`)
    const roots = [
        ...listFiles(path.join(coreDir, 'src/std'), /\.ts$/),
        path.join(coreDir, 'src/customShaders.ts'),
    ]
    return ts.createProgram(roots, {...parsed.options, noEmit: true, skipLibCheck: true})
}

function listFiles(dir: string, pattern: RegExp): string[] {
    const out: string[] = []
    for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
        const full = path.join(dir, entry.name)
        if (entry.isDirectory()) out.push(...listFiles(full, pattern))
        else if (pattern.test(entry.name)) out.push(full)
    }
    return out.sort()
}

// ── How `shaders/std` exposes each module ───────────────────────────────────────────────

interface IndexAccess {
    /** module (relative to src, e.g. `std/paint/fields.ts`) → namespace name it is exported under. */
    namespaces: Map<string, string>
    /** module → the names re-exported at the top level of `shaders/std`. */
    topLevel: Map<string, Set<string>>
    /** Nested namespace indexes (`std/effects/index.ts`) → their own namespace map. */
    nested: Map<string, Map<string, string>>
}

function readIndexAccess(program: ts.Program, srcDir: string): IndexAccess {
    const access: IndexAccess = {namespaces: new Map(), topLevel: new Map(), nested: new Map()}
    const indexFile = program.getSourceFile(path.join(srcDir, 'std/index.ts'))
    if (!indexFile) throw new Error('docs manifest: std/index.ts not in program')
    const resolveModule = (from: ts.SourceFile, specifier: string): string => {
        const abs = path.resolve(path.dirname(from.fileName), specifier)
        const candidates = [`${abs}.ts`, path.join(abs, 'index.ts'), abs]
        const found = candidates.find((c) => fs.existsSync(c))
        return path.relative(srcDir, found ?? `${abs}.ts`)
    }
    const visit = (file: ts.SourceFile, prefix: string | null) => {
        for (const st of file.statements) {
            if (!ts.isExportDeclaration(st) || !st.moduleSpecifier || !ts.isStringLiteral(st.moduleSpecifier)) continue
            const mod = resolveModule(file, st.moduleSpecifier.text)
            if (st.exportClause && ts.isNamespaceExport(st.exportClause)) {
                const ns = prefix ? `${prefix}.${st.exportClause.name.text}` : st.exportClause.name.text
                if (mod.endsWith('index.ts')) {
                    const nestedFile = program.getSourceFile(path.join(srcDir, mod))
                    if (nestedFile) visit(nestedFile, ns)
                } else {
                    access.namespaces.set(mod, ns)
                }
            } else if (st.exportClause && ts.isNamedExports(st.exportClause)) {
                if (st.isTypeOnly) continue
                const set = access.topLevel.get(mod) ?? new Set<string>()
                for (const el of st.exportClause.elements) {
                    if (el.isTypeOnly) continue
                    set.add((el.propertyName ?? el.name).text)
                }
                access.topLevel.set(mod, set)
            }
        }
    }
    visit(indexFile, null)
    return access
}

function accessForModule(module: string, access: IndexAccess): DocsCategory['access'] {
    const ns = access.namespaces.get(module)
    return ns ? {kind: 'namespace', name: ns} : {kind: 'top-level'}
}

/** The names of `module` that are reachable from `shaders/std`; `null` = every export (a namespace). */
function exportedNamesOf(module: string, access: IndexAccess): Set<string> | null {
    if (access.namespaces.has(module)) return null
    return access.topLevel.get(module) ?? new Set()
}

// ── Collecting words from a module ──────────────────────────────────────────────────────

function collectModule(
    file: ts.SourceFile,
    checker: ts.TypeChecker,
    spec: CategorySpec,
    module: string,
    exportedNames: Set<string> | null,
    access: IndexAccess,
    words: Record<string, DocsWord>,
    types: Record<string, DocsType>,
    category: DocsCategory,
): void {
    const isExported = (name: string) => exportedNames === null || exportedNames.has(name)
    for (const st of file.statements) {
        if (!hasExportModifier(st)) continue
        const doc = readJsDoc(st)
        if (doc.tags.internal) continue

        if (ts.isFunctionDeclaration(st) && st.name) {
            if (!isExported(st.name.text)) continue
            const word = makeFunctionWord(st, st.name.text, checker, spec, module, doc, access, file)
            const existing = words[word.id]
            if (existing) {
                // An overload set: one word, every overload signature listed, docs from whichever
                // overload carries them. The implementation signature (the one with a body) is
                // the broadest and is not what an author calls — list only the overloads.
                if (!st.body) existing.signature = existing.signature ? `${existing.signature}\n${word.signature}` : word.signature
                if (!existing.documented && word.documented) Object.assign(existing, {summary: word.summary, description: word.description, examples: word.examples, tips: word.tips, see: word.see, documented: true})
                if (!existing.params.length) existing.params = word.params
                continue
            }
            if (st.body && hasOverloads(file, st.name.text)) {
                // First sight is the implementation with earlier overloads absent (they come first
                // in source, so this only happens when no overload preceded) — keep as is.
            }
            addWord(word)
        } else if (ts.isVariableStatement(st)) {
            for (const decl of st.declarationList.declarations) {
                if (!ts.isIdentifier(decl.name) || !isExported(decl.name.text)) continue
                collectVariable(decl, checker, spec, module, doc, access, file, addWord)
            }
        } else if (ts.isClassDeclaration(st) && st.name) {
            if (!isExported(st.name.text)) continue
            addWord({
                ...baseWord(st.name.text, st.name.text, spec, module, doc, access, file, st),
                kind: 'class',
                signature: `class ${st.name.text}`,
                params: [],
            })
        } else if ((ts.isInterfaceDeclaration(st) || ts.isTypeAliasDeclaration(st)) && st.name) {
            // Types are always recorded when their module is public — signatures mention them.
            if (exportedNames !== null && !typeIsPublic(st.name.text, module, access)) continue
            const id = `${spec.id}.${st.name.text}`
            types[id] = {
                id,
                name: st.name.text,
                category: spec.id,
                module,
                kind: ts.isInterfaceDeclaration(st) ? 'interface' : 'type',
                text: st.getText(file).replace(/^export\s+/, ''),
                summary: doc.summary,
                description: doc.description,
                source: {file: module, line: file.getLineAndCharacterOfPosition(st.getStart(file)).line + 1},
            }
            category.types.push(id)
        }
    }

    function addWord(word: DocsWord): void {
        const target = word.category
        words[word.id] = word
        if (target === spec.id) category.words.push(word.id)
    }
}

/**
 * The module's header comment — the first `/** … *​/` block before any statement — as prose:
 * the `std/paint/light — ` style title prefix dropped, backtick-quoted paths kept.
 */
function moduleHeader(file: ts.SourceFile): string | undefined {
    const text = file.getFullText()
    const first = file.statements[0]
    const limit = first ? first.getStart(file) : text.length
    const m = text.slice(0, limit).match(/\/\*\*([\s\S]*?)\*\//)
    if (!m) return undefined
    const body = m[1]
        .split('\n')
        .map((line) => line.replace(/^\s*\*\s?/, ''))
        .join('\n')
        .trim()
    const stripped = body.replace(/^[`\w/@.-]+\s+[—–-]\s+/, '')
    const paragraphs = stripped.split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, ' ').trim()).filter(Boolean)
    if (!paragraphs.length) return undefined
    const intro = paragraphs[0].charAt(0).toUpperCase() + paragraphs[0].slice(1)
    return [intro, ...paragraphs.slice(1)].join('\n\n')
}

function hasOverloads(file: ts.SourceFile, name: string): boolean {
    let n = 0
    for (const st of file.statements) if (ts.isFunctionDeclaration(st) && st.name?.text === name) n++
    return n > 1
}

function typeIsPublic(name: string, module: string, access: IndexAccess): boolean {
    // Top-level modules export their types through `export type {…}` lines the access map skips
    // (type-only). Treat every exported type of a public module as documentable.
    return access.topLevel.has(module) || access.namespaces.has(module) || name.length > 0
}

function collectVariable(
    decl: ts.VariableDeclaration,
    checker: ts.TypeChecker,
    spec: CategorySpec,
    module: string,
    doc: JsDoc,
    access: IndexAccess,
    file: ts.SourceFile,
    addWord: (w: DocsWord) => void,
): void {
    const name = (decl.name as ts.Identifier).text
    // `export const op = {…} as const` / `{…} satisfies X` — look through the assertion so the
    // object's members become words like any other grouped object.
    let init = decl.initializer
    while (init && (ts.isAsExpression(init) || ts.isSatisfiesExpression(init) || ts.isParenthesizedExpression(init))) init = init.expression
    if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) {
        addWord(makeFunctionWord(init, name, checker, spec, module, doc, access, file, decl))
        return
    }
    if (init && ts.isObjectLiteralExpression(init)) {
        const members: string[] = []
        for (const prop of init.properties) {
            const memberName = prop.name && (ts.isIdentifier(prop.name) || ts.isStringLiteral(prop.name)) ? prop.name.text : null
            if (!memberName) continue
            const memberDoc = readJsDoc(prop)
            if (memberDoc.tags.internal) continue
            const qualified = `${name}.${memberName}`
            let memberInit: ts.Expression | undefined = ts.isPropertyAssignment(prop) ? prop.initializer : undefined
            while (memberInit && (ts.isAsExpression(memberInit) || ts.isSatisfiesExpression(memberInit) || ts.isParenthesizedExpression(memberInit))) memberInit = memberInit.expression
            if (ts.isMethodDeclaration(prop) || (memberInit && (ts.isArrowFunction(memberInit) || ts.isFunctionExpression(memberInit)))) {
                const fn = ts.isMethodDeclaration(prop) ? prop : memberInit as ts.ArrowFunction | ts.FunctionExpression
                const w = makeFunctionWord(fn, memberName, checker, spec, module, memberDoc, access, file, prop, qualified)
                members.push(w.id)
                addWord(w)
            } else if (ts.isShorthandPropertyAssignment(prop)) {
                // `{ steer, seek }` — the member is a function declared elsewhere in the module.
                const symbol = checker.getShorthandAssignmentValueSymbol(prop)
                const target = symbol?.valueDeclaration
                let fn: ts.SignatureDeclaration | undefined
                if (target && ts.isFunctionDeclaration(target)) fn = target
                else if (target && ts.isVariableDeclaration(target) && target.initializer) {
                    let ini: ts.Expression = target.initializer
                    while (ts.isAsExpression(ini) || ts.isSatisfiesExpression(ini) || ts.isParenthesizedExpression(ini)) ini = ini.expression
                    if (ts.isArrowFunction(ini) || ts.isFunctionExpression(ini)) fn = ini
                }
                const targetDoc = target ? readJsDoc(ts.isVariableDeclaration(target) ? target.parent.parent : target) : memberDoc
                const docFor = memberDoc.summary ? memberDoc : targetDoc
                if (docFor.tags.internal) continue
                if (fn) {
                    const w = makeFunctionWord(fn, memberName, checker, spec, module, docFor, access, file, target ?? prop, qualified)
                    members.push(w.id)
                    addWord(w)
                } else {
                    const type = checker.getTypeAtLocation(prop.name)
                    const w: DocsWord = {
                        ...baseWord(memberName, qualified, spec, module, docFor, access, file, target ?? prop),
                        kind: type.getCallSignatures().length ? 'function' : 'constant',
                        signature: `${qualified}: ${checker.typeToString(type, undefined, ts.TypeFormatFlags.NoTruncation)}`,
                        params: [],
                    }
                    members.push(w.id)
                    addWord(w)
                }
            } else if (ts.isPropertyAssignment(prop)) {
                const type = checker.getTypeAtLocation(prop.initializer)
                const w: DocsWord = {
                    ...baseWord(memberName, qualified, spec, module, memberDoc, access, file, prop),
                    kind: type.getCallSignatures().length ? 'function' : 'constant',
                    signature: `${qualified}: ${checker.typeToString(type, undefined, ts.TypeFormatFlags.NoTruncation)}`,
                    params: [],
                }
                members.push(w.id)
                addWord(w)
            }
        }
        addWord({
            ...baseWord(name, name, spec, module, doc, access, file, decl),
            kind: 'object',
            signature: `${name}: { ${members.map((m) => m.split('.').pop()).join(', ')} }`,
            members,
            params: [],
        })
        return
    }
    // A cast/derived value (`export const circle = field(...) as (x, y, r) => Expr`), a class
    // instance, or a plain constant: describe it by its type.
    const type = checker.getTypeAtLocation(decl.name)
    const signatures = type.getCallSignatures()
    if (signatures.length) {
        const sig = signatures[0]
        addWord({
            ...baseWord(name, name, spec, module, doc, access, file, decl),
            kind: 'function',
            signature: `${name}${checker.signatureToString(sig, undefined, ts.TypeFormatFlags.NoTruncation)}`,
            params: sig.getParameters().map((p) => paramFromSymbol(p, checker, doc)),
            returns: doc.tags.returns ?? undefined,
        })
        return
    }
    addWord({
        ...baseWord(name, name, spec, module, doc, access, file, decl),
        kind: 'constant',
        signature: `${name}: ${checker.typeToString(type, undefined, ts.TypeFormatFlags.NoTruncation)}`,
        params: [],
    })
}

function makeFunctionWord(
    fn: ts.SignatureDeclaration,
    name: string,
    checker: ts.TypeChecker,
    spec: CategorySpec,
    module: string,
    doc: JsDoc,
    access: IndexAccess,
    file: ts.SourceFile,
    anchor: ts.Node = fn,
    qualified = name,
): DocsWord {
    const sig = checker.getSignatureFromDeclaration(fn)
    const params: DocsParam[] = fn.parameters.map((p) => ({
        name: `${p.dotDotDotToken ? '...' : ''}${p.name.getText(file)}`,
        type: p.type ? p.type.getText(file) : checker.typeToString(checker.getTypeAtLocation(p), undefined, ts.TypeFormatFlags.NoTruncation),
        optional: !!p.questionToken || !!p.initializer,
        ...(p.initializer ? {defaultValue: p.initializer.getText(file)} : {}),
        ...(doc.params[p.name.getText(file)] ? {doc: doc.params[p.name.getText(file)]} : {}),
    }))
    for (const param of params) if (param.name.startsWith('...')) param.optional = true
    const typeParams = fn.typeParameters?.length ? `<${fn.typeParameters.map((t) => t.getText(file)).join(', ')}>` : ''
    const returnType = fn.type
        ? fn.type.getText(file)
        : sig ? checker.typeToString(checker.getReturnTypeOfSignature(sig), undefined, ts.TypeFormatFlags.NoTruncation) : 'unknown'
    const paramText = params.map((p) => `${p.name}${p.optional && !p.defaultValue && !p.name.startsWith('...') ? '?' : ''}: ${p.type}${p.defaultValue ? ` = ${p.defaultValue}` : ''}`).join(', ')
    return {
        ...baseWord(name, qualified, spec, module, doc, access, file, anchor),
        kind: 'function',
        signature: `${qualified}${typeParams}(${paramText}): ${returnType}`,
        params,
        returns: doc.tags.returns ?? undefined,
    }
}

function baseWord(
    name: string,
    qualified: string,
    spec: CategorySpec,
    module: string,
    doc: JsDoc,
    access: IndexAccess,
    file: ts.SourceFile,
    anchor: ts.Node,
): Omit<DocsWord, 'kind' | 'signature' | 'params'> {
    const category = doc.tags.category ?? spec.id
    const id = `${category}.${qualified}`
    const ns = access.namespaces.get(module)
    const root = qualified.split('.')[0]
    const importInfo = ns
        ? {statement: `import {${ns.split('.')[0]}} from 'shaders/std'`, usage: `${ns}.${qualified}`}
        : {statement: `import {${root}} from 'shaders/std'`, usage: qualified}
    return {
        id,
        name,
        qualifiedName: qualified,
        category,
        module,
        summary: doc.summary,
        description: doc.description,
        examples: doc.tags.examples,
        tips: doc.tags.tips,
        see: doc.tags.see,
        ...(doc.tags.deprecated !== undefined ? {deprecated: doc.tags.deprecated} : {}),
        import: importInfo,
        usedBy: [],
        documented: doc.summary.length > 0,
        source: {file: module, line: file.getLineAndCharacterOfPosition(anchor.getStart(file)).line + 1},
    }
}

function paramFromSymbol(symbol: ts.Symbol, checker: ts.TypeChecker, doc: JsDoc): DocsParam {
    const decl = symbol.valueDeclaration
    const type = decl ? checker.getTypeOfSymbolAtLocation(symbol, decl) : checker.getDeclaredTypeOfSymbol(symbol)
    const optional = !!decl && ts.isParameter(decl) && (!!decl.questionToken || !!decl.initializer)
    return {
        name: symbol.name,
        type: checker.typeToString(type, undefined, ts.TypeFormatFlags.NoTruncation),
        optional,
        ...(doc.params[symbol.name] ? {doc: doc.params[symbol.name]} : {}),
    }
}


function hasExportModifier(node: ts.Node): boolean {
    return !!(ts.canHaveModifiers(node) && ts.getModifiers(node)?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword))
}

// ── JSDoc ───────────────────────────────────────────────────────────────────────────────

interface JsDoc {
    summary: string
    description: string
    params: Record<string, string>
    tags: {
        category?: string
        returns?: string
        deprecated?: string
        internal: boolean
        examples: string[]
        tips: string[]
        see: string[]
    }
}

function readJsDoc(node: ts.Node): JsDoc {
    const out: JsDoc = {summary: '', description: '', params: {}, tags: {internal: false, examples: [], tips: [], see: []}}
    const docs = ts.getJSDocCommentsAndTags(node).filter(ts.isJSDoc)
    const jsdoc = docs[docs.length - 1]
    if (!jsdoc) return out
    const text = normalizeDocText(ts.getTextOfJSDocComment(jsdoc.comment) ?? '')
    const [first, ...rest] = splitParagraphs(text)
    const {summary, remainder} = splitSummary(first ?? '')
    out.summary = summary
    out.description = [remainder, ...rest].filter(Boolean).join('\n\n')
    for (const tag of jsdoc.tags ?? []) {
        const name = tag.tagName.text
        const comment = normalizeDocText(ts.getTextOfJSDocComment(tag.comment) ?? '')
        switch (name) {
            case 'param': {
                const pname = ts.isJSDocParameterTag(tag) ? tag.name.getText() : comment.split(/\s+/)[0]
                out.params[pname] = comment
                break
            }
            case 'returns':
            case 'return':
                out.tags.returns = comment
                break
            case 'category':
                out.tags.category = comment.trim()
                break
            case 'example':
                out.tags.examples.push(stripFence(comment))
                break
            case 'tip':
                out.tags.tips.push(comment)
                break
            case 'see': {
                // `@see rings, tiles` — TypeScript parses the first name as the tag's `name`
                // reference (a JSDocSeeTag), with only the remainder in `comment`; read both.
                const lead = ts.isJSDocSeeTag(tag) && tag.name ? tag.name.getText() : ''
                out.tags.see.push(...`${lead} ${comment}`.split(/[,\s]+/).map((s) => s.replace(/^\{@link\s*|\}$/g, '').replace(/[`.]+$/g, '').replace(/^`/, '')).filter(Boolean))
                break
            }
            case 'deprecated':
                out.tags.deprecated = comment
                break
            case 'internal':
                out.tags.internal = true
                break
        }
    }
    return out
}

function normalizeDocText(text: string): string {
    return text
        .replace(/\r\n/g, '\n')
        .replace(/[ \t]+\n/g, '\n')
        // `{@link word}` / `{@link word|label}` → a backticked name renderers can resolve.
        .replace(/\{@link\s+([^}|\s]+)(?:\s*\|\s*([^}]+))?\}/g, (_m, ref: string, label?: string) => `\`${(label ?? ref).trim()}\``)
        .trim()
}

/**
 * The summary is the FIRST SENTENCE of the first paragraph; whatever follows in that paragraph
 * joins the description. Sentence ends are `.`/`!`/`?` followed by whitespace and a capital, a
 * backtick or a paren — a `.` inside `d.vec4f` or `0.5` does not end a sentence.
 */
function splitSummary(paragraph: string): {summary: string; remainder: string} {
    const m = paragraph.match(/^(.+?[.!?])(?:\s+(?=[A-Z`(\[]))([\s\S]*)$/)
    if (!m || m[1].length < 12) return {summary: paragraph, remainder: ''}
    return {summary: m[1].trim(), remainder: (m[2] ?? '').trim()}
}

function splitParagraphs(text: string): string[] {
    return text.split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, ' ').trim()).filter(Boolean)
}

function stripFence(text: string): string {
    const m = text.match(/^```[a-z]*\n([\s\S]*?)\n```$/)
    return m ? m[1] : text
}

// ── Curation (packages/core/docs/std/<category>.md) ─────────────────────────────────────

export interface Curation {
    title?: string
    intro?: string
    order: string[]
    reach: {when: string; use: string}[]
    example?: string
}

/**
 * Parse a curation file. Convention (all sections optional):
 *
 *     # Title
 *     Intro paragraphs…
 *     ## Reach for it when
 *     | When | Use |
 *     |---|---|
 *     | you need X | `rampOver` |
 *     ## Order
 *     - rampOver
 *     - stops
 *     ## Example
 *     ```ts
 *     …
 *     ```
 */
export function parseCuration(markdown: string): Curation {
    const lines = markdown.replace(/\r\n/g, '\n').split('\n')
    const out: Curation = {order: [], reach: []}
    let section: 'intro' | 'reach' | 'order' | 'example' | 'other' = 'intro'
    const intro: string[] = []
    const example: string[] = []
    let inFence = false
    for (const line of lines) {
        const h1 = line.match(/^#\s+(.+)$/)
        if (h1 && section === 'intro' && !out.title) {
            out.title = h1[1].trim()
            continue
        }
        const h2 = line.match(/^##\s+(.+)$/)
        if (h2 && !inFence) {
            const t = h2[1].trim().toLowerCase()
            section = t.startsWith('reach') ? 'reach' : t === 'order' ? 'order' : t.startsWith('example') ? 'example' : 'other'
            continue
        }
        if (section === 'intro') intro.push(line)
        else if (section === 'reach') {
            const row = line.match(/^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$/)
            if (row && !/^-+$/.test(row[1].trim()) && row[1].trim().toLowerCase() !== 'when') {
                out.reach.push({when: row[1].trim(), use: row[2].trim()})
            }
        } else if (section === 'order') {
            const item = line.match(/^[-*]\s+`?([A-Za-z0-9_.]+)`?/)
            if (item) out.order.push(item[1])
        } else if (section === 'example') {
            if (/^```/.test(line)) {
                inFence = !inFence
                continue
            }
            if (inFence) example.push(line)
        }
    }
    const introText = intro.join('\n').trim()
    if (introText) out.intro = introText
    if (example.length) out.example = example.join('\n')
    return out
}

function applyCuration(coreDir: string, categories: Record<string, DocsCategory>, words: Record<string, DocsWord>): void {
    const dir = path.join(coreDir, 'docs/std')
    if (!fs.existsSync(dir)) return
    for (const category of Object.values(categories)) {
        const file = path.join(dir, `${category.id}.md`)
        if (!fs.existsSync(file)) continue
        const curation = parseCuration(fs.readFileSync(file, 'utf8'))
        category.curated = true
        if (curation.title) category.title = curation.title
        if (curation.intro) category.intro = curation.intro
        if (curation.example) category.example = curation.example
        if (curation.reach.length) category.reach = curation.reach
        // Curated order: names resolve to word ids in this category; unknown names are an error
        // (a typo here would silently hide a word).
        const known = new Set(category.words)
        const ordered: string[] = []
        for (const name of curation.order) {
            const id = `${category.id}.${name}`
            if (!known.has(id) || !words[id]) throw new Error(`docs manifest: ${category.id}.md orders unknown word '${name}'`)
            ordered.push(id)
        }
        for (const entry of category.reach ?? []) {
            for (const m of entry.use.matchAll(/`([A-Za-z0-9_.]+)`/g)) {
                const ref = m[1]
                const id = `${category.id}.${ref}`
                const known = !!words[id] || Object.values(words).some((w) => w.qualifiedName === ref || w.name === ref)
                if (!known) throw new Error(`docs manifest: ${category.id}.md reach table names unknown word '${ref}'`)
            }
        }
        ;(category as DocsCategory & {_order?: string[]})._order = ordered
    }
}

function orderWords(category: DocsCategory, words: Record<string, DocsWord>): string[] {
    const ordered = (category as DocsCategory & {_order?: string[]})._order ?? []
    delete (category as DocsCategory & {_order?: string[]})._order
    const rest = category.words.filter((id) => !ordered.includes(id)).sort((a, b) => words[a].qualifiedName.localeCompare(words[b].qualifiedName))
    return [...ordered, ...rest]
}

// ── Used by (library shaders) ───────────────────────────────────────────────────────────

function applyUsedBy(srcDir: string, words: Record<string, DocsWord>, access: IndexAccess): void {
    const shadersDir = path.join(srcDir, 'shaders')
    if (!fs.existsSync(shadersDir)) return
    const byModuleName = new Map<string, DocsWord[]>()
    for (const w of Object.values(words)) {
        const key = `${w.module}::${w.qualifiedName.split('.')[0]}`
        const arr = byModuleName.get(key) ?? []
        arr.push(w)
        byModuleName.set(key, arr)
    }
    const indexModules = new Map<string, string>() // top-level name → module (via std/index re-exports)
    for (const [mod, names] of access.topLevel) for (const n of names) indexModules.set(n, mod)

    for (const entry of fs.readdirSync(shadersDir, {withFileTypes: true})) {
        if (!entry.isDirectory()) continue
        const file = path.join(shadersDir, entry.name, 'index.ts')
        if (!fs.existsSync(file)) continue
        const text = fs.readFileSync(file, 'utf8')
        const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true)
        const shaderName = readShaderName(text, entry.name)
        for (const st of sf.statements) {
            if (!ts.isImportDeclaration(st) || !ts.isStringLiteral(st.moduleSpecifier)) continue
            const spec = st.moduleSpecifier.text
            if (!spec.startsWith('@coreroot/std')) continue
            const clause = st.importClause
            if (!clause?.namedBindings || !ts.isNamedImports(clause.namedBindings)) continue
            const modFromSpec = spec === '@coreroot/std' ? null : `${spec.replace('@coreroot/', '')}.ts`
            for (const el of clause.namedBindings.elements) {
                const imported = (el.propertyName ?? el.name).text
                const local = el.name.text
                const mod = modFromSpec ?? indexModules.get(imported)
                if (!mod) continue
                const candidates = byModuleName.get(`${mod}::${imported}`) ?? []
                for (const w of candidates) {
                    const qualifiedLocal = w.qualifiedName.includes('.') ? `${local}.${w.qualifiedName.split('.').slice(1).join('.')}` : local
                    const re = new RegExp(`\\b${qualifiedLocal.replace(/\./g, '\\.')}\\b`)
                    const body = text.slice(st.end)
                    if (re.test(body) && !w.usedBy.includes(shaderName)) w.usedBy.push(shaderName)
                }
            }
        }
    }
    for (const w of Object.values(words)) w.usedBy.sort()
}

/**
 * A shader's public name: the `name:` inside its definition call (`defineStd({`, `defineShader({`,
 * or a `componentDefinition … = {` literal) — NOT the first `name:` in the file, which in a
 * simulation shader is often a kernel's. The folder name is the fallback (it matches by convention).
 */
function readShaderName(source: string, folderName: string): string {
    const inDefine = source.match(/\b(?:defineStd|defineShader)\s*(?:<[^>]*>)?\s*\(\s*\{[\s\S]*?\bname:\s*["'`]([^"'`]+)["'`]/)
    if (inDefine) return inDefine[1]
    const inLiteral = source.match(/\bcomponentDefinition\b[^=]*=\s*\{[\s\S]*?\bname:\s*["'`]([^"'`]+)["'`]/)
    if (inLiteral) return inLiteral[1]
    return folderName
}

// ── @see resolution ─────────────────────────────────────────────────────────────────────

function resolveSee(words: Record<string, DocsWord>): void {
    const byName = new Map<string, string[]>()
    for (const w of Object.values(words)) {
        for (const key of [w.name, w.qualifiedName]) {
            const arr = byName.get(key) ?? []
            if (!arr.includes(w.id)) arr.push(w.id)
            byName.set(key, arr)
        }
    }
    for (const w of Object.values(words)) {
        w.see = w.see.map((ref) => {
            if (words[ref]) return ref
            const sameCategory = byName.get(ref)?.find((id) => id.startsWith(`${w.category}.`))
            return sameCategory ?? byName.get(ref)?.[0] ?? ref
        })
    }
}

// ── Version ─────────────────────────────────────────────────────────────────────────────

function readEngineVersion(coreDir: string): string {
    for (const candidate of [path.join(coreDir, '../shaders/package.json'), path.join(coreDir, 'package.json')]) {
        try {
            return JSON.parse(fs.readFileSync(candidate, 'utf8')).version
        } catch {
            /* next */
        }
    }
    return '0.0.0'
}

// ── Renderers shared by the CLI and dotcom ──────────────────────────────────────────────

/** `llms.txt`: the whole vocabulary as compact markdown an agent can load in one read. */
export function renderLlmsText(manifest: DocsManifest, options: {baseUrl?: string} = {}): string {
    const base = options.baseUrl?.replace(/\/$/, '') ?? ''
    const lines: string[] = [
        `# Shaders std vocabulary (engine ${manifest.engineVersion})`,
        '',
        'Compose shaders from these words with `defineShader` from `shaders/std`. Every word is a plain TypeScript',
        'function; the engine lowers the composition to WGSL at runtime. A `wgsl` body is the escape hatch.',
        '',
    ]
    for (const group of manifest.groups) {
        lines.push(`## ${group.title}`, '')
        for (const cid of group.categories) {
            const c = manifest.categories[cid]
            if (!c) continue
            const accessText = c.access.kind === 'namespace' ? `\`import {${c.access.name.split('.')[0]}} from 'shaders/std'\`` : '`shaders/std` top level'
            lines.push(`### ${c.title} (${accessText})${base ? ` — ${base}/docs/primitives/${c.id}` : ''}`)
            if (c.intro) lines.push('', c.intro.split('\n\n')[0])
            lines.push('')
            for (const wid of c.words) {
                const w = manifest.words[wid]
                if (!w || w.kind === 'object') continue
                lines.push(`- \`${w.signature}\`${w.summary ? ` — ${w.summary}` : ''}`)
            }
            lines.push('')
        }
    }
    return lines.join('\n')
}

/** One markdown document per category — the human page body and the per-category agent export. */
export function renderCategoryMarkdown(manifest: DocsManifest, categoryId: string): string {
    const c = manifest.categories[categoryId]
    if (!c) throw new Error(`docs manifest: unknown category ${categoryId}`)
    const out: string[] = [`# ${c.title}`, '']
    const importLine = c.access.kind === 'namespace'
        ? `import {${c.access.name.split('.')[0]}} from 'shaders/std'`
        : `import {${c.words.slice(0, 3).map((id) => manifest.words[id]?.name).filter(Boolean).join(', ')}${c.words.length > 3 ? ', …' : ''}} from 'shaders/std'`
    out.push('```ts', importLine, '```', '')
    if (c.intro) out.push(c.intro, '')
    if (c.reach?.length) {
        out.push('## Reach for it when', '', '| When | Use |', '|---|---|')
        for (const r of c.reach) out.push(`| ${r.when} | ${r.use} |`)
        out.push('')
    }
    if (c.example) out.push('## Example', '', '```ts', c.example, '```', '')
    out.push('## Words', '')
    for (const wid of c.words) {
        const w = manifest.words[wid]
        if (!w) continue
        out.push(`### ${w.qualifiedName}`, '', '```ts', w.signature, '```', '')
        if (w.deprecated) out.push(`> Deprecated: ${w.deprecated}`, '')
        if (w.summary) out.push(w.summary, '')
        if (w.description) out.push(w.description, '')
        if (w.params.length) {
            out.push('| Parameter | Type | Notes |', '|---|---|---|')
            for (const p of w.params) out.push(`| \`${p.name}\`${p.optional ? ' (optional)' : ''} | \`${p.type}\` | ${p.defaultValue ? `default \`${p.defaultValue}\`. ` : ''}${p.doc ?? ''} |`)
            out.push('')
        }
        if (w.returns) out.push(`Returns: ${w.returns}`, '')
        for (const ex of w.examples) out.push('```ts', ex, '```', '')
        for (const tip of w.tips) out.push(`> **Tip.** ${tip}`, '')
        if (w.see.length) out.push(`See also: ${w.see.map((s) => `\`${manifest.words[s]?.qualifiedName ?? s}\``).join(', ')}`, '')
        if (w.usedBy.length) out.push(`Used by: ${w.usedBy.join(', ')}`, '')
    }
    return out.join('\n')
}

// ── CLI ─────────────────────────────────────────────────────────────────────────────────

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isMain) {
    const coreDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
    const outDir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(coreDir, 'dist')
    const manifest = buildDocsManifest({coreDir})
    fs.mkdirSync(outDir, {recursive: true})
    fs.writeFileSync(path.join(outDir, 'docs-manifest.json'), JSON.stringify(manifest, null, 2))
    fs.writeFileSync(path.join(outDir, 'llms.txt'), renderLlmsText(manifest, {baseUrl: 'https://shaders.com'}))
    const {words, documented, curatedCategories} = manifest.coverage
    console.log(`📚 docs-manifest.json: ${words} words, ${documented} documented (${Math.round((documented / words) * 100)}%), ${curatedCategories.length} curated categories → ${outDir}`)
}
