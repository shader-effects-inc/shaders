/**
 * std/document — a `wgsl` envelope written out as a source file: the `defineShader({...})` module
 * code export puts next to the exported shader, so the component lives in the user's project as
 * ordinary, editable code.
 *
 *   wgslComponentSource(envelope, {identifier: 'Halo'})
 *
 * The object written is `wgslComponentData(envelope)`, the same data the editor builds its live
 * definition from, so the file compiles to the shader the editor showed.
 */
import {wgslComponentData} from './wgslEnvelope'
import type {GraphPropConfigData, GraphPropTransformRef} from './define'
import type {WgslEnvelope} from './types'

export interface ComponentSourceOptions {
    /** The exported variable, e.g. `Halo`. Must be a valid identifier. */
    identifier: string
    /** The definition's `name` (what preset JSON and the registry call it). Defaults to the envelope's. */
    name?: string
    /** Lines for a comment at the top of the file, without the `// `. */
    header?: string[]
}

const IDENTIFIER = /^[A-Za-z_$][\w$]*$/
/** Widest a value is written on one line before it breaks across lines. */
const INLINE_WIDTH = 100

function quote(text: string): string {
    return `'${text.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n')}'`
}

function key(name: string): string {
    return IDENTIFIER.test(name) ? name : quote(name)
}

/** A JSON-like value as a JS literal: on one line when short, else one entry per line. */
function literal(value: unknown, indent: string): string {
    if (value === undefined || value === null) return 'null'
    if (typeof value === 'string') return quote(value)
    if (typeof value !== 'object') return String(value)
    const entries = Array.isArray(value)
        ? value.map((item) => literal(item, `${indent}  `))
        : Object.entries(value as Record<string, unknown>)
            .filter(([, v]) => v !== undefined)
            .map(([k, v]) => `${key(k)}: ${literal(v, `${indent}  `)}`)
    const [open, close] = Array.isArray(value) ? ['[', ']'] : ['{', '}']
    if (entries.length === 0) return `${open}${close}`
    const inline = Array.isArray(value) ? `[${entries.join(', ')}]` : `{ ${entries.join(', ')} }`
    if (inline.length + indent.length <= INLINE_WIDTH && !inline.includes('\n')) return inline
    return `${open}\n${entries.map((entry) => `${indent}  ${entry},`).join('\n')}\n${indent}${close}`
}

/** The function a transform ref imports from `shaders/std`. */
function transformImport(transform: GraphPropTransformRef): string {
    return typeof transform === 'object' ? 'selectIndex' : transform
}

/** A transform ref as source: the imported function, or `selectIndex([...])`. */
function transformSource(transform: GraphPropTransformRef, indent: string): string {
    return typeof transform === 'object' ? `selectIndex(${literal(transform.selectIndex, indent)})` : transform
}

/** One prop config, its transform written as the imported function's name. */
function propSource(config: GraphPropConfigData, indent: string): string {
    const {transform, description, ...rest} = config
    const label = (rest.ui as {label?: string} | undefined)?.label
    const fields: string[] = [`default: ${literal(rest.default, `${indent}  `)}`]
    if (transform) fields.push(`transform: ${transformSource(transform, `${indent}  `)}`)
    for (const [k, v] of Object.entries(rest)) {
        if (k !== 'default' && v !== undefined) fields.push(`${key(k)}: ${literal(v, `${indent}  `)}`)
    }
    // The description repeats the label unless the author wrote one
    if (description && description !== label) fields.push(`description: ${quote(description)}`)
    const inline = `{ ${fields.join(', ')} }`
    if (inline.length + indent.length <= INLINE_WIDTH) return inline
    return `{\n${fields.map((field) => `${indent}  ${field},`).join('\n')}\n${indent}}`
}

/** The body inside a template literal: re-indented, with what a template literal would read escaped. */
function bodySource(body: string, indent: string): string {
    const lines = body.replace(/\t/g, '    ').split('\n')
    while (lines.length && !lines[0]!.trim()) lines.shift()
    while (lines.length && !lines[lines.length - 1]!.trim()) lines.pop()
    const margin = Math.min(...lines.filter((line) => line.trim()).map((line) => line.match(/^ */)![0].length))
    return lines
        .map((line) => (line.trim() ? `${indent}${line.slice(margin).trimEnd()}` : ''))
        .join('\n')
        .replace(/\\/g, '\\\\')
        .replace(/`/g, '\\`')
        .replace(/\$\{/g, '\\${')
}

/**
 * A `wgsl` envelope as the source of a module exporting its `defineShader` definition.
 *
 * Imports come from `shaders/std`. Throws StdGraphError when the envelope doesn't validate.
 */
export function wgslComponentSource(envelope: WgslEnvelope, options: ComponentSourceOptions): string {
    if (!IDENTIFIER.test(options.identifier)) throw new Error(`wgslComponentSource: "${options.identifier}" is not a valid identifier`)
    const data = wgslComponentData(envelope)
    const name = options.name ?? data.name

    const transforms = [...new Set(Object.values(data.props).flatMap((config) => (config.transform ? [transformImport(config.transform)] : [])))].sort()
    const imports = ['defineShader', 'wgsl', ...transforms]

    const props = Object.entries(data.props)
        .map(([prop, config]) => `    ${key(prop)}: ${propSource(config, '    ')},`)
        .join('\n')

    const lines = [
        ...(options.header ?? []).map((line) => `// ${line}`),
        `import { ${imports.join(', ')} } from 'shaders/std'`,
        '',
        `export const ${options.identifier} = defineShader({`,
        `  name: ${quote(name)},`,
        ...(data.description ? [`  description: ${quote(data.description)},`] : []),
        ...(data.animatedTime ? [`  animatedTime: { speed: 'speed' },`] : []),
        props ? `  props: {\n${props}\n  },` : '  props: {},',
        `  ${data.field}: wgsl\``,
        bodySource(data.body, '    '),
        '  `,',
        '})',
        '',
    ]
    return lines.join('\n')
}
