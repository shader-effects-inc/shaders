/**
 * Turn a component's IR back into a working definition: every `{$word}` becomes a call to that std
 * word, `{$prop}` a prop reference, `{$wgsl}` a WGSL body, `{$lib}` the library function. Only the
 * allowlisted words can be reached, so evaluating an IR runs no code from the file it came from.
 */
// Maintainer notes. This module imports every word (the editor renders any component), so it lives
// behind `shaders/std/document` and never on an app's runtime path (§8.4 T6). A `{$const}` is
// evaluated once and shared, so a const used twice (an input listed in `inputs` and handed to
// `rasterize`) is the same object, exactly as in the running file.
import type {GpuShaderDefinition} from '../../gpu/contract'
import {defineShader} from '../lower'
import {wgsl} from '../wgsl'
import {p} from '../values'
import {pointer, pointerSpeed} from '../signal'
import {crosses, isZero, isValue, allOf} from '../slots'
import {child} from '../inputs/types'
import {blur} from '../inputs/blur'
import {bloom} from '../inputs/bloom'
import {pass, downsample, directionalBlur} from '../inputs/passes'
import {previousFrame} from '../inputs/history'
import {struct, buffer, kernel} from '../inputs/compute'
import {rasterize} from '../inputs/rasterize'
import {events} from '../inputs/events'
import {pointerTrail} from '../inputs/trail'
import {text, samplePoints} from '../inputs/text'
import {displaceBy, tintToward} from '../filter'
import * as simulate from '../simulate'
import * as op from '../ops'
import * as library from '../library/index'
import {colorStopsTransform} from '../../utilities/colorStops'
import {listPropTransform} from '../../utilities/listProps'
import {
    transformAngle, transformBool01, transformBoolean, transformColor, transformColorSpace, transformEdges, transformPosition,
} from '../../utilities/transformations'
import type {ComponentIR, IRValue} from './parse'
import {CALLABLE_WORDS, COMPONENT_IR_VERSION} from './parse'

type AnyFn = (...args: unknown[]) => unknown

// Built on first use, never at import: a table built by a call at module level is a side effect,
// and would keep every word in any bundle that shares a chunk with this module (§8.4).
let WORDS: Record<string, AnyFn> | null = null
function words(): Record<string, AnyFn> {
    WORDS ??= {
        p, pointer, pointerSpeed, crosses, isZero, isValue, allOf,
        blur, bloom, pass, downsample, directionalBlur, previousFrame,
        struct, buffer, kernel, rasterize, events, pointerTrail, text, samplePoints,
        displaceBy, tintToward,
        'simulate.grid': simulate.grid, 'simulate.particles': simulate.particles, 'simulate.fluid': simulate.fluid,
        ...Object.fromEntries(Object.entries(op).map(([k, v]) => [`op.${k}`, v])),
        wgsl: (spec: unknown) => wgsl(spec as never),
    } as Record<string, AnyFn>
    return WORDS
}

const VALUES: Record<string, unknown> = {
    child, transformColor, transformPosition, transformBoolean, transformBool01, transformEdges, transformAngle,
    transformColorSpace, colorStopsTransform, listPropTransform,
}

/** Thrown when an IR cannot be turned into a definition (an unknown word, a bad argument). */
export class ComponentIRError extends Error {
    constructor(message: string) {
        super(message)
        this.name = 'ComponentIRError'
    }
}

/** Evaluate an IR to the definition its source file would produce when run. */
export function evaluateComponentIR(ir: ComponentIR): GpuShaderDefinition {
    if (!ir || typeof ir !== 'object' || ir.v !== COMPONENT_IR_VERSION) {
        throw new ComponentIRError(`unsupported component IR version ${String(ir?.v)} (this engine reads ${COMPONENT_IR_VERSION})`)
    }
    const constValues = new Map<string, unknown>()
    const constIR = new Map(ir.consts.map((c) => [c.name, c.value]))
    const evaluating = new Set<string>()

    const value = (v: IRValue, path: string): unknown => {
        if (v === null || typeof v !== 'object') return v
        if (Array.isArray(v)) return v.map((x, i) => value(x, `${path}[${i}]`))
        if ('$const' in v) {
            const name = String(v.$const)
            if (constValues.has(name)) return constValues.get(name)
            if (evaluating.has(name)) throw new ComponentIRError(`const ${name} refers to itself`)
            const raw = constIR.get(name)
            if (raw === undefined) throw new ComponentIRError(`${path}: unknown const ${name}`)
            evaluating.add(name)
            const out = value(raw, name)
            evaluating.delete(name)
            constValues.set(name, out)
            return out
        }
        if ('$wgsl' in v) return wgsl(String(v.$wgsl))
        if ('$prop' in v) return p(String(v.$prop))
        if ('$value' in v) {
            const name = String(v.$value)
            if (!(name in VALUES)) throw new ComponentIRError(`${path}: '${name}' is not a value a component may use`)
            return VALUES[name]
        }
        if ('$lib' in v) {
            const name = String(v.$lib)
            const fn = (library as Record<string, unknown>)[name]
            if (!fn) throw new ComponentIRError(`${path}: '${name}' is not in the WGSL library`)
            return fn
        }
        if ('$word' in v) {
            const name = String(v.$word)
            const fn = words()[name]
            if (!fn || !CALLABLE_WORDS.has(name)) throw new ComponentIRError(`${path}: '${name}' is not a word a component may call`)
            const args = Array.isArray(v.args) ? v.args : []
            try {
                return fn(...args.map((a, i) => value(a, `${path}.${name}(${i})`)))
            } catch (e) {
                throw new ComponentIRError(`${path}: ${name}(…): ${e instanceof Error ? e.message : String(e)}`)
            }
        }
        const out: Record<string, unknown> = {}
        for (const [k, x] of Object.entries(v)) out[k] = value(x, path ? `${path}.${k}` : k)
        return out
    }

    const definition = value(ir.definition, '') as Record<string, unknown>
    return defineShader(definition as never)
}
