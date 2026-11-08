/**
 * std/document — shader definitions written as data (see `types.ts`): validate them, lower them
 * to ordinary definitions, and print the primitive catalog a `std-graph` is written against.
 */
import type {GpuShaderDefinition} from '../../gpu/contract'
import {defineStdGraph, validateStdGraph, type StdGraphValidation} from './define'
import {defineWgslEnvelope, validateWgslEnvelope, type WgslEnvelopeValidation} from './wgslEnvelope'
import type {ShaderEnvelope} from './types'

export {defineStdGraph, validateStdGraph, normalizeStdGraph, renderStdGraphManifest, resolveStdGraphWgsl} from './define'
export type {StdGraphValidation} from './define'
export {defineWgslEnvelope, validateWgslEnvelope, wgslComponentData} from './wgslEnvelope'
export type {WgslEnvelopeValidation, WgslComponentData} from './wgslEnvelope'
export {wgslComponentSource} from './source'
export type {ComponentSourceOptions} from './source'
export {STD_GRAPH_LIMITS, StdGraphError} from './types'
export type {GraphArg, GraphNode, GraphPropKind, GraphPropSpec, StdGraphEnvelope, WgslEnvelope, ShaderEnvelope} from './types'

/** Either kind's validation result. */
export type ShaderEnvelopeValidation = StdGraphValidation | WgslEnvelopeValidation

/** Static check of any shader envelope, dispatched on `kind`. Never throws. */
export function validateShaderEnvelope(envelope: unknown): ShaderEnvelopeValidation {
    const kind = (envelope as {kind?: unknown} | null)?.kind
    if (kind === 'std-graph') return validateStdGraph(envelope)
    if (kind === 'wgsl') return validateWgslEnvelope(envelope)
    return {ok: false, error: 'kind must be "std-graph" (composed from primitives) or "wgsl" (a raw body)'}
}

/** Lower any shader envelope to an ordinary definition. Throws StdGraphError when it does not validate. */
export function defineShaderEnvelope(envelope: ShaderEnvelope): GpuShaderDefinition {
    return envelope.kind === 'wgsl' ? defineWgslEnvelope(envelope) : defineStdGraph(envelope)
}

// ── The agent format: a TypeScript file, read without running it ─────────────────────────

export {parseComponentSource, componentSourceForExport, COMPONENT_IR_VERSION, PARSER_VERSION, CALLABLE_WORDS, VALUE_WORDS, LIBRARY_NAMES, ALLOWED_MODULES} from './parse'
export type {ComponentIR, IRValue, ParseIssue, ParseResult, ComponentExportOptions} from './parse'
export {evaluateComponentIR, ComponentIRError} from './evaluate'
import {evaluateComponentIR} from './evaluate'
import type {ComponentIR} from './parse'
import {fingerprint} from './define'

/**
 * A custom component as stored with a preset: the TypeScript source exactly as written (the truth,
 * returned on pull) and the IR read from it (what the editor renders). `hash` fingerprints the
 * source, so a stale IR can be detected and re-derived.
 */
export interface StoredComponent {
    kind: 'source'
    source: string
    ir: ComponentIR
    irVersion: number
    parserVersion: string
    hash: string
}

/** True for a stored agent-format component (as opposed to an older `wgsl` envelope). */
export function isStoredComponent(value: unknown): value is StoredComponent {
    return !!value && typeof value === 'object' && (value as {kind?: unknown}).kind === 'source'
        && typeof (value as {source?: unknown}).source === 'string' && typeof (value as {ir?: unknown}).ir === 'object'
}

/** The fingerprint a stored component records for its source. */
export function sourceHash(source: string): string {
    return fingerprint(source)
}

/** Turn any stored custom component (agent-format source, or an older envelope) into a definition. */
export function defineStoredComponent(value: StoredComponent | ShaderEnvelope): GpuShaderDefinition {
    if (isStoredComponent(value)) return evaluateComponentIR(value.ir)
    return defineShaderEnvelope(value)
}
