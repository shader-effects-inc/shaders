/**
 * std/graph — shader definitions written as data (see `types.ts`): validate them, lower them
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
