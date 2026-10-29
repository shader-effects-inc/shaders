/**
 * std/graph — the `wgsl` envelope: a raw `wgsl` body plus a props manifest, as data.
 *
 *   validateWgslEnvelope(envelope)   static checks — never throws
 *   defineWgslEnvelope(envelope)     → an ordinary GpuShaderDefinition
 *
 * The body is the same one `defineShader({paint: wgsl`…`})` takes, so the binding rules are
 * the `wgsl` escape hatch's (see `../wgsl.ts`). What the document cannot say, the shape of
 * the body decides: reading `child` makes a per-pixel filter, sampling `childTexture` a filter
 * with its own render pass, neither a generator; reading `time` adds a `speed` prop and the
 * layer's own clock.
 */
import type {GpuShaderDefinition} from '../../gpu/contract'
import type {ComponentProps} from '../../types'
import {defineStd} from '../lower'
import {scanIdentifiers, wgsl} from '../wgsl'
import {graphPropConfigs, validateGraphProps} from './define'
import {StdGraphError, type WgslEnvelope} from './types'

const MAX_BODY = 12000

/** Names the body owns; a prop spelled the same would never be bound. */
const BODY_NAMES = new Set(['uv', 'time', 'aspect', 'viewport', 'pointer', 'child', 'childTexture', 'childSampler', 'viewportSize', 'logicalViewportSize'])

export interface WgslEnvelopeValidation {
    ok: boolean
    error?: string
    /** `generator` paints from coordinates; `filter` reads the layer inside it. */
    role?: 'generator' | 'filter'
    /** The body reads `time`, so the definition carries `speed` + the layer clock. */
    usesTime?: boolean
}

/** Static check of a `wgsl` envelope. Never throws. */
export function validateWgslEnvelope(envelope: unknown): WgslEnvelopeValidation {
    try {
        const env = envelope as WgslEnvelope
        if (!env || typeof env !== 'object') throw new StdGraphError('envelope is not an object')
        if (env.kind !== 'wgsl') throw new StdGraphError('kind must be "wgsl"')
        if (typeof env.name !== 'string' || !env.name.trim()) throw new StdGraphError('name is required')
        if (typeof env.body !== 'string' || !env.body.trim()) throw new StdGraphError('body is required: WGSL statements that return a vec4f colour')
        if (env.body.length > MAX_BODY) throw new StdGraphError(`body is ${env.body.length} chars (max ${MAX_BODY})`)
        if (!/\breturn\b/.test(env.body)) throw new StdGraphError('body must `return` a vec4f colour')
        const props = env.props ?? {}
        validateGraphProps(props)
        for (const name of Object.keys(props)) {
            if (BODY_NAMES.has(name)) throw new StdGraphError(`prop "${name}" collides with a name the body is given — rename it`)
        }
        const identifiers = scanIdentifiers(env.body)
        const role = identifiers.has('child') || identifiers.has('childTexture') ? 'filter' : 'generator'
        return {ok: true, role, usesTime: identifiers.has('time')}
    } catch (err) {
        return {ok: false, error: err instanceof Error ? err.message : String(err)}
    }
}

/** Lower a `wgsl` envelope to an ordinary definition. Throws StdGraphError on an invalid envelope. */
export function defineWgslEnvelope(envelope: WgslEnvelope): GpuShaderDefinition {
    const check = validateWgslEnvelope(envelope)
    if (!check.ok) throw new StdGraphError(check.error!)
    const usesTime = !!check.usesTime
    const base = {
        name: envelope.name,
        category: 'Custom',
        description: envelope.description ?? envelope.name,
        ...(usesTime ? {animatedTime: {speed: 'speed'}} : {}),
        props: graphPropConfigs(envelope.props ?? {}, {usesTime, colorSpace: false}) as never,
    }
    const body = wgsl(envelope.body)
    return check.role === 'filter'
        ? defineStd<ComponentProps>({...base, effect: body})
        : defineStd<ComponentProps>({
            ...base,
            boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
            acceptsUVContext: true,
            paint: body,
        })
}
