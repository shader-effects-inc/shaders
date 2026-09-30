/**
 * std — the lowering: StdDefinition → GpuShaderDefinition.
 *
 * Noun effects lower to kit primitives (the normative GPU bodies) wired through the
 * engine scaffolds; the L1 tiers pass their blessed bodies straight through, emitting the
 * same calls under the same hints so a port leaves the compiled WGSL untouched. Either
 * way the output is an ordinary `GpuShaderDefinition` — every downstream surface
 * (registry, generated components, editor metadata, presets) is unchanged.
 */
import type {ComponentProps, PropConfig} from '../types'
import type {Expr, GpuFragmentParams, GpuShaderDefinition, KitTexture} from '../gpu/contract'
import {call, expr, vec4, ZERO} from '../gpu/composer'
import {colorMixing, displace as displaceKit, waves, edges as edgesKit, blend} from '../gpu/kit/index'
import {definePointwiseFilter, isFilterIdentity, type FilterIdentity, type FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import {defineRttFilter} from '../gpu/scaffolds/rttFilter'
import {uvRemapShader} from '../gpu/scaffolds/uvRemapShader'
import {defineSdfShapeShader} from '../gpu/scaffolds/sdfShape'
import type {
    IdentityRule,
    StdCustomDefinition,
    StdDefinition,
    StdGatherFilterDefinition,
    StdGeneratorDefinition,
    StdPointwiseFilterDefinition,
    StdPropConfig,
    StdProps,
    StdShapeDefinition,
    StdWarpDefinition,
    StdWgslFilterDefinition,
} from './types'
import {resolveScalar, uniformOf} from './invoke'
import {isWgslBody, lowerWgsl} from './wgsl'

// ── Props ───────────────────────────────────────────────────────────────────────────────

/** `recompile:` rules → the equivalent `compileTimeWhen` predicate. */
function lowerProps<T extends ComponentProps>(props: StdProps<T>): GpuShaderDefinition<T>['props'] {
    const lowered: Record<string, PropConfig<unknown>> = {}
    for (const key of Object.keys(props) as (keyof T & string)[]) {
        const {recompile, ...rest} = props[key] as StdPropConfig<unknown>
        const config: PropConfig<unknown> = {...rest}
        if (recompile?.kind === 'crosses') {
            const boundary = recompile.value
            config.compileTimeWhen = (prev: unknown, next: unknown) =>
                (prev === boundary) !== (next === boundary)
        } else if (recompile?.kind === 'custom') {
            config.compileTimeWhen = recompile.predicate
        }
        lowered[key] = config
    }
    return lowered as GpuShaderDefinition<T>['props']
}

// ── Identity rules ──────────────────────────────────────────────────────────────────────

interface LoweredIdentity {
    props: string[]
    when: (values: Record<string, unknown>) => boolean
}

function lowerIdentityRule(
    rule: IdentityRule,
    defaults: Record<string, unknown>,
): LoweredIdentity {
    switch (rule.kind) {
        case 'isZero':
        case 'isValue': {
            const target = rule.kind === 'isZero' ? 0 : rule.value
            const declared = defaults[rule.prop]
            return {
                props: [rule.prop],
                when: (values) => (values[rule.prop] ?? declared) === target,
            }
        }
        case 'allOf': {
            const lowered = rule.rules.map((r) => lowerIdentityRule(r, defaults))
            return {
                props: [...new Set(lowered.flatMap((l) => l.props))],
                when: (values) => lowered.every((l) => l.when(values)),
            }
        }
        case 'custom':
            return {props: rule.props, when: rule.when}
    }
}

/**
 * `identityWhen:` → the scaffolds' `FilterIdentity`. The scaffold owns the map-driver
 * bypass refusal; unset values fall back to each prop's declared default.
 */
function lowerIdentity<T extends ComponentProps>(
    rule: IdentityRule | undefined,
    props: StdProps<T>,
): FilterIdentity | undefined {
    if (!rule) return undefined
    const defaults: Record<string, unknown> = {}
    for (const [key, config] of Object.entries(props as Record<string, StdPropConfig<unknown>>)) {
        defaults[key] = config.default
    }
    return lowerIdentityRule(rule, defaults)
}

// ── Species lowerings ───────────────────────────────────────────────────────────────────

function lowerPointwiseFilter<T extends ComponentProps>(definition: StdPointwiseFilterDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, effect, identityWhen, missingChildMessage, props, ...meta} = definition
    const shared = {
        ...meta,
        props: lowerProps(props),
        identity: lowerIdentity(identityWhen, props),
        missingChildMessage,
    }
    if (effect.kind === 'tintToward') {
        const {color, amount} = effect
        return definePointwiseFilter<T>({
            ...shared,
            body: {fn: colorMixing.mixToward, hint: 'mixToward'},
            args: (params) => [uniformOf(color, params), resolveScalar(amount, params)],
        })
    }
    if (isWgslBody(effect)) {
        return definePointwiseFilter<T>({
            ...shared,
            build: lowerWgsl(effect, 'pointwise', wgslHost(definition)),
        })
    }
    const {kind: _kind, ...effectConfig} = effect
    return definePointwiseFilter<T>({...shared, ...effectConfig})
}

/** What a `wgsl` body needs from its definition to bind props and the time clock. */
function wgslHost<T extends ComponentProps>(definition: {name: string; props: StdProps<T>; animatedTime?: {speed: string}}) {
    return {
        name: definition.name,
        props: definition.props as unknown as Record<string, PropConfig<unknown>>,
        animatedTime: definition.animatedTime,
    }
}

/**
 * A filter authored with a `wgsl` body and no species: `gather` when the body samples the
 * child texture, `pointwise` otherwise. Returns the definition with the species filled in.
 */
function inferWgslFilterSpecies<T extends ComponentProps>(
    definition: StdWgslFilterDefinition<T>,
): StdPointwiseFilterDefinition<T> | StdGatherFilterDefinition<T> {
    const species = definition.effect.samplesChild ? 'gather' : 'pointwise'
    return {...definition, species} as StdPointwiseFilterDefinition<T> | StdGatherFilterDefinition<T>
}

function lowerGatherFilter<T extends ComponentProps>(definition: StdGatherFilterDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, effect, identityWhen, missingChildMessage, props, ...meta} = definition
    if (effect.kind === 'displaceBy') return lowerDisplaceBy(definition)
    if (isWgslBody(effect)) {
        return defineRttFilter<T>({
            ...meta,
            props: lowerProps(props),
            identity: lowerIdentity(identityWhen, props),
            missingChildMessage,
            build: lowerWgsl(effect, 'gather', wgslHost(definition)),
            resultAlpha: effect.spec.alpha ?? 'premultiplied',
        })
    }
    const {kind: _kind, ...effectConfig} = effect
    return defineRttFilter<T>({
        ...meta,
        props: lowerProps(props),
        identity: lowerIdentity(identityWhen, props),
        missingChildMessage,
        ...effectConfig,
    })
}

/**
 * The `displaceBy` gather effect — a sim-driven distortion. The simulation is discovered
 * from the effect's field reference; the lowering validates the declared op set, wires the
 * kit runtime harness as the compute hook, derives `usesPointer` from the declared
 * signals, and owns the gather fragment (per-tap edge handling, straight-alpha result).
 */
function lowerDisplaceBy<T extends ComponentProps>(definition: StdGatherFilterDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, effect: effectUnion, identityWhen, missingChildMessage: _mcm, props, ...meta} = definition
    const effect = effectUnion as import('./filter').DisplaceByEffect
    const identity = lowerIdentity(identityWhen, props)
    const sim = effect.field.sim.config
    const outputKey = effect.field.output

    // Validate the declared simulation against the op sets the lowering implements — an
    // unrecognized configuration throws at definition time, never silently degrades.
    const wave = sim.step.find((s) => s.kind === 'op.wave')
    const splat = sim.step.find((s) => s.kind === 'op.splat')
    if (!wave || !splat || sim.step.length !== 2) {
        throw new Error(`std: simulate.grid currently implements exactly [op.wave, op.splat] (got: ${sim.step.map((s) => s.kind).join(', ')})`)
    }
    if (sim.history !== 2) throw new Error(`std: op.wave reads t−1 and t−2 — declare history: 2 (got ${sim.history})`)
    if (sim.derive[outputKey]?.kind !== 'op.gradient') {
        throw new Error(`std: displaceBy consumes a vector field — derive '${outputKey}' must be op.gradient()`)
    }
    if (splat.at.teleportGuard !== 'on') throw new Error(`std: op.splat currently implements teleportGuard: 'on' only`)
    if (sim.rest && sim.rest.settlesWhen !== 'derived-from-damping') {
        throw new Error(`std: simulate.grid rest supports settlesWhen: 'derived-from-damping' only`)
    }

    return {
        ...meta,
        props: lowerProps(props),
        requiresRTT: true,
        requiresChild: true,
        usesPointer: splat.at.kind === 'pointer',

        compute: (params: GpuFragmentParams) => {
            if (!params.childNode) return null
            const runtime = waves.createWaveFieldSim(params, {
                resolution: sim.resolution,
                dampingProp: wave.damping.name,
                radiusProp: splat.radius.name,
                radiusScale: 0.05, // UI radius (0.1–1) → field-space brush radius
                speedMax: splat.amount.max,
            })
            if (!runtime) return null // GPU-free composition: the fragment falls back to zero displacement.
            return {outputs: {[outputKey]: runtime.displacement}, getComputeNodes: runtime.getComputeNodes}
        },

        fragment: (params: GpuFragmentParams): Expr => {
            const {childNode, ctx, computeOutputs, propValues, convertToTexture} = params
            if (!childNode) return ZERO
            const childTex = convertToTexture(childNode)
            // Identity still costs the RTT pass (mirrors defineRttFilter) — sample the centre
            // texel straight-alpha rather than returning the child.
            if (isFilterIdentity(identity, params)) {
                return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [childTex.sample(ctx.uv)])
            }
            const edgeMode = (propValues[effect.edges.name] as number) ?? 0
            const dispTex = computeOutputs?.[outputKey] as KitTexture | undefined
            const disp = dispTex ? dispTex.sample(ctx.uv, 'linearClamp').member('xy') : expr('vec2f(0.0, 0.0)')

            const uvs = call(displaceKit.chromaticDisplaceUVs, 'chromaticDisplaceUVs', [
                ctx.uv, disp, uniformOf(effect.strength, params as FilterParams), uniformOf(effect.chromatic, params as FilterParams),
            ])
            const sampleChild = (uv: Expr) => childTex.sample(uv)
            const rSample = edgesKit.applyEdgeHandlingExpr(uvs.member('rUV'), sampleChild, edgeMode)
            const gSample = edgesKit.applyEdgeHandlingExpr(uvs.member('gUV'), sampleChild, edgeMode)
            const bSample = edgesKit.applyEdgeHandlingExpr(uvs.member('bUV'), sampleChild, edgeMode)
            const combined = vec4(rSample.member('r'), gSample.member('g'), bSample.member('b'), gSample.member('a'))
            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [combined])
        },
    }
}

function lowerWarp<T extends ComponentProps>(definition: StdWarpDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, map, edges, resample, uvRemapIdentityWhen, missingChildMessage, props, ...meta} = definition
    return {
        ...meta,
        props: lowerProps(props),
        requiresRTT: true,
        requiresChild: true,
        ...uvRemapShader(map, {
            edges,
            resample,
            uvRemapIdentityWhen,
            requireChildMessage: missingChildMessage,
        }),
    } as GpuShaderDefinition<T>
}

function lowerShape<T extends ComponentProps>(definition: StdShapeDefinition): GpuShaderDefinition<T> {
    const {role: _role, name, description, category, shape} = definition
    return defineSdfShapeShader<T>({name, description, category, ...shape})
}

function lowerGenerator<T extends ComponentProps>(definition: StdGeneratorDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, paint, props, ...meta} = definition
    return {
        ...meta,
        props: lowerProps(props),
        fragment: isWgslBody(paint) ? lowerWgsl(paint, 'generator', wgslHost(definition)) : paint,
    }
}

function lowerCustom<T extends ComponentProps>(definition: StdCustomDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, gpu, props, ...meta} = definition
    return {
        ...meta,
        props: lowerProps(props),
        fragment: gpu.fragment,
        ...(gpu.uvRemap ? {uvRemap: gpu.uvRemap} : {}),
    }
}

// ── Entry ───────────────────────────────────────────────────────────────────────────────

/**
 * A small stable fingerprint (FNV-1a) of the `wgsl` bodies a definition carries, so a
 * live-edited body under an unchanged name still recomposes (see `GpuShaderDefinition.revision`).
 */
function wgslRevision(definition: StdDefinition<unknown & ComponentProps>): string | undefined {
    const bodies: string[] = []
    const paint = (definition as {paint?: unknown}).paint
    const effect = (definition as {effect?: unknown}).effect
    for (const candidate of [paint, effect]) {
        if (isWgslBody(candidate)) bodies.push(candidate.spec.body, Object.keys(candidate.spec.inputs ?? {}).join(','), candidate.spec.alpha ?? '')
    }
    if (bodies.length === 0) return undefined
    let h = 0x811c9dc5
    for (const ch of bodies.join('\u0000')) {
        h ^= ch.charCodeAt(0)
        h = Math.imul(h, 0x01000193) >>> 0
    }
    return h.toString(16)
}

/** Lower a std definition to the engine contract. */
export function defineStd<T extends ComponentProps>(definition: StdDefinition<T>): GpuShaderDefinition<T> {
    const lowered = lowerStd(definition)
    const revision = wgslRevision(definition as StdDefinition<ComponentProps>)
    return revision ? {...lowered, revision} : lowered
}

function lowerStd<T extends ComponentProps>(definition: StdDefinition<T>): GpuShaderDefinition<T> {
    if (definition.role === 'shape') return lowerShape(definition)
    if (definition.role === 'warp') return lowerWarp(definition)
    if (definition.role === 'generator' && !('species' in definition)) return lowerGenerator(definition)
    if (definition.role === 'filter' && !('species' in definition && definition.species) && isWgslBody((definition as StdWgslFilterDefinition<T>).effect)) {
        return lowerStd(inferWgslFilterSpecies(definition as StdWgslFilterDefinition<T>))
    }
    if ('species' in definition) {
        if (definition.species === 'pointwise') return lowerPointwiseFilter(definition)
        if (definition.species === 'gather') return lowerGatherFilter(definition)
        if (definition.species === 'custom') return lowerCustom(definition)
    }
    const {role} = definition as {role: string}
    throw new Error(`std: unsupported definition shape for role '${role}'`)
}

/**
 * Define a shader component. The public name of {@link defineStd}: a declarative definition
 * (props, role, the paint or effect) lowered to the engine contract, ready for
 * `<CustomShader src={…}>` in any framework, `registerShader`, or a `createShader` preset.
 */
export const defineShader: typeof defineStd = defineStd
