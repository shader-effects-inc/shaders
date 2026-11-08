/**
 * The lowering of `displaceBy`: a grid wave field as a compute hook, and the gather fragment that
 * moves the child along its slope with a chromatic split. Lives with the word so it is bundled
 * only when the word is used.
 */
import type {ComponentProps} from '../types'
import type {Expr, GpuFragmentParams, GpuShaderDefinition, KitTexture} from '../gpu/contract'
import {call, expr, vec4, ZERO} from '../gpu/composer'
import {chromaticDisplaceUVs} from '../gpu/kit/displace'
import {createWaveFieldSim} from '../gpu/kit/waves'
import {applyEdgeHandlingExpr} from '../gpu/kit/edges'
import {unpremultiplyAlpha} from '../gpu/kit/blend'
import {isFilterIdentity, type FilterIdentity, type FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import {uniformOf} from './invoke'
import type {StdGatherFilterDefinition} from './types'
import type {DisplaceLoweringHelpers} from './displaceSymbol'

/**
 * The `displaceBy` gather effect — a sim-driven distortion. The simulation is discovered
 * from the effect's field reference; the lowering validates the declared op set, wires the
 * kit runtime harness as the compute hook, derives `usesPointer` from the declared
 * signals, and owns the gather fragment (per-tap edge handling, straight-alpha result).
 */
export function lowerDisplaceBy<T extends ComponentProps>(definition: StdGatherFilterDefinition<T>, helpers: DisplaceLoweringHelpers): GpuShaderDefinition<T> {
    const lowerProps = helpers.lowerProps as unknown as (p: unknown) => GpuShaderDefinition<T>['props']
    const lowerIdentity = helpers.lowerIdentity as unknown as (r: unknown, p: unknown) => FilterIdentity | undefined
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
    if (typeof sim.resolution !== 'number' || typeof wave.damping === 'number' || typeof splat.radius === 'number') {
        throw new Error('std: displaceBy needs a square grid (resolution: number) and prop-driven damping and radius; use the grid as an input for other setups')
    }
    if (sim.derive[outputKey]?.kind !== 'op.gradient') {
        throw new Error(`std: displaceBy consumes a vector field — derive '${outputKey}' must be op.gradient()`)
    }
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
            const runtime = createWaveFieldSim(params, {
                resolution: sim.resolution as number,
                dampingProp: (wave.damping as {name: string}).name,
                radiusProp: (splat.radius as {name: string}).name,
                radiusScale: 0.05, // UI radius (0.1–1) → field-space brush radius
                speedMax: splat.amount.max,
                teleportGuard: splat.at.teleportGuard,
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
                return call(unpremultiplyAlpha, 'unpremultiplyAlpha', [childTex.sample(ctx.uv)])
            }
            const edgeMode = (propValues[effect.edges.name] as number) ?? 0
            const dispTex = computeOutputs?.[outputKey] as KitTexture | undefined
            const disp = dispTex ? dispTex.sample(ctx.uv, 'linearClamp').member('xy') : expr('vec2f(0.0, 0.0)')

            const uvs = call(chromaticDisplaceUVs, 'chromaticDisplaceUVs', [
                ctx.uv, disp, uniformOf(effect.strength, params as FilterParams), uniformOf(effect.chromatic, params as FilterParams),
            ])
            const sampleChild = (uv: Expr) => childTex.sample(uv)
            const rSample = applyEdgeHandlingExpr(uvs.member('rUV'), sampleChild, edgeMode)
            const gSample = applyEdgeHandlingExpr(uvs.member('gUV'), sampleChild, edgeMode)
            const bSample = applyEdgeHandlingExpr(uvs.member('bUV'), sampleChild, edgeMode)
            const combined = vec4(rSample.member('r'), gSample.member('g'), bSample.member('b'), gSample.member('a'))
            return call(unpremultiplyAlpha, 'unpremultiplyAlpha', [combined])
        },
    }
}

