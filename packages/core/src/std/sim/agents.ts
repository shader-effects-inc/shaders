/**
 * std/sim/agents — thousands of small things that move: flocks, motes, filings, swarms.
 *
 * An agent simulation keeps a position (and whatever else the physics needs) per agent, moves
 * every agent one step per frame, draws each one into a shared picture, and hands that picture
 * to the layer. You declare the physics as a list of `force` or `torque` parts folded by an
 * `integrator`, the look as one `renderAgents` variant, and the once-per-frame values with
 * `agentFrame` helpers. `agentSim` owns the plumbing: the state buffers, the output texture,
 * the first-frame spawn and re-spawn on a seed change, the mobile count cap, the child's
 * picture when one is needed, and the fragment that samples the result.
 *
 * Read from the inside out: forces → integrator → render → `agentSim`. This module re-exports
 * the whole agent vocabulary so one import covers it.
 */
// Maintainer notes. The declarative spread a `species: 'custom'` agent shader composes its
// compute/fragment halves from (the gaussianBlur/ComputeBackedEffect precedent): the shader
// file keeps its layout, uniform struct and constants visible, declares its physics as
// `force`/`torque` parts folded by an `integrator` factory, its render as a `renderAgents`
// variant, and its per-frame CPU math as named `agentFrame` recipes — and spreads
//
//     ...agentSim({layout, params, maxAgents, output, bake, frame, fragment})
//
// into the definition. The noun owns the plumbing: device guards, `createAgentSystem` wiring
// (buffers, output texture, guarded pipelines, init latch), the late child binding, and the
// standard full-canvas bilinear fragment with its GPU-free fallback.
import type {Expr, GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import type {d} from '../../gpu/kit/index'

type AnyWgslStruct = d.AnyWgslStruct
import {ZERO} from '../../gpu/composer'
import {
    createAgentSystem,
    type AgentPipelineSpec, type AgentSystem, type AgentSystemConfig, type DeviceTier,
} from '../../gpu/scaffolds/agentSystem'
import type {ComputeStep} from '../../gpu/compute'
import {
    splat, resolve,
    type OrientedWorldSplatConfig, type PointWorldSplatConfig, type VolumeSplatConfig,
    type ReliefSplatConfig, type RampResolveConfig,
    type WorldSplatLayout, type PointWorldSplatLayout, type RampResolveLayout,
} from './agentRender'

// ── renderAgents ──────────────────────────────────────────────────────────────────────────

/** The two draw steps every agent frame ends with: `splat` draws the agents, `resolve` turns the drawing into the output picture. */
export interface RenderAgentsPipelines {
    splat: AgentPipelineSpec
    resolve: AgentPipelineSpec
}

/**
 * How the agents are drawn, as one declared part: a splat (each agent stamps its shape into a
 * shared canvas) paired with the matching resolve (the canvas becomes the output picture).
 *
 * Four variants, one per kind of agent: `orientedWorld` for agents with a heading (arrows,
 * streaks, comets) colored from rest to excited, `pointWorld` for round motes in one tint,
 * `volume` for a 3D swarm seen in perspective, `relief` for particles that stand on an image.
 * Put the returned `splat` and `resolve` into `bake`'s pipelines and end the program with them.
 *
 * @example
 * ```ts
 * const render = renderAgents.orientedWorld(simLayout, {shape: 'arrow', res: 1024, splatRCap: 16, heading: 'velocity', agitation: 'buffer', comet: true, ramp: {colorSpace: 2, trails: 'on'}, names: {splat: 'boidsSplat', resolve: 'boidsResolve'}})
 * ```
 * @tip The resolve also clears the canvas as it reads it, so no clear step is needed.
 * @see agentSim, splat, resolve, pointSlot
 */
// Kernel factories live in `agentRender`; this surface returns ready pipeline specs.
export const renderAgents = {
    /**
     * Agents with a heading, drawn as a shape pointing the way they move (or the way they are
     * turned), colored from a rest color to an excited color.
     *
     * `heading: 'velocity'` reads the direction from the agent's velocity, `'angle'` from a
     * stored angle plus a `home` part. `agitation: 'buffer'` reads excitement from a per-agent
     * value, `'speed'` from the agent's speed. `comet: true` stretches a glow along the heading.
     * `res` is the canvas side in texels.
     *
     * @example
     * ```ts
     * renderAgents.orientedWorld(simLayout, {shape, res: 1024, splatRCap: 16, heading: 'velocity', agitation: 'buffer', comet: true, ramp: {colorSpace: 2, trails: 'on'}, names: {splat: 'boidsSplat', resolve: 'boidsResolve'}})
     * ```
     * @see pointWorld, volume
     */
    // Boids, MagneticFilings, ParticleFlow.
    orientedWorld(
        layout: WorldSplatLayout & RampResolveLayout,
        cfg: Omit<OrientedWorldSplatConfig, 'name'> & {
            ramp: Omit<RampResolveConfig, 'name' | 'res' | 'invGain'> & {invGain?: number}
            names: {splat: string; resolve: string}
            /** Give the splat the second resource group as well (a shape family's own resources). */
            extraOnSplat?: boolean
        },
    ): RenderAgentsPipelines {
        const {ramp, names, extraOnSplat, ...splatCfg} = cfg
        return {
            splat: {
                kernel: splat.orientedWorld(layout, {...splatCfg, name: names.splat}),
                threads: 'agents',
                ...(extraOnSplat ? {extra: true} : {}),
            },
            resolve: {
                kernel: resolve.ramp(layout, {
                    invGain: 1 / 255,
                    ...ramp,
                    res: cfg.res,
                    name: names.resolve,
                }),
                threads: 'fixed',
                size: [cfg.res, cfg.res],
            },
        }
    },

    /**
     * Round motes with no heading, drawn in one tint with a live softness. Where each mote is
     * placed, how bright it is and how big it is come from `pointSlot` parts.
     *
     * @example
     * ```ts
     * renderAgents.pointWorld(simLayout, {shape, res: 1024, place: pointSlot.orbitalPlace(simLayout, {radius: 0.035, rate: 1.4}), brightness: pointSlot.twinkleBrightness(simLayout, {freq: 2}), bodyRadius: pointSlot.presenceRadius(simLayout), names: {splat: 'motesSplat', resolve: 'motesResolve'}})
     * ```
     * @see orientedWorld, pointSlot
     */
    // FloatingParticles.
    pointWorld(
        layout: PointWorldSplatLayout & Parameters<typeof resolve.tint>[0],
        cfg: Omit<PointWorldSplatConfig, 'name'> & {names: {splat: string; resolve: string}},
    ): RenderAgentsPipelines {
        const {names, ...splatCfg} = cfg
        return {
            splat: {kernel: splat.pointWorld(layout, {...splatCfg, name: names.splat}), threads: 'agents'},
            resolve: {
                kernel: resolve.tint(layout, {res: cfg.res, name: names.resolve}),
                threads: 'fixed',
                size: [cfg.res, cfg.res],
            },
        }
    },

    /**
     * A 3D swarm seen in perspective: nearer particles draw larger and brighter, and the color
     * runs from rest to excited with speed.
     *
     * @example
     * ```ts
     * renderAgents.volume(simLayout, {shape, outRes: 1024, maxSplatSize: 12, extQ: 6, ramp: {colorSpace: 0, exposure: true, trails: 'none'}, names: {splat: 'swarmSplat', glow: 'swarmGlow', profile: 'swarmProfile', resolve: 'swarmResolve'}, extraOnSplat: true})
     * ```
     * @see orientedWorld, relief
     */
    // Particles.
    volume(
        layout: Parameters<typeof splat.volume>[0] & RampResolveLayout,
        cfg: Omit<VolumeSplatConfig, 'names'> & {
            ramp: Omit<RampResolveConfig, 'name' | 'res' | 'invGain'>
            names: VolumeSplatConfig['names'] & {resolve: string}
            extraOnSplat?: boolean
        },
    ): RenderAgentsPipelines {
        const {ramp, names, extraOnSplat, ...splatCfg} = cfg
        return {
            splat: {
                kernel: splat.volume(layout, {...splatCfg, names}),
                threads: 'agents',
                ...(extraOnSplat ? {extra: true} : {}),
            },
            resolve: {
                kernel: resolve.ramp(layout, {invGain: 1 / 256, ...ramp, res: cfg.outRes, name: names.resolve}),
                threads: 'fixed',
                size: [cfg.outRes, cfg.outRes],
            },
        }
    },

    /**
     * Particles standing on an image, seen through a camera: each carries the child's color at
     * its home, and nearer particles win the color where they overlap. `out` is the output size
     * in texels.
     *
     * @example
     * ```ts
     * renderAgents.relief(fieldLayout, {shape, splatRCap: 24, aaW: 0.75, minAlpha: 0.02, fp: 1024, wobbleFreq: 1.3, perspK: 0.9, out: [outW, outH], weightedColor: {alphaK: 1.6}, names: {splat: 'reliefSplat', resolve: 'reliefResolve'}})
     * ```
     * @see volume
     */
    // ParticleField.
    relief(
        layout: Parameters<typeof splat.relief>[0],
        cfg: Omit<ReliefSplatConfig, 'name'> & {
            out: [number, number]
            weightedColor: {alphaK: number}
            names: {splat: string; resolve: string}
        },
    ): RenderAgentsPipelines {
        const {out, weightedColor, names, ...splatCfg} = cfg
        return {
            splat: {kernel: splat.relief(layout, {...splatCfg, name: names.splat}), threads: 'grid'},
            resolve: {
                kernel: resolve.weightedColor(layout, {fp: cfg.fp, alphaK: weightedColor.alphaK, name: names.resolve}),
                threads: 'fixed',
                size: out,
            },
        }
    },
} as const

// ── agentSim ──────────────────────────────────────────────────────────────────────────────

/** What `bake` returns: the steps, their order, and anything the frame needs from composition. */
export interface AgentSimBaked {
    /** Every step by name: the spawn, the physics, the splat, the resolve. */
    pipelines: Record<string, AgentPipelineSpec>
    /** The steps that run each frame, in order (keys of `pipelines`). */
    program: readonly string[]
    /** The spawn step, run once before the first frame and again when `reseed` changes. */
    initStep?: string
    /** Override the output size decided at composition (an aspect-fitted target). */
    outputSize?: [number, number]
    /** A second resource group for every step marked `extra` (a shape's own distance field). */
    extraBindGroup?: unknown
    /** Resources ready now for the keys in `externalKeys` (a solver's texture created in `bake`). */
    bindNow?: Record<string, unknown>
    /** Anything `frame` needs from composition: a solver, a per-shape resolver. */
    setup?: unknown
}

/** How the layer samples the simulation's picture. */
export interface AgentSimFragmentConfig {
    /** The output the fragment samples over the full canvas. */
    // rgba16f is filterable — one bilinear tap.
    output: string
    /** What to show without a GPU: nothing, or the child untouched. */
    fallback: 'transparent' | 'child'
    /** Combine the sampled picture with something else, such as laying it over the child. */
    compose?: (field: Expr, params: GpuFragmentParams) => Expr
}

/** What an agent simulation declares: its state layout, its values, its count, its output, and the bake, frame and fragment halves. */
export interface AgentSimConfig<TParams extends AnyWgslStruct> {
    /** The layout naming the per-agent buffers, the accumulators, the values and the output texture. */
    layout: AgentSystemConfig<TParams>['layout']
    /** The schema of the per-frame values the steps read. */
    params: TParams
    /** How many agents the buffers hold. The count prop can be anything up to it. */
    maxAgents: number
    /** A lower effective count on phones and tablets. The prop's range is unchanged. */
    countCap?: DeviceTier
    /** The output texture: its layout key, the name the fragment reads it under, its size and format. */
    output: AgentSystemConfig<TParams>['output']
    /** Layout keys filled later, through `childTexture` or `bake.bindNow`. */
    externalKeys?: readonly string[]
    /** Read the child as a texture and bind it to this layout key once it exists. */
    // ParticleField.
    childTexture?: {externalKey: string}
    /** Assemble the steps once per composition, from compile-time props. Return null to run nothing. */
    bake: (params: GpuFragmentParams) => AgentSimBaked | null
    /** Build the once-per-frame program: write the values, then return `sys.frame(...)`. */
    // Usually via `sys.frame(...)`, prepending any extra passes the shader owns.
    frame: (
        sys: AgentSystem<TParams>,
        params: GpuFragmentParams,
        setup: unknown,
    ) => (frameParams: unknown) => ComputeStep[] | null
    fragment: AgentSimFragmentConfig
}

/** The two halves `agentSim` adds to a definition: the simulation and the fragment that samples it. */
// The ComputeBackedEffect shape.
export interface AgentSimSpread {
    compute: GpuComputeNode
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/**
 * A simulation of many agents, spread into a definition: `bake` assembles the steps from your
 * declared parts, `frame` writes the per-frame values, and the fragment samples the picture.
 *
 * `bake` runs once per composition with the compile-time props (shape, color space) and
 * returns the pipelines and their order. `frame` runs once per frame: read props, write
 * `sys.writeParams(...)`, return `sys.frame({count})`. The spawn step runs before the first
 * frame and again whenever `reseed` changes. Without a GPU the fragment shows the fallback.
 *
 * @example
 * ```ts
 * ...agentSim<typeof SimParams>({layout: simLayout, params: SimParams, maxAgents: 4096, countCap: {desktop: 4096, mobile: 1200}, output: {key: 'outTex', name: 'boidsTexture', size: [1024, 1024], format: 'rgba16float'}, bake: ({getCpuValue}) => { const render = boidsRender(getCpuValue('agentShape') as string); return {pipelines: {init: {kernel: boidsInitKernel, threads: 'max'}, update: {kernel: boidsUpdateKernel, threads: 'agents'}, splat: render.splat, resolve: render.resolve}, initStep: 'init', program: ['update', 'splat', 'resolve']} }, frame: (sys, {getCpuValue}) => (frameParams) => { const {dt, aspect} = readAgentFrame(frameParams); const count = sys.resolveCount(getCpuValue('count') as number); sys.writeParams({...}); return sys.frame({count}) }, fragment: {output: 'boidsTexture', fallback: 'transparent'}})
 * ```
 * @tip Clamp the count prop with `sys.resolveCount` rather than reading it raw, so a desktop preset still loads on a phone.
 * @see renderAgents, integrator, force, agentFrame, simulate
 */
export function agentSim<TParams extends AnyWgslStruct>(cfg: AgentSimConfig<TParams>): AgentSimSpread {
    return {
        compute: (params: GpuFragmentParams) => {
            if (cfg.childTexture && !params.childNode) return null
            if (!params.gpu?.root) return null // GPU-free (no device): fragment falls back.

            const childTexture = cfg.childTexture ? params.convertToTexture(params.childNode!) : null

            const baked = cfg.bake(params)
            if (!baked) return null

            const sys = createAgentSystem(params, {
                layout: cfg.layout,
                params: cfg.params,
                maxAgents: cfg.maxAgents,
                output: baked.outputSize ? {...cfg.output, size: baked.outputSize} : cfg.output,
                ...(cfg.countCap ? {countCap: cfg.countCap} : {}),
                ...(cfg.externalKeys ? {externalKeys: cfg.externalKeys} : {}),
                ...(baked.extraBindGroup ? {extraBindGroup: baked.extraBindGroup} : {}),
                pipelines: baked.pipelines,
                program: baked.program,
                ...(baked.initStep ? {initStep: baked.initStep} : {}),
            })
            if (!sys) return null
            if (baked.bindNow) sys.bindExternal(baked.bindNow)

            const tick = cfg.frame(sys, params, baked.setup)

            return {
                outputs: childTexture ? {childTexture, ...sys.outputs} : sys.outputs,
                ...(childTexture && cfg.childTexture
                    ? {
                        bindInputs: (resolveInput: (key: string) => {texture: unknown} | undefined) => {
                            const src = resolveInput(childTexture.key)
                            if (!src) return
                            sys.bindExternal({[cfg.childTexture!.externalKey]: src.texture})
                        },
                    }
                    : {}),
                getComputeNodes: tick,
            }
        },
        gpu: {
            fragment: (params: GpuFragmentParams): Expr => {
                const {ctx, childNode, computeOutputs} = params
                const field = computeOutputs?.[cfg.fragment.output] as KitTexture | undefined
                if (!field) return cfg.fragment.fallback === 'child' ? (childNode ?? ZERO) : ZERO
                const sampled = field.sample(ctx.uv, 'linearClamp')
                return cfg.fragment.compose ? cfg.fragment.compose(sampled, params) : sampled
            },
        },
    }
}

// ── Re-exports: the full agent-simulation vocabulary from one import surface ──────────────

export {force, torque, field, drift, channel, integrator, composeForces3, composeSteer2, composeTorques, restOrientationAngle} from './agentForces'
export type {
    AgentForce3, AgentSteer2, AgentTorque, AgentFieldAt, AgentHome2, AgentDrift, AgentChannel, ShapeField3,
} from './agentForces'
export {splat, resolve, pointSlot} from './agentRender'
export {shapeField} from './shapeFields'
export * as agentFrame from './agentFrame'
