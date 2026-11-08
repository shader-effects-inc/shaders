/**
 * The input graph: a component's `inputs`, ordered, built on the GPU, and merged into the one
 * compute hook the engine runs before the component's body. Also the names the body reads.
 */
// Maintainer notes. `planInputs` is pure (runs at definition time): it splits `inputs` into input
// nodes and library functions, finds inputs nested inside others (`blur(downsample(child))`),
// orders them deps-first, and lists the names a body may call (`soft`, `field.slope`). The
// compute hook builds every input once per composition and returns their textures as compute
// outputs keyed `<input>:<output>`, which the body lowering binds (std/wgsl.ts).
import type {GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'
import type {PropConfig} from '../../types'
import {authorError} from '../../gpu/support'
import {isWgslLibFn, type WgslLibFn} from '../wgslLibrary'
import {identifiersIn} from '../wgslSource'
import {imageTexture, layerTexture} from '../wgslResources'
import {frameTracker} from './frame'
import {premultipliedKeys} from './self'
import {
    BUILD_INPUT,
    isChildSource,
    isInputNode,
    isPropRef,
    type InputBuildContext,
    type InputInstance,
    type InputNode,
    type InputSource,
    type InputTexture,
} from './types'

/** One input in build order. */
export interface PlannedInput {
    readonly name: string
    readonly node: InputNode
    /** True when it was not listed in `inputs` but nested inside an input that was. */
    readonly nested: boolean
}

/** A name a body can call, and the compute output it reads. */
export interface InputRead {
    /** What the body writes: `soft` or `field.slope`. */
    readonly call: string
    /** The compute output key: `soft:` or `field:slope`. */
    readonly key: string
    /** `uv` reads sample at a position; `index` reads slot i (`clicks(i)`). */
    readonly by: 'uv' | 'index'
    /** 32-bit texture: read exactly by textureLoad, blended by hand. */
    readonly unfilterable?: boolean
}

export interface InputPlan {
    readonly inputs: readonly PlannedInput[]
    readonly lib: readonly WgslLibFn[]
    readonly reads: readonly InputRead[]
    readonly readsChild: boolean
    /** Names listed in `inputs` that are neither inputs nor library functions. */
    readonly invalid: readonly string[]
}

/** Output key for an input's output (`''` is the default). */
export function outputKey(input: string, output = ''): string {
    return `${input}:${output}`
}

/** Every string inside a node's data (its WGSL), for finding the inputs it calls by name. */
/** All the WGSL text an input node holds (its bodies and kernels), as real multi-line text. */
export function codeIn(node: InputNode): string {
    const parts: string[] = []
    const visit = (value: unknown, depth: number) => {
        if (depth > 6) return
        if (typeof value === 'string') parts.push(value)
        else if (value && typeof value === 'object' && !isInputNode(value)) {
            const spec = (value as {spec?: {body?: unknown}}).spec
            if (spec && typeof spec.body === 'string') parts.push(spec.body)
            for (const v of Array.isArray(value) ? value : Object.values(value)) visit(v, depth + 1)
        }
    }
    for (const v of Object.values(node)) visit(v, 0)
    return parts.join('\n')
}

function nestedNodes(node: InputNode): InputNode[] {
    const out: InputNode[] = []
    const visit = (value: unknown, depth: number) => {
        if (depth > 6 || value === null || typeof value !== 'object') return
        if (value !== node && isInputNode(value)) {
            out.push(value)
            return
        }
        if (Array.isArray(value)) {
            for (const v of value) visit(v, depth + 1)
            return
        }
        for (const v of Object.values(value)) visit(v, depth + 1)
    }
    for (const v of Object.values(node)) visit(v, 0)
    return out
}

/** Split and order a definition's `inputs`. Throws for a dependency loop. */
export function planInputs(inputs: Readonly<Record<string, unknown>> | undefined): InputPlan {
    const lib: WgslLibFn[] = []
    const invalid: string[] = []
    const listed = new Map<InputNode, string>()
    for (const [name, value] of Object.entries(inputs ?? {})) {
        if (isWgslLibFn(value)) {
            if (value.name !== name) {
                throw new Error(`inputs: '${name}' is the library function ${value.name}; list it under its own name ({${value.name}})`)
            }
            lib.push(value)
        } else if (isInputNode(value)) {
            if (listed.has(value)) throw new Error(`inputs: '${name}' and '${listed.get(value)}' are the same input; list it once`)
            listed.set(value, name)
        } else {
            invalid.push(name)
        }
    }
    const ordered: PlannedInput[] = []
    const placed = new Map<InputNode, string>()
    let anon = 0
    const byName = new Map([...listed].map(([node, name]) => [name, node]))
    const place = (node: InputNode, stack: Set<InputNode>) => {
        if (placed.has(node)) return
        if (stack.has(node)) throw new Error('inputs: an input reads itself through another input (a loop)')
        stack.add(node)
        for (const dep of nestedNodes(node)) place(dep, stack)
        // Inputs its WGSL calls by name (`soft(p)` inside a pass) are built first too.
        const ids = identifiersIn(codeIn(node))
        for (const [name, dep] of byName) if (dep !== node && ids.has(name)) place(dep, stack)
        stack.delete(node)
        const name = listed.get(node) ?? `_in${anon++}`
        placed.set(node, name)
        ordered.push({name, node, nested: !listed.has(node)})
    }
    for (const node of listed.keys()) place(node, new Set())
    const reads: InputRead[] = []
    for (const p of ordered) {
        if (p.nested) continue
        const by = p.node.readBy ?? 'uv'
        const unfilterable = !!p.node.unfilterable
        reads.push({call: p.name, key: outputKey(p.name), by, unfilterable})
        for (const out of p.node.outputNames ?? []) reads.push({call: `${p.name}.${out}`, key: outputKey(p.name, out), by, unfilterable})
    }
    return {
        inputs: ordered,
        lib,
        reads,
        readsChild: ordered.some((p) => p.node.readsChild || nodeReadsChild(p.node)),
        invalid,
    }
}

function nodeReadsChild(node: InputNode): boolean {
    return Object.values(node).some((v) => isChildSource(v))
}

// ── Building ────────────────────────────────────────────────────────────────────────────

interface LateTexture {
    key: string
    resolved: unknown
}

/**
 * The compute hook for a component's inputs, merged with the hook the component already has
 * (a shape material's field bake), whose steps run first.
 */
export function inputsCompute(
    plan: InputPlan,
    component: {
        name: string
        props: Readonly<Record<string, PropConfig<unknown>>>
        animated: boolean
        helpers: string | null
    },
    base?: GpuComputeNode,
): GpuComputeNode {
    return (params: GpuFragmentParams) => {
        const baseNode = base ? base(params) : null
        const root = params.gpu?.root
        const device = params.gpu?.device
        if (!root || !device || plan.inputs.length === 0) return baseNode
        const frame = frameTracker(params, component.animated)
        const instances = new Map<string, InputInstance>()
        const lates: LateTexture[] = []
        const reported = new Set<string>()
        let childTexture: InputTexture | null = null

        const late = (kit: KitTexture, premultiplied: boolean, format: GPUTextureFormat): InputTexture => {
            const entry: LateTexture = {key: kit.key, resolved: null}
            lates.push(entry)
            return {
                current: () => entry.resolved,
                premultiplied,
                format,
                size: () => {
                    const t = entry.resolved as {size?: readonly number[]; width?: number; height?: number} | null
                    if (t?.size) return {width: t.size[0], height: t.size[1]}
                    return {width: params.dimensions.width, height: params.dimensions.height}
                },
            }
        }

        const sourceOf = (src: InputSource, output = ''): InputTexture | null => {
            if (isChildSource(src)) {
                if (!params.childNode) return null
                childTexture ??= late(params.convertToTexture(params.childNode), true, 'rgba16float')
                return childTexture
            }
            if (isPropRef(src)) {
                const config = component.props[src.name]
                const type = Array.isArray(config?.ui?.type) ? config?.ui?.type[0] : config?.ui?.type
                if (type === 'layer') return late(layerTexture(params, src.name), true, 'rgba16float')
                if (type === 'image-upload') return late(imageTexture(params, src.name), false, 'rgba8unorm')
                throw new Error(`input source p('${src.name}') must be an image or layer prop (video cannot be read by compute)`)
            }
            if (isInputNode(src)) {
                const planned = plan.inputs.find((p) => p.node === src)
                const inst = planned ? instances.get(planned.name) : undefined
                return inst?.outputs[output] ?? null
            }
            return null
        }

        for (const planned of plan.inputs) {
            const ctx: InputBuildContext = {
                name: planned.name,
                componentName: component.name,
                params,
                root,
                device,
                props: component.props,
                frame,
                lib: plan.lib,
                helpers: component.helpers,
                siblings: () => instances,
                source: sourceOf,
                canvas: () => ({width: params.dimensions.width, height: params.dimensions.height}),
                onCleanup: params.onCleanup,
                onResize: (cb) => params.onResize((size) => cb(size)),
                report: (message) => {
                    if (reported.has(message)) return
                    reported.add(message)
                    authorError(`[shaders] ${component.name}: input '${planned.name}': ${message}`)
                },
            }
            try {
                const inst = planned.node[BUILD_INPUT](ctx)
                if (inst) instances.set(planned.name, inst)
            } catch (e) {
                ctx.report(e instanceof Error ? e.message : String(e))
            }
        }

        const outputs: Record<string, KitTexture> = {}
        for (const [name, inst] of instances) {
            for (const [out, tex] of Object.entries(inst.outputs)) {
                outputs[outputKey(name, out)] = params.registerMediaTexture(() => tex.current(), {unfilterable: tex.format.endsWith('32float')})
                if (tex.premultiplied) premultipliedKeys(params).add(outputKey(name, out))
            }
        }

        return {
            outputs: {...(baseNode?.outputs ?? {}), ...outputs},
            getComputeNodes: (frameParams: unknown) => {
                frame.update(frameParams as never)
                const steps: ComputeStep[] = []
                const before = baseNode?.getComputeNodes(frameParams)
                if (before) steps.push(...before)
                for (const inst of instances.values()) {
                    try {
                        steps.push(...inst.steps())
                    } catch (e) {
                        authorError(`[shaders] ${component.name}: ${e instanceof Error ? e.message : String(e)}`)
                    }
                }
                return steps
            },
            bindInputs: (resolve) => {
                baseNode?.bindInputs?.(resolve)
                for (const l of lates) l.resolved = resolve(l.key)?.texture ?? null
                for (const inst of instances.values()) inst.bindLate?.(resolve)
            },
        }
    }
}
