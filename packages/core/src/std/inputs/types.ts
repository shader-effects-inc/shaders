/**
 * Inputs: what the engine computes for a component before its WGSL body runs. A blurred copy
 * of the layer inside, the layer's own last frame, an extra pass at half resolution, a
 * simulation, a particle field. List them in the definition's `inputs` and read each by name in
 * WGSL: `soft(uv)`, `last(uv)`, `field.slope(uv)`.
 */
// Maintainer notes. Every input word returns a plain data node (JSON-serialisable fields) that
// also carries, under a symbol, the function that builds it on the GPU. Symbols do not
// serialise, so the data stays clean for the IR, and only the words a component imports pull
// their GPU code into a bundle (§8.4 T1 of planning/AGENT_SHADER_FORMAT_PLAN.md).
import type {TgpuRoot} from 'typegpu'
import type {GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'
import type {PropConfig} from '../../types'
import type {PropRef} from '../values'
import type {WgslLibFn} from '../wgslLibrary'
import type {FrameTracker} from './frame'

/** The symbol an input node's GPU builder sits under. */
export const BUILD_INPUT: unique symbol = Symbol.for('shaders.std.buildInput') as never

/** The layer inside this one, as a source for an input (`blur(child, …)`). */
export interface ChildSource {
    readonly kind: 'child'
}

/**
 * The layer inside this one, as a source for an input. `blur(child, {radius: 12})` blurs it.
 *
 * @see blur, bloom, downsample
 */
export const child: ChildSource = Object.freeze({kind: 'child' as const})

/** What an input can read: the child, an image/video/layer prop, or another input. */
export type InputSource = ChildSource | PropRef | InputNode

/**
 * How a body reads an output: `uv` samples it smoothly at a position (`soft(uv)`), `texel` reads
 * the one texel under a position exactly, unblended (values that must not mix across a boundary,
 * like a nearest point or an index), `index` reads slot i of a list (`clicks(i)`).
 */
export type InputReadKind = 'uv' | 'texel' | 'index'

/** A texture an input produced or reads, as a TypeGPU texture created with `sampled` usage. */
export interface InputTexture {
    /** The current texture. Inputs that resize swap it; read it again each frame. */
    readonly current: () => unknown
    /** True when the texture holds premultiplied alpha (the child's render-to-texture does). */
    readonly premultiplied: boolean
    /** The texel format, for raw compute modules that bind it. */
    readonly format: GPUTextureFormat
    /** Pixel size, for modules that `textureLoad`. */
    readonly size: () => {width: number; height: number}
    /** How a body reads this output, when it differs from its input's `readBy`. */
    readonly readBy?: InputReadKind
}

/** One built input: its named outputs (`''` is the default) and its per-frame GPU work. */
export interface InputInstance {
    readonly outputs: Record<string, InputTexture>
    /** The compute steps to run this frame, in order. Called once per frame. */
    steps(): ComputeStep[]
    /** Bind textures the composition allocates after composing (the child's render-to-texture). */
    bindLate?(resolve: (key: string) => {texture: unknown} | undefined): void
    /** Pointer use, so the renderer wires mouse listeners. */
    readonly usesPointer?: boolean
    /** How a body reads it (copied from the node). */
    readonly readBy?: InputReadKind
}

/** What an input builder gets: the component's context and ways to reach its sources. */
export interface InputBuildContext {
    /** The input's name in `inputs`, or a generated name for one nested inside another. */
    readonly name: string
    readonly componentName: string
    readonly params: GpuFragmentParams
    readonly root: TgpuRoot
    readonly device: GPUDevice
    readonly props: Readonly<Record<string, PropConfig<unknown>>>
    readonly frame: FrameTracker
    /** Library functions the component listed (raw compute modules paste their source). */
    readonly lib: readonly WgslLibFn[]
    /** The component's `helpers` WGSL, or null. */
    readonly helpers: string | null
    /** Every other input by name, for raw modules that read one (`field.slope(cell)`). */
    readonly siblings: () => ReadonlyMap<string, InputInstance>
    /** Resolve a source to a texture: the child, a media prop, or another input's output. */
    source(src: InputSource, output?: string): InputTexture | null
    /** Canvas size in device pixels, live. */
    canvas(): {width: number; height: number}
    /** Register work to run when the component is torn down (destroy textures, buffers). */
    onCleanup(cb: () => void): void
    onResize(cb: (size: {width: number; height: number}) => void): void
    /** Report an authoring error once, naming the input. */
    report(message: string): void
}

/** The function that builds an input on the GPU. Returns null when it cannot run (no child). */
export type InputBuilder = (ctx: InputBuildContext) => InputInstance | null

/**
 * An input node: plain data plus its builder. `kind` names the word; the other fields are
 * the word's arguments as data.
 */
export interface InputNode {
    readonly kind: string
    /** Output names this input exposes besides the default (`field.slope`). */
    readonly outputNames?: readonly string[]
    /** Whether the input reads the layer inside (the component then needs a child). */
    readonly readsChild?: boolean
    /** How a body reads it: `name(uv)` (default) or `name(i)` for a list of slots (events). */
    readonly readBy?: InputReadKind
    /** Per-output read kinds, by output name (`''` is the default), overriding `readBy`. */
    readonly outputReads?: Readonly<Record<string, InputReadKind>>
    /** Outputs (by name) whose textures are 32-bit floats, when only some are (see `unfilterable`). */
    readonly unfilterableOutputs?: readonly string[]
    /** Its textures are 32-bit floats: read with textureLoad and blended by hand, never through a filtering sampler. */
    readonly unfilterable?: boolean
    /**
     * It reads the pointer through data the code scan can't see (a `cursor` force, a fluid's
     * stir). Known when the input is made, so `defineShader` can declare `usesPointer` and the
     * renderer attaches its mouse listeners (without them a real mouse never arrives).
     */
    readonly readsPointer?: boolean
    readonly [BUILD_INPUT]: InputBuilder
}

/** True for a value made by an input word. */
export function isInputNode(value: unknown): value is InputNode {
    return typeof value === 'object' && value !== null && typeof (value as Record<symbol, unknown>)[BUILD_INPUT] === 'function'
}

/** True for the `child` source token. */
export function isChildSource(value: unknown): value is ChildSource {
    return typeof value === 'object' && value !== null && (value as {kind?: unknown}).kind === 'child'
}

/** Attach a builder to a data node without making the builder an enumerable (serialised) field. */
export function withBuilder<T extends object>(node: T, build: InputBuilder): T & InputNode {
    Object.defineProperty(node, BUILD_INPUT, {value: build, enumerable: false})
    return node as T & InputNode
}

/** A scalar argument: a number or a prop reference, read live each frame. */
export type ScalarArg = number | PropRef

/** Read a scalar argument's value this frame. */
export function scalarValue(arg: ScalarArg | undefined, params: GpuFragmentParams, fallback: number): number {
    if (arg === undefined) return fallback
    if (typeof arg === 'number') return arg
    const v = params.getCpuValue(arg.name)
    return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

/** True when a scalar argument refers to a prop (so a recompose is not needed when it changes). */
export function isPropRef(value: unknown): value is PropRef {
    return typeof value === 'object' && value !== null && (value as {kind?: unknown}).kind === 'prop'
}
