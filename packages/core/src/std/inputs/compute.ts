/**
 * The raw compute tier: records (`struct`), storage buffers of them (`buffer`), and your own
 * compute passes over them (`kernel`). For anything the simulation words do not cover. The
 * engine still owns the bind groups, workgroup sizes, bounds checks and dispatch.
 */
// Maintainer notes. A buffer is an input with no texture outputs; it exposes its GPU buffer to
// sibling kernels, which bind every buffer their code names as `var<storage, read_write>`. A
// kernel runs once per element of the buffer it is `over` (index in `index`), or once per cell of
// a grid size (`cell: vec2i`), every frame, or once (`when: 'init'`, again on resize).
import type {WgslBody} from '../wgsl'
import {createComputeModule, rewriteForModule, type ModuleResource} from './module'
import {bodyText, moduleFor, siblingReads} from './common'
import {identifiersIn} from '../wgslSource'
import {scalarValue, withBuilder, type InputBuildContext, type InputInstance, type InputNode, type ScalarArg} from './types'

/** The field types a `struct` may hold. */
export type StructFieldType =
    | 'f32' | 'i32' | 'u32'
    | 'vec2f' | 'vec3f' | 'vec4f'
    | 'vec2i' | 'vec3i' | 'vec4i'
    | 'vec2u' | 'vec3u' | 'vec4u'

/** A record type, by name, for buffers and kernels. */
export interface StructType {
    readonly kind: 'struct'
    readonly name: string
    readonly fields: Readonly<Record<string, StructFieldType>>
}

/**
 * Declare a record type. The engine writes the WGSL `struct` for you and sizes buffers from it,
 * so it is type-checked in TypeScript and identical in every pass that uses it.
 *
 * @example
 * ```ts
 * const Star = struct('Star', {pos: 'vec3f', vel: 'vec3f', heat: 'f32'})
 * ```
 * @see buffer, kernel
 */
export function struct(name: string, fields: Record<string, StructFieldType>): StructType {
    if (!/^[A-Z][A-Za-z0-9_]*$/.test(name)) throw new Error(`struct('${name}'): use a capitalised WGSL type name, e.g. 'Particle'`)
    if (Object.keys(fields).length === 0) throw new Error(`struct('${name}'): give it at least one field`)
    return {kind: 'struct', name, fields}
}

const FIELD_SIZE: Record<StructFieldType, {size: number; align: number}> = {
    f32: {size: 4, align: 4}, i32: {size: 4, align: 4}, u32: {size: 4, align: 4},
    vec2f: {size: 8, align: 8}, vec2i: {size: 8, align: 8}, vec2u: {size: 8, align: 8},
    vec3f: {size: 12, align: 16}, vec3i: {size: 12, align: 16}, vec3u: {size: 12, align: 16},
    vec4f: {size: 16, align: 16}, vec4i: {size: 16, align: 16}, vec4u: {size: 16, align: 16},
}

/** The byte stride of one element in a storage array of this struct. */
export function structStride(type: StructType): number {
    let offset = 0
    let align = 4
    for (const t of Object.values(type.fields)) {
        const f = FIELD_SIZE[t]
        offset = Math.ceil(offset / f.align) * f.align + f.size
        align = Math.max(align, f.align)
    }
    return Math.ceil(offset / align) * align
}

/** The WGSL declaration of a struct type. */
export function structWgsl(type: StructType): string {
    return `struct ${type.name} {\n${Object.entries(type.fields).map(([n, t]) => `  ${n}: ${t},`).join('\n')}\n}`
}

export interface BufferInput extends InputNode {
    readonly kind: 'buffer'
    readonly type: StructType
    readonly count: ScalarArg
    readonly max: number
}

/** A storage buffer a sibling kernel can read and write. */
export interface BufferHandle {
    readonly name: string
    readonly type: StructType
    readonly buffer: () => GPUBuffer
    /** How many elements are live this frame (≤ the allocated `max`). */
    readonly count: () => number
}

const buffers = new WeakMap<InputInstance, BufferHandle>()

/** The buffer an input instance owns, if it is a buffer. */
export function bufferOf(instance: InputInstance | undefined): BufferHandle | undefined {
    return instance ? buffers.get(instance) : undefined
}

/**
 * A storage buffer of `count` records of a `struct` type, zero-filled. Kernels name it in their
 * WGSL to read and write it (`stars[index].pos`). `count` may be a prop (with `max` the most it
 * can grow to); only the first `count` records are processed each frame.
 *
 * @example
 * ```ts
 * const Star = struct('Star', {pos: 'vec2f', vel: 'vec2f'})
 * inputs: {
 *   stars: buffer(Star, p('count'), {max: 65536}),
 *   move: kernel(wgsl`stars[index].pos += stars[index].vel * dt;`, {over: 'stars'}),
 * },
 * ```
 * @see kernel, struct
 */
export function buffer(type: StructType, count: ScalarArg, opts: {max?: number} = {}): BufferInput {
    const max = Math.max(1, Math.floor(opts.max ?? (typeof count === 'number' ? count : 65536)))
    if (max > 1 << 22) throw new Error(`buffer: ${max} records is more than the 4,194,304 a buffer may hold`)
    const node = {kind: 'buffer' as const, type, count, max}
    return withBuilder(node, (ctx) => {
        const stride = structStride(type)
        const gpu = ctx.device.createBuffer({size: Math.max(16, stride * max), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${ctx.componentName}:${ctx.name}`})
        ctx.onCleanup(() => gpu.destroy())
        const instance: InputInstance = {outputs: {}, steps: () => []}
        buffers.set(instance, {
            name: ctx.name,
            type,
            buffer: () => gpu,
            count: () => Math.max(0, Math.min(max, Math.floor(scalarValue(count, ctx.params, max)))),
        })
        return instance
    })
}

export interface KernelInput extends InputNode {
    readonly kind: 'kernel'
    readonly body: string
    /** The buffer (by its `inputs` name) to run once per element of, or a fixed grid size. */
    readonly over: string | readonly [number, number]
    readonly iterations: number
    readonly when: 'frame' | 'init'
}

/**
 * Your own compute pass. Over a buffer (`over: 'stars'`) it runs once per live element with
 * `index: u32`; over a grid size (`over: [64, 64]`) once per cell with `cell: vec2i`. It reads and
 * writes every buffer it names, reads other inputs by name (`soft(p)`), and sees props,
 * built-ins (`time`, `dt`, `pointer`…), library functions and helpers. `iterations` repeats it
 * within a frame; `when: 'init'` runs it once, to seed a buffer (and again after a resize).
 *
 * @example
 * ```ts
 * move: kernel(wgsl`
 *   var s = stars[index];
 *   s.vel += (pointer - s.pos) * dt;
 *   s.pos += s.vel * dt;
 *   stars[index] = s;
 * `, {over: 'stars'}),
 * ```
 * @tip A kernel has no return value: write your results into a buffer.
 * @see buffer, struct, simulate
 */
export function kernel(
    body: WgslBody | string,
    opts: {over: string | readonly [number, number]; iterations?: number; when?: 'frame' | 'init'},
): KernelInput {
    const text = bodyText(body)
    const iterations = Math.max(1, Math.min(64, Math.round(opts.iterations ?? 1)))
    const node = {kind: 'kernel' as const, body: text, over: opts.over, iterations, when: opts.when ?? 'frame'}
    return withBuilder(node, (ctx) => buildKernel(ctx, node))
}

function buildKernel(ctx: InputBuildContext, node: {body: string; over: string | readonly [number, number]; iterations: number; when: 'frame' | 'init'}): InputInstance | null {
    const ids = identifiersIn(node.body)
    const used: BufferHandle[] = []
    for (const [name, inst] of ctx.siblings()) {
        const handle = bufferOf(inst)
        if (handle && (ids.has(name) || node.over === name)) used.push(handle)
    }
    let over: BufferHandle | null = null
    if (typeof node.over === 'string') {
        over = used.find((b) => b.name === node.over) ?? null
        if (!over) {
            ctx.report(`over: '${node.over}' is not a buffer listed in inputs (list the buffer before the kernel)`)
            return null
        }
    }
    const structs = [...new Map(used.map((b) => [b.type.name, b.type])).values()]
    const reads = siblingReads(ctx, node.body)
    const mod = moduleFor(ctx)
    const resources: ModuleResource[] = used.map((b) => ({
        kind: 'storage' as const,
        name: b.name,
        type: `array<${b.type.name}>`,
        access: 'read_write' as const,
        buffer: b.buffer,
    }))
    const countBuffer = ctx.device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${ctx.name}:count`})
    ctx.onCleanup(() => countBuffer.destroy())
    resources.push({kind: 'uniform', name: '_k', type: 'vec4u', buffer: () => countBuffer})
    const grid = Array.isArray(node.over) ? (node.over as readonly [number, number]) : null
    const fnName = `_kernel_${ctx.name.replace(/\W/g, '_')}`
    const entry = over
        ? `  let index = gid.x;\n  if (index >= _k.x) { return; }\n  ${fnName}(index);`
        : `  if (gid.x >= _k.x || gid.y >= _k.y) { return; }\n  ${fnName}(vec2i(gid.xy));`
    const signature = over ? `(index: u32)` : `(cell: vec2i)`
    const module = createComputeModule(mod, {
        label: `kernel '${ctx.name}'`,
        decls: structs.map(structWgsl).join('\n'),
        segments: [{label: `kernel '${ctx.name}'`, text: rewriteForModule(`fn ${fnName}${signature} {\n${node.body}\n}`, mod, reads), offset: 1}],
        entry,
        resources,
        textureReads: reads,
        workgroup: over ? [64, 1] : [8, 8],
    })
    const threads = (): readonly [number, number] => (over ? [over.count(), 1] : [grid![0], grid![1]])
    const step = module.step(threads)
    const countData = new Uint32Array(4)
    let initialised = false
    ctx.onResize(() => {
        initialised = false
    })
    const writeCount = () => {
        const [x, y] = threads()
        countData.set([x, y, 0, 0])
        ctx.device.queue.writeBuffer(countBuffer, 0, countData)
    }
    return {
        outputs: {},
        steps() {
            if (node.when === 'init') {
                if (initialised) return []
                // Runs once: wait until the pipeline has compiled, or the one run is lost.
                if (!module.ready()) return []
                initialised = true
            }
            return [writeCount, ...Array.from({length: node.iterations}, () => step)]
        },
    }
}
