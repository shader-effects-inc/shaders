/**
 * Raw compute modules: the WGSL an input runs on the GPU (an agent's kernel, a pass, a
 * simulation step), assembled with everything a body may name: the frame values, the
 * component's props, the library functions it listed, its helpers, and the textures of the
 * other inputs. The engine owns the bind groups, workgroup sizes, bounds checks and dispatch.
 */
// Maintainer notes. These modules bypass TypeGPU on purpose: the code is runtime WGSL text, and
// raw `device.createComputePipelineAsync` with an explicit layout keeps it simple and lets a
// compile error point at the author's line. Each module binds group 0 as: 0 frame uniform, 1 props
// uniform, 2 linear-clamp sampler, then its own resources from 3. Bare names in the author's code
// (`time`, `pointer`, a prop) are rewritten to their uniform fields; calls to inputs (`soft(p)`,
// `field.slope(p)`) to generated sample helpers. Dispatch is an inline compute step that encodes
// and submits its own pass, so queue order keeps it in sequence with TypeGPU-dispatched steps.
import type {GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'
import {rawTimestampWrites} from '../../gpu/gpuTimer'
import type {PropConfig} from '../../types'
import {authorError} from '../../gpu/support'
import {transformPosition} from '../../utilities/transformations'
import {wgslTypeForProp} from '../wgsl'
import {libSource, type WgslLibFn} from '../wgslLibrary'
import {appendCallArgs, blankComments, identifiersIn, parseWgslDecls, replaceIdentifier} from '../wgslSource'

export {appendCallArgs}
import type {FrameTracker} from './frame'
import type {InputTexture} from './types'

// ── Shared per-component context ────────────────────────────────────────────────────────

/** The names every module binds without declaration, and the uniform field each reads. */
export const FRAME_FIELDS = ['time', 'dt', 'frame', 'aspect', 'viewport', 'pointer', 'pointerVelocity', 'pointerDown', 'idle', 'pointerOver'] as const

const FRAME_STRUCT = `struct _Frame {
  time: f32, dt: f32, frame: f32, aspect: f32,
  viewport: vec2f, pointer: vec2f, pointerVelocity: vec2f,
  pointerDown: f32, idle: f32, pointerOver: f32,
}`
// 13 floats (52 bytes, 56 with the struct's 8-byte alignment), rounded up to a 16-byte multiple.
const FRAME_BYTES = 64

interface PropSlot {
    name: string
    type: 'f32' | 'vec2f' | 'vec3f' | 'vec4f'
    offset: number
    position: boolean
}

/** One component instance's module plumbing: its frame and props uniforms and shared sampler. */
export interface ModuleContext {
    readonly device: GPUDevice
    readonly params: GpuFragmentParams
    readonly frame: FrameTracker
    readonly componentName: string
    readonly lib: readonly WgslLibFn[]
    readonly helpers: string | null
    readonly propSlots: readonly PropSlot[]
    readonly frameBuffer: GPUBuffer
    readonly propsBuffer: GPUBuffer
    readonly sampler: GPUSampler
    /** Write the frame and props uniforms for this frame (once per frame, however often called). */
    writeUniforms(): void
}

const contexts = new WeakMap<object, ModuleContext>()

function alignUp(n: number, a: number): number {
    return Math.ceil(n / a) * a
}

function layoutProps(props: Readonly<Record<string, PropConfig<unknown>>>): {slots: PropSlot[]; bytes: number} {
    const typed: Omit<PropSlot, 'offset'>[] = []
    for (const [name, config] of Object.entries(props)) {
        const type = wgslTypeForProp(config)
        if (type !== 'f32' && type !== 'vec2f' && type !== 'vec3f' && type !== 'vec4f') continue
        typed.push({name, type, position: config.transform === (transformPosition as unknown)})
    }
    // Widest first keeps the packing tight: vec4/vec3 at 16, vec2 at 8, f32 at 4.
    const order = {vec4f: 0, vec3f: 1, vec2f: 2, f32: 3}
    typed.sort((a, b) => order[a.type] - order[b.type])
    const slots: PropSlot[] = []
    let offset = 0
    for (const t of typed) {
        const align = t.type === 'f32' ? 4 : t.type === 'vec2f' ? 8 : 16
        offset = alignUp(offset, align)
        slots.push({...t, offset})
        offset += t.type === 'f32' ? 4 : t.type === 'vec2f' ? 8 : t.type === 'vec3f' ? 12 : 16
    }
    return {slots, bytes: Math.max(16, alignUp(offset, 16))}
}

/** The module context for a component instance (one per fragment params object). */
export function moduleContext(init: {
    device: GPUDevice
    params: GpuFragmentParams
    frame: FrameTracker
    componentName: string
    props: Readonly<Record<string, PropConfig<unknown>>>
    lib: readonly WgslLibFn[]
    helpers: string | null
    onCleanup: (cb: () => void) => void
}): ModuleContext {
    const existing = contexts.get(init.params)
    if (existing) return existing
    const {device, params, frame} = init
    const {slots, bytes} = layoutProps(init.props)
    const frameBuffer = device.createBuffer({size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${init.componentName}:frame`})
    const propsBuffer = device.createBuffer({size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${init.componentName}:props`})
    const sampler = device.createSampler({magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'})
    init.onCleanup(() => {
        frameBuffer.destroy()
        propsBuffer.destroy()
    })
    const frameData = new Float32Array(FRAME_BYTES / 4)
    const propData = new Float32Array(bytes / 4)
    let writtenFrame = -1
    const ctx: ModuleContext = {
        device, params, frame,
        componentName: init.componentName,
        lib: init.lib,
        helpers: init.helpers,
        propSlots: slots,
        frameBuffer, propsBuffer, sampler,
        writeUniforms() {
            const v = frame.values
            if (v.frame === writtenFrame) return
            writtenFrame = v.frame
            frameData.set([v.time, v.dt, v.frame, v.aspect, v.viewport[0], v.viewport[1], v.pointer[0], v.pointer[1],
                v.pointerVelocity[0], v.pointerVelocity[1], v.pointerDown, v.idle, v.pointerOver])
            device.queue.writeBuffer(frameBuffer, 0, frameData)
            for (const slot of slots) {
                const raw = params.getCpuValue(slot.name)
                const at = slot.offset / 4
                if (typeof raw === 'number') {
                    propData[at] = raw
                    continue
                }
                const view = raw as {x?: number; y?: number; z?: number; w?: number} | undefined
                const comps = [view?.x ?? 0, view?.y ?? 0, view?.z ?? 0, view?.w ?? 0]
                // Positions are stored with y flipped for the library's words; bodies read uv space.
                if (slot.position) comps[1] = 1 - comps[1]
                const n = slot.type === 'vec2f' ? 2 : slot.type === 'vec3f' ? 3 : slot.type === 'vec4f' ? 4 : 1
                for (let i = 0; i < n; i++) propData[at + i] = comps[i]
            }
            device.queue.writeBuffer(propsBuffer, 0, propData)
        },
    }
    contexts.set(params, ctx)
    return ctx
}

// ── Rewriting the author's code ─────────────────────────────────────────────────────────

/** A texture a module reads by calling its name: `soft(p)` or `field.slope(p)`. */
export interface ModuleTextureRead {
    /** The name the code calls: `soft`, or `field.slope` for a named output. */
    readonly call: string
    readonly texture: InputTexture
    /** `index` reads slot i of row 0 (`clicks(i)`); default `uv`. */
    readonly by?: 'uv' | 'index'
}

function sanitize(name: string): string {
    return name.replace(/[^A-Za-z0-9_]/g, '_')
}

/**
 * Rewrite an author's WGSL so it compiles inside a module: frame names and props read their
 * uniforms, input calls read their textures, and `textureSample` (no implicit derivatives in
 * compute) becomes `textureSampleLevel(…, 0.0)`.
 */
export function rewriteForModule(code: string, ctx: Pick<ModuleContext, 'propSlots'>, reads: readonly ModuleTextureRead[]): string {
    let text = code
    // Named outputs before plain names: `field.slope(` must not become `field(`.
    const sorted = [...reads].sort((a, b) => b.call.length - a.call.length)
    for (const r of sorted) {
        const re = new RegExp(`(?<![\\w.])${r.call.replace('.', '\\s*\\.\\s*')}\\s*\\(`, 'g')
        text = text.replace(re, `_read_${sanitize(r.call)}(`)
    }
    text = appendCallArgs(text, 'textureSample', ['0.0']).replace(/(?<![\w.])textureSample\s*\(/g, 'textureSampleLevel(')
    for (const f of FRAME_FIELDS) text = replaceIdentifier(text, f, `_f.${f}`)
    for (const slot of ctx.propSlots) text = replaceIdentifier(text, slot.name, `_p.${slot.name}`)
    return text
}

/** True for a 32-bit float texture, which needs an unfilterable binding and `_bilinear` reads. */
function isUnfilterable(tex: InputTexture): boolean {
    return tex.format === 'rgba32float' || tex.format === 'rg32float' || tex.format === 'r32float'
}

/** Bilinear blend of four texels by textureLoad: smooth reads of a 32-bit texture with no sampler. */
export const BILINEAR_WGSL = `fn _bilinear(t: texture_2d<f32>, p: vec2f) -> vec4f {
  let size = vec2i(textureDimensions(t));
  let q = p * vec2f(size) - 0.5;
  let i = vec2i(floor(q));
  let f = q - floor(q);
  let m = size - vec2i(1);
  let a = textureLoad(t, clamp(i, vec2i(0), m), 0);
  let b = textureLoad(t, clamp(i + vec2i(1, 0), vec2i(0), m), 0);
  let c = textureLoad(t, clamp(i + vec2i(0, 1), vec2i(0), m), 0);
  let d = textureLoad(t, clamp(i + vec2i(1, 1), vec2i(0), m), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}`

/** WGSL for the helpers that sample each input texture a module reads. */
function readHelpers(reads: readonly ModuleTextureRead[], firstBinding: number): {decls: string; bindings: number} {
    let b = firstBinding
    const parts: string[] = []
    for (const r of reads) {
        const id = sanitize(r.call)
        parts.push(`@group(0) @binding(${b++}) var _t_${id}: texture_2d<f32>;`)
        // 32-bit textures can't use a filtering sampler on every GPU: blend four exact texels.
        const sample = isUnfilterable(r.texture) ? `_bilinear(_t_${id}, p)` : `textureSampleLevel(_t_${id}, _samp, p, 0.0)`
        if (r.by === 'index') {
            parts.push(`fn _read_${id}(i: i32) -> vec4f { return textureLoad(_t_${id}, vec2i(clamp(i, 0, i32(textureDimensions(_t_${id}).x) - 1), 0), 0); }`)
            continue
        }
        parts.push(r.texture.premultiplied
            ? `fn _read_${id}(p: vec2f) -> vec4f { let c = ${sample}; return select(vec4f(c.rgb / c.a, c.a), vec4f(0.0), c.a <= 0.0001); }`
            : `fn _read_${id}(p: vec2f) -> vec4f { return ${sample}; }`)
    }
    if (reads.some((r) => r.by !== 'index' && isUnfilterable(r.texture))) parts.unshift(BILINEAR_WGSL)
    return {decls: parts.join('\n'), bindings: b}
}

// ── Resources and dispatch ──────────────────────────────────────────────────────────────

export type ModuleResource =
    | {readonly kind: 'uniform'; readonly name: string; readonly type: string; readonly buffer: () => GPUBuffer}
    | {readonly kind: 'storage'; readonly name: string; readonly type: string; readonly access: 'read' | 'read_write'; readonly buffer: () => GPUBuffer}
    | {readonly kind: 'texture'; readonly name: string; readonly texture: () => unknown; readonly unfilterable?: boolean}
    | {readonly kind: 'storageTexture'; readonly name: string; readonly format: GPUTextureFormat; readonly texture: () => unknown}

/** A piece of the author's code inside a module, so a compile error can name it and its line. */
export interface ModuleSegment {
    readonly label: string
    readonly text: string
    /** Lines of generated code at the top of `text` before the author's own first line (a fn header). */
    readonly offset?: number
}

export interface ModuleSpec {
    /** Shown in error messages: `kernel 'step'`, `pass 'nebula'`. */
    readonly label: string
    /** Generated declarations (structs, resource-specific helpers) placed before the author's code. */
    readonly decls?: string
    /** The author's code (helpers, bodies), each already rewritten, in order. */
    readonly segments: readonly ModuleSegment[]
    /** The compute entry's body. `gid` is the global invocation id. */
    readonly entry: string
    readonly resources: readonly ModuleResource[]
    readonly textureReads?: readonly ModuleTextureRead[]
    readonly workgroup: readonly [number, number]
    /** Include the listed library functions and the component's helpers (default true). */
    readonly includeShared?: boolean
}

export interface ComputeModule {
    /** A compute step dispatching `threads` invocations (x, y), bounds left to the entry. */
    step(threads: () => readonly [number, number]): ComputeStep
    /** The assembled WGSL (for tests and error reports). */
    readonly code: string
    /** True once a dispatch would actually run: the pipeline has compiled (asynchronously) and
     *  every texture it binds exists. A step that must run exactly once (a grid's `init`) waits
     *  for this, or its single chance lands before the pipeline does and is lost. */
    ready(): boolean
}

const unwrapCache = new WeakMap<object, GPUTexture>()

function rawTexture(params: GpuFragmentParams, tex: unknown): GPUTexture | null {
    if (!tex) return null
    if (typeof (tex as GPUTexture).createView === 'function' && !(tex as {resourceType?: unknown}).resourceType) return tex as GPUTexture
    const cached = unwrapCache.get(tex as object)
    if (cached) return cached
    const raw = params.gpu.root.unwrap(tex as never) as unknown as GPUTexture
    unwrapCache.set(tex as object, raw)
    return raw
}

/** The helpers and library source a module's code reaches (transitively), rewritten for compute. */
function sharedCode(ctx: ModuleContext, code: string, reads: readonly ModuleTextureRead[]): {lib: string; helpers: string} {
    const reached = identifiersIn(code)
    let helperText = ''
    if (ctx.helpers) {
        const decls = parseWgslDecls(ctx.helpers)
        const byName = new Map(decls.fns.map((f) => [f.name, f]))
        const keep = new Set<string>()
        const visit = (name: string) => {
            const fn = byName.get(name)
            if (!fn || keep.has(name)) return
            keep.add(name)
            for (const id of identifiersIn(fn.body)) {
                reached.add(id)
                visit(id)
            }
        }
        for (const id of [...reached]) visit(id)
        const parts = [
            ...decls.structs.map((x) => x.text),
            ...decls.consts.map((x) => x.text),
            ...decls.fns.filter((f) => keep.has(f.name)).map((f) => f.text),
        ]
        helperText = parts.length ? rewriteForModule(parts.join('\n\n'), ctx, reads) : ''
    }
    const lib = ctx.lib.filter((f) => reached.has(f.name) || f.structs.some((st) => reached.has(st)))
    return {lib: lib.length ? libSource(lib) : '', helpers: helperText}
}

/** Build a compute module for a component. Compile errors are reported once, with the author's line. */
export function createComputeModule(ctx: ModuleContext, spec: ModuleSpec): ComputeModule {
    const {device} = ctx
    const includeShared = spec.includeShared ?? true
    // Only the helpers and library functions this module's code reaches are pasted in: a fragment-only
    // function (one using `fwidth`, like the anti-aliasing family) would not compile in compute.
    const shared = includeShared ? sharedCode(ctx, [spec.decls ?? '', spec.entry, ...spec.segments.map((x) => x.text)].join('\n'), spec.textureReads ?? []) : {lib: '', helpers: ''}
    const {lib, helpers} = shared
    const props = ctx.propSlots.length
        ? `struct _Props {\n${ctx.propSlots.map((s) => `  ${s.name}: ${s.type},`).join('\n')}\n}`
        : 'struct _Props { _unused: f32, }'

    const resourceDecls: string[] = []
    let binding = 3
    const layoutEntries: GPUBindGroupLayoutEntry[] = [
        {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'uniform'}},
        {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'uniform'}},
        {binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: {type: 'filtering'}},
    ]
    const bindingFor: {binding: number; get: () => GPUBindingResource | null; identity: () => unknown}[] = []
    for (const r of spec.resources) {
        const b = binding++
        if (r.kind === 'uniform') {
            resourceDecls.push(`@group(0) @binding(${b}) var<uniform> ${r.name}: ${r.type};`)
            layoutEntries.push({binding: b, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'uniform'}})
            bindingFor.push({binding: b, get: () => ({buffer: r.buffer()}), identity: r.buffer})
        } else if (r.kind === 'storage') {
            resourceDecls.push(`@group(0) @binding(${b}) var<storage, ${r.access}> ${r.name}: ${r.type};`)
            layoutEntries.push({binding: b, visibility: GPUShaderStage.COMPUTE, buffer: {type: r.access === 'read' ? 'read-only-storage' : 'storage'}})
            bindingFor.push({binding: b, get: () => ({buffer: r.buffer()}), identity: r.buffer})
        } else if (r.kind === 'texture') {
            resourceDecls.push(`@group(0) @binding(${b}) var ${r.name}: texture_2d<f32>;`)
            layoutEntries.push({binding: b, visibility: GPUShaderStage.COMPUTE, texture: {sampleType: r.unfilterable ? 'unfilterable-float' : 'float'}})
            bindingFor.push({binding: b, get: () => rawTexture(ctx.params, r.texture())?.createView() ?? null, identity: r.texture})
        } else {
            resourceDecls.push(`@group(0) @binding(${b}) var ${r.name}: texture_storage_2d<${r.format}, write>;`)
            layoutEntries.push({binding: b, visibility: GPUShaderStage.COMPUTE, storageTexture: {access: 'write-only', format: r.format}})
            bindingFor.push({binding: b, get: () => rawTexture(ctx.params, r.texture())?.createView() ?? null, identity: r.texture})
        }
    }
    const reads = spec.textureReads ?? []
    const readDecl = readHelpers(reads, binding)
    for (const r of reads) {
        const b = binding++
        layoutEntries.push({binding: b, visibility: GPUShaderStage.COMPUTE, texture: {sampleType: isUnfilterable(r.texture) ? 'unfilterable-float' : 'float'}})
        bindingFor.push({binding: b, get: () => rawTexture(ctx.params, r.texture.current())?.createView() ?? null, identity: r.texture.current})
    }

    const header = [
        FRAME_STRUCT,
        '@group(0) @binding(0) var<uniform> _f: _Frame;',
        props,
        '@group(0) @binding(1) var<uniform> _p: _Props;',
        '@group(0) @binding(2) var _samp: sampler;',
        resourceDecls.join('\n'),
        readDecl.decls,
        spec.decls ?? '',
        lib,
        helpers,
    ].filter((s) => s.trim().length > 0).join('\n\n')

    // Track where each of the author's segments starts, so a compiler line maps back to theirs.
    const segmentStarts: {label: string; line: number; lines: number}[] = []
    let code = header
    for (const seg of spec.segments) {
        code += '\n\n'
        segmentStarts.push({label: seg.label, line: code.split('\n').length + (seg.offset ?? 0), lines: seg.text.split('\n').length - (seg.offset ?? 0)})
        code += seg.text
    }
    code += `\n\n@compute @workgroup_size(${spec.workgroup[0]}, ${spec.workgroup[1]})\nfn _main(@builtin(global_invocation_id) gid: vec3u) {\n${spec.entry}\n}\n`

    const where = (line: number): string => {
        for (const s of segmentStarts) {
            if (line >= s.line && line < s.line + s.lines) return `${s.label}, line ${line - s.line + 1}`
        }
        return `generated code line ${line}`
    }

    let pipeline: GPUComputePipeline | null = null
    let broken = false
    // Errors are reported to the author (below), never left uncaptured: an uncaptured device
    // error latches the whole renderer.
    device.pushErrorScope('validation')
    const layout = device.createBindGroupLayout({entries: layoutEntries, label: spec.label})
    const module = device.createShaderModule({code, label: `${ctx.componentName}: ${spec.label}`})
    void device.popErrorScope().then((e) => {
        if (e && !broken) {
            broken = true
            authorError(`[shaders] ${ctx.componentName}: ${spec.label}: ${e.message.split('\n')[0]}`)
        }
    }).catch(() => undefined)
    void module.getCompilationInfo?.().then((info) => {
        const errors = info.messages.filter((m) => m.type === 'error')
        if (errors.length === 0) return
        broken = true
        const lines = errors.slice(0, 4).map((m) => `  ${where(m.lineNum)}: ${m.message}`).join('\n')
        authorError(`[shaders] ${ctx.componentName}: ${spec.label} failed to compile:\n${lines}`)
    }).catch(() => undefined)
    device.pushErrorScope('validation')
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [layout]})
    void device.popErrorScope().catch(() => undefined)
    device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: {module, entryPoint: '_main'},
        label: `${ctx.componentName}: ${spec.label}`,
    }).then((p) => {
        pipeline = p
    }).catch((e: unknown) => {
        broken = true
        authorError(`[shaders] ${ctx.componentName}: ${spec.label} could not be built: ${e instanceof Error ? e.message : String(e)}`)
    })

    let bindGroup: GPUBindGroup | null = null
    let identities: unknown[] = []
    const ensureBindGroup = (): boolean => {
        const now = bindingFor.map((b) => {
            const id = b.identity()
            return typeof id === 'function' ? (id as () => unknown)() : id
        })
        if (bindGroup && now.every((v, i) => v === identities[i])) return true
        const entries: GPUBindGroupEntry[] = [
            {binding: 0, resource: {buffer: ctx.frameBuffer}},
            {binding: 1, resource: {buffer: ctx.propsBuffer}},
            {binding: 2, resource: ctx.sampler},
        ]
        for (const b of bindingFor) {
            const res = b.get()
            if (!res) return false // a late-bound texture is not there yet
            entries.push({binding: b.binding, resource: res})
        }
        device.pushErrorScope('validation')
        bindGroup = device.createBindGroup({layout, entries, label: spec.label})
        void device.popErrorScope().then((e) => {
            if (e && !broken) {
                broken = true
                authorError(`[shaders] ${ctx.componentName}: ${spec.label}: ${e.message.split('\n')[0]}`)
            }
        }).catch(() => undefined)
        identities = now
        return true
    }

    return {
        code,
        ready: () => !!pipeline && !broken && ensureBindGroup(),
        step(threads) {
            return (timer) => {
                if (!pipeline || broken) return
                ctx.writeUniforms()
                if (!ensureBindGroup()) return
                const [x, y] = threads()
                if (x <= 0 || y <= 0) return
                // Scoped, so a failing dispatch switches this pass off and tells the author instead
                // of raising uncaptured errors every frame (which stop the whole renderer).
                device.pushErrorScope('validation')
                const encoder = device.createCommandEncoder({label: spec.label})
                const pass = encoder.beginComputePass({label: spec.label, timestampWrites: rawTimestampWrites(ctx.params.gpu.root, timer, spec.label)})
                pass.setPipeline(pipeline)
                pass.setBindGroup(0, bindGroup!)
                pass.dispatchWorkgroups(Math.ceil(x / spec.workgroup[0]), Math.ceil(y / spec.workgroup[1]), 1)
                pass.end()
                device.queue.submit([encoder.finish()])
                void device.popErrorScope().then((e) => {
                    if (e && !broken) {
                        broken = true
                        authorError(`[shaders] ${ctx.componentName}: ${spec.label}: ${e.message.split('\n')[0]}`)
                    }
                }).catch(() => undefined)
            }
        },
    }
}

/** Create a 2D texture an input writes in compute and the body samples. */
export function createStorageTexture(
    params: GpuFragmentParams,
    size: {width: number; height: number},
    format: GPUTextureFormat,
    label: string,
    extraUsages: readonly ('render')[] = [],
): unknown {
    const root = params.gpu.root as unknown as {createTexture: (props: unknown) => {$usage: (...u: string[]) => unknown}}
    const create = root.createTexture as unknown as (p: unknown) => {$usage: (...u: string[]) => unknown}
    const tex = create.call(root, {size: [Math.max(1, size.width), Math.max(1, size.height)], format, label})
    return tex.$usage('storage', 'sampled', ...extraUsages)
}

/** Destroy a TypeGPU texture made by `createStorageTexture`. */
export function destroyTexture(tex: unknown): void {
    try {
        (tex as {destroy?: () => void})?.destroy?.()
    } catch {
        /* already gone */
    }
}

/** Every identifier-like token the author's code uses, for deciding what a module needs. */
export function namesUsed(code: string): Set<string> {
    const out = new Set<string>()
    for (const m of blankComments(code).matchAll(/(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)?/g)) {
        out.add(m[0].replace(/\s+/g, ''))
    }
    return out
}
