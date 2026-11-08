/**
 * Shared plumbing for inputs that run raw compute modules: the component's module context, a
 * small per-input uniform block, an output texture that follows the canvas size, and reads of
 * other inputs by the names a body calls.
 */
import {identifiersIn} from '../wgslSource'
import {isWgslBody, type WgslBody} from '../wgsl'
import {createStorageTexture, destroyTexture, moduleContext, type ModuleContext, type ModuleTextureRead} from './module'
import type {InputBuildContext, InputTexture} from './types'

/** The module context for the component an input belongs to. */
export function moduleFor(ctx: InputBuildContext): ModuleContext {
    return moduleContext({
        device: ctx.device,
        params: ctx.params,
        frame: ctx.frame,
        componentName: ctx.componentName,
        props: ctx.props,
        lib: ctx.lib,
        helpers: ctx.helpers,
        onCleanup: ctx.onCleanup,
    })
}

/** A uniform block of `count` vec4f an input writes each frame (its own arguments). */
export function uniformBlock(ctx: InputBuildContext, count: number, label: string): {buffer: GPUBuffer; write: (values: number[]) => void} {
    const bytes = Math.max(16, count * 16)
    const buffer = ctx.device.createBuffer({size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label})
    ctx.onCleanup(() => buffer.destroy())
    const data = new Float32Array(bytes / 4)
    return {
        buffer,
        write(values) {
            data.fill(0)
            data.set(values.slice(0, data.length))
            ctx.device.queue.writeBuffer(buffer, 0, data)
        },
    }
}

/**
 * A storage + sampled texture sized to the canvas times `scale` (or a fixed size), recreated
 * when the canvas resizes. `current()` always returns the live one.
 */
export function canvasTexture(
    ctx: InputBuildContext,
    opts: {scale?: number; fixed?: {width: number; height: number}; format?: GPUTextureFormat; label: string; max?: number; render?: boolean},
): InputTexture & {width: () => number; height: () => number} {
    const format = opts.format ?? 'rgba16float'
    const sizeFor = (canvas: {width: number; height: number}) => {
        if (opts.fixed) return opts.fixed
        const s = opts.scale ?? 1
        const max = opts.max ?? 4096
        return {
            width: Math.max(1, Math.min(max, Math.round(canvas.width * s))),
            height: Math.max(1, Math.min(max, Math.round(canvas.height * s))),
        }
    }
    let size = sizeFor(ctx.canvas())
    const usages = opts.render ? (['render'] as const) : []
    let tex = createStorageTexture(ctx.params, size, format, `${ctx.componentName}:${opts.label}`, usages)
    ctx.onCleanup(() => destroyTexture(tex))
    if (!opts.fixed) {
        ctx.onResize((canvas) => {
            const next = sizeFor(canvas)
            if (next.width === size.width && next.height === size.height) return
            const old = tex
            size = next
            tex = createStorageTexture(ctx.params, size, format, `${ctx.componentName}:${opts.label}`, usages)
            destroyTexture(old)
        })
    }
    return {
        current: () => tex,
        premultiplied: false,
        format,
        size: () => size,
        width: () => size.width,
        height: () => size.height,
    }
}

/** The WGSL text of a body argument (a `wgsl` body or a plain string). */
export function bodyText(body: WgslBody | string): string {
    return isWgslBody(body) ? body.spec.body : String(body)
}

/**
 * Reads of other inputs that this code calls by name (`soft(p)`, `field.slope(p)`), resolved
 * lazily so an input listed later is found once it is built.
 */
export function siblingReads(ctx: InputBuildContext, code: string): ModuleTextureRead[] {
    const ids = identifiersIn(code)
    const reads: ModuleTextureRead[] = []
    for (const [name, inst] of ctx.siblings()) {
        if (name === ctx.name || !ids.has(name)) continue
        for (const [out, tex] of Object.entries(inst.outputs)) {
            const call = out ? `${name}.${out}` : name
            if (out && !new RegExp(`(?<![\\w.])${name}\\s*\\.\\s*${out}\\b`).test(code)) continue
            reads.push({call, texture: tex, by: inst.readBy ?? 'uv'})
        }
    }
    return reads
}

/** Bounds-checked uv for the current invocation, given an output texture variable. */
export function texelPrelude(target: string): string {
    return `  let _size = textureDimensions(${target});
  if (gid.x >= _size.x || gid.y >= _size.y) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(_size);`
}
