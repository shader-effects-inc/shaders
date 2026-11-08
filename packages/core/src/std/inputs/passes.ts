/**
 * Full-screen passes as inputs: `pass` runs your own WGSL into a texture (at a reduced
 * resolution if you like), `downsample` makes a cheap low-resolution copy of a source, and
 * `directionalBlur` smears a source along a direction.
 */
import type {WgslBody} from '../wgsl'
import {appendCallArgs, createComputeModule, rewriteForModule, type ModuleTextureRead} from './module'
import {bodyText, canvasTexture, moduleFor, siblingReads, texelPrelude, uniformBlock} from './common'
import {bufferOf, structWgsl} from './compute'
import {identifiersIn} from '../wgslSource'
import {
    scalarValue,
    withBuilder,
    type InputNode,
    type InputSource,
    type ScalarArg,
} from './types'

/** The formats a pass may write. `rgba16float` keeps HDR values; `rgba8unorm` halves the memory. */
export type PassFormat = 'rgba16float' | 'rgba8unorm' | 'r32float'

export interface PassInput extends InputNode {
    readonly kind: 'pass'
    readonly body: string
    readonly scale: number
    readonly format: PassFormat
}

/**
 * Your own WGSL, rendered into a texture every frame before the body runs, read by name:
 * `clouds(uv)`. The body is like a `paint:` body (it gets `uv`, `time`, props, other inputs,
 * library functions, and returns a `vec4f`), but it runs in a compute pass: no `dpdx`, `fwidth`
 * or `discard`, and `textureSample` becomes `textureSampleLevel` for you. Reading `child` gives
 * the layer inside at this `uv`.
 *
 * `scale` renders at a fraction of the canvas (0.5 is a quarter of the pixels). Use it for
 * expensive work that is smooth anyway: volumetric clouds, a soft height field. `targetSize`
 * (vec2f) is this pass's own size in pixels; use it, not `viewport`, for anything measured in
 * pixels here (one pixel is `1.0 / targetSize.y` in uv).
 *
 * @example
 * ```ts
 * inputs: {clouds: pass(wgsl`return marchClouds(uv, time);`, {scale: 0.5})},
 * paint: wgsl`return clouds(uv);`,
 * ```
 * @see downsample, previousFrame
 */
export function pass(body: WgslBody | string, opts: {scale?: number; format?: PassFormat} = {}): PassInput {
    const text = bodyText(body)
    const scale = Math.min(1, Math.max(0.05, opts.scale ?? 1))
    const format = opts.format ?? 'rgba16float'
    const readsChild = /(?<![\w.])child(?![\w])/.test(text)
    const node = {kind: 'pass' as const, body: text, scale, format, ...(readsChild ? {readsChild: true} : {})}
    return withBuilder(node, (ctx) => {
        const target = canvasTexture(ctx, {scale, format, label: `${ctx.name} (pass)`})
        const reads: ModuleTextureRead[] = siblingReads(ctx, text)
        let prelude = ''
        if (readsChild) {
            const childTex = ctx.source({kind: 'child'})
            if (!childTex) {
                ctx.report('reads `child` but nothing is nested inside this layer')
                return null
            }
            reads.push({call: '_child', texture: childTex})
            prelude = '  let child = _child(uv);\n'
        }
        const mod = moduleFor(ctx)
        // Buffers the body names (`flies[i]`) are bound read-only, with their struct types.
        const ids = identifiersIn(text)
        const used = [...ctx.siblings()].map(([name, inst]) => [name, bufferOf(inst)] as const).filter(([name, b]) => b && ids.has(name))
        const structDecls = [...new Map(used.map(([, b]) => [b!.type.name, structWgsl(b!.type)])).values()].join('\n')
        // `targetSize`: this pass's own size in pixels (the canvas times `scale`), for pixel maths.
        const fnText = `fn _pass_${ctx.name.replace(/\W/g, '_')}(uv: vec2f, targetSize: vec2f) -> vec4f {\n${prelude}${text}\n}`
        const module = createComputeModule(mod, {
            label: `pass '${ctx.name}'`,
            decls: structDecls,
            segments: [{label: `pass '${ctx.name}'`, text: rewriteForModule(fnText, mod, reads), offset: 1 + (prelude ? 1 : 0)}],
            entry: `${texelPrelude('_out')}\n  textureStore(_out, gid.xy, _pass_${ctx.name.replace(/\W/g, '_')}(uv, vec2f(_size)));`,
            resources: [
                {kind: 'storageTexture', name: '_out', format, texture: target.current},
                ...used.map(([name, b]) => ({kind: 'storage' as const, name, type: `array<${b!.type.name}>`, access: 'read' as const, buffer: b!.buffer})),
            ],
            textureReads: reads,
            workgroup: [8, 8],
        })
        const step = module.step(() => [target.width(), target.height()])
        return {
            outputs: {'': target},
            steps: () => [step],
        }
    })
}

export interface DownsampleInput extends InputNode {
    readonly kind: 'downsample'
    readonly source: InputSource
    readonly scale: number
}

/**
 * A smaller copy of a source, box-filtered so it does not alias. Read it by name. Use it to
 * make an expensive read cheap: sample the small copy many times instead of the full one.
 *
 * @example
 * ```ts
 * inputs: {small: downsample(child, {scale: 0.25})},
 * ```
 * @see blur, pass
 */
export function downsample(source: InputSource, opts: {scale: number}): DownsampleInput {
    const scale = Math.min(1, Math.max(0.02, opts.scale))
    const node = {kind: 'downsample' as const, source, scale}
    return withBuilder(node, (ctx) => {
        const input = ctx.source(source)
        if (!input) return null
        const target = canvasTexture(ctx, {scale, label: `${ctx.name} (downsample)`})
        const mod = moduleFor(ctx)
        const reads: ModuleTextureRead[] = [{call: '_src', texture: input}]
        const module = createComputeModule(mod, {
            label: `downsample '${ctx.name}'`,
            segments: [],
            entry: `${texelPrelude('_out')}
  let texel = 0.25 / vec2f(_size);
  let c = (_read__src(uv + vec2f(-texel.x, -texel.y)) + _read__src(uv + vec2f(texel.x, -texel.y))
         + _read__src(uv + vec2f(-texel.x, texel.y)) + _read__src(uv + vec2f(texel.x, texel.y))) * 0.25;
  textureStore(_out, gid.xy, c);`,
            resources: [{kind: 'storageTexture', name: '_out', format: 'rgba16float', texture: target.current}],
            textureReads: reads,
            workgroup: [8, 8],
            includeShared: false,
        })
        const step = module.step(() => [target.width(), target.height()])
        return {outputs: {'': target}, steps: () => (input.current() ? [step] : [])}
    })
}

export interface DirectionalBlurInput extends InputNode {
    readonly kind: 'directionalBlur'
    readonly source: InputSource
    readonly angle: ScalarArg
    readonly length: ScalarArg
}

/**
 * A source smeared along a direction: `angle` in degrees (0 points right, 90 down), `length` in
 * canvas pixels. 32 weighted taps along the streak, at half resolution.
 *
 * @example
 * ```ts
 * inputs: {streak: directionalBlur(child, {angle: p('angle'), length: 60})},
 * ```
 * @see blur
 */
export function directionalBlur(source: InputSource, opts: {angle: ScalarArg; length: ScalarArg}): DirectionalBlurInput {
    const node = {kind: 'directionalBlur' as const, source, angle: opts.angle, length: opts.length}
    return withBuilder(node, (ctx) => {
        const input = ctx.source(source)
        if (!input) return null
        const target = canvasTexture(ctx, {scale: 0.5, label: `${ctx.name} (streak)`})
        const args = uniformBlock(ctx, 1, `${ctx.name}:args`)
        const mod = moduleFor(ctx)
        const module = createComputeModule(mod, {
            label: `directionalBlur '${ctx.name}'`,
            segments: [],
            entry: `${texelPrelude('_out')}
  let dir = vec2f(cos(_args.x), sin(_args.x)) * _args.y / max(_f.viewport, vec2f(1.0));
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var i = 0; i < 32; i++) {
    let t = (f32(i) + 0.5) / 32.0 - 0.5;
    let w = exp(-t * t * 8.0);
    acc += _read__src(uv + dir * t) * w;
    wsum += w;
  }
  textureStore(_out, gid.xy, acc / wsum);`,
            resources: [
                {kind: 'storageTexture', name: '_out', format: 'rgba16float', texture: target.current},
                {kind: 'uniform', name: '_args', type: 'vec4f', buffer: () => args.buffer},
            ],
            textureReads: [{call: '_src', texture: input}],
            workgroup: [8, 8],
            includeShared: false,
        })
        const step = module.step(() => [target.width(), target.height()])
        return {
            outputs: {'': target},
            steps() {
                if (!input.current()) return []
                const angle = (scalarValue(node.angle, ctx.params, 0) * Math.PI) / 180
                args.write([angle, scalarValue(node.length, ctx.params, 40)])
                return [step]
            },
        }
    })
}

// Re-exported for the body lowering, which rewrites `textureSample` the same way in passes.
export {appendCallArgs}
