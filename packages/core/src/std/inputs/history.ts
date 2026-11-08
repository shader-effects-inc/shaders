/**
 * `previousFrame`: what this layer drew last frame, read by name. Trails, afterimages, smears,
 * phosphor persistence, anything that builds on its own past: `max(last(uv) * 0.95, ink)`.
 */
// Maintainer notes. A component that lists `previousFrame` has its output body rendered to a
// texture (std/wgsl.ts wraps the body's colour in `convertToTexture` and records the boundary's key
// here). Each frame, before the render passes, a copy step moves that texture (still holding last
// frame's picture) into the history ring the body samples. Compute steps run before the render
// passes, so the copy always sees the previous frame.
import {selfOutput} from './self'
import {createComputeModule} from './module'
import {canvasTexture, moduleFor, texelPrelude} from './common'
import {withBuilder, type InputNode, type InputTexture} from './types'

export interface PreviousFrameInput extends InputNode {
    readonly kind: 'previousFrame'
    readonly frames: number
    readonly scale: number
}

/**
 * This layer's own output from an earlier frame. With no options it is last frame's picture;
 * `frames: n` reads the one from n frames ago (up to 8). `scale` keeps a smaller history, which
 * is cheaper and blurs slightly with every frame (often what a trail wants).
 *
 * Read it like any input: `last(uv)` gives straight-alpha colour. On the first frame it is
 * transparent.
 *
 * @example
 * ```ts
 * inputs: {last: previousFrame()},
 * paint: wgsl`
 *   let ink = 1.0 - smoothstep(0.0, 0.03, distance(uv, pointer));
 *   return max(last(uv) * fade, vec4f(ink));
 * `,
 * ```
 * @tip The whole body becomes one extra render pass, so a layer with a history costs a little more than one without.
 */
export function previousFrame(opts: {frames?: number; scale?: number} = {}): PreviousFrameInput {
    const frames = Math.max(1, Math.min(8, Math.round(opts.frames ?? 1)))
    const scale = Math.min(1, Math.max(0.1, opts.scale ?? 1))
    const node = {kind: 'previousFrame' as const, frames, scale}
    return withBuilder(node, (ctx) => {
        const self = selfOutput(ctx.params)
        const ring = Array.from({length: frames}, (_, i) => canvasTexture(ctx, {scale, label: `${ctx.name} (history ${i})`}))
        let write = 0
        const selfTexture: InputTexture = {
            current: () => self.texture,
            premultiplied: false,
            format: 'rgba16float',
            size: () => ctx.canvas(),
        }
        const mod = moduleFor(ctx)
        const copies = ring.map((target, i) => createComputeModule(mod, {
            label: `previousFrame '${ctx.name}' [${i}]`,
            segments: [],
            entry: `${texelPrelude('_out')}\n  textureStore(_out, gid.xy, _read__self(uv));`,
            resources: [{kind: 'storageTexture', name: '_out', format: 'rgba16float', texture: target.current}],
            textureReads: [{call: '_self', texture: selfTexture}],
            workgroup: [8, 8],
            includeShared: false,
        }).step(() => [target.width(), target.height()]))
        // The body reads the oldest slot; the copy overwrites it after the read, next frame.
        const oldest = (): InputTexture => ring[(write + 1) % frames]
        return {
            outputs: {
                '': {
                    current: () => oldest().current(),
                    premultiplied: false,
                    format: 'rgba16float',
                    size: () => oldest().size(),
                },
            },
            steps() {
                if (!self.texture) return []
                write = (write + 1) % frames
                return [copies[write]]
            },
            bindLate(resolve) {
                self.texture = self.key ? resolve(self.key)?.texture ?? null : null
            },
        }
    })
}
