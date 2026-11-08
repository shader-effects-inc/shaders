/**
 * `pointerTrail`: where the pointer has been, and how recently, as a field. 1 where it is now,
 * fading to 0 behind it. Cursor heat, comet tails, wipe-to-reveal.
 */
import type {GridSimConfig} from '../sim'
import {buildGrid} from './grid'
import {withBuilder, type InputNode} from './types'

export interface PointerTrailInput extends InputNode {
    readonly kind: 'pointerTrail'
    readonly decay: number
    readonly radius: number
    readonly resolution: number
}

/**
 * The pointer's recent path as a field in `.x` (0–1), read by name: `trail(uv).x`. `decay` is
 * how many seconds a mark takes to halve; `radius` the brush size as a fraction of the height.
 *
 * @example
 * ```ts
 * inputs: {trail: pointerTrail({decay: 0.6, radius: 0.04})},
 * effect: wgsl`return mix(child, vec4f(1.0, 0.4, 0.1, 1.0), trail(uv).x);`,
 * ```
 */
export function pointerTrail(opts: {decay?: number; radius?: number; resolution?: number} = {}): PointerTrailInput {
    const node = {
        kind: 'pointerTrail' as const,
        decay: Math.max(0.01, opts.decay ?? 0.6),
        radius: Math.max(0.002, opts.radius ?? 0.04),
        resolution: Math.max(32, Math.min(1024, Math.round(opts.resolution ?? 256))),
    }
    return withBuilder(node, (ctx) => {
        const canvas = ctx.canvas()
        const aspect = canvas.width / Math.max(1, canvas.height)
        const config: GridSimConfig = {
            resolution: aspect >= 1 ? [node.resolution, Math.round(node.resolution / aspect)] : [Math.round(node.resolution * aspect), node.resolution],
            history: 1,
            derive: {},
            step: [{
                kind: 'op.kernel',
                iterations: 1,
                body: `let c = read(cell);
  let a = vec2f(f32(gridSize.x) / f32(gridSize.y), 1.0);
  let moving = select(0.0, 1.0, length(pointerVelocity) > 0.02 || pointerDown > 0.5);
  let d = length((cellUV - pointer) * a) / ${node.radius};
  let mark = (1.0 - smoothstep(0.6, 1.0, d)) * moving;
  return vec4f(max(c.x * exp2(-dt / ${node.decay}), mark), 0.0, 0.0, 1.0);`,
            }],
        }
        const inst = buildGrid(ctx, config)
        return inst ? {...inst, usesPointer: true} : null
    })
}
