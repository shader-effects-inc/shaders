/**
 * Discrete events as an input: the most recent presses on the canvas, each with where it
 * happened, how long ago, and a random seed of its own. Overlapping transients (click ripples,
 * bursts, rain on a puddle) need a queue, which a boolean prop cannot be.
 */
import {createStorageTexture, destroyTexture} from './module'
import {withBuilder, type InputNode} from './types'

export interface EventsInput extends InputNode {
    readonly kind: 'events'
    readonly source: 'click'
    readonly max: number
    readonly maxAge: number
}

function toHalf(v: number): number {
    // IEEE 754 binary16 from a JS number (round to nearest, no NaN payloads needed here).
    const f = new Float32Array([v])
    const i = new Uint32Array(f.buffer)[0]
    const sign = (i >> 16) & 0x8000
    let exp = ((i >> 23) & 0xff) - 127 + 15
    let mant = i & 0x7fffff
    if (exp <= 0) return sign
    if (exp >= 31) return sign | 0x7c00
    mant += 0x1000
    if (mant & 0x800000) {
        mant = 0
        exp += 1
        if (exp >= 31) return sign | 0x7c00
    }
    return sign | (exp << 10) | (mant >> 13)
}

/**
 * The last `max` presses (up to 32) on the canvas within `maxAge` seconds, read by slot:
 * `clicks(i)` is a `vec4f` of (x, y in uv, age in seconds, seed 0–1). Empty slots have age -1.
 * Slot 0 is the newest.
 *
 * @example
 * ```ts
 * inputs: {clicks: events('click', {max: 8})},
 * paint: wgsl`
 *   var c = 0.0;
 *   for (var i = 0; i < 8; i++) {
 *     let e = clicks(i);
 *     if (e.z < 0.0) { continue; }
 *     let ring = abs(distance(uv, e.xy) - e.z * 0.4);
 *     c += (1.0 - smoothstep(0.0, 0.01, ring)) * exp(-e.z * 2.0);
 *   }
 *   return vec4f(vec3f(c), 1.0);
 * `,
 * ```
 */
export function events(source: 'click', opts: {max?: number; maxAge?: number} = {}): EventsInput {
    const max = Math.max(1, Math.min(32, Math.round(opts.max ?? 8)))
    const maxAge = Math.max(0.1, opts.maxAge ?? 4)
    const node = {kind: 'events' as const, source, max, maxAge, readBy: 'index' as const}
    return withBuilder(node, (ctx) => {
        const tex = createStorageTexture(ctx.params, {width: max, height: 1}, 'rgba16float', `${ctx.name} (events)`)
        ctx.onCleanup(() => destroyTexture(tex))
        const raw = ctx.root.unwrap(tex as never) as unknown as GPUTexture
        const data = new Uint16Array(max * 4)
        return {
            outputs: {'': {current: () => tex, premultiplied: false, format: 'rgba16float', size: () => ({width: max, height: 1})}},
            usesPointer: true,
            readBy: 'index',
            steps() {
                const presses = ctx.frame.presses(maxAge)
                for (let i = 0; i < max; i++) {
                    const e = presses[presses.length - 1 - i]
                    const at = i * 4
                    if (e) {
                        data[at] = toHalf(e.x)
                        data[at + 1] = toHalf(e.y)
                        data[at + 2] = toHalf(e.age)
                        data[at + 3] = toHalf(e.seed)
                    } else {
                        data[at] = 0
                        data[at + 1] = 0
                        data[at + 2] = toHalf(-1)
                        data[at + 3] = 0
                    }
                }
                ctx.device.queue.writeTexture({texture: raw}, data, {bytesPerRow: max * 8}, {width: max, height: 1})
                return []
            },
        }
    })
}
