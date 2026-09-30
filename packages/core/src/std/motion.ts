/**
 * std/motion — Words that turn a clock into rhythm: cycles, pulses, oscillations and easings.
 *
 * The clock is a number of seconds. Read the layer's own clock with `animatedTime(params)`
 * after declaring `animatedTime: {speed: 'speed'}` on the definition. Words like `cycle`
 * and `oscillate` take that clock and return a number you multiply into a color, a radius or
 * a coordinate. `oscillating` and `pulsing` do the same job for a prop-shaped slot on a warp
 * or an effect, so a knob can move on its own without any math in your definition.
 */
// Maintainer notes (not part of the reader-facing reference):
// - Two tiers. The EXPRESSION tier turns a clock Expr into rhythm — cycle phases, pulse
//   trains, oscillations, easings, per-cycle random seeds — for use anywhere algebra goes.
//   The SLOT tier (`oscillating`, `pulsing`) wraps the same shapes as `Scalar` signals, so
//   any ArgSpec slot on a warp or effect noun can be driven by time directly.
// - Slot signals read the global clock by default; `clock: 'node'` reads the definition's
//   `animatedTime` clock instead (declare `animatedTime: {speed: '...'}` — that makes the
//   signal speed-controllable and pausable from the panel).
import type {Expr} from '../gpu/contract'
import {animatedTime} from '../gpu/porters'
import {add, div, floor, fract, local, mul, smoothstep, sin, sub} from './math'
import {Scalar, type SignalSlotParams} from './values'

const TWO_PI = 6.283185307179586

/**
 * How far through a repeating cycle the clock is, 0..1.
 *
 * `period` is the cycle length in seconds. The value ramps from 0 to 1 and snaps back.
 *
 * @example
 * ```ts
 * const sweep = motion.phase(animatedTime(params), params.uniforms.period)
 * ```
 * @see cycle, oscillate
 */
export function phase(t: Expr, period: Expr | number): Expr {
    // fract(t / period)
    return fract(div(t, period))
}

/**
 * A repeating cycle as two numbers: `progress` (0..1 through the current cycle) and
 * `index` (which cycle this is).
 *
 * `period` is the cycle length in seconds. The "every N seconds" word: fade something in
 * with `progress`, and feed `index` to `cycleSeed` when each pass should look different.
 *
 * @example
 * ```ts
 * const {progress, index} = motion.cycle(animatedTime(params), 2) // a new pass every 2 s
 * ```
 * @see cycleSeed, phase, pulseTrain
 */
export function cycle(t: Expr, period: Expr | number): {progress: Expr; index: Expr} {
    const cycles = local(div(t, period), 'cycles')
    return {progress: local(fract(cycles), 'cycleT'), index: local(floor(cycles), 'cycleI')}
}

/**
 * A random but stable number, 0..1, for each cycle index.
 *
 * The same index always gives the same value, so a meteor keeps its angle for the whole
 * pass and picks a new one next time. Change `salt` for a second, independent draw.
 *
 * @example
 * ```ts
 * const hue = motion.cycleSeed(index)
 * const size = motion.cycleSeed(index, 1)
 * ```
 * @see cycle
 */
export function cycleSeed(index: Expr, salt = 0): Expr {
    return fract(mul(sin(add(mul(index, 12.9898), salt * 78.233 + 4.1414)), 43758.5453))
}

/**
 * An on/off rhythm: 1 for the first part of each cycle, 0 for the rest.
 *
 * `period` is seconds per cycle and `duty` is the on fraction, 0..1. `soften` eases both
 * edges over that fraction of the cycle. Leave it at 0 for a hard blink.
 *
 * @example
 * ```ts
 * const blink = motion.pulseTrain(animatedTime(params), {period: 1.5, duty: 0.3, soften: 0.05})
 * ```
 * @see pulsing, cycle, oscillate
 */
export function pulseTrain(t: Expr, opts: {period: Expr | number; duty: Expr | number; soften?: number}): Expr {
    const p = local(phase(t, opts.period), 'pulseT')
    const s = opts.soften ?? 0
    if (s <= 0) return sub(1, smoothstep(opts.duty, add(opts.duty, 0.0001), p))
    const rise = smoothstep(0, s, p)
    const fall = sub(1, smoothstep(sub(opts.duty, s), opts.duty, p))
    return mul(rise, fall)
}

/**
 * A smooth back-and-forth between `min` and `max`.
 *
 * `rate` is cycles per second, default 1. `min` and `max` default to 0 and 1. `offset`
 * shifts the starting point, in radians, so two oscillations can run out of step.
 *
 * @example
 * ```ts
 * const breathe = motion.oscillate(animatedTime(params), {rate: 0.5, min: 0.6, max: 1})
 * ```
 * @see oscillating, pulseTrain, phase
 */
export function oscillate(t: Expr, opts: {rate?: number; min?: Expr | number; max?: Expr | number; offset?: number}): Expr {
    const wave = add(mul(sin(add(mul(t, (opts.rate ?? 1) * TWO_PI), opts.offset ?? 0)), 0.5), 0.5)
    const lo = opts.min ?? 0
    const hi = opts.max ?? 1
    return add(lo, mul(wave, sub(hi, lo)))
}

// ── Easings (unit in → unit out) ─────────────────────────────────────────────────────────

/**
 * A 0..1 value that starts slow and speeds up.
 *
 * @example
 * ```ts
 * const grow = motion.easeIn(progress)
 * ```
 * @see easeOut, easeInOut
 */
export const easeIn = (x: Expr): Expr => mul(x, x)
/**
 * A 0..1 value that starts fast and slows to a stop.
 *
 * @example
 * ```ts
 * const settle = motion.easeOut(progress)
 * ```
 * @see easeIn, easeInOut
 */
export const easeOut = (x: Expr): Expr => sub(1, mul(sub(1, x), sub(1, x)))
/**
 * A 0..1 value that starts slow, speeds up, and slows again at the end.
 *
 * @example
 * ```ts
 * const glide = motion.easeInOut(progress)
 * ```
 * @see easeIn, easeOut
 */
export const easeInOut = (x: Expr): Expr => smoothstep(0, 1, x)

// ── Slot signals — drive any ArgSpec slot with time ──────────────────────────────────────

type SignalClock = 'global' | 'node'

function clockOf(clock: SignalClock | undefined, params: SignalSlotParams): Expr {
    // Warp hooks carry no ctx — signals there always ride the node's animated clock.
    if (clock === 'node' || !params.ctx) return animatedTime(params)
    return params.ctx.time
}

/**
 * A knob that sweeps between `min` and `max` on its own, for any slot that takes a prop.
 *
 * Same options as `oscillate`. By default it follows the page clock. Pass `clock: 'node'`
 * to follow the layer's own speed prop instead, after declaring `animatedTime` on the
 * definition.
 *
 * @example
 * ```ts
 * map: warps.twirl({center: p('center'), intensity: motion.oscillating({rate: 0.25, min: 0.4, max: 1.6, clock: 'node'})})
 * ```
 * @tip Inside `map:` it always follows the layer's clock, so declare `animatedTime: {speed: 'speed'}` there.
 * @see oscillate, pulsing
 */
export function oscillating(opts: {rate?: number; min?: Expr | number; max?: Expr | number; offset?: number; clock?: SignalClock}): Scalar {
    return new Scalar({kind: 'signal', build: (params) => oscillate(clockOf(opts.clock, params), opts)})
}

/**
 * A knob that switches on and off in a rhythm, for any slot that takes a prop.
 *
 * `period` is seconds per cycle, `duty` the on fraction, `soften` the edge ease, as in
 * `pulseTrain`. Follows the page clock unless `clock: 'node'` is set.
 *
 * @example
 * ```ts
 * effect: effects.color.saturate(motion.pulsing({period: 2, duty: 0.5, soften: 0.1})) // color, then gray, then color
 * ```
 * @tip Inside `map:` it always follows the layer's clock, so declare `animatedTime: {speed: 'speed'}` there.
 * @see pulseTrain, oscillating
 */
export function pulsing(opts: {period: number; duty: number; soften?: number; clock?: SignalClock}): Scalar {
    return new Scalar({kind: 'signal', build: (params) => pulseTrain(clockOf(opts.clock, params), opts)})
}
