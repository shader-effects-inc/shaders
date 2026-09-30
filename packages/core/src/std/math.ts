/**
 * std/math — Per-pixel arithmetic for building your own paint, mask or motion from numbers.
 *
 * Every word takes expressions or plain numbers and returns an expression the GPU evaluates
 * once per pixel. Props arrive as `params.uniforms.<name>`, coordinates come from a frame,
 * and the result flows into a color, a coverage value or another word. Use `local` for any
 * value you read more than once.
 */
// Maintainer notes (not part of the reader-facing reference):
// - This is the expression algebra: per-pixel math written in the language instead of
//   `'use gpu'` bodies. Every function builds an `Expr` from expressions and numbers,
//   lowering through the composer's emitter exactly like every other noun. `local()` binds a
//   shared sub-expression to one WGSL local so multi-consumer values evaluate once.
// - Numbers format as f32 literals; every compound operand is parenthesized, so composed
//   expressions never re-associate under WGSL precedence.
import {Expr, type EmitContext} from '../gpu/contract'
import {floatE, vec4 as vec4E, mixExpr, asLocal} from '../gpu/composer'

type Val = Expr | number

const toE = (v: Val): Expr => (typeof v === 'number' ? floatE(v) : v)
const emit = (v: Val, ctx: EmitContext): string => (typeof v === 'number' ? floatE(v)._emit(ctx) : v._emit(ctx))

/** Emit `name(args…)` for a WGSL builtin. */
function callB(name: string, ...args: Val[]): Expr {
    return new Expr((ctx) => `${name}(${args.map((a) => emit(a, ctx)).join(', ')})`)
}

function binary(op: string, a: Val, b: Val): Expr {
    return new Expr((ctx) => `(${emit(a, ctx)} ${op} ${emit(b, ctx)})`)
}

// ── Arithmetic ──────────────────────────────────────────────────────────────────────────

/**
 * The sum of two values.
 *
 * Works on numbers and on vectors of the same size.
 *
 * @example
 * ```ts
 * const lifted = add(mul(noise, 0.5), 0.5) // a −1..1 signal moved into 0..1
 * ```
 * @see sub, mul
 */
export const add = (a: Val, b: Val): Expr => binary('+', a, b)
/**
 * The first value minus the second.
 *
 * @example
 * ```ts
 * const outside = sub(1, coverage) // flip a 0..1 mask
 * ```
 * @see add, neg
 */
export const sub = (a: Val, b: Val): Expr => binary('-', a, b)
/**
 * The product of two values.
 *
 * Multiply a color by a 0..1 mask to fade it, or a coordinate by a prop to scale it.
 *
 * @example
 * ```ts
 * const rgb = mul(params.uniforms.color.member('rgb'), coverage)
 * ```
 * @see div, add
 */
export const mul = (a: Val, b: Val): Expr => binary('*', a, b)
/**
 * The first value divided by the second.
 *
 * Dividing by zero produces infinity on the GPU, so guard a prop with `max` first.
 *
 * @example
 * ```ts
 * const normalized = div(dist, max(params.uniforms.radius, 0.001))
 * ```
 * @see mul, max
 */
export const div = (a: Val, b: Val): Expr => binary('/', a, b)
/**
 * The value with its sign flipped.
 *
 * @example
 * ```ts
 * const falloff = exp(neg(mul(dist, 4)))
 * ```
 * @see sub, abs
 */
export const neg = (a: Val): Expr => new Expr((ctx) => `(-(${emit(a, ctx)}))`)

// ── WGSL builtins ───────────────────────────────────────────────────────────────────────

/**
 * The value without its sign.
 *
 * Turns a signed distance into a distance from the line, so a stroke covers both sides.
 *
 * @example
 * ```ts
 * const fromLine = abs(sub(uv.member('y'), 0.5))
 * ```
 * @see sign, neg
 */
export const abs = (x: Val): Expr => callB('abs', x)
/**
 * The value rounded down to a whole number.
 *
 * Gives the cell index when you tile a coordinate.
 *
 * @example
 * ```ts
 * const column = floor(mul(uv.member('x'), params.uniforms.columns))
 * ```
 * @see fract, ceil
 */
export const floor = (x: Val): Expr => callB('floor', x)
/**
 * The value rounded up to a whole number.
 *
 * @example
 * ```ts
 * const rows = ceil(div(height, cell))
 * ```
 * @see floor
 */
export const ceil = (x: Val): Expr => callB('ceil', x)
/**
 * The fractional part of a value, 0..1.
 *
 * Repeats a coordinate every whole unit, which is how a pattern tiles.
 *
 * @example
 * ```ts
 * const inCell = fract(mul(uv, params.uniforms.tiles)) // 0..1 inside each tile
 * ```
 * @see floor
 */
export const fract = (x: Val): Expr => callB('fract', x)
/**
 * The square root of a value.
 *
 * @example
 * ```ts
 * const dist = sqrt(add(mul(dx, dx), mul(dy, dy)))
 * ```
 * @see length, pow
 */
export const sqrt = (x: Val): Expr => callB('sqrt', x)
/**
 * The first value raised to the power of the second.
 *
 * Shapes a 0..1 value: powers above 1 push it toward 0, powers below 1 toward 1. Keep the
 * base non-negative.
 *
 * @example
 * ```ts
 * const hardened = pow(coverage, params.uniforms.hardness)
 * ```
 * @see sqrt, exp
 */
export const pow = (x: Val, y: Val): Expr => callB('pow', x, y)
/**
 * e raised to the value.
 *
 * With a negative argument it is the classic smooth falloff.
 *
 * @example
 * ```ts
 * const glow = exp(neg(mul(dist, params.uniforms.falloff)))
 * ```
 * @see exp2, log, gaussBell
 */
export const exp = (x: Val): Expr => callB('exp', x)
/**
 * The natural logarithm of a value.
 *
 * @example
 * ```ts
 * const octave = log(max(scale, 1))
 * ```
 * @see exp
 */
export const log = (x: Val): Expr => callB('log', x)
/**
 * The sine of an angle in radians, −1..1.
 *
 * @example
 * ```ts
 * const wave = sin(add(mul(uv.member('x'), 12), time))
 * ```
 * @tip A degrees prop needs `mul(angle, 0.017453292519943295)` first.
 * @see cos, tan
 */
export const sin = (x: Val): Expr => callB('sin', x)
/**
 * The cosine of an angle in radians, −1..1.
 *
 * @example
 * ```ts
 * const spun = rotate2(delta, cos(angle), sin(angle))
 * ```
 * @tip A degrees prop needs `mul(angle, 0.017453292519943295)` first.
 * @see sin, rotate2
 */
export const cos = (x: Val): Expr => callB('cos', x)
/**
 * The tangent of an angle in radians.
 *
 * @example
 * ```ts
 * const slope = tan(mul(params.uniforms.tilt, 0.017453292519943295))
 * ```
 * @see sin, cos, atan2
 */
export const tan = (x: Val): Expr => callB('tan', x)
/**
 * The angle of the point (x, y) in radians, −π..π.
 *
 * Takes y first. Turns an offset from a center into an angle around it.
 *
 * @example
 * ```ts
 * const angle = atan2(delta.member('y'), delta.member('x'))
 * ```
 * @see sin, cos
 */
export const atan2 = (y: Val, x: Val): Expr => callB('atan2', y, x)
/**
 * The smaller of two values.
 *
 * @example
 * ```ts
 * const capped = min(brightness, 1)
 * ```
 * @see max, clamp
 */
export const min = (a: Val, b: Val): Expr => callB('min', a, b)
/**
 * The larger of two values.
 *
 * @example
 * ```ts
 * const safeRadius = max(params.uniforms.radius, 0.001)
 * ```
 * @see min, clamp
 */
export const max = (a: Val, b: Val): Expr => callB('max', a, b)
/**
 * The value limited to the range `lo`..`hi`.
 *
 * @example
 * ```ts
 * const alpha = clamp(add(body, ring), 0, 1)
 * ```
 * @see min, max
 */
export const clamp = (x: Val, lo: Val, hi: Val): Expr => callB('clamp', x, lo, hi)
/**
 * A blend from `a` to `b` by `t`, where 0 gives `a` and 1 gives `b`.
 *
 * Works on numbers and on colors. For two colors, pass `t` through `splat3` to blend every
 * channel by the same amount.
 *
 * @example
 * ```ts
 * const rgb = mix(params.uniforms.colorA.member('rgb'), params.uniforms.colorB.member('rgb'), splat3(coverage))
 * ```
 * @see smoothstep, splat3
 */
export const mix = (a: Val, b: Val, t: Val): Expr => mixExpr(toE(a), toE(b), typeof t === 'number' ? t : t)
/**
 * A hard edge: 0 when `x` is below `edge`, 1 from `edge` on.
 *
 * @example
 * ```ts
 * const lit = step(params.uniforms.threshold, brightness)
 * ```
 * @see smoothstep
 */
export const step = (edge: Val, x: Val): Expr => callB('step', edge, x)
/**
 * A soft edge: 0 below `e0`, 1 above `e1`, an S-curve between.
 *
 * The everyday word for turning a distance into a soft mask. Swap `e0` and `e1` to fade the
 * other way.
 *
 * @example
 * ```ts
 * const coverage = sub(1, smoothstep(radius, add(radius, softness), dist)) // 1 inside, fading out
 * ```
 * @see step, mix
 */
export const smoothstep = (e0: Val, e1: Val, x: Val): Expr => callB('smoothstep', e0, e1, x)
/**
 * −1, 0 or 1 by the sign of the value.
 *
 * @example
 * ```ts
 * const side = sign(sub(uv.member('x'), 0.5))
 * ```
 * @see abs
 */
export const sign = (x: Val): Expr => callB('sign', x)
/**
 * The length of a vector.
 *
 * The distance from the origin, or from a center when you pass the offset from it.
 *
 * @example
 * ```ts
 * const dist = length(sub(uv, params.uniforms.center))
 * ```
 * @see distance, normalize
 */
export const length = (v: Val): Expr => callB('length', v)
/**
 * The distance between two points.
 *
 * @example
 * ```ts
 * const toPointer = distance(uv, params.ctx.pointer)
 * ```
 * @see length
 */
export const distance = (a: Val, b: Val): Expr => callB('distance', a, b)
/**
 * The dot product of two vectors.
 *
 * With two unit vectors it is the cosine of the angle between them, which is how a light
 * direction meets a surface normal.
 *
 * @example
 * ```ts
 * const facing = max(dot(normal, lightDir), 0)
 * ```
 * @see normalize, cross
 */
export const dot = (a: Val, b: Val): Expr => callB('dot', a, b)
/**
 * The vector scaled to length 1.
 *
 * A zero vector gives NaN, so keep the input non-zero.
 *
 * @example
 * ```ts
 * const dir = normalize(sub(params.uniforms.target, params.uniforms.origin))
 * ```
 * @see length, dot
 */
export const normalize = (v: Val): Expr => callB('normalize', v)
/**
 * The cross product of two 3D vectors.
 *
 * @example
 * ```ts
 * const normal = normalize(cross(tangentX, tangentY))
 * ```
 * @see dot, normalize
 */
export const cross = (a: Val, b: Val): Expr => callB('cross', a, b)
/**
 * A 2D point rotated around the origin.
 *
 * Pass the cosine and sine of the angle in radians, computed once, and the same pair
 * rotates every point.
 *
 * @example
 * ```ts
 * const spun = local(rotate2(delta, cos(angle), sin(angle)), 'spun')
 * ```
 * @see cos, sin
 */
export const rotate2 = (p: Val, cosA: Val, sinA: Val): Expr =>
    vec2(sub(mul(member2(p, 'x'), cosA), mul(member2(p, 'y'), sinA)),
        add(mul(member2(p, 'x'), sinA), mul(member2(p, 'y'), cosA)))
const member2 = (v: Val, m: 'x' | 'y'): Val => (typeof v === 'number' ? v : v.member(m))

/**
 * The direction `i` bounced off a surface with unit normal `n`.
 *
 * @example
 * ```ts
 * const bounce = reflect(viewDir, normal)
 * ```
 * @see refract, normalize
 */
export const reflect = (i: Val, n: Val): Expr => callB('reflect', i, n)
/**
 * The direction `i` bent through a surface with unit normal `n`.
 *
 * `eta` is the ratio of refractive indices, about 0.66 going from air into glass.
 *
 * @example
 * ```ts
 * const bent = refract(viewDir, normal, 0.66)
 * ```
 * @see reflect
 */
export const refract = (i: Val, n: Val, eta: Val): Expr => callB('refract', i, n, eta)
/**
 * 2 raised to the value.
 *
 * @example
 * ```ts
 * const octaveScale = exp2(octave)
 * ```
 * @see exp
 */
export const exp2 = (x: Val): Expr => callB('exp2', x)

/**
 * A random but stable angle in radians, 0..2π, from a seed.
 *
 * Use it to start each of several sine motions at a different point so they do not move in
 * step. `salt` is a pair of constants. Give each independent phase its own pair.
 *
 * @example
 * ```ts
 * const wobble = sin(add(mul(t, 0.7), hashPhase(params.uniforms.seed, {k: 53.7, big: 3847.2})))
 * ```
 * @see sin
 */
export const hashPhase = (seed: Val, salt: {k: number; big: number}): Expr =>
    mul(fract(mul(sin(mul(seed, salt.k)), salt.big)), 6.283185307179586)
// Maintainer note: `fract(sin(seed·k)·big)·τ`, the standard cheap decorrelator for sine
// schedules. The salt is the raw `(k, big)` pair so ported schedules keep their exact
// historical constants.

// ── Comparison & selection ──────────────────────────────────────────────────────────────

/**
 * True when `a` is less than `b`.
 *
 * A condition for `select`.
 *
 * @example
 * ```ts
 * const rgb = select(lt(dist, radius), insideRgb, outsideRgb)
 * ```
 * @see le, gt, ge, select
 */
export const lt = (a: Val, b: Val): Expr => binary('<', a, b)
/**
 * True when `a` is less than or equal to `b`.
 *
 * @example
 * ```ts
 * const inBand = select(le(abs(d), width), 1, 0)
 * ```
 * @see lt, gt, ge, select
 */
export const le = (a: Val, b: Val): Expr => binary('<=', a, b)
/**
 * True when `a` is greater than `b`.
 *
 * @example
 * ```ts
 * const highlights = select(gt(luma, 0.5), luma, 0)
 * ```
 * @see ge, lt, le, select
 */
export const gt = (a: Val, b: Val): Expr => binary('>', a, b)
/**
 * True when `a` is greater than or equal to `b`.
 *
 * @example
 * ```ts
 * const rightHalf = select(ge(uv.member('x'), 0.5), 1, 0)
 * ```
 * @see gt, lt, le, select
 */
export const ge = (a: Val, b: Val): Expr => binary('>=', a, b)
/**
 * `ifTrue` when the condition holds, otherwise `ifFalse`.
 *
 * Both values are computed at every pixel, so this picks rather than skips work. For a soft
 * transition use `mix` or `smoothstep` instead.
 *
 * @example
 * ```ts
 * const rgb = select(lt(dist, radius), params.uniforms.fill.member('rgb'), params.uniforms.background.member('rgb'))
 * ```
 * @see lt, gt, mix
 */
export const select = (cond: Expr, ifTrue: Val, ifFalse: Val): Expr =>
    new Expr((ctx) => `select(${emit(ifFalse, ctx)}, ${emit(ifTrue, ctx)}, ${emit(cond, ctx)})`)
// Maintainer note: WGSL `select` takes the false value first — the argument swap is handled here.

// ── Constructors & structure ────────────────────────────────────────────────────────────

/**
 * A plain number as an expression.
 *
 * Rarely needed: every word here accepts numbers directly. Reach for it when something
 * wants an expression and you only have a constant.
 *
 * @example
 * ```ts
 * const half = float(0.5)
 * ```
 * @see vec2, vec3, vec4
 */
export const float = (n: number): Expr => floatE(n)
/**
 * A 2D vector from two values.
 *
 * @example
 * ```ts
 * const offset = vec2(params.uniforms.shiftX, params.uniforms.shiftY)
 * ```
 * @see vec3, vec4
 */
export const vec2 = (x: Val, y: Val): Expr => callB('vec2f', x, y)
/**
 * A 3D vector from three values, often a color without alpha.
 *
 * @example
 * ```ts
 * const warmWhite = vec3(1, 0.96, 0.9)
 * ```
 * @see vec2, vec4, splat3
 */
export const vec3 = (x: Val, y: Val, z: Val): Expr => callB('vec3f', x, y, z)
/**
 * A 4D vector, usually the final color with alpha.
 *
 * Takes four numbers, or a vec3 followed by alpha. Return one of these from `paint:`.
 *
 * @example
 * ```ts
 * return vec4(rgb, coverage)
 * ```
 * @see vec3, vec2
 */
export const vec4 = (...parts: Val[]): Expr => vec4E(...parts.map(toE))
/**
 * One number copied into all three channels of a vec3.
 *
 * The usual way to blend two colors by a single amount with `mix`.
 *
 * @example
 * ```ts
 * const rgb = mix(darkRgb, lightRgb, splat3(brightness))
 * ```
 * @see mix, vec3
 */
export const splat3 = (x: Val): Expr => callB('vec3f', x)

/**
 * A value computed once per pixel and shared by everything that reads it.
 *
 * Wrap any expression you use twice or more. `hint` is a short name that appears in the
 * generated code and only has to be readable.
 *
 * @example
 * ```ts
 * const dist = local(length(sub(uv, params.uniforms.center)), 'dist')
 * ```
 * @tip Without it, the same math is repeated at every place you use the value.
 */
export const local = (e: Expr, hint: string): Expr => asLocal(e, hint)
// Maintainer note: binds the expression to one WGSL `let` so multi-consumer values evaluate once.

// ── Derived (general shapes used across the catalog) ───────────────────────────────────

/**
 * A soft hinge: about 0 for values well below zero, about `x` above, rounded near zero.
 *
 * `k` is the width of the rounded corner. Use it where a hard `max(x, 0)` would show a
 * visible crease.
 *
 * @example
 * ```ts
 * const past = local(softPlus(sub(beam.along, beam.length), mul(softness, 0.6)), 'past')
 * ```
 * @see max, smoothstep
 */
export const softPlus = (x: Expr, k: Val): Expr => {
    // Smooth relu: (x + sqrt(x² + k²)) / 2 — the standard soft elbow.
    const xe = local(x, 'sp')
    return mul(add(xe, sqrt(add(mul(xe, xe), mul(k, k)))), 0.5)
}

/**
 * A bell curve: 1 at zero, falling toward 0 on both sides.
 *
 * `k` sets the width. Larger values give a narrower bell. Pass a distance or a centered
 * coordinate as `x`.
 *
 * @example
 * ```ts
 * const profile = gaussBell(div(beam.across, beamWidth), 2.5)
 * ```
 * @see exp, smoothstep
 */
export const gaussBell = (x: Expr, k: Val): Expr => exp(neg(mul(mul(x, x), k)))
// Maintainer note: `exp(−k·x²)` over a normalized coordinate.
