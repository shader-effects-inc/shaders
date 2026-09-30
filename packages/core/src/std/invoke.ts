/**
 * Context values, and the helpers a hand-written `paint:` or `effect:` builder uses to read props and context on the GPU.
 *
 * `ctx` names the context values (`ctx.uv`, `ctx.time`, …) as tokens you can pass to a word
 * or bind in a `wgsl` body. The rest is for a builder function: `paintFrame` gives the
 * coordinate and frame size to draw in, `uniformOf` reads a prop, `resolveScalar` and
 * `resolveArg` turn any value a word accepts into a GPU value, and `pointwiseOp` wraps a
 * compiled GPU function as a pointwise filter effect.
 */
// Maintainer notes: the generic invocation layer. Vocabulary modules (std/effects/*,
// std/paint/*, std/warps/*) build named nouns on top of these helpers: an `ArgSpec` is
// declarative data describing one GPU-body argument (a prop binding, a scalar value graph, a
// context token, or a number literal), and `pointwiseOp` wires a kit body into the pointwise
// filter species from specs alone — so a noun like `saturate(p('intensity'))` is one line and
// carries no plumbing.
import type {Expr, GpuFragmentParams} from '../gpu/contract'
import {call, expr, floatE} from '../gpu/composer'
import {fields} from '../gpu/kit/index'
import type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import type {PointwiseEffect} from './types'
import {Scalar, type PropRef, type ScalarSource, type SignalSlotParams} from './values'

/** One of the context values as a token you can pass where a word takes a value: `ctx.uv`, `ctx.time`, … */
export interface CtxToken {
    readonly kind: 'ctx'
    readonly name: 'uv' | 'aspect' | 'time' | 'viewportSize' | 'logicalViewportSize' | 'pointer'
}

/**
 * The context values as tokens: `ctx.uv`, `ctx.time`, `ctx.aspect`, `ctx.pointer`, `ctx.viewportSize`, `ctx.logicalViewportSize`.
 *
 * Pass one where a word takes a value, or bind it under another name in a `wgsl` body's
 * `inputs`. `uv` is 0–1 across the canvas with y down. `time` is the global clock in seconds
 * and never pauses; for the layer's own speed-scaled clock declare `animatedTime` and read
 * `time` in a body or `animatedTime(params)` in a builder. `aspect` is width / height.
 * `pointer` is the mouse in uv. `viewportSize` is the frame in device pixels,
 * `logicalViewportSize` in CSS pixels.
 *
 * @example
 * ```ts
 * paint: wgsl({inputs: {t: ctx.time, px: ctx.viewportSize}, body: 'return vec4f(fract(uv * px / 40.0), sin(t) * 0.5 + 0.5, 1.0);'})
 * ```
 * @tip A warp's `map:` has no context; a `ctx` token there throws when the shader is built.
 * @see p, paintFrame, wgsl
 */
export const ctx = {
    /** The pixel's coordinate: 0–1 across the canvas, y down. */
    uv: {kind: 'ctx', name: 'uv'} as CtxToken,
    /** The canvas's width divided by its height. */
    aspect: {kind: 'ctx', name: 'aspect'} as CtxToken,
    /** The global clock in seconds. Never pauses; for the layer's own speed-scaled clock declare `animatedTime`. */
    time: {kind: 'ctx', name: 'time'} as CtxToken,
    /** The frame size in device pixels, as a `vec2f`. */
    viewportSize: {kind: 'ctx', name: 'viewportSize'} as CtxToken,
    /** The frame size in CSS pixels, as a `vec2f`, for feature counts that should not change with screen density. */
    logicalViewportSize: {kind: 'ctx', name: 'logicalViewportSize'} as CtxToken,
    /** The mouse position in uv (0–1, y down). */
    pointer: {kind: 'ctx', name: 'pointer'} as CtxToken,
} as const

/** Anything a word accepts as an input value: a prop ref `p('x')`, a `Scalar`, a context token, or a plain number. */
export type ArgSpec = PropRef | Scalar | CtxToken | number

/** The least a value needs to resolve against: the props, plus `ctx` where the host has one (a warp's `map:` does not). */
export type SlotParams = SignalSlotParams

/**
 * The coordinate and frame size a hand-written `paint:` builder should draw in.
 *
 * Returns `uv` and `viewport` as GPU values. Inside a distortion the uv is the distorted one,
 * and inside a sized box the viewport is the box, so a generator built on these behaves like
 * the library's own.
 *
 * @example
 * ```ts
 * paint: (params) => {
 *   const {uv} = paintFrame(params)
 *   const center = uniformOf(p('center'), params)
 *   const d = length(sub(uv, vec2(center.member('x'), sub(1, center.member('y')))))
 *   return vec4(splat3(smoothstep(0, 0.5, d)), 1)
 * }
 * ```
 * @see ctx, uniformOf
 */
// The uvContext idiom every generator paint shares: evaluate against the composed UV/viewport
// when a UV-propagating parent supplies one, else the raw canvas frame.
export function paintFrame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

/**
 * The live GPU value of a prop, for a hand-written builder: `uniformOf(p('radius'), params)`.
 *
 * Colors come back as `vec4f` in linear RGB with alpha, positions as `vec2f`, numbers as
 * `f32`. Read a part with `.member('rgb')` or `.member('x')`. Throws when the name is not one
 * of the definition's props.
 *
 * @example
 * ```ts
 * paint: (params) => {
 *   const color = uniformOf(p('color'), params)
 *   const gain = mul(uniformOf(p('intensity'), params), 0.5)
 *   return vec4(mul(color.member('rgb'), gain), color.member('a'))
 * }
 * ```
 * @tip A position prop's y is stored flipped: use `sub(1, center.member('y'))` to compare it with `uv`.
 * @see p, resolveScalar, resolveArg, paintFrame
 */
// Resolves a prop binding to its uniform accessor (throws on a dangling name).
export function uniformOf(ref: PropRef, params: {uniforms: Record<string, Expr>}): Expr {
    const accessor = params.uniforms[ref.name]
    if (!accessor) throw new Error(`std: effect binds unknown prop '${ref.name}'`)
    return accessor
}

/**
 * A `Scalar` or prop ref as a GPU value, for a hand-written builder.
 *
 * Products (`.times()`), radial masks and signals are built out here. A radial mask needs a
 * `paint:` or `effect:` host; it throws in a warp's `map:`.
 *
 * @example
 * ```ts
 * const amount = resolveScalar(radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity')), params)
 * ```
 * @see uniformOf, resolveArg, Scalar
 */
export function resolveScalar(input: Scalar | ScalarSource, params: SlotParams): Expr {
    const node: ScalarSource = input instanceof Scalar ? input.node : input
    switch (node.kind) {
        case 'prop':
            return uniformOf(node, params)
        case 'mul':
            return resolveScalar(node.a, params).mul(resolveScalar(node.b, params))
        case 'radialMask':
            if (!params.ctx) throw new Error('std: radialMask needs a fragment host (warp slots have no ctx)')
            return call(fields.radialFalloffMask, 'radialFalloffMask', [
                params.ctx.uv, params.ctx.aspect,
                uniformOf(node.center, params), uniformOf(node.radius, params), uniformOf(node.falloff, params),
            ])
        case 'signal':
            return node.build(params)
    }
}

/**
 * Any input value a word accepts (a prop ref, a `Scalar`, a context token, a number) as a GPU value inside a filter's builder.
 *
 * @example
 * ```ts
 * effect: pointwise({build: (params) => mix(params.childNode, vec4(0, 0, 0, 1), resolveArg(p('amount'), params))})
 * ```
 * @see resolveArgIn, uniformOf, resolveScalar
 */
export function resolveArg(spec: ArgSpec, params: FilterParams): Expr {
    return resolveArgIn(spec, params)
}

/**
 * `resolveArg` for any host, including a warp's `map:` factory.
 *
 * A warp has no context, so a `ctx` token or a radial mask throws there; props, scalars and
 * numbers resolve everywhere.
 *
 * @example
 * ```ts
 * map: (params) => ({map: (uv) => add(uv, mul(resolveArgIn(p('offset'), params), 0.1))})
 * ```
 * @see resolveArg
 */
export function resolveArgIn(spec: ArgSpec, params: SlotParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar) return resolveScalar(spec, params)
    switch (spec.kind) {
        case 'prop':
            return uniformOf(spec, params)
        case 'ctx':
            if (!params.ctx) throw new Error(`std: ctx.${spec.name} needs a fragment host (warp slots have no ctx)`)
            return params.ctx[spec.name]
        default:
            return resolveScalar(spec, params)
    }
}

/** Never-used guard so `expr` stays available to vocabulary modules re-exporting it. */
export {expr as rawExpr}

/**
 * Wrap a compiled GPU function as a pointwise filter effect: the child's color goes in first, then `args`.
 *
 * This is how the library's color words are made. `fn` takes `(color: vec4f, ...args)` and
 * returns the new color; `hint` names it in the generated code; `args` are the values that
 * follow the color, given as prop refs, scalars, context tokens or numbers. `extra.compose`
 * post-processes the result and `extra.setup` runs once per build.
 *
 * @example
 * ```ts
 * export function saturate(intensity: ArgSpec): PointwiseEffect {
 *   return pointwiseOp(colorOps.saturate, 'saturate', [intensity])
 * }
 * ```
 * @see resolveArg, p
 */
export function pointwiseOp(
    fn: unknown,
    hint: string,
    args: ArgSpec[],
    extra?: {compose?: PointwiseEffect['compose']; setup?: PointwiseEffect['setup']},
): PointwiseEffect {
    return {
        kind: 'pointwise',
        body: {fn, hint},
        args: (params) => args.map((spec) => resolveArg(spec, params)),
        compose: extra?.compose,
        setup: extra?.setup,
    }
}
