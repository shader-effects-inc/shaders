/**
 * std/filter — words for the `effect:` field of a filter. A filter is a component that changes
 * the layer inside it (the child): `defineShader({name, props, effect: tintToward(p('color'),
 * {amount: 0.5})})`.
 *
 * There are two kinds of effect. A *pointwise* effect edits the child's color at this pixel
 * and nothing else, so it is cheap and keeps the child's alpha. A *gather* effect first draws
 * the child to a texture so it can read the child's other pixels too: neighbours for a blur,
 * a shifted position for a distortion. `tintToward` and `paintThrough` are pointwise;
 * `displaceBy` is a gather. `pointwise` and `gather` let you write your own of either kind.
 * The kind is inferred from the effect, so a definition needs no `role` or `species`.
 */
// Maintainer notes. Two tiers per species:
//  - Noun effects (`tintToward`, `displaceBy`) — pure vocabulary; the GPU body is a kit
//    primitive the lowering wires up, so the authored file contains no GPU code.
//  - `pointwise(...)` / `gather(...)` — the L1 tier for filters whose body is a blessed
//    bespoke `'use gpu'` fn or tap chain. Same species, same engine mechanics.
// Constructors capture data only; all mechanics (child guards, identity bypass with driver
// refusal, call wiring, the per-species alpha discipline) belong to the lowering.
import {call} from '../gpu/composer'
import {blend} from '../gpu/kit/index'
import type {Paint} from './paint/fields'
import {local, mul, vec4} from './math'
import type {GatherEffect, PointwiseEffect} from './types'
import type {PropRef, ScalarInput} from './values'

/** What `tintToward` returns: the color prop and the amount, ready for `effect:`. */
export interface TintTowardEffect {
    readonly kind: 'tintToward'
    readonly color: PropRef
    readonly amount: ScalarInput
}

/**
 * Mix the child's color toward one color by an amount, leaving its alpha alone.
 *
 * `color` is a color prop. `amount` runs 0 (no change) to 1 (fully that color) and can be a
 * number, a prop, or a value that varies per pixel such as a mask. Pointwise.
 *
 * @example
 * ```ts
 * effect: tintToward(p('color'), {amount: radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity'))})
 * ```
 * @tip Pair it with `identityWhen: isZero('intensity')` so the filter costs nothing at zero.
 * @see paintThrough, pointwise
 */
export function tintToward(color: PropRef, opts: {amount: ScalarInput}): TintTowardEffect {
    // Lowered to `colorMixing.mixToward` (linear-RGB mix of rgb, alpha carried through).
    return {kind: 'tintToward', color, amount: opts.amount}
}

/**
 * What `displaceBy` returns: the simulation field and the props it reads, ready for `effect:`.
 */
export interface DisplaceByEffect {
    readonly kind: 'displaceBy'
    readonly field: import('./sim').SimOutputRef
    readonly strength: PropRef
    readonly chromatic: PropRef
    /** A prop marked `compileTime: true` holding `'stretch'`, `'transparent'`, `'mirror'` or `'wrap'`. */
    readonly edges: PropRef
}

/**
 * Push the child's pixels around by a vector field from one of your simulations.
 *
 * `field` is a simulation output, such as `waves.output('displacement')` from
 * `simulate.grid`. `strength` scales the push (the library uses 0–20), capped at about 15%
 * of the canvas. `chromatic` (0–3) splits red and blue slightly apart from green along the
 * push, like a lens. `edges` is a prop marked `compileTime: true` holding `'stretch'`,
 * `'transparent'`, `'mirror'` or `'wrap'`. Gather.
 *
 * @example
 * ```ts
 * effect: displaceBy(waves.output('displacement'), {strength: p('intensity'), chromatic: p('chromaticSplit'), edges: p('edges')})
 * ```
 * @tip The simulation must derive the field with `op.gradient()`; the definition fails loudly otherwise.
 * @see gather
 */
export function displaceBy(
    field: import('./sim').SimOutputRef,
    opts: {strength: PropRef; chromatic: PropRef; edges: PropRef},
): DisplaceByEffect {
    // A gather effect: the child renders to a texture, taps are premultiplied, and the result is
    // straight alpha — the alpha discipline is the species', never the author's. The lowering
    // validates the sim's op set ([op.wave, op.splat], history 2, derive = op.gradient), wires
    // the kit wave-field runtime as the compute hook, and samples R/G/B at
    // `displace.chromaticDisplaceUVs` with per-tap edge handling.
    return {kind: 'displaceBy', field, strength: opts.strength, chromatic: opts.chromatic, edges: opts.edges}
}

/**
 * Write your own pointwise effect: a function of the child's color at this pixel.
 *
 * Give it `build`, which receives `params.childNode` (the child's color here) and
 * `params.uniforms` (your props), and returns the new color. Keep the child's alpha unless
 * you mean to change it. No texture is involved, so it is the cheapest kind of filter.
 *
 * @example
 * ```ts
 * effect: pointwise({build: ({childNode, uniforms}) => math.vec4(math.mul(childNode.member('rgb'), uniforms.gain), childNode.member('a'))})
 * ```
 * @see gather, tintToward, paintThrough
 */
export function pointwise(config: Omit<PointwiseEffect, 'kind'>): PointwiseEffect {
    // L1 tier: a pointwise filter around a blessed bespoke `'use gpu'` body. `body`/`args`/
    // `compose`/`setup` are the library-body form; `build` is the pure-Expr form. Exactly one of
    // `body`/`build` must be given (the scaffold throws otherwise).
    return {kind: 'pointwise', ...config}
}

/**
 * Write your own gather effect: a function that can read the child anywhere.
 *
 * Give it `build`, which receives the child as a texture (`params.texture`, or
 * `params.sampleStraight(uv)` for plain colors), `params.ctx.uv` for this pixel and
 * `params.uniforms` for your props. Return the new color. Set `resultAlpha: 'straight'` when
 * you built the color from `sampleStraight`. Costs one extra render of the child.
 *
 * @example
 * ```ts
 * effect: gather({resultAlpha: 'straight', build: ({sampleStraight, ctx, uniforms}) => sampleStraight(math.add(ctx.uv, uniforms.shift))})
 * ```
 * @tip Reach for `pointwise` first. Only gather when the effect needs pixels other than this one.
 * @see pointwise, displaceBy
 */
export function gather(config: Omit<GatherEffect, 'kind'>): GatherEffect {
    // L1 tier: a gather filter around a blessed bespoke tap chain over the child texture. Taps on
    // `texture` are premultiplied; the species appends the unpremultiply tail unless
    // `resultAlpha: 'straight'`. Identity bypasses to a centre sample, never to the raw child.
    return {kind: 'gather', ...config}
}

/**
 * Show a paint through the child, so the child becomes the mask.
 *
 * Like CSS `mask-image` the other way round: the paint is the content and the child decides
 * where it shows. `by: 'alpha'` (the default) reveals the paint wherever the child is opaque,
 * which gives a gradient through text. `by: 'luminance'` reveals it where the child is
 * bright. The paint's own alpha still applies. Pointwise.
 *
 * @example
 * ```ts
 * effect: paintThrough(paint.rampOver(paint.dist.radial({center: p('center'), radius: p('radius'), aspect: 1, skew: 0}), paint.pair(p('colorA'), p('colorB'), p('colorSpace'))))
 * ```
 * @see tintToward, pointwise
 */
export function paintThrough(paint: Paint, opts: {by?: 'alpha' | 'luminance'} = {}): PointwiseEffect {
    return {
        kind: 'pointwise',
        build: (params) => {
            const painted = local(paint(params), 'painted')
            const mask = opts.by === 'luminance'
                ? call(blend.luminance, 'luminance', [params.childNode.member('rgb')])
                : params.childNode.member('a')
            return vec4(painted.member('rgb'), mul(painted.member('a'), mask))
        },
    }
}
