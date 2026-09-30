/**
 * std/paint/compose — stacking paints and finishing the output.
 *
 * `layers` stacks several colors into one, back to front, each with an optional opacity and
 * an optional "seen through" factor. `layered` builds and sums N variations of one thing.
 * `emissiveAlpha` turns light into a color with alpha so it composites over transparency, and
 * `dithered` hides banding on long soft ramps. These close a `paint:` function.
 */
// Maintainer notes (not part of the reference):
//  - `layers` is the declarative composite. Additive is the default (emissive light); `screen`
//    soft-saturates instead of summing. The alternative — a hand-balanced `add(add(add(…)))`
//    tree — hides the scene's structure from the next reader.
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {tone as toneKit} from '../../gpu/kit/index'
import {paintFrame} from '../invoke'
import {add, clamp, div, local, max, mul, sub, vec4} from '../math'

/**
 * One entry in `layers`: a color, or a color with options.
 *
 * `blend` is `'add'` (the default, sums light) or `'screen'` (brightens without blowing out).
 * `opacity` scales the layer. `behind` is a 0–1 factor the layer is seen through, such as the
 * transmittance of fog in front of it.
 */
export type Layer =
    | Expr
    | {
        paint: Expr
        /** `'add'` (default) sums light; `'screen'` folds it in with 1−(1−a)(1−b). */
        blend?: 'add' | 'screen'
        /** Scales the layer's contribution. */
        opacity?: Expr | number
        /** A 0–1 factor the layer is seen through (content behind a medium). */
        behind?: Expr
    }

/**
 * Builds N variations of one thing from a data list and sums them with weights.
 *
 * Give it a list of plain data and a function that turns one entry into `{value, weight}`.
 * You get `weightedSum` and `totalWeight` back and choose the blend yourself, usually
 * `weightedSum / (totalWeight + 0.001)`. Slice the list to change how many layers are built.
 *
 * @example
 * ```ts
 * const {weightedSum, totalWeight} = layered(CURTAINS.slice(0, count), ({offset, weight}) => ({value: curtain(offset), weight: math.mul(alpha, weight)}))
 * ```
 * @tip Cheap depth: aurora curtains, cloud banks, echo trails are all a few layers of one recipe.
 * @see layers
 */
export function layered<T>(
    layerData: readonly T[],
    build: (layer: T, index: number) => {value: Expr; weight: Expr},
): {weightedSum: Expr; totalWeight: Expr} {
    // weightedSum = Σ value·weight and totalWeight = Σ weight, both in list order. The caller owns
    // the blend because the guard and the remap are look decisions. Compile-time layer counts are
    // just `list.slice(0, n)` — disabled layers aren't emitted.
    let weightedSum: Expr | undefined
    let totalWeight: Expr | undefined
    layerData.forEach((layer, index) => {
        const {value, weight} = build(layer, index)
        const contribution = mul(value, weight)
        weightedSum = weightedSum === undefined ? contribution : add(weightedSum, contribution)
        totalWeight = totalWeight === undefined ? weight : add(totalWeight, weight)
    })
    if (!weightedSum || !totalWeight) throw new Error('std: layered([]) needs at least one layer')
    return {weightedSum, totalWeight}
}

/**
 * Stacks colors into one, first entry at the back.
 *
 * Each entry is a color, or `{paint, blend, opacity, behind}` (see `Layer`). Entries add
 * together unless one asks for `'screen'`. Needs at least one entry.
 *
 * @example
 * ```ts
 * const rgb = layers([
 *   {paint: sky, opacity: 0.6},
 *   {paint: stars, behind: fog.transmittance},
 *   {paint: glow, blend: 'screen'},
 * ])
 * ```
 * @see layered, emissiveAlpha
 */
export function layers(entries: Layer[]): Expr {
    let acc: Expr | undefined
    for (const entry of entries) {
        const spec = entry instanceof Object && 'paint' in entry ? entry : {paint: entry as Expr}
        let term = spec.paint
        if (spec.opacity !== undefined) term = mul(term, spec.opacity)
        if (spec.behind) term = mul(term, spec.behind)
        acc = acc === undefined
            ? term
            : spec.blend === 'screen'
                ? sub(add(acc, term), mul(acc, term))
                : add(acc, term)
    }
    if (!acc) throw new Error('std: layers([]) needs at least one layer')
    return acc
}

/**
 * Turns an rgb light value into a color with alpha, so glows composite over transparency.
 *
 * Alpha is the brightest channel and the color is scaled up to match, so over black the layer
 * looks exactly like the light it is, and over anything else like a translucent tint. Use it
 * as the last step of a glow, spill or flare. Not for opaque bodies.
 *
 * @example
 * ```ts
 * paint: (params) => compose.emissiveAlpha(math.mul(params.uniforms.color.member('rgb'), coverage))
 * ```
 * @see layers, dithered
 */
export function emissiveAlpha(rgb: Expr, hint = 'emissive'): Expr {
    const c = local(rgb, hint)
    const a = local(clamp(max(max(c.member('x'), c.member('y')), c.member('z')), 0, 1), `${hint}A`)
    return vec4(div(c, max(a, 0.0001)), a)
}

/**
 * Adds a tiny amount of noise to a final color so long soft gradients show no banding.
 *
 * Apply it to the finished rgba, never to a value that still feeds a palette.
 *
 * @example
 * ```ts
 * paint: (params) => dithered(standardPalette(field, params), params)
 * ```
 * @see emissiveAlpha
 */
export function dithered(color: Expr, params: GpuFragmentParams): Expr {
    // ±0.002 hash dither seeded from the composed UV and the definition's clock (the LightLeak idiom).
    const {uv, viewport} = paintFrame(params)
    return call(toneKit.rgbDither, 'rgbDither', [color, uv, viewport, animatedTime(params)])
}
