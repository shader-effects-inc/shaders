/**
 * Identity and recompile rules: tell the engine when a filter does nothing, and when a prop change needs a rebuild.
 *
 * An identity rule sits on a filter definition's `identityWhen:`. While it holds, the filter
 * is skipped and the layer inside passes through untouched, at no cost. A recompile rule sits
 * on a prop's `recompile:` and says which changes rebuild the shader instead of just updating
 * a value. The two pair up: `recompile: crosses(0)` on the prop, `identityWhen: isZero(...)` on
 * the definition, so the skip switches on and off exactly at the threshold.
 */
// Maintainer notes: these replace function-valued escape hatches with data — `crosses(v)`
// replaces a hand-written `compileTimeWhen` predicate, the identity rules replace
// hand-written `identity.when` closures. The lowerings live in `lower.ts`; the scaffolds own
// the bypass (pointwise returns the child raw, gather samples the centre texel) and the
// map-driver refusal.
import type {IdentityRule, RecompileRule} from './types'

/**
 * Rebuild the shader only when the prop crosses `value`, not on every change.
 *
 * Put it on a prop's `recompile:`. Pair it with an identity rule at the same value: the rule
 * decides whether the filter is skipped, and this is what lets that decision change.
 *
 * @example
 * ```ts
 * props: {shift: {default: 0, recompile: crosses(0)}},
 * effect: hueRotate(p('shift')),
 * identityWhen: isZero('shift'),
 * ```
 * @tip Without it a runtime prop never rebuilds, so a filter that started skipped stays skipped.
 * @see isZero, isValue, recompileWhen
 */
export function crosses(value: number): RecompileRule {
    return {kind: 'crosses', value}
}

/**
 * Rebuild when your own test of the old and new value says so, for a threshold `crosses` cannot express.
 *
 * @example
 * ```ts
 * blur: {default: 0, recompile: recompileWhen((prev, next) => (prev > 0) !== (next > 0))}
 * ```
 * @see crosses
 */
// The loose fn signature mirrors PropConfig.compileTimeWhen.
export function recompileWhen(predicate: (prev: any, next: any) => boolean): RecompileRule {
    return {kind: 'custom', predicate}
}

/**
 * Skip the filter entirely while `prop` is exactly 0.
 *
 * Put it on `identityWhen:`. An unset prop counts as its default. A prop that is mapped per
 * pixel by a driver disables the skip automatically, since the value then varies across the
 * canvas. Give the prop `recompile: crosses(0)` so the skip switches on and off.
 *
 * @example
 * ```ts
 * identityWhen: isZero('intensity')
 * ```
 * @see isValue, allOf, crosses
 */
export function isZero(prop: string): IdentityRule {
    return {kind: 'isZero', prop}
}

/**
 * Skip the filter entirely while `prop` equals `value`, for a filter whose no-op is not 0.
 *
 * An unset prop counts as its default; a per-pixel mapped prop disables the skip. Give the
 * prop `recompile: crosses(value)` so the skip switches on and off.
 *
 * @example
 * ```ts
 * props: {intensity: {default: 1, recompile: crosses(1)}},
 * effect: saturate(p('intensity')),
 * identityWhen: isValue('intensity', 1),
 * ```
 * @see isZero, allOf, crosses
 */
export function isValue(prop: string, value: unknown): IdentityRule {
    return {kind: 'isValue', prop, value}
}

/**
 * Skip the filter only while every rule holds, for a no-op that needs several props at rest.
 *
 * @example
 * ```ts
 * identityWhen: allOf(isZero('brightness'), isZero('contrast'))
 * ```
 * @see isZero, isValue
 */
export function allOf(...rules: IdentityRule[]): IdentityRule {
    return {kind: 'allOf', rules}
}

/**
 * Skip the filter when your own test of the props' values says so.
 *
 * List in `props` every prop the test reads: a per-pixel mapped prop among them disables the
 * skip, and the engine needs the list to know which.
 *
 * @example
 * ```ts
 * identityWhen: identityWhenever(['amount', 'mode'], (v) => v.amount === 0 || v.mode === 'off')
 * ```
 * @see isZero, isValue, allOf
 */
export function identityWhenever(props: string[], when: (values: Record<string, unknown>) => boolean): IdentityRule {
    return {kind: 'custom', props, when}
}
