/**
 * List props: reading an array-valued prop (any number of lights, points, anchors) in a hand-written builder, and looping over its items.
 *
 * Declare the prop with `listPropConfig`: each item is a small record of typed fields
 * (`position`, `color`, `number`, `boolean`) and the list has a `maxItems` cap. In the
 * builder, `listOf(params, 'lights')` gives the live item count and each item's fields, and
 * `accumulate` adds a per-item value over all of them in a real GPU loop. Adding or removing
 * an item at runtime never rebuilds the shader.
 */
// Maintainer notes: a list prop `lights` with item fields `{position, color, intensity}`
// arrives in the fragment as `uniforms.lights_position[i]` / `uniforms.lights_color[i]` /
// `uniforms.lights_intensity[i]` (vec4 lanes, see the packing note in `utilities/listProps`)
// plus `uniforms.lightsCount`. `listOf` names those; `accumulate` is the loop — a real WGSL
// `for` over the RUNTIME count (adding an item never recompiles), whose body may hoist
// `local()`s freely (they land inside the loop, the `guarded` rule: anything shared with the
// enclosing recipe goes through `deps`).
import type {EmitContext} from '../gpu/contract'
import {Expr} from '../gpu/contract'
import {listCountName, listFieldName} from '../utilities/listProps'

let listCounter = 0

const raw = (name: string): Expr => new Expr(() => name)

/** One item's fields at a loop index, read by field name and kind. */
export interface ListItemAt {
    /** A `position` field as `vec2f`, in the engine's stored form (y flipped: 1 − the authored y). */
    position: (field: string) => Expr
    /** A `color` field as `vec4f`, linear RGB with alpha. */
    color: (field: string) => Expr
    /** A `number` or `boolean` field as `f32` (true is 1, false is −1). */
    number: (field: string) => Expr
}

/** A list prop bound for reading: the live item count and each item's fields. */
export interface ListRef {
    /** How many items the list holds right now, as an `f32`. */
    count: Expr
    /** The fields of item `i`. */
    at: (i: Expr) => ListItemAt
}

/**
 * Read a list prop inside a hand-written builder: its live item count and each item's fields.
 *
 * `listOf(params, 'lights')` gives `count` and `at(i)`, whose `.position('position')`,
 * `.color('color')` and `.number('intensity')` read item `i` by field name. Throws when
 * `prop` is not a list prop or a field name is not in its item spec. Loop with `accumulate`.
 *
 * @example
 * ```ts
 * const lights = listOf(params, 'lights')
 * const rgb = accumulate(lights, (i) => mul(lights.at(i).color('color').member('rgb'), lights.at(i).number('intensity')), {zero: 'vec3f'})
 * ```
 * @tip A position field's y is stored flipped; use `sub(1, pos.member('y'))` to compare it with `uv`.
 * @see accumulate
 */
export function listOf(params: {uniforms: Record<string, Expr>}, prop: string): ListRef {
    const lane = (field: string, i: Expr): Expr => {
        const arr = params.uniforms[listFieldName(prop, field)]
        if (!arr) throw new Error(`std: list '${prop}' has no field '${field}'`)
        return new Expr((ctx) => `${arr._emit(ctx)}[${i._emit(ctx)}]`)
    }
    const count = params.uniforms[listCountName(prop)]
    if (!count) throw new Error(`std: '${prop}' is not a list prop`)
    return {
        count,
        at: (i) => ({
            position: (field) => lane(field, i).member('xy'),
            color: (field) => lane(field, i),
            number: (field) => lane(field, i).member('x'),
        }),
    }
}

/**
 * Add up a per-item value over every item of a list, in a real GPU loop over the live count.
 *
 * `body(i)` returns the value for item `i`. `zero` picks the sum's type: `'f32'` (default) or
 * `'vec3f'` for a color. Anything the body reads that was computed outside the loop (a
 * normal, a surface point) goes in `deps`, so it is emitted once before the loop instead of
 * inside it. `hint` names the sum in the generated code.
 *
 * @example
 * ```ts
 * const lights = listOf(params, 'lights')
 * const rgb = accumulate(lights, (i) => {
 *   const light = lights.at(i)
 *   const pos = light.position('position')
 *   const d = length(sub(uv, vec2(pos.member('x'), sub(1, pos.member('y')))))
 *   return mul(light.color('color').member('rgb'), div(light.number('intensity'), add(1, mul(d, d))))
 * }, {zero: 'vec3f', deps: [uv], hint: 'lightSum'})
 * ```
 * @see listOf
 */
// Statement machinery: `var acc = zero; for (i < count) { acc += body(i) }`. See the module
// notes for the hoisting rule.
export function accumulate(
    list: ListRef,
    body: (i: Expr) => Expr,
    opts: {zero?: 'f32' | 'vec3f'; deps?: Expr[]; hint?: string} = {},
): Expr {
    const id = listCounter++
    const zero = opts.zero === 'vec3f' ? 'vec3f(0.0)' : '0.0'
    const hint = opts.hint ?? 'sum'
    return new Expr((ctx) =>
        ctx.memo(`accumulate:${id}`, () => {
            for (const dep of opts.deps ?? []) dep._emit(ctx)
            const acc = ctx.freshLocal(hint)
            const i = ctx.freshLocal('item')
            const countText = list.count._emit(ctx)
            const stmts: string[] = []
            const scoped: EmitContext = {
                external: (v, h) => ctx.external(v, h),
                statement: (w) => stmts.push(w),
                freshLocal: (h) => ctx.freshLocal(h),
                memo: (k, f) => ctx.memo(k, f),
            }
            const term = body(raw(i))._emit(scoped)
            ctx.statement(`var ${acc} = ${zero};`)
            ctx.statement([
                `for (var ${i} = 0u; ${i} < u32(${countText}); ${i}++) {`,
                ...stmts.map((w) => `  ${w}`),
                `  ${acc} += ${term};`,
                `}`,
            ].join('\n'))
            return acc
        }),
    )
}
