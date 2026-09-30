/**
 * Props and values: how a definition points at its own props (`p('radius')`) and combines them into a number, without writing GPU code.
 *
 * A prop ref is a live binding: the word you pass it to reads the prop on the GPU every
 * frame, drivers and animation included. A `Scalar` is a number built from props and masks
 * (`radialMask(…).times(p('intensity'))`) that any word taking a `ScalarInput` accepts.
 */
// Maintainer notes: the value graph. Nouns and slots reference values through these nodes,
// never through Exprs or uniforms directly — the authored file stays strict-literal data, and
// the lowering (`lower.ts` / `invoke.ts`) turns nodes into Exprs against the composition
// params. Phase 0 slice: prop bindings and scalar-producing nouns with `.times()`
// composition; the full ScalarInput union (lengths, signals, field-valued params) lands with
// later phases.

/** A reference to one of the definition's props by name. Make one with `p`. */
export interface PropRef {
    readonly kind: 'prop'
    readonly name: string
}

/**
 * Reference one of the definition's props by name, wherever a word asks for a value.
 *
 * Pass it where a std word takes a color, a position or a number and the word reads that
 * prop live on the GPU. The name must be one of the definition's `props`; a wrong name
 * throws when the shader is built.
 *
 * @example
 * ```ts
 * effect: tintToward(p('color'), {amount: p('intensity')})
 * ```
 * @tip A prop ref is a binding, not a snapshot: changing the prop later needs no rebuild.
 * @see Scalar, uniformOf, ctx
 */
export function p(name: string): PropRef {
    return {kind: 'prop', name}
}

/** The shapes a `Scalar` can hold: a prop, a product of two values, a radial mask, or a live signal. */
export type ScalarSource =
    | PropRef
    | {readonly kind: 'mul'; readonly a: ScalarSource; readonly b: ScalarSource}
    | {readonly kind: 'radialMask'; readonly center: PropRef; readonly radius: PropRef; readonly falloff: PropRef}
    | {readonly kind: 'signal'; readonly build: (params: SignalSlotParams) => import('../gpu/contract').Expr}

/** What a signal is given when it builds: the props, and the context values where the host has them (a warp's `map:` has no `ctx`). */
export interface SignalSlotParams {
    props: import('../gpu/contract').Expr
    uniforms: Record<string, import('../gpu/contract').Expr>
    ctx?: import('../gpu/contract').GpuFragmentParams['ctx']
}

/**
 * A number built from props: what `radialMask(…)`, `motion.oscillating(…)` and `.times()` return.
 *
 * Pass it anywhere a word accepts a `ScalarInput` (a prop ref or a Scalar): an effect's
 * `amount`, a mask's strength. Chain `.times(p('intensity'))` to scale it by a prop.
 *
 * @example
 * ```ts
 * effect: tintToward(p('color'), {amount: radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity'))})
 * ```
 * @see p, resolveScalar
 */
// Wraps the data node and carries the fluent combinators; serialization reads `.node`.
export class Scalar {
    constructor(readonly node: ScalarSource) {}

    /** Multiply by a prop or another Scalar. */
    times(other: Scalar | PropRef): Scalar {
        return new Scalar({kind: 'mul', a: this.node, b: other instanceof Scalar ? other.node : other})
    }
}

/** What a number-valued input accepts: a prop ref `p('x')` or a `Scalar`. */
export type ScalarInput = Scalar | PropRef
