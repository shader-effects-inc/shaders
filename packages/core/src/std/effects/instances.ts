/**
 * Repeat the layer inside: many copies laid out in a grid, around a circle or along a line,
 * each copy a little different. `repeatInstances` returns the fragment for `gpu.fragment`;
 * `grid`, `radial`, `line` and `byMode` describe the layout it uses, and the recipe's
 * `variation` object says how each copy scales, turns, fades and jitters.
 *
 * Copies are cut from the layer's own bounds when the engine knows them (declare
 * `wantsBoundsParams: true`), otherwise from the whole canvas, and spread across this layer's
 * bounding box. Radius, spacing and gaps are fractions of the canvas (height for vertical
 * measures and the radial orbit, width for `gapX`); angles are in degrees unless noted.
 */
// Maintainer: part bodies live in kit/instanceParts (string emitters, not tgpu.fns, because the
// per-instance texture sample sits inside a runtime-count loop behind a non-uniform cull); this
// module resolves `p()` slots and assembles the loop. The source rect follows childBoundsParams
// (fallback: full canvas) and the placement field follows ownBoundsParams the same way.
import {Expr, type EmitContext, type GpuFragmentParams} from '../../gpu/contract'
import {call, expr, ZERO} from '../../gpu/composer'
import {blend, instanceParts} from '../../gpu/kit/index'
import type {PropRef} from '../values'

const {formatFloat} = instanceParts

// ── Placement specs ─────────────────────────────────────────────────────────────────────

/**
 * The props of a grid layout: `columns` and `rows` (counts), `gapX` and `gapY` (fractions of
 * the canvas width and height), `stagger` (0 to 1, shifts every other row by that much of a
 * cell) and `flip`, a compile-time select that mirrors alternate cells (`'none'`,
 * `'alternate-flip-x'`, `'alternate-flip-y'`, `'both'`).
 */
export interface GridSlots {
    columns: PropRef
    rows: PropRef
    gapX: PropRef
    gapY: PropRef
    stagger: PropRef
    /** A compile-time select: `'none'`, `'alternate-flip-x'`, `'alternate-flip-y'` or `'both'`. */
    flip: PropRef
}
/**
 * The props of a radial layout: `count`, `radius` (a fraction of the canvas height),
 * `startAngle` and `sweep` (degrees), and `faceCenter`, a boolean that turns each copy to face
 * the middle of the circle.
 */
export interface RadialSlots {
    count: PropRef
    radius: PropRef
    startAngle: PropRef
    sweep: PropRef
    faceCenter: PropRef
}
/**
 * The props of a line layout: `count`, `direction` (degrees, 0 points right) and `spacing`
 * between copies (a fraction of the canvas height).
 */
export interface LineSlots {
    count: PropRef
    direction: PropRef
    spacing: PropRef
}

/** One layout, built by `grid`, `radial` or `line`. */
export type PlacementSpec =
    | {kind: 'grid'; slots: GridSlots}
    | {kind: 'radial'; slots: RadialSlots}
    | {kind: 'line'; slots: LineSlots}

/**
 * Copies in rows and columns, filling this layer's bounding box.
 *
 * Pass it as the recipe's `placement`, or as one member of `byMode`.
 *
 * @example
 * ```ts
 * placement: grid({columns: p('columns'), rows: p('rows'), gapX: p('gapX'), gapY: p('gapY'), stagger: p('stagger'), flip: p('flip')})
 * ```
 * @tip Mark `flip` `compileTime: true`. Only the chosen mirroring is compiled in.
 * @see radial, line, byMode
 */
export const grid = (slots: GridSlots): PlacementSpec => ({kind: 'grid', slots})
/**
 * Copies around a circle centred on the layer inside.
 *
 * `sweep` at 360 spaces them evenly all the way round; smaller sweeps make an arc starting at
 * `startAngle`.
 *
 * @example
 * ```ts
 * placement: radial({count: p('count'), radius: p('radius'), startAngle: p('startAngle'), sweep: p('sweep'), faceCenter: p('faceCenter')})
 * ```
 * @see grid, line, byMode
 */
export const radial = (slots: RadialSlots): PlacementSpec => ({kind: 'radial', slots})
/**
 * Copies in a row, marching away from the layer inside in one direction.
 *
 * @example
 * ```ts
 * placement: line({count: p('count'), direction: p('direction'), spacing: p('spacing')})
 * ```
 * @see grid, radial, byMode
 */
export const line = (slots: LineSlots): PlacementSpec => ({kind: 'line', slots})

/** A layout chosen by a select prop, built by `byMode`. */
export interface PlacementPick {
    mode: PropRef
    members: {grid: PlacementSpec; radial: PlacementSpec; linear: PlacementSpec}
}
/**
 * Lets one select prop switch between a grid, a radial and a linear layout.
 *
 * `mode` is a compile-time select with the values `'grid'`, `'radial'` and `'linear'`; only
 * the chosen layout is compiled in.
 *
 * @example
 * ```ts
 * placement: byMode(p('mode'), {grid: grid({...}), radial: radial({...}), linear: line({...})})
 * ```
 * @see grid, radial, line
 */
// Maintainer: structural read — `propValues[mode]` may be the raw string or its transformed
// number (MODE_MAP); only the active member emits.
export const byMode = (mode: PropRef, members: PlacementPick['members']): PlacementPick => ({mode, members})

// ── Variation & recipe ──────────────────────────────────────────────────────────────────

/**
 * How each copy differs from the last. `scale` and `opacity` multiply with every copy (1 keeps
 * them equal), `rotation` adds per copy in radians (give the prop a degrees-to-radians
 * transform). The `jitter` props (0 to 1) add seeded randomness to position, rotation, scale
 * and opacity; `seed` re-rolls it. `phase` (0 to 1) slides the whole layout by one cell or one
 * copy and wraps seamlessly, for looping animation.
 */
export interface VariationSlots {
    scale: PropRef
    rotation: PropRef
    opacity: PropRef
    jitter: {position: PropRef; rotation: PropRef; scale: PropRef; opacity: PropRef}
    seed: PropRef
    /** Layout animation phase, 0 to 1; 1 is the same layout as 0. */
    phase: PropRef
}

/** Everything `repeatInstances` needs: what to copy, where to put the copies, and how they vary. */
export interface RepeatInstancesRecipe {
    /** Crop insets on the copied area, each a fraction (0 to 0.49) of its width or height. */
    source: {crop: {left: PropRef; right: PropRef; top: PropRef; bottom: PropRef}}
    placement: PlacementPick | PlacementSpec
    variation: VariationSlots
    /** Hue rotation added per copy, in radians. Off at 0; give the prop `recompile: crosses(0)`. */
    hueShift?: PropRef
    /** A compile-time select: `'forward'` stacks later copies on top, `'backward'` underneath. */
    order: PropRef
}

const MODE_MAP: Record<string, number> = {grid: 0, radial: 1, linear: 2}
const FLIP_MAP: Record<string, number> = {'none': 0, 'alternate-flip-x': 1, 'alternate-flip-y': 2, 'both': 3}

/**
 * Many copies of the layer inside, laid out and varied as the recipe says.
 *
 * Returns the fragment for `gpu.fragment` on a definition with `requiresRTT`, `requiresChild`
 * and `wantsBoundsParams` set. Copies stack with normal alpha in the recipe's `order`. The
 * result is straight alpha.
 *
 * @example
 * ```ts
 * gpu: {fragment: repeatInstances({source: {crop: {left: p('cropLeft'), right: p('cropRight'), top: p('cropTop'), bottom: p('cropBottom')}}, placement: radial({count: p('count'), radius: p('radius'), startAngle: p('startAngle'), sweep: p('sweep'), faceCenter: p('faceCenter')}), variation: {scale: p('instanceScale'), rotation: p('instanceRotation'), opacity: p('instanceOpacity'), jitter: {position: p('jitterPosition'), rotation: p('jitterRotation'), scale: p('jitterScale'), opacity: p('jitterOpacity')}, seed: p('seed'), phase: p('phase')}, hueShift: p('hueShift'), order: p('zOrder')})}
 * ```
 * @tip `hueShift` costs nothing at 0. Give the prop `recompile: crosses(0)` so it switches on cleanly.
 * @see grid, radial, line, byMode
 */
// Maintainer: placement frame → per-instance variation (→ hue-shift stage when the prop is
// non-zero) → premultiplied over-accumulation of the child sampled per instance, unpremultiplied
// at the end like every RTT-sampling filter. `order` and `mode`/`flip` are structural reads
// (propValues), so they must be compile-time props.
export function repeatInstances(recipe: RepeatInstancesRecipe) {
    return (params: GpuFragmentParams): Expr => {
        const {uniforms, childNode, ctx, propValues, childBoundsParams, ownBoundsParams, convertToTexture} = params
        if (!childNode) return ZERO

        const childTex = convertToTexture(childNode)
        const texKey = childTex.key

        // Structural reads: only the active placement member / stages are emitted.
        const structural = (ref: PropRef, map: Record<string, number>): number => {
            const v = propValues[ref.name]
            return typeof v === 'number' ? v : (map[v as string] ?? 0)
        }
        const spec: PlacementSpec = 'members' in recipe.placement
            ? [recipe.placement.members.grid, recipe.placement.members.radial, recipe.placement.members.linear][
                structural(recipe.placement.mode, MODE_MAP)] ?? recipe.placement.members.grid
            : recipe.placement
        const zBack = recipe.order ? (propValues[recipe.order.name] === 'backward' || propValues[recipe.order.name] === 1) : false
        const hueActive = recipe.hueShift ? ((propValues[recipe.hueShift.name] as number) ?? 0) !== 0 : false

        return new Expr((ec: EmitContext) => {
            const f: instanceParts.InstanceEmitFrame = {p: ec.freshLocal('rep'), L: formatFloat}
            const {p, L} = f
            const R = (ref: PropRef): string => {
                const accessor = uniforms[ref.name]
                if (!accessor) throw new Error(`std: repeatInstances binds unknown prop '${ref.name}'`)
                return accessor._emit(ec)
            }
            const stmts: string[] = []

            stmts.push(`let ${p}_uv = ${ctx.uv._emit(ec)};`)
            stmts.push(`let ${p}_aspect = ${ctx.aspect._emit(ec)};`)
            stmts.push(`let ${p}_pix = vec2f(${p}_uv.x * ${p}_aspect, ${p}_uv.y);`)

            // Source rect: the child layer bounds when available, else the full canvas.
            const src = {
                cx: childBoundsParams ? childBoundsParams.centerX._emit(ec) : L(0.5),
                cy: childBoundsParams ? childBoundsParams.centerY._emit(ec) : L(0.5),
                hw: childBoundsParams ? childBoundsParams.halfWidth._emit(ec) : L(0.5),
                hh: childBoundsParams ? childBoundsParams.halfHeight._emit(ec) : L(0.5),
            }
            stmts.push(...instanceParts.sourceRect(f, src, {
                left: R(recipe.source.crop.left), right: R(recipe.source.crop.right),
                top: R(recipe.source.crop.top), bottom: R(recipe.source.crop.bottom),
            }))

            // Placement field: the consumer's own bounding box when active, else the canvas.
            stmts.push(...instanceParts.placementField(f, {
                cx: ownBoundsParams ? ownBoundsParams.centerX._emit(ec) : L(0.5),
                cy: ownBoundsParams ? ownBoundsParams.centerY._emit(ec) : L(0.5),
                hw: ownBoundsParams ? ownBoundsParams.halfWidth._emit(ec) : L(0.5),
                hh: ownBoundsParams ? ownBoundsParams.halfHeight._emit(ec) : L(0.5),
            }))

            stmts.push(`let ${p}_phase = ${R(recipe.variation.phase)};`)
            stmts.push(`let ${p}_seed = ${R(recipe.variation.seed)};`)

            // Radial / line placements anchor to the child layer position (fallback: field centre).
            const anchorSq = childBoundsParams
                ? `vec2f((${childBoundsParams.centerX._emit(ec)}) * ${p}_aspect, ${childBoundsParams.centerY._emit(ec)})`
                : `${p}_fieldCenterSq`
            const placement: instanceParts.PlacementPart =
                spec.kind === 'grid'
                    ? instanceParts.gridPlacement({
                        columns: R(spec.slots.columns), rows: R(spec.slots.rows),
                        gapX: R(spec.slots.gapX), gapY: R(spec.slots.gapY), stagger: R(spec.slots.stagger),
                    }, structural(spec.slots.flip, FLIP_MAP))
                    : spec.kind === 'radial'
                        ? instanceParts.radialPlacement(anchorSq, {
                            count: R(spec.slots.count), sweep: R(spec.slots.sweep),
                            startAngle: R(spec.slots.startAngle), radius: R(spec.slots.radius),
                            faceCenter: R(spec.slots.faceCenter),
                        })
                        : instanceParts.linePlacement(anchorSq, {
                            count: R(spec.slots.count), direction: R(spec.slots.direction), spacing: R(spec.slots.spacing),
                        })

            stmts.push(...instanceParts.accumulateInstances(f, {
                placement,
                variation: instanceParts.instanceVariation(f, {
                    scale: R(recipe.variation.scale), rotation: R(recipe.variation.rotation),
                    opacity: R(recipe.variation.opacity),
                    jitterPosition: R(recipe.variation.jitter.position), jitterRotation: R(recipe.variation.jitter.rotation),
                    jitterScale: R(recipe.variation.jitter.scale), jitterOpacity: R(recipe.variation.jitter.opacity),
                }),
                hue: hueActive && recipe.hueShift ? instanceParts.instanceHueShift(f, R(recipe.hueShift)) : [],
                zBack,
                texKey,
            }))

            for (const s of stmts) ec.statement(s)

            // The accumulator is premultiplied → straight alpha, like every RTT-sampling filter.
            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [expr(`${p}_accum`)])._emit(ec)
        })
    }
}
