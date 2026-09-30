/**
 * std/paint/patterns — repeating patterns: checkers, stripes, zigzags, rings, falling
 * streaks, lattices of squares, hexagons, triangles, bricks, truchet arcs, woven threads and
 * isometric cubes, plus the print-screen filters (halftone dots, CMYK plates, dithering).
 *
 * Most words here return a mask: a number per pixel, 1 on the pattern and 0 off it. Mix two
 * colors through it with `strokeOver`, or hang it under one color as alpha with `withAlpha`.
 * The lattice patterns share one coordinate system: `cellFrame` sets the cell count and the
 * rotation, and the line words draw inside it. Every slot takes a prop as `p('name')`, so the
 * pattern follows the editor's controls with no extra wiring. Generator words go in `paint:`;
 * the halftone and dither words are filters and go in `effect:`.
 */
// Maintainer notes. A paint part is `(params) => Expr`; parts whose result is read by more than
// one consumer are memoised per composition (`shared`, below) so the readers share ONE lowered
// Expr object — exactly a `const` local in a fused builder. Parts own the UV-context idiom (a
// UV-propagating parent supplies a distorted `uvContext` + `effectiveViewportSize`; standalone
// falls back to `ctx.uv` / `ctx.viewportSize` — see `paintFrame` in std/invoke) and their own
// animated-time reads (`animatedTime: {speed}` stays declared on the definition). The GPU bodies
// live in `gpu/kit/patternPaints.ts` (frames, cell parts, per-lattice fields, halftone and
// dither stages); the composition lives in each shader file, the parts live here.
import type {Expr, GpuFragmentParams, GpuMapSampleUVs} from '../../gpu/contract'
import {call, vec4, mixExpr, animatedTime, arrayExpr, floatE} from '../../gpu/porters'
import {patternPaints, cells as cellKit} from '../../gpu/kit/index'
import type {RttFilterParams} from '../../gpu/scaffolds/rttFilter'
import type {GatherEffect} from '../types'
import {gather} from '../filter'
import type {PropRef} from '../values'
import {uniformOf, paintFrame} from '../invoke'
import {mixColorsIn} from './fields'
import {local} from '../math'

/**
 * A pattern value: something you place in `paint:` or hand to another pattern word, yielding
 * a color or a mask per pixel.
 */
export type PatternPaint = (params: GpuFragmentParams) => Expr

/** What a pattern slot accepts: a prop `p('name')` or another pattern value. */
export type PaintSlot = PropRef | PatternPaint

function resolveSlot(slot: PaintSlot, params: GpuFragmentParams): Expr {
    return typeof slot === 'function' ? slot(params) : uniformOf(slot, params)
}

/**
 * Memoise a part per composition so every reader shares ONE lowered Expr object — the
 * slot-graph equivalent of a `const` local in a fused builder.
 */
function shared<P extends object>(build: (params: P) => Expr): (params: P) => Expr {
    const cache = new WeakMap<P, Expr>()
    return (params) => {
        const hit = cache.get(params)
        if (hit) return hit
        const e = build(params)
        cache.set(params, e)
        return e
    }
}

// ═══ Lattice parts ══════════════════════════════════════════════════════════════════════════════

// Maintainer note: the three conventions are preserved exactly from the generators they came
// from — the Y flip and the rotation sign are part of each pattern's look, not a normalisable
// difference. `'flippedY'` = `latticeFrameFlipY` (grid, triangular grid), `'clockwise'` =
// `latticeFrameCW` (hex grid, truchet, isometric cubes), `'plain'` = `latticeFrame` (weave).
/**
 * Which way a lattice's rows and rotation run.
 *
 * `'flippedY'` counts rows up from the bottom and rotates counter-clockwise (grids,
 * triangles). `'clockwise'` counts rows down from the top and rotates clockwise (hexagons,
 * truchet, cubes). `'plain'` counts down from the top and rotates counter-clockwise (weave).
 * Use the one named in the line word's example; the others still work but shift the look.
 */
export type CellFrameConvention = 'flippedY' | 'clockwise' | 'plain'

const FRAME_BODIES = {
    flippedY: {fn: () => patternPaints.latticeFrameFlipY, hint: 'latticeFrameFlipY'},
    clockwise: {fn: () => patternPaints.latticeFrameCW, hint: 'latticeFrameCW'},
    plain: {fn: () => patternPaints.latticeFrame, hint: 'latticeFrame'},
} as const

/**
 * The shared coordinate system of a tiled pattern: how many cells fit down the canvas
 * height and how far the whole lattice is rotated.
 *
 * `cells` counts cells along the canvas height, so cells stay square as the canvas resizes.
 * `rotation` is in degrees about the canvas centre. Pass the result as the `frame` of
 * `gridLines`, `hexLines`, `triangleLines`, `truchetArcs`, `weaveThreads` or `isoCubeFaces`.
 *
 * @example
 * ```ts
 * const field = gridLines({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'flippedY'}), thickness: p('thickness'), softness: p('softness'), variation: p('variation')})
 * ```
 * @tip One frame can feed several line words. It is evaluated once however many read it.
 * @see gridLines, hexLines, triangleLines, truchetArcs, weaveThreads, isoCubeFaces
 */
export function cellFrame(slots: {cells: PropRef; rotation: PropRef; convention: CellFrameConvention}): PatternPaint {
    // Maps the paint UV (uvContext-aware) into rotated, aspect-corrected lattice space scaled
    // to `cells` lattice units; everything drawn IN the cells composes on top.
    const body = FRAME_BODIES[slots.convention]
    return shared((params) => {
        const {uv, viewport} = paintFrame(params)
        return call(body.fn(), body.hint, [uv, viewport, uniformOf(slots.cells, params), uniformOf(slots.rotation, params)])
    })
}

/**
 * A color lightened or darkened per cell by a shade factor, alpha untouched.
 *
 * `factor` is the `shade` output of a lattice word: 1 keeps the color, above 1 brightens,
 * below 1 darkens. Use it on the cell fill so every tile reads slightly different.
 *
 * @example
 * ```ts
 * paint: strokeOver({fill: vary(p('cellColor'), field.shade), stroke: p('color'), mask: field.lines, space: p('colorSpace')})
 * ```
 * @see gridLines, hexLines, brickCourses, strokeOver
 */
export function vary(color: PaintSlot, factor: PaintSlot): PatternPaint {
    return (params) => {
        const c = resolveSlot(color, params)
        return vec4(c.member('rgb').mul(resolveSlot(factor, params)), c.member('a'))
    }
}

/**
 * Two colors mixed through a mask: `fill` where the mask is 0, `stroke` where it is 1.
 *
 * The everyday way to color a pattern. `space` is a color-space prop (`'linear'`, `'oklch'`,
 * `'oklab'`, `'hsl'`, `'hsv'`, `'lch'`); leave it out for a linear mix. Alpha interpolates
 * too, so a transparent fill gives a lines-only pattern.
 *
 * @example
 * ```ts
 * paint: strokeOver({fill: p('colorA'), stroke: p('colorB'), mask: checkerCells({cells: p('cells'), softness: p('softness')}), space: p('colorSpace')})
 * ```
 * @tip The `space` prop must be `compileTime: true`. Changing it recompiles the shader.
 * @see withAlpha, mixOf, vary
 */
export function strokeOver(layers: {fill: PaintSlot; stroke: PaintSlot; mask: PaintSlot; space?: PropRef}): PatternPaint {
    // The mix is the color space's `mixColors` variant (alpha-weighted; output alpha is the
    // weighted sum), chosen at composition time from the compile-time `space` value.
    return shared((params) => call(mixColorsIn(layers.space, params), 'mixColors', [
        resolveSlot(layers.fill, params), resolveSlot(layers.stroke, params), resolveSlot(layers.mask, params),
    ]))
}

/**
 * Makes map-driven props change per whole cell instead of per pixel, so a mapped dot size or
 * line thickness grows cell by cell rather than being cut off mid-cell.
 *
 * Set it on the definition's `mapSampleUVs` field, not in `paint:`. Give it the same `cells`
 * count and `rotation` (degrees) the pattern uses so the snapping matches the lattice.
 * `props` lists the prop names to snap.
 *
 * @example
 * ```ts
 * mapSampleUVs: sampleMapsAtCellCentres({cells: p('cells'), rotation: p('rotation'), props: ['thickness', 'softness']})
 * ```
 * @tip Only matters when a prop is driven by a map (an image or another layer). Plain values are unaffected.
 * @see cellFrame, dotLattice, gridLines
 */
export function sampleMapsAtCellCentres(slots: {
    cells: PropRef
    rotation?: PropRef
    props: string[]
}): GpuMapSampleUVs {
    // Samples the listed mapped props at the CELL CENTRE rather than the fragment position.
    // Uses the raw canvas UV/viewport (the global, not effectiveViewportSize) because map
    // sampling happens in the composer against the canvas, before any uvContext applies.
    return ({uniforms, ctx}) => {
        const params = {uniforms}
        const cellCenter = slots.rotation
            ? call(patternPaints.gridCellCenterUV, 'gridCellCenterUV', [
                ctx.uv, ctx.viewportSize, uniformOf(slots.cells, params), uniformOf(slots.rotation, params),
            ])
            : call(cellKit.cellCentreUV, 'cellCentreUV', [ctx.uv, ctx.viewportSize, uniformOf(slots.cells, params)])
        const out: Record<string, Expr> = {}
        for (const prop of slots.props) out[prop] = cellCenter
        return out
    }
}

// ═══ Compose parts ═════════════════════════════════════════════════════════════════════════════

/**
 * A color with its alpha replaced by a separately built mask.
 *
 * The usual way to hang a pattern under one color: the mask becomes the coverage and the
 * color's own alpha is dropped.
 *
 * @example
 * ```ts
 * paint: withAlpha(p('color'), dotLattice({density: p('density'), dotSize: p('dotSize'), offset: p('offset'), speedVariance: p('speedVariance'), twinkle: p('twinkle')}))
 * ```
 * @tip To keep the color's own alpha as well, multiply it in: `withAlpha(color, times(alphaOf(color), mask))`.
 * @see alphaOf, times, opaque
 */
export function withAlpha(color: PaintSlot, alpha: PaintSlot): PatternPaint {
    return (params) => {
        const c = resolveSlot(color, params)
        return vec4(c.member('rgb'), resolveSlot(alpha, params))
    }
}

/**
 * The alpha channel of a color, as a number per pixel.
 *
 * @example
 * ```ts
 * paint: withAlpha(streakColor, times(alphaOf(streakColor), streaks.mask))
 * ```
 * @see rgbOf, withAlpha
 */
export function alphaOf(color: PaintSlot): PatternPaint {
    return (params) => resolveSlot(color, params).member('a')
}

/**
 * The RGB channels of a color, without its alpha.
 *
 * Finish it with `opaque` or `withAlpha` to make a color again.
 *
 * @example
 * ```ts
 * paint: opaque(mixOf(times(rgbOf(cubeColor), faces.tone), rgbOf(p('lineColor')), faces.wire))
 * ```
 * @see alphaOf, opaque
 */
export function rgbOf(color: PaintSlot): PatternPaint {
    return (params) => resolveSlot(color, params).member('rgb')
}

/**
 * The product of two values: a mask scaled by a prop, an RGB dimmed by a shade, two masks
 * combined.
 *
 * @example
 * ```ts
 * mask: times(faces.random, p('colorVariation'))
 * ```
 * @see mixOf, vary
 */
export function times(a: PaintSlot, b: PaintSlot): PatternPaint {
    return (params) => resolveSlot(a, params).mul(resolveSlot(b, params))
}

/**
 * A straight blend from one value to another by an amount from 0 to 1.
 *
 * Works on colors, RGB and single numbers alike, channel by channel with no color-space
 * handling. For a color mix in a chosen color space use `strokeOver`.
 *
 * @example
 * ```ts
 * mixOf(alphaOf(p('colorA')), alphaOf(p('colorB')), bands)
 * ```
 * @see strokeOver, times
 */
export function mixOf(from: PaintSlot, to: PaintSlot, amount: PaintSlot): PatternPaint {
    return (params) => mixExpr(resolveSlot(from, params), resolveSlot(to, params), resolveSlot(amount, params))
}

/**
 * An RGB value finished as a fully opaque color.
 *
 * @example
 * ```ts
 * paint: opaque(mixOf(times(rgbOf(cubeColor), faces.tone), rgbOf(p('lineColor')), faces.wire))
 * ```
 * @see rgbOf, withAlpha
 */
export function opaque(rgb: PaintSlot): PatternPaint {
    return (params) => vec4(resolveSlot(rgb, params), 1.0)
}

// ═══ Mask and field parts ══════════════════════════════════════════════════════════════════════

/**
 * A checkerboard mask: 1 on one set of squares, 0 on the other, with clean edges at any size.
 *
 * `cells` counts squares along the canvas height (squares stay square as the canvas
 * resizes). `softness` 0–1 blurs the edges; 0 is crisp. Color it with `strokeOver`.
 *
 * @example
 * ```ts
 * paint: strokeOver({fill: p('colorA'), stroke: p('colorB'), mask: checkerCells({cells: p('cells'), softness: p('softness')}), space: p('colorSpace')})
 * ```
 * @tip Stays sharp inside a distortion, which makes it a good test pattern for warps.
 * @see stripeBands, gridLines, strokeOver
 */
export function checkerCells(slots: {cells: PropRef; softness: PropRef}): PatternPaint {
    // The analytically anti-aliased checker blend factor (Quilez 2D filter over the
    // screen-space footprint), Y-flipped aspect-corrected cell coordinate.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.checkerBlend, 'checkerBlend', [
            uv, viewport, uniformOf(slots.cells, params), uniformOf(slots.softness, params),
        ])
    }
}

/**
 * A mask of straight parallel stripes that scroll with the layer's clock.
 *
 * `angle` is in degrees. `density` is the number of stripe pairs across the canvas height.
 * `balance` 0–1 is how much of each pair is the 0 side (the fill in `strokeOver`).
 * `softness` 0–1 blurs the edges. `offset` shifts the phase; 1 is one whole pair. Declare
 * `animatedTime: {speed: 'speed'}` on the definition; speed 0 holds the stripes still.
 *
 * @example
 * ```ts
 * paint: strokeOver({fill: p('colorA'), stroke: p('colorB'), mask: stripeBands({angle: p('angle'), density: p('density'), balance: p('balance'), softness: p('softness'), offset: p('offset')}), space: p('colorSpace')})
 * ```
 * @see zigzagBands, checkerCells, ringWaves
 */
export function stripeBands(slots: {
    angle: PropRef
    density: PropRef
    balance: PropRef
    softness: PropRef
    offset: PropRef
}): PatternPaint {
    // The analytically anti-aliased directional stripe mask (Quilez 1D filter, `balance` as the
    // duty threshold: 0 below it, 1 above), scrolled by the node's accumulated animation time
    // plus `offset`. The projection pivots on the canvas centre so `angle` spins in place.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.stripesMask, 'stripesMask', [
            uv,
            viewport,
            uniformOf(slots.angle, params),
            uniformOf(slots.density, params),
            animatedTime(params),
            uniformOf(slots.offset, params),
            uniformOf(slots.balance, params),
            uniformOf(slots.softness, params),
        ])
    }
}

/**
 * A mask of chevron (zigzag) stripes that scroll with the layer's clock.
 *
 * `count` is the number of stripe pairs across the canvas. `angle` is in degrees. `balance`
 * 0–1 is how much of each pair is the 0 side. `softness` 0–1 blurs the edges. `offset` shifts
 * the phase; 1 is one whole pair. Declare `animatedTime: {speed: 'speed'}` on the definition.
 *
 * @example
 * ```ts
 * const bands = zigzagBands({count: p('count'), angle: p('angle'), balance: p('balance'), softness: p('softness'), offset: p('offset')})
 * ```
 * @tip Reuse one `bands` value for both the color mix and an alpha mix. It is evaluated once.
 * @see stripeBands, checkerCells
 */
export function zigzagBands(slots: {
    count: PropRef
    angle: PropRef
    balance: PropRef
    softness: PropRef
    offset: PropRef
}): PatternPaint {
    // The fwidth-anti-aliased chevron stripe mask, scrolled by the node's accumulated animation
    // time plus `offset`. Shared: Chevron reads it twice (color mix + alpha mix).
    return shared((params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.chevronMask, 'chevronMask', [
            uv, viewport,
            uniformOf(slots.count, params),
            uniformOf(slots.angle, params),
            uniformOf(slots.balance, params),
            uniformOf(slots.softness, params),
            uniformOf(slots.offset, params),
            animatedTime(params),
        ])
    })
}

/**
 * A mask of concentric rings spreading out from a point, animated by the layer's clock.
 *
 * `center` is a position prop. `frequency` sets how tightly the rings pack; higher is more
 * rings. `thickness` 0–1 is how much of each ring period is ring (0.5 is even bands, 1 is
 * solid). `softness` 0–3 blurs the ring edges. `phase` is in radians. Declare
 * `animatedTime: {speed: 'speed'}` on the definition; positive speed moves the rings outward.
 *
 * @example
 * ```ts
 * paint: strokeOver({fill: p('colorB'), stroke: p('colorA'), mask: ringWaves({center: p('center'), frequency: p('frequency'), thickness: p('thickness'), softness: p('softness'), phase: p('phase')})})
 * ```
 * @tip Keep `softness` above 0. At exactly 0 the edge is an undefined hard step rather than a crisp one.
 * @see stripeBands, dotLattice
 */
export function ringWaves(slots: {
    center: PropRef
    frequency: PropRef
    thickness: PropRef
    softness: PropRef
    phase: PropRef
}): PatternPaint {
    // The mask of concentric rings emanating from `center` (a sin of the aspect-corrected
    // distance, banded by thickness/softness), animated by the node's accumulated time plus
    // `phase`. `center` arrives already transformed (`(x, 1 - y)`); the body un-flips it.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.ripplesMask, 'ripplesMask', [
            uniformOf(slots.center, params),
            uniformOf(slots.frequency, params),
            uniformOf(slots.softness, params),
            uniformOf(slots.thickness, params),
            uniformOf(slots.phase, params),
            animatedTime(params),
            uv,
            viewport,
        ])
    }
}

/**
 * Streaks that fall across the canvas in columns, each column at its own random speed.
 *
 * Returns two values: `mask` is the streak coverage, and `fade` runs from 0 at the tail to 1
 * at the leading edge, for a color ramp along each streak. `angle` is in degrees (90 falls
 * down, 0 moves right). `density` is the number of columns across the canvas.
 * `speedVariance` 0–1 spreads the per-column speeds. `trailLength` 0–1 is the streak length
 * as a fraction of the spacing. `strokeWidth` 0–1 is the width as a fraction of a column.
 * `rounding` 0–1 rounds the leading cap. `balance` 0–1 moves the midpoint of `fade`. Declare
 * `animatedTime: {speed: 'speed'}` on the definition.
 *
 * @example
 * ```ts
 * const streaks = fallingStreaks({angle: p('angle'), density: p('density'), speedVariance: p('speedVariance'), trailLength: p('trailLength'), strokeWidth: p('strokeWidth'), rounding: p('rounding'), balance: p('balance')})
 * const streakColor = strokeOver({fill: p('trailColor'), stroke: p('leadColor'), mask: streaks.fade, space: p('colorSpace')})
 * // paint: withAlpha(streakColor, times(alphaOf(streakColor), streaks.mask))
 * ```
 * @see stripeBands, dotLattice, withAlpha
 */
export function fallingStreaks(slots: {
    angle: PropRef
    density: PropRef
    speedVariance: PropRef
    trailLength: PropRef
    strokeWidth: PropRef
    rounding: PropRef
    balance: PropRef
}): {mask: PatternPaint; fade: PatternPaint} {
    // Directional streaks with per-column random speed/phase and a rounded leading cap, driven
    // by the node's accumulated animation time. One fused body returns vec2(mask, fade); the
    // two readers share it through `shared` + `local`.
    const field = shared<GpuFragmentParams>((params) => {
        const {uv, viewport} = paintFrame(params)
        return local(call(patternPaints.fallingLinesField, 'fallingLinesField', [
            uv, viewport,
            uniformOf(slots.angle, params),
            uniformOf(slots.density, params),
            uniformOf(slots.speedVariance, params),
            uniformOf(slots.trailLength, params),
            uniformOf(slots.strokeWidth, params),
            uniformOf(slots.rounding, params),
            uniformOf(slots.balance, params),
            animatedTime(params),
        ]), 'fallingField')
    })
    return {mask: (params) => field(params).member('x'), fade: (params) => field(params).member('y')}
}

/**
 * A mask of round dots on a square grid, with optional row stagger, drift and twinkle.
 *
 * `density` counts dots along the canvas height. `dotSize` 0–1 is the dot diameter as a
 * fraction of a cell; 1 touches the neighbours. `offset` 0–1 shifts every other row (0.5 is
 * the polka-dot stagger). `speedVariance` 0–1 gives rows different drift speeds. `twinkle`
 * 0–1 pulses each dot on its own phase. Declare `animatedTime: {speed: 'speed'}` on the
 * definition; the drift follows that clock while the twinkle keeps running at speed 0.
 *
 * @example
 * ```ts
 * paint: withAlpha(p('color'), dotLattice({density: p('density'), dotSize: p('dotSize'), offset: p('offset'), speedVariance: p('speedVariance'), twinkle: p('twinkle')}))
 * ```
 * @tip Add `sampleMapsAtCellCentres` when `dotSize` or `twinkle` is map-driven, so dots grow whole instead of being clipped.
 * @see sampleMapsAtCellCentres, gridLines, ringWaves
 */
export function dotLattice(slots: {
    density: PropRef
    dotSize: PropRef
    offset: PropRef
    speedVariance: PropRef
    twinkle: PropRef
}): PatternPaint {
    // The coverage of a square lattice of anti-aliased discs with optional brick-style row
    // stagger, per-row animated drift (the node's accumulated time × per-row random speed), and
    // a per-dot twinkle on the GLOBAL clock (`ctx.time`, so it oscillates while paused).
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.dotGridAlpha, 'dotGridAlpha', [
            uv,
            viewport,
            uniformOf(slots.density, params),
            uniformOf(slots.dotSize, params),
            uniformOf(slots.offset, params),
            uniformOf(slots.speedVariance, params),
            uniformOf(slots.twinkle, params),
            animatedTime(params),
            params.ctx.time,
        ])
    }
}

/**
 * Square grid lines drawn in a `cellFrame`, plus a per-cell shade for the fill.
 *
 * Returns `lines` (1 on a line, 0 inside a cell) and `shade` (a brightness factor around 1
 * for `vary`). `thickness` 1 draws lines about 2% of a cell wide; 0 draws none. `softness`
 * 0–1 blurs the lines. `variation` 0–1 sets how far `shade` strays from 1. Draw it with the
 * `'flippedY'` convention.
 *
 * @example
 * ```ts
 * const field = gridLines({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'flippedY'}), thickness: p('thickness'), softness: p('softness'), variation: p('variation')})
 * // paint: strokeOver({fill: vary(p('cellColor'), field.shade), stroke: p('color'), mask: field.lines, space: p('colorSpace')})
 * ```
 * @tip Rotate the frame 45 degrees for a diamond crosshatch.
 * @see cellFrame, hexLines, triangleLines, vary
 */
export function gridLines(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    variation: PropRef
}): {lines: PatternPaint; shade: PatternPaint} {
    // Quilez axis integrals over the dpdx/dpdy footprint of the frame coordinate; the body
    // returns vec2(lineMask, cellVariationFactor) and both readers share it.
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.gridField, 'gridField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params), uniformOf(slots.variation, params),
    ]), 'gridField'))
    return {lines: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * Honeycomb lines (pointy-top hexagons) drawn in a `cellFrame`, plus a per-cell shade.
 *
 * Returns `lines` (1 on a line) and `shade` (a brightness factor around 1 for `vary`).
 * `thickness` 1 draws lines about 2% of a cell wide; 0 draws none. `softness` 0–1 blurs the
 * lines. `variation` 0–1 sets how far `shade` strays from 1. Draw it with the `'clockwise'`
 * convention.
 *
 * @example
 * ```ts
 * const field = hexLines({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}), thickness: p('thickness'), softness: p('softness'), variation: p('variation')})
 * ```
 * @see cellFrame, gridLines, isoCubeFaces, vary
 */
export function hexLines(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    variation: PropRef
}): {lines: PatternPaint; shade: PatternPaint} {
    // Pointy-top hex tiling (period √3 × 1) in the frame coordinate; returns
    // vec2(lineMask, variationFactor), shared by both readers.
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.hexGridField, 'hexGridField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params), uniformOf(slots.variation, params),
    ]), 'hexField'))
    return {lines: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * A lattice of equilateral triangles drawn in a `cellFrame`, with rows that can drift.
 *
 * Returns `lines` (1 on a line) and `shade` (a brightness factor around 1 per triangle for
 * `vary`). `thickness` 1 draws lines about 4% of a cell wide; 0 draws none. `softness` 0–1
 * blurs the lines. `variation` 0–1 sets how far `shade` strays from 1. `speedVariance` 0–1
 * gives rows different drift speeds. Declare `animatedTime: {speed: 'speed'}` on the
 * definition; speed 0 holds still. Draw it with the `'flippedY'` convention.
 *
 * @example
 * ```ts
 * const field = triangleLines({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'flippedY'}), thickness: p('thickness'), softness: p('softness'), variation: p('variation'), speedVariance: p('speedVariance')})
 * ```
 * @see cellFrame, gridLines, hexLines, vary
 */
export function triangleLines(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    variation: PropRef
    speedVariance: PropRef
}): {lines: PatternPaint; shade: PatternPaint} {
    // Skewed equilateral-triangle lattice with per-row animated drift (the node's accumulated
    // time × per-row random speed); returns vec2(lineMask, variationFactor), shared.
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.triangularGridField, 'triangularGridField', [
        slots.frame(params),
        uniformOf(slots.thickness, params),
        uniformOf(slots.softness, params),
        uniformOf(slots.variation, params),
        uniformOf(slots.speedVariance, params),
        animatedTime(params),
    ]), 'triField'))
    return {lines: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * A brick wall: staggered rows with mortar gaps, optional rotation and per-row drift.
 *
 * Returns `bricks` (1 on a brick, 0 in the mortar) and `shade` (a brightness factor around 1
 * per brick for `vary`). `cellsX` is bricks per row across the canvas width and `cellsY` the
 * number of rows down its height. `mortar` 0–1 widens the gaps (1 is 30% of a brick), equal
 * in pixels both ways. `softness` 0–1 blurs the edges. `variation` 0–1 sets how far `shade`
 * strays from 1. `rotation` is in degrees. `offset` slides the rows; 1 is one brick.
 * `speedVariance` 0–1 and `seed` randomise the per-row drift. Declare
 * `animatedTime: {speed: 'speed'}` on the definition. It brings its own frame, so there is
 * no `frame` slot.
 *
 * @example
 * ```ts
 * const field = brickCourses({cellsX: p('cellsX'), cellsY: p('cellsY'), mortar: p('mortar'), softness: p('softness'), variation: p('variation'), rotation: p('rotation'), offset: p('offset'), speedVariance: p('speedVariance'), seed: p('seed')})
 * // paint: strokeOver({fill: p('colorMortar'), stroke: vary(p('colorBrick'), field.shade), mask: field.bricks, space: p('colorSpace')})
 * ```
 * @see gridLines, cellFrame, vary
 */
export function brickCourses(slots: {
    cellsX: PropRef
    cellsY: PropRef
    mortar: PropRef
    softness: PropRef
    variation: PropRef
    rotation: PropRef
    offset: PropRef
    speedVariance: PropRef
    seed: PropRef
}): {bricks: PatternPaint; shade: PatternPaint} {
    // Brick keeps its OWN frame part (`patternPaints.brickFrame`) rather than the shared
    // `cellFrame` slot — its raw-UV, non-uniform-cell framing is not byte-equivalent to any
    // lattice frame convention (Gate-C candidate; see the note on `brickFrame` in the kit).
    // Returns vec2(brickMask, variationFactor), shared by both readers.
    const field = shared<GpuFragmentParams>((params) => {
        const {uv, viewport} = paintFrame(params)
        return local(call(patternPaints.brickField, 'brickField', [
            uv, viewport,
            uniformOf(slots.cellsX, params),
            uniformOf(slots.cellsY, params),
            uniformOf(slots.mortar, params),
            uniformOf(slots.softness, params),
            uniformOf(slots.variation, params),
            uniformOf(slots.rotation, params),
            uniformOf(slots.offset, params),
            uniformOf(slots.speedVariance, params),
            uniformOf(slots.seed, params),
            animatedTime(params),
        ]), 'brickField')
    })
    return {bricks: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * A maze of quarter-circle arcs, two per tile, flipped at random so they join into flowing
 * curves.
 *
 * Returns the arc mask (1 on an arc). `thickness` 2 draws arcs about 2% of a tile wide; 0
 * draws none. `softness` 0–1 blurs the edges. `seed` picks a different set of flips. Draw it
 * in a `'clockwise'` `cellFrame`.
 *
 * @example
 * ```ts
 * paint: strokeOver({fill: p('colorA'), stroke: p('colorB'), mask: truchetArcs({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}), thickness: p('thickness'), softness: p('softness'), seed: p('seed')}), space: p('colorSpace')})
 * ```
 * @see cellFrame, gridLines, weaveThreads
 */
export function truchetArcs(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    seed: PropRef
}): PatternPaint {
    // Quarter-circle arc tiles, orientation hashed per tile (`seed` offsets the cell coordinate
    // before the hash, reshuffling the maze).
    return (params) => call(patternPaints.truchetField, 'truchetField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params), uniformOf(slots.seed, params),
    ])
}

/**
 * Two sets of threads woven over and under each other, returned as a finished color.
 *
 * `gap` 0–0.5 is the empty margin on each side of a thread, as a fraction of a cell; 0 packs
 * them tight. `colors` is `[horizontal, vertical]`. The gaps are transparent, so this goes
 * straight into `paint:`. Draw it in a `'plain'` `cellFrame`.
 *
 * @example
 * ```ts
 * paint: weaveThreads({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'plain'}), gap: p('gap'), colors: [p('colorA'), p('colorB')]})
 * ```
 * @see cellFrame, gridLines, truchetArcs
 */
export function weaveThreads(slots: {
    frame: PatternPaint
    gap: PropRef
    /** `[horizontal, vertical]` thread colors. */
    colors: [PropRef, PropRef]
}): PatternPaint {
    // Interlaced horizontal/vertical thread bands with a checkerboard over-under rule,
    // compositing the two colors by per-thread alpha weight — not a color-space mix, so there
    // is no `space` slot.
    return (params) => call(patternPaints.weaveColor, 'weaveColor', [
        slots.frame(params),
        uniformOf(slots.gap, params),
        uniformOf(slots.colors[0], params),
        uniformOf(slots.colors[1], params),
    ])
}

/**
 * Tumbling blocks: three shaded rhombus faces per hexagon that read as a field of 3D cubes.
 *
 * Returns three values. `tone` is the face shading (1 on top, 0.74 on the right, 0.5 on the
 * left) to multiply into the cube color. `random` is a 0–1 value per cube for varying cube
 * colors. `wire` is the edge-line mask (1 on an edge). `thickness` 1 draws edges about 4% of
 * a cell wide; 0 draws none. `softness` 0–1 blurs them. Draw it in a `'clockwise'`
 * `cellFrame`.
 *
 * @example
 * ```ts
 * const faces = isoCubeFaces({frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}), thickness: p('thickness'), softness: p('softness')})
 * // paint: opaque(mixOf(times(rgbOf(cubeColor), faces.tone), rgbOf(p('lineColor')), faces.wire))
 * ```
 * @see cellFrame, hexLines, times, opaque
 */
export function isoCubeFaces(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
}): {random: PatternPaint; tone: PatternPaint; wire: PatternPaint} {
    // Rhombille field in a flat-top hex tiling (period 1 × √3, the transpose of hexLines);
    // returns vec3(cubeRand, faceTone, edgeWire), shared by the three readers.
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.isoCubeField, 'isoCubeField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params),
    ]), 'isoCubeField'))
    return {
        random: (params) => field(params).member('x'),
        tone: (params) => field(params).member('y'),
        wire: (params) => field(params).member('z'),
    }
}

// ═══ Print-screen parts (halftone) ═════════════════════════════════════════════════════════════

/**
 * The dot coverage of one halftone screen: 1 inside a dot, 0 between, with the dot size set
 * by an ink amount.
 *
 * A raw building block for a custom press inside a gather `build`. `uv` is the pixel's uv
 * and `aspect` the canvas aspect (`ctx.uv`, `ctx.aspect`). `angle` is the screen angle in
 * degrees. `intensity` 0–1 is the ink amount at this pixel; 0 leaves no dot. `frequency` is
 * the number of dots down the canvas height.
 *
 * @example
 * ```ts
 * const dots = dotScreenMask(ctx.uv, ctx.aspect, uniforms.angle, intensity, uniforms.frequency)
 * ```
 * @see inkTransmission, cmykPress, dotScreen
 */
export function dotScreenMask(uv: Expr, aspect: Expr, angle: Expr, intensity: Expr, frequency: Expr): Expr {
    // The rotated dot-screen coverage for one ink plate; dot radius follows `intensity`.
    return call(patternPaints.halftonePlateGrid, 'halftonePlateGrid', [uv, aspect, angle, intensity, frequency])
}

/**
 * How much light passes one ink plate: white where the dot mask is 0, the ink color where
 * it is 1.
 *
 * Returns RGB. Multiply the plates together over a paper color to lay the inks down in
 * order. The ink's alpha scales its strength.
 *
 * @example
 * ```ts
 * printed = printed.mul(inkTransmission(uniforms.cyanColor, dotScreenMask(ctx.uv, ctx.aspect, uniforms.cyanAngle, cyan, uniforms.frequency)))
 * ```
 * @see dotScreenMask, cmykPress
 */
export function inkTransmission(inkColor: Expr, inkMask: Expr): Expr {
    // Subtractive ink lay-down: white where the plate leaves paper bare, fading toward the ink
    // color (× ink alpha) where the dot covers. Plates multiply together.
    return call(patternPaints.halftoneTransmission, 'halftoneTransmission', [inkColor, inkMask])
}

/**
 * A classic halftone filter: the layer inside it redrawn as a grid of dots that grow where
 * it is bright.
 *
 * Goes in `effect:`. `angle` is the screen angle in degrees (45 is the print default).
 * `frequency` is the number of dots down the canvas height. The dots keep the child's own
 * color; between dots the output is transparent.
 *
 * @example
 * ```ts
 * effect: dotScreen({angle: p('angle'), frequency: p('frequency')})
 * ```
 * @see cmykPress, chosenBy, quantise
 */
export function dotScreen(slots: {angle: PropRef; frequency: PropRef}): GatherEffect {
    // The child sampled once (straight alpha), its brightness modulating a single rotated dot
    // plate; the body multiplies all four channels by the dot pattern.
    return gather({
        resultAlpha: 'straight',
        build: ({sampleStraight, ctx, uniforms}): Expr => {
            const childColor = sampleStraight(ctx.uv)
            return call(patternPaints.halftoneClassic, 'halftoneClassic', [
                childColor, ctx.uv, ctx.aspect, uniformOf(slots.angle, {uniforms}), uniformOf(slots.frequency, {uniforms}),
            ])
        },
    })
}

/**
 * One ink plate of a `cmykPress`: which channel it prints, its screen-angle prop (degrees)
 * and its ink-color prop.
 */
export interface InkPlateSpec {
    readonly channel: 'cyan' | 'magenta' | 'yellow' | 'black'
    readonly screenAngle: PropRef
    readonly ink: PropRef
}

/**
 * Declares one ink plate for `cmykPress`.
 *
 * `channel` is `'cyan'`, `'magenta'`, `'yellow'` or `'black'`. `screenAngle` is an angle prop
 * in degrees and `ink` a color prop. Plates print in the order you list them.
 *
 * @example
 * ```ts
 * inkPlate({channel: 'cyan', screenAngle: p('cyanAngle'), ink: p('cyanColor')})
 * ```
 * @see cmykPress
 */
export function inkPlate(spec: InkPlateSpec): InkPlateSpec {
    return spec
}

const PLATE_CHANNELS = {
    cyan: {fn: () => patternPaints.halftoneChannelC, hint: 'halftoneChannelC'},
    magenta: {fn: () => patternPaints.halftoneChannelM, hint: 'halftoneChannelM'},
    yellow: {fn: () => patternPaints.halftoneChannelY, hint: 'halftoneChannelY'},
    black: {fn: () => patternPaints.halftoneChannelK, hint: 'halftoneChannelK'},
} as const

/**
 * A four-color print filter: the layer inside it separated into ink plates, each screened
 * into dots at its own angle and laid down on paper in order.
 *
 * Goes in `effect:`. `paper` is the color shown where no ink lands. `frequency` is the number
 * of dots down the canvas height. `misprint` offsets each plate in uv (0 is perfect
 * registration, 0.005 is visible fringing), in a direction that turns a quarter turn per
 * plate starting from `misprintAngle` (degrees). `plates` is a list from `inkPlate`; the
 * standard screen angles are cyan 15, magenta 75, yellow 0, black 45. The result keeps the
 * child's alpha.
 *
 * @example
 * ```ts
 * effect: cmykPress({paper: p('paperColor'), frequency: p('frequency'), misprint: p('misprint'), misprintAngle: p('misprintAngle'), plates: [inkPlate({channel: 'cyan', screenAngle: p('cyanAngle'), ink: p('cyanColor')}), inkPlate({channel: 'black', screenAngle: p('blackAngle'), ink: p('blackColor')})]})
 * ```
 * @tip Any subset of the four plates works. Two plates give a duotone or risograph look.
 * @see inkPlate, dotScreen, dotScreenMask, inkTransmission
 */
export function cmykPress(slots: {
    paper: PropRef
    frequency: PropRef
    misprint: PropRef
    misprintAngle: PropRef
    plates: InkPlateSpec[]
}): GatherEffect {
    // One subtractive plate per ink, laid down in order. Each plate samples the child at its
    // own registration offset around `misprintAngle` (quarter turns per plate index;
    // mis-registration shows as color fringing), reads its CMYK channel, screens it through
    // `dotScreenMask`, and multiplies its `inkTransmission` into the paper. Alpha follows the
    // un-offset centre sample; every tap is straight-alpha.
    return gather({
        resultAlpha: 'straight',
        build: ({sampleStraight, ctx, uniforms}): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, {uniforms})
            let printed = u(slots.paper).member('rgb')
            slots.plates.forEach((plate, index) => {
                const sample = sampleStraight(call(patternPaints.halftonePlateUV, 'halftonePlateUV', [
                    ctx.uv, u(slots.misprintAngle), floatE(index * 90), u(slots.misprint), ctx.aspect,
                ]))
                const channel = PLATE_CHANNELS[plate.channel]
                const intensity = call(channel.fn(), channel.hint, [sample])
                printed = printed.mul(inkTransmission(u(plate.ink), dotScreenMask(ctx.uv, ctx.aspect, u(plate.screenAngle), intensity, u(slots.frequency))))
            })
            return vec4(printed, sampleStraight(ctx.uv).member('a'))
        },
    })
}

/**
 * Picks one whole effect recipe from a select prop, decided when the shader compiles.
 *
 * `prop` must be a `compileTime: true` prop. `read` turns its raw value into one of the keys
 * of `recipes`; accept both the option string and its transformed value. Every recipe must
 * be a gather effect with the same alpha handling and no setup step, or this throws as soon
 * as it is called.
 *
 * @example
 * ```ts
 * effect: chosenBy(p('style'), (raw) => (raw === 'cmyk' || raw === 1 ? 'cmyk' : 'classic'), {classic: dotScreen({angle: p('angle'), frequency: p('frequency')}), cmyk: cmykPress({paper: p('paperColor'), frequency: p('frequency'), misprint: p('misprint'), misprintAngle: p('misprintAngle'), plates})})
 * ```
 * @tip Changing the prop recompiles the shader, which is the point: each style pays only for its own code.
 * @see dotScreen, cmykPress
 */
export function chosenBy<K extends string>(prop: PropRef, read: (raw: unknown) => K, recipes: Record<K, GatherEffect>): GatherEffect {
    // Compile-time recipe switch on a structural enum prop. `read` maps the raw compile-time
    // value (robust to a pre-transform string) to a case key. All cases must share their alpha
    // discipline and carry no setup hooks, because the merged gather declares one of each.
    const cases = Object.values(recipes) as GatherEffect[]
    const resultAlpha = cases[0]?.resultAlpha
    for (const c of cases) {
        if (c.resultAlpha !== resultAlpha || c.setup) throw new Error('std: chosenBy recipes must share resultAlpha and carry no setup hooks')
    }
    return gather({
        resultAlpha,
        build: (params): Expr => recipes[read(params.propValues[prop.name])].build(params),
    })
}

// ═══ Print-screen parts (dither) ═══════════════════════════════════════════════════════════════

// Compile-time enum readers — robust to a raw string (a preset loaded before the prop transform
// ran) as well as the bridge-mapped number.
const ditherPatternOf = (raw: unknown): number => {
    if (typeof raw === 'number') return raw
    const patterns: Record<string, number> = {bayer2: 0, bayer4: 1, bayer8: 2, clusteredDot: 3, blueNoise: 4, whiteNoise: 5, floydSteinberg: 6}
    return patterns[raw as string] ?? 1
}
const ditherColorModeOf = (raw: unknown): number => (typeof raw === 'number' ? raw : raw === 'source' ? 1 : 0)

/** A value computed per pixel from the layer inside a gather effect, such as a dither level. */
export type GatherStage = (params: RttFilterParams) => Expr

/**
 * The pixel grid a dither works on, from `pixelGrid`: the cell `size` prop, the cell
 * coordinate `coord`, and `source`, the child's color sampled once per cell.
 */
export interface PixelGrid {
    readonly size: PropRef
    readonly coord: GatherStage
    readonly source: GatherStage
}

/**
 * The pixel grid of a dither: the layer inside it chopped into square cells of a given size.
 *
 * `size` is a prop in pixels of the authored frame, so the cell count stays the same when the
 * canvas renders at a higher resolution. Build it once and pass the same grid to `quantise`
 * and `ditherInks`.
 *
 * @example
 * ```ts
 * const grid = pixelGrid({size: p('pixelSize')})
 * ```
 * @see quantise, ditherInks
 */
export function pixelGrid(slots: {size: PropRef}): PixelGrid {
    // The grid is sized against the LOGICAL (authored-frame) resolution so the dot count stays
    // constant under infinite-canvas resolution scaling. `coord` is the dither cell coordinate,
    // `source` the child sampled once per cell (the pixellated source color); both shared.
    const coord = shared<RttFilterParams>((params) =>
        local(call(patternPaints.ditherCoord, 'ditherCoord', [params.ctx.uv, params.ctx.logicalViewportSize, uniformOf(slots.size, params)]), 'ditherCoord'))
    const source = shared<RttFilterParams>((params) =>
        local(params.sampleStraight(call(patternPaints.ditherPixUV, 'ditherPixUV', [coord(params), uniformOf(slots.size, params), params.ctx.logicalViewportSize])), 'ditherSource'))
    return {size: slots.size, coord, source}
}

/** @internal */
export function orderedThreshold(pattern: number, coord: Expr): Expr {
    // The threshold FIELD for a compile-time pattern code (0 bayer2, 1 bayer4, 2 bayer8,
    // 3 clusteredDot, 4 blueNoise, 5 whiteNoise) at the dither cell coordinate — the closed-form
    // Bayer / clustered-dot lattice values or the blue/white-noise hashes. `quantise` feeds it
    // to `ditherOrderedResult`. Floyd–Steinberg (6) is not a threshold field — its quantisation
    // diffuses error serially — and unknown codes fall back to white noise.
    if (pattern === 4) return call(patternPaints.ditherBlueNoise, 'ditherBlueNoise', [coord])
    if (pattern === 5 || pattern > 6 || pattern < 0) return call(patternPaints.ditherWhiteNoise, 'ditherWhiteNoise', [coord])
    const periodic = call(patternPaints.ditherPeriodic, 'ditherPeriodic', [coord])
    switch (pattern) {
        case 0: return periodic.member('x') // bayer2
        case 1: return periodic.member('y') // bayer4
        case 2: return periodic.member('z') // bayer8
        default: return periodic.member('w') // clusteredDot
    }
}

/**
 * Reduces each grid cell of the layer inside to 0 or 1 using a dither pattern.
 *
 * Returns a per-pixel level for `ditherInks`. `pattern` is a `compileTime: true` select prop
 * with one of `bayer2`, `bayer4`, `bayer8`, `clusteredDot`, `blueNoise`, `whiteNoise` or
 * `floydSteinberg`. `threshold` 0–1 shifts the cut point; 0.5 is neutral. `spread` 0–1 is
 * how much of the brightness range dithers; lower leaves more solid areas.
 *
 * @example
 * ```ts
 * const levels = quantise({grid, pattern: p('pattern'), threshold: p('threshold'), spread: p('spread')})
 * ```
 * @tip Floyd–Steinberg is the costliest pattern: it re-reads 64 cells per pixel. Bayer 4 is the cheap default.
 * @see pixelGrid, ditherInks
 */
export function quantise(slots: {grid: PixelGrid; pattern: PropRef; threshold: PropRef; spread: PropRef}): GatherStage {
    // The ordered modes compose an `orderedThreshold` field with the cell luminance (Rec.601 ×
    // alpha); the compile-time Floyd–Steinberg mode has no threshold field — it samples the 64
    // block-cell luminances and runs its tile-confined serpentine error diffusion, reading off
    // this fragment's cell. Confining diffusion to an 8×8 tile keeps the pattern temporally
    // stable.
    return (params) => {
        const u = (ref: PropRef) => uniformOf(ref, params)
        const threshold = u(slots.threshold)
        const spread = u(slots.spread)
        const coord = slots.grid.coord(params)
        const pattern = ditherPatternOf(params.propValues[slots.pattern.name])
        if (pattern === 6) {
            const gridRes = params.ctx.logicalViewportSize
            const pixelSize = u(slots.grid.size)
            const blockOrigin = call(patternPaints.ditherBlockOrigin, 'ditherBlockOrigin', [coord])
            const lic = call(patternPaints.ditherLocalCellIndex, 'ditherLocalCellIndex', [coord, blockOrigin])
            const lums: Expr[] = []
            for (let ly = 0; ly < 8; ly++) {
                for (let lx = 0; lx < 8; lx++) {
                    const cellUV = call(patternPaints.ditherCellUV, 'ditherCellUV', [blockOrigin, floatE(lx), floatE(ly), pixelSize, gridRes])
                    lums.push(call(patternPaints.ditherLuma, 'ditherLuma', [params.sampleStraight(cellUV)]))
                }
            }
            return call(patternPaints.ditherFloydSteinberg, 'ditherFloydSteinberg', [arrayExpr('f32', lums), lic, threshold, spread])
        }
        const luminance = call(patternPaints.ditherLuma, 'ditherLuma', [slots.grid.source(params)])
        return call(patternPaints.ditherOrderedResult, 'ditherOrderedResult', [orderedThreshold(pattern, coord), luminance, threshold, spread])
    }
}

/**
 * A dither filter: colors the 0/1 levels from `quantise`, either with two colors of your own
 * or with darkened and brightened versions of the layer's own pixels.
 *
 * Goes in `effect:`. `mode` is a `compileTime: true` select prop, `'custom'` or `'source'`.
 * `colors` is `[dark, light]` for custom mode; a transparent dark color lets the background
 * show through.
 *
 * @example
 * ```ts
 * effect: ditherInks({grid, levels: quantise({grid, pattern: p('pattern'), threshold: p('threshold'), spread: p('spread')}), mode: p('colorMode'), colors: [p('colorA'), p('colorB')]})
 * ```
 * @see pixelGrid, quantise, dotScreen
 */
export function ditherInks(slots: {
    grid: PixelGrid
    levels: GatherStage
    mode: PropRef
    colors: [PropRef, PropRef]
}): GatherEffect {
    // The compile-time `mode` branches — custom mixes `colors[0]`→`colors[1]` by the level,
    // source darkens (×0.3) / brightens (×1.3) the grid's own pixellated child color. Every tap
    // is straight-alpha.
    return gather({
        resultAlpha: 'straight',
        build: (params): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, params)
            const levels = slots.levels(params)
            if (ditherColorModeOf(params.propValues[slots.mode.name]) === 0) {
                return call(patternPaints.ditherComposeCustom, 'ditherComposeCustom', [u(slots.colors[0]), u(slots.colors[1]), levels])
            }
            return call(patternPaints.ditherComposeSource, 'ditherComposeSource', [slots.grid.source(params), levels])
        },
    })
}
