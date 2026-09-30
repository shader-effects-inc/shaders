/**
 * std/effects/reveal — wipes and dissolves: words that make the layer inside disappear in a chosen order.
 *
 * `reveal` is the effect. It takes a **coverage**, a number per pixel from 0 to 1 saying when
 * that pixel goes (0 first, 1 last). As `progress` climbs from 0 to 1 the pixels below it fade
 * out across a soft front `feather` wide. The `coverage` words produce the ordering (a straight
 * edge, a circle, a clock sweep, a grid of cells, a noise pattern) and the modifiers reshape it
 * (`.folded()`, `.tiled(n)`, `.bands(n)`, `.shuffled()`). Only the child's alpha changes; its
 * colors stay put. A wipe reads as its recipe:
 *
 *     effect: reveal({
 *         coverage: coverage.directional(p('angle')).bands(p('barCount')).shuffled(),
 *         progress: p('progress'), feather: p('softness'), invert: p('invert'),
 *     })
 */
// Maintainer notes: every pointwise wipe is the same effect with an interchangeable coverage
// coordinate. `reveal` lowers to the kit's reveal tail (revealMask → applyReveal), scaling the
// STRAIGHT-alpha child's alpha with RGB preserved (no render-to-texture). Coverages come from the
// `coverage` namespace, one producer per coordinate family (directional projection, radial
// distance, angular sweep, cell grid, organic noise), and compose through the modifier methods.
import type {Expr} from '../../gpu/contract'
import {call, floatE} from '../../gpu/porters'
import {reveal as kit} from '../../gpu/kit/index'
import type {FilterParams} from '../../gpu/scaffolds/pointwiseFilter'
import type {PointwiseEffect} from '../types'
import {resolveArg, type ArgSpec} from '../invoke'
import type {PropRef} from '../values'

/**
 * The order pixels disappear in: a number per pixel from 0 (goes first) to 1 (goes last).
 *
 * Get one from a `coverage` word, then reshape it with the methods below. Each method returns
 * a new coverage, so they chain. Pass the result to `reveal`.
 *
 * @example
 * ```ts
 * coverage: coverage.directional(p('angle')).folded()
 * ```
 * @see coverage, reveal
 */
export class Coverage {
    constructor(readonly build: (params: FilterParams) => Expr) {}

    /**
     * Mirror the order about its middle so one moving edge becomes two opening outward, like
     * barn doors. The center goes first, both ends last.
     */
    folded(): Coverage {
        return new Coverage((params) => call(kit.foldAboutCenter, 'foldAboutCenter', [this.build(params)]))
    }

    /**
     * Repeat the order `count` times so every strip wipes at once, like venetian blinds.
     */
    tiled(count: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.tilePhase, 'tilePhase', [this.build(params), resolveArg(count, params)]))
    }

    /**
     * Cut the order into `count` bands that vanish whole, one after another (rings popping
     * over a radial coverage). Chain `.shuffled()` to randomize the band order.
     */
    bands(count: ArgSpec): BandedCoverage {
        return new BandedCoverage(this, count)
    }
}

/**
 * A coverage cut into bands. Bands vanish in order, or in a fixed random order after `.shuffled()`.
 *
 * @example
 * ```ts
 * coverage: coverage.radial(p('center')).bands(p('rings'))
 * ```
 * @see Coverage
 */
export class BandedCoverage extends Coverage {
    constructor(base: Coverage, count: ArgSpec) {
        super((params) =>
            call(kit.bandMidpointCoord, 'bandMidpointCoord', [base.build(params), resolveArg(count, params)]))
        this.base = base
        this.count = count
    }

    private readonly base: Coverage
    private readonly count: ArgSpec

    /**
     * Vanish the bands in a random order that never changes between frames (random bars).
     */
    shuffled(): Coverage {
        // A hash of the band index, no time dependence, so only progress moves it.
        return new Coverage((params) =>
            call(kit.bandShuffleCoord, 'bandShuffleCoord', [this.base.build(params), resolveArg(this.count, params)]))
    }
}

/**
 * The canvas cut into a grid of square cells, waiting for you to pick the order the cells go in.
 *
 * Not a coverage on its own. Call `.diamond()`, `.checker()` or `.shuffled()` to get one.
 *
 * @example
 * ```ts
 * coverage: coverage.cells(p('blockSize')).shuffled()
 * ```
 * @see coverage
 */
export class CellCoverage {
    constructor(private readonly size: ArgSpec) {}

    /** Every cell opens as a growing diamond from its center, all at once. */
    diamond(): Coverage {
        return new Coverage((params) =>
            call(kit.cellDiamondCoord, 'cellDiamondCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(this.size, params)]))
    }

    /** Cells vanish in checkerboard order: one color of squares sweeps corner to corner, then the other. */
    checker(): Coverage {
        return new Coverage((params) =>
            call(kit.cellCheckerCoord, 'cellCheckerCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(this.size, params)]))
    }

    /** Cells vanish in a random order that never changes between frames (a block dissolve). */
    shuffled(): Coverage {
        // A hash of the integer cell index, no time dependence, so only progress moves it.
        return new Coverage((params) =>
            call(kit.cellShuffleCoord, 'cellShuffleCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(this.size, params)]))
    }
}

/** Which way an angular sweep turns. Baked into the shader, so a change rebuilds it. */
export type AngularDirection = 'cw' | 'ccw' | 'both'

const ANGULAR_MODES: Record<AngularDirection, number> = {cw: 0, ccw: 1, both: 2}

/**
 * The orders a wipe can follow. Each word returns a `Coverage` for `reveal`.
 *
 * All of them read the canvas in uv (0–1 across, y down) and correct for its aspect ratio, so
 * a circle stays round and a diagonal stays at its true angle.
 *
 * @example
 * ```ts
 * coverage: coverage.radial(p('center'))
 * ```
 * @see reveal, Coverage
 */
export const coverage = {
    /**
     * A straight edge sweeping across the canvas.
     *
     * `angleDeg` is in degrees; 0 wipes left to right, 90 top to bottom. The edge reaches both
     * ends of the canvas exactly at whatever angle you give.
     *
     * @example
     * ```ts
     * coverage: coverage.directional(p('angle'))
     * ```
     * @see radial, angular
     */
    directional(angleDeg: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.directionalCoord, 'directionalCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(angleDeg, params)]))
    },

    /**
     * A circle growing out from `center`, an iris wipe.
     *
     * `center` is a position prop (`transform: transformPosition`). The circle is sized to the
     * farthest corner, so it clears the whole canvas even from an off-center start.
     *
     * @example
     * ```ts
     * coverage: coverage.radial(p('center'))
     * ```
     * @see directional, angular
     */
    radial(center: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.radialCornerNormCoord, 'radialCornerNormCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(center, params)]))
    },

    /**
     * A clock hand sweeping around `center`.
     *
     * `startDeg` is where the sweep begins, in degrees. `direction` is `'cw'`, `'ccw'` or
     * `'both'` (two hands opening from the start and meeting opposite it), either as a literal
     * or as a select prop declared `compileTime: true`.
     *
     * @example
     * ```ts
     * coverage: coverage.angular(p('center'), p('startAngle'), p('direction'))
     * ```
     * @see radial, directional
     */
    angular(center: ArgSpec, startDeg: ArgSpec, direction: PropRef | AngularDirection): Coverage {
        // `direction` is STRUCTURAL: baked into the compiled shader as a mode literal.
        return new Coverage((params) => {
            const value = typeof direction === 'string'
                ? direction
                : ((params.propValues[direction.name] as AngularDirection | undefined) ?? 'cw')
            const mode = ANGULAR_MODES[value] ?? 0
            return call(kit.angularCoord, 'angularCoord', [
                params.ctx.uv, params.ctx.aspect,
                resolveArg(center, params), resolveArg(startDeg, params), floatE(mode),
            ])
        })
    },

    /**
     * A grid of square cells `size` wide.
     *
     * `size` is a fraction of the canvas width (0.01–0.5). Finish with `.diamond()`,
     * `.checker()` or `.shuffled()` to say how the cells go.
     *
     * @example
     * ```ts
     * coverage: coverage.cells(p('blockSize')).checker()
     * ```
     * @see noise
     */
    cells(size: ArgSpec): CellCoverage {
        return new CellCoverage(size)
    },

    /**
     * Soft blobs eating the image away, the classic film dissolve.
     *
     * `scale` sets the blob size (0.5–20; higher is smaller blobs). `seed` picks a different
     * pattern. The pattern itself never moves; only `progress` does.
     *
     * @example
     * ```ts
     * coverage: coverage.noise(p('scale'), p('seed'))
     * ```
     * @see cells
     */
    noise(scale: ArgSpec, seed: ArgSpec): Coverage {
        // A stable 3-octave fbm, no time dependence.
        return new Coverage((params) =>
            call(kit.fbmCoverageCoord, 'fbmCoverageCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(scale, params), resolveArg(seed, params)]))
    },
}

/** What `reveal` needs: the order pixels go in, and the three timing props every wipe shares. */
export interface RevealSlots {
    /** The order pixels disappear in, from a `coverage` word. */
    coverage: Coverage
    /** How far the wipe has gone, 0–1 (0 fully visible, 1 fully gone). */
    progress: ArgSpec
    /** Width of the soft edge, 0–1 (0 is a hard edge). */
    feather: ArgSpec
    /** A boolean prop (`transform: transformBoolean`) that reverses the order. */
    invert: ArgSpec
}

/**
 * Fade the child away in coverage order as `progress` goes from 0 to 1.
 *
 * At 0 the child is fully visible, at 1 it is gone, whatever `feather` is. Only the child's
 * alpha changes; its colors stay put. Turning `invert` on reverses the order, so the last
 * pixels go first.
 *
 * @example
 * ```ts
 * effect: reveal({coverage: coverage.directional(p('angle')), progress: p('progress'), feather: p('feather'), invert: p('invert')})
 * ```
 * @tip Animate `progress` from outside (a prop driven by scroll or a tween); the wipe itself has no clock.
 * @see coverage, Coverage
 */
export function reveal(slots: RevealSlots): PointwiseEffect {
    // Pixels vanish in coverage order as `progress` grows, crossing a ±`feather` soft front
    // remapped so both progress extremes clear completely. Scales the straight-alpha child's
    // alpha with RGB preserved: pointwise, no render-to-texture.
    return {
        kind: 'pointwise',
        body: {fn: kit.applyReveal, hint: 'applyReveal'},
        args: (params) => [
            call(kit.revealMask, 'revealMask', [
                slots.coverage.build(params),
                resolveArg(slots.progress, params),
                resolveArg(slots.feather, params),
                resolveArg(slots.invert, params),
            ]),
        ],
    }
}
