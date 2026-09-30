/**
 * std/warps — words for the `map:` field of a warp. A warp is a component that moves the
 * pixels of the layer inside it (the child): for every pixel on screen, the map says where in
 * the child to look. Pass props with `p('name')` or plain numbers.
 *
 * A definition with a `map:` is a warp: `defineShader({name, props, map: warps.twirl({center:
 * p('center'), intensity: p('intensity')})})`. Positions are in uv (0–1 across the canvas, y
 * down) and angles are in degrees unless a word says otherwise. Most maps can look outside the
 * child, so the definition also says what happens there with `edges`: `'prop'` (the default)
 * reads an `edges` prop holding `'stretch'`, `'transparent'`, `'mirror'` or `'wrap'`; a fixed
 * mode number (0 stretch, 1 transparent, 2 mirror, 3 wrap) bakes one choice in; `'none'` skips
 * edge handling for a map that never leaves the child. `resample` picks the texture filter:
 * `'catmullRom'` (the default, sharp under magnification) or `'bilinear'` (one tap, softer).
 * Animated warps read the layer's clock, so declare `animatedTime: {speed: 'speed'}` on the
 * definition.
 */
// Maintainer notes. Each producer takes a slot object (prop bindings via `p()`, or plain
// numbers) and returns a `UvMapSource`: the single coordinate map (screen UV → source UV,
// plus coverage where the warp clips) that the warp role turns into both engine paths — the
// RTT fragment and the analytic uvRemap fold — so they cannot drift. The GPU bodies live in
// the kit (`warpMaps`); these functions only wire slots into them.
import type {Expr, KitTexture} from '../gpu/contract'
import {call, expr, floatE, animatedTime} from '../gpu/porters'
import {warpMaps, edges} from '../gpu/kit/index'
import {lerpToIdentity, selectMap} from '../gpu/scaffolds/uvRemapShader'
import type {UvMapSource, UvRemapHookParams} from '../gpu/scaffolds/uvRemapShader'
import type {ArgSpec} from './invoke'
import {resolveScalar, uniformOf} from './invoke'
import {Scalar} from './values'
import type {GatherEffect} from './types'
import type {PropRef} from './values'

/**
 * Resolve a warp slot to an Expr: prop bindings, number literals, and Scalar value-graphs
 * (including time signals — which ride the node's animated clock here, since warp hooks
 * carry no `ctx`; declare `animatedTime: {speed: '…'}` on the definition).
 */
function arg(spec: ArgSpec, params: UvRemapHookParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar) return resolveScalar(spec, params)
    if (typeof spec === 'object' && spec !== null && 'kind' in spec) {
        if (spec.kind === 'prop') return uniformOf(spec as PropRef, params)
        return resolveScalar(spec as never, params)
    }
    throw new Error('std/warps: warp slots accept prop bindings, numbers, and Scalar signals')
}

/**
 * Mirror the child across a line through `center` at `angle`.
 *
 * `center` is in uv, `angle` in degrees. One side of the line stays as it is and the other side
 * shows its reflection.
 *
 * @example
 * ```ts
 * map: warps.mirrorLine({center: p('center'), angle: p('angle')})
 * ```
 * @tip Pair it with an `edges` prop that defaults to `'mirror'` so the reflected half never runs out of picture.
 * @see flip, kaleidoscope
 */
export function mirrorLine(slots: {center: ArgSpec; angle: ArgSpec}): UvMapSource {
    // The reflection body returns a hard 0/1 side selector alongside the reflected coordinate,
    // so `selectMap` picks per pixel between the original coordinate (near side) and the
    // reflected one (far side). Selecting COORDINATES and sampling once is pixel-identical to
    // mixing two sampled colors by the same hard selector, with one texture fetch instead of
    // two, and both halves go through the same reconstruction filter so the filtering never
    // changes across the mirror line.
    return (params) => {
        // `Expr` has no CSE — `member()` re-emits its whole subtree at each use site — so calling
        // this twice emits `mirrorReflect(...)` twice in the WGSL. Same text, no extra cost.
        const reflect = (uv: Expr, aspect: Expr): Expr =>
            call(warpMaps.mirrorReflect, 'mirrorReflect', [arg(slots.center, params), arg(slots.angle, params), uv, aspect])
        return selectMap(
            (uv) => uv,
            (uv, aspect) => reflect(uv, aspect).member('xy'),
            (uv, aspect) => reflect(uv, aspect).member('z'),
        )
    }
}

/**
 * Flip the child horizontally, vertically, or both.
 *
 * `flipX` and `flipY` are boolean props declared with `transform: transformBoolean`. The flip
 * never looks outside the child, so set `edges: 'none'` on the definition.
 *
 * @example
 * ```ts
 * map: warps.flip({flipX: p('flipX'), flipY: p('flipY')}),
 * edges: 'none',
 * ```
 * @see mirrorLine
 */
export function flip(slots: {flipX: ArgSpec; flipY: ArgSpec}): UvMapSource {
    // `flipX`/`flipY` arrive as ±1 uniforms (transformBoolean maps false → −1, not 0), so the
    // body compares them rather than using them as mix factors. The map sends [0,1] onto [0,1],
    // which is why `edges: 'none'` is legitimate here and nowhere else in the library.
    return (params) => ({
        map: (uv) => call(warpMaps.flipUV, 'flipUV', [uv, arg(slots.flipX, params), arg(slots.flipY, params)]),
    })
}

/**
 * Curve the child like a bent sheet seen in perspective, so its ends swell or shrink.
 *
 * `strength` runs −1 to 1 and its sign picks the bend direction. `angle` is the bend axis in
 * degrees. `falloff` (0–1) keeps more of the middle flat and pushes the bend toward the ends.
 * Where the sheet leaves the frame there is nothing to show, so pair it with an `edges` prop
 * that defaults to `'transparent'`.
 *
 * @example
 * ```ts
 * map: warps.bend({strength: p('strength'), falloff: p('falloff'), angle: p('angle')})
 * ```
 * @see perspective, cornerPin
 */
export function bend(slots: {strength: ArgSpec; falloff: ArgSpec; angle: ArgSpec}): UvMapSource {
    // The inverse bent-sheet projection under real perspective; the closed-form quadratic and its
    // silhouette guard are documented on `warpMaps.bendRemap`. Past the visible silhouette the
    // coordinate is pushed far out of [0,1] so the edge mode decides.
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.bendRemap, 'bendRemap', [
            arg(slots.strength, params), arg(slots.falloff, params), arg(slots.angle, params), uv, aspect,
        ]),
    })
}

/**
 * Magnify or pinch the child inside a disc around `center`.
 *
 * `strength` runs −1 to 1: positive bulges out, negative pinches in. `radius` sets the size of
 * the disc (1 reaches about half the canvas height). `falloff` runs 0 (hard edge) to 1 (soft).
 *
 * @example
 * ```ts
 * map: warps.bulge({center: p('center'), strength: p('strength'), radius: p('radius'), falloff: p('falloff')})
 * ```
 * @see twirl, stretch
 */
export function bulge(slots: {center: ArgSpec; strength: ArgSpec; radius: ArgSpec; falloff: ArgSpec}): UvMapSource {
    // The effective disc is `radius * 0.5` in aspect-corrected uv; the body negates `strength` so
    // a positive value scales the delta down (magnification). See `warpMaps.bulgeUV`.
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.bulgeUV, 'bulgeUV', [
            arg(slots.center, params), arg(slots.strength, params), arg(slots.radius, params), arg(slots.falloff, params), uv, aspect,
        ]),
    })
}

/**
 * Twist the child around `center`, turning more the farther a pixel is from it.
 *
 * `intensity` is the twist in radians per unit of distance, so the canvas edge turns by about
 * `intensity` radians. Negative values twist the other way. The library uses −5 to 5.
 *
 * @example
 * ```ts
 * map: warps.twirl({center: p('center'), intensity: p('intensity')})
 * ```
 * @see bulge, concentricRings
 */
export function twirl(slots: {center: ArgSpec; intensity: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.twirlUV, 'twirlUV', [
            arg(slots.center, params), arg(slots.intensity, params), uv, aspect,
        ]),
    })
}

/**
 * Fold the child into mirrored pie slices around `center`.
 *
 * `segments` is the number of slices (2 or more). `angle` rotates the whole pattern, in
 * degrees. Pair it with an `edges` prop that defaults to `'mirror'`.
 *
 * @example
 * ```ts
 * map: warps.kaleidoscope({center: p('center'), segments: p('segments'), angle: p('angle')})
 * ```
 * @see mirrorLine, toPolar
 */
export function kaleidoscope(slots: {center: ArgSpec; segments: ArgSpec; angle: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.kaleidoscopeUV, 'kaleidoscopeUV', [
            arg(slots.center, params), arg(slots.segments, params), arg(slots.angle, params), uv, aspect,
        ]),
    })
}

/**
 * Stretch the child along one direction, starting at `center`.
 *
 * `angle` is the direction in degrees. Content on the side `angle` points to is pulled out;
 * the other side is left alone. `strength` (0–1) sets how far. `falloff` (0–1) softens the
 * start of the stretch, from a hard line at 0 to a long ramp at 1.
 *
 * @example
 * ```ts
 * map: warps.stretch({center: p('center'), strength: p('strength'), angle: p('angle'), falloff: p('falloff')})
 * ```
 * @see bulge, bend
 */
export function stretch(slots: {center: ArgSpec; strength: ArgSpec; angle: ArgSpec; falloff: ArgSpec}): UvMapSource {
    // The parallel component of the delta is divided by `1 + strength·100·mask`, where the mask
    // ramps in over `mix(0.001, 75, falloff)` units past the center. See `warpMaps.stretchUV`.
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.stretchUV, 'stretchUV', [
            arg(slots.center, params), arg(slots.strength, params), arg(slots.angle, params), arg(slots.falloff, params), uv, aspect,
        ]),
    })
}

/** Waveform names → mode numbers, matching the `waveType` prop transform. */
const WAVE_MODES: Record<string, number> = {sine: 0, triangle: 1, square: 2, sawtooth: 3, bounce: 4}

/** Normalise a `waveType` prop value (mapped number, or a raw string for robustness). */
function waveTypeMode(raw: unknown): number {
    if (typeof raw === 'number') return raw
    return WAVE_MODES[raw as string] ?? 0
}

/** Pick the active waveform body from the compile-time waveType (0=sine … 4=bounce). */
function waveBodyFor(waveType: number): {fn: unknown; hint: string} {
    switch (waveType) {
        case 1: return {fn: warpMaps.waveDistortTriangle, hint: 'waveDistortTriangle'}
        case 2: return {fn: warpMaps.waveDistortSquare, hint: 'waveDistortSquare'}
        case 3: return {fn: warpMaps.waveDistortSawtooth, hint: 'waveDistortSawtooth'}
        case 4: return {fn: warpMaps.waveDistortBounce, hint: 'waveDistortBounce'}
        default: return {fn: warpMaps.waveDistortSine, hint: 'waveDistortSine'}
    }
}

/**
 * Ripple the child with a travelling wave.
 *
 * `angle` is the travel direction in degrees, `frequency` the number of waves across the
 * canvas, `strength` (0–1) how far pixels move. `waveType` is a prop marked
 * `compileTime: true` holding `'sine'`, `'triangle'`, `'square'`, `'sawtooth'` or `'bounce'`.
 * The wave moves with the layer's clock, so declare `animatedTime: {speed: 'speed'}`.
 *
 * @example
 * ```ts
 * map: warps.wave({strength: p('strength'), frequency: p('frequency'), angle: p('angle'), waveType: p('waveType')})
 * ```
 * @tip Changing `waveType` recompiles the shader; the other inputs update live.
 * @see flowNoise, barOffset
 */
export function wave(slots: {strength: ArgSpec; frequency: ArgSpec; angle: ArgSpec; waveType: PropRef}): UvMapSource {
    // `waveType` is read from compile-time propValues: the producer branches once per composition
    // and only the active waveform body is emitted — no runtime branch. The phase advances by the
    // definition's animated clock at half rate.
    return (params) => {
        const {fn, hint} = waveBodyFor(waveTypeMode(params.propValues[slots.waveType.name]))
        const t = animatedTime(params).mul(0.5)
        return {
            map: (uv, aspect) => call(fn, hint, [
                uv, aspect, arg(slots.angle, params), arg(slots.frequency, params), arg(slots.strength, params), t,
            ]),
        }
    }
}

/**
 * Unroll the child around `center`, so angle becomes x and distance becomes y.
 *
 * `wrap` scales how many times the child repeats around (1 is once). `radius` scales the
 * distance axis. `intensity` runs 0 (no change) to 1 (the full polar unwrap), and the values
 * between are a real partial warp, not a cross-fade.
 *
 * @example
 * ```ts
 * map: warps.toPolar({center: p('center'), wrap: p('wrap'), radius: p('radius'), intensity: p('intensity')})
 * ```
 * @see fromPolar, kaleidoscope
 */
export function toPolar(slots: {center: ArgSpec; wrap: ArgSpec; radius: ArgSpec; intensity: ArgSpec}): UvMapSource {
    // `lerpToIdentity` blends COORDINATES, not colors. The angle from `atan2` is normalised to
    // [0, 1] and scaled by `wrap`; the radius is scaled by `radius`.
    return (params) => lerpToIdentity(
        {
            map: (uv, aspect) => call(warpMaps.polarCoordsUV, 'polarCoordsUV', [
                arg(slots.center, params), arg(slots.wrap, params), arg(slots.radius, params), uv, aspect,
            ]),
        },
        () => arg(slots.intensity, params),
    )
}

/**
 * Wrap the child into a circle around `center`, reading its x as angle and its y as distance.
 *
 * The reverse of `toPolar`. `scale` sets how large the circle is. `intensity` runs 0 (no
 * change) to 1 (the full wrap).
 *
 * @example
 * ```ts
 * map: warps.fromPolar({center: p('center'), scale: p('scale'), intensity: p('intensity')})
 * ```
 * @see toPolar
 */
export function fromPolar(slots: {center: ArgSpec; scale: ArgSpec; intensity: ArgSpec}): UvMapSource {
    // Unlike the other maps, `center.x` is not aspect-corrected in `rectCoordsUV` (the rectangular
    // X is aspect-divided and added to the raw center.x).
    return (params) => lerpToIdentity(
        {
            map: (uv, aspect) => call(warpMaps.rectCoordsUV, 'rectCoordsUV', [
                arg(slots.center, params), arg(slots.scale, params), uv, aspect,
            ]),
        },
        () => arg(slots.intensity, params),
    )
}

/**
 * Pin the four corners of the child to new positions, with true perspective between them.
 *
 * The corners are positions in uv. `amount` (0–1) blends each corner from its normal place
 * (0) to the pinned position (1). Pixels outside the pinned quad are clipped to transparent
 * whatever the edge mode.
 *
 * @example
 * ```ts
 * map: warps.cornerPin({topLeft: p('topLeft'), topRight: p('topRight'), bottomRight: p('bottomRight'), bottomLeft: p('bottomLeft'), amount: p('amount')})
 * ```
 * @see perspective, bend
 */
export function cornerPin(slots: {
    topLeft: ArgSpec
    topRight: ArgSpec
    bottomRight: ArgSpec
    bottomLeft: ArgSpec
    amount: ArgSpec
}): UvMapSource {
    // A square→quad homography (Heckbert), inverted analytically. The body returns a packed
    // `vec3f` (projected coordinate in `.xy`, front-facing gate in `.z`), so the map returns the
    // COVERAGE form: the warp role applies coverage to sampled alpha on the fragment path and to
    // the coverage mask on the analytic path, clipping the folded-back region either way.
    return (params) => ({
        map: (uv) => {
            const packed = call(warpMaps.cornerPinSample, 'cornerPinSample', [
                uv, arg(slots.amount, params),
                arg(slots.topLeft, params), arg(slots.topRight, params), arg(slots.bottomRight, params), arg(slots.bottomLeft, params),
            ])
            return {uv: packed.member('xy'), coverage: packed.member('z')}
        },
    })
}

/**
 * Tilt the child in 3D, as if it were a card turned away from the camera.
 *
 * `pan` and `tilt` are the turns in degrees (−90 to 90). `fov` is the camera's field of view
 * in degrees: wider means stronger perspective. `zoom` scales the picture (1 is unchanged).
 * `center` is the pivot and `offset` shifts the result, both in uv.
 *
 * @example
 * ```ts
 * map: warps.perspective({center: p('center'), pan: p('pan'), tilt: p('tilt'), fov: p('fov'), zoom: p('zoom'), offset: p('offset')})
 * ```
 * @see cornerPin, bend
 */
export function perspective(slots: {
    center: ArgSpec
    pan: ArgSpec
    tilt: ArgSpec
    fov: ArgSpec
    zoom: ArgSpec
    offset: ArgSpec
}): UvMapSource {
    // The map ignores `aspect` — it samples in raw UV space, unlike the other warps here.
    return (params) => ({
        map: (uv) => call(warpMaps.perspectiveUV, 'perspectiveUV', [
            arg(slots.center, params), arg(slots.pan, params), arg(slots.tilt, params),
            arg(slots.fov, params), arg(slots.zoom, params), arg(slots.offset, params), uv,
        ]),
    })
}

/**
 * Cut the child into rings around `center` and spin each ring by its own amount.
 *
 * `rings` is how many rings fit in one unit of distance. `intensity` is the largest still
 * rotation a ring can have, in degrees. `smoothness` (0–1) blends neighbouring rings instead
 * of cutting between them. `seed` reshuffles the per-ring amounts and `speedRandomness`
 * (0–1) lets rings spin at different rates. Declare `animatedTime: {speed: 'speed'}`.
 *
 * @example
 * ```ts
 * map: warps.concentricRings({center: p('center'), intensity: p('intensity'), rings: p('rings'), smoothness: p('smoothness'), seed: p('seed'), speedRandomness: p('speedRandomness')})
 * ```
 * @see twirl, barOffset
 */
export function concentricRings(slots: {
    center: ArgSpec
    intensity: ArgSpec
    rings: ArgSpec
    smoothness: ArgSpec
    seed: ArgSpec
    speedRandomness: ArgSpec
}): UvMapSource {
    // Per-ring hashed static + animated rotation, blended between adjacent rings along the
    // shortest angular path. See `warpMaps.concentricSpinUV`.
    return (params) => {
        const t = animatedTime(params)
        return {
            map: (uv, aspect) => call(warpMaps.concentricSpinUV, 'concentricSpinUV', [
                arg(slots.center, params), arg(slots.intensity, params), arg(slots.rings, params),
                arg(slots.smoothness, params), arg(slots.seed, params), arg(slots.speedRandomness, params),
                t, uv, aspect,
            ]),
        }
    }
}

/**
 * Push the child around with a slowly flowing liquid motion.
 *
 * `strength` is how far pixels move, in uv (the library uses 0–0.5). `detail` scales the flow
 * pattern (higher is finer). `seed` is a prop that picks a different pattern, even when still.
 * Two clocks drive it: declare `animatedTime: {speed: 'speed'}` for the drift and
 * `extraAnimatedTimes: {evolution: 'evolutionSpeed'}` for the pattern slowly reshaping.
 *
 * @example
 * ```ts
 * map: warps.flowNoise({strength: p('strength'), detail: p('detail'), seed: p('seed')})
 * ```
 * @tip Speed 0 on both clocks gives a fixed distortion you can still change with `seed`.
 * @see wave, liquidDisplace
 */
export function flowNoise(slots: {strength: ArgSpec; detail: ArgSpec; seed: PropRef}): UvMapSource {
    // Three noise layers drifting through a 3D field, combined and normalised so the flow speed
    // is constant and `strength` is the only magnitude control. Both clocks run at 1/10 rate.
    // `seed` is added to both clocks on the GPU, shifting the noise domain along the
    // drift/evolution axes — a different static pattern per seed value, even at speed 0.
    return (params) => {
        const time = animatedTime(params, slots.seed.name).mul(0.1)
        const evolutionTime = animatedTime(params, slots.seed.name, '_animTime_evolution').mul(0.1)
        return {
            map: (uv, aspect) => call(warpMaps.flowFieldUV, 'flowFieldUV', [
                uv, aspect, arg(slots.detail, params), arg(slots.strength, params), time, evolutionTime,
            ]),
        }
    }
}

/**
 * Slice the child into strips that slide out of frame in alternating directions.
 *
 * A wipe transition. `progress` runs 0 (whole) to 1 (gone). `angle` is the cut direction in
 * degrees and `count` the number of strips. Set `edges: 1` (transparent) on the definition so
 * the vacated space clears instead of smearing.
 *
 * @example
 * ```ts
 * map: warps.slicedSlide({angle: p('angle'), count: p('sliceCount'), progress: p('progress')}),
 * edges: 1,
 * ```
 * @see barOffset
 */
export function slicedSlide(slots: {angle: ArgSpec; count: ArgSpec; progress: ArgSpec}): UvMapSource {
    // The "shredder" transition. At progress 1 every lookup is out of bounds, so the fixed
    // TRANSPARENT edge mode is what makes it a wipe. SliceWipe also sets `resample: 'bilinear'`
    // to keep the single tap it has always used.
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.sliceWipeUV, 'sliceWipeUV', [
            uv, aspect, arg(slots.angle, params), arg(slots.count, params), arg(slots.progress, params),
        ]),
    })
}

/**
 * Cut the child into parallel bars and shift each one sideways by a random amount.
 *
 * A glitch or fractured look. `count` is the number of bars across the canvas, `angle` their
 * direction in degrees, `intensity` (0–1) how far they shift. `seed` reshuffles the offsets.
 * Bars drift with the layer's clock, so declare `animatedTime: {speed: 'speed'}`.
 *
 * @example
 * ```ts
 * map: warps.barOffset({count: p('count'), angle: p('angle'), intensity: p('intensity'), seed: p('seed')})
 * ```
 * @see slicedSlide, concentricRings
 */
export function barOffset(slots: {count: ArgSpec; angle: ArgSpec; intensity: ArgSpec; seed: ArgSpec}): UvMapSource {
    // Each bar is offset by a hash for a fractured or glitch-like look and drifts at its own
    // hashed rate. `count` refers to bars across the longest viewport dimension.
    return (params) => {
        const t = animatedTime(params)
        return {
            map: (uv, aspect) => call(warpMaps.barShiftUV, 'barShiftUV', [
                arg(slots.count, params), arg(slots.angle, params), arg(slots.intensity, params),
                arg(slots.seed, params), t, uv, aspect,
            ]),
        }
    }
}

/**
 * Move the child by a displacement field your own simulation produces, one grid cell at a time.
 *
 * `output` names the simulation output (the same name you gave the field), `gridSize` the
 * number of cells across the shorter side of the canvas. Every pixel in a cell moves together,
 * by at most a tenth of the canvas. Add `uvRemapIdentityWhen: ({computeOutputs}) =>
 * !computeOutputs?.<output>` to the definition so the warp does nothing until the field exists.
 *
 * @example
 * ```ts
 * map: warps.gridCellDisplace({gridSize: p('gridSize'), output: 'displacement'}),
 * uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement,
 * ```
 * @see liquidDisplace
 */
export function gridCellDisplace(slots: {gridSize: ArgSpec; output: string}): UvMapSource {
    // Snap the incoming UV to the centre of its `gridSize` cell, read the displacement texture
    // there, offset (±0.1 clamp), then edge-handle. When the compute texture is absent (GPU-free
    // composition, or before the compute hook has run) the map substitutes a zero displacement
    // and keeps its structure identical — `uvRemapIdentityWhen` makes the analytic path return
    // the incoming remap verbatim in that case instead.
    return (params) => ({
        map: (uv, aspect) => {
            const dispTex = params.computeOutputs?.[slots.output] as KitTexture | undefined
            const gridCellUV = call(warpMaps.gridCellSnap, 'gridCellSnap', [uv, arg(slots.gridSize, params), aspect])
            const disp = dispTex ? dispTex.sample(gridCellUV, 'linearClamp').member('xy') : expr('vec2f(0.0, 0.0)')
            return call(warpMaps.gridDistortOffsetUV, 'gridDistortOffsetUV', [uv, disp])
        },
    })
}

/**
 * Move the child smoothly by a displacement field your own simulation produces.
 *
 * The liquid or cloth look. `output` names the simulation output. `intensity` scales the
 * movement (the library uses 0–20), capped at about 15% of the canvas. Add
 * `uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.<output>` to the definition so
 * the warp does nothing until the field exists.
 *
 * @example
 * ```ts
 * map: warps.liquidDisplace({intensity: p('intensity'), output: 'displacement'}),
 * uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement,
 * ```
 * @see gridCellDisplace, flowNoise
 */
export function liquidDisplace(slots: {intensity: ArgSpec; output: string}): UvMapSource {
    // Sampled continuously at the incoming UV, scaled by `intensity * 0.1` (±0.15 clamp). Same
    // absent-texture handling as `gridCellDisplace`: zero displacement here, with
    // `uvRemapIdentityWhen` covering the analytic path on the definition.
    return (params) => ({
        map: (uv) => {
            const dispTex = params.computeOutputs?.[slots.output] as KitTexture | undefined
            const disp = dispTex ? dispTex.sample(uv, 'linearClamp').member('xy') : expr('vec2f(0.0, 0.0)')
            return call(warpMaps.liquifyOffsetUV, 'liquifyOffsetUV', [uv, disp, arg(slots.intensity, params)])
        },
    })
}

/** Channel-mode names → mode numbers, matching the `channelMode` prop transform. */
const DISPLACE_CHANNEL_MODES: Record<string, number> = {twoAxis: 0, directional: 1}

/** Normalise a `channelMode` prop value (mapped number, or a raw string for robustness). */
function displaceChannelMode(raw: unknown): number {
    if (typeof raw === 'number') return raw
    return DISPLACE_CHANNEL_MODES[raw as string] ?? 0
}

/**
 * Distort the child using another layer's pixels as a displacement map.
 *
 * This one goes in `effect:`, not `map:`, because it reads a second layer. `source` is a prop
 * marked `compileTime: true` that holds the other layer's id. `amount` (0–1) sets how far
 * pixels move. `channels` is a prop holding `'twoAxis'` (red moves x, green moves y) or
 * `'directional'` (brightness pushes along `angle`, in degrees). Mid-grey means no movement.
 * `edges` is a prop marked `compileTime: true` holding `'stretch'`, `'transparent'`,
 * `'mirror'` or `'wrap'`. With no source selected the child passes through unchanged.
 *
 * @example
 * ```ts
 * effect: warps.displaceByLayer({source: p('source'), amount: p('amount'), channels: p('channelMode'), angle: p('angle'), edges: p('edges')})
 * ```
 * @tip An animated source layer displaces live.
 * @see liquidDisplace
 */
export function displaceByLayer(slots: {
    source: PropRef
    amount: ArgSpec
    channels: PropRef
    angle: ArgSpec
    edges: PropRef
}): GatherEffect {
    // The classic Photoshop/AE displacement, driven cross-layer rather than by a procedural
    // field, hence a gather effect: the source arrives via `getLayerTexture`, the same shared RTT
    // boundary prop maps use, so the layer renders once even when it's also visible on canvas.
    // `channels` picks the applier at composition (red/green two-axis vs luminance along
    // `angle`). No source selected (or an unresolvable id) → a sharp passthrough of the child,
    // exactly as sampled at the incoming coordinate.
    return {
        kind: 'gather',
        build: (params) => {
            const source = params.getLayerTexture(slots.source.name)
            if (!source) return params.texture.sample(params.ctx.uv)
            const edgeMode = (params.propValues[slots.edges.name] as number) ?? 0
            const src = source.sample(params.ctx.uv)
            const distorted = displaceChannelMode(params.propValues[slots.channels.name]) === 1
                ? call(warpMaps.dmDisplaceLuminance, 'dmDisplaceLuminance', [params.ctx.uv, src, params.ctx.aspect, arg(slots.amount, params), arg(slots.angle, params)])
                : call(warpMaps.dmDisplaceRG, 'dmDisplaceRG', [params.ctx.uv, src, params.ctx.aspect, arg(slots.amount, params)])
            return edges.sampleRemappedExpr(params.texture, distorted, edgeMode)
        },
    }
}
