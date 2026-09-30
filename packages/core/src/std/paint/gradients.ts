/**
 * std/paint/gradients — complete gradient paints: a flat fill, a color wheel, a point cloud, a beam.
 *
 * Each word binds your props and returns something a generator's `paint:` field takes. They
 * are finished looks, not building blocks. For a gradient of your own shape, compose a
 * `dist.*` field with `rampOver` from the fields words instead.
 */
// Maintainer notes (not part of the reference):
//  - Built from the kit's gradient bodies (`kit/gradientPaints.ts`). Metric gradients
//    (linear/radial/conic/diamond/spiral) and noise-field paints are COMPOSITIONS — see
//    `std/paint/fields.ts` for the field pipeline; light recipes (bursts, godrays, leaks)
//    compose the parts in `std/paint/light.ts`.
//  - The UV idiom — standalone renders read `ctx.uv`; wrapped by a UV-propagating parent the
//    composed `uvContext` wins. Aspect uses the effective viewport (resize-fit box) when
//    present, else the canvas viewport.
//  - Structural slots (`space`, `mode`) are compile-time props: the word reads their CPU values
//    and JS-branches, emitting only the selected WGSL path.
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE, vec4} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {d, gradientPaints, colorMixing} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {local} from '../math'
import {uniformOf, resolveScalar, type ArgSpec} from '../invoke'

/** What a generator's `paint:` field takes: a function that returns the color at this pixel. */
export type Paint = (params: GpuFragmentParams) => Expr

// ── Shared resolution helpers ───────────────────────────────────────────────────────────

/** Resolve a scalar-ish slot (prop, scalar graph, or number literal) to an Expr. */
function arg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

/** The composed UV + effective viewport every gradient paint evaluates against. */
function paintFrame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

/** Read a structural (compile-time) slot's CPU value. */
function structural(ref: PropRef, params: GpuFragmentParams): unknown {
    return params.propValues[ref.name]
}

// ── Paint words ─────────────────────────────────────────────────────────────────────────

/**
 * A flat fill in one color prop.
 *
 * The color's alpha passes through, so a half-transparent color gives a half-transparent layer.
 *
 * @example
 * ```ts
 * paint: solidColor(p('color'))
 * ```
 */
export function solidColor(color: PropRef): Paint {
    // The bound prop is already a P3-linear rgba uniform, so the paint is a direct read.
    return (params) => uniformOf(color, params)
}

/**
 * A striped gradient that slides across the canvas, cycling through its colors over time.
 *
 * `mode` is a `compileTime` prop: `'rainbow'` generates a full spectrum, `'custom'` loops
 * `palette.a` to `b` to `c` and back, mixed in the color space named by the `space` prop
 * (also `compileTime`, ignored in rainbow mode). `direction` in degrees turns the stripes
 * about the canvas center, `scale` sets how many color cycles cross the canvas (1 is a
 * single wide sweep). It moves on the layer's clock, so declare `animatedTime: {speed: 'speed'}`.
 *
 * @example
 * ```ts
 * paint: colorWheel({mode: p('mode'), direction: p('angle'), scale: p('scale'), palette: {a: p('colorA'), b: p('colorB'), c: p('colorC')}, space: p('colorSpace')})
 * ```
 * @see pointCloudGradient
 */
export function colorWheel(slots: {
    mode: PropRef
    direction: ArgSpec
    scale: ArgSpec
    palette: {a: PropRef; b: PropRef; c: PropRef}
    space: PropRef
}): Paint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const animTime = animatedTime(params)
        const t = call(gradientPaints.colorWheelT, 'colorWheelT', [
            uv, viewport, arg(slots.direction, params), arg(slots.scale, params), animTime,
        ])

        // `mode` is a structural string prop → JS-branch the whole paint (recomposes on change).
        const mode = (structural(slots.mode, params) as string) ?? 'rainbow'
        if (mode === 'rainbow') {
            return call(gradientPaints.colorWheelRainbow, 'colorWheelRainbow', [t])
        }
        // Custom 3-color cycle: phase → (segmentMix, segmentIndex), the three per-segment
        // blends mixed at builder level in the structural color space, then the segment pick.
        const phase = local(call(gradientPaints.colorWheelPhase, 'colorWheelPhase', [t]), 'wheelPhase')
        const si = phase.member('x')
        const floorT3 = phase.member('y')
        const spaceMode = (structural(slots.space, params) as number) ?? 0
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        const c01 = call(variant, 'mixColors', [uniformOf(slots.palette.a, params), uniformOf(slots.palette.b, params), si])
        const c12 = call(variant, 'mixColors', [uniformOf(slots.palette.b, params), uniformOf(slots.palette.c, params), si])
        const c20 = call(variant, 'mixColors', [uniformOf(slots.palette.c, params), uniformOf(slots.palette.a, params), si])
        return call(gradientPaints.colorWheelSelect, 'colorWheelSelect', [c01, c12, c20, floorT3])
    }
}

/**
 * Five colored points placed anywhere on the canvas, each pixel taking the colors of the points nearest it.
 *
 * Pass exactly five `{color, position}` prop pairs (positions made with `transformPosition`).
 * `smoothness` (0–5) widens each point's reach as it rises. The mix happens in the color
 * space named by the `space` prop (mark it `compileTime`).
 *
 * @example
 * ```ts
 * paint: pointCloudGradient({points: [{color: p('colorA'), position: p('positionA')}, {color: p('colorB'), position: p('positionB')}, {color: p('colorC'), position: p('positionC')}, {color: p('colorD'), position: p('positionD')}, {color: p('colorE'), position: p('positionE')}], smoothness: p('smoothness'), space: p('colorSpace')})
 * ```
 * @tip For a softer, animated version with any number of points, see `scatterField` in the fields words.
 * @see colorWheel
 */
export function pointCloudGradient(slots: {
    points: {color: PropRef; position: PropRef}[]
    smoothness: ArgSpec
    space: PropRef
}): Paint {
    // The kit body returns the four incremental running-weighted-average factors; the paint folds
    // the five colors through them in the structural space. `smoothness` inverts into the
    // inverse-distance power.
    if (slots.points.length !== 5) throw new Error('pointCloudGradient expects exactly 5 points')
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const factors = local(call(gradientPaints.mpgFactors, 'mpgFactors', [
            uv, viewport,
            ...slots.points.map((pt) => uniformOf(pt.position, params)),
            arg(slots.smoothness, params),
        ]), 'mpgFactors')
        const spaceMode = (structural(slots.space, params) as number) ?? 0
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        let acc: Expr = uniformOf(slots.points[0].color, params)
        const channels = ['x', 'y', 'z', 'w'] as const
        for (let i = 1; i < 5; i++) {
            acc = call(variant, 'mixColors', [acc, uniformOf(slots.points[i].color, params), factors.member(channels[i - 1])])
        }
        return acc
    }
}

/**
 * The `extraFields` a `beam` definition must declare.
 *
 * Spread it as `extraFields: beamPreconvertedFields` next to the `beam` paint.
 *
 * @example
 * ```ts
 * extraFields: beamPreconvertedFields,
 * ```
 * @see beam
 */
export const beamPreconvertedFields = {
    // For a non-linear color space the forward P3→working-space conversion of the two endpoint
    // colors is pixel-invariant, so `beam` computes it ONCE per frame on the CPU (dirty-keyed)
    // into these vec3 fields and the GPU reads them via `mixPreconvertedVariants[mode]`.
    /** @internal */
    convA: {schema: d.vec3f, initial: [0, 0, 0]},
    /** @internal */
    convB: {schema: d.vec3f, initial: [0, 0, 0]},
}

/**
 * A glowing beam of light between two points.
 *
 * `from` and `to` are position props (made with `transformPosition`). `thickness` (0–2) and
 * `softness` each take a `start` and `end` value so the beam can taper. The cross-section
 * shades from `colors.inside` at the core to `colors.outside` at the edge, mixed in the color
 * space named by the `space` prop (mark it `compileTime`). Declare
 * `extraFields: beamPreconvertedFields` on the definition.
 *
 * @example
 * ```ts
 * paint: beam({from: p('startPosition'), to: p('endPosition'), thickness: {start: p('startThickness'), end: p('endThickness')}, softness: {start: p('startSoftness'), end: p('endSoftness')}, colors: {inside: p('insideColor'), outside: p('outsideColor')}, space: p('colorSpace')})
 * ```
 * @see beamPreconvertedFields
 */
export function beam(slots: {
    from: PropRef
    to: PropRef
    thickness: {start: ArgSpec; end: ArgSpec}
    softness: {start: ArgSpec; end: ArgSpec}
    colors: {inside: PropRef; outside: PropRef}
    space: PropRef
}): Paint {
    // The pixel projects onto the segment; thickness and softness taper between the start/end
    // values along it. Linear mixes per pixel; a non-linear space preconverts both endpoints on
    // the CPU each frame into `beamPreconvertedFields` and pays only the weighted mix +
    // back-conversion per pixel. Alpha is the endpoint average scaled by the glow.
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const mode = (structural(slots.space, params) as number) ?? 0

        const field = local(call(gradientPaints.beamField, 'beamField', [
            uv, viewport, uniformOf(slots.from, params), uniformOf(slots.to, params),
            arg(slots.thickness.start, params), arg(slots.thickness.end, params),
            arg(slots.softness.start, params), arg(slots.softness.end, params),
        ]), 'beamField')
        const colorT = field.member('x')
        const alpha = field.member('y')
        const inside = uniformOf(slots.colors.inside, params)
        const outside = uniformOf(slots.colors.outside, params)

        let beamColorRGB: Expr
        if (mode !== 0) {
            let lastKey = ''
            params.onBeforeRender(() => {
                const a = params.getCpuValue(slots.colors.inside.name) as {x: number; y: number; z: number} | undefined
                const b = params.getCpuValue(slots.colors.outside.name) as {x: number; y: number; z: number} | undefined
                if (!a || !b) return
                const key = `${a.x},${a.y},${a.z}|${b.x},${b.y},${b.z}`
                if (key === lastKey) return
                lastKey = key
                const ca = colorMixing.convertP3ToMixSpaceCPU(a.x, a.y, a.z, mode)
                const cb = colorMixing.convertP3ToMixSpaceCPU(b.x, b.y, b.z, mode)
                params.setExtraField('convA', [ca[0], ca[1], ca[2]])
                params.setExtraField('convB', [cb[0], cb[1], cb[2]])
            })
            const variant = colorMixing.mixPreconvertedVariants[mode as keyof typeof colorMixing.mixPreconvertedVariants] ?? colorMixing.mixPreconvertedLinear
            const mixed = call(variant, 'mixPreconvertedColors', [
                params.uniforms.convA, params.uniforms.convB, inside.member('a'), outside.member('a'), colorT,
            ])
            beamColorRGB = mixed.member('rgb')
        } else {
            const mixed = call(colorMixing.mixColorsLinear, 'mixColors', [inside, outside, colorT])
            beamColorRGB = mixed.member('rgb')
        }

        // avgAlpha = (inside.a + outside.a) / 2, then scaled by the glow alpha.
        const avgAlpha = inside.member('a').add(outside.member('a')).mul(0.5)
        return vec4(beamColorRGB, avgAlpha.mul(alpha))
    }
}
