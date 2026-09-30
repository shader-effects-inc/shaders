/**
 * Hand-made looks for the layer inside: a watercolor wash, a chalk drawing, ASCII characters
 * and copper-plate line engraving. Each look is a short recipe. Build the shared piece once
 * (a wobbled sample position, a character grid, a ring of edge taps), hand it to the stages
 * that read it, and close with the word that returns the effect for `effect:` (`wash`,
 * `chalkSketch`, `linework`) or the fragment for `gpu.fragment` (`glyphTint`).
 *
 * Every stage takes an object of props (`p('name')`) or other stages. The closing words return
 * a gather effect: the layer inside renders to a texture first so the look can read the pixels
 * around each one. The result is straight alpha and, except for `linework`, keeps the alpha of
 * the layer inside.
 */
// Maintainer notes. Stages whose result is read by more than one consumer are memoised per
// composition (`shared`) so the readers share one lowered Expr — exactly a `const` local in a
// fused builder. The gather words own the species' alpha discipline (resultAlpha 'straight').
// The ASCII words are custom-tier stages over `GpuFragmentParams` because their glyph atlas is
// rasterised on the CPU — the host lifecycle stays with the definition and is handed in through
// the `atlas` slot. `orientedBoxField` (Text's placement kernel) and the engraving plates live
// here too; they are engine-internal and hidden from the reference.
import type {Expr, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {call, vec4, floatE, expr, ZERO, asLocal} from '../../gpu/porters'
import {stylizePaints, blend, tgpu, d, std, constants, noise} from '../../gpu/kit/index'
import type {RttFilterParams} from '../../gpu/scaffolds/rttFilter'
import type {GatherEffect} from '../types'
import {gather} from '../filter'
import type {PropRef} from '../values'
import {uniformOf} from '../invoke'

/** A stage that reads the layer inside (rendered to a texture) and gives one value per pixel. */
export type GatherStage = (params: RttFilterParams) => Expr

/**
 * Memoise a stage per composition so every reader shares ONE lowered value — the slot-graph
 * equivalent of a `const` local in a fused builder.
 */
function shared<P extends object, V>(build: (params: P) => V): (params: P) => V {
    const cache = new WeakMap<P, V>()
    return (params) => {
        const hit = cache.get(params)
        if (hit !== undefined) return hit
        const v = build(params)
        cache.set(params, v)
        return v
    }
}

// ═══ Watercolor parts ═══════════════════════════════════════════════════════════════════════════

/**
 * A sample position that drifts slowly past hard edges, so paint bleeds instead of stopping.
 *
 * `amount` scales the drift in logical pixels (about four pixels per unit; 0 turns it off).
 * Build it once and pass the same stage as `at` to both `kuwahara` and `wash`, so the brush and
 * the original sample agree.
 *
 * @example
 * ```ts
 * const wet = bleedWarp({amount: p('bleed')})
 * ```
 * @tip Give the `amount` prop `recompile: crosses(0)` so turning the bleed off removes the work.
 * @see kuwahara, wash
 */
// Maintainer: at 0 the stage collapses to ctx.uv at composition time (a compile-time branch, hence
// the crosses(0) rule). The wobble is authored in LOGICAL pixels (logicalViewportSize) so it holds
// the same physical size at any DPR. Shared by the original sample and every brush tap.
export function bleedWarp(slots: {amount: PropRef}): GatherStage {
    return shared((params) => {
        const amount = (params.propValues[slots.amount.name] as number) ?? 0
        if (amount === 0) return params.ctx.uv
        return asLocal(call(stylizePaints.watercolorUvBase, 'watercolorUvBase', [
            params.ctx.uv, params.ctx.logicalViewportSize, uniformOf(slots.amount, params), params.ctx.time,
        ]), 'wcUvBase')
    })
}

/** The brush `kuwahara` builds: four overlapping patches of samples around each pixel. Pass it to `wash`. */
export type KuwaharaBrush = (params: RttFilterParams) => {quadrants: {sum: Expr; sq: Expr}[]; invN: number}

/**
 * A painter's brush: the flattening that turns fine detail into soft patches of one color.
 *
 * `radius` is the brush size in logical pixels, 1 to 8, and must be a compile-time prop. `at`
 * is the sample position, normally the same `bleedWarp` stage you hand to `wash`. Returns the
 * brush for `wash`.
 *
 * @example
 * ```ts
 * brush: kuwahara({at: wet, radius: p('radius')})
 * ```
 * @tip Mark `radius` `compileTime: true`. A hidden prop with a fixed default is the usual choice.
 * @see wash, bleedWarp
 */
// Maintainer: 4 overlapping Kuwahara quadrants of taps around the `at` frame, accumulating
// premultiplied sum + rgb sum² per quadrant at the builder level. The tap loop is unrolled in JS
// (hence compile-time radius), stride-2 taps at texel-pair midpoints so one bilinear fetch covers
// a 2×2 block; every tap is hoisted into a local so it samples exactly once. The footprint is
// authored in LOGICAL pixels so it stays the same physical size at any DPR.
export function kuwahara(slots: {at: GatherStage; radius: PropRef}): KuwaharaBrush {
    return shared((params) => {
        // compileTime brush radius (clamps round(radius) to [1,8], with a `?? 4` fallback).
        const R = Math.max(1, Math.min(8, Math.round((params.propValues[slots.radius.name] as number) ?? 4)))
        // At least 2 taps per axis — a 1-tap quadrant has zero variance by construction, so all
        // four sigmas tie and the Kuwahara pick degenerates into a fixed diagonal offset sample.
        const taps = Math.max(2, Math.ceil((R + 1) / 2))
        const invN = 1 / (taps * taps)
        const uvBase = slots.at(params)
        const lvp = params.ctx.logicalViewportSize

        const quadrant = (dx: number, dy: number): {sum: Expr; sq: Expr} => {
            let sum: Expr = expr('vec4f(0.0, 0.0, 0.0, 0.0)')
            let sq: Expr = expr('vec3f(0.0, 0.0, 0.0)')
            for (let j = 0; j < taps; j++) {
                for (let i = 0; i < taps; i++) {
                    const uv = call(stylizePaints.watercolorTapUV, 'watercolorTapUV',
                        [uvBase, lvp, floatE((2 * i + 0.5) * dx), floatE((2 * j + 0.5) * dy)])
                    const c = asLocal(params.texture.sample(uv), 'wcTap')
                    const rgb = c.member('rgb')
                    sum = sum.add(c)
                    sq = sq.add(rgb.mul(rgb))
                }
            }
            return {sum, sq}
        }
        return {quadrants: [quadrant(-1, -1), quadrant(1, -1), quadrant(-1, 1), quadrant(1, 1)], invN}
    })
}

/**
 * Fine paper grain for the pigment to settle into, one brightness value per screen pixel.
 *
 * `amount` runs 0 to 1; at 0 the grain is left out entirely. Returns the stage for the `grain`
 * slot of `wash`. Pass the same prop as `paper` there too.
 *
 * @example
 * ```ts
 * grain: paperGrain({amount: p('paper')})
 * ```
 * @tip Give the `amount` prop `recompile: crosses(0)`.
 * @see wash
 */
// Maintainer: device-pixel value noise (watercolorGrain over viewportSize). amount === 0 folds to
// a literal 0 at composition time, hence the crosses(0) rule.
export function paperGrain(slots: {amount: PropRef}): GatherStage {
    return (params) => {
        const amount = (params.propValues[slots.amount.name] as number) ?? 0
        return amount === 0 ? floatE(0) : call(stylizePaints.watercolorGrain, 'watercolorGrain', [params.ctx.uv, params.ctx.viewportSize])
    }
}

/**
 * The watercolor look: the layer inside flattened into soft patches of color over paper grain.
 *
 * Closes the recipe and returns the effect for `effect:`. `strength` runs from 0 (the original)
 * to 1 (the full wash). `paper` (0 to 1) sets how much grain shows and how far the result tints
 * toward `paperColor`. `at`, `brush` and `grain` are the stages from `bleedWarp`, `kuwahara`
 * and `paperGrain`.
 *
 * @example
 * ```ts
 * effect: wash({at: wet, brush: kuwahara({at: wet, radius: p('radius')}), grain: paperGrain({amount: p('paper')}), paper: p('paper'), paperColor: p('paperColor'), strength: p('strength')})
 * ```
 * @tip `strength` at 0 skips the whole effect. Give the prop `recompile: crosses(0)`.
 * @see kuwahara, bleedWarp, paperGrain
 */
// Maintainer: the winning-quadrant Kuwahara mean (lowest variance) mixed back toward the original
// by `strength`, broken by the paper grain toward `paperColor`. The gather runs in PREMULTIPLIED
// space and unpremultiplies once in watercolorCompose, so the result is straight alpha.
// `strength` at 0 returns the original sample at composition time (the whole gather compiled
// away), hence the crosses(0) rule.
export function wash(slots: {
    at: GatherStage
    brush: KuwaharaBrush
    grain: GatherStage
    paper: PropRef
    paperColor: PropRef
    strength: PropRef
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: (params): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, params)
            const original = asLocal(params.sampleStraight(slots.at(params)), 'wcOriginal')

            // strength=0 → compose mixes entirely back to the original, so skip the gather outright.
            const strength = (params.propValues[slots.strength.name] as number) ?? 1
            if (strength === 0) return original

            const {quadrants: [q0, q1, q2, q3], invN} = slots.brush(params)
            const grain = slots.grain(params)
            const finalRGB = call(stylizePaints.watercolorCompose, 'watercolorCompose', [
                q0.sum, q1.sum, q2.sum, q3.sum, q0.sq, q1.sq, q2.sq, q3.sq,
                floatE(invN), original, u(slots.paperColor), u(slots.paper), u(slots.strength), grain,
            ])
            // Output is straight alpha (the winning quadrant mean is unpremultiplied in compose).
            return vec4(finalRGB, original.member('a'))
        },
    })
}

// ═══ Chalk parts ════════════════════════════════════════════════════════════════════════════════

/** The ring of edge samples `sobelTaps` builds around each pixel. Pass it to `chalkSketch`. */
export type SobelRing = (params: RttFilterParams) => {sobelA: Expr; sobelB: Expr; centre: Expr; centreLum: Expr}

/**
 * Eight brightness samples in a ring around each pixel, for finding edges.
 *
 * `spacing` is the distance to the ring in device pixels (about 0.5 to 4; thicker strokes at
 * larger values). Returns the ring for `chalkSketch`.
 *
 * @example
 * ```ts
 * edges: sobelTaps({spacing: p('edgeThickness')})
 * ```
 * @see chalkSketch
 */
// Maintainer: 8 straight-alpha tap luminances at `spacing` texels around the fragment, packed
// `[tl, t, tr, l]` / `[r, bl, b, br]`, plus the centre sample and its luminance. Multi-tap → a
// true gather, never a uvRemap candidate.
export function sobelTaps(slots: {spacing: PropRef}): SobelRing {
    return shared((params) => {
        const {sampleStraight, ctx} = params
        const viewport = ctx.viewportSize
        const lumAt = (ox: number, oy: number): Expr => {
            const uv = call(stylizePaints.chalkOffsetUV, 'chalkOffsetUV', [ctx.uv, viewport, uniformOf(slots.spacing, params), floatE(ox), floatE(oy)])
            return call(stylizePaints.chalkLuma, 'chalkLuma', [sampleStraight(uv)])
        }
        const tl = lumAt(-1, -1)
        const t = lumAt(0, -1)
        const tr = lumAt(1, -1)
        const l = lumAt(-1, 0)
        const r = lumAt(1, 0)
        const bl = lumAt(-1, 1)
        const bo = lumAt(0, 1)
        const br = lumAt(1, 1)
        const centre = sampleStraight(ctx.uv)
        const centreLum = call(stylizePaints.chalkLuma, 'chalkLuma', [centre])
        return {sobelA: vec4(tl, t, tr, l), sobelB: vec4(r, bl, bo, br), centre, centreLum}
    })
}

/**
 * The chalk-drawing look: edges become chalk strokes and dark areas fill with cross-hatching.
 *
 * Closes the recipe and returns the effect for `effect:`. `sensitivity` (0 to 1) sets how faint
 * an edge still draws. `hatchScale` is the hatch line spacing in device pixels. `shading` (0 to
 * 1) is how much cross-hatch fills the dark regions and `grain` (0 to 1) how dusty the strokes
 * are. `board` and `chalk` are colors. Alpha follows the layer inside.
 *
 * @example
 * ```ts
 * effect: chalkSketch({edges: sobelTaps({spacing: p('edgeThickness')}), sensitivity: p('edgeSensitivity'), hatchScale: p('hatchScale'), shading: p('shading'), grain: p('grain'), board: p('boardColor'), chalk: p('chalkColor')})
 * ```
 * @see sobelTaps, linework
 */
// Maintainer: the Sobel edge magnitude from the ring becomes the stroke outline, three hatch line
// families (gated by darkness at 0.22 / 0.5 / 0.78) fill darker regions, and value-noise dust
// breaks up the strokes over the board color. Straight alpha throughout (every tap is
// unpremultiplied).
export function chalkSketch(slots: {
    edges: SobelRing
    sensitivity: PropRef
    hatchScale: PropRef
    shading: PropRef
    grain: PropRef
    board: PropRef
    chalk: PropRef
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: (params): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, params)
            const {sobelA, sobelB, centre, centreLum} = slots.edges(params)
            const misc = vec4(centreLum, centre.member('a'), 0, 0)
            const tuning = vec4(u(slots.sensitivity), u(slots.hatchScale), u(slots.shading), u(slots.grain))
            return call(stylizePaints.chalkboardCompose, 'chalkboardCompose', [
                sobelA, sobelB, params.ctx.uv, params.ctx.viewportSize, misc, tuning, u(slots.board), u(slots.chalk),
            ])
        },
    })
}

// ═══ ASCII parts (custom tier — CPU-rasterised glyph atlas) ═════════════════════════════════════

/** A stage of the ASCII look: one value per pixel, computed in the fragment. */
export type FragmentStage = (params: GpuFragmentParams) => Expr

/**
 * The character grid: which cell each pixel falls in, and where it sits inside that cell.
 *
 * `cellSize` is the cell size in pixels on a 1080-pixel-tall canvas; it scales with the canvas
 * height. `spacing` is how much of the cell the character fills (1 fills it). Build it once and
 * share it with `cellSample`, `glyphFor` and `glyphTint`.
 *
 * @example
 * ```ts
 * const grid = charGrid({cellSize: p('cellSize'), spacing: p('spacing')})
 * ```
 * @see cellSample, glyphFor, glyphTint
 */
// Maintainer: returns the AsciiGrid struct (`cellCenter`, `cellUV`, `isOutside`), memoised per
// composition and shared by the cell sample, the glyph lookup and the tint.
export function charGrid(slots: {cellSize: PropRef; spacing: PropRef}): FragmentStage {
    return shared((params) => call(stylizePaints.asciiGrid, 'asciiGrid', [
        params.ctx.uv, params.ctx.viewportSize, uniformOf(slots.cellSize, params), uniformOf(slots.spacing, params),
    ]))
}

/**
 * The color of the layer inside at the centre of each character cell.
 *
 * One sample per cell, straight alpha. Feeds `glyphFor` (to pick the character) and `glyphTint`
 * (to color it).
 *
 * @example
 * ```ts
 * const cell = cellSample({grid})
 * ```
 * @see charGrid, glyphFor
 */
// Maintainer: the child converted to a texture and sampled once at `cellCenter`, unpremultiplied.
export function cellSample(slots: {grid: FragmentStage}): FragmentStage {
    return shared((params) => {
        const child = params.convertToTexture(params.childNode!)
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [child.sample(slots.grid(params).member('cellCenter'))])
    })
}

/**
 * The character drawn in each cell, picked by how bright the cell is.
 *
 * `gamma` bends the brightness curve: below 1 favours the light characters, above 1 the dark
 * ones. `atlas` is your function that draws the character set into a texture and returns it
 * (the Ascii shader shows one). The definition needs three extra fields the atlas fills in:
 * `_charCount`, `_atlasScale` and `_atlasSize`.
 *
 * @example
 * ```ts
 * glyph: glyphFor({grid, cell, gamma: p('gamma'), atlas: setupGlyphAtlas})
 * ```
 * @see glyphTint, cellSample
 */
// Maintainer: cell brightness (gamma-curved, inverted) → character index → atlas cell, sampled
// nearest/clamped at the cell-local UV. The `atlas` slot runs the host raster lifecycle at
// composition time and returns the glyph texture.
export function glyphFor(slots: {
    grid: FragmentStage
    cell: FragmentStage
    gamma: PropRef
    atlas: (params: GpuFragmentParams) => KitTexture
}): FragmentStage {
    return (params) => {
        const atlasKit = slots.atlas(params)
        const atlasUV = call(stylizePaints.asciiAtlasUV, 'asciiAtlasUV', [
            slots.cell(params).member('rgb'), uniformOf(slots.gamma, params),
            params.uniforms._charCount, params.uniforms._atlasSize, params.uniforms._atlasScale,
            slots.grid(params).member('cellUV'),
        ])
        return atlasKit.sample(atlasUV, 'nearestClamp')
    }
}

/**
 * The finished ASCII look: each character tinted with its cell's color, everything else transparent.
 *
 * Returns the fragment for `gpu.fragment`. Cells whose alpha is below `alphaThreshold` (0 to 1)
 * turn transparent. With `preserveAlpha` on, the output keeps the alpha of the layer inside;
 * off, visible characters are fully opaque.
 *
 * @example
 * ```ts
 * gpu: {fragment: glyphTint({grid, cell, glyph, alphaThreshold: p('alphaThreshold'), preserveAlpha: p('preserveAlpha')})}
 * ```
 * @see glyphFor, charGrid
 */
// Maintainer: background (glyph brightness < 0.1), out-of-cell and below-threshold pixels get
// alpha 0. Returns ZERO without a child (the custom tier owns the child guard).
export function glyphTint(slots: {
    grid: FragmentStage
    cell: FragmentStage
    glyph: FragmentStage
    alphaThreshold: PropRef
    preserveAlpha: PropRef
}): FragmentStage {
    return (params) => {
        if (!params.childNode) return ZERO
        const cellColor = slots.cell(params)
        const glyph = slots.glyph(params)
        return call(stylizePaints.asciiCompose, 'asciiCompose', [
            glyph.member('rgb'), cellColor, slots.grid(params).member('isOutside'),
            uniformOf(slots.alphaThreshold, params), uniformOf(slots.preserveAlpha, params),
        ])
    }
}

// ═══ Placed-box frame ═════════════════════════════════════════════════════════════════════════

const OBF_DEG_TO_RAD = constants.DEG_TO_RAD

/** @internal The oriented-box placement kernel behind the Text shader. */
// Oriented-box placement field: aspect-corrected screen UV → rotated box-local UV + inside mask.
// `center` is the TRANSFORMED position prop (transformPosition stores (x, 1-y)), so `1 - center.y`
// recovers the authored y. Returns vec3(localU, localV, insideMask); the consumer samples its
// content texture at `.xy` and gates by `.z`. No V-flip — media textures written via
// `copyExternalImageToTexture` are top-left origin, matching the local V directly.
export const orientedBoxField = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.vec3f)(
    (uv, aspect, center, rotationDeg, halfW, halfH) => {
        'use gpu'
        const aspectUV = d.vec2f(uv.x * aspect, uv.y)
        const centerPos = d.vec2f(center.x * aspect, 1.0 - center.y)
        const dx = aspectUV.x - centerPos.x
        const dy = aspectUV.y - centerPos.y
        const rotRad = rotationDeg * OBF_DEG_TO_RAD
        const cosR = std.cos(rotRad)
        const sinR = std.sin(rotRad)
        const rdx = dx * cosR + dy * sinR
        const rdy = dy * cosR - dx * sinR
        const localU = rdx / (halfW * 2.0) + 0.5
        const localV = rdy / (halfH * 2.0) + 0.5
        const insideMask = std.step(0.0, localU) * std.step(localU, 1.0) * std.step(0.0, localV) * std.step(localV, 1.0)
        return d.vec3f(localU, localV, insideMask)
    },
)

// ═══ Line-engraving parts ═════════════════════════════════════════════════════════════════════

const HP_TAU = constants.TAU
const HP_DEG_TO_RAD = constants.DEG_TO_RAD

/** @internal One hatched line plate of the engraving; `linework` is the word. */
// One hatched line plate. Lines run along `angleDeg`; the coordinate perpendicular to them
// carries a cosine wave whose crests are inked. `level` (0 = black, 1 = white) slides the ink
// threshold across the wave, so darker tone → wider ink until the lines merge to solid;
// `reliefPhase` shifts the wave with image brightness — the classic engraved "lines climb over
// the form" look. `aa` is the analytic per-pixel wave footprint used for the smoothstep edge.
export const hatchPlate = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (p, angleDeg, frequency, reliefPhase, level, aa) => {
        'use gpu'
        const angleRad = angleDeg * HP_DEG_TO_RAD
        const c = std.cos(angleRad)
        const s = std.sin(angleRad)
        // Coordinate perpendicular to the line direction (lines run along (c, s)).
        const w = p.x * (s * -1.0) + p.y * c
        const phase = w * frequency * HP_TAU + reliefPhase
        const v = std.cos(phase)
        // level 0 (black) → threshold below the wave → solid ink; level 1 (white) → above → none.
        const thr = std.mix(-1.15, 1.15, std.clamp(level, 0.0, 1.0))
        return std.smoothstep(thr - aa, thr + aa, v)
    })

/** @internal The spiral plate of the engraving; `linework` is the word. */
// The spiral plate: an Archimedean spiral around `c` — rings spaced 1/frequency apart, each
// advancing one full spacing per turn, so the line spirals continuously outward from the centre
// (the classic guilloché portrait cut). Same tonal threshold + relief displacement as the
// straight plate; the angular term's footprint grows toward the centre, so the spiral naturally
// tightens into a dense knot there.
export const spiralPlate = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (p, c, frequency, reliefPhase, level, aa) => {
        'use gpu'
        const dv = p.sub(c)
        const r = std.length(dv)
        const theta = std.atan2(dv.y, dv.x)
        const phase = (r * frequency - theta * (1.0 / HP_TAU)) * HP_TAU + reliefPhase
        const v = std.cos(phase)
        const thr = std.mix(-1.15, 1.15, std.clamp(level, 0.0, 1.0))
        return std.smoothstep(thr - aa, thr + aa, v)
    })

/** @internal The per-style engraving composite factory behind `linework`. */
// The linework composite (factory — `style` is a compile-time JS branch: 0 = single line plate,
// 1 = cross-hatch (base + a +72° plate engaging in the shadows + a −38° plate in the deepest
// blacks — the classic copper-plate build-up), 2 = one continuous spiral cut around `center`).
//
// Shared tonal front-end: pivot-0.5 contrast curve on luminance, wavy burin-stroke domain warp,
// analytic AA, brightness→phase relief — then the style's plate stack inked between `paper`
// and `ink`. p0 = (angleDeg, frequency, relief, waviness); p1 = (contrast, viewportY, 0, 0).
export function makeLineworkComposite(style: number) {
    return tgpu.fn(
        [d.vec4f, d.vec2f, d.f32, d.vec4f, d.vec4f, d.vec2f, d.vec4f, d.vec4f], d.vec4f)(
        (childColor, uv, aspect, paper, ink, center, p0, p1) => {
            'use gpu'
            const angleDeg = p0.x
            const frequency = p0.y
            const relief = p0.z
            const waviness = p0.w
            const contrast = p1.x
            const viewportY = p1.y

            // Tonal level: luminance through a pivot-0.5 contrast curve.
            const lum = std.dot(d.vec3f(childColor.x, childColor.y, childColor.z), d.vec3f(0.299, 0.587, 0.114))
            const level = std.clamp(0.5 + (lum - 0.5) * contrast, 0.0, 1.0)

            // Wavy domain: two decorrelated noise channels bend the line work organically.
            const pa = d.vec2f(uv.x * aspect, uv.y)
            const nx = noise.mxNoiseFloat2(pa.mul(3.1))
            const ny = noise.mxNoiseFloat2(pa.mul(3.1).add(d.vec2f(7.31, 3.77)))
            const p = pa.add(d.vec2f(nx, ny).mul(waviness * 0.06))

            // Analytic AA: the wave's max per-pixel footprint (phase slope / viewport height).
            const aa = std.clamp(frequency * HP_TAU / std.max(viewportY, 1.0), 0.02, 1.0)

            // Brightness shifts the wave phase by up to ~1.5 periods (frequency-independent relief).
            const reliefPhase = relief * level * 9.42

            let cover = d.f32(0)
            if (style === 2) {
                // Spiral cut around the configurable centre (transformPosition stores (x, 1 − y)).
                const sc = d.vec2f(center.x * aspect, 1.0 - center.y)
                cover = spiralPlate(p, sc, frequency, reliefPhase, level, aa)
            } else if (style === 1) {
                // Base plate + shadow cross plate (+72°) + deep-black plate (−38°).
                const ink1 = hatchPlate(p, angleDeg, frequency, reliefPhase, level, aa)
                const level2 = std.clamp(level / 0.55, 0.0, 1.0)
                const ink2 = hatchPlate(p, angleDeg + 72.0, frequency * 0.92, reliefPhase * 0.7, level2, aa)
                const level3 = std.clamp(level / 0.28, 0.0, 1.0)
                const ink3 = hatchPlate(p, angleDeg - 38.0, frequency * 1.13, reliefPhase * 0.5, level3, aa)
                cover = std.max(ink1, std.max(ink2, ink3))
            } else {
                cover = hatchPlate(p, angleDeg, frequency, reliefPhase, level, aa)
            }

            const rgb = std.mix(d.vec3f(paper.x, paper.y, paper.z), d.vec3f(ink.x, ink.y, ink.z), d.vec3f(cover))
            const alpha = childColor.w * std.mix(paper.w, ink.w, cover)
            return d.vec4f(rgb.x, rgb.y, rgb.z, alpha)
        }).$name(`lineworkComposite_${style}`)
}

/** Style names → plate-stack modes, matching the linework `style` prop transform. */
const LINEWORK_MODES: Record<string, number> = {line: 0, crosshatch: 1, spiral: 2}

/**
 * The engraving look: the layer inside redrawn as ink lines that swell where it is dark.
 *
 * Closes the recipe and returns the effect for `effect:`. `style` is a compile-time select:
 * `'line'` (one set of parallel lines), `'crosshatch'` (more sets appear in the shadows) or
 * `'spiral'` (one line coiling out from `center`, a position prop). `frequency` is how many
 * lines span the canvas height. `angle` is in degrees. `relief` (0 to 2) is how far brightness
 * pushes the lines sideways, `waviness` (0 to 1) how much they meander, and `contrast` (0.25 to
 * 3) the tone curve before inking. `ink` and `paper` are colors.
 *
 * @example
 * ```ts
 * effect: linework({style: p('style'), frequency: p('frequency'), angle: p('angle'), center: p('center'), relief: p('relief'), waviness: p('waviness'), contrast: p('contrast'), ink: p('inkColor'), paper: p('paperColor')})
 * ```
 * @tip Set `blendWithChildren: false` on the definition. The lines replace the layer rather than sit on it.
 * @see chalkSketch
 */
// Maintainer: sample the composed child once (unpremultiplied) and redraw it as luminance-
// displaced line work between `paper` and `ink`. `style` binds a compile-time prop (a number or
// one of LINEWORK_MODES' names) — exactly one plate stack is emitted per composition. The
// composite works on and returns STRAIGHT alpha, so no unpremultiply tail is appended.
export function linework(slots: {
    style: PropRef
    frequency: PropRef
    angle: PropRef
    center: PropRef
    relief: PropRef
    waviness: PropRef
    contrast: PropRef
    ink: PropRef
    paper: PropRef
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: ({sampleStraight, ctx, uniforms, propValues}): Expr => {
            const childColor = sampleStraight(ctx.uv)
            const raw = propValues[slots.style.name]
            const style = typeof raw === 'number' ? raw : (LINEWORK_MODES[String(raw)] ?? 1)
            const composite = makeLineworkComposite(style)
            const p0 = vec4(uniforms[slots.angle.name], uniforms[slots.frequency.name], uniforms[slots.relief.name], uniforms[slots.waviness.name])
            const p1 = vec4(uniforms[slots.contrast.name], ctx.viewportSize.member('y'), floatE(0), floatE(0))
            return call(composite, `lineworkComposite_${style}`, [
                childColor, ctx.uv, ctx.aspect, uniforms[slots.paper.name], uniforms[slots.ink.name], uniforms[slots.center.name], p0, p1,
            ])
        },
    })
}
