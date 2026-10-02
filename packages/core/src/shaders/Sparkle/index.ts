import type {Expr, GpuShaderDefinition} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {crosses, defineStd, isZero} from "@coreroot/std"
import type {RttFilterParams} from "@coreroot/std"
import {brightestNear, gatherStack} from "@coreroot/std/effects/blurs"
import {scatterPoints} from "@coreroot/std/paint/patterns"
import {starGlint} from "@coreroot/std/paint/light"
import {flashes} from "@coreroot/std/motion"
import {add, div, dot, max, min, mix, mul, local, smoothstep, splat3, sqrt, step, vec2, vec3, vec4} from "@coreroot/std/math"

export interface ComponentProps {
    size: number
    intensity: number
    threshold: number
    expand: number
    rayLength: number
    colorize: number
    speed: number
    seed: number
}

// The look's constants. Two scattered grids, the second at 2/3 the spacing, so the glints
// never line up. Brightness is judged perceptually (√ of linear luminance, close to sRGB
// lightness), and glints ramp in over THRESHOLD_BAND above `threshold`.
const GRID_SCALES = [1, 2 / 3]
const THRESHOLD_BAND = 0.12
// Star shape in CSS px: core sharpness, ray falloff across / along (at rayLength 1), ray gain.
const GLINT = {core: 0.9, rayWidth: 1.6, rayLength: 0.45, rays: 0.25}
// Each point blinks at its own rate in [0.8, 2.0] rad/s; ~45% of points are the bright ones.
const BLINK = {minRate: 0.8, rateSpread: 1.2, sharpness: 8 as const}
const BRIGHT_SHARE = 0.45
const DIM_LEVEL = 0.35
// How much the child's colour (normalised to unit luminance) carries into a tinted glint.
const HUE_GAIN = 0.6
const LUMA = [0.2126, 0.7152, 0.0722] as const
// Expand looks round each point on a ring of this many taps (only compiled in when expand > 0).
const EXPAND_TAPS = 4

/**
 * One grid of glints: scatter points over the four nearest cells, read the child at each (or
 * the brightest spot within `expand` of it), and light a blinking star wherever it's bright,
 * tinted toward that colour. Stars fade out by half a cell, so long rays never clip.
 */
function glintLayer(params: RttFilterParams, px: Expr, gridScale: number, layer: number, t: Expr): Expr {
    const u = params.uniforms
    const cellSize = local(mul(u.size, gridScale), `sparkleCell${layer}`)
    const expanding = ((params.propValues.expand as number) ?? 0) > 0
    const expandUV = local(div(vec2(u.expand, u.expand), params.ctx.logicalViewportSize), `sparkleExpand${layer}`)
    const rayRate = local(div(GLINT.rayLength, max(u.rayLength, 0.01)), `sparkleRays${layer}`)

    return scatterPoints(px, {cellSize, jitter: 0.6, seed: add(u.seed, layer * 0.31)})
        .map(({point, offset, random}) => {
            const source = brightestNear(params.sampleStraight, div(point, params.ctx.logicalViewportSize), {
                radius: expandUV,
                taps: expanding ? EXPAND_TAPS : 0,
                angle: layer * 45,
            })
            const presence = smoothstep(u.threshold, add(u.threshold, THRESHOLD_BAND), sqrt(source.luma))

            const blink = flashes(t, {
                rate: add(BLINK.minRate, mul(random.member('z'), BLINK.rateSpread)),
                offset: mul(random.member('x'), 6.283185307179586),
                sharpness: BLINK.sharpness,
            })
            const level = add(DIM_LEVEL, mul(1 - DIM_LEVEL, step(BRIGHT_SHARE, random.member('z'))))
            const star = starGlint(offset, {
                core: GLINT.core,
                rayWidth: GLINT.rayWidth,
                rayLength: rayRate,
                rays: GLINT.rays,
                reach: mul(cellSize, 0.5),
            })

            // Normalise by the colour's own (straight) luminance; `source.luma` is alpha-weighted.
            const rgb = local(source.color.member('rgb'), 'sparkleRgb')
            const hue = mul(div(rgb, max(dot(rgb, vec3(...LUMA)), 0.001)), HUE_GAIN)
            const tint = mix(splat3(1), hue, u.colorize)
            return mul(tint, mul(mul(mul(star, blink), presence), mul(level, u.intensity)))
        })
        .reduce((a, b) => add(a, b))
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Sparkle",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Stylize",
    description: "Twinkling star glints over the bright parts of the layer inside",
    animatedTime: {speed: 'speed'},
    props: {
        size: {
            default: 100,
            description: 'Spacing between glints, in CSS pixels (smaller means more glints)',
            ui: {type: ['range', 'map'], min: 10, max: 200, step: 1, label: 'Spacing', group: 'Effect'},
        },
        intensity: {
            default: 10,
            description: 'Brightness of the glints',
            ui: {type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'Intensity', group: 'Effect'},
        },
        threshold: {
            default: 0.2,
            description: 'How bright the layer must be before glints appear: 0 allows all but black, 0.5 is mid-grey and up, 1 only pure white',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Threshold', group: 'Effect'},
        },
        expand: {
            default: 0,
            description: 'Lets glints appear up to this many CSS pixels outside the bright areas',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Expand', group: 'Effect'},
        },
        rayLength: {
            default: 5,
            description: 'Length of the four rays around each glint (they fade out by half the spacing)',
            ui: {type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'Ray Length', group: 'Effect'},
        },
        colorize: {
            default: 0.45,
            description: 'How much each glint takes on the colour beneath it (0 is pure white)',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Colorize', group: 'Effect'},
        },
        speed: {
            default: 1,
            description: 'How fast the glints twinkle',
            ui: {type: 'range', min: 0, max: 5, step: 0.05, label: 'Speed', group: 'Animation'},
        },
        seed: {
            default: 0,
            description: 'Re-rolls where the glints sit and when they blink',
            ui: {type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation'},
        },
    },
    identityWhen: isZero('intensity'),

    // The child at this pixel plus every grid's glints, added as light. Glints may spill past
    // the child's edge, so coverage grows by the brightest glint channel.
    effect: gatherStack((params) => {
        const px = local(mul(params.ctx.uv, params.ctx.logicalViewportSize), 'sparklePx')
        const t = local(animatedTime(params), 'sparkleT')
        const glow = local(
            GRID_SCALES.map((scale, layer) => glintLayer(params, px, scale, layer, t)).reduce((a, b) => add(a, b)),
            'sparkleGlow',
        )
        const child = local(params.texture.sample(params.ctx.uv), 'sparkleChild')
        const spill = max(max(glow.member('r'), glow.member('g')), glow.member('b'))
        return vec4(add(child.member('rgb'), glow), min(add(child.member('a'), spill), 1))
    }, []),
    missingChildMessage: 'You must pass a child component into the Sparkle shader.',
})

export default componentDefinition
