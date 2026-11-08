import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, schema, uniformOf} from "@coreroot/std"
import {surfaceOf} from "@coreroot/std/frames"
import {quadBlend, warpStep} from "@coreroot/std/paint/fields"
import {volumeNoiseAt} from "@coreroot/std/paint/materials"
import {
    abs, add, clamp, cos, div, local, max, mix, mul, pow, rotate2, sin, smoothstep, splat3, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

// ── The FlowingGradient look (this data IS the look) ──────────────────────────────────────
// Uses 4π (full precision) for the noise-rotation range; the two chained warp levels carry
// their decorrelation offsets and reaches; fold masks peak where the color flips.
const FOUR_PI = 4 * Math.PI
const WARP_LEVELS = [
    {scale: 2.0, offsets: [[5.2, 1.3], [1.7, 9.2]] as [[number, number], [number, number]], reach: 0.25},
    {scale: 1.5, offsets: [[8.3, 2.8], [3.1, 7.4]] as [[number, number], [number, number]], reach: 0.2},
]

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorC: Parameters<typeof transformColor>[0]
    colorD: Parameters<typeof transformColor>[0]
    colorSpace: string
    speed: number
    distortion: number
    seed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "FlowingGradient",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Liquid silk gradient with organic flowing color bands",
    acceptsUVContext: true,
    // Per-node animated time, read inside the field part.
    animatedTime: { speed: 'speed' },
    // PRECONVERTED-COLORS RECIPE (Beam): for a non-linear color space the forward P3→working-space
    // conversion of the four endpoint colors is pixel-invariant, so the quadBlend part computes it ONCE per
    // frame on the CPU (dirty-keyed) into these vec3 extraFields and the GPU reads them via the
    // mixPreconverted* variants — the per-pixel cost drops from 6 forward + 3 back conversions to
    // one back-conversion.
    extraFields: {
        convA: { schema: schema.vec3f, initial: [0, 0, 0] },
        convB: { schema: schema.vec3f, initial: [0, 0, 0] },
        convC: { schema: schema.vec3f, initial: [0, 0, 0] },
        convD: { schema: schema.vec3f, initial: [0, 0, 0] }
    },
    props: {
        colorA: {
            default: '#0a0015',
            transform: transformColor,
            description: 'Deep background color',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#6b17e6',
            transform: transformColor,
            description: 'Primary accent color',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        colorC: {
            default: '#ff4d6a',
            transform: transformColor,
            description: 'Secondary accent color',
            ui: { type: 'color', label: 'Color C', group: 'Colors' }
        },
        colorD: {
            default: '#ff6b35',
            transform: transformColor,
            description: 'Tertiary accent color',
            ui: { type: 'color', label: 'Color D', group: 'Colors' }
        },
        colorSpace: {
            default: 'oklch',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        },
        speed: {
            default: 1,
            description: 'Animation speed',
            ui: { type: 'range', min: 0, max: 10, step: 0.1, label: 'Speed', group: 'Effect' }
        },
        distortion: {
            default: 0.5,
            description: 'Organic distortion intensity',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Distortion', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: 'Random seed for variation',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        }
    },

    // The fused rotate/warp/shade kernel stays one named part, hoisted to a local (each member
    // read would otherwise re-emit the whole 6-noise body); the separable tail — the 2D 4-color
    // blend in the compile-time color space, then the fold lighting — is the visible composition.
    // The FlowingGradient look: rotate the canvas by a drifting noise angle, shear it with a
    // sine wave, fold it through two chained domain warps, split the folded field into two
    // blend factors with variable hard/soft edges, 2D-blend four colors by them, and light the
    // folds. The data at the top of this file is the look; the words are the language.
    paint: (params) => {
        const {uv, viewport} = surfaceOf(params)
        // Guarded aspect (a zero-height canvas frame must not render NaN).
        const aspect = local(div(viewport.member('x'), max(viewport.member('y'), 1e-6)), 'aspect')
        const t = local(mul(animatedTime(params), 0.075), 'flowTime')
        const seedOff = local(mul(uniformOf(p('seed'), params), 0.17), 'flowSeed')
        const strength = local(uniformOf(p('distortion'), params), 'flowStrength')
        const centred = local(sub(uv, 0.5), 'flowCentred')

        // Organic rotation: one noise read spins the (aspect-squashed) canvas; time flows
        // through both spatial dims for directional motion. The squash/re-stretch aspect
        // handling is this look's own — deliberately NOT the shared D-1 framing.
        const spinNoise = volumeNoiseAt(vec3(
            add(mul(centred.member('x'), 0.8), mul(t, 0.4)),
            sub(mul(centred.member('y'), 0.8), mul(t, 0.25)), seedOff))
        const angle = local(mul(mul(spinNoise, FOUR_PI), strength), 'flowAngle')
        const squashed = vec2(centred.member('x'), div(centred.member('y'), aspect))
        const spun = local(rotate2(squashed, cos(angle), sin(angle)), 'flowSpun')
        const rot = local(vec2(spun.member('x'), mul(spun.member('y'), aspect)), 'flowRot')

        // Subtle sine shear: the y wave reads the just-updated x — the sequential dependence
        // IS the shear.
        const shearT = mul(t, 2)
        const shearX = local(add(rot.member('x'),
            mul(div(sin(add(mul(rot.member('y'), 5), shearT)), 50), strength)), 'flowShearX')
        const shearY = add(rot.member('y'),
            mul(div(sin(add(mul(shearX, 7.5), shearT)), 25), strength))

        // Two chained domain-warp levels; each level's noise samples ride along for the tails.
        const level1 = local(warpStep({
            at: vec2(shearX, shearY), scale: WARP_LEVELS[0].scale, z: add(mul(t, 0.5), seedOff),
            offsets: WARP_LEVELS[0].offsets, reach: WARP_LEVELS[0].reach, strength,
        }), 'flowWarp1')
        const tz = local(add(mul(t, 0.3), seedOff), 'flowTz')
        const level2 = local(warpStep({
            at: level1.member('xy'), scale: WARP_LEVELS[1].scale, z: tz,
            offsets: WARP_LEVELS[1].offsets, reach: WARP_LEVELS[1].reach, strength,
        }), 'flowWarp2')

        // Color zones with variable hard/soft edges: hardness comes from level 1's x-noise so
        // hard and soft boundary stretches flow with the warp.
        const zoneNoise = volumeNoiseAt(vec3(
            mul(level2.member('x'), 1.8), mul(level2.member('y'), 1.8), tz))
        const edgeHardness = local(pow(add(mul(level1.member('z'), 0.5), 0.5), 1.5), 'flowHardness')
        const edgeWidth = local(mix(0.5, 0.008, edgeHardness), 'flowEdge')
        const t1 = local(smoothstep(mul(edgeWidth, -1), edgeWidth, zoneNoise), 'flowT1')
        const t2 = local(smoothstep(mul(edgeWidth, -1), edgeWidth, level1.member('w')), 'flowT2')

        // 2D 4-color blend in the compile-time colorSpace.
        const base = local(quadBlend({
            a: p('colorA'), b: p('colorB'), c: p('colorC'), d: p('colorD'), space: p('colorSpace'),
        })(t1, t2, params), 'flowBase')

        // Fold lighting: peaks where the color flips fastest, plus surface-curvature brightness.
        const foldMask = (blend: typeof t1) => sub(1, abs(sub(mul(blend, 2), 1)))
        const foldHighlight = mul(max(foldMask(t1), foldMask(t2)), edgeHardness)
        const surfaceLight = add(mul(add(mul(level2.member('z'), 0.5), 0.5), 0.12), 0.94)
        const litScale = mul(surfaceLight, add(1, mul(foldHighlight, 0.35)))
        return vec4(clamp(mul(base.member('rgb'), litScale), splat3(0), splat3(1)), base.member('a'))
    },
})

export default componentDefinition
