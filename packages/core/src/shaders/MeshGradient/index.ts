import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, uniformOf, crosses, schema} from "@coreroot/std"
import {surfaceOf} from "@coreroot/std/frames"
import {scatterField, scatteredAnchors, warpedPoint, wrapRamp, standardPalette} from "@coreroot/std/paint/fields"
import {volumeNoiseAt} from "@coreroot/std/paint/materials"
import {dithered} from "@coreroot/std/paint/compose"
import {add, clamp, cos, div, exp, length, local, max, mul, rotate2, sin, smoothstep, sub, vec2, vec3} from "@coreroot/std/math"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {isMobileGpuViewport} from "@coreroot/utilities/device"

// ── The MeshGradient look (this data IS the look) ─────────────────────────────────────────
// Domain-warp framing: field frequency, baked warp strength (the retired distortion prop's
// validated default) and morph rate.
const DOMAIN_SCALE = 1.6
const WARP_AMOUNT = 0.48
const WARP_RATE = 0.125
// Max vortex rotation (radians) at the frame's far corners when swirl = 1 (paper-shaders' 3.0).
const SWIRL_MAX = 3.0
// Softness-variation field: frequency over the warped query space, evolution rate, and the
// exp() strength that turns the signed noise into a symmetric ÷/× multiplier on the exponent.
const VAR_FREQ = 1.9
const VAR_STRENGTH = 1.4
const VAR_RATE = 0.045
// Max extra palette half-cycles the wrapping prop can add (wrapping 1 → 1 + WRAP_CYCLES).
const WRAP_CYCLES = 9
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    colorSpace: string
    count: number
    smoothness: number
    variation: number
    swirl: number
    drift: number
    wrapping: number
    speed: number
    seed: number
}

// Default palette — a dark-to-light multi-hue ramp (deep indigo → violet → magenta → orange →
// golden). The wrapped seams sweep THESE hues, so the default must be a rich ramp, not two colors.
const DEFAULT_STOPS: ColorStop[] = [
    {color: '#1a0533', position: 0},
    {color: '#6d2fd1', position: 0.26},
    {color: '#e04b9e', position: 0.52},
    {color: '#ff8c42', position: 0.76},
    {color: '#ffdf8e', position: 1},
]

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "MeshGradient",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Flowing mesh gradient of soft drifting color swaths whose seams wrap through the palette",
    acceptsUVContext: true,
    // Per-node animated time, read inside the field part.
    animatedTime: { speed: 'speed' },
    // The scatter constellation, hoisted off the pixel: the 8 anchor positions depend only on
    // count/seed/drift/aspect/time, so `scatteredAnchors` computes them once per frame on the CPU
    // and writes them here (8 vec2s packed 4-per-vec4 — uniform arrays need a 16-byte stride).
    extraFields: {
        meshAnchors: { schema: schema.arrayOf(schema.vec4f, 4), initial: new Array(16).fill(0) },
    },
    props: {
        colorA: {
            default: '#1a0533',
            transform: transformColor,
            description: 'First gradient color (two-color fallback when stops are cleared)',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#ffdf8e',
            transform: transformColor,
            description: 'Second gradient color (two-color fallback when stops are cleared)',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Unlike the other gradient shaders (default null → legacy pair), MeshGradient DEFAULTS to a
        // full multi-stop palette — the wrapped-seam look needs a dark-to-light multi-hue ramp.
        stops: {...colorStopsPropConfig(), default: DEFAULT_STOPS},
        colorSpace: {
            default: 'oklab',
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
        // Runtime uniform, NOT compileTime: the fixed-8 gated loop emits identical WGSL for every
        // count, so a compile-time prop would recompile per slider detent for zero per-frame gain.
        count: {
            default: 5,
            description: 'Number of color points scattered across the canvas',
            ui: { type: 'range', min: 2, max: 8, step: 1, label: 'Points', group: 'Effect' }
        },
        smoothness: {
            default: 2,
            description: 'How smoothly the color points blend into each other',
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.01, label: 'Smoothness', group: 'Effect' }
        },
        // Crossing 0 recomposes: at exactly 0 the variation noise read is compiled out (one fewer
        // Perlin-3D per pixel). Scrubbing anywhere above 0 is a plain uniform write.
        variation: {
            default: 0.35,
            recompile: crosses(0),
            description: 'Varies edge softness across the canvas — some color boundaries crisp, others diffuse',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Variation', group: 'Effect' }
        },
        swirl: {
            default: 0.3,
            description: 'Vortex rotation that spirals the field around the frame centre — negative values spin the other way',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Swirl', group: 'Effect' }
        },
        drift: {
            default: 0.5,
            description: 'How far the color points wander from their home positions',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Drift', group: 'Effect' }
        },
        wrapping: {
            default: 0,
            description: 'Wraps the palette back through itself where colors meet — banded seams where the field pinches, smooth elsewhere',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Wrapping', group: 'Effect' }
        },
        speed: {
            default: 1,
            description: 'Animation speed',
            ui: { type: 'range', min: 0, max: 10, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: 'Random seed for the point layout and palette assignment',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        }
    },

    // The fused warp/swirl/IDW field kernel stays one named part; the separable tail — palette
    // wrapping, the standard multi-stop palette (reading colorA/colorB/stops/colorSpace), and
    // the always-on output dither — is the visible composition.
    // The MeshGradient look: swirl the canvas into a gentle vortex, warp it organically, blend
    // a drifting constellation of color anchors by nearness (softness varying across the
    // field), then wrap the palette and dither the output.
    paint: (params) => {
        const {uv, viewport} = surfaceOf(params)
        // Guarded aspect (a zero-height canvas frame must not render NaN).
        const aspect = local(div(viewport.member('x'), max(viewport.member('y'), 1e-6)), 'aspect')
        const t = local(animatedTime(params), 'meshTime')
        const seed = uniformOf(p('seed'), params)

        // Vortex swirl about the frame centre (paper-shaders' move): rotation grows with
        // radius, so the centre stays put and the field wraps into a spiral toward the edges.
        const centred = local(vec2(sub(mul(uv.member('x'), aspect), mul(aspect, 0.5)), sub(uv.member('y'), 0.5)), 'meshCentred')
        const swirlAngle = local(mul(mul(-SWIRL_MAX, uniformOf(p('swirl'), params)), smoothstep(0, 1, length(centred))), 'swirlAngle')
        const swirled = local(rotate2(centred, cos(swirlAngle), sin(swirlAngle)), 'meshRot')
        const uvS = local(vec2(add(swirled.member('x'), mul(aspect, 0.5)), add(swirled.member('y'), 0.5)), 'meshSwirled')

        // Organic domain warp — the folded creases are where the field compresses and the
        // palette wraps. The warped position is divided back down so anchor distances stay in
        // UV units.
        // Mobile tier (coarse pointer, ≤1366px): one warp level instead of two — half the noise
        // reads, shallower folds. Decided at build time like sdf3d's march budget.
        const warped = local(warpedPoint({
            at: vec3(mul(uvS.member('x'), DOMAIN_SCALE), mul(uvS.member('y'), DOMAIN_SCALE), mul(seed, 7.31)),
            time: mul(t, WARP_RATE),
            amount: WARP_AMOUNT,
            levels: isMobileGpuViewport() ? 1 : 2,
        }), 'meshWarp')
        const q = local(vec2(div(warped.member('x'), DOMAIN_SCALE), div(warped.member('y'), DOMAIN_SCALE)), 'meshQ')

        // Blend softness: invert smoothness (higher = blobs spread further), then let a slow
        // signed noise field scale the exponent log-symmetrically — one boundary drifts between
        // crisp and diffuse along its length. Sampled in warped space so the hard and soft
        // regions flow with the field.
        const basePower = div(8, add(uniformOf(p('smoothness'), params), 0.5))
        // variation === 0 is compiled out (recompile: crosses(0)): exp(0) = 1, so the clamp of the
        // bare exponent is the identical result without the noise read.
        // Any value other than 0/undefined keeps the read (a driver object is a live, nonzero value).
        const variation = params.propValues.variation
        const variationActive = variation !== undefined && variation !== 0
        const softness = variationActive
            ? (() => {
                const varNoise = volumeNoiseAt(vec3(mul(q.member('x'), VAR_FREQ), mul(q.member('y'), VAR_FREQ),
                    add(mul(seed, 3.17), mul(t, VAR_RATE))))
                return clamp(mul(basePower, exp(mul(mul(varNoise, uniformOf(p('variation'), params)), VAR_STRENGTH))), 0.6, 16)
            })()
            : clamp(basePower, 0.6, 16)

        // The field: nearness-blended color anchors, drifting. The anchors themselves are
        // pixel-invariant, so they arrive pre-computed from the CPU each frame.
        const anchors = scatteredAnchors(params, {count: p('count'), seed: p('seed'), drift: p('drift'), field: 'meshAnchors'})
        const field = scatterField({
            at: q,
            count: uniformOf(p('count'), params),
            seed,
            drift: uniformOf(p('drift'), params),
            aspect,
            time: t,
            softness,
            anchors,
        })

        // Wrap the palette (identity at wrapping 0), look it up, and dither the output.
        const wrapped = wrapRamp({t: field, cycles: add(1, mul(uniformOf(p('wrapping'), params), WRAP_CYCLES))})
        return dithered(standardPalette(wrapped, params), params)
    },
})

export default componentDefinition
