import {defineShader, p, uniformOf} from "@coreroot/std"
import {animatedTime} from "@coreroot/gpu/porters"

const PI = 3.141592653589793
import {surfaceOf} from "@coreroot/std/frames"
import {colorLadder3} from "@coreroot/std/paint/fields"
import {surfaceNoiseAt} from "@coreroot/std/paint/materials"
import {add, div, local, max, mul, sin, smoothstep, vec2} from "@coreroot/std/math"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorC: Parameters<typeof transformColor>[0]
    scale: number
    turbulence: number
    speed: number
    seed: number
    colorSpace: string
}

// std generator: noise-warped sine veins driving a base → tone → depth color ladder. The
// hand-tuned octave decorrelation constants live in the kit body — they are the look.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Marble",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Classic marble swirl and vein texture using noise-warped sine waves",
    acceptsUVContext: true,
    // Per-node animated time drifting the turbulence field.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#ffffff",
            transform: transformColor,
            description: "Base background color of the marble",
            ui: { type: 'color', label: 'Base', group: 'Colors' }
        },
        colorB: {
            default: "#3a2d54",
            transform: transformColor,
            description: "Secondary marble tone",
            ui: { type: 'color', label: 'Tone', group: 'Colors' }
        },
        colorC: {
            default: "#0f0f0f",
            transform: transformColor,
            description: "Deepest marble color",
            ui: { type: 'color', label: 'Depth', group: 'Colors' }
        },
        scale: {
            default: 2.0,
            description: "Scale and density of the marble vein pattern",
            ui: { type: ['range', 'map'], min: 0.1, max: 10, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        turbulence: {
            default: 10,
            description: "Amount of noise-driven distortion applied to the veins",
            ui: { type: ['range', 'map'], min: 0, max: 50, step: 0.5, label: 'Turbulence', group: 'Effect' }
        },
        speed: {
            default: 0.05,
            description: "Animation speed",
            ui: { type: 'range', min: 0, max: 0.25, step: 0.005, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: "Random seed for pattern variation",
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },
    // The Marble look: three hand-tuned noise octaves bend a sine vein wave across the plane;
    // the wave splits into two bands (the deep one modulated by a slow cross-axis depth noise),
    // and the bands drive the base → tone → depth color ladder. The octaves stay hand-written
    // rather than a shared fbm: each uses a DIFFERENT seed multiplier and time rate (1.7/0.7,
    // 3.1/1.3) — decorrelation a uniform per-octave stride cannot reproduce.
    paint: (params) => {
        const {uv, viewport} = surfaceOf(params)
        // Guarded aspect (a zero-height canvas frame must not render NaN).
        const aspect = local(div(viewport.member('x'), max(viewport.member('y'), 1e-6)), 'aspect')
        const t = local(animatedTime(params), 'marbleTime')
        const seed = uniformOf(p('seed'), params)
        const pos = local(mul(vec2(mul(uv.member('x'), aspect), uv.member('y')), uniformOf(p('scale'), params)), 'plane')
        const x = pos.member('x')
        const y = pos.member('y')

        // Turbulence: three decorrelated octaves, drifting on the clock.
        const turbulence = local(add(add(
            surfaceNoiseAt(vec2(add(x, seed), add(y, t))),
            mul(surfaceNoiseAt(vec2(add(mul(x, 2), mul(seed, 1.7)), add(mul(y, 2), mul(t, 0.7)))), 0.5)),
            mul(surfaceNoiseAt(vec2(add(mul(x, 4), mul(seed, 3.1)), add(mul(y, 4), mul(t, 1.3)))), 0.25)), 'marbleTurb')

        // The vein wave: a turbulence-bent sine over x, squashed to [0,1].
        const vein = local(add(mul(sin(add(mul(x, PI), mul(turbulence, uniformOf(p('turbulence'), params)))), 0.5), 0.5), 'veinT')
        // How deep the deepest veins read: a slow cross-axis noise (~[0.2, 0.8]).
        const depth = add(mul(surfaceNoiseAt(vec2(add(mul(y, 1.5), mul(seed, 2)), mul(x, 1.2))), 0.3), 0.5)

        // Two bands → the three-color ladder in the compile-time colorSpace.
        return colorLadder3({a: p('colorA'), b: p('colorB'), c: p('colorC'), space: p('colorSpace')})(
            smoothstep(0.3, 0.6, vein), mul(smoothstep(0.6, 0.9, vein), depth), params)
    }
})

export default componentDefinition
