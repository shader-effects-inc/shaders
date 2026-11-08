import {defineShader, p} from "@coreroot/std"
import {alphaOf, mixOf, strokeOver, withAlpha, zigzagBands} from "@coreroot/std/paint/patterns"
import {transformColor, transformAngle, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    count: number
    angle: Parameters<typeof transformAngle>[0]
    balance: number
    softness: number
    speed: number
    offset: number
    colorSpace: string
}

// The anti-aliased zigzag stripe mask, scrolled by the animation clock plus `offset` — shared
// by the color mix and the separate alpha mix.
const bands = zigzagBands({
    count: p('count'),
    angle: p('angle'),
    balance: p('balance'),
    softness: p('softness'),
    offset: p('offset'),
})

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Chevron",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Animated chevron / zigzag stripe pattern",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and this generator rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    // Per-node animated time driven by `speed`; the paint noun reads it as the scroll phase.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#000000",
            transform: transformColor,
            description: "First color",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Second color",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        count: {
            default: 5,
            description: "Number of chevron pairs visible",
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Count', group: 'Effect' }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: "Rotation angle of the chevrons",
            ui: { type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Angle', group: 'Effect' }
        },
        balance: {
            default: 0.5,
            description: "Ratio of the two colors",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Balance', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Edge softness",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        speed: {
            default: 0,
            description: "Animation speed",
            ui: { type: 'range', min: -2, max: 2, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        offset: {
            default: 0,
            description: "Phase offset for pattern positioning",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Offset', group: 'Animation' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },

    // The recipe: the zigzag mask mixes the two colors in the chosen color space, with the
    // alpha channel interpolated separately (a straight mix of the two color alphas).
    paint: withAlpha(
        strokeOver({fill: p('colorA'), stroke: p('colorB'), mask: bands, space: p('colorSpace')}),
        mixOf(alphaOf(p('colorA')), alphaOf(p('colorB')), bands),
    )
})

export default componentDefinition
