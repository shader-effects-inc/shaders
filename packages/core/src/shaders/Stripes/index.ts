import {defineShader, p} from "@coreroot/std"
import {stripeBands, strokeOver} from "@coreroot/std/paint/patterns"
import {transformColor, transformAngle, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    angle: Parameters<typeof transformAngle>[0]
    density: number
    balance: number
    softness: number
    speed: number
    offset: number
    colorSpace: string
}

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Stripes",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Alternating colored stripes with animation",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and Stripes rasterises at full canvas resolution against that distorted UV. The
    // Quilez analytical filter operates on the looked-up UV's derivatives, preserving stripe
    // crispness through arbitrary distortions — much better than RTT bilinear.
    acceptsUVContext: true,
    // Per-node animated time (createAnimatedTime equivalent): the renderer registers `_animTime`
    // and advances it by `deltaTime * speed` (speed=0 pauses). The paint noun reads it as the
    // stripe scroll phase.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#000000",
            transform: transformColor,
            description: "First stripe color",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Second stripe color",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        angle: {
            default: 45,
            transform: transformAngle,
            description: "Angle of stripes in degrees",
            ui: { type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Angle', group: 'Effect' }
        },
        density: {
            default: 5,
            description: "Number of stripe pairs visible",
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Density', group: 'Effect' }
        },
        balance: {
            default: 0.5,
            description: "Ratio of the two colors",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Balance', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Edge softness",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Softness', group: 'Effect' }
        },
        speed: {
            default: 0.2,
            description: "Animation speed",
            ui: { type: 'range', min: -1, max: 1, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        offset: {
            default: 0,
            description: "Phase offset for pattern positioning",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Offset', group: 'Animation' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },

    // The recipe: the anti-aliased stripe mask (scrolled by the animation clock plus `offset`)
    // mixes the two colors in the chosen color space.
    paint: strokeOver({
        fill: p('colorA'),
        stroke: p('colorB'),
        mask: stripeBands({
            angle: p('angle'),
            density: p('density'),
            balance: p('balance'),
            softness: p('softness'),
            offset: p('offset'),
        }),
        space: p('colorSpace'),
    })
})

export default componentDefinition
