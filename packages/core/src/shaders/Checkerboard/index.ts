import {defineShader, p} from "@coreroot/std"
import {checkerCells, strokeOver} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    cells: number,
    softness: number,
    colorSpace: string
}

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Checkerboard",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Classic checkerboard pattern with two alternating colors",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and Checkerboard rasterises at full canvas resolution against that distorted UV.
    // The Quilez analytical filter operates on the looked-up UV's derivatives, preserving cell
    // crispness through arbitrary distortions — much better than RTT bilinear.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#cccccc",
            transform: transformColor,
            description: "First color of the checkerboard pattern",
            ui: {
                type: 'color',
                label: 'Color A',
                group: 'Colors'
            }
        },
        colorB: {
            default: "#999999",
            transform: transformColor,
            description: "Second color of the checkerboard pattern",
            ui: {
                type: 'color',
                label: 'Color B',
                group: 'Colors'
            }
        },
        cells: {
            default: 8,
            description: "Number of cells along the canvas height (creates square cells). Switch the unit to px in the editor to set an absolute cell size instead.",
            ui: {
                type: ['range', 'map'],
                min: 1,
                max: 50,
                step: 1,
                label: 'Cells',
                group: 'Effect',
                // Inverse-dimensional: the plain number is a cell COUNT (legacy / resize-relative),
                // but the unit chip can switch to px, where the value is
                // an absolute cell size that resolves to `count = canvasHeight / px` (resize-stable).
                dimensional: 'count-canvas-height'
            }
        },
        softness: {
            default: 0,
            description: "Smoothness of the transition between colors (0 = hard edges, 1 = very soft)",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 1,
                step: 0.1,
                label: 'Softness',
                group: 'Effect'
            }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        }
    },

    // The recipe: the anti-aliased checker blend factor mixes the two colors in the chosen
    // color space.
    paint: strokeOver({
        fill: p('colorA'),
        stroke: p('colorB'),
        mask: checkerCells({cells: p('cells'), softness: p('softness')}),
        space: p('colorSpace'),
    })
})

export default componentDefinition
