import {defineStd, p} from "@coreroot/std"
import {cellFrame, hexLines, strokeOver, vary} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions, transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    cells: number
    thickness: number
    rotation: Parameters<typeof transformAngle>[0]
    softness: number
    variation: number
    colorSpace: string
}

// The honeycomb line field (pointy-top hexagons), drawn in a clockwise lattice frame: `lines`
// masks the strokes, `shade` carries the per-cell variation factor.
const field = hexLines({
    frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}),
    thickness: p('thickness'),
    softness: p('softness'),
    variation: p('variation'),
})

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineStd<ComponentProps>({
    name: "HexGrid",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Honeycomb hexagonal grid pattern",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and HexGrid rasterises at full canvas resolution against that distorted UV. The
    // fwidth-based AA operates on the looked-up UV's derivatives, preserving line crispness through
    // arbitrary distortions — much better than RTT bilinear.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#000000",
            transform: transformColor,
            description: "Cell fill color",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Grid line color",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        cells: {
            default: 8,
            description: "Number of hexagons along the canvas height (the width fits as many as the aspect ratio allows)",
            ui: { type: ['range', 'map'], min: 1, max: 40, step: 1, label: 'Cells', group: 'Effect' }
        },
        thickness: {
            default: 1,
            description: "Thickness of the hex grid lines",
            ui: { type: ['range', 'map'], min: 0, max: 10, step: 0.1, label: 'Thickness', group: 'Effect' }
        },
        rotation: {
            default: 0,
            transform: transformAngle,
            description: "Rotation of the grid in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Softness of the line edges (0 = crisp, 1 = very soft)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        variation: {
            default: 0,
            description: "Per-cell random lightening/darkening of the cell fill color",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Variation', group: 'Effect' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },

    // The recipe: a clockwise lattice, the honeycomb line field drawn in it, per-cell variation
    // on the fill, lines stroked over in the chosen color space.
    paint: strokeOver({
        fill: vary(p('colorA'), field.shade),
        stroke: p('colorB'),
        mask: field.lines,
        space: p('colorSpace'),
    })
})

export default componentDefinition
