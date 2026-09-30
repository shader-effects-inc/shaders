import {defineStd, p} from "@coreroot/std"
import {cellFrame, gridLines, strokeOver, vary, sampleMapsAtCellCentres} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0],
    cellColor: Parameters<typeof transformColor>[0],
    cells: number,
    thickness: number,
    softness: number,
    variation: number,
    rotation: number,
    colorSpace: string
}

// The grid line field, drawn in a flipped-Y square lattice frame: `lines` masks the strokes,
// `shade` carries the per-cell variation factor.
const field = gridLines({
    frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'flippedY'}),
    thickness: p('thickness'),
    softness: p('softness'),
    variation: p('variation'),
})

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineStd<ComponentProps>({
    name: "Grid",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Simple grid lines pattern with adjustable thickness and rotation",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and Grid rasterises at full canvas resolution against that distorted UV. The
    // dpdx/dpdy-based AA operates on the looked-up UV's derivatives, preserving line crispness
    // through arbitrary distortions — much better than RTT bilinear.
    acceptsUVContext: true,
    props: {
        color: {
            default: "#ffffff",
            transform: transformColor,
            description: "The color of the grid lines",
            ui: { type: 'color', label: 'Line Color', group: 'Colors' }
        },
        cellColor: {
            default: "transparent",
            transform: transformColor,
            description: "Fill color of the cells (transparent = lines only). Pair with Variation for a tiled look.",
            ui: { type: 'color', label: 'Cell Color', group: 'Colors' }
        },
        cells: {
            default: 10,
            description: "Number of cells along the canvas height (cells stay square; the width fits as many as the aspect ratio allows)",
            ui: { type: ['range', 'map'], min: 1, max: 50, step: 1, label: 'Cells', group: 'Effect' }
        },
        thickness: {
            default: 1,
            description: "Thickness of grid lines (normalized, 0.0-1.0)",
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'Thickness', group: 'Effect' }
        },
        rotation: {
            default: 0,
            description: "Rotation of the grid in degrees. At 45° this produces a crosshatch/diamond pattern.",
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
    // Sample mapped props at the cell centre to prevent line thickness from varying per-fragment
    // within a cell (which clips lines at source boundaries instead of transitioning whole lines).
    mapSampleUVs: sampleMapsAtCellCentres({
        cells: p('cells'),
        rotation: p('rotation'),
        props: ['thickness', 'cells', 'rotation', 'softness'],
    }),

    // The recipe: a flipped-Y square lattice, the grid line field drawn in it, per-cell
    // variation on the cell fill, lines stroked over in the chosen color space. A transparent
    // cell color collapses to lines-only.
    paint: strokeOver({
        fill: vary(p('cellColor'), field.shade),
        stroke: p('color'),
        mask: field.lines,
        space: p('colorSpace'),
    })
})

export default componentDefinition
