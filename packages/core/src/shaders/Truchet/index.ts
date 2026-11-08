import {defineShader, p} from "@coreroot/std"
import {cellFrame, strokeOver, truchetArcs} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions, transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    cells: number
    thickness: number
    rotation: Parameters<typeof transformAngle>[0]
    softness: number
    seed: number
    colorSpace: string
}

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Truchet",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Quarter-circle arc tiles that connect to form organic, maze-like flowing curves",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and Truchet rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#000000",
            transform: transformColor,
            description: "Background color between the arcs",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Arc line color",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        cells: {
            default: 10,
            description: "Number of tiles across the shortest canvas edge",
            ui: { type: ['range', 'map'], min: 2, max: 40, step: 1, label: 'Cells', group: 'Effect' }
        },
        thickness: {
            default: 2,
            description: "Thickness of the arc lines",
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'Thickness', group: 'Effect' }
        },
        rotation: {
            default: 0,
            transform: transformAngle,
            description: "Rotation of the tiling in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Softness of the arc edges (0 = crisp, 1 = very soft)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: "Random seed — changes which tiles flip, producing a different maze pattern",
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

    // The recipe: a clockwise lattice, quarter-circle arc tiles drawn in it (orientation hashed
    // per tile, `seed` reshuffles the maze), arcs stroked over the background in the chosen
    // color space.
    paint: strokeOver({
        fill: p('colorA'),
        stroke: p('colorB'),
        mask: truchetArcs({
            frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}),
            thickness: p('thickness'),
            softness: p('softness'),
            seed: p('seed'),
        }),
        space: p('colorSpace'),
    })
})

export default componentDefinition
