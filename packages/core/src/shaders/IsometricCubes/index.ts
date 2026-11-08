import {defineShader, p} from "@coreroot/std"
import {cellFrame, isoCubeFaces, mixOf, opaque, rgbOf, strokeOver, times} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions, transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    lineColor: Parameters<typeof transformColor>[0]
    cells: number
    thickness: number
    rotation: Parameters<typeof transformAngle>[0]
    softness: number
    colorVariation: number
    colorSpace: string
}

// The rhombille (tumbling-blocks) field, drawn in a clockwise lattice frame — three shaded
// rhombus faces per hexagon read as a 3D cube. `random` is the per-cube color hash, `tone` the
// face shading, `wire` the edge-line mask.
const faces = isoCubeFaces({
    frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'clockwise'}),
    thickness: p('thickness'),
    softness: p('softness'),
})

// Per-cube random blend between the two cube colors, scaled by the variation amount.
const cubeColor = strokeOver({
    fill: p('colorA'),
    stroke: p('colorB'),
    mask: times(faces.random, p('colorVariation')),
    space: p('colorSpace'),
})

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "IsometricCubes",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Isometric tumbling-blocks tiling — a 3D cube illusion (rhombille pattern)",
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#7c5cff",
            transform: transformColor,
            description: "Base cube color",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ff5c9d",
            transform: transformColor,
            description: "Second cube color, mixed in per-cube for variation",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        lineColor: {
            default: "#000000",
            transform: transformColor,
            description: "Color of the cube edge lines",
            ui: { type: 'color', label: 'Edges', group: 'Colors' }
        },
        cells: {
            default: 6,
            description: "Number of cubes across the shortest canvas edge",
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Cells', group: 'Effect' }
        },
        thickness: {
            default: 1,
            description: "Thickness of the cube edge lines (0 = no edges)",
            ui: { type: ['range', 'map'], min: 0, max: 10, step: 0.1, label: 'Thickness', group: 'Effect' }
        },
        rotation: {
            default: 0,
            transform: transformAngle,
            description: "Rotation of the tiling in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Softness of the edge lines (0 = crisp, 1 = very soft)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        colorVariation: {
            default: 1,
            description: "Per-cube random blend between the two colors (0 = uniform cubes, the clean 3D look)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Variation', group: 'Colors' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },

    // The recipe: the face tone shades each cube color, the edge wire draws over in the line
    // color, fully opaque output.
    paint: opaque(
        mixOf(times(rgbOf(cubeColor), faces.tone), rgbOf(p('lineColor')), faces.wire),
    )
})

export default componentDefinition
