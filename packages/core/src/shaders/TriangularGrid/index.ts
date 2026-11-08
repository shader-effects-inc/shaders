import {defineShader, p} from "@coreroot/std"
import {cellFrame, strokeOver, triangleLines, vary} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions, transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    cells: number
    thickness: number
    rotation: Parameters<typeof transformAngle>[0]
    softness: number
    variation: number
    speed: number
    speedVariance: number
    colorSpace: string
}

// The skewed equilateral-triangle line field (with per-row animated drift), drawn in a
// flipped-Y square lattice frame: `lines` masks the strokes, `shade` carries the per-cell
// variation factor.
const field = triangleLines({
    frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'flippedY'}),
    thickness: p('thickness'),
    softness: p('softness'),
    variation: p('variation'),
    speedVariance: p('speedVariance'),
})

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "TriangularGrid",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Tiling grid of equilateral triangles with optional animated row offsets",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and TriangularGrid rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    // Per-node animated time (createAnimatedTime equivalent); the paint noun reads it as the drift.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#1a1a1a",
            transform: transformColor,
            description: "Triangle fill color",
            ui: { type: 'color', label: 'Fill', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Grid line color",
            ui: { type: 'color', label: 'Lines', group: 'Colors' }
        },
        cells: {
            default: 8,
            description: "Number of triangle rows down the shortest canvas edge",
            ui: { type: ['range', 'map'], min: 1, max: 40, step: 1, label: 'Cells', group: 'Effect' }
        },
        thickness: {
            default: 1,
            description: "Thickness of the grid lines (0 = no lines)",
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
            description: "Per-triangle random lightening/darkening of the fill color",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Variation', group: 'Effect' }
        },
        speed: {
            default: 0,
            description: "Animates the triangle rows drifting horizontally (0 = static)",
            ui: { type: 'range', min: -3, max: 3, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        speedVariance: {
            default: 0.3,
            description: "Per-row random speed variance for irregular drifting motion",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Speed Variance', group: 'Animation' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },

    // The recipe: a flipped-Y lattice, the triangle line field drawn in it, per-cell variation
    // on the fill, lines stroked over in the chosen color space.
    paint: strokeOver({
        fill: vary(p('colorA'), field.shade),
        stroke: p('colorB'),
        mask: field.lines,
        space: p('colorSpace'),
    })
})

export default componentDefinition
