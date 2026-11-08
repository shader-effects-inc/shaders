import {defineShader, p} from "@coreroot/std"
import {brickCourses, strokeOver, vary} from "@coreroot/std/paint/patterns"
import {transformColor, transformAngle, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorBrick: Parameters<typeof transformColor>[0]
    colorMortar: Parameters<typeof transformColor>[0]
    cellsX: number
    cellsY: number
    mortar: number
    softness: number
    variation: number
    rotation: Parameters<typeof transformAngle>[0]
    speed: number
    offset: number
    speedVariance: number
    seed: number
    colorSpace: string
}

// The brick-course field: staggered rows with mortar gaps, rotation, static offset, and per-row
// animated drift. `bricks` masks the brick coverage, `shade` carries the per-brick variation.
const field = brickCourses({
    cellsX: p('cellsX'),
    cellsY: p('cellsY'),
    mortar: p('mortar'),
    softness: p('softness'),
    variation: p('variation'),
    rotation: p('rotation'),
    offset: p('offset'),
    speedVariance: p('speedVariance'),
    seed: p('seed'),
})

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "BrickPattern",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Classic brick wall pattern with alternating rows and mortar gaps",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and this generator rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    // Per-node animated time driven by `speed`; the paint noun reads it as the row drift.
    animatedTime: { speed: 'speed' },
    props: {
        colorBrick: {
            default: "#000000",
            transform: transformColor,
            description: "Brick color",
            ui: { type: 'color', label: 'Brick Color', group: 'Colors' }
        },
        colorMortar: {
            default: "#ffffff",
            transform: transformColor,
            description: "Mortar / gap color",
            ui: { type: 'color', label: 'Mortar Color', group: 'Colors' }
        },
        cellsX: {
            default: 8,
            description: "Number of bricks per row",
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Bricks per Row', group: 'Effect' }
        },
        cellsY: {
            default: 10,
            description: "Number of brick rows",
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Rows', group: 'Effect' }
        },
        mortar: {
            default: 0.05,
            description: "Width of mortar gaps — equal pixel thickness in both directions",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Mortar Gap', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Softness of the brick edges (0 = crisp, 1 = very soft)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        variation: {
            default: 0,
            description: "Per-brick random lightening/darkening of the brick color",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Variation', group: 'Effect' }
        },
        rotation: {
            default: 0,
            transform: transformAngle,
            description: "Rotation of the pattern in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        speed: {
            default: 0,
            description: "Animation speed",
            ui: { type: 'range', min: -2, max: 2, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        offset: {
            default: 0,
            description: "Static horizontal offset — shifts the brick pattern without animating",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Offset', group: 'Animation' }
        },
        speedVariance: {
            default: 0,
            description: "How much each row's speed varies — at high values rows move at different speeds and directions",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Speed Variance', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: "Random seed for per-row speed variation",
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for interpolation',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        }
    },

    // The recipe: the brick-course field (fused with its own rotated frame — the frame's aspect
    // feeds the mortar thickness compensation), per-brick variation on the brick fill, bricks
    // stroked over the mortar in the chosen color space.
    paint: strokeOver({
        fill: p('colorMortar'),
        stroke: vary(p('colorBrick'), field.shade),
        mask: field.bricks,
        space: p('colorSpace'),
    })
})

export default componentDefinition
