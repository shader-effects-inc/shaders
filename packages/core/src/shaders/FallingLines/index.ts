import {defineShader, p} from "@coreroot/std"
import {alphaOf, fallingStreaks, strokeOver, times, withAlpha} from "@coreroot/std/paint/patterns"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorSpace: string
    angle: number
    speed: number
    speedVariance: number
    density: number
    trailLength: number
    balance: number
    strokeWidth: number
    rounding: number
}

// The streak field, driven by the node's accumulated animation time: `mask` is the streak
// coverage, `fade` the lead→trail position within a streak.
const streaks = fallingStreaks({
    angle: p('angle'),
    density: p('density'),
    speedVariance: p('speedVariance'),
    trailLength: p('trailLength'),
    strokeWidth: p('strokeWidth'),
    rounding: p('rounding'),
    balance: p('balance'),
})

// Trail color fading toward the lead color along each streak, in the chosen color space.
const streakColor = strokeOver({
    fill: p('colorB'),
    stroke: p('colorA'),
    mask: streaks.fade,
    space: p('colorSpace'),
})

export const componentDefinition = defineShader<ComponentProps>({
    name: "FallingLines",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Directional falling lines with a leading-to-trailing color fade",
    acceptsUVContext: true,
    // Per-node accumulated time driven by the `speed` prop; the paint noun reads it as the drift.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#ffffff",
            transform: transformColor,
            description: "Color at the leading edge of each line",
            ui: { type: 'color', label: 'Lead Color', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff00",
            transform: transformColor,
            description: "Color at the trailing edge (transparent by default)",
            ui: { type: 'color', label: 'Trail Color', group: 'Colors' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for interpolation between lead and trail colors',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        },
        angle: {
            default: 90,
            description: "Direction of movement in degrees (90=down, 270=up, 0=right, 180=left)",
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        speed: {
            default: 0.5,
            description: "Movement speed",
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        speedVariance: {
            default: 0.3,
            description: "Per-line speed variance (0=uniform, 1=high variance)",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Speed Variance', group: 'Animation' }
        },
        density: {
            default: 15,
            description: "Number of line columns across the canvas",
            ui: { type: 'range', min: 1, max: 60, step: 1, label: 'Density', group: 'Effect' }
        },
        trailLength: {
            default: 0.35,
            description: "Streak length relative to spacing (0=point, 1=continuous)",
            ui: { type: ['range', 'map'], min: 0.01, max: 1, step: 0.01, label: 'Trail Length', group: 'Effect' }
        },
        balance: {
            default: 0.5,
            description: "Color mix midpoint (0.5=linear, 0=all trailing/colorB, 1=all leading/colorA)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Balance', group: 'Colors' }
        },
        strokeWidth: {
            default: 0.15,
            description: "Line thickness as fraction of column width (0=hairline, 1=full width)",
            ui: { type: ['range', 'map'], min: 0.02, max: 1, step: 0.02, label: 'Stroke Width', group: 'Effect' }
        },
        rounding: {
            default: 1,
            description: "Rounds the leading edge (0=flat/square, 1=fully rounded cap)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Rounding', group: 'Effect' }
        }
    },

    // The recipe: the streak coverage masks the faded color's alpha.
    paint: withAlpha(streakColor, times(alphaOf(streakColor), streaks.mask))
})

export default componentDefinition
