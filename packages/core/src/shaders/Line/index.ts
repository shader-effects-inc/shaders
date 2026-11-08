import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {strokedSegment} from "@coreroot/std/paint/figures"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

// Style/cap selects stored as runtime uniforms (like Circle's strokePosition) — a few
// std.select branches per pixel is cheaper than a recompile on every option change.
const LINE_STYLE_MODES: Record<string, number> = {solid: 0, dashed: 1, dotted: 2}
export const transformLineStyle = (value: string): number => LINE_STYLE_MODES[value] ?? 0

const LINE_CAP_MODES: Record<string, number> = {square: 0, rounded: 1}
export const transformLineCap = (value: string): number => LINE_CAP_MODES[value] ?? 1

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    pointA: Parameters<typeof transformPosition>[0]
    pointB: Parameters<typeof transformPosition>[0]
    thickness: number
    style: string
    dashLength: number
    gapLength: number
    capStart: string
    capEnd: string
}

const CAP_OPTIONS = [
    {label: 'Square', value: 'square'},
    {label: 'Rounded', value: 'rounded'}
]

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Line",
    role: 'generator',
    boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
    category: "Shapes",
    description: "Draw a straight line between two points with color, thickness, and solid, dashed, or dotted styles",
    acceptsUVContext: true,
    props: {
        color: {
            default: "#ffffff",
            transform: transformColor,
            description: "The color of the line",
            ui: {
                type: 'color',
                label: 'Color',
                group: 'Colors'
            }
        },
        pointA: {
            default: {
                x: 0.2,
                y: 0.5
            },
            transform: transformPosition,
            description: "The start point of the line",
            ui: {
                type: 'position',
                label: 'Point A',
                group: 'Position',
                units: ['%', 'px']
            }
        },
        pointB: {
            default: {
                x: 0.8,
                y: 0.5
            },
            transform: transformPosition,
            description: "The end point of the line",
            ui: {
                type: 'position',
                label: 'Point B',
                group: 'Position',
                units: ['%', 'px']
            }
        },
        thickness: {
            default: 0.01,
            description: "The thickness of the line. A value of one (1) matches the canvas height.",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 0.25,
                step: 0.001,
                label: 'Thickness',
                group: 'Effect',
                dimensional: 'canvas-height'
            }
        },
        style: {
            default: 'solid',
            transform: transformLineStyle,
            description: "The line style: solid, dashed, or dotted. Dashes and dots are spaced to land exactly on both endpoints.",
            ui: {
                type: 'select',
                options: [
                    {label: 'Solid', value: 'solid'},
                    {label: 'Dashed', value: 'dashed'},
                    {label: 'Dotted', value: 'dotted'}
                ],
                label: 'Style',
                group: 'Style'
            }
        },
        dashLength: {
            default: 0.05,
            description: "The length of each dash, including its caps",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 0.5,
                step: 0.001,
                label: 'Dash Length',
                group: 'Style',
                dimensional: 'canvas-height',
                condition: {style: 'dashed'}
            }
        },
        gapLength: {
            default: 0.025,
            description: "The gap between dashes, or the spacing between dots",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 0.5,
                step: 0.001,
                label: 'Gap Length',
                group: 'Style',
                dimensional: 'canvas-height',
                condition: {style: ['dashed', 'dotted']}
            }
        },
        capStart: {
            default: 'rounded',
            transform: transformLineCap,
            description: "Cap shape at the start of the line (point A). Rounded caps center on the endpoint; square caps end flat exactly at it. Also shapes the A-facing end of each dash and dot.",
            ui: {
                type: 'select',
                options: CAP_OPTIONS,
                label: 'Start Cap',
                group: 'Style'
            }
        },
        capEnd: {
            default: 'rounded',
            transform: transformLineCap,
            description: "Cap shape at the end of the line (point B). Rounded caps center on the endpoint; square caps end flat exactly at it. Also shapes the B-facing end of each dash and dot.",
            ui: {
                type: 'select',
                options: CAP_OPTIONS,
                label: 'End Cap',
                group: 'Style'
            }
        }
    },
    paint: strokedSegment({
        from: p('pointA'),
        to: p('pointB'),
        width: p('thickness'),
        style: p('style'),
        dashLength: p('dashLength'),
        gapLength: p('gapLength'),
        capStart: p('capStart'),
        capEnd: p('capEnd'),
        color: p('color'),
    })
})

export default componentDefinition
