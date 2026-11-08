import {defineShader, p} from "@coreroot/std"
import {backToP3, overRgb, ribbons, tonePow} from "@coreroot/std/paint/noise"
import {transformPosition, transformColorSpace, colorSpaceOptions, transformBoolean} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

// Hard cap on strands; drives the Strand Count slider max.
const STRAND_MAX = 16

export interface ComponentProps {
    speed: number
    amplitude: number
    frequency: number
    lineCount: number
    lineWidth: number
    softness: number
    spread: number
    stops: ColorStop[] | null
    colorSpace: string
    colorScale: number
    colorVariance: number
    colorSpeed: number
    pinEdges: boolean
    start: Parameters<typeof transformPosition>[0]
    end: Parameters<typeof transformPosition>[0]
}

// std generator: flowing ribbons along the start → end path, colored from the multi-stop
// gradient in its working space. The ribbon field reads both clocks declared below — the main
// one drives the wave motion, the `color` extra clock scrolls the colors.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Strands",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Flowing ribbons of light with a multi-color gradient",
    acceptsUVContext: true,
    // Two independent clocks: `speed` drives the wave motion, `colorSpeed` scrolls the colors.
    animatedTime: { speed: 'speed' },
    extraAnimatedTimes: { color: 'colorSpeed' },
    props: {
        speed: {
            default: 0.5,
            description: 'Overall animation speed',
            ui: { type: 'range', min: 0.0, max: 2.0, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        amplitude: {
            default: 2,
            description: 'How far the strands wave away from their resting line',
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.1, label: 'Amplitude', group: 'Effect' }
        },
        frequency: {
            default: 0.3,
            description: 'How many waves run along each strand',
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.1, label: 'Frequency', group: 'Effect' }
        },
        lineCount: {
            default: 8,
            description: 'Number of strands',
            ui: { type: 'range', min: 1, max: STRAND_MAX, step: 1, label: 'Strand Count', group: 'Effect' }
        },
        lineWidth: {
            default: 0.05,
            description: 'Thickness of each strand’s solid core',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.05, label: 'Strand Width', group: 'Effect' }
        },
        softness: {
            default: 0.05,
            description: 'Softness of each strand’s edge (0 is crisp, higher is feathered)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.05, label: 'Strand Softness', group: 'Effect' }
        },
        spread: {
            default: 0.2,
            description: 'How far the strands fan out across the field',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.05, label: 'Spread', group: 'Effect' }
        },
        pinEdges: {
            default: false,
            transform: transformBoolean,
            description: 'Pin the strands so they converge to the start and end points',
            ui: { type: 'checkbox', label: 'Pin Edges', group: 'Effect' }
        },
        // Multi-stop gradient. Strands are colored by sampling this gradient; color flows along the
        // path over time and (via Color Variance) can differ from strand to strand.
        stops: {
            ...colorStopsPropConfig(),
            default: [
                { color: '#00e5ff', position: 0 },
                { color: '#5b8cff', position: 0.34 },
                { color: '#b06bff', position: 0.67 },
                { color: '#ff5fa2', position: 1 }
            ]
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space used to blend the gradient',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
        },
        colorScale: {
            default: 1,
            description: 'How many times the gradient repeats along the strands',
            ui: { type: ['range', 'map'], min: 1, max: 6, step: 1, label: 'Color Scale', group: 'Colors' }
        },
        colorVariance: {
            default: 1,
            description: 'At 0 every strand shares the same color; at 1 the strands spread across the full gradient',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.05, label: 'Color Variance', group: 'Colors' }
        },
        colorSpeed: {
            default: 1,
            description: 'Speed at which the colors flow along the strands (loops)',
            ui: { type: 'range', min: 0, max: 2, step: 0.05, label: 'Color Speed', group: 'Colors' }
        },
        start: {
            default: { x: 0, y: 0.5 },
            transform: transformPosition,
            description: 'Starting point of the strands',
            ui: { type: 'position', label: 'Start', group: 'Position' }
        },
        end: {
            default: { x: 1, y: 0.5 },
            transform: transformPosition,
            description: 'Ending point of the strands',
            ui: { type: 'position', label: 'End', group: 'Position' }
        }
    },

    // The ribbon accumulation runs in the working color space (an atomic runtime-count
    // reduce), closed by one back-conversion at the compile-time colorSpace and the tone lift.
    paint: overRgb(
        ribbons({
            from: p('start'),
            to: p('end'),
            count: p('lineCount'),
            width: p('lineWidth'),
            amplitude: p('amplitude'),
            frequency: p('frequency'),
            softness: p('softness'),
            spread: p('spread'),
            pinEdges: p('pinEdges'),
            colorScale: p('colorScale'),
            colorVariance: p('colorVariance'),
        }),
        backToP3({space: p('colorSpace')}),
        tonePow(),
    )
})

export default componentDefinition
