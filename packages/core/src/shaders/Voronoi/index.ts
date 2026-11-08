import {defineShader, p} from "@coreroot/std"
import {rampOver, share, stops} from "@coreroot/std/paint/fields"
import {borderOverlay, cellBorders, cellDistances, cellFill} from "@coreroot/std/paint/noise"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    colorBorder: Parameters<typeof transformColor>[0]
    scale: number
    speed: number
    seed: number
    edgeIntensity: number
    edgeSoftness: number
    colorSpace: string
}

// std generator: F1/F2 cellular-distance ratio → the two-color / multi-stop ramp, with separate
// boundary-line coloring. The cells are always fully random (no jitter control).

// The nearest-2 distances, shared: the fill gradient and the border mask read one evaluation.
const cells = share(cellDistances({scale: p('scale'), seed: p('seed')}))

export const componentDefinition = defineShader<ComponentProps>({
    name: "Voronoi",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Cellular pattern where each pixel is colored by its distance to the nearest of many scattered points",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and Voronoi rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    // Per-node animated time drifting the cell points.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#3186cf",
            transform: transformColor,
            description: "Color near each cell's center point",
            ui: { type: 'color', label: 'Color A (Center)', group: 'Colors' }
        },
        colorB: {
            default: "#fc02dd",
            transform: transformColor,
            description: "Color at cell boundaries, far from any center point",
            ui: { type: 'color', label: 'Color B (Edge)', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
        colorBorder: {
            default: "#000000",
            transform: transformColor,
            description: "Color of the cell boundary lines",
            ui: { type: 'color', label: 'Border Color', group: 'Colors' }
        },
        scale: {
            default: 6,
            description: "Number of cells across the canvas",
            ui: { type: ['range', 'map'], min: 1, max: 20, step: 0.5, label: 'Scale', group: 'Effect' }
        },
        speed: {
            default: 0.5,
            description: "Animation speed — how fast the cell points drift",
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: "Random seed — shifts the cell pattern without changing the overall structure",
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        edgeIntensity: {
            default: 0.5,
            description: "Controls how much of the cell interior is filled by the edge color. Low = center color dominates with a sharp boundary. High = edge color spreads further into the cell.",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Edge Intensity', group: 'Effect' }
        },
        edgeSoftness: {
            default: 0.05,
            description: "Width of the cell boundary lines.",
            ui: { type: ['range', 'map'], min: 0, max: 0.4, step: 0.005, label: 'Edge Softness', group: 'Effect' }
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

    // One shared nearest-2 cell evaluation feeds both the F1/F2 fill (through the stop ramp)
    // and the boundary-line mask, which overlays the border color on top. The border color
    // stays separate — it is not a gradient endpoint.
    paint: borderOverlay(
        rampOver(
            share(cellFill(cells, {edgeIntensity: p('edgeIntensity')})),
            stops(p('colorSpace')),
        ),
        cellBorders(cells, {softness: p('edgeSoftness'), scale: p('scale')}),
        {color: p('colorBorder')},
    )
})

export default componentDefinition
