import {defineShader, p} from "@coreroot/std"
import {ditherInks, pixelGrid, quantise} from "@coreroot/std/paint/patterns"
import {transformColor} from "@coreroot/utilities/transformations"

// Local compile-time enum transforms (applied by the bridge → propValues carries the mapped number).
const transformPattern = (value: string): number => {
    const patterns: Record<string, number> = {bayer2: 0, bayer4: 1, bayer8: 2, clusteredDot: 3, blueNoise: 4, whiteNoise: 5, floydSteinberg: 6}
    return patterns[value] ?? 1
}
const transformColorMode = (value: string): number => (value === 'source' ? 1 : 0)

export interface ComponentProps {
    pattern: string
    pixelSize: number
    threshold: number
    spread: number
    colorMode: string
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
}

// The shared pixel-grid frame: the dither cell coordinate plus the child sampled once per cell,
// sized against the logical resolution so the dot count survives resolution scaling.
const grid = pixelGrid({size: p('pixelSize')})

// std gather filter: samples the child's render-to-texture per dither cell.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Dither",
    role: 'filter',
    species: 'gather',
    category: "Stylize",
    description: "Dithering effect with multiple pattern options",
    // Stylization: blendWithChildren is false, so the dithered result replaces the child —
    // transparent output shows the background rather than revealing the child beneath. Set it to
    // true if you want transparent dither output to reveal the child through it.
    blendWithChildren: false,
    boundingBoxDeclaration: {
        aspectRatio: null
    },
    props: {
        pattern: {
            default: "bayer4",
            transform: transformPattern,
            compileTime: true,
            description: "Dithering pattern algorithm",
            ui: {
                type: 'select',
                options: [
                    { label: 'Bayer 2x2', value: 'bayer2' },
                    { label: 'Bayer 4x4', value: 'bayer4' },
                    { label: 'Bayer 8x8', value: 'bayer8' },
                    { label: 'Clustered Dot', value: 'clusteredDot' },
                    { label: 'Blue Noise', value: 'blueNoise' },
                    { label: 'White Noise', value: 'whiteNoise' },
                    { label: 'Floyd-Steinberg', value: 'floydSteinberg' }
                ],
                label: 'Pattern',
                group: 'Effect'
            }
        },
        pixelSize: {
            default: 4,
            description: "Size of dithering pixels",
            ui: { type: 'range', min: 1, max: 20, step: 1, label: 'Pixel Size', group: 'Effect' }
        },
        threshold: {
            default: 0.5,
            description: "Luminance threshold for dithering",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Threshold', group: 'Effect' }
        },
        spread: {
            default: 1.0,
            description: "How much of the luminance range participates in dithering (lower = more solid areas)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Spread', group: 'Effect' }
        },
        colorMode: {
            default: "custom",
            transform: transformColorMode,
            compileTime: true,
            description: "How colors are determined",
            ui: {
                type: 'select',
                options: [
                    { label: 'Custom Colors', value: 'custom' },
                    { label: 'Source Colors', value: 'source' }
                ],
                label: 'Color Mode',
                group: 'Colors'
            }
        },
        colorA: {
            default: "transparent",
            transform: transformColor,
            description: "Dark color for dithering",
            ui: { type: 'color', label: 'Dark Color', condition: { colorMode: 'custom' }, group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Light color for dithering",
            ui: { type: 'color', label: 'Light Color', condition: { colorMode: 'custom' }, group: 'Colors' }
        }
    },

    // The recipe: grid → quantise (an ordered threshold field, or Floyd–Steinberg's serpentine
    // block diffusion, per the compile-time pattern) → ink the levels (custom two-color mix, or
    // the grid's own pixellated source color, per the compile-time color mode).
    effect: ditherInks({
        grid,
        levels: quantise({grid, pattern: p('pattern'), threshold: p('threshold'), spread: p('spread')}),
        mode: p('colorMode'),
        colors: [p('colorA'), p('colorB')],
    }),
})

export default componentDefinition
