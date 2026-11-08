import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {isolines} from "@coreroot/std/effects/color"
import {transformColor, transformBoolean} from "@coreroot/utilities/transformations"

// Local compile-time enum transforms (referenced by the props block; applied by the bridge, so
// propValues.<prop> carries the mapped number).
const transformSource = (value: string): number => (value === 'alpha' ? 1 : 0)
const transformColorMode = (value: string): number => (value === 'custom' ? 1 : 0)

export interface ComponentProps {
    levels: number
    lineWidth: number
    softness: number
    gamma: number
    invert: boolean
    source: Parameters<typeof transformSource>[0]
    colorMode: Parameters<typeof transformColorMode>[0]
    lineColor: Parameters<typeof transformColor>[0]
    backgroundColor: Parameters<typeof transformColor>[0]
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ContourLines",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Draw topographical contour lines based on luminance or alpha",
    // Stylization: composite the line pattern OVER the child so transparent pixels reveal it
    // (composeReplaceFilter's blendWithChildren branch, leaf-sibling composition path).
    blendWithChildren: false, // This effect replaces its children rather than blending over them.
    props: {
        levels: {
            default: 5,
            description: "Number of contour levels",
            ui: { type: ['range', 'map'], min: 2, max: 30, step: 1, label: 'Levels', group: 'Effect' }
        },
        lineWidth: {
            default: 2,
            description: "Width of the contour lines in pixels",
            ui: { type: ['range', 'map'], min: 0.5, max: 5, step: 0.1, label: 'Line Width', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Edge softness of the lines (0 = sharp, 1 = soft)",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        gamma: {
            default: 0.5,
            description: "Contour distribution. <1 clusters in bright, >1 clusters in dark",
            ui: { type: ['range', 'map'], min: 0.1, max: 2, step: 0.01, label: 'Gamma', group: 'Effect' }
        },
        invert: {
            default: false,
            description: "Invert the source values",
            transform: transformBoolean,
            ui: { type: 'checkbox', label: 'Invert', group: 'Effect' }
        },
        source: {
            default: 'luminance',
            description: "Use luminance or alpha channel for contours",
            transform: transformSource,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    { label: 'Luminance', value: 'luminance' },
                    { label: 'Alpha', value: 'alpha' }
                ],
                label: 'Source',
                group: 'Effect'
            }
        },
        colorMode: {
            default: 'source',
            description: "Use source image colors or custom colors",
            transform: transformColorMode,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    { label: 'Source', value: 'source' },
                    { label: 'Custom', value: 'custom' }
                ],
                label: 'Color Mode',
                group: 'Colors'
            }
        },
        lineColor: {
            default: '#000000',
            description: "Color of the contour lines (custom mode)",
            transform: transformColor,
            ui: { type: 'color', label: 'Line Color', group: 'Colors', condition: { colorMode: 'custom' } }
        },
        backgroundColor: {
            default: 'transparent',
            description: "Background color (custom mode)",
            transform: transformColor,
            ui: { type: 'color', label: 'Background Color', group: 'Colors', condition: { colorMode: 'custom' } }
        }
    },
    // compileTime `source`/`colorMode` branch inside the noun (only the active path emits).
    effect: isolines({
        source: p('source'),
        levels: p('levels'),
        lineWidth: p('lineWidth'),
        softness: p('softness'),
        gamma: p('gamma'),
        invert: p('invert'),
        colorMode: p('colorMode'),
        lineColor: p('lineColor'),
        backgroundColor: p('backgroundColor'),
    }),
})

export default componentDefinition
