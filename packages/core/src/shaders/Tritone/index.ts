import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {tritone} from "@coreroot/std/effects/color"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorC: Parameters<typeof transformColor>[0]
    blendMid: number
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Tritone",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Map colors to three tones: shadows, midtones, highlights",
    props: {
        colorA: {
            default: "#ce1bea",
            transform: transformColor,
            description: 'First color (used for shadows/darkest areas)',
            ui: {
                type: 'color',
                label: 'Color A (Shadows)',
                group: 'Colors'
            }
        },
        colorB: {
            default: "#2fff00",
            transform: transformColor,
            description: 'Second color (used for midtones)',
            ui: {
                type: 'color',
                label: 'Color B (Midtones)',
                group: 'Colors'
            }
        },
        colorC: {
            default: "#ffff00",
            transform: transformColor,
            description: 'Third color (used for highlights/brightest areas)',
            ui: {
                type: 'color',
                label: 'Color C (Highlights)',
                group: 'Colors'
            }
        },
        blendMid: {
            default: 0.5,
            description: 'Midpoint position between the three colors',
            ui: {
                type: ['range', 'map'],
                min: 0.0,
                max: 1.0,
                step: 0.1,
                label: 'Midpoint',
                group: 'Effect'
            }
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
    // No `missingChildMessage`: the tone filters fail SILENTLY (transparent), by design.
    effect: tritone({colorA: p('colorA'), colorB: p('colorB'), colorC: p('colorC'), blendMid: p('blendMid'), colorSpace: p('colorSpace')}),
})

export default componentDefinition
