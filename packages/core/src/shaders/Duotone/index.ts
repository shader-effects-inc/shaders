import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {duotone} from "@coreroot/std/effects/color"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    blend: number
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Duotone",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Map colors to two tones based on luminance",
    props: {
        colorA: {
            default: "#ff0000",
            transform: transformColor,
            description: 'First color (used for darker areas)',
            ui: {
                type: 'color',
                label: 'Color A',
                group: 'Colors'
            }
        },
        colorB: {
            default: "#023af4",
            transform: transformColor,
            description: 'Second color (used for brighter areas)',
            ui: {
                type: 'color',
                label: 'Color B',
                group: 'Colors'
            }
        },
        blend: {
            default: 0.5,
            description: 'Blend point between the two colors',
            ui: {
                type: ['range', 'map'],
                min: 0.0,
                max: 1.0,
                step: 0.1,
                label: 'Blend',
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
    effect: duotone({colorA: p('colorA'), colorB: p('colorB'), blend: p('blend'), colorSpace: p('colorSpace')}),
})

export default componentDefinition
