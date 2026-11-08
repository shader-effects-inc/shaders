import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {solidColor} from "@coreroot/std/paint/gradients"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    color: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "SolidColor",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Textures",
    description: "Fill the canvas with a single solid color",
    acceptsUVContext: true,
    props: {
        color: {
            default: "#5b18ca",
            transform: transformColor,
            description: 'The solid color to display',
            ui: { type: 'color', label: 'Color', group: 'Colors' }
        }
    },
    paint: solidColor(p('color'))
})

export default componentDefinition
