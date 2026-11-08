import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {grayscale} from "@coreroot/std/effects/color"

export interface ComponentProps {}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Grayscale",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Convert colors to black and white",
    props: {},
    effect: grayscale(),
    missingChildMessage: 'You must pass a child component into the Grayscale shader.',
})

export default componentDefinition
