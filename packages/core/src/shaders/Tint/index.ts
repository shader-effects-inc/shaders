import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {tint} from "@coreroot/std/effects/color"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    amount: number
    preserveLuminosity: boolean
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Tint",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Apply a color tint to the image",
    props: {
        color: {
            default: "#ff8800",
            transform: transformColor,
            description: 'Tint color',
            ui: { type: 'color', label: 'Tint Color', group: 'Colors' }
        },
        amount: {
            default: 0.5,
            description: 'Tint amount (0 = no tint, 1 = full tint)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Amount', group: 'Effect' }
        },
        preserveLuminosity: {
            default: true,
            compileTime: true,
            description: 'Preserve original brightness',
            transform: (v: boolean) => (v ? 1 : 0),
            ui: { type: 'checkbox', label: 'Preserve Luminosity', group: 'Effect' }
        }
    },
    // `preserveLuminosity` picks a compile-time body PAIR inside the noun — only the chosen
    // variant emits into WGSL.
    effect: tint({color: p('color'), amount: p('amount'), preserveLuminosity: p('preserveLuminosity')}),
    // NO identity bypass at amount=0: it would hold for the plain body, but the preserve body
    // (the DEFAULT) rescales by `originalLum / max(tintedLum, 1e-4)`, which is exactly 1 only once
    // luminance ≥ 1e-4 — below that it darkens. Not a provable no-op, so it stays unbypassed.
    missingChildMessage: 'You must pass a child component into the Tint shader.',
})

export default componentDefinition
