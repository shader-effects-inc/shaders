import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isValue} from "@coreroot/std"
import {exposure} from "@coreroot/std/effects/color"

export interface ComponentProps {
    exposure: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Exposure",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Multiplicative exposure (gain) on the child. Unlike additive brightness, it scales values around black — blacks stay black while highlights can be pushed past 1.0 to feed HDR bloom and tone-mapping roll-off. Pointwise, so no extra render pass is needed.",
    props: {
        exposure: {
            default: 1,
            description: 'Brightness multiplier (gain). 1 = unchanged, >1 pushes highlights past white for HDR bloom, <1 darkens.',
            // Recompile only when gain crosses 1 (identity ↔ scale) — drives the identity bypass.
            recompile: crosses(1),
            ui: { type: ['range', 'map'], min: 0, max: 8, step: 0.1, label: 'Exposure', group: 'Adjustments' }
        }
    },
    effect: exposure(p('exposure')),
    // gain=1 is an exact no-op (rgb * 1.0).
    identityWhen: isValue('exposure', 1),
    missingChildMessage: 'You must pass a child component into the Exposure shader.',
})

export default componentDefinition
