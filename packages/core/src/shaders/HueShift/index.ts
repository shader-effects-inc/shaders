import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isZero} from "@coreroot/std"
import {hueRotate} from "@coreroot/std/effects/color"

export interface ComponentProps {
    shift: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "HueShift",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Rotate hue around the color wheel",
    props: {
        shift: {
            default: 0,
            description: 'The amount to shift the hue by',
            // Recompile only when shift crosses 0 (identity ↔ rotation). Drives the identity
            // bypass below. Operates on raw degrees (0 deg = 0 rad).
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Shift', group: 'Adjustments' }
        }
    },
    // `shift` carries RAW DEGREES; the deg→rad conversion lives in the body.
    effect: hueRotate(p('shift')),
    // At shift=0 the rotation matrix is identity (raw degrees; 0 deg ⇒ 0 rad).
    identityWhen: isZero('shift'),
    missingChildMessage: 'You must pass a child component into the Hue Shift shader.',
})

export default componentDefinition
