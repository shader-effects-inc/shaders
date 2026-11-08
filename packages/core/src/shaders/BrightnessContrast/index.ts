import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isZero, allOf} from "@coreroot/std"
import {brightnessContrast} from "@coreroot/std/effects/color"

export interface ComponentProps {
    brightness: number
    contrast: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "BrightnessContrast",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Adjustments",
    description: "Adjust brightness and contrast of the image",
    props: {
        brightness: {
            default: 0,
            description: 'Brightness adjustment (-1 to 1)',
            // Recompile only when brightness crosses 0 — drives the identity bypass below.
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Brightness', group: 'Adjustments' }
        },
        contrast: {
            default: 0,
            description: 'Contrast adjustment (-1 to 1)',
            // The `+ 1` value transform is folded into the GPU body, so the identity value here is
            // the RAW 0, not 1.
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Contrast', group: 'Adjustments' }
        }
    },
    // Both uniforms carry the RAW slider values; the body folds contrast's `+ 1`.
    effect: brightnessContrast({brightness: p('brightness'), contrast: p('contrast')}),
    // brightness=0 AND raw contrast=0 (→ contrast factor 1) is an exact no-op: (rgb-0.5)*1+0.5+0.
    // Both defaults are 0, so a freshly-added BrightnessContrast is free until it is adjusted.
    identityWhen: allOf(isZero('brightness'), isZero('contrast')),
    missingChildMessage: 'You must pass a child component into the Brightness Contrast shader.',
})

export default componentDefinition
