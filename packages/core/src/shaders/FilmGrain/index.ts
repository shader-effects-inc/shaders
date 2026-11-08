import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, crosses, isZero, schema} from "@coreroot/std"
import {filmGrain} from "@coreroot/std/effects/color"

export interface ComponentProps {
    strength: number
    bias: number
    animated: boolean
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "FilmGrain",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Analog film grain texture overlay, weighted toward darker areas",
    // `animTime` is the grain clock the `filmGrain` noun drives each frame (only when `animated`).
    extraFields: {
        animTime: { schema: schema.f32, initial: 0 }
    },
    props: {
        strength: {
            default: 0.5,
            description: 'Intensity of the film grain noise',
            // Recompile only when strength crosses 0 (identity ↔ grain) — drives the bypass below.
            recompile: crosses(0),
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.025, label: 'Strength', group: 'Effect' }
        },
        bias: {
            default: 2.0,
            description: 'Concentrates grain in darker areas. Higher values focus grain more heavily on shadows; 0 applies grain uniformly.',
            ui: { type: 'range', min: 0, max: 10, step: 0.1, label: 'Bias', group: 'Effect' }
        },
        animated: {
            default: false,
            description: 'When enabled, the grain pattern changes each frame for a dynamic film effect',
            ui: { type: 'checkbox', label: 'Animated', group: 'Effect' },
            transform: (value: boolean) => value ? 1.0 : 0.0
        }
    },
    // No `missingChildMessage`: FilmGrain fails silently (transparent) when childless.
    effect: filmGrain({strength: p('strength'), bias: p('bias'), animated: p('animated')}),
    // strength=0 zeroes the grain amplitude — skips the hash AND the grain clock entirely.
    identityWhen: isZero('strength'),
})

export default componentDefinition
