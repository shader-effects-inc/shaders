import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformPosition, transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    center: Parameters<typeof transformPosition>[0]
    rings: number
    feather: number
    invert: boolean
}

// IrisWipe's radial coverage banded into `rings` concentric rings: whole rings vanish one after
// another from the center outward, pulsing like rain rings instead of sweeping smoothly. `feather`
// cross-fades adjacent rings' timing (wide feather ≈ a soft radial sweep, 0 = discrete ring pops).
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "RippleWipe",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Wipe the content away in concentric rings pulsing out from a center point",
    props: {
        progress: {
            default: 0.5,
            description: "How far the ripple has expanded (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Point the rings expand from",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        rings: {
            default: 8,
            description: "Number of concentric rings",
            ui: { type: ['range', 'map'], min: 2, max: 40, step: 1, label: 'Rings', group: 'Transition' }
        },
        feather: {
            default: 0.2,
            description: "How softly adjacent rings' timing overlaps (0 = discrete ring pops)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Wipe from the outside inward instead",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.radial(p('center')).bands(p('rings')),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Ripple Wipe shader.',
})

export default componentDefinition
