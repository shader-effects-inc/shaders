import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {reveal, coverage} from "@coreroot/std/effects/reveal"
import {transformPosition, transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    progress: number
    startAngle: number
    direction: string
    center: Parameters<typeof transformPosition>[0]
    feather: number
    invert: boolean
}

// A clock-hand sweep around `center`: the angular coverage wipes the growing wedge to transparent.
// `direction` is a compile-time cpu-only STRING (no transform — an inline transform would write
// the string into an f32 field); the angular coverage bakes it to a mode literal.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "RadialWipe",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Sweep the content away in a clock-hand arc around a center point",
    props: {
        progress: {
            default: 0.5,
            description: "How far the sweep has travelled (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        startAngle: {
            default: 0,
            description: "Angle in degrees where the sweep begins",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Start Angle', group: 'Transition' }
        },
        direction: {
            default: 'cw',
            compileTime: true,
            description: "Which way the sweep rotates",
            ui: {
                type: 'select',
                options: [
                    {label: 'Clockwise', value: 'cw'},
                    {label: 'Counter-Clockwise', value: 'ccw'},
                    {label: 'Both', value: 'both'}
                ],
                label: 'Direction',
                group: 'Transition'
            }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Point the sweep rotates around",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        feather: {
            default: 0.1,
            description: "Softness of the sweeping edge",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Feather', group: 'Transition' }
        },
        invert: {
            default: false,
            transform: transformBoolean,
            description: "Reverse which side of the sweep is wiped",
            ui: { type: 'checkbox', label: 'Invert', group: 'Transition' }
        }
    },
    effect: reveal({
        coverage: coverage.angular(p('center'), p('startAngle'), p('direction')),
        progress: p('progress'),
        feather: p('feather'),
        invert: p('invert'),
    }),
    missingChildMessage: 'You must pass a child component into the Radial Wipe shader.',
})

export default componentDefinition
