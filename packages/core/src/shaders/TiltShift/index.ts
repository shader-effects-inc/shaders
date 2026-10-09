import {defineStd, p, crosses} from "@coreroot/std"
import {tiltShift} from "@coreroot/std/effects/blurs"

import {transformAngle, transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    intensity: number
    width: number
    falloff: number
    angle: Parameters<typeof transformAngle>[0]
    center: Parameters<typeof transformPosition>[0]
    dispersion: number
    jitter: number
}

export const componentDefinition = defineStd<ComponentProps>({
    name: "TiltShift",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Blurs",
    description: "Selective focus blur mimicking tilt-shift photography",
    requiresRTT: true,
    requiresChild: true,
    props: {
        intensity: {
            default: 50,
            description: 'Maximum blur intensity at edges',
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Effect'}
        },
        width: {
            default: 0.3,
            description: 'Width of the sharp focus area',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Width', group: 'Effect'}
        },
        falloff: {
            default: 0.3,
            description: 'Distance over which blur transitions to full strength',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Falloff', group: 'Effect'}
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: 'Rotation angle of the focus line (in degrees)',
            ui: {type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect'}
        },
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: 'Center point of the focus line',
            ui: {type: 'position', label: 'Center', group: 'Position'}
        },
        dispersion: {
            default: 0,
            description: 'Blurs red and blue by different amounts for a lens-like color fringe outside the focus band; negative swaps the colors',
            recompile: crosses(0),
            ui: {type: 'range', min: -1, max: 1, step: 0.01, label: 'Dispersion', group: 'Lens'}
        },
        jitter: {
            default: 0,
            description: 'Per-pixel noise on the tap spacing that hides banding at large blur radii',
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Jitter', group: 'Lens'}
        }
    },

    // Compute-backed: variable-radius Gaussian banded around the focus line; the fragment
    // re-derives the same blur amount to keep in-focus pixels crisp at canvas resolution.
    // The lens dials are off at 0; the chromatic kernel is only built while dispersion is set.
    ...tiltShift({
        intensity: p('intensity'), width: p('width'), falloff: p('falloff'), angle: p('angle'), center: p('center'),
        detail: {dispersion: p('dispersion'), jitter: p('jitter')},
    }),
})

export default componentDefinition
