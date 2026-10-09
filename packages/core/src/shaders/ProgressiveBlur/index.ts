import {defineStd, p, crosses} from "@coreroot/std"
import {progressiveBlur} from "@coreroot/std/effects/blurs"

import {transformAngle} from "@coreroot/utilities/transformations"
import {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    intensity: number
    angle: Parameters<typeof transformAngle>[0]
    center: Parameters<typeof transformPosition>[0]
    falloff: number
    dispersion: number
    jitter: number
}

export const componentDefinition = defineStd<ComponentProps>({
    name: "ProgressiveBlur",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Blurs",
    description: "Blur that increases progressively in one direction",
    requiresRTT: true,
    requiresChild: true,
    props: {
        intensity: {
            default: 50,
            description: 'Maximum intensity of the blur effect',
            ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Effect'}
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: 'Direction of the blur gradient (in degrees)',
            ui: {type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect'}
        },
        center: {
            default: {
                x: 0,
                y: 0.5
            },
            transform: transformPosition,
            description: 'Center point where blur begins',
            ui: {type: 'position', label: 'Center', group: 'Position'}
        },
        falloff: {
            default: 1,
            description: 'Distance over which blur transitions to full strength',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Falloff', group: 'Effect'}
        },
        dispersion: {
            default: 0,
            description: 'Blurs red and blue by different amounts for a lens-like color fringe that grows with the blur; negative swaps the colors',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Dispersion', group: 'Lens'}
        },
        jitter: {
            default: 0,
            description: 'Per-pixel noise on the tap spacing that hides banding at large blur radii',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Jitter', group: 'Lens'}
        }
    },

    // Compute-backed: variable-radius Gaussian whose radius map ramps directionally from `center`
    // along `angle`; a map driver on `intensity` samples the per-pixel max radius from the source.
    // The lens dials are off at 0; the chromatic kernel is only built while dispersion is set.
    ...progressiveBlur({
        intensity: p('intensity'), angle: p('angle'), center: p('center'), falloff: p('falloff'),
        detail: {dispersion: p('dispersion'), jitter: p('jitter')},
    }),
})

export default componentDefinition
