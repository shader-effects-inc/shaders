import {defineStd, p, crosses} from "@coreroot/std"
import {motionBlur, blurPath} from "@coreroot/std/effects/blurs"
import {transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    intensity: number
    angle: Parameters<typeof transformAngle>[0]
    dispersion: number
    falloff: number
    focus: number
    bias: number
    highlights: number
    jitter: number
}

export const componentDefinition = defineStd<ComponentProps>({
    name: "LinearBlur",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Blurs",
    description: "Directional motion blur in a specific angle",
    props: {
        intensity: {
            default: 30,
            description: 'Intensity of the linear blur effect',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Effect' }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: 'Direction of the linear blur (in degrees)',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        dispersion: {
            default: 0,
            description: 'Spectral color fringing along the streak, like light through a lens; negative flips which end is red',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Dispersion', group: 'Lens'}
        },
        falloff: {
            default: 0,
            description: 'Concentrates the dispersion away from the center; 0 fringes everywhere equally',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Falloff', group: 'Lens'}
        },
        focus: {
            default: 0,
            description: 'Radius of a clean zone around the center that stays sharp',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Focus', group: 'Lens'}
        },
        bias: {
            default: 0,
            description: 'Trails the streak off one side of each pixel like a comet; the sign picks the direction',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Bias', group: 'Streak'}
        },
        highlights: {
            default: 0,
            description: 'Lets bright pixels streak further than dark ones, like light rays',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Highlights', group: 'Streak'}
        },
        jitter: {
            default: 0,
            description: 'Per-pixel noise on the tap spacing that hides banding in long streaks',
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Jitter', group: 'Streak'}
        }
    },

    // Every detail dial is off at 0; the plain 32-tap gather is compiled until one is set. The
    // lens dials (falloff, focus) measure from the canvas center, since a linear streak has none.
    effect: motionBlur({
        path: blurPath.linear(p('angle')),
        amount: p('intensity'),
        detail: {
            dispersion: p('dispersion'), falloff: p('falloff'), focus: p('focus'),
            bias: p('bias'), highlights: p('highlights'), jitter: p('jitter'),
        },
    }),
})

export default componentDefinition
