import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {gatherStack} from "@coreroot/std/effects/blurs"
import {chromaticFan} from "@coreroot/std/effects/lens"
import {transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    strength: number
    angle: Parameters<typeof transformAngle>[0]
    redOffset: number
    greenOffset: number
    blueOffset: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ChromaticAberration",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Separate RGB channels for a prismatic distortion effect",
    props: {
        strength: {
            default: 0.2,
            description: 'Overall strength of the chromatic aberration effect',
            ui: {
                type: ['range', 'map'],
                min: 0.0,
                max: 1.0,
                step: 0.01,
                label: 'Strength',
                group: 'Effect'
            }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: 'Direction of the chromatic aberration in degrees',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 360,
                step: 1,
                label: 'Angle',
                group: 'Effect'
            }
        },
        redOffset: {
            default: -1.0,
            description: 'Red channel offset multiplier',
            ui: {
                type: ['range', 'map'],
                min: -2.0,
                max: 2.0,
                step: 0.1,
                label: 'Red Offset',
                group: 'Effect'
            }
        },
        greenOffset: {
            default: 0.0,
            description: 'Green channel offset multiplier',
            ui: {
                type: ['range', 'map'],
                min: -2.0,
                max: 2.0,
                step: 0.1,
                label: 'Green Offset',
                group: 'Effect'
            }
        },
        blueOffset: {
            default: 1.0,
            description: 'Blue channel offset multiplier',
            ui: {
                type: ['range', 'map'],
                min: -2.0,
                max: 2.0,
                step: 0.1,
                label: 'Blue Offset',
                group: 'Effect'
            }
        }
    },

    // The chromatic fan, literally: three per-channel offset taps along `angle` recombined into
    // one color (alpha from the centred green sample — no 4th tap needed). The gather species
    // owns the RTT boundary and the premultiplied → straight unpremultiply tail.
    effect: gatherStack(chromaticFan({
        strength: p('strength'),
        angle: p('angle'),
        red: p('redOffset'),
        green: p('greenOffset'),
        blue: p('blueOffset'),
    }), []),
})

export default componentDefinition
