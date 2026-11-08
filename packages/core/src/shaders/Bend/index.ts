import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {bend} from "@coreroot/std/warps"
import {edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    strength: number
    falloff: number
    angle: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Bend",
    role: 'warp',
    category: "Distortions",
    description: "Bends the ends of the frame toward you like a curved display — content at the edges swells closer under real perspective, or curls away when the strength goes negative",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        strength: {
            default: 0.5,
            description: "How hard the ends bend — positive curls them toward you (edges magnify), negative curls them away (edges recede into the frame)",
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        falloff: {
            default: 0,
            description: "Concentrates the bend toward the ends — 0 curves the whole frame, higher keeps the middle flat and bends only near the edges",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Falloff', group: 'Effect' }
        },
        angle: {
            default: 0,
            description: "Direction of the bend axis in degrees — 0 bends the left/right ends, 90 bends the top/bottom",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        edges: edgesPropConfig('transparent', 'How to handle edges when the bend pulls content away from the frame border')
    },
    map: bend({strength: p('strength'), falloff: p('falloff'), angle: p('angle')}),
    missingChildMessage: 'You must pass a child component into the Bend shader.',
})

export default componentDefinition
