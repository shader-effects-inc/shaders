import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {barOffset} from "@coreroot/std/warps"
import {transformAngle} from "@coreroot/utilities/transformations"
import {edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    count: number
    angle: Parameters<typeof transformAngle>[0]
    intensity: number
    seed: number
    speed: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "BarShift",
    role: 'warp',
    category: "Distortions",
    description: "Slices content into parallel bars, each offset independently for a fractured or glitch-like effect",
    boundingBoxDeclaration: { aspectRatio: null },
    // Per-node animated time: each bar drifts at its own hash rate.
    animatedTime: { speed: 'speed' },
    props: {
        count: {
            default: 6,
            description: 'Number of bars across the longest viewport dimension',
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Count', group: 'Effect' }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: 'Angle of bar orientation in degrees (0 = vertical bars, 90 = horizontal bars)',
            ui: { type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Angle', group: 'Effect' }
        },
        intensity: {
            default: 0.15,
            description: 'Maximum displacement per bar',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: 'Randomization seed for per-bar offset variation',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        speed: {
            default: 0,
            description: 'Animation speed — each bar drifts at its own rate and direction',
            ui: { type: 'range', min: -2, max: 2, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        edges: edgesPropConfig('mirror', 'How to handle edges when distortion pushes content out of bounds')
    },
    map: barOffset({count: p('count'), angle: p('angle'), intensity: p('intensity'), seed: p('seed')}),
    missingChildMessage: 'You must pass a child component into the BarShift shader.',
})

export default componentDefinition
