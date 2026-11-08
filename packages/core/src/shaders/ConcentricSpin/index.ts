import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {concentricRings} from "@coreroot/std/warps"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"
// Type-only: `center` accepts the same input shape `transformPosition` does (numbers or px strings).
import type {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    intensity: number
    rings: number
    smoothness: number
    seed: number
    speed: number
    speedRandomness: number
    edges: string
    center: Parameters<typeof transformPosition>[0]
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ConcentricSpin",
    role: 'warp',
    category: "Distortions",
    description: "Concentric rings that each rotate the underlying image by different amounts",
    boundingBoxDeclaration: { aspectRatio: null },
    // Per-node animated time: rings spin continuously by speed.
    animatedTime: { speed: 'speed' },
    props: {
        intensity: {
            default: 20,
            description: 'Maximum rotation angle per ring',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Effect' }
        },
        rings: {
            default: 8,
            description: 'Number of concentric rings',
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 1, label: 'Rings', group: 'Effect' }
        },
        smoothness: {
            default: 0.03,
            description: 'Softness of transitions between rings',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Smoothness', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: 'Randomization seed for per-ring rotation variation',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        speed: {
            default: 0.1,
            description: 'Speed of continuous ring rotation',
            ui: { type: 'range', min: -5, max: 5, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        speedRandomness: {
            default: 0.5,
            description: 'How much each ring varies in rotation speed and direction',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Speed Randomness', group: 'Animation' }
        },
        edges: edgesPropConfig('mirror', 'How to handle edges when distortion pushes content out of bounds'),
        center: centerPropConfig('Center point of the concentric rings', {label: 'Center Position'})
    },
    map: concentricRings({
        center: p('center'),
        intensity: p('intensity'),
        rings: p('rings'),
        smoothness: p('smoothness'),
        seed: p('seed'),
        speedRandomness: p('speedRandomness'),
    }),
    missingChildMessage: 'You must pass a child component into the ConcentricSpin shader.',
})

export default componentDefinition
