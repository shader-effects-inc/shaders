import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {flowNoise} from "@coreroot/std/warps"
import {transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    strength: number
    detail: number
    speed: number
    evolutionSpeed: number
    seed: number
    edges: Parameters<typeof transformEdges>[0]
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "FlowField",
    role: 'warp',
    category: "Distortions",
    description: "Fluid-like distortion with constant smooth motion",
    boundingBoxDeclaration: { aspectRatio: null },
    // Two independent animated clocks: the primary `speed` (flow drift) → `_animTime`, and
    // `evolutionSpeed` (pattern reshape) → `_animTime_evolution`. Both advance CPU-side each frame.
    animatedTime: { speed: 'speed' },
    extraAnimatedTimes: { evolution: 'evolutionSpeed' },
    props: {
        strength: {
            default: 0.15,
            description: "Intensity of the flow distortion",
            ui: { type: ['range', 'map'], min: 0, max: 0.5, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        detail: {
            default: 2,
            description: "Scale of the flow patterns",
            ui: { type: ['range', 'map'], min: 0.5, max: 5, step: 0.1, label: 'Detail', group: 'Effect' }
        },
        speed: {
            default: 0,
            description: "Speed of the flow",
            ui: { type: 'range', min: 0, max: 20, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        evolutionSpeed: {
            default: 0,
            description: "How fast the flow field pattern reshapes over time",
            ui: { type: 'range', min: 0, max: 20, step: 0.1, label: 'Evolution Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: "Random seed for flow pattern variation",
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        edges: {
            default: 'mirror',
            description: 'How to handle edges when distortion pushes content out of bounds',
            transform: transformEdges,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Transparent', value: 'transparent'},
                    {label: 'Mirror', value: 'mirror'},
                    {label: 'Wrap', value: 'wrap'}
                ],
                label: 'Edges',
                group: 'Effect'
            }
        }
    },
    map: flowNoise({strength: p('strength'), detail: p('detail'), seed: p('seed')}),
})

export default componentDefinition
