import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, simulate, op, pointer, pointerSpeed, displaceBy} from "@coreroot/std"
import {transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    intensity: number
    decay: number
    radius: number
    chromaticSplit: number
    edges: string
}

// std gather filter driven by a declared simulation: a damped wave equation over
// a 128² height field (2-history ping-pong — engine-owned), stirred by the pointer (velocity
// tracked, teleport-guarded), its gradient derived as an RG vector field that displaces the child
// with a chromatic R/B split. Rest semantics are declared: the engine derives the settle window
// from the damping value and stops dispatching once the field has decayed. GPU bodies live in the
// kit behind the nouns: `waves.buildWaveFieldKernels` (propagate + gradient) and
// `displace.chromaticDisplaceUVs`.
const waves = simulate.grid({
    resolution: 128,
    history: 2, // the wave equation reads t−1 AND t−2
    step: [
        op.wave({ damping: p('decay') }),
        op.splat({
            at: pointer({ teleportGuard: true }),
            amount: pointerSpeed({ max: 2 }),
            radius: p('radius'),
        }),
    ],
    derive: { displacement: op.gradient() },
    rest: { settlesWhen: 'derived-from-damping' },
})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "CursorRipples",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Fluid-like ripple distortion",
    props: {
        intensity: {
            default: 10,
            description: 'Strength of the ripple distortion',
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        decay: {
            default: 10,
            description: 'How quickly ripples fade (higher = faster)',
            ui: { type: 'range', min: 0, max: 20, step: 0.1, label: 'Decay', group: 'Effect' }
        },
        radius: {
            default: 0.5,
            description: 'Radius of cursor influence',
            ui: { type: 'range', min: 0.1, max: 1, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        chromaticSplit: {
            default: 1,
            description: 'RGB channel separation along ripple edges',
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.1, label: 'Chromatic Split', group: 'Effect' }
        },
        edges: {
            default: 'stretch',
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
    effect: displaceBy(waves.output('displacement'), {
        strength: p('intensity'),
        chromatic: p('chromaticSplit'),
        edges: p('edges'),
    }),
})

export default componentDefinition
