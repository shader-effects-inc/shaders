import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {wave} from "@coreroot/std/warps"
import {edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    strength: number
    frequency: number
    speed: number
    angle: number
    waveType: string
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "WaveDistortion",
    role: 'warp',
    category: "Distortions",
    description: "Wave-based distortion with multiple waveform types",
    boundingBoxDeclaration: { aspectRatio: null },
    // Per-node animated time (createAnimatedTime equivalent): the renderer registers `_animTime`
    // and advances it by `deltaTime * speed` (speed=0 pauses). The wave producer reads it.
    animatedTime: { speed: 'speed' },
    props: {
        strength: {
            default: 0.3,
            description: 'Distortion intensity',
            ui: {
                type: ['range', 'map'],
                min: 0.0,
                max: 1.0,
                step: 0.01,
                label: 'Strength',
                group: 'Effect'
            }
        },
        frequency: {
            default: 1,
            description: 'Number of bends/waves',
            ui: {
                type: ['range', 'map'],
                min: 0.1,
                max: 10.0,
                step: 0.1,
                label: 'Frequency',
                group: 'Effect'
            }
        },
        speed: {
            default: 1,
            description: 'Animation speed',
            ui: {
                type: 'range',
                min: 0,
                max: 5,
                step: 0.1,
                label: 'Speed',
                group: 'Animation'
            }
        },
        angle: {
            default: 0,
            description: 'Direction of wave distortion in degrees',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 360,
                step: 1,
                label: 'Angle',
                group: 'Effect'
            }
        },
        waveType: {
            default: 'sine',
            description: 'Shape of the distortion wave',
            compileTime: true,
            transform: (value: string) => {
                const types: Record<string, number> = {
                    'sine': 0,
                    'triangle': 1,
                    'square': 2,
                    'sawtooth': 3,
                    'bounce': 4
                }
                return types[value] ?? 0
            },
            ui: {
                type: 'select',
                options: [
                    {label: 'Sine', value: 'sine'},
                    {label: 'Triangle', value: 'triangle'},
                    {label: 'Square', value: 'square'},
                    {label: 'Sawtooth', value: 'sawtooth'},
                    {label: 'Bounce', value: 'bounce'}
                ],
                label: 'Wave Type',
                group: 'Effect'
            }
        },
        edges: edgesPropConfig('stretch', 'How to handle edges when distortion pushes content out of bounds')
    },

    // `waveType` is compileTime, so the producer branches once per composition and only the
    // active waveform body is emitted — no runtime branch on a compileTime prop.
    map: wave({strength: p('strength'), frequency: p('frequency'), angle: p('angle'), waveType: p('waveType')}),
    missingChildMessage: 'You must pass a child component into the WaveDistortion shader.',
})

export default componentDefinition
