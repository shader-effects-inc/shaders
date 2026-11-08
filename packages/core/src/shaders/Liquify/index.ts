import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {liquidDisplace} from "@coreroot/std/warps"
import {gridSim} from "@coreroot/std/sim/grids"
import {springLatticeField} from "@coreroot/std/effects/pointerFields"
import {edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    intensity: number
    stiffness: number
    damping: number
    radius: number
    edges: string
}

// std warp: one coordinate map drives both engine paths (the RTT fragment and the analytic
// UV fold), fed by this node's own compute-generated displacement field.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Liquify",
    role: 'warp',
    category: "Interactive",
    description: "Liquid-like interactive deformation effect",
    boundingBoxDeclaration: { aspectRatio: null },
    usesPointer: true,
    props: {
        intensity: {
            default: 10,
            description: 'Scale of the fabric displacement effect',
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        stiffness: {
            default: 3,
            description: 'Fabric rigidity (higher = stiffer canvas, lower = stretchy silk)',
            ui: { type: 'range', min: 1, max: 30, step: 0.5, label: 'Stiffness', group: 'Effect' }
        },
        damping: {
            default: 3,
            description: 'How quickly fabric motion settles',
            ui: { type: 'range', min: 0, max: 10, step: 0.1, label: 'Damping', group: 'Effect' }
        },
        radius: {
            default: 1,
            description: 'Cursor influence area',
            ui: { type: 'range', min: 0.1, max: 1.5, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        edges: edgesPropConfig('stretch', 'How to handle edges when distortion pushes content out of bounds')
    },

    // WebGPU compute: the spring-lattice field sim (a spring-mass cloth over ping-pong state
    // buffers, cursor-impulse driven, settle-gated), publishing the RG displacement texture the
    // map below samples. CHILD-INDEPENDENT — the sim also runs on the analytic fold path.
    ...gridSim(springLatticeField({
        stiffness: p('stiffness'), damping: p('damping'), radius: p('radius'),
        output: 'displacement',
    })),

    // One map drives both hooks: bend the incoming UV by this node's OWN compute-generated
    // displacement (the spring-lattice cloth field), then edge-handle. Same displacement-texture
    // shape as GridDistortion — see the `uvRemapIdentityWhen` note there for why the
    // absent-texture case is handled asymmetrically between the two hooks.
    map: liquidDisplace({intensity: p('intensity'), output: 'displacement'}),
    uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement,
})

export default componentDefinition
