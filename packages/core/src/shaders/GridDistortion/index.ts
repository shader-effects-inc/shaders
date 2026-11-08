import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, recompileWhen, p} from "@coreroot/std"
import {gridCellDisplace} from "@coreroot/std/warps"
import {gridSim} from "@coreroot/std/sim/grids"
import {pointerSplatField, clampSplatGridSize} from "@coreroot/std/effects/pointerFields"
import {edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    intensity: number
    decay: number
    radius: number
    gridSize: number
    edges: string
}

// std warp: one coordinate map drives both engine paths (the RTT fragment and the analytic
// UV fold), fed by this node's own compute-generated displacement field.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "GridDistortion",
    role: 'warp',
    category: "Interactive",
    description: "Interactive grid distortion controlled by mouse position",
    boundingBoxDeclaration: { aspectRatio: null },
    usesPointer: true,
    props: {
        intensity: {
            default: 1,
            description: 'Strength of the distortion effect',
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        decay: {
            default: 3,
            description: 'Rate of distortion decay (higher = faster)',
            ui: { type: 'range', min: 0, max: 10, step: 0.1, label: 'Decay', group: 'Effect' }
        },
        radius: {
            default: 1,
            description: 'Radius of the distortion effect',
            ui: { type: 'range', min: 0, max: 3, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        gridSize: {
            default: 20,
            // The compute grid resolution (buffer index stride, texture size, dispatch grid) is baked
            // per-compose from this value, so a change to the effective (clamped, floored) grid must
            // rebuild the compute. The recompile rule fires only when that integer changes — dragging
            // within one cell count is free.
            recompile: recompileWhen((prev, next) => clampSplatGridSize(prev as number) !== clampSplatGridSize(next as number)),
            description: 'Resolution of the distortion grid (higher = more detailed)',
            ui: { type: ['range', 'map'], min: 8, max: 128, step: 1, label: 'Grid Size', group: 'Effect' }
        },
        edges: edgesPropConfig('stretch', 'How to handle edges when distortion pushes content out of bounds')
    },

    // WebGPU compute: the pointer-splat field sim (a mouse-driven grid displacement field —
    // dissipate + Gaussian cursor-velocity splat + clamp, settle-gated), publishing the RG
    // displacement texture the map below samples. CHILD-INDEPENDENT. The grid resolution is
    // baked per-compose (recompile rule above).
    ...gridSim(pointerSplatField({
        gridSize: p('gridSize'), decay: p('decay'), intensity: p('intensity'), radius: p('radius'),
        output: 'displacement',
    })),

    // One map drives both hooks: snap the incoming UV to its grid cell, read this node's OWN
    // compute-generated displacement there, offset, then edge-handle.
    //
    // The displacement source is what makes this shader need the warp role's one asymmetric option.
    // When the compute texture is absent (GPU-free composition, or before the compute hook has run)
    // `uvRemap` must hand back the incoming `{uv, mask}` verbatim — hence `uvRemapIdentityWhen`. The
    // fragment path instead substitutes a zero displacement inside the map and keeps its RTT pass
    // structurally identical either way. Two correct answers to two different questions.
    map: gridCellDisplace({gridSize: p('gridSize'), output: 'displacement'}),
    uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement,
})

export default componentDefinition
