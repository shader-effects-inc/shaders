import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {perspective} from "@coreroot/std/warps"
import {transformPosition} from "@coreroot/utilities/transformations"
import {centerPropConfig, edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    pan: number
    tilt: number
    fov: number
    zoom: number
    offset: Parameters<typeof transformPosition>[0]
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Perspective",
    role: 'warp',
    category: "Distortions",
    description: "Rotate the plane in 3D space with pan and tilt",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        center: centerPropConfig("Center point of rotation"),
        pan: {
            default: 0,
            description: "Horizontal rotation (left/right)",
            ui: { type: ['range', 'map'], min: -90, max: 90, step: 0.1, label: 'Pan', group: 'Effect' }
        },
        tilt: {
            default: 0,
            description: "Vertical rotation (up/down)",
            ui: { type: ['range', 'map'], min: -90, max: 90, step: 0.1, label: 'Tilt', group: 'Effect' }
        },
        fov: {
            default: 60,
            description: "Field of view - controls perspective intensity",
            ui: { type: ['range', 'map'], min: 30, max: 120, step: 1, label: 'FOV', group: 'Effect' }
        },
        zoom: {
            default: 1,
            description: "Zoom in to fill the frame after rotation",
            ui: { type: ['range', 'map'], min: 0.5, max: 3, step: 0.1, label: 'Zoom', group: 'Effect' }
        },
        offset: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Shift the result in X/Y",
            ui: { type: 'position', label: 'Offset', group: 'Position' }
        },
        edges: edgesPropConfig('transparent', 'How to handle edges')
    },

    // No `missingChildMessage`: this shader fails silently on a missing child.
    map: perspective({
        center: p('center'),
        pan: p('pan'),
        tilt: p('tilt'),
        fov: p('fov'),
        zoom: p('zoom'),
        offset: p('offset'),
    }),
})

export default componentDefinition
