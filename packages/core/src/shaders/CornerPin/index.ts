import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {cornerPin} from "@coreroot/std/warps"
import {transformPosition} from "@coreroot/utilities/transformations"
import {edgesPropConfig} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    topLeft: Parameters<typeof transformPosition>[0]
    topRight: Parameters<typeof transformPosition>[0]
    bottomLeft: Parameters<typeof transformPosition>[0]
    bottomRight: Parameters<typeof transformPosition>[0]
    amount: number
    edges: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "CornerPin",
    role: 'warp',
    category: "Distortions",
    description: "Pin each corner of the content to an arbitrary position for a free perspective warp",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        topLeft: {
            default: { x: 0, y: 0 },
            transform: transformPosition,
            description: "Position of the top-left corner",
            ui: { type: 'position', label: 'Top Left', group: 'Corners' }
        },
        topRight: {
            default: { x: 1, y: 0 },
            transform: transformPosition,
            description: "Position of the top-right corner",
            ui: { type: 'position', label: 'Top Right', group: 'Corners' }
        },
        bottomLeft: {
            default: { x: 0, y: 1 },
            transform: transformPosition,
            description: "Position of the bottom-left corner",
            ui: { type: 'position', label: 'Bottom Left', group: 'Corners' }
        },
        bottomRight: {
            default: { x: 1, y: 1 },
            transform: transformPosition,
            description: "Position of the bottom-right corner",
            ui: { type: 'position', label: 'Bottom Right', group: 'Corners' }
        },
        amount: {
            default: 1,
            description: "Blends the warp in and out — 0 returns the content to its original rectangle, 1 fully pins the corners",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Amount', group: 'Effect' }
        },
        edges: edgesPropConfig('transparent', 'How to handle areas outside the pinned quad')
    },

    // Silent on a missing child, as this shader has always been.
    map: cornerPin({
        topLeft: p('topLeft'),
        topRight: p('topRight'),
        bottomRight: p('bottomRight'),
        bottomLeft: p('bottomLeft'),
        amount: p('amount'),
    }),
})

export default componentDefinition
