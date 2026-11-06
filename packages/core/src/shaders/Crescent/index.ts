import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {crescent} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    innerRatio: number
    offset: number
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Crescent",
    description: "Crescent moon shape — an outer circle with an inner circle subtracted",
    shape: {
        // Outer circle `radius`; the bite circle is `innerRatio` of it, shifted by `offset`.
        distance: ({x, y}, u) => crescent(x, y, u('radius'), u('innerRatio'), u('offset')),
        colorDescription: "Fill color of the crescent",
        centerDescription: "Center position of the crescent",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.3,
                description: "Outer circle radius",
                ui: { type: 'range', min: 0.05, max: 1, step: 0.01, label: 'Size', group: 'Shape' }
            },
            innerRatio: {
                default: 0.8,
                description: "Inner (bite) circle radius as a fraction of outer radius",
                ui: { type: 'range', min: 0.3, max: 1.2, step: 0.01, label: 'Bite Size', group: 'Shape' }
            },
            offset: {
                default: 0.2,
                description: "Horizontal distance the bite circle is shifted from center",
                ui: { type: 'range', min: 0.01, max: 0.5, step: 0.01, label: 'Bite Offset', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
