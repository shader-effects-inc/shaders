import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {teardrop} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    height: number | DimensionalValue | DimensionalString
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Teardrop",
    description: "Teardrop — a rounded bulb tapering to a sharp point",
    shape: {
        // Bulb `radius`, length `height` from bulb to point.
        distance: ({x, y}, u) => teardrop(x, y, u('radius'), u('height')),
        colorDescription: "Fill color of the teardrop",
        centerDescription: "Center position of the teardrop",
        bounds: {size: {width: 'radius', height: 'height'}},
        shapeProps: {
            radius: {
                default: 0.22,
                description: "Radius of the rounded bulb",
                ui: { type: 'range', min: 0.02, max: 0.6, step: 0.01, label: 'Width', group: 'Shape' }
            },
            height: {
                default: 0.4,
                description: "Length from the bulb to the point",
                ui: { type: 'range', min: 0.1, max: 0.7, step: 0.01, label: 'Length', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
