import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {heart} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Heart",
    description: "Heart shape with adjustable size",
    shape: {
        // iq's heart; `radius` is the fit radius.
        distance: ({x, y}, u) => heart(x, y, u('radius')),
        colorDescription: "Fill color of the heart",
        centerDescription: "Center position of the heart",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.32,
                description: "Size of the heart (fit radius)",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Size', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
