import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {ellipse} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radiusX: number | DimensionalValue | DimensionalString
    radiusY: number | DimensionalValue | DimensionalString
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Ellipse",
    description: "Ellipse with independently adjustable horizontal and vertical radii",
    shape: {
        distance: ({x, y}, u) => ellipse(x, y, u('radiusX'), u('radiusY')),
        colorDescription: "Fill color of the ellipse",
        centerDescription: "Center position of the ellipse",
        bounds: {size: {width: 'radiusX', height: 'radiusY'}},
        shapeProps: {
            radiusX: {
                default: 0.35,
                description: "Horizontal semi-axis radius",
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Width', group: 'Shape' }
            },
            radiusY: {
                default: 0.2,
                description: "Vertical semi-axis radius",
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Height', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
