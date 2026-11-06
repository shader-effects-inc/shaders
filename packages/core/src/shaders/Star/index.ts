import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {star} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    sides: number
    innerRatio: number
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Star",
    description: "Classic star polygon with straight sides and sharp pointed tips",
    shape: {
        // Tip `radius`, `sides` points, inner-vertex radius = `innerRatio` of the tip radius.
        distance: ({x, y}, u) => star(x, y, u('radius'), u('sides'), u('innerRatio')),
        colorDescription: "Fill color of the star",
        centerDescription: "Center position of the star",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.4,
                description: "Outer tip radius — distance from center to the pointed tips",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Radius', group: 'Shape' }
            },
            sides: {
                default: 5,
                description: "Number of points on the star",
                ui: { type: 'range', min: 3, max: 12, step: 1, label: 'Points', group: 'Shape' }
            },
            innerRatio: {
                default: 0.4,
                description: "Inner vertex radius as a ratio of outer radius (0.382 = golden-ratio 5-star)",
                ui: { type: 'range', min: 0.1, max: 0.9, step: 0.01, label: 'Inner Ratio', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
