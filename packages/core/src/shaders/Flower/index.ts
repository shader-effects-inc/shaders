import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {flower} from "@coreroot/std/shape"
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
    name: "Flower",
    description: "Petal shape with N lobes and adjustable inner-to-outer radius ratio",
    shape: {
        // Petal-tip `radius`, `sides` petals, valley radius = `innerRatio` of the tip radius.
        distance: ({x, y}, u) => flower(x, y, u('radius'), u('sides'), u('innerRatio')),
        colorDescription: "Fill color of the flower",
        centerDescription: "Center position of the flower",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.4,
                description: "Outer petal tip radius in UV space",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Radius', group: 'Shape' }
            },
            sides: {
                default: 5,
                description: "Number of petals",
                ui: { type: 'range', min: 3, max: 12, step: 1, label: 'Petals', group: 'Shape' }
            },
            innerRatio: {
                default: 0.4,
                description: "Inner valley radius as a ratio of outer radius — lower values make deeper notches",
                ui: { type: 'range', min: 0.1, max: 0.95, step: 0.01, label: 'Inner Ratio', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
