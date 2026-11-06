import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {vesica} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    spread: number
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Vesica",
    description: "Vesica piscis (lens shape) formed by the intersection of two overlapping circles",
    shape: {
        // Circle `radius`, center separation `spread` (as a fraction of the radius).
        distance: ({x, y}, u) => vesica(x, y, u('radius'), u('spread')),
        colorDescription: "Fill color of the vesica",
        centerDescription: "Center position of the vesica",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.35,
                description: "Radius of the two overlapping circles",
                ui: { type: 'range', min: 0.05, max: 1, step: 0.01, label: 'Size', group: 'Shape' }
            },
            spread: {
                default: 0.5,
                description: "Circle separation — 0 = full circle overlap, 1 = infinitely thin lens",
                ui: { type: 'range', min: 0.05, max: 0.95, step: 0.01, label: 'Spread', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
