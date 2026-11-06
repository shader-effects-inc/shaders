import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {roundedRect} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    width: number | DimensionalValue | DimensionalString
    height: number | DimensionalValue | DimensionalString
    rounding: number | DimensionalValue | DimensionalString
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "RoundedRect",
    description: "Rounded rectangle with adjustable width, height, and corner rounding",
    shape: {
        // Half-`width`, half-`height`, corner `rounding`.
        distance: ({x, y}, u) => roundedRect(x, y, u('width'), u('height'), u('rounding')),
        colorDescription: "Fill color of the rectangle",
        centerDescription: "Center position of the rectangle",
        bounds: {size: {width: 'width', height: 'height'}},
        shapeProps: {
            width: {
                default: 0.35,
                description: "Half-width of the rectangle",
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Width', group: 'Shape' }
            },
            height: {
                default: 0.25,
                description: "Half-height of the rectangle",
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Height', group: 'Shape' }
            },
            rounding: {
                default: 0.05,
                description: "Corner rounding radius — set to min(width, height) for a pill shape",
                ui: { type: 'range', min: 0, max: 0.5, step: 0.01, label: 'Rounding', group: 'Shape', dimensional: 'canvas-height' }
            }
        }
    },
})

export default componentDefinition
