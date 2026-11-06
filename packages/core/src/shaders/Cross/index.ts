import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {cross} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    thickness: number
    rounding: number
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Cross",
    description: "Plus / cross shape with adjustable arm length, width, and rounding",
    shape: {
        // Arm half-length `radius`, arm half-width `thickness`, corner `rounding`.
        distance: ({x, y}, u) => cross(x, y, u('radius'), u('thickness'), u('rounding')),
        colorDescription: "Fill color of the cross",
        centerDescription: "Center position of the cross",
        rotationDescription: "Rotation in degrees (45° turns a plus into an ×)",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.35,
                description: "Arm half-length — distance from center to the end of each arm",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Arm Length', group: 'Shape' }
            },
            thickness: {
                default: 0.08,
                description: "Arm half-width — controls how wide each arm is",
                ui: { type: 'range', min: 0.01, max: 0.5, step: 0.005, label: 'Arm Width', group: 'Shape' }
            },
            rounding: {
                default: 0,
                description: "Corner rounding — rounds the arm ends and concave corners",
                ui: { type: 'range', min: 0, max: 0.2, step: 0.005, label: 'Rounding', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
