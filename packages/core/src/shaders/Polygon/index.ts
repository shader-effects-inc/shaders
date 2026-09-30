import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {mix} from "@coreroot/std/math"
import {circle, polygon} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue
    sides: number
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
    name: "Polygon",
    description: "Regular polygon with adjustable sides and corner rounding",
    shape: {
        // The sharp regular-polygon field blended toward a circle by `rounding` (0 = polygon, 1 = circle).
        distance: ({x, y}, u) =>
            mix(polygon(x, y, u('radius'), u('sides')), circle(x, y, u('radius')), u('rounding')),
        colorDescription: "Fill color of the polygon",
        centerDescription: "Center position of the polygon",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.4,
                description: "Inradius — distance from the center to the middle of each side, in UV space (the vertices sit a little farther out)",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Radius', group: 'Shape' }
            },
            sides: {
                default: 6,
                description: "Number of sides (3 = triangle, 4 = square, 6 = hexagon, etc.)",
                ui: { type: 'range', min: 3, max: 12, step: 1, label: 'Sides', group: 'Shape' }
            },
            rounding: {
                default: 0,
                description: "Corner rounding — 0 is sharp vertices, 1 morphs into a circle",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Rounding', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
