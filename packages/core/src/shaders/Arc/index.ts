import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {mul} from "@coreroot/std/math"
import {arc, APERTURE_TO_HALF_ANGLE} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number | DimensionalValue | DimensionalString
    aperture: number
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Arc",
    description: "Pie sector (arc wedge) with adjustable radius and aperture angle",
    shape: {
        // `aperture` is in degrees; `arc` takes the half-aperture in radians (aperture · π/360).
        distance: ({x, y}, u) => arc(x, y, u('radius'), mul(u('aperture'), APERTURE_TO_HALF_ANGLE)),
        colorDescription: "Fill color of the arc",
        centerDescription: "Center position of the arc",
        bounds: {size: {width: 'radius', height: 'radius'}},
        shapeProps: {
            radius: {
                default: 0.38,
                description: "Radius of the pie sector",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Size', group: 'Shape' }
            },
            aperture: {
                default: 270,
                description: "Aperture angle of the wedge in degrees (360 = full circle)",
                ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Aperture', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
