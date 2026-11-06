import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {mul} from "@coreroot/std/math"
import {circle} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0],
    radius: number | DimensionalValue | DimensionalString,
    softness: number | DimensionalValue | DimensionalString,
    center: Parameters<typeof transformPosition>[0],
    strokeThickness: number,
    strokeColor: Parameters<typeof transformColor>[0],
    strokePosition: string,
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Circle",
    description: "Generate a circle with adjustable size and softness",
    shape: {
        // `radius` names the full visual size; the edge sits at half of it.
        distance: ({x, y}, u) => circle(x, y, mul(u('radius'), 0.5)),
        // Rotationally symmetric: no rotation prop, and the overlay shows no rotation handle.
        rotatable: false,
        colorDescription: "The color of the circle",
        centerDescription: "The center point of the circle",
        centerLabel: 'Center Position',
        // `radius` names the FULL visual size here (the shader halves it), unlike the half-extent
        // size props on the rest of the fleet.
        bounds: {size: {width: 'radius', height: 'radius', as: 'canvas-height'}},
        shapeProps: {
            radius: {
                default: 1,
                description: "The radius of the circle. A value of one (1) is sets the circle to fit the canvas.",
                ui: {
                    type: ['range', 'map'],
                    min: 0,
                    max: 2,
                    step: 0.01,
                    label: 'Radius',
                    group: 'Effect',
                    units: ['%', 'px']
                }
            }
        },
        stroke: {
            strokeColorDescription: "The color of the stroke outline",
            strokePositionDescription: 'Position of the stroke relative to the circle edge',
        },
        colorSpaceDescription: 'Color space for blending fill and stroke colors in soft edges',
        // Circle's softness and stroke thickness are map-driveable with widened ranges (softness is a
        // canvas-height distance, not the fleet's 0–0.1 antialiasing nudge), so they replace the
        // fleet-standard pair rather than being generated from it.
        propOverrides: {
            softness: {
                default: 0,
                description: "Edge softness. Lower values like zero (0) are sharp, higher values like one (1) are softer.",
                ui: {
                    type: ['range', 'map'],
                    min: 0,
                    max: 1,
                    step: 0.01,
                    label: 'Softness',
                    group: 'Effect',
                    dimensional: 'canvas-height'
                }
            },
            strokeThickness: {
                default: 0,
                description: "The thickness of the stroke outline. Zero (0) means no stroke.",
                ui: {
                    type: ['range', 'map'],
                    min: 0,
                    max: 0.5,
                    step: 0.01,
                    label: 'Stroke Thickness',
                    group: 'Stroke'
                }
            },
        },
        // Circle predates the fleet-canonical ordering: radius/softness come before center. Prop order
        // is load-bearing (it drives the uniform struct layout and the settings-panel order), so it is
        // pinned rather than normalized.
        propOrder: ['origin', 'color', 'radius', 'softness', 'center', 'strokeThickness', 'strokeColor', 'strokePosition', 'colorSpace'],
    },
})

export default componentDefinition
