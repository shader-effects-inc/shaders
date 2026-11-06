import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {parallelogram} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    width: number | DimensionalValue | DimensionalString
    height: number | DimensionalValue | DimensionalString
    skew: number
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Parallelogram",
    description: "Parallelogram with adjustable width, height and skew",
    shape: {
        // iq's parallelogram: half-`width`, half-`height`, top-edge `skew`.
        distance: ({x, y}, u) => parallelogram(x, y, u('width'), u('height'), u('skew')),
        colorDescription: "Fill color of the parallelogram",
        centerDescription: "Center position of the parallelogram",
        bounds: {size: {width: 'width', height: 'height'}},
        shapeProps: {
            width: {
                default: 0.32,
                description: "Half-width of the parallelogram",
                ui: { type: 'range', min: 0.02, max: 1, step: 0.01, label: 'Width', group: 'Shape' }
            },
            height: {
                default: 0.22,
                description: "Half-height of the parallelogram",
                ui: { type: 'range', min: 0.02, max: 1, step: 0.01, label: 'Height', group: 'Shape' }
            },
            skew: {
                default: 0.15,
                description: "Horizontal skew of the top edge",
                ui: { type: 'range', min: -0.4, max: 0.4, step: 0.01, label: 'Skew', group: 'Shape' }
            }
        }
    },
})

export default componentDefinition
