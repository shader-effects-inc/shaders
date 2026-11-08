import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {ring} from "@coreroot/std/shape"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    radius: number
    thickness: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader({
    role: 'shape',
    name: "Ring",
    description: "Annular ring (donut) with adjustable radius and band thickness",
    shape: {
        // Midline `radius`, half-band `thickness`.
        distance: ({x, y}, u) => ring(x, y, u('radius'), u('thickness')),
        // Rotationally symmetric: no rotation prop, and the overlay shows no rotation handle.
        rotatable: false,
        colorDescription: "Fill color of the ring",
        centerDescription: "Center position of the ring",
        // The box tracks the ring's OUTER edge (midline radius + half-band thickness), which no single
        // prop names — hence the custom bounds rather than a width/height binding.
        bounds: {
            custom: {
                propBindings: {
                    x: { prop: 'center', as: 'position-x' },
                    y: { prop: 'center', as: 'position-y' }
                },
                // Box tracks the outer edge of the ring band (midline radius + half-band thickness).
                computeBounds(props, cw, ch) {
                    const radius = (props.radius as number) ?? 0.5
                    const thickness = (props.thickness as number) ?? 0
                    // typeof guard: px-unit DimensionalValue axes baseline to 0.5, never NaN.
                    const cx = typeof props.center?.x === 'number' ? props.center.x : 0.5
                    const cy = typeof props.center?.y === 'number' ? props.center.y : 0.5
                    const diameter = (radius + thickness) * 2 * ch
                    return { centerXPx: cx * cw, centerYPx: cy * ch, widthPx: diameter, heightPx: diameter, rotationDeg: 0 }
                },
                // Resizing drives the outer radius; band thickness is preserved.
                writeBounds(bounds, currentProps, cw, ch) {
                    const thickness = (currentProps.thickness as number) ?? 0
                    const newOuter = Math.max(0, bounds.widthPx / (2 * ch))
                    return {
                        center: { ...(currentProps.center ?? { x: 0.5, y: 0.5 }), x: bounds.centerXPx / cw, y: bounds.centerYPx / ch },
                        radius: Math.max(0, newOuter - thickness)
                    }
                }
            }
        },
        shapeProps: {
            radius: {
                default: 0.3,
                description: "Distance from center to the ring's midline in UV space",
                ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Radius', group: 'Shape' }
            },
            thickness: {
                default: 0.07,
                description: "Half-width of the ring band — total ring width is twice this value",
                ui: { type: 'range', min: 0.005, max: 0.3, step: 0.005, label: 'Thickness', group: 'Shape' }
            }
        },
        stroke: {
            // A ring has two edges, and softness feathers both.
            softnessDescription: "Edge softness for antialiasing (applied to both inner and outer ring edges)",
            strokeThicknessMax: 0.1,
            strokePositionDescription: "Position of the stroke relative to the ring edge",
        },
    },
})

export default componentDefinition
