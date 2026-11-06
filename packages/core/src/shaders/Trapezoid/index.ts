import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {trapezoid} from "@coreroot/std/shape"
import {halfUvFromProp} from "@coreroot/utilities/halfUvProps"
import type {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"

export interface ComponentProps {
    origin: BoundingBoxOrigin
    color: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    bottomWidth: number | DimensionalValue | DimensionalString
    topWidth: number
    height: number | DimensionalValue | DimensionalString
    rotation: number
    softness: number
    strokeThickness: number
    strokeColor: Parameters<typeof transformColor>[0]
    strokePosition: string
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd({
    role: 'shape',
    name: "Trapezoid",
    description: "Trapezoid with adjustable top and bottom widths and height",
    shape: {
        // iq's trapezoid takes (r1, r2, he) = the half-widths at y<0 / y>0 and the half-height.
        // In screen space y grows downward, so r1=topWidth, r2=bottomWidth keeps the visual
        // labelling correct — SDF argument order, not prop declaration order.
        distance: ({x, y}, u) => trapezoid(x, y, u('topWidth'), u('bottomWidth'), u('height')),
        colorDescription: "Fill color of the trapezoid",
        centerDescription: "Center position of the trapezoid",
        bounds: {
            custom: {
                propBindings: {
                    x: { prop: 'center', as: 'position-x' },
                    y: { prop: 'center', as: 'position-y' },
                    rotation: { prop: 'rotation', as: 'degrees' }
                },

                // The visual width is the WIDER of the two edges — a width binding on bottomWidth alone
                // lies whenever topWidth > bottomWidth (box narrower than the shape). Both width props are
                // half-widths in canvas-height units (half-canvas-height: full px = value × 2 × ch); a px
                // DimensionalValue carries the full pixel size directly, which `halfUvFromProp` normalizes.
                computeBounds(props, cw, ch) {
                    const bottom = halfUvFromProp(props.bottomWidth, ch, 0.35, 'x', cw)
                    const top = typeof props.topWidth === 'number' ? props.topWidth : 0.2
                    const halfH = halfUvFromProp(props.height, ch, 0.25, 'y', cw)
                    const cx = typeof props.center?.x === 'number' ? props.center.x : 0.5
                    const cy = typeof props.center?.y === 'number' ? props.center.y : 0.5
                    return {
                        centerXPx: cx * cw,
                        centerYPx: cy * ch,
                        widthPx: Math.max(top, bottom) * 2 * ch,
                        heightPx: halfH * 2 * ch,
                        rotationDeg: (props.rotation as number) ?? 0
                    }
                },

                // W resize scales BOTH edge widths proportionally (the taper is the shape's identity —
                // stretching only one edge would mutate it); H drives height directly. px-unit values keep
                // their px object shape so absolute-sized shapes stay absolute.
                writeBounds(bounds, currentProps, cw, ch) {
                    const keepUnit = (cur: any, newHalfUV: number): any =>
                        cur != null && typeof cur === 'object' && cur.unit === 'px'
                            ? { value: Math.max(0, Math.round(newHalfUV * 2 * ch)), unit: 'px' }
                            : Math.max(0.001, newHalfUV)
                    const bottom = halfUvFromProp(currentProps.bottomWidth, ch, 0.35, 'x', cw)
                    const top = typeof currentProps.topWidth === 'number' ? currentProps.topWidth : 0.2
                    const oldMax = Math.max(top, bottom)
                    const newMax = ch > 0 ? bounds.widthPx / (2 * ch) : oldMax
                    const factor = oldMax > 0 ? newMax / oldMax : 1
                    // cw/ch are 0 until the canvas has been measured; dividing there would write NaN
                    // into center and lose the shape. Keep the current centre in that case.
                    const prevCenter = currentProps.center ?? { x: 0.5, y: 0.5 }
                    return {
                        center: {
                            ...prevCenter,
                            x: cw > 0 ? bounds.centerXPx / cw : (prevCenter.x ?? 0.5),
                            y: ch > 0 ? bounds.centerYPx / ch : (prevCenter.y ?? 0.5)
                        },
                        bottomWidth: keepUnit(currentProps.bottomWidth, bottom * factor),
                        topWidth: Math.max(0.001, top * factor),
                        height: keepUnit(currentProps.height, ch > 0 ? bounds.heightPx / (2 * ch) : halfUvFromProp(currentProps.height, ch, 0.25, 'y', cw)),
                        rotation: bounds.rotationDeg
                    }
                },

                // writeBounds absorbs W and H into independent props — resize is not aspect-locked.
                freeResize: true
            }
        },
        shapeProps: {
            bottomWidth: {
                default: 0.35,
                description: "Half-width of the bottom edge",
                // dimensional marker replaces the old width propBinding as the px→UV resolution source
                // (the box now measures via computeBounds, which doesn't feed the dimensional plan).
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Bottom Width', group: 'Shape', units: ['%', 'px'], dimensional: 'half-canvas-height', dimensionalAxis: 'x' }
            },
            topWidth: {
                default: 0.2,
                description: "Half-width of the top edge",
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Top Width', group: 'Shape' }
            },
            height: {
                default: 0.25,
                description: "Half-height of the trapezoid",
                ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Height', group: 'Shape', units: ['%', 'px'], dimensional: 'half-canvas-height', dimensionalAxis: 'y' }
            }
        }
    },
})

export default componentDefinition
