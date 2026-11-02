import Color from 'colorjs.io'
import type { DimensionalValue } from '../types'

/**
 * A single position-axis input: a plain UV number, a CSS-style keyword ('left'/'center'/…),
 * or a DimensionalValue carrying px/uv units (resolved to UV by the renderer before transform).
 */
export type PositionAxisInput = number | string | DimensionalValue

// Color space configuration for transformColor functions
// 'p3-linear': Uses Display P3 linear (default for backwards compatibility)
// 'srgb': Uses sRGB linear (matches design tools like Figma)
let colorSpaceMode: 'p3-linear' | 'srgb' = 'p3-linear'

export const setColorSpaceMode = (mode: 'p3-linear' | 'srgb') => {
    colorSpaceMode = mode
}

/**
 * Convert a true/false boolean into an integer 1/-1 representation
 * @param value
 */
export const transformBoolean = (value: boolean) => value? 1 : -1

/** A boolean as 0 (false) or 1 (true), for props a WGSL body reads as a number. */
export const transformBool01 = (value: unknown) => (value ? 1 : 0)

/**
 * Parse a CSS color string into finite RGBA components in the target color
 * space. Defensive against:
 *   - colorjs.io throwing on malformed input (returns transparent black)
 *   - "transparent" keyword and other CSS values whose RGB channels are NaN
 *     by spec — colorjs.io preserves the NaN through color-space conversion,
 *     and downstream `vec4(NaN, …)` poisons the shader graph: shader compile
 *     calls getTypeFromLength on a node that resolved to null and crashes
 *     with "Cannot read properties of null (reading '0')".
 * Returns finite numbers with NaN channels coerced to 0.
 */
function parseColorChannels(value: string): { r: number; g: number; b: number; a: number } {
    try {
        const threeColor = new Color(value)
        const targetSpace = colorSpaceMode === 'srgb' ? 'srgb-linear' : 'p3-linear'
        const coords = threeColor.to(targetSpace).coords
        // colorjs.io returns a BOXED Number object for functional notations (rgba()/hsl()/rgb(…/…)),
        // so a raw Number.isFinite(threeColor.alpha) is false → alpha silently collapsed to 0 (the
        // color rendered fully transparent). Number() unboxes it before the check; a genuinely
        // non-finite alpha defaults OPAQUE, not invisible.
        const alpha = Number(threeColor.alpha)
        return {
            r: Number.isFinite(coords?.[0]) ? coords[0] : 0,
            g: Number.isFinite(coords?.[1]) ? coords[1] : 0,
            b: Number.isFinite(coords?.[2]) ? coords[2] : 0,
            a: Number.isFinite(alpha) ? alpha : 1,
        }
    } catch {
        // Silently fall back. colorjs.io throws on a wide range of edge-case
        // inputs (the literal `0`, empty strings, partial state during prop
        // updates) — many of these are transient and not worth surfacing.
        return { r: 0, g: 0, b: 0, a: 0 }
    }
}

/**
 * Parse a CSS color string into finite RGBA channels in the active color space.
 *
 * This is a prop `transform` marker: a shader's color prop references it, and the GPU uniform
 * bridge swaps it for `transformColorGpu` (which builds the packed `d.vec4f`) by function
 * identity — so this body is never invoked to feed the renderer. It stays a real, distinct
 * function (the identity the bridge matches on) that returns the parsed channels for any direct
 * caller.
 * @param value
 * @returns `{ r, g, b, a }` finite channels (NaN coerced to 0)
 */
export const transformColor = (value: string) => parseColorChannels(value)

/**
 * Converts a position value into both a GPU node and raw data.
 * Accepts either an object {x: number, y: number} or a CSS-style string
 * like "top left", "center", "bottom right", etc.
 * Normalizes coordinates to a 0-1 range and inverts the y-axis
 * (so y=0 is top, y=1 is bottom, aligning with typical screen coordinates).
 *
 * Like `transformColor`, this is a prop `transform` marker matched by identity in the GPU
 * uniform bridge (which swaps in `transformPositionGpu`), so it is not called to feed the
 * renderer; it returns the parsed `{ x, y }` (Y-flipped) for any direct caller.
 *
 * @param value - The position value, either {x, y} or a string.
 * @returns `{ x, y }` in UV space (y inverted so 0 = top, 1 = bottom)
 */
export const transformPosition = (value: {x: PositionAxisInput, y: PositionAxisInput} | string) => {
    let x = 0.5
    let y = 0.5

    const parsePositionValue = (val: PositionAxisInput, isVertical: boolean = false): number => {
        if (typeof val === 'number') {
            return val
        }
        // Defensive: a DimensionalValue should be resolved to a plain number by the renderer
        // before reaching here; if not, fall back to its raw value (treated as UV).
        if (typeof val === 'object' && val !== null) {
            return val.value
        }

        const str = val.toLowerCase().trim()
        if (isVertical) {
            if (str === 'top') return 0
            if (str === 'bottom') return 1
            if (str === 'center') return 0.5
        } else {
            if (str === 'left') return 0
            if (str === 'right') return 1
            if (str === 'center') return 0.5
        }
        
        console.warn(`Invalid position value: ${val}. Defaulting to center.`)
        return 0.5
    }

    if (typeof value === 'string') {
        const parts = value.toLowerCase().trim().split(/\s+/);
        // Horizontal alignment
        if (parts.includes('left')) {
            x = 0
        } else if (parts.includes('right')) {
            x = 1
        } else if (parts.includes('center') || parts.length === 1 && (parts[0] === 'top' || parts[0] === 'bottom')) {
            // If only vertical is specified, horizontal is center
            x = 0.5
        }

        // Vertical alignment
        if (parts.includes('top')) {
            y = 0
        } else if (parts.includes('bottom')) {
            y = 1
        } else if (parts.includes('center') || parts.length === 1 && (parts[0] === 'left' || parts[0] === 'right')) {
            // If only horizontal is specified, vertical is center
            y = 0.5
        }
        // Handle a single "center" keyword
        if (parts.length === 1 && parts[0] === 'center') {
            x = 0.5
            y = 0.5
        }

    } else if (typeof value === 'object' && value !== null) {
        x = parsePositionValue(value.x, false)
        y = parsePositionValue(value.y, true)
    } else {
        console.warn(`Invalid position value provided: ${value}. Defaulting to center.`)
    }

    const finalY = 1.0 - y
    return { x, y: finalY }
}

/**
 * Transform angle values into normalized degrees (0-360).
 * Supports numerical angles with automatic wrapping and CSS-style gradient directions.
 * 
 * @param value - Angle value (number or CSS gradient direction string)
 * @returns Normalized angle in degrees (0-360)
 */
export const transformAngle = (value: number | string): number => {
    if (typeof value === 'number') {
        // Normalize numerical angles to 0-360 range
        return ((value % 360) + 360) % 360
    }

    const str = value.toLowerCase().trim()
    switch (str) {
        // Basic directions
        case 'to right':
            return 0 // left-to-right
        case 'to bottom':
            return 90 // top-to-bottom
        case 'to left':
            return 180 // right-to-left
        case 'to top':
            return 270 // bottom-to-top

        // Diagonal directions
        case 'to bottom right':
        case 'to right bottom':
            return 45
        case 'to bottom left':
        case 'to left bottom':
            return 135
        case 'to top left':
        case 'to left top':
            return 225
        case 'to top right':
        case 'to right top':
            return 315

        // "From" directions (opposite of "to")
        case 'from left':
            return 0
        case 'from top':
            return 90
        case 'from right':
            return 180
        case 'from bottom':
            return 270
        case 'from top left':
            return 45
        case 'from top right':
            return 135
        case 'from bottom right':
            return 225
        case 'from bottom left':
            return 315

        default:
            // Try to parse as a number with units
            const numMatch = str.match(/^(-?\d*\.?\d+)(deg|rad|turn)?$/)
            if (numMatch) {
                const [, valueStr, unit = 'deg'] = numMatch
                const numValue = parseFloat(valueStr)

                switch (unit) {
                    case 'deg':
                        return ((numValue % 360) + 360) % 360
                    case 'rad':
                        return (((numValue * 180 / Math.PI) % 360) + 360) % 360
                    case 'turn':
                        return (((numValue * 360) % 360) + 360) % 360
                    default:
                        return ((numValue % 360) + 360) % 360
                }
            }

            console.warn(`Invalid angle value: ${value}. Defaulting to 0 degrees.`)
            return 0
    }

}

/**
 * Transform edge handling mode strings into numeric values for shader use.
 * Controls how content behaves when distorted UVs fall outside the 0-1 range.
 *
 * @param value - Edge mode string ('stretch', 'transparent', 'mirror', or 'wrap')
 * @returns Numeric mode value (0 = stretch, 1 = transparent, 2 = mirror, 3 = wrap)
 */
export const transformEdges = (value: string): number => {
    const modes: Record<string, number> = {
        'stretch': 0,      // Clamp to edge (stretch edge pixels)
        'transparent': 1,  // Fade to transparent/alpha
        'mirror': 2,       // Reflect/mirror content at edges
        'wrap': 3          // Tile/repeat content
    }

    const mode = modes[value.toLowerCase()]
    if (mode === undefined) {
        console.warn(`Invalid edge mode: ${value}. Defaulting to 'stretch'.`)
        return 0
    }

    return mode
}

/**
 * Transform stroke-position mode strings into numeric values for shader use. Shared by every
 * analytic shape shader (Circle/Ellipse/Cross/Heart/…), which previously each defined an
 * identical local copy — a single shared reference lets the GPU bridge's `gpuTransformFor`
 * recognise it (it is a pure `string → number`, so the GPU-side transform IS this
 * function). Invalid input defaults to 'center' (1); the value always comes from a fixed select.
 *
 * @param value - Stroke position string ('outside', 'center', or 'inside')
 * @returns Numeric mode value (0 = outside, 1 = center, 2 = inside)
 */
export const transformStrokePosition = (value: string): number => {
    const modes: Record<string, number> = {'outside': 0, 'center': 1, 'inside': 2}
    return modes[value?.toLowerCase()] ?? 1
}

/**
 * Color space options for UI select dropdowns
 * Centralized configuration for color interpolation modes
 */
export const colorSpaceOptions = [
    {label: 'Linear RGB', value: 'linear'},
    {label: 'OKLCh', value: 'oklch'},
    {label: 'OKLAB', value: 'oklab'},
    {label: 'HSL', value: 'hsl'},
    {label: 'HSV', value: 'hsv'},
    {label: 'LCH', value: 'lch'}
]

/**
 * Transform color space interpolation mode strings into numeric values for shader use.
 * Controls how colors are mixed/interpolated in gradient effects.
 *
 * @param value - Color space string ('linear', 'oklch', 'oklab', 'hsl', 'hsv', or 'lch')
 * @returns Numeric mode value (0 = linear RGB, 1 = OKLCh, 2 = OKLAB, 3 = HSL, 4 = HSV, 5 = LCH)
 */
export const transformColorSpace = (value: string): number => {
    const modes: Record<string, number> = {
        'linear': 0,   // Linear RGB interpolation (default)
        'oklch': 1,    // OKLCh color space (perceptually uniform, vibrant)
        'oklab': 2,    // OKLAB color space (perceptually uniform)
        'hsl': 3,      // HSL color space (hue-saturation-lightness)
        'hsv': 4,      // HSV color space (hue-saturation-value)
        'lch': 5       // LCH color space (CIE Lab cylindrical)
    }

    const mode = modes[value.toLowerCase()]
    if (mode === undefined) {
        console.warn(`Invalid color space mode: ${value}. Defaulting to 'linear'.`)
        return 0
    }

    return mode
}