import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {linework} from "@coreroot/std/effects/stylize"
import {transformAngle, transformColor, transformPosition} from "@coreroot/utilities/transformations"

// Style selector: 0 = line (single plate), 1 = cross-hatch (3 plates), 2 = spiral.
const transformStyle = (value: string): number => (value === 'spiral' ? 2 : value === 'line' ? 0 : 1)

export interface ComponentProps {
    style: string
    frequency: number
    angle: Parameters<typeof transformAngle>[0]
    center: Parameters<typeof transformPosition>[0]
    relief: number
    waviness: number
    contrast: number
    inkColor: Parameters<typeof transformColor>[0]
    paperColor: Parameters<typeof transformColor>[0]
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Engraving",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Copper-plate line engraving — the image is redrawn as flowing line work whose weight swells with darkness, lines displaced by the form like a banknote portrait: a single plate, cross-hatched shadow plates, or one continuous spiral cut",
    // Stylization replaces the child: transparent output reveals the background, not the child.
    blendWithChildren: false,
    props: {
        style: {
            default: "crosshatch",
            transform: transformStyle,
            compileTime: true,
            description: "The engraving cut: a single line plate, cross-hatched plates building up the shadows, or one continuous spiral",
            ui: {
                type: 'select',
                options: [
                    { label: 'Line', value: 'line' },
                    { label: 'Cross-Hatch', value: 'crosshatch' },
                    { label: 'Spiral', value: 'spiral' }
                ],
                label: 'Style',
                group: 'Lines'
            }
        },
        frequency: {
            default: 90,
            description: "Line density — how many engraved lines span the height of the canvas",
            ui: { type: ['range', 'map'], min: 20, max: 300, step: 1, label: 'Frequency', group: 'Lines' }
        },
        angle: {
            default: 8,
            transform: transformAngle,
            description: "Direction of the base line work (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Lines', condition: { style: 'line' } }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Center the spiral coils outward from",
            ui: { type: 'position', label: 'Center', group: 'Lines', condition: { style: 'spiral' } }
        },
        relief: {
            default: 0.6,
            description: "How strongly the image brightness displaces the lines — the engraved 'lines climb over the form' effect",
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Relief', group: 'Lines' }
        },
        waviness: {
            default: 0.35,
            description: "Organic meander of the line work, like a hand-pulled burin stroke",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Waviness', group: 'Lines' }
        },
        contrast: {
            default: 1.15,
            description: "Tonal contrast applied before the lines are cut — higher pushes mid-tones toward pure line or pure paper",
            ui: { type: ['range', 'map'], min: 0.25, max: 3, step: 0.01, label: 'Contrast', group: 'Tone' }
        },
        inkColor: {
            default: "#1a1410",
            transform: transformColor,
            description: "Color of the engraved ink",
            ui: { type: 'color', label: 'Ink', group: 'Colors' }
        },
        paperColor: {
            default: "#f4eee2",
            transform: transformColor,
            description: "Paper color shown between the lines",
            ui: { type: 'color', label: 'Paper', group: 'Colors' }
        }
    },

    // The whole effect is the linework gather recipe: the compileTime `style` bakes exactly one
    // plate stack; tone, domain warp and relief live behind the noun.
    effect: linework({
        style: p('style'),
        frequency: p('frequency'),
        angle: p('angle'),
        center: p('center'),
        relief: p('relief'),
        waviness: p('waviness'),
        contrast: p('contrast'),
        ink: p('inkColor'),
        paper: p('paperColor'),
    }),
})

export default componentDefinition
