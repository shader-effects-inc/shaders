import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, dist, pair} from "@coreroot/std/paint/fields"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    strokeWidth: number,
    strokeFalloff: number,
    softness: number,
    speed: number,
    center: Parameters<typeof transformPosition>[0],
    scale: number,
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Spiral",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Rotating spiral pattern with animated movement",
    acceptsUVContext: true,
    // Per-node animated time: the renderer registers `_animTime` and advances it by
    // `deltaTime * speed` (speed=0 pauses, negative reverses).
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: "#000000",
            transform: transformColor,
            description: "Background color",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#ffffff",
            transform: transformColor,
            description: "Spiral stroke color",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        strokeWidth: {
            default: 0.5,
            description: "Thickness of spiral stroke",
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Stroke Width', group: 'Effect' }
        },
        strokeFalloff: {
            default: 0,
            description: "Stroke losing width further from center",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Stroke Falloff', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: "Color transition sharpness (0 = hard edge, 1 = smooth fade)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Softness', group: 'Effect' }
        },
        speed: {
            default: 1.0,
            description: "Animation speed (negative values reverse direction)",
            ui: { type: 'range', min: -3, max: 3, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: "The center point of the spiral",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        scale: {
            default: 1.0,
            description: "Scale factor for spiral bands (higher = more bands, lower = fewer bands)",
            ui: { type: ['range', 'map'], min: 0.1, max: 5, step: 0.1, label: 'Scale', group: 'Effect' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        }
    },
    paint: rampOver(
        dist.spiral({center: p('center'), scale: p('scale'), width: p('strokeWidth'), falloff: p('strokeFalloff'), softness: p('softness')}),
        pair(p('colorA'), p('colorB'), p('colorSpace')),
    )
})

export default componentDefinition
