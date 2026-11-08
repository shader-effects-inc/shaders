import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {coverageMix, withFrame, radialFrame, modulate, softRayLobes, featherMask} from "@coreroot/std/paint/light"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    background: Parameters<typeof transformColor>[0]
    center: Parameters<typeof transformPosition>[0]
    rayCount: number
    softness: number
    radius: number
    feather: number
    speed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "SunBurst",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Radial sunburst rays emanating from a center point",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and SunBurst rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    // Per-node animated time, read by the paint's rotation phase.
    animatedTime: { speed: 'speed' },
    props: {
        color: {
            default: "#ffdd88",
            transform: transformColor,
            description: "Ray color",
            ui: { type: 'color', label: 'Color', group: 'Colors' }
        },
        background: {
            default: "#000000",
            transform: transformColor,
            description: "Background color",
            ui: { type: 'color', label: 'Background', group: 'Colors' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: "Center point of the sunburst",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        rayCount: {
            default: 12,
            description: "Number of rays",
            ui: { type: 'range', min: 3, max: 64, step: 1, label: 'Ray Count', group: 'Effect' }
        },
        softness: {
            default: 0.3,
            description: "Softness of ray edges",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        radius: {
            default: 0.8,
            description: "How far the rays extend from the center",
            ui: { type: ['range', 'map'], min: 0, max: 1.2, step: 0.01, label: 'Radius', group: 'Effect' }
        },
        feather: {
            default: 0.5,
            description: "How gradually the rays fade at their outer edge",
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.01, label: 'Feather', group: 'Effect' }
        },
        speed: {
            default: 0.2,
            description: "Rotation speed — positive values rotate clockwise",
            ui: { type: 'range', min: -2, max: 2, step: 0.1, label: 'Speed', group: 'Animation' }
        }
    },
    // The recipe: a radial frame around the centre → spinning ray lobes × the outer feather
    // (positive speed = clockwise, hence the negative spin), colored by coverage over the
    // background.
    paint: coverageMix({
        color: p('color'),
        background: p('background'),
        coverage: withFrame(radialFrame(p('center')), modulate(
            softRayLobes({count: p('rayCount'), softness: p('softness'), spin: -1}),
            featherMask({radius: p('radius'), feather: p('feather')}),
        ), 'burstFrame'),
        hint: 'burstAlpha',
    })
})

export default componentDefinition
