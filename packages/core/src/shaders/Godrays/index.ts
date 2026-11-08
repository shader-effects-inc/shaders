import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {coverageOver, withFrame, seamlessRadialFrame, noiseRays} from "@coreroot/std/paint/light"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    density: number
    intensity: number
    spotty: number
    speed: number
    rayColor: string
    backgroundColor: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Godrays",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Volumetric light rays emanating from a point",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        center: {
            default: { x: 0, y: 0 },
            transform: transformPosition,
            description: "The center point of the god rays",
            ui: {type: 'position', label: 'Center', group: 'Position'}
        },
        density: {
            default: 0.3,
            description: 'Frequency of ray sectors',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Density', group: 'Effect'}
        },
        intensity: {
            default: 0.8,
            description: 'Ray visibility within sectors',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Intensity', group: 'Effect'}
        },
        spotty: {
            default: 1,
            description: 'Density of spots on rays (higher = more spots)',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Spotty', group: 'Effect'}
        },
        speed: {
            default: 0.5,
            description: 'Animation speed of the rays',
            ui: {type: 'range', min: 0, max: 2, step: 0.1, label: 'Speed', group: 'Animation'}
        },
        rayColor: {
            default: "#4283fb",
            transform: transformColor,
            description: 'Color of the light rays',
            ui: {type: 'color', label: 'Ray Color', group: 'Colors'}
        },
        backgroundColor: {
            default: "transparent",
            transform: transformColor,
            description: 'Background color',
            ui: {type: 'color', label: 'Background Color', group: 'Colors'}
        }
    },
    // The recipe: the seam-free radial frame → the two-layer noise ray stack → the ray color
    // composited over the background by coverage (straight alpha, linear RGB).
    paint: coverageOver({
        color: p('rayColor'),
        background: p('backgroundColor'),
        coverage: withFrame(seamlessRadialFrame(p('center')), noiseRays({
            density: p('density'),
            intensity: p('intensity'),
            spotty: p('spotty'),
        })),
    }),
})

export default componentDefinition
