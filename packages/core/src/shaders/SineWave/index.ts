import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {sineStroke} from "@coreroot/std/paint/figures"
import {transformColor, transformPosition, transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps{
    color: Parameters<typeof transformColor>[0]
    amplitude: number
    frequency: number
    speed: number
    angle: Parameters<typeof transformAngle>[0]
    position: Parameters<typeof transformPosition>[0]
    thickness: number
    softness: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "SineWave",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Animated wave with thickness and softness",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and SineWave rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    // Per-node animated time, read by the sineStroke noun.
    animatedTime: { speed: 'speed' },
    props: {
        color: {
            default: "#ffffff",
            transform: transformColor,
            description: "The color of the sine wave",
            ui: {
                type: 'color',
                label: 'Color',
                group: 'Colors'
            }
        },
        amplitude: {
            default: 0.15,
            description: "The height/amplitude of the sine wave",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 1,
                step: 0.1,
                label: 'Amplitude',
                group: 'Effect'
            }
        },
        frequency: {
            default: 1,
            description: "The frequency/number of wave cycles",
            ui: {
                type: ['range', 'map'],
                min: 0.1,
                max: 20,
                step: 0.1,
                label: 'Frequency',
                group: 'Effect'
            }
        },
        speed: {
            default: 1,
            description: "The animation speed of the wave",
            ui: {
                type: 'range',
                min: -5,
                max: 5,
                step: 0.1,
                label: 'Speed',
                group: 'Animation'
            }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: "The rotation angle of the wave (in degrees)",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 360,
                step: 1,
                label: 'Angle',
                group: 'Effect'
            }
        },
        position: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: "The center position of the wave",
            ui: {
                type: 'position',
                label: 'Position',
                group: 'Position'
            }
        },
        thickness: {
            default: 0.2,
            description: "The thickness of the wave line",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 2,
                step: 0.1,
                label: 'Thickness',
                group: 'Effect'
            }
        },
        softness: {
            default: 0.4,
            description: "Edge softness of the wave line",
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 1,
                step: 0.1,
                label: 'Softness',
                group: 'Effect'
            }
        }
    },
    paint: sineStroke({
        position: p('position'),
        angle: p('angle'),
        frequency: p('frequency'),
        amplitude: p('amplitude'),
        thickness: p('thickness'),
        softness: p('softness'),
        color: p('color'),
    })
})

export default componentDefinition
