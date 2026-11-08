import {defineShader, p} from "@coreroot/std"
import {ringWaves, strokeOver} from "@coreroot/std/paint/patterns"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0],
    colorA: Parameters<typeof transformColor>[0],
    colorB: Parameters<typeof transformColor>[0],
    speed: number,
    frequency: number,
    softness: number,
    thickness: number,
    phase: number
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "Ripples",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Concentric animated ripples emanating from a point",
    acceptsUVContext: true,
    // Per-node animated time: the renderer registers `_animTime` and advances it by
    // `deltaTime * speed` (speed=0 pauses, negative reverses). The paint noun reads it as the
    // outward ripple motion.
    animatedTime: { speed: 'speed' },
    props: {
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: "The center point where ripples emanate from",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        colorA: {
            default: "#ffffff",
            transform: transformColor,
            description: "Color of the ripple waves",
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: "#000000",
            transform: transformColor,
            description: "Background color between ripples",
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        speed: {
            default: 1.0,
            description: 'Speed of ripple animation',
            ui: { type: 'range', min: -5, max: 5, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        frequency: {
            default: 20.0,
            description: 'Number of ripples/spacing between them',
            ui: { type: ['range', 'map'], min: 1, max: 80, step: 0.1, label: 'Frequency', group: 'Effect' }
        },
        softness: {
            default: 0,
            description: 'Softness of ripple edges',
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.1, label: 'Softness', group: 'Effect' }
        },
        thickness: {
            default: 0.5,
            description: 'Thickness of each ripple band',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Thickness', group: 'Effect' }
        },
        phase: {
            default: 0,
            description: 'Phase offset for ripple animation',
            ui: { type: ['range', 'map'], min: 0, max: 6.28, step: 0.1, label: 'Phase', group: 'Animation' }
        }
    },

    // The recipe: the concentric ring mask (animated by the node's clock plus `phase`) strokes
    // the ring color over the background, alpha-weighted in linear space.
    paint: strokeOver({
        fill: p('colorB'),
        stroke: p('colorA'),
        mask: ringWaves({
            center: p('center'),
            frequency: p('frequency'),
            thickness: p('thickness'),
            softness: p('softness'),
            phase: p('phase'),
        }),
    })
})

export default componentDefinition
