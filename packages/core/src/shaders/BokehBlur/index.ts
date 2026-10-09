import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineStd, p as prop} from "@coreroot/std"
import {bokehDefocus} from "@coreroot/std/effects/blurs"
import {transformAngle} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    radius: number
    highlightGain: number
    highlightThreshold: number
    bladeShape: string
    bladeCount: number
    bladeRotation: Parameters<typeof transformAngle>[0]
    chromaticFringe: number
    catEye: number
}

const MAX_RADIUS = 100

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "BokehBlur",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Blurs",
    description: "Photographic lens blur where bright highlights bloom into aperture-shaped discs",
    requiresRTT: true,
    requiresChild: true,
    props: {
        radius: {
            default: 50,
            description: 'Defocus amount — how far the lens blur spreads',
            ui: {type: ['range', 'map'], min: 0, max: MAX_RADIUS, step: 1, label: 'Radius', group: 'Effect'}
        },
        highlightGain: {
            default: 4,
            description: 'How strongly bright highlights bloom into discs',
            ui: {type: ['range', 'map'], min: 0, max: 10, step: 0.1, label: 'Highlight Bloom', group: 'Highlights'}
        },
        highlightThreshold: {
            default: 0.6,
            description: 'Brightness above which a highlight forms a disc',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Highlight Threshold', group: 'Highlights'}
        },
        bladeShape: {
            default: 'blades',
            description: 'Aperture shape the highlights bloom into — bladed iris or a novelty cut-out (heart, star, …)',
            ui: {
                type: 'select',
                options: [
                    {label: 'Blades', value: 'blades'},
                    {label: 'Circle', value: 'circle'},
                    {label: 'Star', value: 'star'},
                    {label: 'Heart', value: 'heart'},
                    {label: 'Flower', value: 'flower'},
                    {label: 'Cross', value: 'cross'},
                    {label: 'Ring', value: 'ring'}
                ],
                label: 'Blade Shape',
                group: 'Aperture'
            }
        },
        bladeCount: {
            default: 6,
            description: 'Blade or point count for the Blades, Star and Flower shapes — 0–2 blades is a circular iris',
            ui: {type: 'range', min: 0, max: 9, step: 1, label: 'Aperture Blades', group: 'Aperture'}
        },
        bladeRotation: {
            default: 0,
            transform: transformAngle,
            description: 'Rotation of the aperture shape (in degrees)',
            ui: {type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Blade Rotation', group: 'Aperture'}
        },
        chromaticFringe: {
            default: 0.2,
            description: 'Lens color fringing on the disc edges',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Chromatic Fringe', group: 'Aperture'}
        },
        catEye: {
            default: 0,
            description: 'Clips the discs into cat-eye shapes toward the frame corners, like a real lens vignetting its aperture',
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Cat Eye', group: 'Aperture'}
        }
    },

    // The two halves of the bokeh-defocus recipe: the aperture-table gather compute
    // (uniform-radius or map-driven fork) + the defocused-buffer sampling fragment.
    ...bokehDefocus({
        radius: prop('radius'),
        gain: prop('highlightGain'),
        threshold: prop('highlightThreshold'),
        shape: prop('bladeShape'),
        blades: prop('bladeCount'),
        rotation: prop('bladeRotation'),
        fringe: prop('chromaticFringe'),
        catEye: prop('catEye'),
    }),
})

export default componentDefinition
