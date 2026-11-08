import {defineShader, p} from "@coreroot/std"
import {
    worleyNoise,
    worleyHash,
    transformWorleyMode as transformMode,
    transformWorleyDistance as transformDistance,
} from "@coreroot/std/paint/noise"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

// The shared legacy sin-fract cell hash, re-exported under this shader's historical name for any
// external importer.
export {worleyHash}

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorSpace: string
    scale: number
    mode: string
    distance: string
    octaves: number
    lacunarity: number
    persistence: number
    jitter: number
    contrast: number
    balance: number
    seed: number
    speed: number
}

// std generator: a 9-neighbour cellular fold → F1/F2 distances under a selectable metric, reduced
// per `mode`, over up to 4 gated fractal octaves. mode/distance/octaves are compile-time props
// (structural hash) read as CPU values by the noun.
export const componentDefinition = defineShader<ComponentProps>({
    name: "WorleyNoise",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Cellular noise field — distance-based, with selectable feature combinations and fractal octaves",
    acceptsUVContext: true,
    // Per-node animated time driving how fast each cell's point drifts.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Color where the noise field is low (typically near cell centers)',
            ui: { type: 'color', label: 'Color A (Low)', group: 'Colors' }
        },
        colorB: {
            default: '#000000',
            transform: transformColor,
            description: 'Color where the noise field is high (typically near cell boundaries)',
            ui: { type: 'color', label: 'Color B (High)', group: 'Colors' }
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
        },
        scale: {
            default: 6,
            description: 'Number of cells across the canvas at the base octave',
            ui: { type: ['range', 'map'], min: 1, max: 30, step: 0.5, label: 'Scale', group: 'Effect' }
        },
        mode: {
            default: 'f1',
            transform: transformMode,
            compileTime: true,
            description: 'Field type. F1 = distance to nearest point. F2 = distance to second-nearest. F2 − F1 emphasises cell boundaries.',
            ui: {
                type: 'select',
                options: [
                    { label: 'F1 (Cells)', value: 'f1' },
                    { label: 'F2 (Wider Cells)', value: 'f2' },
                    { label: 'F2 − F1 (Edges)', value: 'f2MinusF1' },
                    { label: 'F1 + F2', value: 'f1PlusF2' },
                    { label: 'F1 × F2', value: 'f1TimesF2' }
                ],
                label: 'Mode',
                group: 'Effect'
            }
        },
        distance: {
            default: 'euclidean',
            transform: transformDistance,
            compileTime: true,
            description: 'Distance metric. Euclidean = round cells. Manhattan = diamond. Chebyshev = square.',
            ui: {
                type: 'select',
                options: [
                    { label: 'Euclidean', value: 'euclidean' },
                    { label: 'Manhattan', value: 'manhattan' },
                    { label: 'Chebyshev', value: 'chebyshev' }
                ],
                label: 'Distance',
                group: 'Effect'
            }
        },
        octaves: {
            default: 1,
            compileTime: true,
            description: 'Number of fractal layers stacked at progressively finer scales',
            ui: { type: 'range', min: 1, max: 4, step: 1, label: 'Octaves', group: 'Effect' }
        },
        lacunarity: {
            default: 2,
            description: 'Scale multiplier between octaves (only active when Octaves > 1)',
            ui: { type: ['range', 'map'], min: 1.5, max: 4, step: 0.1, label: 'Lacunarity', group: 'Effect', condition: { octaves: [2, 3, 4] } }
        },
        persistence: {
            default: 0.5,
            description: 'Amplitude multiplier between octaves (only active when Octaves > 1)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Persistence', group: 'Effect', condition: { octaves: [2, 3, 4] } }
        },
        jitter: {
            default: 1,
            description: 'How much each cell\'s point drifts inside its cell. 0 = rigid grid (banded look), 1 = fully random.',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Jitter', group: 'Effect' }
        },
        contrast: {
            default: 1,
            description: 'Steepness of the gradient between low and high regions',
            ui: { type: ['range', 'map'], min: 0.25, max: 4, step: 0.01, label: 'Contrast', group: 'Effect' }
        },
        balance: {
            default: 0,
            description: 'Shifts the gradient midpoint. Negative pulls the field toward Color A, positive toward Color B.',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Balance', group: 'Effect' }
        },
        seed: {
            default: 0,
            description: 'Random seed — shifts the cell pattern without changing its overall structure',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
        },
        speed: {
            default: 0.5,
            description: 'Animation speed — how fast each cell\'s point drifts',
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Speed', group: 'Animation' }
        }
    },

    paint: worleyNoise({
        scale: p('scale'),
        jitter: p('jitter'),
        lacunarity: p('lacunarity'),
        persistence: p('persistence'),
        contrast: p('contrast'),
        balance: p('balance'),
        seed: p('seed'),
        mode: p('mode'),
        distance: p('distance'),
        octaves: p('octaves'),
        space: p('colorSpace'),
    })
})

export default componentDefinition
