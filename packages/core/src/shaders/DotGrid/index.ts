import {defineStd, p} from "@coreroot/std"
import {dotLattice, withAlpha, sampleMapsAtCellCentres} from "@coreroot/std/paint/patterns"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps{
    color: Parameters<typeof transformColor>[0],
    density: number
    dotSize: number
    offset: number
    speed: number
    speedVariance: number
    twinkle: number
}

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineStd<ComponentProps>({
    name: "DotGrid",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Grid of dots with optional twinkling animation",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and DotGrid rasterises at full canvas resolution against that distorted UV. The
    // fwidth-based AA operates on the looked-up UV's derivatives, preserving dot crispness through
    // arbitrary distortions — much better than RTT bilinear.
    acceptsUVContext: true,
    // Per-node animated time: the renderer registers `_animTime`
    // and advances it by `deltaTime * speed` (speed=0 pauses). The paint noun reads it as the drift.
    animatedTime: { speed: 'speed' },
    props: {
        color: {
            default: '#ffffff',
            description: 'The color of the dot',
            transform: transformColor,
            ui: {
                type: 'color',
                label: 'Color',
                group: 'Colors'
            }
        },
        density: {
            default: 30,
            description: 'The number of dots along the canvas height (the width fits as many as the aspect ratio allows)',
            ui: {
                type: 'range',
                min: 1,
                max: 200,
                step: 1,
                label: 'Density',
                group: 'Effect'
            }
        },
        dotSize: {
            default: 0.3,
            description: 'The size of each dot, zero (0) being invisible, one (1) filled the grid with no gaps',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 1,
                step: 0.01,
                label: 'Dot Size',
                group: 'Effect'
            }
        },
        offset: {
            default: 0,
            description: 'Horizontal stagger of alternating rows (0.5 = classic polka-dot brick offset)',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 1,
                step: 0.01,
                label: 'Row Offset',
                group: 'Effect'
            }
        },
        speed: {
            default: 0,
            description: 'Animates the rows drifting horizontally (0 = static)',
            ui: {
                type: 'range',
                min: -3,
                max: 3,
                step: 0.01,
                label: 'Speed',
                group: 'Animation'
            }
        },
        speedVariance: {
            default: 0.3,
            description: 'Per-row random speed variance for irregular drifting motion',
            ui: {
                type: 'range',
                min: 0,
                max: 1,
                step: 0.01,
                label: 'Speed Variance',
                group: 'Animation'
            }
        },
        twinkle: {
            default: 0,
            description: 'Intensity of the twinkle effect (0 = off, 1 = full twinkle)',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 1,
                step: 0.1,
                label: 'Twinkle',
                group: 'Animation'
            }
        }
    },
    // Sample mapped props (dotSize, twinkle) at the cell centre rather than the fragment position.
    // Without this, dotSize varies per-fragment within a cell, clipping dots at source boundaries
    // instead of keeping them fully circular.
    mapSampleUVs: sampleMapsAtCellCentres({
        cells: p('density'),
        props: ['dotSize', 'twinkle'],
    }),

    // The recipe: the dot-lattice coverage carries the alpha under the dot color.
    paint: withAlpha(p('color'), dotLattice({
        density: p('density'),
        dotSize: p('dotSize'),
        offset: p('offset'),
        speedVariance: p('speedVariance'),
        twinkle: p('twinkle'),
    }))
})

export default componentDefinition
