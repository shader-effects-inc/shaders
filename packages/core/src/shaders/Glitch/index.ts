import {defineShader, p} from "@coreroot/std"
import {gatherStack, glitchFrame, glitchRgbSplit, colorBarFills, distortedScanlines} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    intensity: number
    speed: number
    rgbShift: number
    blockDensity: number
    colorBarIntensity: number
    mirrorAmount: number
    scanlineIntensity: number
}

// The shared glitch geometry every stage reads — computed once per composition.
const glitchGeometry = glitchFrame({
    intensity: p('intensity'),
    speed: p('speed'),
    blockDensity: p('blockDensity'),
    mirrorAmount: p('mirrorAmount'),
})

export const componentDefinition = defineShader<ComponentProps>({
    name: "Glitch",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Digital glitch that melts pixels and distorts colors",
    props: {
        intensity: {
            default: 0.5,
            description: 'Overall glitch strength and frequency of glitch bursts',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Effect' }
        },
        speed: {
            default: 1,
            description: 'How fast the glitch pattern evolves',
            ui: { type: 'range', min: 0.1, max: 5, step: 0.1, label: 'Speed', group: 'Effect' }
        },
        rgbShift: {
            default: 5,
            description: 'Amount of chromatic aberration (RGB channel splitting)',
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.5, label: 'RGB Shift', group: 'Effect' }
        },
        blockDensity: {
            default: 10,
            description: 'Base number of horizontal glitch bands',
            ui: { type: ['range', 'map'], min: 2, max: 50, step: 1, label: 'Block Density', group: 'Effect' }
        },
        colorBarIntensity: {
            default: 0.2,
            description: 'Intensity of vivid neon color bar overlay in glitch regions',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Color Bars', group: 'Effect' }
        },
        mirrorAmount: {
            default: 0.3,
            description: 'Chance of glitch blocks showing mirrored/flipped content',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Mirror', group: 'Effect' }
        },
        scanlineIntensity: {
            default: 0.2,
            description: 'Visibility of CRT-style horizontal scanlines in distorted areas',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Scanlines', group: 'Effect' }
        }
    },

    // Digital corruption over one shared geometry frame (burst pulse, jitter bands, block
    // shifts, mirror flips) — its hashes feed the RGB-split sampling, then color-bar fills and
    // distortion-gated scanlines. `ctx.time` is the global clock.
    effect: gatherStack(glitchRgbSplit(glitchGeometry, {shift: p('rgbShift')}), [
        colorBarFills(glitchGeometry, {intensity: p('colorBarIntensity')}),
        distortedScanlines(glitchGeometry, {intensity: p('scanlineIntensity')}),
    ]),
})

export default componentDefinition
