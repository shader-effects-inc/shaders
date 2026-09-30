import {defineStd, p} from "@coreroot/std"
import {gatherStack, rgbSplit, adjust, scanlines, phosphorMask, vignetteOverlay, opaque} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    pixelSize: number,
    colorShift: number,
    scanlineIntensity: number,
    scanlineFrequency: number,
    brightness: number,
    contrast: number,
    vignetteIntensity: number,
    vignetteRadius: number
}

export const componentDefinition = defineStd<ComponentProps>({
    name: "CRTScreen",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Retro CRT monitor simulation with scanlines",
    props: {
        pixelSize: {
            default: 128,
            description: 'Density of the RGB phosphor stripes — about half this many across the canvas width (higher = smaller, finer pixels)',
            ui: { type: ['range', 'map'], min: 8, max: 128, step: 1, label: 'Pixel Size', group: 'Effect' }
        },
        colorShift: {
            default: 1,
            description: 'Chromatic aberration amount',
            ui: { type: ['range', 'map'], min: 0, max: 10, step: 0.1, label: 'Color Shift', group: 'Effect' }
        },
        scanlineIntensity: {
            default: 0.3,
            description: 'Strength of horizontal scanlines',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Scanline Intensity', group: 'Effect' }
        },
        scanlineFrequency: {
            default: 200,
            description: 'Number of scanlines across screen',
            ui: { type: ['range', 'map'], min: 100, max: 800, step: 10, label: 'Scanline Frequency', group: 'Effect' }
        },
        brightness: {
            default: 1,
            description: 'Screen brightness boost',
            ui: { type: ['range', 'map'], min: 0.5, max: 2, step: 0.1, label: 'Brightness', group: 'Adjustments' }
        },
        contrast: {
            default: 1,
            description: 'Screen contrast enhancement',
            ui: { type: ['range', 'map'], min: 0.5, max: 2, step: 0.1, label: 'Contrast', group: 'Adjustments' }
        },
        vignetteIntensity: {
            default: 1,
            description: 'Strength of corner darkening effect (0 = off)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Vignette Intensity', group: 'Effect' }
        },
        vignetteRadius: {
            default: 0.5,
            description: 'How far the vignette extends inward (0 = edges only, 1 = reaches center)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Vignette Radius', group: 'Effect' }
        }
    },

    // The retro-monitor stack, literally — RGB-split sampling at three chromatic-aberration UVs,
    // then adjust → scanlines → phosphor mask → vignette, finished as an opaque screen. Not a
    // uvRemap candidate (three sample UVs + a screen-space overlay stack).
    effect: gatherStack(rgbSplit({amount: p('colorShift')}), [
        adjust({brightness: p('brightness'), contrast: p('contrast')}),
        scanlines({frequency: p('scanlineFrequency'), intensity: p('scanlineIntensity')}),
        phosphorMask({pitch: p('pixelSize')}),
        vignetteOverlay({radius: p('vignetteRadius'), intensity: p('vignetteIntensity')}),
        opaque(),
    ]),
})

export default componentDefinition
