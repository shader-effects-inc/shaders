import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {spectralLens} from "@coreroot/std/effects/lens"
import {LENS_MAX_SAMPLES} from "@coreroot/gpu/kit/lensParts"
import {transformAngle, transformPosition} from "@coreroot/utilities/transformations"

// Port of Paper Shaders' lens-distortion (MIT, github.com/paper-design/shaders). Separates the
// child into shifting color layers (the chromatic aberration of a lens) and warps the geometry
// outward/inward like barrel/pincushion distortion, with an optional circular lens crop, swirl,
// noise scatter and film grain. The whole effect is the std spectralLens recipe below — the
// runtime-count spectral fan lives in the kit's lens statement parts.

export interface ComponentProps {
    center: Parameters<typeof transformPosition>[0]
    spread: number
    bias: number
    angle: Parameters<typeof transformAngle>[0]
    perspective: number
    count: number
    dispersion: number
    dispersionShift: number
    dispersionColor: number
    focusCenter: number
    focusEdges: number
    swirl: number
    noise: number
    noiseFrequency: number
    noiseOffset: number
    lensBulge: number
    lensCircle: number
    grainMixer: number
    grainOverlay: number
}

const MAX_SAMPLES = LENS_MAX_SAMPLES

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "LensDistortion",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Split content into shifting chromatic layers with barrel or pincushion lens warp. Powered by Paper Shaders.",
    requiresRTT: true,
    requiresChild: true,
    props: {
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: 'The center point of the lens effect — the focus zone, lens warp, circle crop and swirl all pivot around it',
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        spread: {
            default: 0.6,
            description: 'Strength of the color split; how far the color layers are pushed apart (0 is off)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Spread', group: 'Spread' }
        },
        angle: {
            default: 0,
            transform: transformAngle,
            description: 'Direction of the spread in degrees',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Spread' }
        },
        perspective: {
            default: 0.1,
            description: 'Shapes the spread direction from a straight line (0) to a radial burst out from the centre (1)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Perspective', group: 'Spread' }
        },
        bias: {
            default: 1,
            description: 'Shifts the colors toward one end of the spread; 0 spaces them evenly',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Bias', group: 'Spread' }
        },
        count: {
            default: 35,
            description: 'Number of sampled color layers along the spread; higher is smoother and costlier',
            ui: { type: ['range', 'map'], min: 2, max: MAX_SAMPLES, step: 1, label: 'Count', group: 'Spread' }
        },
        dispersion: {
            default: 1,
            description: 'Overall amount of color dispersion; 1 gives each layer its own color from the spectrum, 0 keeps the original color',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Dispersion', group: 'Dispersion' }
        },
        dispersionShift: {
            default: 0,
            description: 'Balance of the dispersion between a soft circular zone at the centre and the rest; -1 keeps it in the centre only, 1 at the edges only',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Shift', group: 'Dispersion' }
        },
        dispersionColor: {
            default: 0.6,
            description: 'Rotates the dispersion colors around the hue wheel, 0 to 1 for a full turn',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Hue', group: 'Dispersion' }
        },
        focusCenter: {
            default: 0.8,
            description: 'Reduces the spread in a circular zone at the centre; 0 keeps it full to the centre',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Focus Center', group: 'Focus' }
        },
        focusEdges: {
            default: 1,
            description: 'Reduces the spread toward the edges; 1 restores the original image there',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Focus Edges', group: 'Focus' }
        },
        lensBulge: {
            default: 0,
            description: 'Radial lens warp; positive bulges out like a fisheye, negative pinches in like a pincushion',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Bulge', group: 'Lens' }
        },
        lensCircle: {
            default: 0,
            description: 'Squeezes pixels outside the inscribed circle inward so the outline becomes a circle',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Circle', group: 'Lens' }
        },
        swirl: {
            default: 0.35,
            description: 'Rotates the color layers around the centre by an angle growing along the spread; 0 is off',
            ui: { type: ['range', 'map'], min: -1, max: 1, step: 0.01, label: 'Swirl', group: 'Lens' }
        },
        noise: {
            default: 0,
            description: 'Scatters the spread direction with noise; 0 is off',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Noise', group: 'Noise' }
        },
        noiseFrequency: {
            default: 0.25,
            description: 'Frequency of the direction noise; higher is finer (no effect with noise at 0)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Frequency', group: 'Noise' }
        },
        noiseOffset: {
            default: 0,
            description: 'Offsets the noise pattern for a different seed (no effect with noise at 0)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Seed', group: 'Noise' }
        },
        grainMixer: {
            default: 0,
            description: 'Scatters the spread with grain noise, breaking up the edges of the color layers',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Grain Mixer', group: 'Grain' }
        },
        grainOverlay: {
            default: 0,
            description: 'Post-processing black/white grain overlay',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Grain Overlay', group: 'Grain' }
        }
    },

    // The whole fragment is the spectral-lens recipe: lensGeometry → spreadAxis → spectralFan →
    // grainOverlay over the child RTT.
    gpu: {fragment: spectralLens({
        center: p('center'), spread: p('spread'), bias: p('bias'), angle: p('angle'),
        perspective: p('perspective'), count: p('count'),
        dispersion: p('dispersion'), dispersionShift: p('dispersionShift'), dispersionColor: p('dispersionColor'),
        focusCenter: p('focusCenter'), focusEdges: p('focusEdges'), swirl: p('swirl'),
        noise: p('noise'), noiseFrequency: p('noiseFrequency'), noiseOffset: p('noiseOffset'),
        lensBulge: p('lensBulge'), lensCircle: p('lensCircle'),
        grainMixer: p('grainMixer'), grainOverlay: p('grainOverlay'),
    })},
})

export default componentDefinition
