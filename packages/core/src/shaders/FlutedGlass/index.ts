import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, recompileWhen, p} from "@coreroot/std"
import {gatherStack, fluteFrame, refractedTaps, fluteHighlight} from "@coreroot/std/effects/blurs"

import {transformColor, transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    shape: string
    angle: number
    frequency: number
    softness: number
    waveAmplitude: number
    waveFrequency: number
    speed: number
    refraction: number
    aberration: number
    lightAngle: number
    highlight: number
    highlightSoftness: number
    highlightColor: Parameters<typeof transformColor>[0]
    edges: string
}

// The flute geometry (refracted UV + chromatic offset + flute slope) both stages read.
const flute = fluteFrame({
    shape: p('shape'),
    angle: p('angle'),
    frequency: p('frequency'),
    softness: p('softness'),
    waveAmplitude: p('waveAmplitude'),
    waveFrequency: p('waveFrequency'),
    refraction: p('refraction'),
    aberration: p('aberration'),
})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "FlutedGlass",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Distortions",
    description: "Full-screen fluted glass effect — refracts content through repeating cylindrical bars",
    animatedTime: { speed: 'speed' },
    props: {
        shape: {
            default: 'bars',
            description: 'Cross-section shape of each flute',
            compileTime: true,
            transform: (value: string) => ({ 'bars': 0, 'rounded': 1, 'waves': 2 } as Record<string, number>)[value] ?? 0,
            ui: {
                type: 'select',
                options: [
                    {label: 'Bars', value: 'bars'},
                    {label: 'Rounded', value: 'rounded'},
                    {label: 'Waves', value: 'waves'}
                ],
                label: 'Shape',
                group: 'Effect'
            }
        },
        angle: {
            default: 0,
            description: 'Direction of the flutes in degrees (0 = vertical bars)',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        frequency: {
            default: 10,
            description: 'Number of flutes across the longest viewport axis',
            ui: { type: ['range', 'map'], min: 1, max: 20, step: 1, label: 'Frequency', group: 'Effect' }
        },
        softness: {
            default: 0.5,
            description: 'How smoothly distortion fades from each flute centre to its edge (0 = flat middle / sharp seams, 1 = gentle curve)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Effect' }
        },
        waveAmplitude: {
            default: 0.06,
            description: 'How far each flute sways horizontally as it travels (Waves shape only)',
            ui: { type: ['range', 'map'], min: 0, max: 0.5, step: 0.01, label: 'Wave Amplitude', group: 'Effect', condition: { shape: 'waves' } }
        },
        waveFrequency: {
            default: 1.5,
            description: 'How many sways fit along each flute (Waves shape only)',
            ui: { type: ['range', 'map'], min: 0.1, max: 10, step: 0.1, label: 'Wave Frequency', group: 'Effect', condition: { shape: 'waves' } }
        },
        speed: {
            default: 0,
            description: 'Animation speed — drifts the flute pattern over time and flows wave perturbations',
            ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        refraction: {
            default: 1.5,
            description: 'How aggressively each flute bends content beneath it',
            ui: { type: ['range', 'map'], min: 0, max: 4, step: 0.01, label: 'Refraction', group: 'Effect' }
        },
        aberration: {
            default: 0.2,
            description: 'Chromatic aberration — splits RGB along the refraction direction at flute seams',
            // Recompile only when aberration crosses on/off (1-sample vs 3-sample chromatic split).
            recompile: recompileWhen((prev, next) => ((prev as number) > 0) !== ((next as number) > 0)),
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Aberration', group: 'Effect' }
        },
        lightAngle: {
            default: 30,
            description: 'Direction the light source is coming from (0 = head-on, 90 = grazing)',
            ui: { type: 'range', min: -90, max: 90, step: 1, label: 'Light Angle', group: 'Highlight' }
        },
        highlight: {
            default: 0.2,
            description: 'Strength of the specular reflection on each flute',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Highlight', group: 'Highlight' }
        },
        highlightSoftness: {
            default: 0.3,
            description: 'Spread of the specular peak (0 = pin-tight, 1 = broad sheen)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Highlight Softness', group: 'Highlight' }
        },
        highlightColor: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Color of the specular highlight',
            ui: { type: 'color', label: 'Highlight Color', group: 'Highlight' }
        },
        edges: {
            default: 'mirror',
            description: 'How to handle edges when distortion samples beyond the canvas',
            transform: transformEdges,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Transparent', value: 'transparent'},
                    {label: 'Mirror', value: 'mirror'},
                    {label: 'Wrap', value: 'wrap'}
                ],
                label: 'Edges',
                group: 'Effect'
            }
        }
    },

    // Geometry → edge-handled refracted tap(s) → Blinn specular over the flute slope. Silent on
    // a missing child, as this shader has always been.
    effect: gatherStack(
        refractedTaps(flute, {aberration: p('aberration'), edges: p('edges')}),
        [fluteHighlight(flute, {
            lightAngle: p('lightAngle'),
            highlight: p('highlight'),
            softness: p('highlightSoftness'),
            color: p('highlightColor'),
        })],
        {resultAlpha: 'straight'},
    ),
})

export default componentDefinition
