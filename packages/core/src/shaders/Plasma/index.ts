import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, tone, scaledVolume, warped, noiseField, stops} from "@coreroot/std/paint/fields"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
    density: number
    speed: number
    intensity: number
    warp: number
    contrast: number
    balance: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Plasma",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Animated effect of glowing plasma",
    acceptsUVContext: true,
    // Per-node animated time, read by the paint's warp clock.
    animatedTime: { speed: 'speed' },
    props: {
        density: {
            default: 2,
            description: "Density of the plasma pattern",
            ui: { type: ['range', 'map'], min: 0, max: 4, step: 0.1, label: 'Density', group: 'Effect' }
        },
        speed: {
            default: 2,
            description: "Animation speed",
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Speed', group: 'Effect' }
        },
        intensity: {
            default: 1.5,
            description: "Brightness and spread of the plasma glow",
            ui: { type: ['range', 'map'], min: 0.1, max: 3, step: 0.1, label: 'Intensity', group: 'Colors' }
        },
        warp: {
            default: 0.4,
            description: "How much the flow distorts and swirls",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Warp', group: 'Effect' }
        },
        contrast: {
            default: 1,
            description: 'Push darks darker and lights lighter',
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.1, label: 'Contrast', group: 'Colors' }
        },
        balance: {
            default: 50,
            description: 'Skew color balance toward A (higher) or B (lower)',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Balance', group: 'Colors' }
        },
        colorA: {
            default: '#7018be',
            description: 'Primary color',
            transform: transformColor,
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#000000',
            description: 'Secondary color',
            transform: transformColor,
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        stops: colorStopsPropConfig(),
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
        }
    },
    // Domain-warped MaterialX noise on an aspect-corrected volume slab, tone-shaped, then
    // ramped through the stops/pair palette.
    paint: rampOver(
        tone(
            scaledVolume(
                warped(noiseField('mx3'), {amount: p('warp'), amountScale: 4, timeScale: 0.125}),
                {scale: p('density')},
            ),
            {glow: p('intensity'), contrast: p('contrast'), balance: p('balance'), invert: true},
        ),
        stops(p('colorSpace')),
    )
})

export default componentDefinition
