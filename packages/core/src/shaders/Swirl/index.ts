import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {rampOver, flowField, threshold, pulse, share, shimmered, stops} from "@coreroot/std/paint/fields"
import {transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

// The signed multi-layer flow pattern, evaluated once per composition (see `paint`).
const swirlPatternField = share(flowField({detail: p('detail')}))

export interface ComponentProps{
    colorA: string
    colorB: string
    stops: ColorStop[] | null
    speed: number
    detail: number
    blend: number
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Swirl",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Textures",
    description: "Flowing swirl pattern with multi-layered noise",
    // Per-node animated time, read by the paint's flow clock.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: '#1275d8',
            description: 'Primary gradient color',
            transform: transformColor,
            ui: {
                type: 'color',
                label: 'Color A',
                group: 'Colors'
            }
        },
        colorB: {
            default: '#e19136',
            description: 'Secondary gradient color',
            transform: transformColor,
            ui: {
                type: 'color',
                label: 'Color B',
                group: 'Colors'
            }
        },
        stops: colorStopsPropConfig(),
        speed: {
            default: 1,
            description: 'Flow animation speed',
            ui: {
                type: 'range',
                min: 0,
                max: 5,
                step: 0.1,
                label: 'Speed',
                group: 'Animation'
            }
        },
        detail: {
            default: 1,
            description: 'Level of detail and intricacy in the swirl patterns',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 5,
                step: 0.1,
                label: 'Detail',
                group: 'Effect'
            }
        },
        blend: {
            default: 50,
            description: 'Skew color balance toward A (lower values) or B (higher values)',
            ui: {
                type: ['range', 'map'],
                min: 0,
                max: 100,
                step: 1,
                label: 'Blend',
                group: 'Effect'
            }
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
        }
    },
    // The layered flow pattern feeds BOTH the color threshold and the shimmer gain, so it
    // is shared (one evaluation). Blend (0–100) remaps to a ±0.3 threshold bias.
    paint: shimmered(
        rampOver(
            threshold(swirlPatternField, {low: 0.3, high: 0.7, bias: {base: p('blend'), add: -50, mul: 0.006}}),
            stops(p('colorSpace')),
        ),
        pulse(swirlPatternField, {speed: 2.5, span: 8, depth: 0.015}),
    )
})

export default componentDefinition
