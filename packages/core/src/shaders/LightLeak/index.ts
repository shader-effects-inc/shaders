import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {asLocal, animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, resolveArg} from "@coreroot/std"
import {screenGlow, beamFrame, slowDrift, beamBloom, beamStreaks, dithered, heatRamp} from "@coreroot/std/paint/light"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    position: Parameters<typeof transformPosition>[0]
    spread: number
    intensity: number
    streaks: number
    colorHot: Parameters<typeof transformColor>[0]
    colorMid: Parameters<typeof transformColor>[0]
    colorFringe: Parameters<typeof transformColor>[0]
    flicker: number
    speed: number
    seed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "LightLeak",
    role: 'filter',
    species: 'pointwise',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Photorealistic film light leak — warm overexposed light bleeding in from a draggable anchor point, with streak bands, chromatic fringing, and slow breathing that evolves over time",
    usesPointer: false,
    animatedTime: { speed: 'speed' },
    props: {
        position: {
            default: { x: 0.95, y: 0.4 },
            transform: transformPosition,
            description: 'Anchor point of the leak — the light beams from here toward the canvas centre',
            ui: { type: 'position', label: 'Position', group: 'Leak', units: ['%', 'px'] }
        },
        spread: {
            default: 0.55,
            description: 'How far the light bleeds into the frame',
            ui: { type: ['range', 'map'], min: 0.1, max: 1.5, step: 0.01, label: 'Spread', group: 'Leak' }
        },
        intensity: {
            default: 0.15,
            description: 'Exposure strength of the leak',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Leak' }
        },
        streaks: {
            default: 0.6,
            description: 'Strength of the parallel streak bands further into the frame',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Streaks', group: 'Leak' }
        },
        colorHot: {
            default: '#fff3c4',
            transform: transformColor,
            description: 'Core color where the leak is fully overexposed',
            ui: { type: 'color', label: 'Hot Core', group: 'Colors' }
        },
        colorMid: {
            default: '#ff7a2f',
            transform: transformColor,
            description: 'Main body color of the leak',
            ui: { type: 'color', label: 'Midtone', group: 'Colors' }
        },
        colorFringe: {
            default: '#a63d8f',
            transform: transformColor,
            description: 'Outer fringe color where the light fades out',
            ui: { type: 'color', label: 'Fringe', group: 'Colors' }
        },
        flicker: {
            default: 0.35,
            description: 'Breathing of the leak intensity',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Flicker', group: 'Animation' }
        },
        speed: {
            default: 1,
            description: 'Speed of the breathing, drift and evolution. 0 freezes the leak.',
            ui: { type: 'range', min: 0, max: 4, step: 0.1, label: 'Speed', group: 'Animation' }
        },
        seed: {
            default: 0,
            description: 'Random seed — re-rolls the drift and breathing pattern',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation' }
        }
    },
    // The recipe: the beam-local frame at the anchor → slow drift/breathing signals → bloom +
    // shoulder + drifting streak bands accumulated into an overexposed heat field → hash dither →
    // the chromatic heat ramp, screen-composited over the child like exposure (the leak is
    // emitted light, so it raises alpha over transparent areas too).
    effect: screenGlow((params) => {
        const t = asLocal(animatedTime(params).mul(0.6).add(resolveArg(p('seed'), params).mul(7.9)), 'leakClock')
        const frame = asLocal(beamFrame(p('position'))(params), 'leakFrame')
        const drift = asLocal(slowDrift({seed: p('seed'), flicker: p('flicker')})(t, params), 'leakDrift')
        const bloom = asLocal(beamBloom({spread: p('spread')})(frame, drift, params), 'leakBloom')
        const streaks = beamStreaks({spread: p('spread'), strength: p('streaks')})(frame, drift, params)
        // Composite heat field: main ×1.25 + shoulder + streaks ×0.85, scaled by intensity
        // (×2.2 so the 0–1 slider reaches full overexposure) and the breathing flicker.
        const heat = bloom.member('x').mul(1.25).add(bloom.member('y')).add(streaks.mul(0.85))
            .mul(resolveArg(p('intensity'), params)).mul(2.2).mul(drift.member('x'))
        return heatRamp({hot: p('colorHot'), mid: p('colorMid'), fringe: p('colorFringe')})
            .taps(dithered(heat, params.ctx.uv, t), params)
    }),
    missingChildMessage: 'You must pass a child component into the Light Leak shader.',
})

export default componentDefinition
