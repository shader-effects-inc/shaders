import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, uniformOf} from "@coreroot/std"
import {surfaceOf, segmentFrame} from "@coreroot/std/frames"
import {
    add, clamp, div, exp, gaussBell, local, max, mix, mul, neg, smoothstep, softPlus, splat3, sub, vec4,
} from "@coreroot/std/math"
import {hueWheel} from "@coreroot/std/paint/fields"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    position: Parameters<typeof transformPosition>[0]
    splitPosition: Parameters<typeof transformPosition>[0]
    beamWidth: number
    spread: number
    softness: number
    startFalloff: number
    endFalloff: number
    intensity: number
    beamColor: Parameters<typeof transformColor>[0]
    saturation: number
    speed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Prism",
    role: 'generator',
    category: "Textures",
    description: "A beam of light that fans out and splits into a slowly-rotating rainbow past a controllable point. Transparent background — composite over a dark layer.",
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    acceptsUVContext: true,
    animatedTime: {speed: 'speed'},
    props: {
        position: {
            default: { x: 0.18, y: 0.18 },
            transform: transformPosition,
            description: 'Where the beam originates',
            ui: { type: 'position', label: 'Source', group: 'Beam' }
        },
        beamWidth: {
            default: 0.04,
            description: 'Thickness of the white beam',
            ui: { type: ['range', 'map'], min: 0.002, max: 0.3, step: 0.002, label: 'Beam Width', group: 'Beam' }
        },
        intensity: {
            default: 1.6,
            description: 'Brightness (above 1 blows the core to white)',
            ui: { type: ['range', 'map'], min: 0, max: 4, step: 0.01, label: 'Intensity', group: 'Beam' }
        },
        beamColor: {
            default: '#ffffff',
            transform: transformColor,
            description: 'color of the beam before it splits',
            ui: { type: 'color', label: 'Beam Color', group: 'Beam' }
        },
        startFalloff: {
            default: 0.15,
            description: 'How softly the beam fades in at its source',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Start Falloff', group: 'Beam' }
        },
        endFalloff: {
            default: 0.6,
            description: 'How quickly the rainbow fades out with distance',
            ui: { type: ['range', 'map'], min: 0.02, max: 2, step: 0.01, label: 'End Falloff', group: 'Beam' }
        },
        splitPosition: {
            default: { x: 0.5, y: 0.42 },
            transform: transformPosition,
            description: 'Where the beam starts to fan out and split (projected onto the beam axis)',
            ui: { type: 'position', label: 'Split Point', group: 'Spectrum' }
        },
        spread: {
            default: 0.7,
            description: 'How widely the rainbow diffuses as it travels past the split point',
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Spread', group: 'Spectrum' }
        },
        softness: {
            default: 0.12,
            description: 'How quickly the white beam transitions into the rainbow',
            ui: { type: ['range', 'map'], min: 0.001, max: 1, step: 0.005, label: 'Softness', group: 'Spectrum' }
        },
        saturation: {
            default: 0.95,
            description: 'Rainbow saturation — 0 = white, 1 = full color',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Saturation', group: 'Spectrum' }
        },
        speed: {
            default: 0.1,
            description: 'Speed the hue slowly rotates through the rainbow (0 = static)',
            ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Hue Speed', group: 'Spectrum' }
        }
    },

    // The whole effect in the language: a segment frame from the source to the split point,
    // a soft elbow where the beam passes the split, a longitudinal fade envelope, a widening
    // gaussian fan across the axis, and a white → hue-wheel blend rotating on the node clock.
    paint: (params: GpuFragmentParams): Expr => {
        const u = (name: string) => uniformOf(p(name), params)
        const beam = segmentFrame({from: p('position'), to: p('splitPosition')})(params, surfaceOf(params))

        // The elbow past the split point — a soft relu so the fan opens smoothly.
        const soft = local(max(u('softness'), 0.001), 'soft')
        const past = local(softPlus(sub(beam.along, beam.length), mul(soft, 0.6)), 'past')

        // Longitudinal envelope: fade in from the source, exponential decay past the split.
        const fadeIn = smoothstep(0, max(u('startFalloff'), 0.001), beam.along)
        const fadeOut = exp(neg(div(past, max(u('endFalloff'), 0.001))))
        const longEnv = local(mul(fadeIn, fadeOut), 'longEnv')

        // Lateral fan: the beam widens past the split; gaussian profile across it.
        const fanWidth = max(add(u('beamWidth'), mul(past, u('spread'))), 0.001)
        const an = local(div(beam.across, fanWidth), 'an')
        const profile = gaussBell(an, 2.5)

        // White core → rotating rainbow past the split.
        const splitSat = smoothstep(beam.length, add(beam.length, soft), beam.along)
        const hue = add(mul(an, 0.42), animatedTime(params))
        const col = mix(u('beamColor').member('rgb'), hueWheel(hue), splat3(mul(splitSat, u('saturation'))))

        const amt = local(mul(profile, longEnv), 'amt')
        const rgb = local(mul(mul(col, amt), u('intensity')), 'beamRgb')
        return vec4(rgb.member('xyz'), clamp(amt, 0, 1))
    }
})

export default componentDefinition
