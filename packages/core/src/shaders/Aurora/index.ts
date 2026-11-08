import type {Expr, GpuShaderDefinition} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineShader, p, uniformOf} from "@coreroot/std"
import {surfaceOf} from "@coreroot/std/frames"
import {colorLadder3} from "@coreroot/std/paint/fields"
import {volumeNoiseAt} from "@coreroot/std/paint/materials"
import {wavyLine} from "@coreroot/std/paint/noise"
import {rayBands} from "@coreroot/std/paint/light"
import {layered} from "@coreroot/std/paint/compose"
import {softBand} from "@coreroot/std/mask"
import {
    add, clamp, div, exp, hashPhase, local, max, mix, mul, pow, smoothstep, splat3, sub, vec3, vec4,
} from "@coreroot/std/math"
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

// ── The Aurora look (this data IS the look) ───────────────────────────────────────────────
// The curtain path's wave stack (each with its historical hash-phase salt), the ray phase
// salt, and the layer stack (time offsets + fading weights).
const ARC_WAVES = [
    {freq: 0.7, speed: 0.4, amount: 0.15, salt: {k: 12.9, big: 4758.5}},
    {freq: 1.6, speed: 0.65, amount: 0.08, salt: {k: 78.2, big: 2847.1}},
    {freq: 2.9, speed: 0.3, amount: 0.04, salt: {k: 41.6, big: 1593.7}},
]
const RAY_SALT = {k: 53.7, big: 3847.2}
// Layer 1 rides at full weight (weight 1 emits no multiply); the per-layer seed offsets are
// fround-folded so each literal equals the historical runtime f32 product exactly.
const LAYERS = [
    {timeOffset: 0.0, weight: 1},
    {timeOffset: 3.1, weight: 0.65},
    {timeOffset: 6.7, weight: 0.45},
    {timeOffset: 9.4, weight: 0.3},
]

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorC: Parameters<typeof transformColor>[0]
    colorSpace: string
    balance: number
    intensity: number
    curtainCount: number
    speed: number
    waviness: number
    rayDensity: number
    height: number
    center: Parameters<typeof transformPosition>[0]
    seed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Aurora",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Mesmerizing aurora borealis with layered curtains, vertical rays, and flowing light.",
    acceptsUVContext: true,
    // Per-node animated time, read inside the field part.
    animatedTime: { speed: 'speed' },
    props: {
        colorA: {
            default: '#a533f8',
            transform: transformColor,
            description: 'Edge color at the curtain base',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#22ee88',
            transform: transformColor,
            description: 'Core color in the bright center',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        colorC: {
            default: '#1694e8',
            transform: transformColor,
            description: 'Tip color at the ray ends',
            ui: { type: 'color', label: 'Color C', group: 'Colors' }
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
        balance: {
            default: 50,
            description: 'Shifts color distribution across the curtain height',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Balance', group: 'Colors' }
        },
        intensity: {
            default: 80,
            description: 'Overall aurora brightness',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Aurora' }
        },
        curtainCount: {
            default: 4,
            compileTime: true,
            description: 'Number of aurora curtain layers',
            ui: { type: 'range', min: 1, max: 4, step: 1, label: 'Curtains', group: 'Aurora' }
        },
        speed: {
            default: 5,
            description: 'Animation speed',
            ui: { type: 'range', min: -10, max: 10, step: 0.1, label: 'Speed', group: 'Aurora' }
        },
        waviness: {
            default: 50,
            description: 'How much the curtains undulate',
            ui: { type: ['range', 'map'], min: 0, max: 200, step: 1, label: 'Waviness', group: 'Aurora' }
        },
        rayDensity: {
            default: 20,
            description: 'Density of vertical ray structures',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Ray Detail', group: 'Aurora' }
        },
        height: {
            default: 120,
            description: 'How tall the aurora extends',
            ui: { type: ['range', 'map'], min: 10, max: 200, step: 1, label: 'Height', group: 'Scene' }
        },
        center: {
            default: { x: 0.5, y: 0 },
            transform: transformPosition,
            description: 'Center position of the aurora',
            ui: { type: 'position', label: 'Center', group: 'Scene' }
        },
        seed: {
            default: 0,
            description: 'Random seed for variation',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Aurora' }
        }
    },
    // The Aurora look: a wavy line across the sky, a soft curtain hung around it whose height
    // flickers with vertical rays, stacked four times with fading weights, colored A → B → C
    // by height and glowing by the stack's accumulated light. The data at the top of this
    // file is the look; the words are the language.
    paint: (params) => {
        const {uv, viewport} = surfaceOf(params)
        // Guarded aspect (a zero-height canvas frame must not render NaN).
        const aspect = local(div(viewport.member('x'), max(viewport.member('y'), 1e-6)), 'aspect')
        const t = local(animatedTime(params), 'auroraTime')
        const seed = uniformOf(p('seed'), params)
        const center = uniformOf(p('center'), params)
        const centerX = center.member('x')
        const centerY = sub(1, center.member('y'))
        // Prop conditioning: sliders are 0–100/0–200 ranges, the field works in unit terms.
        const wav = mul(uniformOf(p('waviness'), params), 0.01)
        const rayAmt = mul(uniformOf(p('rayDensity'), params), 0.01)
        const auroraHeight = mul(uniformOf(p('height'), params), 0.01)
        const intens = mul(uniformOf(p('intensity'), params), 0.018)
        const balanceExp = mix(2.5, 0.4, mul(uniformOf(p('balance'), params), 0.01))

        // One curtain (an authoring-time mixin — emitted per layer). Yields {colorT, alpha}.
        const curtain = (layerIdx: number, timeOffset: number): {colorT: Expr; alpha: Expr} => {
            const tSlow = local(add(mul(t, 0.06), timeOffset), 'tSlow')
            const lSeed = local(add(seed, Math.fround(layerIdx * Math.fround(31.7))), 'lSeed')
            const wx = local(mul(mul(sub(uv.member('x'), centerX), aspect), 2), 'wx')

            // The path: a wavy line, wobbled by THE single noise read per curtain (path,
            // rays and section brightness all share this one sample).
            const arc = wavyLine({along: wx, time: tSlow, waves: ARC_WAVES.map((w) => ({
                ...w, phase: hashPhase(lSeed, w.salt),
            }))})
            const noiseWarp = local(volumeNoiseAt(vec3(mul(wx, 0.5), mul(tSlow, 0.25), mul(lSeed, 0.1))), 'curtainNoise')
            const pathY = add(add(centerY, mul(arc, wav)), mul(mul(noiseWarp, 0.14), wav))

            // Vertical rays flicker the curtain's height.
            const rays = local(rayBands({
                along: add(wx, mul(noiseWarp, 2.5)),
                frequency: mul(mix(6, 20, rayAmt), 0.15),
                time: tSlow,
                speed: 0.1,
                phase: hashPhase(lSeed, RAY_SALT),
                sharpness: 1.5,
            }), 'curtainRays')
            const localHeight = mul(auroraHeight, mix(0.4, 1.0, rays))

            // The curtain: a soft band hung around the path — sharp fade-in below, long
            // ray-flickered fade-out above.
            const distFromPath = local(sub(uv.member('y'), pathY), 'distFromPath')
            const topExtent = local(mul(localHeight, add(mul(rays, 0.4), 0.6)), 'topExtent')
            const envelope = softBand({
                distance: distFromPath,
                below: {from: -0.04, to: 0.015},
                above: {to: topExtent},
            })

            // Height through the band drives color; brightness sections along the path and
            // boosts near the core.
            const colorT = local(clamp(div(distFromPath, add(topExtent, 0.001)), 0, 1), 'colorT')
            const sectionBright = mix(0.4, 1.0, add(mul(noiseWarp, 0.5), 0.5))
            const coreD = sub(colorT, 0.2)
            const coreBright = add(mul(exp(mul(mul(coreD, coreD), -6)), 0.3), 1)

            const alpha = local(mul(mul(mul(envelope, rays), sectionBright), coreBright), 'curtainAlpha')
            return {colorT, alpha}
        }

        // Stack the curtains — curtainCount is compile-time, so disabled layers aren't emitted
        // (a gated layer contributed exactly 0).
        const count = Math.max(1, Math.min(4, Math.round((params.propValues.curtainCount as number) ?? 4)))
        const {weightedSum, totalWeight} = layered(LAYERS.slice(0, count), ({timeOffset, weight}, layerIdx) => {
            const {colorT, alpha} = curtain(layerIdx, timeOffset)
            return {value: colorT, weight: weight === 1 ? alpha : mul(alpha, weight)}
        })
        const total = local(totalWeight, 'auroraAlpha')

        // Balance remap: the weighted-mean height parameter, bent by the balance exponent,
        // split into the two ladder factors; brightness = accumulated light × intensity.
        const avgT = local(pow(div(weightedSum, add(total, 0.001)), balanceExp), 'auroraAvgT')
        const ladder = colorLadder3({a: p('colorA'), b: p('colorB'), c: p('colorC'), space: p('colorSpace')})
        const color = ladder(smoothstep(0, 0.2, avgT), smoothstep(0.4, 0.85, avgT), params)

        // Glow compose: scale by brightness (clamped), alpha from the glow itself.
        const brightness = local(mul(total, intens), 'auroraGlow')
        return vec4(
            clamp(mul(color.member('rgb'), brightness), splat3(0), splat3(1)),
            clamp(mul(brightness, 0.5), 0, 1))
    },
})

export default componentDefinition
