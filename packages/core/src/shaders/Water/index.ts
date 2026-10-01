import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, marchedNormal, fieldSlope, flowWarp, keyLightAt,
    grazingFlat, schlickFresnel, opticalThickness, beerLambert, sharpGlint, surfaceNoiseAt,
    reflect, refract, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {direction} from "@coreroot/std/frames"
import {
    abs, add, clamp, div, dot, float, length, local, mix, mul, normalize, pow,
    sin, smoothstep, splat3, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// Three-octave value noise for the wave height field — big swells plus mid + fine ripple bands.
// mxNoiseFloat2 is ~[-1,1]; the final ×0.52 keeps it there.
const waterFbm = (uvE: Expr, hint: string): Expr => {
    const uv = local(uvE, hint)
    const n0 = surfaceNoiseAt(uv)
    const n1 = mul(surfaceNoiseAt(add(mul(uv, 2.07), vec2(4.3, 9.1))), 0.5)
    const n2 = mul(surfaceNoiseAt(add(mul(uv, 4.13), vec2(1.7, 8.2))), 0.25)
    return mul(add(add(n0, n1), n2), 0.52)
}

// Single-octave noise for finite-difference gradient taps (the upper octaves' derivatives average
// out under the relief multiplier — dropping them is imperceptible + saves noise calls).
const waterNoise1 = (uv: Expr): Expr => mul(surfaceNoiseAt(uv), 0.52)

// Procedural sky driving both the reflection and the see-through transmission: a vertical
// gradient (dark below horizon → pale at it → blue overhead) with a sun disc steered by the light
// angle and tightened by sharpness. Rotated into the env frame. Callers pass local()'d args.
const waterSky = (rx: Expr, ry: Expr, lx: Expr, ly: Expr, envC: Expr, envS: Expr, sharp: Expr, hint: string): Expr => {
    const ex = local(sub(mul(rx, envC), mul(ry, envS)), `${hint}Ex`)
    const ey = local(add(mul(rx, envS), mul(ry, envC)), `${hint}Ey`)
    const up = local(clamp(add(mul(ey, 0.5), 0.5), 0, 1), `${hint}Up`)
    const horizon = vec3(0.90, 0.96, 1.0)
    const zenith = vec3(0.20, 0.46, 0.85)
    const below = vec3(0.04, 0.10, 0.16)
    const aboveCol = mix(horizon, zenith, splat3(smoothstep(0.5, 1.0, up)))
    const base = mix(below, aboveCol, splat3(smoothstep(0.40, 0.55, up)))
    const sunCenter = vec2(mul(lx, 0.8), add(mul(ly, 0.8), 0.12))
    const sunRad = local(mix(0.5, 0.13, sharp), `${hint}SunRad`)
    const sd = length(sub(vec2(ex, ey), sunCenter))
    const sun = smoothstep(sunRad, mul(sunRad, 0.15), sd)
    return add(base, mul(vec3(1.0, 0.97, 0.88), mul(sun, 1.5)))
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    waterColor: Parameters<typeof transformColor>[0]
    clarity: number
    depth: number
    shallows: number
    choppiness: number
    waveScale: number
    swirl: number
    speed: number
    caustics: number
    causticScale: number
    reflection: number
    envRotation: number
    sharpness: number
    lightAngle: number
    foam: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Water",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Translucent water — a refractive, light-absorbing body that deepens to its color with thickness, wrapped in a wind-driven wave surface that reflects a procedural sky, with Fresnel-bright grazing edges and foam breaking on the crests and shoreline",
    animatedTime: {speed: 'speed'},
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the water shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the water shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the water shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        waterColor: {
            default: '#00bfeb',
            transform: transformColor,
            description: 'The color the water saturates toward with depth',
            ui: { type: 'color', label: 'Water Color', group: 'Water' }
        },
        clarity: {
            default: 0.4,
            description: 'How much of the reflected sky transmits through the body',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Clarity', group: 'Water' }
        },
        depth: {
            default: 5,
            description: 'How quickly the water darkens and saturates toward its color with thickness',
            ui: { type: 'range', min: 0, max: 8, step: 0.01, label: 'Depth', group: 'Water' }
        },
        shallows: {
            default: 0.5,
            description: 'Brightness of the thin shallow water at the edges — high fades the rim to light translucent water, low keeps the edges close to the deep water color',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Shallows', group: 'Water' }
        },
        caustics: {
            default: 0.15,
            description: 'Intensity of the light caustics bubbling up through the body — bright veins of focused light churning inside the water',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Caustics', group: 'Water' }
        },
        causticScale: {
            default: 6,
            description: 'Size of the caustic veins — higher = finer, busier light patterns inside the body',
            ui: { type: 'range', min: 1, max: 20, step: 0.1, label: 'Caustic Scale', group: 'Water' }
        },
        choppiness: {
            default: 0.8,
            description: 'Height of the wind-driven waves — the depth of the folds that bend the reflection',
            ui: { type: 'range', min: 0, max: 1.5, step: 0.01, label: 'Choppiness', group: 'Waves' }
        },
        waveScale: {
            default: 3.5,
            description: 'Scale of the wave fronts — higher = smaller, busier ripples',
            ui: { type: 'range', min: 0.5, max: 8, step: 0.01, label: 'Wave Scale', group: 'Waves' }
        },
        swirl: {
            default: 0.6,
            description: 'Swirl of the flow — domain-warps the wave fronts into curling eddies',
            ui: { type: 'range', min: 0, max: 1.5, step: 0.01, label: 'Swirl', group: 'Waves' }
        },
        speed: {
            default: 0.5,
            description: 'Speed of the rolling waves. 0 pauses the surface.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Waves' }
        },
        reflection: {
            default: 2,
            description: 'Strength of the reflected sky across the surface',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Reflection', group: 'Surface' }
        },
        envRotation: {
            default: 0,
            description: 'Rotates the reflected sky — spins where the bright sky and sun fall',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Env Rotation', group: 'Surface' }
        },
        sharpness: {
            default: 0.1,
            description: 'Tightness of the sun glint and reflected sun',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Sharpness', group: 'Surface' }
        },
        lightAngle: {
            default: 30,
            description: 'Direction of the sun, in degrees — drives the glint and where the sun reflects',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Surface' }
        },
        foam: {
            default: 0.3,
            description: 'White foam breaking on the wave crests and along the shoreline edge',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Foam', group: 'Surface' }
        },
        edgeSoftness: {
            default: 0.05,
            description: 'Softness of the shape boundary edge',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Surface' }
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: { type: 'shape', label: 'Shape', group: 'Shape' }
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type'
        }
    },

    // The shape-effect spine: 'raw' surface-pattern coords (the waves ride the shape surface) +
    // sampler routing + the forward 3-tap field frame. The translucent-water material as algebra:
    // geometric normal + wind-driven wave relief + reflected/transmitted procedural sky +
    // Beer–Lambert absorbing body + internal caustics + Fresnel reflection + sun glint + foam,
    // all inside the shape guard so outside pixels pay only the condition.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)
            const vol = frame.volumetric

            const field = surfaceField(frame, params, {scale: u.scale})
            const {sdf, pxH} = field

            const light = direction(u.lightAngle, 'light')
            const lx = light.member('x')
            const ly = light.member('y')
            const env = direction(u.envRotation, 'env')
            const envC = env.member('x')
            const envS = env.member('y')

            // ── Field gradient + geometric normal (the volumetric/flat split is build-time) ──
            const grad = fieldSlope(field)
            let nGeo: Expr
            if (vol) {
                nGeo = marchedNormal(field)
            } else {
                const rimT = clamp(div(mul(sdf, -1), 0.05), 0, 1)
                const rimBevel = local(mul(sub(1, rimT), 1.6), 'rimBevel')
                nGeo = local(normalize(vec3(mul(grad.member('x'), rimBevel), mul(grad.member('y'), rimBevel), -1)), 'nGeo')
            }

            // ── Wind-driven wave relief ──────────────────────────────────────────────
            const pat = surfacePattern(frame, field)
            const fx = pat.member('x')
            const fy = pat.member('y')
            const freq = local(mul(u.waveScale, 2.4), 'freq')
            const flowT = local(mul(t, 0.2), 'flowT')
            const q = local(vec2(mul(mul(fx, freq), 0.8), mul(mul(fy, freq), 1.25)), 'q')
            const wAmt = mul(u.swirl, 0.9)
            // The shared flow-warp part (the same table LiquidMetal's molten folds read).
            const w = flowWarp(mul(q, 0.6), flowT)
            const qw = local(add(add(q, mul(w, wAmt)), vec2(0, mul(flowT, -1))), 'qw')
            const re = 0.06
            const h0 = local(waterFbm(qw, 'fbmH0'), 'h0')
            const h0n1 = local(waterNoise1(qw), 'h0n1')
            const hx = clamp(div(sub(waterNoise1(add(qw, vec2(re, 0))), h0n1), re), -4, 4)
            const hy = clamp(div(sub(waterNoise1(add(qw, vec2(0, re))), h0n1), re), -4, 4)
            const fineRe = 0.025
            const qf = local(add(add(mul(qw, 3.3), vec2(mul(flowT, 1.7), mul(flowT, -1.15))), vec2(11.3, 4.7)), 'qf')
            const f0n1 = local(waterNoise1(qf), 'f0n1')
            const fxg = clamp(div(sub(waterNoise1(add(qf, vec2(fineRe, 0))), f0n1), fineRe), -4, 4)
            const fyg = clamp(div(sub(waterNoise1(add(qf, vec2(0, fineRe))), f0n1), fineRe), -4, 4)
            const relief = local(mul(u.choppiness, 0.5), 'relief')
            const reliefFine = local(mul(u.choppiness, 0.22), 'reliefFine')
            const n = local(normalize(vec3(
                sub(sub(nGeo.member('x'), mul(hx, relief)), mul(fxg, reliefFine)),
                sub(sub(nGeo.member('y'), mul(hy, relief)), mul(fyg, reliefFine)),
                nGeo.member('z'),
            )), 'n')

            const sharp = local(clamp(u.sharpness, 0, 1), 'sharp')

            // ── Reflected view ray → sky ─────────────────────────────────────────────
            const R = local(reflect(vec3(0, 0, 1), n), 'R')
            const reflCol = local(mul(waterSky(R.member('x'), R.member('y'), lx, ly, envC, envS, sharp, 'refl'), u.reflection), 'reflCol')

            // ── Absorbing, refractive body: the shared optical-thickness reading of the
            //    field + Beer–Lambert transmission, absorption derived from the water color ──
            const thickness = local(mul(opticalThickness(field, frame), u.depth), 'thickness')
            const depthT = local(clamp(mul(thickness, 1.3), 0, 1), 'depthT')
            const wc = local(u.waterColor.member('rgb'), 'wc')
            const absorb = vec3(
                mul(sub(1, wc.member('x')), 2.2),
                mul(sub(1, wc.member('y')), 2.2),
                mul(sub(1, wc.member('z')), 2.2),
            )
            const transmit = local(beerLambert(thickness, absorb), 'transmit')
            const shallowsC = clamp(u.shallows, 0, 1)
            const shallowCol = mix(wc, vec3(1, 1, 1), splat3(shallowsC))
            const bodyTint = local(add(mul(mix(shallowCol, wc, splat3(depthT)), transmit), mul(wc, mul(depthT, 0.18))), 'bodyTint')

            const kl = keyLightAt(light, -0.7)
            const L = kl.member('L')
            const ndl = dot(n, L)
            const diffuse = local(add(mul(pow(clamp(add(mul(ndl, 0.5), 0.5), 0, 1), 0.8), 0.5), 0.5), 'diffuse')

            const eta = 1.0 / 1.333
            const tDir = local(refract(vec3(0, 0, 1), n, float(eta)), 'tDir')
            const transSky = mul(waterSky(tDir.member('x'), tDir.member('y'), lx, ly, envC, envS, sharp, 'trans'), transmit)
            const clarityT = mul(clamp(u.clarity, 0, 1), sub(1, mul(depthT, 0.8)))
            const body0 = mul(mix(bodyTint, add(mul(transSky, 0.6), mul(bodyTint, 0.4)), splat3(clarityT)), diffuse)

            // ── Internal caustics ────────────────────────────────────────────────────
            const causT = local(mul(t, 3), 'causT')
            const causP = local(add(mul(vec2(fx, fy), u.causticScale), vec2(mul(sin(mul(causT, 0.15)), 0.4), mul(mul(causT, 0.3), -1))), 'causP')
            const c1 = sub(1, abs(waterFbm(causP, 'fbmC1')))
            const c2 = sub(1, abs(waterFbm(add(mul(causP, 1.9), vec2(3.7, mul(mul(causT, 0.18), -1))), 'fbmC2')))
            const causV = local(pow(clamp(add(mul(c1, 0.6), mul(c2, 0.4)), 0, 1), 3), 'causV')
            const causMask = local(mul(add(mul(depthT, 0.8), 0.2), u.caustics), 'causMask')
            const body1 = mul(body0, add(1, mul(mul(causV, causMask), 0.7)))
            const body2 = add(body1, mul(vec3(0.55, 0.82, 1.0), mul(mul(causV, causMask), 0.45)))

            // ── Fresnel reflection mix + sun glint + foam ────────────────────────────
            const fresMix = clamp(schlickFresnel(grazingFlat(n), {gain: 0.96}), 0, 1)
            const surface0 = mix(body2, reflCol, splat3(fresMix))

            const ndh = clamp(dot(n, kl.member('H')), 0, 1)
            const spec = sharpGlint(ndh, sharp, vec4(40, 600, 0.3, 1.3))
            const surface1 = add(surface0, mul(vec3(1.0, 0.98, 0.92), spec))

            const crestH = clamp(add(mul(h0, 0.5), 0.5), 0, 1)
            const crest = smoothstep(0.62, 0.92, crestH)
            const shore = sub(1, smoothstep(0, 0.045, mul(sdf, -1)))
            const foamMask = local(mul(clamp(add(crest, mul(shore, 0.7)), 0, 1), u.foam), 'foamMask')
            const surface2 = local(mix(surface1, vec3(1, 1, 1), splat3(mul(foamMask, 0.85))), 'waterRgb')

            const transition = silhouette(field, u.edgeSoftness)
            return guarded(insideShape(sdf, pxH), vec4(surface2, transition), ZERO, 'water')
        },
    }),
})

export default componentDefinition
