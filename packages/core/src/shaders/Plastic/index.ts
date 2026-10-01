import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {call, ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, marchedNormal, fieldSlope, nudgeNormal,
    grazingFlat, keyLightAt, reflect, exp2, surfaceNoise, surfaceNoiseAt, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {direction} from "@coreroot/std/frames"
import {
    abs, add, clamp, div, dot, local, max, mix, mul, normalize, pow, smoothstep,
    splat3, sqrt, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {colorMixing, colorStops as kitColorStops} from "@coreroot/gpu/kit"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// Soft-edged box window (1 inside [lo,hi], 0 outside, `soft` edge). The reversed-edge smoothstep
// form `smoothstep(hi+soft, hi-soft, x)` is rewritten `1 - smoothstep(hi-soft, hi+soft, x)` —
// provably identical (smoothstep is symmetric) + satisfies WGSL's edge0 < edge1.
const envBox = (x: Expr, lo: number, hi: number, softE: Expr, hint: string): Expr => {
    const s = local(softE, `${hint}S`)
    const xx = local(x, `${hint}X`)
    const a = smoothstep(sub(lo, s), add(lo, s), xx)
    const b = sub(1, smoothstep(sub(hi, s), add(hi, s), xx))
    return mul(a, b)
}

// Ridged crumple field — |noise| gradient flips at zero crossings into the hard creases of blown
// plastic. Two decorrelated octaves; `seedOff` shifts the domain (re-rolls the dents).
const crumpleField = (px: Expr, py: Expr, cScale: Expr, seedOff: Expr): Expr => {
    const p1 = vec2(add(mul(px, cScale), seedOff), add(mul(py, cScale), mul(seedOff, 0.617)))
    const p2 = vec2(
        add(add(mul(mul(px, cScale), 2.3), 13.1), seedOff),
        add(add(mul(mul(py, cScale), 2.3), 7.7), mul(seedOff, 0.617)),
    )
    return add(
        abs(surfaceNoiseAt(p1)),
        mul(abs(surfaceNoiseAt(p2)), 0.45),
    )
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    stops: ColorStop[] | null
    gradientAngle: number
    colorSpace: string
    lightAngle: number
    shading: number
    roughness: number
    reflectivity: number
    crumple: number
    crumpleScale: number
    crumpleCoverage: number
    seed: number
    speed: number
    rim: number
    rimColor: Parameters<typeof transformColor>[0]
    thickness: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Plastic",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Glossy molded plastic with photorealistic studio reflections, driven in a custom shape",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine: 'raw' surface-pattern coords + sampler routing + the forward 3-tap
    // field frame. The glossy-plastic material as algebra: pillow / 3D normal + ridged crumple
    // perturbation + procedural studio (window/mullion/strip/fill/dome) + Fresnel-weighted
    // reflection + Blinn glint + wrapped diffuse over the pre-mixed body gradient + contact AO +
    // rim light — all inside the shape guard so outside pixels pay only the condition.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u, propValues} = params
            const t = animatedTime(params)
            const vol = frame.volumetric

            const field = surfaceField(frame, params, {scale: u.scale})
            const {sdf, pxH} = field
            const sdfUV = local(frame.sdfUV!, 'sdfUV')

            // Body gradient factor (geometry-only) → pre-mixed body color: multi-stop when >1
            // active stops, else the two-color path (Swirl/Voronoi pattern). colorSpace is
            // compile-time.
            const gDir = direction(u.gradientAngle, 'gDir')
            const gT = smoothstep(0, 1, clamp(add(div(dot(sub(sdfUV.member('xy'), vec2(0.5, 0.5)), gDir), 0.7), 0.5), 0, 1))
            const colorSpaceMode = (propValues.colorSpace as number) ?? 0
            const stopCount = (propValues.stopCount as number) ?? 0
            let baseColor: Expr
            if (stopCount > 1) {
                baseColor = kitColorStops.mixColorStopsRuntime(
                    gT,
                    {
                        colorsArray: u.colorsArray,
                        positionsArray: u.positionsArray,
                        convertedColorsArray: u.convertedColorsArray,
                        stopCount: u.stopCount,
                    },
                    colorSpaceMode,
                )
            } else {
                const variant = colorMixing.mixColorsVariants[colorSpaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
                baseColor = call(variant, 'mixColors', [u.colorA, u.colorB, gT])
            }
            const baseRgb = baseColor.member('rgb')

            // Light axis drifts slowly (env orbit): envDrift = 6·animTime.
            const light = direction(add(u.lightAngle, mul(t, 6)), 'light')
            const lx = light.member('x')
            const ly = light.member('y')

            const grad = fieldSlope(field)
            const transition = silhouette(field, u.edgeSoftness)

            // Pattern coordinates (shape-locked on 3D).
            const pat = surfacePattern(frame, field, {uv: sdfUV})
            const pdx = pat.member('x')
            const pdy = pat.member('y')

            // ── Base surface normal (the volumetric/flat split is build-time) ────────
            const thicknessRange = max(mul(u.thickness, 0.3), 0.01)
            const depthNorm = local(clamp(div(mul(sdf, -1), thicknessRange), 0, 1), 'depthNorm')
            let nBase: Expr
            if (vol) {
                nBase = marchedNormal(field, 4, 'nBase')
            } else {
                const bevelSlope = local(mul(sqrt(max(sub(1, mul(depthNorm, depthNorm)), 0)), 2.4), 'bevelSlope')
                nBase = local(normalize(vec3(mul(grad.member('x'), bevelSlope), mul(grad.member('y'), bevelSlope), -1)), 'nBase')
            }

            // ── Crumple: ridged-noise normal perturbation ─────────────────────────────
            const cScale = local(mul(u.crumpleScale, 4), 'cScale')
            const cAmp = mul(u.crumple, 0.9)
            const seedOff = local(mul(u.seed, 7.31), 'seedOff')
            const neps = 0.012
            const cr0 = local(crumpleField(pdx, pdy, cScale, seedOff), 'cr0')
            const crx = div(sub(crumpleField(add(pdx, neps), pdy, cScale, seedOff), cr0), neps)
            const cry = div(sub(crumpleField(pdx, add(pdy, neps), cScale, seedOff), cr0), neps)
            const coverage = clamp(u.crumpleCoverage, 0, 1)
            const covNoise = add(mul(surfaceNoise(
                add(add(mul(pdx, 2.6), seedOff), 31.7),
                add(add(mul(pdy, 2.6), mul(seedOff, 0.617)), 17.3),
            ), 0.5), 0.5)
            const covThreshold = local(sub(1.25, mul(coverage, 1.75)), 'covThreshold')
            const covMask = smoothstep(sub(covThreshold, 0.25), add(covThreshold, 0.25), covNoise)
            const crumpleW = local(mul(mul(cAmp, covMask), add(mul(smoothstep(0, 0.35, depthNorm), 0.85), 0.15)), 'crumpleW')
            const n = nudgeNormal(nBase, mul(mul(crx, crumpleW), 0.25), mul(mul(cry, crumpleW), 0.25))

            const grazing = grazingFlat(n)

            // ── Reflection vector → procedural studio ─────────────────────────────────
            const R = local(reflect(vec3(0, 0, 1), n), 'R')
            const rough = local(clamp(u.roughness, 0, 1), 'rough')
            const grain = surfaceNoise(mul(pdx, 140), mul(pdy, 140))
            const rJit = local(mul(mul(rough, grain), 0.18), 'rJit')
            const eu = local(add(add(mul(R.member('x'), lx), mul(R.member('y'), ly)), rJit), 'eu')
            const ev = local(add(add(mul(R.member('x'), mul(ly, -1)), mul(R.member('y'), lx)), mul(rJit, 0.6)), 'ev')
            const soft = local(mix(0.02, 0.45, rough), 'soft')

            const windowBox = mul(envBox(eu, 0.18, 0.95, soft, 'win1'), envBox(ev, -0.42, 0.42, soft, 'win2'))
            const mullion = mul(mul(envBox(ev, -0.045, 0.045, mul(soft, 0.6), 'mull'), 0.8), sub(1, rough))
            const keyBank = mul(windowBox, sub(1, mullion))
            const strip = mul(mul(envBox(eu, -0.16, -0.05, mul(soft, 0.7), 'strip1'), envBox(ev, -0.85, 0.85, soft, 'strip2')), 0.85)
            const fill = mul(mul(envBox(eu, -0.95, -0.45, mul(soft, 1.6), 'fill1'), envBox(ev, -0.6, 0.6, mul(soft, 1.6), 'fill2')), 0.3)
            const dome = mul(clamp(add(mul(eu, 0.5), 0.5), 0, 1), 0.12)
            const env = mul(add(add(add(keyBank, strip), fill), dome), sub(1, mul(rough, 0.55)))

            // ── Fresnel-weighted reflection + Blinn glint ─────────────────────────────
            const fresnel = add(0.3, mul(pow(grazing, 3), 0.7))
            const specW = mul(u.reflectivity, fresnel)
            const kl = keyLightAt(light, -0.8)
            const L = kl.member('L')
            const ndh = clamp(dot(n, kl.member('H')), 0, 1)
            const shininess = exp2(mix(9, 4, rough))
            const glint = mul(mul(pow(ndh, shininess), 1.6), sub(1, mul(rough, 0.7)))
            const specular = local(mul(add(env, glint), specW), 'specular')

            // ── Body color (pre-mixed gradient) + wrapped diffuse + AO + rim ──────────
            const ndl = clamp(dot(n, L), -1, 1)
            const lit = local(add(mul(ndl, 0.5), 0.5), 'lit')
            const diffuse = mix(sub(1, mul(u.shading, 0.75)), add(1, mul(u.shading, 0.45)), lit)
            const ao = mix(0.6, 1, smoothstep(0, 1, clamp(div(mul(sdf, -1), 0.025), 0, 1)))
            const rimGlow = mul(mul(mul(pow(grazing, 3.5), u.rim), add(mul(lit, 0.6), 0.4)), 1.6)

            const plasticRgb = local(add(add(mul(mul(baseRgb, diffuse), ao), splat3(specular)), mul(u.rimColor.member('rgb'), rimGlow)), 'plasticRgb')
            return guarded(insideShape(sdf, pxH), vec4(plasticRgb, transition), ZERO, 'plastic')
        },
    }),
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the plastic shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the plastic shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the plastic shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        colorA: {
            default: '#f5f5f7',
            transform: transformColor,
            description: 'Body color at the start of the gradient',
            ui: { type: 'color', label: 'Color A', group: 'Colors' }
        },
        colorB: {
            default: '#d9dade',
            transform: transformColor,
            description: 'Body color at the end of the gradient',
            ui: { type: 'color', label: 'Color B', group: 'Colors' }
        },
        // Multi-stop gradient: when set (length > 1) these stops override colorA/colorB.
        stops: colorStopsPropConfig(),
        gradientAngle: {
            default: 90,
            description: 'Direction of the body color gradient in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Gradient Angle', group: 'Colors' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for gradient color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        },
        thickness: {
            default: 0.5,
            description: 'Puffiness — how far the rounded bevel reaches inward from the edge before the surface flattens',
            ui: { type: 'range', min: 0.05, max: 1, step: 0.01, label: 'Puffiness', group: 'Plastic' }
        },
        roughness: {
            default: 0.12,
            description: 'Surface roughness — 0 = polished gloss with crisp mirror reflections, 1 = matte satin',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Roughness', group: 'Plastic' }
        },
        reflectivity: {
            default: 1,
            description: 'Strength of the environment reflections and specular highlights',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Reflectivity', group: 'Plastic' }
        },
        crumple: {
            default: 0.25,
            description: 'Surface crumple — irregular dents that break reflections into hard organic shapes, like blown or vacuum-formed plastic',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Crumple', group: 'Plastic' }
        },
        crumpleScale: {
            default: 1,
            description: 'Size of the crumple dents — higher = smaller, denser dents',
            ui: { type: 'range', min: 0.5, max: 10, step: 0.1, label: 'Crumple Scale', group: 'Plastic' }
        },
        crumpleCoverage: {
            default: 1,
            description: 'How much of the surface is crumpled — 0 = pristine, 0.5 = patches of crumple, 1 = fully crumpled',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Crumple Coverage', group: 'Plastic' }
        },
        seed: {
            default: 0,
            description: 'Random seed — re-rolls the crumple pattern for reflection variety',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Plastic' }
        },
        edgeSoftness: {
            default: 0.05,
            description: 'Softness of the shape boundary edge',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Plastic' }
        },
        lightAngle: {
            default: 315,
            description: 'Direction the studio key light comes from, in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Lighting' }
        },
        speed: {
            default: 1,
            description: 'Speed of the environment drift — the studio reflections slowly orbit the surface. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        shading: {
            default: 0.35,
            description: 'Diffuse shading contrast — how strongly the body darkens away from the light',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Shading', group: 'Lighting' }
        },
        rim: {
            default: 0.25,
            description: 'Fresnel rim light intensity on grazing surfaces',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Rim Light', group: 'Lighting' }
        },
        rimColor: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Color of the fresnel rim light',
            ui: { type: 'color', label: 'Rim Color', group: 'Lighting' }
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

})

export default componentDefinition
