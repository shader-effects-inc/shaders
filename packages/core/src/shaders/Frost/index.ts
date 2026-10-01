import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {call, ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, fieldSlope, nudgeNormal, keyLightAt, grazingFlat,
    opticalThickness, beerLambert, surfaceNoise, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {direction} from "@coreroot/std/frames"
import {
    add, clamp, div, dot, float, gt, local, max, mix, mul, normalize, pow, sin, smoothstep,
    splat3, sqrt, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import type {Expr} from "@coreroot/gpu/porters"
import {noise} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    iceColor: Parameters<typeof transformColor>[0]
    gloss: number
    ambient: number
    edgeSoftness: number
    density: number
    absorption: number
    scatter: number
    frostAmount: number
    frostDepth: number
    frostScale: number
    frostRoughness: number
    sparkle: number
    fresnel: number
    lightAngle: number
    speed: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Frost",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Photoreal frozen ice — thickness-driven subsurface scattering that reads as a solid block, with fine frost crystals creeping in from the edges",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine: 'raw' surface-pattern coords (the crystal noise reads .g/.b) +
    // 'firstLobe' chord (first-entry → FIRST-exit), NOT the default 'span' (→ last-exit): Frost
    // drives Beer–Lambert absorption from the chord (.r), so 'span' would count internal air gaps
    // (e.g. an extruded logo's hole) as ice and step the thickness at the rear silhouette — showing
    // a spurious dark band of the INTERNAL structure through the frosted front face. 'firstLobe'
    // measures front-wall thickness only, staying continuous across holes (see Glass/ThinFilm).
    //
    // The frozen-ice material as algebra: subsurface Beer–Lambert absorption + rim frost crystals
    // (a guarded region — the noise taps run only where the frost field is live) + sparkle +
    // surface shading, all inside the shape guard so outside pixels pay only the condition.
    ...shapedSurface({
        pattern: 'raw',
        chord: 'firstLobe',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)
            const vol = frame.volumetric
            const EPS = 0.01

            const field = surfaceField(frame, params, {scale: u.scale})
            const {s0, sX, sY, sdf, pxH} = field
            const grad = fieldSlope(field)
            const transition = silhouette(field, u.edgeSoftness)

            // Pattern coordinates: volumetric = raw surface coords (.g/.b); flat = sdfUV − 0.5.
            const pat = surfacePattern(frame, field)
            const pdx = pat.member('x')
            const pdy = pat.member('y')

            // Base surface normal + steepness (the volumetric/flat split is build-time).
            const depthNorm = local(clamp(div(mul(sdf, -1), max(0.3, 0.01)), 0, 1), 'depthNorm')
            let nBase: Expr
            let steepAtten: Expr
            if (vol) {
                const ddx = local(clamp(div(sub(sX!.member('w'), s0.member('w')), EPS), -4, 4), 'ddx')
                const ddy = local(clamp(div(sub(sY!.member('w'), s0.member('w')), EPS), -4, 4), 'ddy')
                nBase = local(normalize(vec3(ddx, ddy, -1)), 'nBase')
                const steepness = clamp(add(mul(ddx, ddx), mul(ddy, ddy)), 0, 2)
                steepAtten = local(div(1, add(1, mul(steepness, 2))), 'steepAtten')
            } else {
                const bevelSlope = local(mul(sqrt(max(sub(1, mul(depthNorm, depthNorm)), 0)), 1.8), 'bevelSlope')
                nBase = local(normalize(vec3(mul(grad.member('x'), bevelSlope), mul(grad.member('y'), bevelSlope), -1)), 'nBase')
                steepAtten = float(1)
            }

            // Optical thickness (the "solid ice" cue) — the shared field-thickness part.
            const thickness = local(mul(opticalThickness(field, frame), u.density), 'thickness')
            const depthT = local(clamp(mul(thickness, 1.2), 0, 1), 'depthT')

            // Subsurface: Beer–Lambert absorption (red absorbs fast → blue core).
            const absorb = mul(vec3(1.6, 0.85, 0.45), u.absorption)
            const transmit = local(beerLambert(thickness, absorb), 'transmit')
            const scatteredRgb = local(mul(mix(vec3(1, 1, 1), u.iceColor.member('rgb'), splat3(depthT)), transmit), 'scatteredRgb')

            // Frost crystals: edge-concentrated, feathery + sparkly — the noise taps and Worley
            // cracks run only where the frost field is live (a guarded region).
            const frostDepthU = max(mul(u.frostDepth, 0.3), 0.01)
            const edgeMask = sub(1, smoothstep(0, frostDepthU, mul(sdf, -1)))
            const frostField = local(mul(edgeMask, u.frostAmount), 'frostField')
            const frostScale = local(mul(u.frostScale, 60), 'frostScl')
            const feps = 0.004
            const fn0 = local(surfaceNoise(mul(pdx, frostScale), mul(pdy, frostScale)), 'fn0')
            const fnx = div(sub(surfaceNoise(mul(add(pdx, feps), frostScale), mul(pdy, frostScale)), fn0), feps)
            const fny = div(sub(surfaceNoise(mul(pdx, frostScale), mul(add(pdy, feps), frostScale)), fn0), feps)
            const frostNormalAmp = local(mul(mul(frostField, u.frostRoughness), steepAtten), 'frostNormalAmp')
            const wScale = mul(u.frostScale, 40)
            const wc = call(noise.mxWorleyNoiseFloat2Pub, 'mxWorleyNoiseFloat2Pub', [vec2(mul(pdx, wScale), mul(pdy, wScale)), float(1)])
            const frostFx = guarded(
                gt(frostField, 0),
                vec3(mul(mul(fnx, frostNormalAmp), 0.03), mul(mul(fny, frostNormalAmp), 0.03), mul(smoothstep(0.25, 0, wc), frostField)),
                vec3(0, 0, 0),
                'frostFx',
                [pat, frostField, frostScale, steepAtten],
            )
            const fx3 = local(frostFx, 'fx3')
            const n = nudgeNormal(nBase, fx3.member('x'), fx3.member('y'))

            // Lighting — the shared key-light frame (Frost's elevation is −0.6).
            const kl = keyLightAt(direction(u.lightAngle, 'light'), -0.6)
            const L = kl.member('L')
            const ndl = local(dot(n, L), 'ndl')
            const ndh = local(clamp(dot(n, kl.member('H')), 0, 1), 'ndh')

            const wrap = add(mul(ndl, 0.5), 0.5)
            const sssDiffuse = pow(wrap, 0.7)
            const backLight = clamp(add(mul(dot(L, vec3(0, 0, 1)), 0.5), 0.5), 0, 1)
            const forwardGlow = mul(mul(mul(transmit, backLight), u.scatter), sub(1, mul(depthT, 0.4)))
            const sssRgb = add(mul(scatteredRgb, add(mul(sssDiffuse, 0.6), 0.5)), mul(forwardGlow, 0.5))

            // Tiny twinkling crystal glints concentrated in the rim frost (guarded like the crystals).
            const glintField = local(add(mul(surfaceNoise(add(mul(pdx, 190), 13.7), sub(mul(pdy, 190), 4.3)), 0.5), 0.5), 'glintField')
            const twinkle = add(mul(sin(add(mul(t, 3), mul(glintField, 40))), 0.5), 0.5)
            const sparkleGlint = guarded(
                gt(frostField, 0),
                mul(mul(mul(smoothstep(0.72, 0.96, mul(glintField, twinkle)), add(mul(pow(ndh, 6), 0.6), 0.4)), frostField), u.sparkle),
                float(0),
                'sparkleGlint',
                [pat, frostField, ndh],
            )
            const frostRgb = mul(vec3(0.95, 0.98, 1.0), add(mul(fx3.member('z'), 0.4), mul(sparkleGlint, 4)))

            // Surface shading.
            const grazing = grazingFlat(n)
            const fres = mul(pow(grazing, 3), u.fresnel)
            const fresnelRgb = mul(vec3(0.85, 0.93, 1.0), mul(fres, 1.5))
            const broadSpec = local(mul(mul(pow(ndh, 16), 0.4), u.gloss), 'broadSpec')
            const ambient = mul(vec3(0.40, 0.46, 0.55), u.ambient)

            const iceRgb = local(add(add(add(add(sssRgb, ambient), splat3(broadSpec)), fresnelRgb), frostRgb), 'iceRgb')
            return guarded(insideShape(sdf, pxH), vec4(iceRgb, transition), ZERO, 'frost')
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
            description: 'Center position of the ice shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the ice shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the ice shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        iceColor: {
            default: '#0597fc',
            transform: transformColor,
            description: 'Ice tint — the color deep ice saturates toward (thin edges stay near white)',
            ui: { type: 'color', label: 'Ice Color', group: 'Ice' }
        },
        ambient: {
            default: 0,
            description: 'Cool ambient fill so shadowed ice never goes black',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Ambient', group: 'Ice' }
        },
        edgeSoftness: {
            default: 0.05,
            description: 'Softness of the shape boundary edge',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Ice' }
        },
        density: {
            default: 5,
            description: 'Optical depth multiplier — how thick and deep the ice reads',
            ui: { type: 'range', min: 0, max: 10, step: 0.01, label: 'Density', group: 'Subsurface' }
        },
        absorption: {
            default: 1,
            description: 'Beer–Lambert absorption strength — higher drives a deeper blue-teal core',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Absorption', group: 'Subsurface' }
        },
        scatter: {
            default: 0.5,
            description: 'Forward back-lit glow transmitted through thin ice',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Scatter', group: 'Subsurface' }
        },
        frostAmount: {
            default: 0.8,
            description: 'Overall density of the rim frost crystals',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Frost', group: 'Frost' }
        },
        frostDepth: {
            default: 0.4,
            description: 'How far the frost creeps inward from the rim',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Frost Depth', group: 'Frost' }
        },
        frostScale: {
            default: 1,
            description: 'Crystal size — higher = finer, busier frost',
            ui: { type: 'range', min: 0.3, max: 4, step: 0.01, label: 'Frost Scale', group: 'Frost' }
        },
        frostRoughness: {
            default: 3,
            description: 'Feathery roughness of the frost crystal surface',
            ui: { type: 'range', min: 1, max: 5, step: 0.01, label: 'Frost Roughness', group: 'Frost' }
        },
        sparkle: {
            default: 0.5,
            description: 'Tiny bright crystal glints catching the light at the rim',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Sparkle', group: 'Frost' }
        },
        gloss: {
            default: 0.6,
            description: 'Surface sheen — the broad soft icy specular',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Gloss', group: 'Surface' }
        },
        fresnel: {
            default: 0.02,
            description: 'Cold rim glow on grazing surfaces',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Fresnel', group: 'Surface' }
        },
        lightAngle: {
            default: 300,
            description: 'Key light direction in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Surface' }
        },
        speed: {
            default: 0.3,
            description: 'Sparkle twinkle speed. 0 freezes the sparkle.',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Speed', group: 'Animation' }
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
