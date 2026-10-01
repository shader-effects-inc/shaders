import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, fieldSlope, surfaceNoise, cosineRainbow, silhouette,
    guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {
    add, clamp, div, dot, float, local, mix, mul, normalize, pow, sin, smoothstep,
    splat3, sub, vec3, vec4,
} from "@coreroot/std/math"
import {transformPosition} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"


const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })
const THIRD = 1 / 3
const TWOTHIRD = 2 / 3
// Fixed studio light direction (300°) — the look comes from the surface + animated spectrum, not a
// steerable light, so this isn't a prop. Folded to fractional literals.
const LIGHT_X = Math.cos((300 * Math.PI) / 180)
const LIGHT_Y = Math.sin((300 * Math.PI) / 180)
// The key-light frame's L: normalize(LIGHT_X, LIGHT_Y, −0.9) pre-folded at f32 — these exact
// literals are the folded values the lighting was tuned against (an f64 refold lands one ulp off
// on x). H stays a runtime normalize.
const L_X = 0.37164708971977234
const L_Y = -0.6437116265296936
const L_Z = -0.6689647436141968

// Cosine rainbow palette — smooth spectral cycle, the signature diffraction-grating holo look.
// t wraps every 1.0. Phases stay 1/3 and 2/3 exactly — thinFilm's copy uses the truncated
// 0.3333/0.6667, so `cosinePalette` takes them as arguments rather than unifying (a pixel change).
const holoRainbow = (t: Expr): Expr => cosineRainbow(t, [0, THIRD, TWOTHIRD])

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    hueShift: number
    foilScale: number
    saturation: number
    roughness: number
    speed: number
    crinkle: number
    crinkleScale: number
    sparkle: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Holographic",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Iridescent holographic foil sticker with animated rainbow sheen and glitter flakes",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine: 'raw' surface-pattern coords + the forward field frame at the foil's
    // own eps 0.008 stencil. The holographic-foil material as algebra: surface normal + laminate
    // wrinkles + diffraction hue (cosine rainbow) + metallic sheen + glitter flakes + grain, all
    // inside the shape guard so outside pixels pay only the condition. Macro fields
    // (blotch/wrinkle) use surface-locked pattern coords in volumetric; micro fields (flakes ride
    // the pattern, grain stays screen-stable in shape space).
    ...shapedSurface({
        pattern: 'raw',
        stencil: {kind: 'forward', eps: 0.008},
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = local(animatedTime(params), 'animTime')
            const vol = frame.volumetric
            const EPS = 0.008

            const field = surfaceField(frame, params, {scale: u.scale})
            const {s0, sX, sY, sdf, pxH} = field
            const transition = silhouette(field, u.edgeSoftness)

            const suv = local(frame.sdfUV!, 'suv')
            const pat = surfacePattern(frame, field, {uv: suv})
            const pdx = pat.member('x')
            const pdy = pat.member('y')
            const qdx = sub(suv.member('x'), 0.5)
            const qdy = sub(suv.member('y'), 0.5)

            // Surface normal (2D pillow vs volumetric view-depth reconstruction) — build-time split.
            let baseNx: Expr
            let baseNy: Expr
            let steepAtten: Expr
            if (vol) {
                const bx = local(clamp(div(sub(sX!.member('w'), s0.member('w')), EPS), -1.5, 1.5), 'baseNxVol')
                const by = local(clamp(div(sub(sY!.member('w'), s0.member('w')), EPS), -1.5, 1.5), 'baseNyVol')
                baseNx = bx
                baseNy = by
                const steepness = clamp(add(mul(bx, bx), mul(by, by)), 0, 2)
                steepAtten = local(div(1, add(1, mul(steepness, 2))), 'steepAtten')
            } else {
                const grad = fieldSlope(field, EPS)
                const edgeT = clamp(div(mul(sdf, -1), 0.06), 0, 1)
                const domeSlope = local(mul(sub(1, edgeT), 0.35), 'domeSlope')
                baseNx = local(mul(grad.member('x'), domeSlope), 'baseNx')
                baseNy = local(mul(grad.member('y'), domeSlope), 'baseNy')
                steepAtten = float(1)
            }

            // Laminate wrinkles (attenuated on steep 3D walls).
            const crScale = local(mul(u.crinkleScale, 1.35), 'crScale')
            const crAmp = local(mul(mul(u.crinkle, 0.8), steepAtten), 'crAmp')
            const neps = 0.02
            const w0 = local(surfaceNoise(add(mul(pdx, crScale), 2.7), add(mul(pdy, crScale), 11.3)), 'w0')
            const wx = div(sub(surfaceNoise(add(mul(add(pdx, neps), crScale), 2.7), add(mul(pdy, crScale), 11.3)), w0), neps)
            const wy = div(sub(surfaceNoise(add(mul(pdx, crScale), 2.7), add(mul(add(pdy, neps), crScale), 11.3)), w0), neps)
            const n = local(normalize(vec3(
                add(baseNx, mul(mul(wx, crAmp), 0.4)),
                add(baseNy, mul(mul(wy, crAmp), 0.4)),
                -1,
            )), 'n')

            // Diffraction hue.
            const fScale = local(mul(u.foilScale, 2.2), 'fScale')
            const drift = local(mul(t, 0.06), 'drift')
            const blotch = add(
                surfaceNoise(add(mul(pdx, fScale), mul(drift, LIGHT_X)), add(mul(pdy, fScale), mul(drift, LIGHT_Y))),
                mul(surfaceNoise(
                    sub(add(mul(mul(pdx, fScale), 2.1), 5.2), mul(mul(drift, LIGHT_X), 1.6)),
                    sub(add(mul(mul(pdy, fScale), 2.1), 9.4), mul(mul(drift, LIGHT_Y), 1.6)),
                ), 0.4),
            )
            const normalTilt = add(mul(n.member('x'), LIGHT_X), mul(n.member('y'), LIGHT_Y))
            const ramp = mul(add(mul(pdx, LIGHT_X), mul(pdy, LIGHT_Y)), 0.8)
            const hue = local(add(add(add(add(mul(blotch, 0.45), ramp), mul(normalTilt, 1.4)), mul(t, 0.18)), div(u.hueShift, 360)), 'hue')

            const saturationC = clamp(u.saturation, 0, 1)
            const spectral = holoRainbow(hue)
            const silver = vec3(0.72, 0.73, 0.76)
            const foilTint = local(mix(silver, spectral, splat3(mul(saturationC, 0.85))), 'foilTint')

            // Lighting on the foil — the pre-folded fixed key light (see the L_* constants above).
            const roughnessC = local(clamp(u.roughness, 0, 1), 'roughnessC')
            const L = local(vec3(L_X, L_Y, L_Z), 'L')
            const H = local(normalize(add(L, vec3(0, 0, -1))), 'H')
            const ndl = local(clamp(add(mul(dot(n, L), 0.5), 0.5), 0, 1), 'ndl')
            const ndh = local(clamp(dot(n, H), 0, 1), 'ndh')
            const sheen = mul(mul(pow(ndh, 7), 0.45), sub(1, mul(roughnessC, 0.35)))
            const specCore = mul(mul(pow(ndh, mix(float(42), float(12), roughnessC)), 0.85), sub(1, mul(roughnessC, 0.55)))
            const specRgb = add(mul(foilTint, sheen), mul(mix(foilTint, vec3(1, 1, 1), splat3(0.6)), specCore))

            // Glitter flakes.
            const flakeN = surfaceNoise(mul(pdx, 190), mul(pdy, 190))
            const flakePhase = local(surfaceNoise(add(mul(pdx, 190), 47.3), sub(mul(pdy, 190), 31.1)), 'flakePhase')
            const twinkle = local(add(mul(sin(add(mul(t, 2.2), mul(flakePhase, 12.6))), 0.5), 0.5), 'twinkle')
            const flakeMask = mul(mul(smoothstep(0.52, 0.62, add(mul(flakeN, 0.75), mul(twinkle, 0.3))), clamp(u.sparkle, 0, 1)), steepAtten)
            const flakeRgb = mul(mul(holoRainbow(add(add(hue, 0.33), mul(flakePhase, 0.5))), add(0.85, mul(twinkle, 0.55))), add(mul(ndl, 0.6), 0.7))

            // Roughness grain.
            const grain = surfaceNoise(mul(qdx, 420), mul(qdy, 420))
            const grainMul = add(1, mul(mul(grain, roughnessC), 0.28))

            const foilBase = add(mul(foilTint, add(mul(ndl, 0.5), 0.62)), specRgb)
            const foilRgb = local(mul(mix(foilBase, flakeRgb, splat3(flakeMask)), grainMul), 'foilRgb')

            return guarded(insideShape(sdf, pxH), vec4(foilRgb, transition), ZERO, 'holo')
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
            description: 'Center position of the foil shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the foil shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the foil shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        hueShift: {
            default: 0,
            description: 'Rotates the rainbow spectrum across the foil',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Hue Shift', group: 'Foil' }
        },
        foilScale: {
            default: 1.5,
            description: 'Size of the iridescent color patches — higher = smaller, busier patches',
            ui: { type: 'range', min: 0.3, max: 6, step: 0.01, label: 'Foil Scale', group: 'Foil' }
        },
        saturation: {
            default: 0.75,
            description: 'Rainbow saturation — 0 = plain silver foil, 1 = fully spectral',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Saturation', group: 'Foil' }
        },
        roughness: {
            default: 0.25,
            description: 'Matte laminate grain — softens the gloss with fine paper-like noise',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Roughness', group: 'Foil' }
        },
        speed: {
            default: 1,
            description: 'Animation speed — the spectrum drifts across the foil as if the sticker is tilting in your hand. 0 pauses.',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        crinkle: {
            default: 0.5,
            description: 'Laminate wrinkles — perturbs the surface so the sheen and rainbow swirl locally instead of lying flat',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Crinkle', group: 'Foil' }
        },
        crinkleScale: {
            default: 1,
            description: 'Size of the laminate wrinkles — higher = smaller, denser wrinkles',
            ui: { type: 'range', min: 0.3, max: 6, step: 0.01, label: 'Crinkle Scale', group: 'Foil' }
        },
        sparkle: {
            default: 0.4,
            description: 'Glitter flakes embedded in the foil — tiny facets that refract a shifted hue and twinkle as the spectrum drifts',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Sparkle', group: 'Foil' }
        },
        edgeSoftness: {
            default: 0.05,
            description: 'Softness of the shape boundary edge',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Foil' }
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
