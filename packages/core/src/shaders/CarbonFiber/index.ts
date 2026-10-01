import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {ZERO} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, geometricNormal, viewRay, grazingOf, reflect,
    anisoSpecular, lightVec3, tiltAlong, surfaceNoise, studioSoftboxes, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {direction, directionFrame} from "@coreroot/std/frames"
import {
    abs, add, clamp, div, dot, float, floor, fract, ge, local, min, mix, mul,
    pow, select, sin, smoothstep, splat3, sub, vec4,
} from "@coreroot/std/math"
import {constants} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const PI = constants.PI
const TWO_PI_SEVEN = 2 * Math.PI * 7 // filament frequency (~7 filaments per tow)
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    lightColor: Parameters<typeof transformColor>[0]
    darkColor: Parameters<typeof transformColor>[0]
    weaveStyle: string
    weaveAngle: number
    weaveScale: number
    relief: number
    fiberSheen: number
    roughness: number
    clearcoat: number
    environment: number
    envRotation: number
    lightAngle: number
    bevelWidth: number
    bevelShape: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "CarbonFiber",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Photorealistic woven carbon fibre — interlaced tows whose anisotropic sheen flips ninety degrees cell to cell, raised into a quilted weave and finished with a glossy clearcoat that mirrors the studio. Plain or twill, in any shape.",
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the carbon shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the carbon shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the carbon shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        lightColor: {
            default: '#b9bdc6',
            transform: transformColor,
            description: 'The bright tone of the fibre sheen — recolor for forged/colored carbon (red, blue, bronze)',
            ui: { type: 'color', label: 'Fiber Sheen', group: 'Carbon' }
        },
        darkColor: {
            default: '#0b0c0e',
            transform: transformColor,
            description: 'The deep resin tone the weave falls to in shadow — near-black for classic carbon',
            ui: { type: 'color', label: 'Resin Color', group: 'Carbon' }
        },
        weaveStyle: {
            default: 'twill',
            compileTime: true,
            description: 'Weave pattern — plain (checkerboard) or 2×2 twill (the diagonal supercar weave)',
            ui: {
                type: 'select',
                options: [{ label: 'Twill (2×2)', value: 'twill' }, { label: 'Plain', value: 'plain' }],
                label: 'Weave',
                group: 'Carbon'
            }
        },
        weaveScale: {
            default: 30,
            description: 'Number of woven tows across the shape — higher = finer, denser weave',
            ui: { type: 'range', min: 2, max: 50, step: 0.5, label: 'Weave Scale', group: 'Carbon' }
        },
        weaveAngle: {
            default: 0,
            description: 'Rotates the whole weave grid in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Weave Angle', group: 'Carbon' }
        },
        relief: {
            default: 0.5,
            description: 'Depth of the quilted weave — how far each tow bulges, catching light on its crown and shadowing the valleys between',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Weave Depth', group: 'Carbon' }
        },
        fiberSheen: {
            default: 1.2,
            description: 'Strength of the anisotropic fibre sheen — the directional satin glint running along each tow',
            ui: { type: 'range', min: 0, max: 1.5, step: 0.01, label: 'Fiber Sheen', group: 'Carbon' }
        },
        roughness: {
            default: 0.5,
            description: 'Softness of the fibre sheen — 0 = tight crisp glints, 1 = soft matte satin',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Roughness', group: 'Surface' }
        },
        clearcoat: {
            default: 1.2,
            description: 'Strength of the glossy lacquer coat — the sharp studio reflections of the wet clearcoat over the weave',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Clearcoat', group: 'Surface' }
        },
        ...glassShellProps({
            environment: {
                default: 2,
                description: 'Strength of the reflected studio lighting seen in the clearcoat',
                ui: {group: 'Surface'},
            },
            envRotation: {ui: {group: 'Surface'}},
            lightAngle: {
                default: 215,
                description: 'Direction of the key light for the fibre sheen, in degrees',
                ui: {group: 'Surface'},
            },
        }),
        bevelWidth: {
            default: 0.05,
            description: 'Width of the edge bevel, relative to the shape — thin for machined jewellery edges, wide for soft pillowed metal',
            ui: { type: ['range', 'map'], min: 0.005, max: 0.2, step: 0.001, label: 'Bevel Width', group: 'Surface' }
        },
        bevelShape: {
            default: 0,
            description: 'Bevel profile — 0 = one smooth round fillet, 1 = machined: a steep outer fillet, a chamfer plateau and an inner knee, each catching its own line of light',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Bevel Profile', group: 'Surface' }
        },
        ...glassShellProps({
            edgeSoftness: {ui: {group: 'Surface'}},
        }),
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

    // The shape-effect spine: 'raw' surface-pattern coords (the weave rides the geometry) +
    // sampler routing + the forward 3-tap field frame. The woven-carbon material as algebra, built
    // per compile-time weave style (only the selected parity math emits): geometric normal + weave
    // layout (per-cell perpendicular tows) + tow-crown relief + fibre filaments + Ward anisotropic
    // sheen + seam AO + one crisp clearcoat tap of the shared 5-softbox studio, inside the shape
    // guard. The cell parity uses a floored `mod` (`s − m·floor(s/m)`): cell indices go negative
    // under rotation / surface coords, and a truncated mod would flip the parity sign.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u, propValues} = params
            const weaveStyle = ((propValues.weaveStyle as string) || 'twill') === 'plain' ? 'plain' : 'twill'

            const field = surfaceField(frame, params, {scale: u.scale})
            const {sdf, pxH} = field
            const nGeo = geometricNormal(frame, field, {bevelWidth: u.bevelWidth, bevelShape: u.bevelShape})

            // ── Weave layout: warp/weft in the weave's direction frame ─────────────
            const weave = directionFrame(u.weaveAngle, 'weave')
            const {along: warp, across: weft} = weave.coordsOf(surfacePattern(frame, field))

            const wA = local(mul(warp, u.weaveScale), 'wA')
            const wB = local(mul(weft, u.weaveScale), 'wB')
            const cellX = local(floor(wA), 'cellX')
            const cellY = local(floor(wB), 'cellY')
            const fA = local(sub(fract(wA), 0.5), 'fA')
            const fB = local(sub(fract(wB), 0.5), 'fB')

            // parity 0 → tow runs along the WARP axis; 1 → along the WEFT. plain = checkerboard;
            // twill = diagonal 2-tow floats (compile-time weave selection).
            let parity: Expr
            if (weaveStyle === 'plain') {
                const s2 = local(add(cellX, cellY), 's2')
                parity = local(sub(s2, mul(2, floor(div(s2, 2)))), 'parity')
            } else {
                const s4 = local(add(sub(cellX, cellY), 64), 's4')
                const m4 = sub(s4, mul(4, floor(div(s4, 4))))
                parity = local(select(ge(m4, 2), float(1), float(0)), 'parity')
            }

            // parity picks which axis the tow (and its filaments) runs along per cell.
            const fiber = local(mix(weave.tangent, weave.perp, parity), 'fiber')
            const crossAxis = local(mix(weave.perp, weave.tangent, parity), 'crossAxis')

            // ── Weave relief + fibre filaments ──────────────────────────────────────
            const acrossLocal = local(mix(fB, fA, parity), 'acrossLocal')
            const alongLocal = local(mix(warp, weft, parity), 'alongLocal')
            const reliefAmt = mul(u.relief, 0.32)
            const bulge = mul(sin(mul(acrossLocal, PI)), reliefAmt)
            const filJit = mul(surfaceNoise(mul(alongLocal, 2.0), mul(acrossLocal, 40.0)), 0.18)
            const filament = mul(add(sin(mul(acrossLocal, TWO_PI_SEVEN)), filJit), 0.13)
            const crossTilt = local(add(bulge, filament), 'crossTilt')
            const n = tiltAlong(nGeo, crossAxis, crossTilt)

            // Seam ambient occlusion — the dark gaps where tows cross under.
            const edge = min(sub(0.5, abs(fA)), sub(0.5, abs(fB)))
            const ao = mix(0.5, 1.0, smoothstep(0.0, 0.05, edge))

            // ── Perspective view ray ────────────────────────────────────────────────
            const viewI = viewRay(params, field)
            const grazing = grazingOf(n, viewI)
            const R = local(reflect(viewI, n), 'R')
            const rough = local(clamp(u.roughness, 0, 1), 'rough')

            // ── Anisotropic fibre sheen (Ward-style, per-cell tangent) ──────────────
            const L = lightVec3(direction(u.lightAngle, 'light'), -0.7)
            const baseA = local(mix(0.06, 0.5, rough), 'baseA')
            const sheen = anisoSpecular({
                normal: n, tangent: fiber, light: L, view: viewI,
                alphas: {along: mul(baseA, 6.0), across: baseA},
                gain: u.fiberSheen,
            })

            const ndl = clamp(add(mul(dot(n, L), 0.5), 0.5), 0, 1)

            // ── Carbon body: resin → fibre sheen, occluded in the valleys ───────────
            const bodyLum = clamp(add(add(0.035, mul(ndl, 0.10)), sheen), 0, 1.3)
            const body = mul(mix(u.darkColor.member('rgb'), u.lightColor.member('rgb'), splat3(bodyLum)), ao)

            // ── Glossy clearcoat: one crisp tap of the SHARED 5-softbox studio bank
            //    (the same bank BrushedMetal smears) at CarbonFiber's sky gain 0.3,
            //    keyRad 0.16, no drift ──
            const coatLum = studioSoftboxes(R.member('x'), R.member('y'), {
                rotation: direction(u.envRotation, 'env'), keyRadius: 0.16, drift: 0, strength: u.environment, skyGain: 0.3,
            })
            const coatF = add(0.05, mul(pow(grazing, 5), 0.95))
            const coat = local(mul(mul(coatLum, coatF), u.clearcoat), 'coat')

            const rgb = add(body, splat3(coat))

            const transition = silhouette(field, u.edgeSoftness)
            return guarded(insideShape(sdf, pxH), vec4(rgb, transition), ZERO, 'carbonFiber')
        },
    }),
})

export default componentDefinition
