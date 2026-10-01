import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, geometricNormal, nudgeNormal, perlinSlope,
    flowWarp, viewRay, keyLightAt, sharpGlint, reflect, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {direction} from "@coreroot/std/frames"
import {
    add, clamp, length, local, mix, mul, sin, smoothstep, sub, vec2, vec4,
} from "@coreroot/std/math"
import {sdf3d} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {isMobileGpuViewport} = sdf3d

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// ─────────────────────────────────────────────────────────────────────────────
// A single softbox bank of the procedural studio, rotated into the env frame — LiquidMetal's own
// 3-bank look (overhead key / lower-left / right), one inline copy per reflection tap. Returns the
// reflected luminance at (rx, ry). Callers pass hoisted env/key locals.
//   bank = [cx, cy, rmul, br]
// ─────────────────────────────────────────────────────────────────────────────
function liquidStudio(rx: Expr, ry: Expr, envC: Expr, envS: Expr, keyRad: Expr, drift: Expr, envStr: Expr): Expr {
    // Rotate the reflection into the studio frame (Env Rotation).
    const ex = local(sub(mul(rx, envC), mul(ry, envS)), 'ex')
    const ey = local(add(mul(rx, envS), mul(ry, envC)), 'ey')
    const p = local(vec2(ex, ey), 'p')
    // Dark floor → brighter sky gradient (squared keeps the floor truly dark).
    const sky = local(smoothstep(-0.7, 0.85, ey), 'sky')
    let lum: Expr = mul(mul(sky, sky), 0.4)
    // Box 0: overhead key (0.0, 0.55, 1.05, 1.75).
    const r0 = local(mul(keyRad, 1.05), 'r0')
    const box0 = smoothstep(r0, mul(r0, 0.2), length(sub(p, vec2(add(0, drift), 0.55))))
    lum = add(lum, mul(mul(box0, 1.75), envStr))
    // Box 1: lower-left bank (-0.45, -0.4, 1.25, 1.15).
    const r1 = local(mul(keyRad, 1.25), 'r1')
    const box1 = smoothstep(r1, mul(r1, 0.2), length(sub(p, vec2(add(-0.45, drift), -0.4))))
    lum = add(lum, mul(mul(box1, 1.15), envStr))
    // Box 2: right bank (0.55, 0.1, 1.15, 1.1).
    const r2 = local(mul(keyRad, 1.15), 'r2')
    const box2 = smoothstep(r2, mul(r2, 0.2), length(sub(p, vec2(add(0.55, drift), 0.1))))
    return add(lum, mul(mul(box2, 1.1), envStr))
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    lightColor: Parameters<typeof transformColor>[0]
    darkColor: Parameters<typeof transformColor>[0]
    turbulence: number
    ripple: number
    warp: number
    sharpness: number
    environment: number
    envRotation: number
    dispersion: number
    lightAngle: number
    speed: number
    bevelWidth: number
    bevelShape: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "LiquidMetal",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Flowing liquid chrome — a molten reflective surface that wraps the shape, sweeping a procedural studio reflection across animated folds with crisp speculars and prismatic edges",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine: 'raw' surface-pattern coords (the molten relief reads .g/.b) +
    // sampler routing + the forward 3-tap field frame. The flowing-chrome material as algebra:
    // geometric normal + domain-warped molten relief (shared flow-warp + clamped-Perlin-gradient
    // parts) + reflected procedural studio + crisp specular + prismatic dispersion, inside the
    // shape guard. The volumetric/flat normal split and the mobile chromatic-split drop are
    // build-time branches.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)
            const mobile = isMobileGpuViewport()

            const field = surfaceField(frame, params, {scale: u.scale})
            const {sdf, pxH} = field
            const nGeo = geometricNormal(frame, field, {bevelWidth: u.bevelWidth, bevelShape: u.bevelShape})

            // ── Flowing molten relief: shared flow-warp drifts the domain, shared
            //    clamped-Perlin-gradient (gain 0.62) reads the fold slopes ────────────
            const pat = surfacePattern(frame, field)
            const freq = local(mul(u.ripple, 2.6), 'freq')
            const flowT = local(mul(t, 0.12), 'flowT')
            const q = local(vec2(mul(pat.member('x'), freq), mul(pat.member('y'), freq)), 'q')
            const wAmt = mul(u.warp, 0.9)
            const w = flowWarp(mul(q, 0.6), flowT)
            const qw = add(q, mul(w, wAmt))
            const hg = local(perlinSlope(qw, 0.62), 'hg')
            const relief = local(mul(u.turbulence, 0.5), 'relief')
            const n = nudgeNormal(nGeo, mul(mul(hg.member('x'), relief), -1), mul(mul(hg.member('y'), relief), -1))

            // ── Reflect the perspective view ray + procedural studio ─────────────────
            const viewI = viewRay(params, field)
            const R = local(reflect(viewI, n), 'R')
            const envStr = local(u.environment, 'envStr')
            const sharp = local(clamp(u.sharpness, 0, 1), 'sharp')
            const keyRad = local(mix(0.5, 0.18, sharp), 'keyRad')
            const drift = local(mul(sin(mul(t, 0.07)), 0.05), 'drift')
            const env = direction(u.envRotation, 'env')
            const envC = env.member('x')
            const envS = env.member('y')

            // Crisp specular glint from the key light (sharpness tightens it) — the shared
            // key-light frame + sharpness-steered glint lobe (endpoints are LiquidMetal's look).
            const kl = keyLightAt(direction(u.lightAngle, 'light'), -0.7)
            const Hh = local(kl.member('H'), 'Hh')
            const ndh = clamp(add(add(mul(n.member('x'), Hh.member('x')), mul(n.member('y'), Hh.member('y'))), mul(n.member('z'), Hh.member('z'))), 0, 1)
            const spec = local(sharpGlint(ndh, sharp, vec4(60.0, 900.0, 0.4, 1.4)), 'spec')

            // ── Per-channel dispersion (mobile: shared eval; desktop: chromatic split — a
            //    build-time device branch) ──
            const lumG = local(add(liquidStudio(R.member('x'), R.member('y'), envC, envS, keyRad, drift, envStr), spec), 'lumG')
            let lumR: Expr
            let lumB: Expr
            if (mobile) {
                lumR = lumG
                lumB = lumG
            } else {
                const grazeD = sub(1, clamp(mul(n.member('z'), -1), 0, 1))
                const dth = local(mul(mul(u.dispersion, 0.09), add(0.4, mul(grazeD, 1.3))), 'dth')
                const lumBoffset = local(add(liquidStudio(add(R.member('x'), mul(R.member('y'), dth)), sub(R.member('y'), mul(R.member('x'), dth)), envC, envS, keyRad, drift, envStr), spec), 'lumBoffset')
                lumB = lumBoffset
                lumR = sub(mul(lumG, 2), lumBoffset)
            }

            const dark = local(u.darkColor.member('rgb'), 'dark')
            const light = local(u.lightColor.member('rgb'), 'light')
            const rgb = vec4(
                mix(dark.member('x'), light.member('x'), clamp(lumR, 0, 1.8)),
                mix(dark.member('y'), light.member('y'), clamp(lumG, 0, 1.8)),
                mix(dark.member('z'), light.member('z'), clamp(lumB, 0, 1.8)),
                silhouette(field, u.edgeSoftness),
            )
            return guarded(insideShape(sdf, pxH), rgb, ZERO, 'liquidMetal')
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
            description: 'Center position of the metal shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the metal shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the metal shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        lightColor: {
            default: '#eef0f6',
            transform: transformColor,
            description: 'The bright tone of the chrome highlights',
            ui: { type: 'color', label: 'Light Color', group: 'Metal' }
        },
        darkColor: {
            default: '#141414',
            transform: transformColor,
            description: 'The dark tone the reflection falls to in shadow — recolor for tinted metal (gold, steel, copper…)',
            ui: { type: 'color', label: 'Dark Color', group: 'Metal' }
        },
        turbulence: {
            default: 1,
            description: 'How molten the surface is — the depth of the flowing folds that bend the reflection',
            ui: { type: 'range', min: 0, max: 1.5, step: 0.01, label: 'Turbulence', group: 'Liquid' }
        },
        ripple: {
            default: 4,
            description: 'Scale of the molten folds — higher = smaller, busier ripples',
            ui: { type: 'range', min: 0.5, max: 8, step: 0.01, label: 'Ripple Scale', group: 'Liquid' }
        },
        warp: {
            default: 1.5,
            description: 'Swirl of the flow — domain-warps the folds into liquid curls',
            ui: { type: 'range', min: 0, max: 1.5, step: 0.01, label: 'Swirl', group: 'Liquid' }
        },
        speed: {
            default: 0.5,
            description: 'Speed of the molten flow and the slowly drifting reflection. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Liquid' }
        },
        ...glassShellProps({
            environment: {
                default: 1.5,
                description: 'Strength of the reflected studio lighting — the softboxes seen across the metal',
                ui: {group: 'Surface'},
            },
            envRotation: {ui: {group: 'Surface'}},
        }),
        sharpness: {
            default: 0.6,
            description: 'Tightness of the specular highlights — 1 = razor chrome glints, 0 = soft satin',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Sharpness', group: 'Surface' }
        },
        ...glassShellProps({
            dispersion: {
                description: 'Prismatic color fringing along the reflection edges',
                ui: {group: 'Surface'},
            },
            lightAngle: {
                default: 265,
                description: 'Direction of the key light for the specular glints, in degrees',
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

})

export default componentDefinition
