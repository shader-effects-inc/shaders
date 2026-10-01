import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {call, ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, surfacePattern, marchedNormal, nudgeNormal, perlinSlope, viewRay,
    grazingOf, reflect, neutralTone, silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {
    abs, add, clamp, cos, div, exp, float, length, local, max, min, mix, mul, sin, smoothstep,
    splat3, sqrt, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {noise, effects, lighting, sdf3d, constants} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {bevelSin} = effects.bevel
const {fieldGradient} = lighting
const {isMobileGpuViewport} = sdf3d

const DEG_TO_RAD = constants.DEG_TO_RAD
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// ─────────────────────────────────────────────────────────────────────────────
// The procedural chrome studio — one RGB (linear, HDR) radiance tap for a reflection direction,
// inlined per tap (Chrome's own look; 1 tap on mobile, 3 spectrally-offset taps on desktop).
// Modelled on a luxury product-photography setup shot against white:
//   - a vertical surround gradient: deep graphite floor (faintly cool) → pale ceiling;
//   - a THIN warm strip light hugging the horizon (the amber band chrome logos live on);
//   - a broad cool wash just below the horizon (the smoked steel-blue mid zone);
//   - one very large frontal softbox slightly above the lens (a projected rounded-rect —
//     straight edges, not a blob — this is what flat faces mirror);
//   - two large black flags either side of the softbox (broad graphite reflections).
// `boxSoftIn` blurs the softbox edge (satin↔polished), `spectral` scales both colored strips,
// `shadows` sets how deep the black-flag/floor reflections crush. The caller rotates the
// direction for Env Rotation / the studio orbit first. The flags are CONTINUOUS features of the
// sphere (pure sideways-ness) — windowing them to camera-facing directions carves crescent-shaped
// rings into curved faces. Scalars/colors arrive pre-hoisted from the caller.
// ─────────────────────────────────────────────────────────────────────────────
function chromeStudio(exIn: Expr, eyIn: Expr, ezIn: Expr, boxSoftIn: Expr, stripSoft: Expr, spectral: Expr, shadows: Expr, warmC: Expr, coolC: Expr): Expr {
    const ex = local(exIn, 'ex')
    const ey = local(eyIn, 'ey')
    const ez = local(ezIn, 'ez')

    // Azimuthal side weighting: warm accents live toward +x, cool toward −x.
    const horiz = local(div(ex, add(length(vec2(ex, ey)), 0.0001)), 'horiz')
    const warmSide = local(add(0.35, mul(0.65, smoothstep(-0.7, 0.7, horiz))), 'warmSide')
    const coolSide = local(add(0.45, mul(0.55, smoothstep(-0.7, 0.7, mul(horiz, -1)))), 'coolSide')

    // Vertical surround: floor → compact gray mid → pale ceiling.
    const up = local(smoothstep(-0.65, 1.0, ey), 'up')
    const graphite = add(mul(coolC, 0.03), vec3(0.003, 0.003, 0.004))
    const baseLo = mix(vec3(0.17, 0.175, 0.19), graphite, splat3(shadows))
    const baseHi = vec3(0.56, 0.57, 0.6)
    let env: Expr = mix(baseLo, baseHi, splat3(mul(up, up)))
    const ceil = smoothstep(0.3, 0.9, ey)
    env = add(env, mul(vec3(0.55, 0.55, 0.56), ceil))

    // Cool wash below the horizon (biased opposite the warm side).
    const cd = local(div(add(ey, 0.45), 0.2), 'cd')
    const coolBand = exp(mul(mul(cd, cd), -1))
    env = add(env, mul(coolC, mul(mul(mul(coolBand, spectral), 0.2), coolSide)))

    // Frontal softbox: projected rounded rect, windowed to camera-facing directions.
    const facing = local(smoothstep(0.18, -0.1, ez), 'facing')
    const pz = local(max(sub(0.32, ez), 0.06), 'pz')
    const px = local(div(ex, pz), 'px')
    const py = local(div(ey, pz), 'py')
    const bx = local(sub(abs(px), 0.62), 'bx')
    const by = local(sub(abs(sub(py, 0.3)), 0.28), 'by')
    const inside = min(max(bx, by), 0)
    const dbox = local(add(length(max(vec2(bx, by), vec2(0, 0))), inside), 'dbox')
    const boxSoft = local(add(0.05, mul(boxSoftIn, 0.4)), 'boxSoft')
    const box = mul(smoothstep(boxSoft, mul(boxSoft, -0.6), dbox), facing)
    env = add(env, mul(vec3(2.9, 2.87, 2.82), box))

    // Black flags: continuous sideways darkening, applied BEFORE the strip lights.
    const flagMask = smoothstep(0.5, 0.92, abs(ex))
    const flagDepth = mix(0.55, 0.07, shadows)
    env = mul(env, mix(float(1), flagDepth, flagMask))

    // Warm rim-glow hugging the softbox edge.
    const gw = local(add(0.065, mul(stripSoft, 0.05)), 'gw')
    const dOut = local(max(dbox, 0), 'dOut')
    const warmGlow = mul(exp(mul(mul(div(dOut, gw), div(dOut, gw)), -1)), smoothstep(-0.06, 0.1, dbox))
    env = add(env, mul(warmC, mul(mul(mul(mul(warmGlow, spectral), 1.3), warmSide), facing)))
    // Faint warm floor bounce for the deep-grazing directions the box never covers.
    const fb = local(div(add(ey, 0.12), 0.05), 'fb')
    env = add(env, mul(warmC, mul(mul(mul(mul(exp(mul(mul(fb, fb), -1)), spectral), 0.5), warmSide), sub(1, facing))))

    // Low bounce card: a faint white strip under the lens.
    const cbx = local(sub(abs(px), 0.7), 'cbx')
    const cby = local(sub(abs(add(py, 0.78)), 0.05), 'cby')
    const dcard = local(add(length(max(vec2(cbx, cby), vec2(0, 0))), min(max(cbx, cby), 0)), 'dcard')
    const card = mul(smoothstep(boxSoft, mul(boxSoft, -0.6), dcard), facing)
    return add(env, mul(vec3(0.5, 0.5, 0.52), card))
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    tint: Parameters<typeof transformColor>[0]
    warmColor: Parameters<typeof transformColor>[0]
    coolColor: Parameters<typeof transformColor>[0]
    bevelWidth: number
    bevelShape: number
    curvature: number
    waviness: number
    environment: number
    envRotation: number
    softness: number
    spectral: number
    dispersion: number
    shadows: number
    speed: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Chrome",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Studio-lit mirror chrome — a precision-machined shape with a controllable polished bevel, softly convex faces and a photographic studio reflected in them: one huge frontal softbox, black flags, and thin warm/cool strip lights that fringe amber and ice-blue exactly where the reflections break",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine: 'raw' surface-pattern coords (the waviness rides the geometry) +
    // sampler routing + the forward 3-tap field frame. The chrome material as algebra: a
    // controllable multi-facet bevel + face dome + pressed-metal waviness + a perspective view
    // ray + the procedural studio (3 spectrally-offset taps on desktop, 1 on mobile — a
    // build-time device branch), graded through the PBR-neutral shoulder, inside the shape guard.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)
            const vol = frame.volumetric
            const mobile = isMobileGpuViewport()
            const EPS = 0.01
            const PR = 0.0081 // Lorentzian pillow radius² (0.09²)

            const field = surfaceField(frame, params, {scale: u.scale})
            const {s0, sX, sY, sdf, pxH} = field

            // ── Shape-local pattern coords (flat: sdf space; 3D: surface-locked) ─────
            const pat = surfacePattern(frame, field)
            const fx = pat.member('x')
            const fy = pat.member('y')

            // ── Geometric normal (volumetric/flat split is build-time) ───────────────
            let nGeo: Expr
            if (vol) {
                nGeo = marchedNormal(field)
            } else {
                // Field gradient (in-plane outward direction).
                const grad = local(call(fieldGradient, 'fieldGradient', [s0, sX!, sY!, float(EPS)]), 'grad')
                const glen = local(max(length(grad), 0.0001), 'glen')
                const gx = local(div(grad.member('x'), glen), 'gx')
                const gy = local(div(grad.member('y'), glen), 'gy')
                // Bevel: progress t across a controllable-width rim, facet boundaries wobbled
                // along the edge (polishing-line jitter).
                const bw = max(u.bevelWidth, 0.003)
                const depth = local(mul(sdf, -1), 'depth')
                const t0 = local(clamp(div(depth, bw), 0, 1), 't0')
                const rimN = local(call(noise.perlin12d, 'perlin12d', [vec2(mul(fx, 21), mul(fy, 21))]), 'rimN')
                const tt = clamp(add(t0, mul(mul(mul(mul(mul(rimN.member('x'), u.waviness), 0.22), t0), sub(1, t0)), 4)), 0, 1)
                const sBev = local(call(bevelSin, 'bevelSin', [tt, u.bevelShape]), 'sBev')
                // Pillow: a subtle SDF-following convexity that fades with interior depth
                // (Lorentzian — slope must die out well before the medial ridge).
                const sPillow = mul(mul(u.curvature, 0.14), div(PR, add(mul(depth, depth), PR)))
                // Sphere dome: the whole mark curves gently around the shape centre.
                const flen = local(length(vec2(fx, fy)), 'flen')
                const dgx = local(div(fx, max(flen, 0.0001)), 'dgx')
                const dgy = local(div(fy, max(flen, 0.0001)), 'dgy')
                const sSphere = local(mul(mul(u.curvature, 0.75), smoothstep(0, 0.5, flen)), 'sSphere')
                // Combine in-plane tilts: bevel/pillow along the SDF gradient + the sphere dome.
                const sEdge = local(max(sBev, sPillow), 'sEdge')
                const txy = local(vec2(add(mul(gx, sEdge), mul(dgx, sSphere)), add(mul(gy, sEdge), mul(dgy, sSphere))), 'txy')
                const tl = local(min(length(txy), 0.9995), 'tl')
                const tn = local(mul(txy, div(tl, max(length(txy), 0.0001))), 'tn')
                const cz = sqrt(sub(1, mul(tl, tl)))
                nGeo = local(vec3(tn.member('x'), tn.member('y'), mul(cz, -1)), 'nGeo')
            }

            // ── Pressed-metal waviness: the shared clamped-Perlin-gradient part at gain 1 ──
            const wq = vec2(add(mul(fx, 1.05), mul(t, 0.08)), mul(fy, 1.05))
            const wg = local(perlinSlope(wq, 1.0), 'wg')
            const wAmt = local(mul(u.waviness, 0.2), 'wAmt')
            const n = nudgeNormal(nGeo, mul(wg.member('x'), wAmt), mul(wg.member('y'), wAmt))

            // ── Perspective view ray + reflection ─────────────────────────────────────
            const viewI = viewRay(params, field, 0.55)
            const R = local(reflect(viewI, n), 'R')
            const tiltV = grazingOf(n, viewI)

            // ── Rotate the studio: Env Rotation + the LIVE orbit (two incommensurate sways
            //    around the vertical axis plus a slight nod). Screen uv.y grows downward; the
            //    studio is authored y-up, so flip first. ──
            const rotA = local(add(mul(u.envRotation, DEG_TO_RAD), mul(sin(mul(t, 0.23)), 0.05)), 'rotA')
            const envC = local(cos(rotA), 'envC')
            const envS = local(sin(rotA), 'envS')
            const upY = local(mul(R.member('y'), -1), 'upY')
            const rx0 = local(sub(mul(R.member('x'), envC), mul(upY, envS)), 'rx0')
            const ry0 = local(add(mul(R.member('x'), envS), mul(upY, envC)), 'ry0')
            const orbit = local(add(mul(sin(mul(t, 0.5)), 0.4), mul(sin(add(mul(t, 0.19), 1.7)), 0.2)), 'orbit')
            const oc = local(cos(orbit), 'oc')
            const os = local(sin(orbit), 'os')
            const rx = local(add(mul(rx0, oc), mul(R.member('z'), os)), 'rx')
            const rz0 = local(sub(mul(R.member('z'), oc), mul(rx0, os)), 'rz0')
            const nod = local(mul(sin(add(mul(t, 0.31), 0.9)), 0.12), 'nod')
            const nc = local(cos(nod), 'nc')
            const ns = local(sin(nod), 'ns')
            const ry = local(sub(mul(ry0, nc), mul(rz0, ns)), 'ry')
            const rz = local(add(mul(rz0, nc), mul(ry0, ns)), 'rz')

            // ── Spectral taps: per-channel elevation offset → prismatic fringes focused on
            //    the fast-curving edges (tilt² keeps flat faces clean). Flat faces see a
            //    softened studio; the bevel sees it crisp. ──
            const faceness = smoothstep(0.75, 1.0, mul(n.member('z'), -1))
            const soft = local(u.softness, 'soft')
            const softEff = local(add(soft, mul(faceness, 0.2)), 'softEff')
            const spectral = local(u.spectral, 'spectral')
            const shadows = local(u.shadows, 'shadows')
            const warmC = local(u.warmColor.member('rgb'), 'warmC')
            const coolC = local(u.coolColor.member('rgb'), 'coolC')
            const envG = local(chromeStudio(rx, ry, rz, softEff, soft, spectral, shadows, warmC, coolC), 'envG')
            let envR: Expr
            let envB: Expr
            if (mobile) {
                envR = envG.member('x')
                envB = envG.member('z')
            } else {
                const delta = local(mul(u.dispersion, add(0.004, mul(mul(tiltV, tiltV), 0.06))), 'delta')
                const envRt = local(chromeStudio(rx, add(ry, delta), rz, softEff, soft, spectral, shadows, warmC, coolC), 'envRt')
                const envBt = local(chromeStudio(rx, sub(ry, delta), rz, softEff, soft, spectral, shadows, warmC, coolC), 'envBt')
                envR = envRt.member('x')
                envB = envBt.member('z')
            }
            const env = vec3(envR, envG.member('y'), envB)

            // ── Grade: exposure → tint → filmic shoulder ──────────────────────────────
            const rgb = neutralTone(mul(mul(env, u.environment), u.tint.member('rgb')))

            return guarded(insideShape(sdf, pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), ZERO, 'chrome')
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
            description: 'Center position of the chrome shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the chrome shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the chrome shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        tint: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Metal tint multiplied into the reflection — white for silver chrome, warm ivory for gold, dusky rose for copper',
            ui: { type: 'color', label: 'Tint', group: 'Metal' }
        },
        warmColor: {
            default: '#ffa94d',
            transform: transformColor,
            description: 'Color of the warm strip light hugging the studio horizon — the amber band that fringes the reflections',
            ui: { type: 'color', label: 'Warm Light', group: 'Metal' }
        },
        coolColor: {
            default: '#82a7e6',
            transform: transformColor,
            description: 'Color of the cool wash below the horizon — the steel-blue zone the darker reflections fall into',
            ui: { type: 'color', label: 'Cool Light', group: 'Metal' }
        },
        bevelWidth: {
            default: 0.028,
            description: 'Width of the polished edge bevel, relative to the shape — thin for machined jewellery edges, wide for soft pillowed metal',
            ui: { type: ['range', 'map'], min: 0.005, max: 0.2, step: 0.001, label: 'Bevel Width', group: 'Bevel' }
        },
        bevelShape: {
            default: 0.55,
            description: 'Bevel profile — 0 = one smooth round fillet, 1 = machined: a steep outer fillet, a chamfer plateau and an inner knee, each catching its own line of light',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Bevel Profile', group: 'Bevel' }
        },
        curvature: {
            default: 0.5,
            description: 'Convex doming of the faces — sweeps the big soft studio gradients across otherwise flat metal',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Curvature', group: 'Bevel' }
        },
        waviness: {
            default: 0.15,
            description: 'Pressed-metal imperfection — gently bends the face reflections and wobbles the bevel light lines so the surface reads as real',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Waviness', group: 'Bevel' }
        },
        ...glassShellProps({
            environment: {
                description: 'Exposure of the reflected studio — the overall brilliance of the chrome',
                ui: {group: 'Studio'},
            },
            envRotation: {
                description: 'Rotates the whole reflected studio — spins where the softbox, flags and strips fall',
                ui: {type: ['range', 'map'], group: 'Studio'},
            },
        }),
        softness: {
            default: 0.3,
            description: 'Softness of the studio edges — 0 = razor polished reflections, 1 = satin diffusion',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Softness', group: 'Studio' }
        },
        spectral: {
            default: 0.9,
            description: 'The colored accent lights: an amber glow hugging the softbox edge and an ice-blue wash below it. 0 = a fully achromatic black-and-white studio.',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Accent Lights', group: 'Studio' }
        },
        ...glassShellProps({
            dispersion: {
                default: 0.3,
                description: 'Prismatic RGB splitting concentrated on the fast-curving bevel — thin rainbow micro-fringes along the edge highlights, distinct from the broad accent lights',
                ui: {group: 'Studio'},
            },
        }),
        shadows: {
            default: 0.7,
            description: 'Depth of the dark studio reflections — how hard the black side flags and floor crush toward graphite. 0 lifts them to a soft gray studio.',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shadows', group: 'Studio' }
        },
        speed: {
            default: 0.5,
            description: 'Speed of the studio orbit — the whole lighting environment slowly wanders around the piece, sliding the reflections across faces and edges. 0 freezes it.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Studio' }
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
