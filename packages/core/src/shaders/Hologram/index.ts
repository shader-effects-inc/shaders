import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface, guarded, insideShape, surfaceNoise, hashNoise} from "@coreroot/std/paint/materials"
import {
    abs, add, clamp, div, exp, float, fract, local, max, min, mix, mul, normalize, pow, sin,
    smoothstep, splat3, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {constants} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })
const TWO_PI = constants.TWO_PI

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    color: Parameters<typeof transformColor>[0]
    brightness: number
    fill: number
    edgeGlow: number
    depthLines: number
    depthScale: number
    scanlines: number
    scanlineScale: number
    scanlineSpeed: number
    sweep: number
    flicker: number
    distortion: number
    grain: number
    speed: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Hologram",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Volumetric sci-fi hologram — a translucent emissive projection of the shape with fresnel-lit edges, depth scan-lines that wrap 3D forms, CRT scanlines, flicker and beam wobble",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine: patternMode 'none' (Hologram reads .r + .a only), 'firstLobe' chord,
    // the 5-tap central-difference stencil at eps 0.008, and a placeUV slot — the beam wobble
    // displaces the SAMPLING uv while the screen-locked projector layers keep the raw ctx.uv.
    // The volumetric-hologram material as algebra: optical thickness + fresnel rim + depth
    // scan-lines + CRT scanlines/sweep/flicker/grain + chromatic fringe, inside the shape guard.
    ...shapedSurface({
        chord: 'firstLobe',
        stencil: {kind: 'central', eps: 0.008},
        // Beam instability: a continuous horizontal wobble applied to the SAMPLING uv only (the
        // projector grid stays screen-locked).
        placeUV: (params) => {
            const {uniforms: u, ctx} = params
            const t = animatedTime(params)
            const uvY = ctx.uv.member('y')
            const wob = add(mul(sin(add(mul(uvY, 38), mul(t, 2.3))), 0.5), mul(sin(sub(mul(uvY, 11.5), mul(t, 1.7))), 0.5))
            return vec2(add(ctx.uv.member('x'), mul(mul(wob, u.distortion), 0.012)), uvY)
        },
        surface: (frame, params) => {
            const {uniforms: u, ctx} = params
            const t = local(animatedTime(params), 'animTime')
            const vol = frame.volumetric
            const EPS = 0.008

            const sC = local(frame.surf0!, 'sC')
            const pxH = local(div(1, ctx.viewportSize.member('y')), 'pxH')
            const sdf = local(div(sC.member('x'), u.scale), 'sdf')

            const sR = local(frame.surfX!, 'sR')
            const sL = local(frame.surfL!, 'sL')
            const sU = local(frame.surfY!, 'sU')
            const sD = local(frame.surfD!, 'sD')

            const sharp = max(mul(u.edgeSoftness, 0.5), 0.001)
            const gradX = local(div(sub(sR.member('x'), sL.member('x')), EPS * 2), 'gradX')

            // Optical thickness.
            const thickness = local(max(mul(sdf, -1), 0), 'thickness')
            const thin = local(exp(mul(thickness, -5.5)), 'thin')

            // Surface normal + fresnel rim + depth metric (volumetric split is build-time).
            let rimBase: Expr
            let depthMetric: Expr
            if (vol) {
                const nx = clamp(div(sub(sR.member('w'), sL.member('w')), EPS * 2), -2, 2)
                const ny = clamp(div(sub(sU.member('w'), sD.member('w')), EPS * 2), -2, 2)
                const nrm = local(normalize(vec3(nx, ny, -1)), 'nrm')
                const surfFres = pow(clamp(sub(1, abs(nrm.member('z'))), 0, 1), 2.2)
                rimBase = add(mul(thin, 0.55), surfFres)
                depthMetric = sC.member('w')
            } else {
                rimBase = thin
                depthMetric = thickness
            }
            const rim = local(mul(rimBase, u.edgeGlow), 'rim')

            // Translucent volume fill (Beer–Lambert).
            const bodyFill = mul(sub(1, exp(mul(thickness, -5))), u.fill)

            // Depth scan-lines.
            const phase = sub(mul(depthMetric, u.depthScale), mul(t, 1.2))
            const f = local(fract(phase), 'phaseFract')
            const distI = min(f, sub(1, f))
            const contour = mul(smoothstep(0.06, 0, distI), u.depthLines)

            // Projector layers (screen-space).
            const scanPhase = sub(mul(ctx.uv.member('y'), u.scanlineScale), mul(t, mul(u.scanlineSpeed, 3)))
            const scan = add(mul(sin(mul(scanPhase, TWO_PI)), 0.5), 0.5)
            const scanMask = local(mix(float(1), scan, u.scanlines), 'scanMask')

            const sweepPos = fract(mul(t, 0.16))
            const sweepD = local(sub(ctx.uv.member('y'), sweepPos), 'sweepD')
            const sweepGlow = local(mul(exp(mul(mul(sweepD, sweepD), -650)), u.sweep), 'sweepGlow')

            const shimmer = add(mul(sin(mul(t, 40)), 0.04), mul(sin(mul(t, 23.3)), 0.03))
            const dropNoise = add(mul(surfaceNoise(mul(t, 6), 11), 0.5), 0.5)
            const drop = smoothstep(0.6, 0.95, dropNoise)
            const flick = mul(add(1, shimmer), sub(1, mul(mul(u.flicker, drop), 0.7)))

            const gn = hashNoise(add(mul(ctx.uv.member('x'), ctx.viewportSize.member('x')), mul(t, 57.3)), mul(ctx.uv.member('y'), ctx.viewportSize.member('y')))
            const grainMul = add(1, mul(mul(sub(gn, 0.5), u.grain), 0.6))

            // Compose.
            const glow = add(add(bodyFill, rim), contour)
            const core = local(mul(add(mul(mul(mul(glow, scanMask), flick), grainMul), mul(sweepGlow, scanMask)), u.brightness), 'core')

            const ca = add(mul(u.distortion, 0.12), 0.04)
            const fringe = local(mul(mul(mul(rim, gradX), ca), 0.1), 'fringe')
            const rCore = add(core, fringe)
            const bCore = sub(core, fringe)

            const color = u.color.member('rgb')
            const rgb = vec3(mul(color.member('x'), rCore), mul(color.member('y'), core), mul(color.member('z'), bCore))
            const hot = clamp(add(mul(rim, 0.5), sweepGlow), 0, 0.8)
            const hotRgb = local(mix(rgb, splat3(core), splat3(hot)), 'hotRgb')

            const rb1 = clamp(mul(div(mul(sdf, -1), sharp), 32), 0, 1)
            const transition = smoothstep(0, 1, rb1)
            const alpha = mul(clamp(core, 0, 1), transition)
            return guarded(insideShape(sdf, pxH), vec4(hotRgb, alpha), ZERO, 'hologram')
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
            description: 'Center position of the hologram shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the hologram shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the hologram shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        color: {
            default: '#5ec8ff',
            transform: transformColor,
            description: 'Projection tint — the color the whole hologram emits (classic sci-fi cyan by default)',
            ui: { type: 'color', label: 'Color', group: 'Style' }
        },
        brightness: {
            default: 2.5,
            description: 'Overall emission gain. Bright edges blow out toward white for a hot, projected look.',
            ui: { type: 'range', min: 0.2, max: 3, step: 0.01, label: 'Brightness', group: 'Style' }
        },
        edgeSoftness: {
            default: 0.04,
            description: 'Softness of the shape boundary edge — keep low for crisp projection edges',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Style' }
        },
        fill: {
            default: 0.45,
            description: 'Translucent interior glow. Driven by optical thickness, so thicker parts of a 3D form glow more — the core volumetric cue.',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Volume Fill', group: 'Volume' }
        },
        edgeGlow: {
            default: 1,
            description: 'Fresnel rim intensity — bright glowing silhouette and, on 3D shapes, the grazing internal walls of extruded logos',
            ui: { type: 'range', min: 0, max: 2.5, step: 0.01, label: 'Edge Glow', group: 'Volume' }
        },
        depthLines: {
            default: 0.5,
            description: 'Iso-depth scan-lines that hug the surface and reveal the 3D form like a volumetric scan. On flat shapes they become concentric contour rings.',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Depth Lines', group: 'Volume' }
        },
        depthScale: {
            default: 26,
            description: 'Density of the depth scan-lines — higher packs more contour bands through the volume',
            ui: { type: 'range', min: 4, max: 70, step: 1, label: 'Depth Density', group: 'Volume' }
        },
        scanlines: {
            default: 0.55,
            description: 'CRT scanline contrast — how dark the gaps between horizontal projection lines get',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Scanlines', group: 'Scanlines' }
        },
        scanlineScale: {
            default: 130,
            description: 'Number of horizontal scanlines across the canvas height',
            ui: { type: 'range', min: 20, max: 400, step: 1, label: 'Scanline Count', group: 'Scanlines' }
        },
        scanlineSpeed: {
            default: 1,
            description: 'How fast the scanlines drift upward — 0 holds them static',
            ui: { type: 'range', min: 0, max: 5, step: 0.01, label: 'Scanline Drift', group: 'Scanlines' }
        },
        sweep: {
            default: 0.1,
            description: 'Brightness of the refresh bar that sweeps vertically through the projection',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Sweep Bar', group: 'Scanlines' }
        },
        flicker: {
            default: 0.3,
            description: 'Unstable-projector flicker — a fast shimmer plus occasional brightness dropouts',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Flicker', group: 'Interference' }
        },
        distortion: {
            default: 0.1,
            description: 'Continuous horizontal wobble and chromatic edge fringing — the wavering of an unstable beam',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Distortion', group: 'Interference' }
        },
        grain: {
            default: 0.7,
            description: 'Fine projection static — animated per-pixel noise over the emission',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Grain', group: 'Interference' }
        },
        speed: {
            default: 1,
            description: 'Master animation speed for every animated layer. 0 freezes the projection.',
            ui: { type: 'range', min: 0, max: 3, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: { type: 'shape', label: 'Shape', group: 'Shape' }
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file — when non-empty, activates SVG mode and triggers a shader recompile'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type — triggers recompile when shape is switched. When empty, derived from shape JSON at mount time.'
        }
    },

})

export default componentDefinition
