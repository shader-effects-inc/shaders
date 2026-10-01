import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {direction, directionFrame} from "@coreroot/std/frames"
import {
    shapedSurface, surfaceField, geometricNormal, surfacePattern, viewRay, grazingOf, reflect,
    grainNoise, tiltNormal, smearAlong, anisoSpecular, wardAlphas, lightVec3, fresnelBoost, tintRamp, studioSoftboxes,
    silhouette, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {add, clamp, local, mix, mul, sin, sub, vec4} from "@coreroot/std/math"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// The procedural studio is the SHARED 5-softbox bank at BrushedMetal's sky gain 0.38 —
// CarbonFiber's clearcoat reads the same bank at 0.3.
const BRUSHED_SKY_GAIN = 0.38

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    lightColor: Parameters<typeof transformColor>[0]
    darkColor: Parameters<typeof transformColor>[0]
    brushAngle: number
    anisotropy: number
    grain: number
    grainScale: number
    roughness: number
    environment: number
    envRotation: number
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
    name: "BrushedMetal",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Photorealistic brushed metal — a satin anodised surface combed with fine directional grain that smears the reflected studio and key light into the long anisotropic streaks of real brushed aluminium, steel, gold or copper",
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
            default: '#e9ebef',
            transform: transformColor,
            description: 'The bright tone of the polished grain — recolor for gold (#ffe8b0), copper (#ffc8a0) or steel',
            ui: { type: 'color', label: 'Light Color', group: 'Metal' }
        },
        darkColor: {
            default: '#26282d',
            transform: transformColor,
            description: 'The dark tone the brushed grain falls to in shadow. Tint it warm for gold/bronze, cool for steel.',
            ui: { type: 'color', label: 'Dark Color', group: 'Metal' }
        },
        brushAngle: {
            default: 0,
            description: 'Direction the surface is brushed, in degrees — the grain and every reflection streak run along this axis',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Brush Angle', group: 'Brush' }
        },
        anisotropy: {
            default: 0.4,
            description: 'How directional the brushing is — 1 = long smeared streaks along the grain, 0 = isotropic satin with no direction',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Anisotropy', group: 'Brush' }
        },
        grain: {
            default: 0.4,
            description: 'Depth of the individual brush scratches — the fine satin texture combed into the metal',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Grain', group: 'Brush' }
        },
        grainScale: {
            default: 2,
            description: 'Density of the brush lines — higher = finer, tighter scratches',
            ui: { type: 'range', min: 0.25, max: 4, step: 0.01, label: 'Grain Scale', group: 'Brush' }
        },
        roughness: {
            default: 0.5,
            description: 'Overall satin softness — 0 = near-polished with crisp reflections, 1 = soft matte brushed finish',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Roughness', group: 'Surface' }
        },
        ...glassShellProps({
            environment: {
                default: 1.4,
                description: 'Strength of the reflected studio lighting — the softboxes seen smeared across the metal',
                ui: {group: 'Surface'},
            },
            envRotation: {ui: {group: 'Surface'}},
            lightAngle: {
                default: 215,
                description: 'Direction of the key light for the anisotropic specular streak, in degrees',
                ui: {group: 'Surface'},
            },
        }),
        speed: {
            default: 0.4,
            description: 'Speed of the slowly drifting reflection — the studio gently orbits the surface. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Surface' }
        },
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

    // Brushed metal as a recipe: a grain frame combed into the surface → two-scale relief (broad
    // sheen + fine hairlines) tilting the normal → the reflected studio smeared along the grain →
    // an anisotropic key-light streak → Fresnel lift → two-tone metal tint. The constants here
    // ARE the look; the machinery is vocabulary.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)

            const field = surfaceField(frame, params, {scale: u.scale})
            const nGeo = geometricNormal(frame, field, {bevelWidth: u.bevelWidth, bevelShape: u.bevelShape})

            // ── The brush grain: relief in the grain frame, tilting the normal ──────
            const brush = directionFrame(u.brushAngle, 'brush')
            const gc = brush.coordsOf(surfacePattern(frame, field))
            const aniso = local(u.anisotropy, 'aniso')
            const dens = local(mul(u.grainScale, 150), 'dens')
            const driftA = local(mul(sin(mul(t, 0.05)), 0.35), 'driftA')
            const sheenAmt = local(add(mul(aniso, 0.5), 0.08), 'sheenAmt')

            const sheen = local(grainNoise(gc, {freq: [1.3, 6.5], drift: driftA}), 'sheen')
            const lines = local(add(
                grainNoise(gc, {freq: [4.5, dens]}),
                mul(grainNoise(gc, {freq: [8.0, mul(dens, 2.3)], offset: [3.1, 7.7]}), 0.5),
            ), 'lines')
            const reliefAcross = local(add(mul(sheen, sheenAmt), mul(lines, mul(u.grain, 0.16))), 'rAcross')
            const reliefAlong = local(mul(
                grainNoise(gc, {freq: [1.3, 6.5], offset: [15.21, 0], drift: driftA}),
                mul(sheenAmt, 0.12),
            ), 'rAlong')
            const n = tiltNormal(nGeo, brush, {across: reliefAcross, along: reliefAlong})

            // ── View + the reflected studio, smeared along the grain ────────────────
            const view = viewRay(params, field)
            const rough = local(clamp(u.roughness, 0, 1), 'rough')
            const R = local(reflect(view, n), 'R')
            const studio = {
                rotation: direction(u.envRotation, 'env'),
                keyRadius: local(mix(0.42, 0.16, sub(1, rough)), 'keyRad'),
                drift: local(mul(sin(mul(t, 0.07)), 0.05), 'drift'),
                strength: local(u.environment, 'envStr'),
                skyGain: BRUSHED_SKY_GAIN,
            }
            const env = smearAlong(
                {
                    center: R,
                    axis: brush.tangent,
                    spread: mul(mul(aniso, add(mul(rough, 0.6), 0.25)), 0.42),
                    hint: 'envLum',
                },
                (x, y) => studioSoftboxes(x, y, studio),
            )

            // ── Key-light streak + compose ───────────────────────────────────────────
            const spec = anisoSpecular({
                normal: n, tangent: brush.tangent, view,
                light: lightVec3(direction(u.lightAngle, 'light'), -0.7),
                alphas: wardAlphas(rough, aniso),
                gain: mix(0.45, 1.1, sub(1, rough)),
            })
            const lit = local(clamp(mul(add(env, spec), fresnelBoost(grazingOf(n, view), {amount: 0.6})), 0, 1.7), 'lit')
            const rgb = tintRamp(lit, u.darkColor, u.lightColor)

            return guarded(insideShape(field.sdf, field.pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), ZERO, 'brushedMetal')
        },
    }),
})

export default componentDefinition
