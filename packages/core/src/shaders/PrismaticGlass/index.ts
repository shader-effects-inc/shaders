import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {animatedTime, call} from "@coreroot/gpu/porters"
import {defineShader, crosses} from "@coreroot/std"
import {direction} from "@coreroot/std/frames"
import {
    shapedSurface, fdSlope, guarded, interleavedNoise, reflect, refract, schlickFresnel, beerLambert,
    roundedSlab, throughSlab, spectralSum, areaLightStudio, thinFilmColor, causticCast,
} from "@coreroot/std/paint/materials"
import {
    add, clamp, div, dpdxFine, dpdyFine, fract, fwidthFine, length, local, lt, max, min, mix, mul, neg, normalize,
    rotate2, sin, smoothstep, splat3, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {blend, sdf3d} from "@coreroot/gpu/kit"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {unpremultiplyAlpha} = blend
const {isMobileGpuViewport} = sdf3d

const DEFAULT_SHAPE_CONFIG = JSON.stringify({type: 'sphere3D', radius: 0.35})

// ── The look's constants ────────────────────────────────────────────────────────────────────
// Slices of the spectrum each pixel traces. 8 with per-pixel jitter reads as a continuous
// spectrum; the loop body (one backdrop tap + two studio lookups) is what each slice costs and
// dominates the glass interior, so phones (isMobileGpuViewport) trace 5.
const SPECTRAL_SAMPLES = 8
const SPECTRAL_SAMPLES_MOBILE = 5
// Index-of-refraction spread across the spectrum at dispersion 1. Stylised: dense flint is
// about 0.03, so this is a prism that splits light the way people picture it.
const DISPERSION_SPREAD = 0.22
// The key light sways around `lightAngle` so highlights and spectra slide across the glass.
const LIGHT_SWAY_DEG = 17
const LIGHT_SWAY_RATE = 0.25
// The back face bends the lookup about as much again as the front: the filter footprint the
// front-face derivative measures, scaled up to cover both faces.
const BACK_FACE_FOOTPRINT = 2
// The widest the slices spread over the pixel's footprint (before the back-face factor). Eight
// taps filter a few pixels cleanly; spread wider they print ghost copies of the backdrop.
const MAX_FOOTPRINT_PX = 4
// Where the spectrum's two ends land this many pixels apart, slices hold fixed wavelengths:
// jittered they would turn into speckle (thick, high-IOR glass), fixed they read as clean bands.
const STEADY_SPECTRUM_PX = [1.5, 6] as const
// Where the surface turns steeply (the rim) the slices land far apart, so they sample in a fixed
// pattern there and neighbours agree (smooth spectral bands, never speckle). Measured on the
// surface slope, which is smooth per pixel.
const STEADY_FROM_SLOPE = 1.5
const STEADY_FULL_SLOPE = 4
// The marched depth is unreliable within ~2 texels of the silhouette (it blends into the missed-
// ray background), so the face tilt fades out over these texels.
const TILT_FADE_TEXELS = [2.5, 5] as const
// The edge: anti-aliased over at least EDGE_FEATHER_PX either side of the silhouette, and softened
// inward by `edgeSoftness` (shape units, so it scales with the shape) up to EDGE_SOFTNESS_MAX. The
// band where a 3D surface turns to its grazing rim, in pixels (inside the softened edge).
const EDGE_FEATHER_PX = 0.75
const EDGE_SOFTNESS_MAX = 0.04
const RIM_BLEND_PX = 3
// Absorption per unit of tint depth, and the steepest a surface may slope before it is clamped.
const ABSORB_GAIN = 3
const MAX_SLOPE = 6
// Two irrational strides walk each slice to its own spot in the pixel (the R2 sequence).
const R2_STEP = [0.7548777, 0.5698403] as const

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    refraction: number
    dispersion: number
    backdrop: number
    bevel: number
    edgeSoftness: number
    tintColor: Parameters<typeof transformColor>[0]
    absorption: number
    reflections: number
    innerFire: number
    coating: number
    keyColor: Parameters<typeof transformColor>[0]
    rimColor: Parameters<typeof transformColor>[0]
    lightAngle: number
    caustics: number
    speed: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "PrismaticGlass",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Physically traced prismatic glass: every colour refracts through the front and back faces on its own, light trapped inside escapes as spectral fire, a studio reflects in the surface under a shifting thin-film coat, and the glass can cast a colour-split caustic on what lies behind it",
    requiresRTT: true,
    requiresChild: true,
    animatedTime: {speed: 'speed'},
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: {type: 'origin', label: 'Origin', group: 'Position'}
        },
        center: {
            default: {x: 0.5, y: 0.5},
            transform: transformPosition,
            description: 'Center position of the glass shape',
            ui: {type: 'position', label: 'Center', group: 'Position', units: ['%', 'px']}
        },
        scale: {
            default: 1,
            description: 'Scale of the glass shape (1 = default size)',
            ui: {type: ['range', 'map'], min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position'}
        },
        rotation: {
            default: 0,
            description: 'Rotation of the glass shape in degrees',
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position'}
        },
        refraction: {
            default: 1.6,
            description: 'Index of refraction: 1.33 water, 1.5 window glass, 1.7 flint, 2.4 diamond. Higher bends the layers beneath more and traps more light inside as fire.',
            ui: {type: ['range', 'map'], min: 1, max: 2.5, step: 0.01, label: 'Refraction (IOR)', group: 'Glass'}
        },
        dispersion: {
            default: 0.75,
            description: 'How far the glass splits white light into a spectrum: rainbow fringes on every edge it bends, colour in the fire and the caustic. 0 is clear, colourless glass.',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Dispersion', group: 'Glass'}
        },
        backdrop: {
            default: 0.3,
            description: 'How far behind the glass the layers beneath sit. Further bends and inverts them more, like a ball lens, and pushes the cast shadow further away.',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Backdrop Distance', group: 'Glass'}
        },
        bevel: {
            default: 0.12,
            description: 'Width of the rounded glass edge on flat shapes and logos (3D shapes use their real form)',
            ui: {type: ['range', 'map'], min: 0.01, max: 0.4, step: 0.005, label: 'Bevel', group: 'Glass'}
        },
        edgeSoftness: {
            default: 0.25,
            description: 'How gently the glass fades into the layers beneath at its outline: 0 is a crisp, anti-aliased edge, higher melts the rim into the backdrop',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Edge Softness', group: 'Glass'}
        },
        tintColor: {
            default: '#ffffff',
            transform: transformColor,
            description: 'The colour the glass body absorbs toward, deeper where the glass is thicker. White is clear glass.',
            ui: {type: 'color', label: 'Tint', group: 'Glass'}
        },
        absorption: {
            default: 1,
            description: 'How strongly the tint colours the glass: 0 keeps it clear, higher reads as thick coloured glass (amber, emerald, smoke)',
            ui: {type: ['range', 'map'], min: 0, max: 4, step: 0.01, label: 'Tint Depth', group: 'Glass'}
        },
        reflections: {
            default: 1,
            description: 'Brightness of the studio the glass reflects: the softbox and strip lights on its surface and in the light bouncing inside it',
            ui: {type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Reflections', group: 'Light'}
        },
        innerFire: {
            default: 1.2,
            description: 'Light the back faces reflect and send back out through the front: the spectral sparkle inside a gem',
            ui: {type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Inner Fire', group: 'Light'}
        },
        coating: {
            default: 0.3,
            description: 'A thin-film coat on the surface whose colour shifts with the viewing angle, like a lens coating or a soap skin',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Coating', group: 'Light'}
        },
        keyColor: {
            default: '#fff3e4',
            transform: transformColor,
            description: 'Colour of the key softbox the glass reflects and the light it casts',
            ui: {type: 'color', label: 'Key Light', group: 'Light'}
        },
        rimColor: {
            default: '#9ec2ff',
            transform: transformColor,
            description: 'Colour of the strip lights at the sides, which draw the glass rim',
            ui: {type: 'color', label: 'Rim Light', group: 'Light'}
        },
        lightAngle: {
            default: 225,
            description: 'Where the key light comes from, in degrees clockwise from the right (225 = upper left). The cast shadow falls the other way.',
            ui: {type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Light'}
        },
        caustics: {
            default: 0,
            recompile: crosses(0),
            description: 'The soft shadow and the focused, colour-split light the glass casts on the layers behind it. Reads best when those layers look like a lit surface: a wall, paper, a studio sweep. 0 is off and costs nothing.',
            ui: {type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Caustics', group: 'Light'}
        },
        speed: {
            default: 1,
            description: 'Speed of the slowly swaying key light and the flowing coat. 0 holds them still.',
            ui: {type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Light'}
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: {type: 'shape', label: 'Shape', group: 'Shape'}
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'Legacy: URL of a pre-generated SDF .bin for an SVG shape. SVG shapes are generated in real time from the svgUrl in shape; this is only the fallback for older presets'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type — triggers recompile when shape is switched. When empty, derived from shape JSON at mount time.'
        }
    },

    // Prismatic glass as a recipe: rebuild the glass body (thickness, slope, the tilt between its
    // faces) from the shape → measure how fast the refracted and reflected lookups race across the
    // pixel → cast the shadow and caustic → trace every slice of the spectrum in through the front
    // face and out through the back, landing on the layers beneath or bouncing back out as fire,
    // each slice at its own spot in the pixel → reflect a crisp studio in the front face, through a
    // thin-film coat. One field tap at the pixel (with analytic slope) and one where the light
    // leaves the back face; everything else is optics.
    ...shapedSurface({
        chord: 'firstLobe',
        bakedGradients: 'all',
        stencil: 'centre',
        child: 'required',
        surface: (frame, params) => {
            const {uniforms: u, ctx, propValues} = params
            const t = animatedTime(params)
            const childTexture = frame.childTexture!
            const placement = local(frame.sdfUV!, 'placement')
            const s0 = local(frame.surf0!, 's0')
            const field = local(s0.member('x'), 'field') // signed distance; −half the thickness inside a 3D shape

            // ── Placement: one pixel in shape units, and the shape's rotation ────────────────
            const vh = ctx.viewportSize.member('y')
            const aspect = local(div(ctx.viewportSize.member('x'), vh), 'aspect')
            const px = local(div(1, mul(u.scale, vh)), 'pxShape')
            const rot = direction(u.rotation, 'placeRot')
            const toScreen = (v: Expr): Expr => rotate2(v, rot.member('x'), rot.member('y'))
            const toShape = (v: Expr): Expr => rotate2(v, rot.member('x'), neg(rot.member('y')))
            const toUV = (v: Expr): Expr => vec2(div(mul(v.member('x'), u.scale), aspect), mul(v.member('y'), u.scale))
            const slopeAt = (sample: Expr, at: Expr, hint: string): Expr => {
                if (frame.bakedGradients) return local(toScreen(sample.member('yz')), hint)
                const fd = fdSlope((p) => frame.sampler(p).member('x'), at, 0.002, `${hint}Fd`)
                return local(toScreen(vec2(fd.dx, fd.dy)), hint)
            }
            const capSlope = (g: Expr, hint: string): Expr =>
                local(mul(g, min(1, div(MAX_SLOPE, max(length(g), 1e-5)))), hint)

            const child = local(call(unpremultiplyAlpha, 'unpremultiplyAlpha', [childTexture.sample(ctx.uv)]), 'child')

            // ── The glass body: half its thickness, its slope, and the tilt between its faces ──
            const g = slopeAt(s0, placement, 'slope')

            // ── Coverage: the edge feathered over ~1.5 px, evenly on both sides ──────────────
            // The field divided by its own analytic slope is the distance to the silhouette in
            // pixels, per pixel. A 3D field (minus half the thickness) plunges near-vertically just
            // inside the rim, so feathering the raw field would fade only the outside half and
            // leave a hard, stepped edge; in pixels the feather is symmetric on any shape.
            const edgePx = local(div(field, max(mul(length(g), px), 1e-7)), 'edgePx')
            const softPx = local(max(div(mul(u.edgeSoftness, EDGE_SOFTNESS_MAX), px), EDGE_FEATHER_PX), 'softPx')
            const coverage = local(sub(1, smoothstep(neg(softPx), EDGE_FEATHER_PX, edgePx)), 'coverage')
            let half: Expr
            let thicknessSlope: Expr
            let tilt: Expr
            if (frame.volumetric) {
                half = local(max(neg(field), 0), 'half')
                // At the silhouette a real glass wall turns vertical, but the feathered pixels just
                // outside read the gentle slope of the OUTSIDE distance: blend to the grazing rim
                // over the last two pixels so the edge pixels shade as rim, not as a 45° face.
                const rimward = local(normalize(add(neg(g), vec2(1e-6, 0))), 'rimward')
                thicknessSlope = local(mix(
                    capSlope(neg(g), 'innerSlope'),
                    mul(rimward, MAX_SLOPE),
                    smoothstep(-RIM_BLEND_PX, 0, edgePx),
                ), 'thicknessSlope')
                // The plane midway between the faces tilts on rotated solids: the front face's slope
                // (the marched depth's analytic slope) less the thickness slope. Exact on flat
                // facets; faded out near the rim, where the marched depth meets the background.
                if (frame.depthSlope && frame.texelSpan) {
                    const midSlope = local(add(toScreen(frame.depthSlope(placement)), thicknessSlope), 'midSlope')
                    tilt = local(mul(
                        mul(midSlope, min(1, div(3, max(length(midSlope), 1e-5)))),
                        smoothstep(TILT_FADE_TEXELS[0], TILT_FADE_TEXELS[1], div(half, max(frame.texelSpan, px))),
                    ), 'tilt')
                } else {
                    tilt = vec2(0, 0)
                }
            } else {
                const slab = roundedSlab(neg(field), u.bevel, 'slab')
                half = slab.half
                thicknessSlope = capSlope(mul(normalize(add(g, vec2(1e-6, 0))), neg(slab.slope)), 'thicknessSlope')
                tilt = vec2(0, 0)
            }
            const thickness = local(mul(half, 2), 'thickness')
            const view = vec3(0, 0, 1)
            const front = local(normalize(vec3(sub(tilt.member('x'), thicknessSlope.member('x')), sub(tilt.member('y'), thicknessSlope.member('y')), -1)), 'front')
            const mirror = local(reflect(view, front), 'mirror')

            // ── Pixel footprints: how fast the refracted lookup and the reflection move ────────
            // Measured before any branch (derivatives need every pixel), per pixel (fine
            // derivatives, never shared across a 2×2 block). Each slice takes its own spot in the
            // pixel and follows it through the lookup's Jacobian — the pixel's real footprint on
            // the layers beneath — so a racing lookup (the rim) is filtered, not aliased.
            const inward = local(refract(view, front, div(1, u.refraction)), 'inward')
            const frontTravel = local(mul(inward.member('xy'), div(thickness, max(inward.member('z'), 0.25))), 'frontTravel')
            const lookupFront = local(add(ctx.uv, toUV(frontTravel)), 'lookupFront')
            const maxStep = local(div(MAX_FOOTPRINT_PX, vh), 'maxStep')
            const capStep = (step: Expr, hint: string): Expr =>
                local(mul(mul(step, min(1, div(maxStep, max(length(step), 1e-7)))), BACK_FACE_FOOTPRINT), hint)
            const stepX = capStep(dpdxFine(lookupFront), 'lookupStepX')
            const stepY = capStep(dpdyFine(lookupFront), 'lookupStepY')
            const steady = local(smoothstep(STEADY_FROM_SLOPE, STEADY_FULL_SLOPE, length(thicknessSlope)), 'steady')
            const envBlur = local(min(length(fwidthFine(mirror)), 1.5), 'envBlur')
            const pixel = mul(ctx.uv, ctx.viewportSize)
            const spotJitter = local(mix(vec2(interleavedNoise(pixel.member('yx')), interleavedNoise(add(pixel.member('yx'), vec2(5.3, 11.9)))), vec2(0.5, 0.5), steady), 'spotJitter')

            // ── The key light, swaying slowly; the shadow and caustic it casts ────────────────
            const keyDir = direction(add(u.lightAngle, mul(sin(mul(t, LIGHT_SWAY_RATE)), LIGHT_SWAY_DEG)), 'keyDir')
            const key = local(normalize(vec3(mul(keyDir.member('x'), 0.7), mul(keyDir.member('y'), 0.7), -0.72)), 'key')
            const keyRGB = u.keyColor.member('rgb')
            const rimRGB = u.rimColor.member('rgb')
            let wall: Expr = child.member('rgb')
            if (typeof propValues.caustics !== 'number' || propValues.caustics > 0) {
                // Light travels along −key, so the glass lands down-light, further with distance.
                const landing = mul(neg(key.member('xy')), div(add(u.backdrop, 0.12), neg(key.member('z'))))
                const cast = causticCast((p) => frame.gradSampler(p).member('x'), {
                    at: placement,
                    offset: toShape(landing),
                    slant: neg(key.member('z')),
                    focus: clamp(mul(sub(u.refraction, 1), 1.4), 0, 0.9),
                    spread: u.dispersion,
                })
                wall = add(mul(child.member('rgb'), sub(1, mul(cast.shadow, min(u.caustics, 1)))), mul(mul(keyRGB, cast.light), u.caustics))
            }
            wall = local(wall, 'wall')

            // ── Inside the glass ─────────────────────────────────────────────────────────────
            // The back face where a mid-spectrum ray actually reaches it.
            const exitAt = local(add(placement, toShape(frontTravel)), 'exitAt')
            const sExit = local(frame.sampler(exitAt), 'sExit')
            const gExit = slopeAt(sExit, exitAt, 'slopeExit')
            const exitSlope = frame.volumetric
                ? capSlope(neg(gExit), 'exitSlope')
                : capSlope(mul(normalize(add(gExit, vec2(1e-6, 0))), neg(roundedSlab(neg(sExit.member('x')), u.bevel, 'slabExit').slope)), 'exitSlope')
            const back = local(normalize(vec3(add(tilt.member('x'), exitSlope.member('x')), add(tilt.member('y'), exitSlope.member('y')), -1)), 'back')

            // A studio: a big key softbox, a cool strip each side, a strip overhead, in a room
            // that opens onto the layers beneath.
            const topRGB = local(mix(keyRGB, rimRGB, 0.5), 'topRGB')
            const studio = areaLightStudio({
                panels: [
                    {toward: key, size: [0.55, 0.3], rounding: 0.08, color: keyRGB, gain: mul(u.reflections, 2.6), hotspot: 1.4},
                    {toward: normalize(vec3(mul(keyDir.member('x'), -0.8), -0.15, -0.58)), size: [0.055, 0.9], softness: 0.015, color: rimRGB, gain: mul(u.reflections, 2.2)},
                    {toward: normalize(vec3(mul(keyDir.member('x'), 0.9), 0.1, -0.42)), size: [0.04, 0.9], softness: 0.015, color: rimRGB, gain: mul(u.reflections, 1.2)},
                    {toward: vec3(0, -0.848, -0.53), size: [1.05, 0.04], softness: 0.015, color: topRGB, gain: mul(u.reflections, 1.4)},
                ],
                room: {
                    floor: splat3(0.008),
                    ceiling: mul(add(mul(keyRGB, 0.14), splat3(0.01)), u.reflections),
                    horizon: mul(topRGB, mul(u.reflections, 0.09)),
                },
                behind: wall,
            })

            // Every slice of the spectrum, in through the front and out through the back.
            const absorb = local(mul(sub(splat3(1), u.tintColor.member('rgb')), mul(u.absorption, ABSORB_GAIN)), 'absorb')
            const spread = local(mul(u.dispersion, DISPERSION_SPREAD), 'spread')
            // How far apart the red and violet ends of the spectrum land, in pixels.
            const landingAt = (offset: number, hint: string): Expr => throughSlab({
                view, front, back, ior: add(u.refraction, mul(spread, offset)), thickness, behind: u.backdrop, hint,
            }).landing
            const fanUV = toUV(sub(landingAt(0.5, 'violetEnd'), landingAt(-0.5, 'redEnd')))
            const fanPx = length(mul(fanUV, ctx.viewportSize))
            const jitter = local(mix(interleavedNoise(pixel), 0.5,
                max(steady, smoothstep(STEADY_SPECTRUM_PX[0], STEADY_SPECTRUM_PX[1], fanPx))), 'jitter')
            const radiance = spectralSum({
                samples: isMobileGpuViewport() ? SPECTRAL_SAMPLES_MOBILE : SPECTRAL_SAMPLES,
                jitter,
                deps: [front, back, thickness, stepX, stepY, spotJitter, envBlur, absorb, spread, wall, topRGB, ...studio.deps],
            }, (band) => {
                const pass = throughSlab({
                    view, front, back,
                    ior: add(u.refraction, mul(spread, sub(band.t, 0.5))),
                    thickness,
                    behind: u.backdrop,
                })
                const spot = sub(fract(add(spotJitter, mul(vec2(R2_STEP[0], R2_STEP[1]), band.index))), vec2(0.5, 0.5))
                const lookup = add(add(ctx.uv, toUV(pass.landing)), add(mul(stepX, spot.member('x')), mul(stepY, spot.member('y'))))
                const beyond = mix(studio.room(pass.exit, envBlur), childTexture.sampleLevel(lookup).member('rgb'), pass.square)
                const body = local(beerLambert(pass.path, absorb), 'body')
                const fire = mul(studio.light(pass.bounce, mul(envBlur, 0.5)), mul(sub(1, pass.escape), mul(mul(body, body), u.innerFire)))
                return add(mul(mul(beyond, body), pass.escape), fire)
            })

            // The front face: Fresnel reflection of the studio through the coat.
            const facing = local(clamp(neg(front.member('z')), 0, 1), 'facing')
            const f0r = div(sub(u.refraction, 1), add(u.refraction, 1))
            const f0 = local(mul(f0r, f0r), 'f0')
            const fresnel = local(schlickFresnel(sub(1, facing), {r0: f0, gain: sub(1, f0)}), 'fresnel')
            const filmNm = add(300, mul(add(0.5, mul(sin(add(add(mul(half, 18), mul(placement.member('x'), 4)), sub(mul(t, 0.5), mul(placement.member('y'), 3)))), 0.5)), 220))
            const coat = mix(splat3(1), mul(thinFilmColor(facing, filmNm), 2), u.coating)
            const glass = add(mul(radiance, sub(1, fresnel)), mul(mul(studio.light(mirror, envBlur), fresnel), coat))

            const shaded = guarded(
                lt(edgePx, EDGE_FEATHER_PX + 1),
                vec4(glass, 1),
                vec4(0, 0, 0, 0),
                'prismatic',
                [wall, front, mirror, tilt, thickness, half, stepX, stepY, envBlur, steady, spotJitter, key, keyDir, placement],
            )
            return vec4(mix(wall, shaded.member('rgb'), coverage), mix(child.member('a'), 1, coverage))
        },
    }),
})

export default componentDefinition
