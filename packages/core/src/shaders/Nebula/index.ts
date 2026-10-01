import type {Expr, GpuShaderDefinition} from "@coreroot/gpu/porters"
import {ZERO, animatedTime, call} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {direction} from "@coreroot/std/frames"
import {
    shapedSurface, surfaceField, geometricNormal, viewRay, reflect,
    keyLightAt, studioSoftboxes, surfaceNoiseAt, shellRefract, rotationSensor,
    twinkle, pointStars, dualLobeGlint, hashNoise, neutralTone, silhouette,
    guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {interiorRay, parallaxPlane, turbulentMedium, volumeMarch} from "@coreroot/std/paint/volume"
import {layers} from "@coreroot/std/paint/compose"
import {
    abs, add, clamp, dot, float, local, mix, mul, pow, smoothstep,
    splat3, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {colorMixing, d, sdf, sdf3d} from "@coreroot/gpu/kit"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {isMobileGpuViewport, VOLUMETRIC_FIELD_EXTRA_FIELDS} = sdf3d
const {ANALYTIC_SDF_EXTRA_FIELDS} = sdf

// A glass prism is the shader's identity — default to the marched hexagonal prism so the drop-in
// reads as gas sealed inside a solid (the Crystal precedent for a material-specific default shape).
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// The interior march: emission/absorption samples along the view chord through the shape. Six
// steps is enough because banding is bought off with a per-pixel hash jitter of the march offset
// (fine static grain) instead of more steps, and the gas field itself is low-frequency. Mobile
// compiles a 4-step variant (the Chrome/LiquidMetal build-time device branch) — the emission and
// absorption gains are normalized by step count, so the two tiers match in brightness.
const NEBULA_STEPS_DESKTOP = 6
const NEBULA_STEPS_MOBILE = 4

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    coreColor: Parameters<typeof transformColor>[0]
    gasColor: Parameters<typeof transformColor>[0]
    veilColor: Parameters<typeof transformColor>[0]
    colorSpace: string
    density: number
    cavity: number
    dust: number
    gasScale: number
    billow: number
    glow: number
    seed: number
    stars: number
    starScale: number
    twinkle: number
    refraction: number
    environment: number
    highlight: number
    highlightSoftness: number
    lightAngle: number
    edgeSoftness: number
    speed: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Nebula",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "A volumetric gas nebula sealed inside polished glass; billowing emission clouds with hollow dark cavities and hot glowing cores, star fields drifting at real depth behind the gas, all refracted through the curved walls of the shape and dressed with studio reflections",
    animatedTime: {speed: 'speed'},
    // The shape-effect spine with `pattern: 'raw'` — but the GAS domain is deliberately anchored
    // in PLACEMENT space (sdfUV), not surface-pattern space: the field march is orthographic
    // along view depth, so (sdfUV − 0.5, entryDepth + s) IS the real interior ray position,
    // continuous across the facets of a 3D solid (surface-locked coords would smear the gas into
    // per-face streaks). The raw pattern coords are kept for ONE job: (placement − pattern) is a
    // per-pixel ROTATION SENSOR (zero at rest on every shape, growing with 3D rotation), which
    // drives the star planes' counter-parallax — rotating the prism pans the starfield behind it
    // like orbiting a window, each layer at its own rate.
    //
    // The material is a genuine (unrolled) volumetric integration, not a surface trick:
    //   interior ray  — entry depth from the field tap's `.w` (0 on flat shapes) + path length
    //                   from the shared opticalThickness chord, sheared per-pixel along the
    //                   perspective view ray so deeper samples genuinely parallax;
    //   gas           — per step, one-octave domain warp (the billowing folds) over a 3-octave
    //                   3D fbm, thresholded by a cavity level (the hollow carved voids real
    //                   nebulae have) and faded near the glass walls so the gas floats free of
    //                   the container; front-to-back emission/absorption compositing gives
    //                   self-shadowing and depth occlusion for free;
    //   color        — thin veils → body gas → hot core by local density, dimming with depth
    //                   into the prism (internal atmospheric perspective);
    //   stars         — two shape-locked worley point planes at different depths, occluded by
    //                   the accumulated gas transmittance (far plane fully behind the gas);
    //   glass shell   — geometric normal (marched volumetric / bevelled flat, resolved at build
    //                   time), Schlick-fresnel-weighted studio softbox reflection, a sharp key
    //                   glint.
    ...shapedSurface({
        pattern: 'raw',
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = animatedTime(params)
            const steps = isMobileGpuViewport() ? NEBULA_STEPS_MOBILE : NEBULA_STEPS_DESKTOP
            // Compile-time color-space ramps (the Duotone/scenic convention): veil → gas → core
            // blend in the selected space. For a non-linear space the three endpoint conversions
            // are pixel-invariant, so they run ONCE per frame on the CPU (dirty-keyed — the
            // Beam/quadBlend pattern) into the conv* extraFields; each march step then pays two
            // in-space weighted mixes + a single back-conversion instead of four forward + two
            // back conversions. Linear RGB mixes directly (there is no conversion to hoist).
            const spaceMode = (params.propValues.colorSpace as number) ?? 0
            let veilGasMix: (tv: Expr) => Expr
            let rampMix: (t1: Expr, t2: Expr, hint: string) => Expr
            if (spaceMode !== 0) {
                let lastKey = ''
                params.onBeforeRender(() => {
                    const cols = (['veilColor', 'gasColor', 'coreColor'] as const).map(
                        (name) => params.getCpuValue(name) as {x: number; y: number; z: number} | undefined,
                    )
                    if (cols.some((c) => !c)) return
                    const key = cols.map((c) => `${c!.x},${c!.y},${c!.z}`).join('|')
                    if (key === lastKey) return
                    lastKey = key
                    const fields = ['convVeil', 'convGas', 'convCore'] as const
                    cols.forEach((c, idx) => {
                        const conv = colorMixing.convertP3ToMixSpaceCPU(c!.x, c!.y, c!.z, spaceMode)
                        params.setExtraField(fields[idx], [conv[0], conv[1], conv[2]])
                    })
                })
                const back = colorMixing.mixPreconvertedVariants[spaceMode as keyof typeof colorMixing.mixPreconvertedVariants]
                    ?? colorMixing.mixPreconvertedLinear
                veilGasMix = (tv) => call(back, 'mixPreconvertedColors', [
                    u.convVeil, u.convGas, u.veilColor.member('a'), u.gasColor.member('a'), tv,
                ])
                rampMix = (t1, t2, hint) => {
                    const ab = local(call(colorMixing.mixPreconvertedInSpace, 'mixInSpace', [
                        u.convVeil, u.veilColor.member('a'), u.convGas, u.gasColor.member('a'), t1,
                    ]), `${hint}AB`)
                    return call(back, 'mixPreconvertedColors', [
                        ab.member('conv'), u.convCore, ab.member('alpha'), u.coreColor.member('a'), t2,
                    ])
                }
            } else {
                veilGasMix = (tv) => call(colorMixing.mixColorsLinear, 'mixColors', [u.veilColor, u.gasColor, tv])
                rampMix = (t1, t2) => call(colorMixing.mixColorsLinear, 'mixColors', [
                    call(colorMixing.mixColorsLinear, 'mixColors', [u.veilColor, u.gasColor, t1]), u.coreColor, t2,
                ])
            }

            const field = surfaceField(frame, params, {scale: u.scale})
            const {sdf, pxH} = field

            // ── Shell geometry ───────────────────────────────────────────────────────
            const n = geometricNormal(frame, field, {bevelWidth: float(0.09), bevelShape: float(0)})
            const view = viewRay(params, field)

            // ── Interior frame: placement coords bent at the curved walls, the rotation
            //    sensor for star counter-parallax, and the interior ray everything rides ──
            const bend = local(mul(u.refraction, 0.16), 'bend')
            const interior = shellRefract(frame, n, bend)
            const {x: sx, y: sy} = interior.centered
            const sensor = rotationSensor(frame)
            const ray = interiorRay(field, frame, {origin: interior, view, shear: 0.55, lengthScale: 0.9})
            const pathLen = ray.length

            // ── The gas domain ───────────────────────────────────────────────────────
            const freq = local(mul(u.gasScale, 3.1), 'freq')
            const drift = local(mul(t, 0.14), 'drift')
            // The seed pans the whole domain to a different region of noise space — without it
            // every instance reads the SAME neighbourhood of the field (the shape spans under a
            // noise cell at low gasScale), so composition quirks repeat on every canvas.
            const seedOff = local(mul(u.seed, 7.31), 'seedOff')
            const jit = local(mul(hashNoise(mul(interior.x, 311.7), mul(interior.y, 173.3)), 0.15 / steps), 'jit')
            // Macro structure: one low-frequency field deciding where the cloud banks live at all,
            // so the nebula has large-scale anatomy instead of uniform soup. Its frequency is
            // FIXED (not gasScale-scaled) so low gasScale can't collapse it to a single lobe.
            const macro = local(mul(surfaceNoiseAt(vec2(
                add(add(mul(sx, 1.3), mul(drift, 0.6)), mul(seedOff, 0.37)),
                add(mul(sy, 1.3), mul(seedOff, 0.61)),
            )), 0.5), 'macro')
            const cav = local(mix(-0.35, 0.5, u.cavity), 'cav')
            const billowAmt = local(mul(u.billow, 0.55), 'billowAmt')
            const core = local(u.coreColor.member('rgb'), 'coreRgb')
            const veil = local(u.veilColor.member('rgb'), 'veilRgb')
            // Emission/absorption per step: a floor plus a path-length slope, so a thin prism
            // still glows (art direction) while a deep chord genuinely accumulates more gas.
            const eGain = local(mul(add(0.3, mul(pathLen, 0.9)), 6.0 / steps), 'eGain')
            const aGain = local(mul(mul(add(0.5, mul(pathLen, 2.2)), u.density), 3.0 / steps), 'aGain')

            // ── The march: front-to-back emission/absorption ─────────────────────────
            // Transmittance is PER-CHANNEL: the cold dust lanes absorb blue harder than red, so
            // gas and stars seen through them redden — interstellar extinction, the astrophoto cue.
            const march = volumeMarch(
                {steps, entry: ray.entry, length: ray.length, jitter: jit, emissionGain: eGain, hint: 'gasAcc'},
                ({i, t: fi, z}) => {
                    const pos = ray.at(z)
                    const gasN = turbulentMedium({x: pos.x, y: pos.y, z}, {
                        frequency: freq, drift, seed: seedOff, billow: billowAmt, hint: `${i}`,
                    })
                    // Wall fade: the gas floats free of the glass, densest mid-chord.
                    const wall = smoothstep(0, 0.11, mul(pathLen, Math.min(fi, 1 - fi)))
                    // The fine octave doubles as a ridge term (1 − |n|): wispy filaments threading
                    // the cloud banks, at zero extra noise cost.
                    const ridge = sub(1, clamp(mul(abs(gasN.fine), 1.6), 0, 1))
                    const dens = local(mul(mul(
                        smoothstep(0, 0.5, sub(add(gasN.density, macro), cav)), wall),
                        add(0.65, mul(ridge, 0.7))), `dens${i}`)
                    // Cold dust lanes: the mid octave's positive band, an absorb-only medium that
                    // crosses the bright banks as dark reddening tendrils.
                    const dustD = local(mul(mul(smoothstep(0.45, 0.75, gasN.mid), wall), u.dust), `dust${i}`)
                    // Thin veils → body gas → hot core (mixed in the selected color space),
                    // dimming with depth into the prism.
                    const cB = local(rampMix(smoothstep(0.05, 0.75, dens), smoothstep(0.68, 1.0, dens), `cB${i}`), `cB${i}`).member('rgb')
                    const dim = 1 - fi * 0.45
                    // A faint veil haze survives inside the carved voids — deep space is never
                    // perfectly empty, and it keeps the dark side of the volume readable.
                    const haze = mul(smoothstep(-0.7, 0.3, gasN.density), mul(0.05, wall))
                    const emit = add(add(
                        mul(cB, mul(dens, dim)),
                        mul(core, mul(pow(dens, 3), mul(u.glow, 1.2 * dim)))),
                        mul(veil, mul(haze, dim)),
                    )
                    const absorb = add(
                        splat3(mul(dens, aGain)),
                        mul(vec3(1.5, 1.0, 0.55), mul(dustD, mul(aGain, 2.2))),
                    )
                    return {emit, absorb}
                },
            )

            // ── Star planes: two parallax planes at different depths, tinted point stars ──
            // Rotating the shape counter-pans each plane via the rotation sensor at its own
            // rate — the far plane drifts more, the layered parallax of a window into depth.
            const starFreq = local(mul(u.starScale, 34), 'starFreq')
            const twinkleAmt = local(mul(u.twinkle, 0.5), 'twinkleAmt')
            const farPlane = parallaxPlane(ray, {depth: 1.1, pan: sensor, panRate: 1.3, hint: 'far'})
            const far = pointStars(local(vec2(mul(farPlane.x, starFreq), mul(farPlane.y, starFreq)), 'farP'), {
                variationFreq: 0.13, radius: [0.004, 0.013], keep: [0.35, 0.75], brightness: [0.3, 1.2],
                shimmer: (bv) => twinkle(t, bv, {amount: twinkleAmt, rate: 1.7, spread: 40}),
                gain: u.stars, tint: {from: [0.72, 0.82, 1.0], to: [1.0, 0.92, 0.82], gain: 1.6},
                hint: 'starFar',
            })
            const nearPlane = parallaxPlane(ray, {depth: 0.25, pan: sensor, panRate: 0.55, hint: 'near'})
            const nearFreq = mul(starFreq, 0.56)
            const near = pointStars(local(vec2(add(mul(nearPlane.x, nearFreq), 7.3), add(mul(nearPlane.y, nearFreq), 2.9)), 'nearP'), {
                variationFreq: 0.11, radius: [0.003, 0.01], keep: [0.45, 0.8], brightness: [0.35, 1.3],
                shimmer: (bv) => twinkle(t, bv, {amount: twinkleAmt, rate: 2.3, spread: 34}),
                gain: u.stars, tint: {from: [0.75, 0.85, 1.0], to: [1.0, 0.95, 0.88], gain: 1.8},
                hint: 'starNear',
            })

            // ── The glass shell ──────────────────────────────────────────────────────
            const lightDir = direction(u.lightAngle, 'light')
            const kl = keyLightAt(lightDir, -0.7)
            const ndh = local(clamp(dot(n, kl.member('H')), 0, 1), 'ndh')
            const R = local(reflect(view, n), 'R')
            const envLum = local(studioSoftboxes(R.member('x'), R.member('y'), {
                rotation: lightDir,
                keyRadius: 0.2,
                drift: 0,
                strength: u.environment,
                skyGain: 0.32,
            }), 'envLum')
            const spec = dualLobeGlint(ndh, {
                core: [140, 1.3], halo: [22, 0.14],
                softness: u.highlightSoftness, gain: u.highlight,
            })

            // ── The scene, back to front: deep-space floor, the gas volume, two star
            //    planes seen through it, then the glass dressing on top ────────────────
            const rgb = neutralTone(layers([
                {paint: veilGasMix(float(0.25)).member('rgb'), opacity: 0.06, behind: march.transmittance},
                march.color,
                {paint: far.rgb!, behind: march.transmittance},
                {paint: near.rgb!, behind: mix(march.transmittance, splat3(1), 0.35)},
                {paint: mul(vec3(0.85, 0.92, 1.08), envLum), opacity: 0.38},
                mul(vec3(1.0, 0.98, 0.95), spec),
            ]))
            return guarded(insideShape(sdf, pxH), vec4(rgb, silhouette(field, u.edgeSoftness)), ZERO, 'nebula')
        },
    }),
    // Overrides the spine spread above: the shapedSurface field set (keep in sync with its
    // `extraFields` return) + the three CPU-preconverted ramp endpoints the onBeforeRender
    // driver in `surface` writes (see the color-space ramp comment there).
    extraFields: {
        ...VOLUMETRIC_FIELD_EXTRA_FIELDS, ...ANALYTIC_SDF_EXTRA_FIELDS,
        convVeil: {schema: d.vec3f, initial: [0, 0, 0]},
        convGas: {schema: d.vec3f, initial: [0, 0, 0]},
        convCore: {schema: d.vec3f, initial: [0, 0, 0]},
    },
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the nebula shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the nebula shape (1 = default size)',
            ui: { type: ['range', 'map'], min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the nebula shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        coreColor: {
            default: '#ffd9b0',
            transform: transformColor,
            description: 'The hot color the densest gas cores glow with — the embedded newborn stars',
            ui: { type: 'color', label: 'Core Color', group: 'Nebula' }
        },
        gasColor: {
            default: '#ff5e7a',
            transform: transformColor,
            description: 'The main emission color of the gas body — H-alpha rose by default',
            ui: { type: 'color', label: 'Gas Color', group: 'Nebula' }
        },
        veilColor: {
            default: '#3a5bdb',
            transform: transformColor,
            description: 'The color of the thin outer veils and the deep-space background behind the gas',
            ui: { type: 'color', label: 'Veil Color', group: 'Nebula' }
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space the veil → gas → core ramps blend in',
            ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Nebula' }
        },
        density: {
            default: 1,
            description: 'Optical density of the gas — how strongly clouds occlude the stars and gas behind them',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Density', group: 'Nebula' }
        },
        cavity: {
            default: 0,
            description: 'How hollowed-out the nebula is — 0 fills the volume with cloud banks, high carves large dark voids between them',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Cavity', group: 'Nebula' }
        },
        dust: {
            default: 0.55,
            description: 'Cold dark dust lanes threading the gas — absorb-only tendrils that redden the clouds and stars behind them',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Dust', group: 'Nebula' }
        },
        gasScale: {
            default: 0.4,
            description: 'Feature size of the gas — higher = finer, busier cloud detail',
            ui: { type: ['range', 'map'], min: 0.1, max: 1, step: 0.01, label: 'Gas Scale', group: 'Nebula' }
        },
        billow: {
            default: 0.6,
            description: 'Turbulent folding of the clouds — 0 = smooth drifting banks, 1 = heavily churned billows',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Billow', group: 'Nebula' }
        },
        glow: {
            default: 0,
            description: 'Hot emission bloom of the dense cores',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Glow', group: 'Nebula' }
        },
        seed: {
            default: 0,
            description: 'Random seed — pans the gas field to a different region of noise space for a new cloud composition',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Nebula' }
        },
        stars: {
            default: 0.07,
            description: 'Brightness of the star fields drifting at depth behind and inside the gas',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Stars', group: 'Stars' }
        },
        starScale: {
            default: 1,
            description: 'Density of the star fields — higher = smaller, busier stars',
            ui: { type: ['range', 'map'], min: 0.25, max: 3, step: 0.01, label: 'Star Scale', group: 'Stars' }
        },
        twinkle: {
            default: 0.35,
            description: 'Intensity of the star twinkle (0 = steady stars, 1 = full shimmer)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Twinkle', group: 'Stars' }
        },
        ...glassShellProps({
            refraction: {description: 'How strongly the curved glass walls bend the interior seen near the edges'},
            environment: {
                default: 0.7,
                description: 'Strength of the studio softboxes reflected in the polished glass surface',
            },
            highlight: {
                default: 1.15,
                description: 'Sharp key-light glint on the glass',
                ui: {group: 'Glass'},
            },
            highlightSoftness: {ui: {group: 'Glass'}},
            lightAngle: {description: 'Direction of the key light and reflected studio, in degrees'},
            edgeSoftness: true,
        }),
        speed: {
            default: 0.3,
            description: 'Speed of the slowly churning gas and twinkling stars. 0 freezes the nebula.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Animation' }
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
