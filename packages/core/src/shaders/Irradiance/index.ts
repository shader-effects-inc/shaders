import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {
    shapedSurface, surfaceField, geometricNormal, silhouette,
    placementPoint, continuedField, nearestEdge, fieldSlope, pointLightFrom, lambert, inverseSquare,
    exposureTone, guarded, insideShape,
} from "@coreroot/std/paint/materials"
import {irradianceField, shadowVisibility} from "@coreroot/std/paint/radiance"
import {emissiveAlpha} from "@coreroot/std/paint/compose"
import {listOf, accumulate} from "@coreroot/std/lists"
import {add, dot, div, float, gt, local, max, mix, mul, normalize, pow, smoothstep, splat3, sub, vec2, vec3, vec4} from "@coreroot/std/math"
import {transformPosition, transformColor, transformBoolean} from "@coreroot/utilities/transformations"
import {listPropConfig} from "@coreroot/utilities/listProps"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {isMobileGpuViewport} from "@coreroot/utilities/device"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin, MousePositionConfig} from "@coreroot/types"

const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

/** One light: where it sits (a mouse-position driver may stand in), its color, its brightness. */
export interface IrradianceLight extends Record<string, unknown> {
    position: {x: number; y: number} | MousePositionConfig
    color: string
    intensity: number
}

// Two lights out of the box — a warm key high on the left, a cool fill low on the right — so the
// color mixing where their spills meet is visible at first render.
const DEFAULT_LIGHTS: IrradianceLight[] = [
    {position: {x: 0.3, y: 0.3}, color: '#ffb347', intensity: 3},
    {position: {x: 0.72, y: 0.7}, color: '#7a5cff', intensity: 2},
]
// The list's capacity (the fixed size of the packed uniform arrays; ≤ the compute ABI's MAX_LIGHTS).
const MAX_LIGHTS = 6

// ── The look's constants ──────────────────────────────────────────────────────────────────
// The per-frame ray budget of the gathered field (stratified rays per texel): a full burst on the
// first frame after a change, a lighter budget while the light is being dragged (motion masks the
// noise), small refinement gathers folded into the running mean once still. Every tier is
// denoised and accumulated, so the still image converges far beyond visible noise either way.
const RAYS = {burst: 128, motion: 64, refine: 32, refineFrames: 12}
// Mobile-class GPUs cast half the rays.
const MOBILE_RAY_DIVISOR = 2
const MARCH_STEPS = 32
const SHADOW_STEPS = 12
// Texels per side of the gathered irradiance texture (a smooth field — modest is plenty).
const FIELD_RES = 512
const FIELD_RES_MOBILE = 320
// Width of the directly-seen emitting edge, in device pixels.
const CORE_WIDTH_PX = 2.5
// Forward-difference step for edge normals, in field units (the spine's default).
const STENCIL_EPS = 0.01
// The body: how much of its own color an unlit shape shows, and the key-light glint on its
// bevel / 3D form.
const BODY_AMBIENT = 0.3
const GLINT_EXPONENT = 60
const GLINT_GAIN = 0.6

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    lights: IrradianceLight[]
    lightHeight: number
    lightRange: number
    reach: number
    core: number
    wrap: number
    shadows: boolean
    shadowSoftness: number
    bodyColor: Parameters<typeof transformColor>[0]
    bodyLight: number
    bevelWidth: number
    edgeSoftness: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

/** The gather budget: the ray schedule (halved on mobile-class GPUs), shadows on/off, field res. */
function gatherSettings(propValues: Record<string, unknown>): {rays: typeof RAYS; shadowSteps: number; resolution: number} {
    const shadowsRaw = propValues.shadows
    const shadows = typeof shadowsRaw === 'number' ? shadowsRaw > 0 : shadowsRaw !== false
    const mobile = isMobileGpuViewport()
    const div = mobile ? MOBILE_RAY_DIVISOR : 1
    return {
        rays: {burst: RAYS.burst / div, motion: RAYS.motion / div, refine: RAYS.refine / div, refineFrames: RAYS.refineFrames},
        shadowSteps: shadows ? SHADOW_STEPS : 0,
        resolution: mobile ? FIELD_RES_MOBILE : FIELD_RES,
    }
}

/** The props the gather reads each frame. */
const GATHER_PROPS = {
    lights: {prop: 'lights', position: 'position', color: 'color', intensity: 'intensity'},
    reach: 'reach', wrap: 'wrap', lightRange: 'lightRange', shadowSoftness: 'shadowSoftness', steps: MARCH_STEPS,
} as const

const spine = shapedSurface({
        // Centre tap only, and the CHEAP one: the open canvas only gates on it (coverage, the rim
        // and body branches), so it pays one bilinear tap — not three, not the bicubic. The taps a
        // normal needs (quality centre + neighbours) are taken INSIDE the branches (`stencilled`).
        stencil: 'centre',
        centreTap: 'fast',
        surface: (frame, params) => {
            const {uniforms: u, propValues} = params
            const {shadowSteps} = gatherSettings(propValues)
            const shadows = shadowSteps > 0

            // The field, honest far from the shape (unscaled: distances are in the shape's units).
            const field = surfaceField(frame, params)
            // The forward stencil, on demand: the field re-tapped at quality in the centre (the body's
            // 3D normal reads its depth) with its two neighbour taps bound.
            const stencilled = (hint: string) => ({
                ...field,
                s0: local(frame.sampler(frame.sdfUV!), `${hint}S0`),
                sX: local(frame.gradSampler(add(frame.sdfUV!, vec2(STENCIL_EPS, 0))), `${hint}X`),
                sY: local(frame.gradSampler(add(frame.sdfUV!, vec2(0, STENCIL_EPS))), `${hint}Y`),
            })
            const px = local(div(field.pxH, max(u.scale, 0.001)), 'pxUV')
            const far = continuedField(frame, field, params)

            // The lights, as the list prop's uniforms: per item a placement-space position and an
            // rgb color scaled by its intensity.
            const lights = listOf(params, 'lights')
            const lightAt = (i: Expr) => {
                const item = lights.at(i)
                return {
                    uv: placementPoint(params, item.position('position'), 'lightUV'),
                    rgb: mul(item.color('color').member('rgb'), item.number('intensity')),
                }
            }

            // ── The gathered light field (compute), read bilinearly ─────────────────
            const irradiance = irradianceField({...gatherSettings(propValues), ...GATHER_PROPS}).sample(params) ?? splat3(0)

            // ── The emitter itself: the lit edge seen directly, a few pixels wide ────────
            // The gather converges to half the edge radiance beside a lit edge; the hot rim of a
            // real render is the emitting boundary in view. Same lighting, same shadow rays.
            const rim = local(sub(1, smoothstep(0, mul(px, CORE_WIDTH_PX), far.sdf)), 'rim')
            const rimLight = (): Expr => {
                const edge = nearestEdge(frame.sdfUV!, far.sdf, add(fieldSlope(stencilled('rim'), STENCIL_EPS), far.outward))
                return accumulate(lights, (i) => {
                    const light = lightAt(i)
                    const onEdge = pointLightFrom({position: light.uv, height: u.lightHeight}, edge.point, 'rimLight')
                    let struck = mul(lambert(edge.normal, onEdge.toLight, {wrap: u.wrap}), inverseSquare(onEdge.distance, u.lightRange))
                    if (shadows) {
                        struck = mul(struck, shadowVisibility({
                            from: add(edge.point, mul(edge.normal, mul(px, 2))),
                            toward: light.uv,
                            field: far.at,
                            steps: SHADOW_STEPS,
                            surfaceEps: px,
                            softness: u.shadowSoftness,
                            hint: 'rimVisibility',
                        }))
                    }
                    return mul(light.rgb, struck)
                }, {zero: 'vec3f', hint: 'rimSum', deps: [edge.point, edge.normal]})
            }
            const emitter = guarded(gt(rim, 0), mul(rimLight(), mul(rim, u.core)), splat3(0), 'emitter', [px])

            // Exposure burns the rim to white where the lights pile up; each light keeps its color
            // where it falls alone.
            const glow = emissiveAlpha(exposureTone(add(irradiance, emitter)), 'glow')

            // ── The body: the same lights landing on the bevelled face or the marched form ──
            // Only shaded inside the silhouette (the coverage is exactly 0 outside).
            const shadeBody = (): Expr => {
                const n = geometricNormal(frame, stencilled('body'), {bevelWidth: u.bevelWidth, bevelShape: float(0), eps: STENCIL_EPS})
                const lit = accumulate(lights, (i) => {
                    const light = lightAt(i)
                    const onBody = pointLightFrom({position: light.uv, height: u.lightHeight}, frame.sdfUV!, 'bodyLight')
                    const bodyLit = local(mul(inverseSquare(onBody.distance, u.lightRange), u.bodyLight), 'bodyLit')
                    const halfVec = normalize(add(onBody.L, vec3(0, 0, -1)))
                    const glint = mul(pow(max(dot(n, halfVec), 0), GLINT_EXPONENT), GLINT_GAIN)
                    // Diffuse in the body's color, the glint in the light's.
                    return add(
                        mul(u.bodyColor.member('rgb'), mul(lambert(n, onBody.L), bodyLit)),
                        mul(light.rgb, mul(glint, bodyLit)),
                    )
                }, {zero: 'vec3f', hint: 'bodySum', deps: [n]})
                return exposureTone(add(mul(u.bodyColor.member('rgb'), BODY_AMBIENT), lit))
            }
            const body = guarded(insideShape(field.sdf, px), shadeBody(), splat3(0), 'body', [])

            // ── Compose: body over spill ─────────────────────────────────────────────
            const cov = local(silhouette({...field, pxH: px}, u.edgeSoftness), 'cov')
            return vec4(mix(glow.member('rgb'), body, cov), mix(glow.member('a'), 1, cov))
        },
})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Irradiance",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Photorealistic light spilling around the edge of any 2D, SVG, or 3D shape — any number of movable colored point lights strike the silhouette and the lit edges irradiate their surroundings, gathered in a compute pass with real cast shadows, mixing where they meet and burning to white at the rim",
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the shape (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        // The lights, as a LIST prop: add, remove and drag each one; any light's position can be
        // mapped to the mouse.
        lights: listPropConfig<IrradianceLight>({
            maxItems: MAX_LIGHTS,
            minItems: 1,
            itemLabel: 'Light',
            item: {
                position: {kind: 'position', default: {x: 0.3, y: 0.3}, label: 'Position', description: 'Where this light sits — the edges facing it are struck and spill light outward. Map it to the mouse to carry the light around the shape.'},
                color: {kind: 'color', default: '#ffb347', label: 'Color', description: 'The color this light casts'},
                intensity: {kind: 'number', default: 3, label: 'Intensity', min: 0, max: 8, step: 0.01, description: 'Brightness of this light — past ~2 the rim it strikes overexposes to white'},
            },
        }, {
            default: DEFAULT_LIGHTS,
            description: 'The point lights striking the shape — each with its own position, color and brightness',
            label: 'Lights',
            group: 'Lights',
        }),
        lightHeight: {
            default: 0.5,
            description: 'Height of the lights above the shape plane, relative to the shape — low lights graze the body, high lights flood it evenly',
            ui: { type: ['range', 'map'], min: 0.05, max: 3, step: 0.01, label: 'Light Height', group: 'Light' }
        },
        lightRange: {
            default: 1.2,
            description: 'Reach of each light — the distance at which it has fallen to half strength, relative to the shape',
            ui: { type: ['range', 'map'], min: 0.1, max: 4, step: 0.01, label: 'Range', group: 'Light' }
        },
        reach: {
            default: 3,
            description: 'How far from the shape the spilled light is gathered, relative to the shape — the glow fades out toward this distance',
            ui: { type: ['range', 'map'], min: 0.5, max: 8, step: 0.01, label: 'Reach', group: 'Glow' }
        },
        core: {
            default: 1,
            description: 'The lit edge seen directly — the white-hot line along the rim where the lights strike',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Core', group: 'Glow' }
        },
        wrap: {
            default: 0.1,
            description: 'How far light wraps around edges that face away from it — 0 leaves the far side dark',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Wrap', group: 'Glow' }
        },
        // Compile-time enum read RAW from propValues (the PixelSort precedent): shadow rays are a
        // kernel variant, so the toggle is structural.
        shadows: {
            default: true,
            compileTime: true,
            transform: transformBoolean,
            description: 'Cast shadows — the body blocks each light from edges it stands in front of (keeps holes and concave pockets dark)',
            ui: { type: 'checkbox', label: 'Shadows', group: 'Glow' }
        },
        shadowSoftness: {
            default: 0.15,
            description: 'Penumbra width of the cast shadows — 0 is razor sharp',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shadow Softness', group: 'Glow' }
        },
        bodyColor: {
            default: '#000000',
            transform: transformColor,
            description: 'Color of the shape itself — black for a pure occluder, lighter to see the lights land on its form',
            ui: { type: 'color', label: 'Body Color', group: 'Body' }
        },
        bodyLight: {
            default: 0.6,
            description: 'How much the lights illuminate the body — the bevel of a flat shape or the form of a 3D one',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Body Light', group: 'Body' }
        },
        bevelWidth: {
            default: 0.01,
            description: 'Width of the rounded edge a flat shape catches the light on, relative to the shape',
            ui: { type: ['range', 'map'], min: 0.005, max: 0.2, step: 0.001, label: 'Bevel Width', group: 'Body' }
        },
        ...glassShellProps({
            edgeSoftness: {ui: {group: 'Body'}},
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

    // Irradiance as a recipe: the light field is GATHERED ONCE per change in a compute pass —
    // per texel of a fixed-res irradiance texture, a stratified fan of rays traced against the shape
    // field, each ray that reaches the boundary seeing that edge lit by EVERY light (Lambert facing,
    // inverse-square, a shadow ray back to each light), each light's color landing where it falls;
    // the mean, denoised, IS the irradiance (seamless, thinning with the emitter's angular size,
    // occluded where the body stands in the way). The fragment reads it bilinearly and adds what
    // must be full-res: the emitting edge seen directly and the body — the same lights landing on
    // the bevelled face / 3D form. Exposure burns the rim to white. A static scene costs one texture
    // sample per pixel.
    ...spine,
    compute: (params) => irradianceField({...gatherSettings(params.propValues), ...GATHER_PROPS}).compute(params),
})

export default componentDefinition
