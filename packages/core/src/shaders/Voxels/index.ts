import type {GpuShaderDefinition, Expr} from "@coreroot/gpu/porters"
import {call} from "@coreroot/gpu/porters"
import {defineStd, p} from "@coreroot/std"
import {voxelSurface} from "@coreroot/std/paint/voxels"
import {mixColorsIn} from "@coreroot/std/paint/fields"
import {lambert, grazingOf, schlickFresnel} from "@coreroot/std/paint/materials"
import {add, clamp, cos, dot, exp2, local, max, mix, mul, neg, normalize, pow, sin, smoothstep, sub, vec3, vec4} from "@coreroot/std/math"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import {constants} from "@coreroot/gpu/kit"
import type {BoundingBoxOrigin} from "@coreroot/types"

const DEG_TO_RAD = constants.DEG_TO_RAD
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// ── The look's constants ─────────────────────────────────────────────────────────────────────────
// Wrap lighting keeps the shadow side of a voxel readable instead of pitch black.
const KEY_WRAP = 0.12
// Ground bounce is a darker copy of the sky so downward faces stay in the same color family.
const GROUND_GAIN = 0.35
// Blinn–Phong exponent range swept by `glossiness`: 2^3 (satin) → 2^7.5 (lacquer).
const GLOSS_EXP_RANGE: [number, number] = [3, 7.5]
// Per-voxel tonal jitter at full `colorVariation` (±30%).
const VARIATION_RANGE = 0.6
// Seam band: width fraction of the face half-size, and how dark the seam goes at full strength.
const SEAM_WIDTH: [number, number] = [0.04, 0.12]
const SEAM_DARKEN = 0.75

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    voxelSize: number
    voxelShape: string
    voxelScale: number
    fill: number
    bevel: number
    depth: number
    gridSpace: string
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorMode: string
    colorVariation: number
    colorSpace: string
    lightAngle: number
    lightElevation: number
    lightColor: Parameters<typeof transformColor>[0]
    lightIntensity: number
    ambientColor: Parameters<typeof transformColor>[0]
    ambient: number
    shadows: number
    shadowSoftness: number
    ao: number
    glossiness: number
    specular: number
    seams: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Voxels",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Rebuild any shape out of voxels — a flat shape becomes chunky pixel art, a 3D shape a lit voxel model with smooth ambient occlusion, cast shadows, a key light you can orbit or drive, and per-cube color",
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: {type: 'origin', label: 'Origin', group: 'Position'},
        },
        center: {
            default: {x: 0.5, y: 0.5},
            transform: transformPosition,
            description: 'Center position of the voxel model',
            ui: {type: 'position', label: 'Center', group: 'Position', units: ['%', 'px']},
        },
        scale: {
            default: 1,
            description: 'Scale of the voxel model (1 = default size)',
            ui: {type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position'},
        },
        rotation: {
            default: 0,
            description: 'Rotation of the voxel model in degrees',
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position'},
        },
        voxelSize: {
            default: 0.035,
            description: 'Edge length of one voxel relative to the shape field — small for fine detail, large for chunky blocks',
            ui: {type: 'range', min: 0.006, max: 0.15, step: 0.001, label: 'Voxel Size', group: 'Voxels'},
        },
        voxelShape: {
            default: 'cube',
            compileTime: true,
            description: 'What each cell is built from: hard cubes, rounded cubes (see Bevel) or spheres',
            ui: {
                type: 'select',
                options: [
                    {label: 'Cube', value: 'cube'},
                    {label: 'Rounded Cube', value: 'rounded'},
                    {label: 'Sphere', value: 'sphere'},
                ],
                label: 'Voxel Shape', group: 'Voxels',
            },
        },
        voxelScale: {
            default: 1,
            description: 'Size of each voxel inside its cell — 1 = touching neighbours, lower opens gaps between the blocks',
            ui: {type: 'range', min: 0.3, max: 1, step: 0.01, label: 'Voxel Fill', group: 'Voxels'},
        },
        fill: {
            default: 0,
            description: 'How eagerly cells fill in along the surface — negative carves a thinner model, positive puffs it out',
            ui: {type: 'range', min: -0.5, max: 0.5, step: 0.01, label: 'Surface Bias', group: 'Voxels'},
        },
        bevel: {
            default: 0.08,
            description: 'Edge rounding of each voxel as a fraction of its size — a hairline chamfer catching light on cubes, the corner radius of rounded cubes',
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Bevel', group: 'Voxels'},
        },
        depth: {
            default: 0.12,
            description: 'Thickness given to flat shapes (2D shapes and flat SVGs) so they become a slab of voxels — 3D shapes ignore it',
            ui: {type: 'range', min: 0.01, max: 0.6, step: 0.005, label: 'Depth', group: 'Voxels'},
        },
        gridSpace: {
            default: 'shape',
            compileTime: true,
            description: 'Model: the grid rotates with the shape like a built voxel model. Screen: the grid is fixed to the canvas and the shape moves through it like a 3D pixelation',
            ui: {
                type: 'select',
                options: [
                    {label: 'Model', value: 'shape'},
                    {label: 'Screen', value: 'view'},
                ],
                label: 'Grid', group: 'Voxels',
            },
        },
        colorA: {
            default: '#6ea8ff',
            transform: transformColor,
            description: 'Base color of the voxels (the first color of the palette mode)',
            ui: {type: 'color', label: 'Color A', group: 'Colors'},
        },
        colorB: {
            default: '#1d2b4f',
            transform: transformColor,
            description: 'Second palette color — blended in by height, depth, per voxel or on the side faces',
            ui: {type: 'color', label: 'Color B', group: 'Colors'},
        },
        colorMode: {
            default: 'height',
            compileTime: true,
            description: 'How the two colors are laid over the model: one solid color, a gradient by height or depth, a random pick per voxel, or Color A on top faces and Color B on the sides',
            ui: {
                type: 'select',
                options: [
                    {label: 'Solid', value: 'solid'},
                    {label: 'Height', value: 'height'},
                    {label: 'Depth', value: 'depth'},
                    {label: 'Random', value: 'random'},
                    {label: 'Top / Sides', value: 'faces'},
                ],
                label: 'Palette', group: 'Colors',
            },
        },
        colorVariation: {
            default: 0.25,
            description: 'Per-voxel tonal jitter — the slightly-off shades that make blocks read as individual bricks',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Variation', group: 'Colors'},
        },
        colorSpace: {
            default: 'oklab',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'},
        },
        lightAngle: {
            default: 225,
            description: 'Direction the key light comes from, in degrees around the canvas — shadows fall away from it',
            ui: {type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Light'},
        },
        lightElevation: {
            default: 42,
            description: 'How high the key light sits above the canvas — low for long raking shadows, high for a flat top-down light',
            ui: {type: ['range', 'map'], min: 5, max: 85, step: 1, label: 'Light Elevation', group: 'Light'},
        },
        lightColor: {
            default: '#fff2df',
            transform: transformColor,
            description: 'color of the key light',
            ui: {type: 'color', label: 'Light Color', group: 'Light'},
        },
        lightIntensity: {
            default: 1.1,
            description: 'Strength of the key light',
            ui: {type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Light Intensity', group: 'Light'},
        },
        ambientColor: {
            default: '#9fb4d8',
            transform: transformColor,
            description: 'Sky color of the ambient light — the fill on faces the key light misses',
            ui: {type: 'color', label: 'Ambient Color', group: 'Light'},
        },
        ambient: {
            default: 0.55,
            description: 'Strength of the ambient sky/ground fill',
            ui: {type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Ambient', group: 'Light'},
        },
        shadows: {
            default: 0.85,
            description: 'Darkness of the shadows voxels cast onto each other',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shadows', group: 'Light'},
        },
        shadowSoftness: {
            default: 0.35,
            description: 'Penumbra of the cast shadows — 0 is razor sharp, higher spreads the light into a soft area source whose shadows sharpen at contact',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shadow Softness', group: 'Light'},
        },
        ao: {
            default: 1,
            description: 'Ambient occlusion in the creases between voxels — the soft contact darkening that sells the geometry',
            ui: {type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Occlusion', group: 'Light'},
        },
        glossiness: {
            default: 0.35,
            description: 'How tight the specular highlight is — satin plastic at 0, lacquered at 1',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Glossiness', group: 'Surface'},
        },
        specular: {
            default: 0.6,
            description: 'Strength of the specular highlight and the fresnel rim on grazing faces',
            ui: {type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Specular', group: 'Surface'},
        },
        seams: {
            default: 0.2,
            description: 'Dark seam lines along every voxel edge — 0 for seamless blocks, higher for a drawn-outline brick look',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Seams', group: 'Surface'},
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: {type: 'shape', label: 'Shape', group: 'Shape'},
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file',
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type',
        },
    },

    // Voxels as a recipe: the voxel spine hands over the decoded frame (exact normal, baked AO, the
    // shadow-map shadow, cell identity, face edge, depth) and the look is lit here — a palette over
    // the frame, a wrapped key light cut by the shadow, hemisphere ambient occluded by the AO, a
    // Blinn–Phong lobe + Schlick rim, seam lines. The constants above ARE the look.
    ...voxelSurface({
        style: p('voxelShape'),
        gridSpace: p('gridSpace'),
        voxelSize: p('voxelSize'),
        fill: p('fill'),
        voxelScale: p('voxelScale'),
        bevel: p('bevel'),
        shadowSoftness: p('shadowSoftness'),
        lightAngle: p('lightAngle'),
        lightElevation: p('lightElevation'),
        depth: p('depth'),
        surface: (vox, _frame, params) => {
            const {uniforms: u, propValues} = params
            const n = vox.normal

            // ── Palette over the frame ──────────────────────────────────────────────────────
            const mixVariant = mixColorsIn(p('colorSpace'), params)
            const mixAB = (t: Expr): Expr => call(mixVariant, 'mixColors', [u.colorA, u.colorB, t]).member('rgb')
            // Up-facing faces have a negative material-space y normal.
            const upness = local(clamp(neg(n.member('y')), 0, 1), 'upness')
            let base: Expr
            switch (propValues.colorMode) {
                case 'height': base = mixAB(sub(1, vox.heightT)); break
                case 'depth': base = mixAB(vox.depthT); break
                case 'random': base = mixAB(vox.cellHash); break
                case 'faces': base = mixAB(sub(1, smoothstep(0.35, 0.75, upness))); break
                default: base = u.colorA.member('rgb')
            }
            const jitter = add(1, mul(sub(vox.cellHash, 0.5), mul(u.colorVariation, VARIATION_RANGE)))
            const albedo = local(mul(base, jitter), 'albedo')

            // ── Key light: wrapped Lambert, cut by the baked cast shadow ────────────────────
            const az = mul(u.lightAngle, DEG_TO_RAD)
            const el = mul(u.lightElevation, DEG_TO_RAD)
            const cosEl = local(cos(el), 'cosEl')
            const L = local(normalize(vec3(mul(cos(az), cosEl), mul(sin(az), cosEl), neg(sin(el)))), 'L')
            const ndl = local(lambert(n, L, {wrap: KEY_WRAP}), 'ndl')
            const lit = local(sub(1, mul(vox.shadow, u.shadows)), 'lit')
            const keyRgb = local(mul(u.lightColor.member('rgb'), u.lightIntensity), 'keyRgb')
            const direct = mul(keyRgb, mul(ndl, lit))

            // ── Ambient: sky over ground by up-ness, occluded by the baked AO ───────────────
            const occlusion = local(sub(1, mul(u.ao, sub(1, vox.ao))), 'occl')
            const sky = local(u.ambientColor.member('rgb'), 'sky')
            const hemi = mix(mul(sky, GROUND_GAIN), sky, upness)
            const ambient = local(mul(mul(hemi, u.ambient), max(occlusion, 0)), 'ambient')

            // ── Specular lobe + fresnel rim, both shadowed/occluded ─────────────────────────
            const view = vox.view
            const H = local(normalize(sub(L, view)), 'H')
            const ndh = clamp(dot(n, H), 0, 1)
            const shininess = local(exp2(mix(GLOSS_EXP_RANGE[0], GLOSS_EXP_RANGE[1], u.glossiness)), 'shininess')
            const lobe = mul(pow(ndh, shininess), mix(0.2, 1.4, u.glossiness))
            const specRgb = mul(keyRgb, mul(mul(lobe, u.specular), lit))
            const rim = schlickFresnel(grazingOf(n, view), {r0: 0.02, gain: mul(u.glossiness, 0.5)})
            const rimRgb = mul(sky, mul(mul(rim, mul(u.specular, 0.5)), max(occlusion, 0)))

            // ── Seams along the voxel edges ─────────────────────────────────────────────────
            const seamWidth = add(SEAM_WIDTH[0], mul(u.seams, SEAM_WIDTH[1]))
            const seamLine = smoothstep(sub(1, seamWidth), 1, vox.edge)
            const seamShade = sub(1, mul(mul(u.seams, SEAM_DARKEN), seamLine))

            // ── Compose ─────────────────────────────────────────────────────────────────────
            const rgb = mul(add(mul(albedo, add(direct, ambient)), add(specRgb, rimRgb)), seamShade)

            // One texel sample's color + hit coverage — the spine blends four per pixel and guards.
            return vec4(clamp(rgb, vec3(0, 0, 0), vec3(1, 1, 1)), vox.coverage)
        },
    }),
})

export default componentDefinition
