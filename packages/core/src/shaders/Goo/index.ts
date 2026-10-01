import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineStd} from "@coreroot/std"
import {shapedSurface, guarded, keyLightAt, grazingFlat} from "@coreroot/std/paint/materials"
import {direction} from "@coreroot/std/frames"
import {
    add, clamp, cos, div, dot, exp, float, le, local, max, min, mix, mul, normalize, pow, sin,
    smoothstep, sqrt, sub, vec3, vec4,
} from "@coreroot/std/math"
import type {Expr} from "@coreroot/gpu/porters"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"


const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

// Number of metaball charges. Build-time constant — the charge sum unrolls into the algebra with
// each charge's constants baked as literals.
const N_CHARGES = 7

// Deterministic JS hash for per-charge constants (frequencies, phases). Graph-build time only.
const jhash = (x: number): number => {
    const s = Math.sin(x * 12.9898) * 43758.5453
    return s - Math.floor(s)
}

// Per-charge baked constants: a golden-angle "sunflower" base position + hashed drift/breathe
// frequencies + phases. `posFreq[i] = (bx, by, sx, sy)`, `phase[i] = (br, phx, phy, phr)`.
const POS_FREQ: [number, number, number, number][] = []
const PHASE: [number, number, number, number][] = []
for (let i = 0; i < N_CHARGES; i++) {
    const theta = i * 2.399963
    const rr = Math.sqrt((i + 0.5) / N_CHARGES)
    POS_FREQ.push([rr * Math.cos(theta), rr * Math.sin(theta), 0.5 + jhash(i + 1), 0.5 + jhash(i + 7)])
    PHASE.push([0.5 + jhash(i + 13), jhash(i + 2) * 6.2831853, jhash(i + 5) * 6.2831853, jhash(i + 9) * 6.2831853])
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    gooColor: Parameters<typeof transformColor>[0]
    translucency: number
    absorb: number
    containment: number
    edgeSoftness: number
    blobScale: number
    spread: number
    merge: number
    bulge: number
    threshold: number
    lightAngle: number
    wetness: number
    fresnel: number
    specColor: Parameters<typeof transformColor>[0]
    ambient: number
    speed: number
    wobble: number
    breathe: number
    seed: number
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Goo",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Photoreal wet liquid — animated 3D metaball blobs that merge and bulge to loosely form the shape, with sliding wet highlights, a clearcoat rim and a translucent body",
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
            description: 'Center position of the goo',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the goo (1 = default size)',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the goo in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        gooColor: {
            default: '#37c95a',
            transform: transformColor,
            description: 'Base liquid color (a glossy slime green by default)',
            ui: { type: 'color', label: 'Goo Color', group: 'Goo' }
        },
        translucency: {
            default: 0.6,
            description: 'Subsurface glow strength — deep lobes transmit more color',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Translucency', group: 'Goo' }
        },
        absorb: {
            default: 2,
            description: 'How fast thin areas clear and thick areas saturate (Beer–Lambert)',
            ui: { type: 'range', min: 0, max: 6, step: 0.01, label: 'Absorption', group: 'Goo' }
        },
        containment: {
            default: 0.9,
            description: '0 = free blobs that ignore the shape, 1 = goo fills the shape and its surface conforms to the 3D form (highlights track the faces as it rotates)',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Containment', group: 'Goo' }
        },
        edgeSoftness: {
            default: 0.06,
            description: 'Softness of the blob coverage edge',
            ui: { type: 'range', min: 0, max: 0.3, step: 0.005, label: 'Edge Softness', group: 'Goo' }
        },
        blobScale: {
            default: 0.3,
            description: 'Base radius of each liquid lobe',
            ui: { type: 'range', min: 0.1, max: 1, step: 0.01, label: 'Blob Size', group: 'Blobs' }
        },
        spread: {
            default: 1,
            description: 'How far the lobes scatter across the shape',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Spread', group: 'Blobs' }
        },
        merge: {
            default: 0.6,
            description: 'How much the lobes fuse together — low = distinct beads, high = one fused mass',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Merge', group: 'Blobs' }
        },
        bulge: {
            default: 1,
            description: 'How 3D-rounded each lobe reads — dome height of the wet surface',
            ui: { type: 'range', min: 0, max: 1.5, step: 0.01, label: 'Bulge', group: 'Blobs' }
        },
        threshold: {
            default: 0.5,
            description: 'Iso-surface level — lower makes fatter, fuller blobs',
            ui: { type: 'range', min: 0.2, max: 0.9, step: 0.01, label: 'Threshold', group: 'Blobs' }
        },
        lightAngle: {
            default: 300,
            description: 'Light direction in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Lighting' }
        },
        wetness: {
            default: 2,
            description: 'Master scale on the sharp specular highlight and clearcoat rim',
            ui: { type: 'range', min: 0, max: 5, step: 0.01, label: 'Highlights', group: 'Lighting' }
        },
        fresnel: {
            default: 0.25,
            description: 'Clearcoat fresnel rim strength wrapping each lobe',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Fresnel', group: 'Lighting' }
        },
        specColor: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Highlight / rim color (white reads as a clear wet coat)',
            ui: { type: 'color', label: 'Highlight Color', group: 'Lighting' }
        },
        ambient: {
            default: 0.25,
            description: 'Base fill so shadowed goo isn\'t black',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Ambient', group: 'Lighting' }
        },
        speed: {
            default: 0.5,
            description: 'Drift and breathe speed of the blobs. 0 pauses.',
            ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Speed', group: 'Animation' }
        },
        wobble: {
            default: 0.5,
            description: 'Drift amplitude — how far the lobe centers wander',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Wobble', group: 'Animation' }
        },
        breathe: {
            default: 0.3,
            description: 'Radius pulsing amplitude — lobes swell and shrink',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Breathe', group: 'Animation' }
        },
        seed: {
            default: 1,
            description: 'Variation offset — shifts the blob arrangement',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Animation' }
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

    // The shape-effect spine: Goo reads the field's .r (membership) + .a (front-depth) only —
    // patternMode 'none', 'firstLobe' chord, and its own tight eps 0.004 forward stencil (the
    // metaball fill gradient). The wet-liquid material as algebra: an animated 7-charge metaball
    // potential (unrolled, constants baked) shaped by the container field, an antialiased
    // iso-contour coverage, a normal from the fill gradient + 3D shape relief, and wet Blinn
    // shading + Beer–Lambert subsurface — all inside the distance-cull guard, so far pixels pay
    // only the centre-distance test.
    ...shapedSurface({
        chord: 'firstLobe',
        stencil: {kind: 'forward', eps: 0.004},
        surface: (frame, params) => {
            const {uniforms: u} = params
            const t = local(animatedTime(params, 'seed'), 'gooT') // seed-offset drift/breathe clock (speed=0 pauses)
            const EPS = 0.004

            const s0 = local(frame.surf0!, 's0')
            const sdfUV = local(frame.sdfUV!, 'sdfUV')
            const px = local(sub(sdfUV.member('x'), 0.5), 'px')
            const py = local(sub(sdfUV.member('y'), 0.5), 'py')
            const distC = sqrt(add(mul(px, px), mul(py, py)))

            const spread = local(mul(u.spread, 0.45), 'spread')
            const driftAmp = local(mul(u.wobble, 0.12), 'driftAmp')
            const breatheAmp = local(u.breathe, 'breatheAmp')
            const radiusMul = add(0.65, mul(u.merge, 0.7))
            const baseR = local(mul(mul(u.blobScale, 0.38), radiusMul), 'baseR')
            const maxReach = add(add(add(spread, mul(baseR, add(1, breatheAmp))), driftAmp), 0.2)

            // Summed Gaussian metaball potential at 3 points (centre + x/y neighbour), one charge
            // at a time (build-time unroll over the baked charge constants).
            let potC: Expr | null = null
            let potX: Expr | null = null
            let potY: Expr | null = null
            for (let i = 0; i < N_CHARGES; i++) {
                const pf = POS_FREQ[i]
                const ph = PHASE[i]
                const cx = local(add(mul(pf[0], spread), mul(driftAmp, sin(add(mul(t, pf[2]), ph[1])))), `cx${i}`)
                const cy = local(add(mul(pf[1], spread), mul(driftAmp, cos(add(mul(t, pf[3]), ph[2])))), `cy${i}`)
                const rVal = local(mul(baseR, add(1, mul(breatheAmp, sin(add(mul(t, ph[0]), ph[3]))))), `rVal${i}`)
                const invR2 = local(div(1, max(mul(rVal, rVal), 0.0001)), `invR2_${i}`)
                const dcx = local(sub(px, cx), `dcx${i}`)
                const dcy = local(sub(py, cy), `dcy${i}`)
                const termC = exp(mul(mul(add(mul(dcx, dcx), mul(dcy, dcy)), invR2), -1))
                const dxx = local(sub(add(px, EPS), cx), `dxx${i}`)
                const termX = exp(mul(mul(add(mul(dxx, dxx), mul(dcy, dcy)), invR2), -1))
                const dyy = local(sub(add(py, EPS), cy), `dyy${i}`)
                const termY = exp(mul(mul(add(mul(dcx, dcx), mul(dyy, dyy)), invR2), -1))
                potC = potC ? add(potC, termC) : termC
                potX = potX ? add(potX, termX) : termX
                potY = potY ? add(potY, termY) : termY
            }

            // Soft inside-mask (reversed-edge smoothstep → 1 - smoothstep).
            const inside0 = local(sub(1, smoothstep(-0.06, 0.06, s0.member('x'))), 'inside0')
            const insideX = local(sub(1, smoothstep(-0.06, 0.06, frame.surfX!.member('x'))), 'insideX')
            const insideY = local(sub(1, smoothstep(-0.06, 0.06, frame.surfY!.member('x'))), 'insideY')

            // Occupancy height field: blend free blobs → shape-confined + filled.
            const contain = u.containment
            const H0 = local(add(mul(potC!, mix(float(1), inside0, contain)), mul(inside0, contain)), 'H0')
            const Hx = local(add(mul(potX!, mix(float(1), insideX, contain)), mul(insideX, contain)), 'Hx')
            const Hy = local(add(mul(potY!, mix(float(1), insideY, contain)), mul(insideY, contain)), 'Hy')

            const soft = local(max(u.edgeSoftness, 0.001), 'soft')
            const coverage = smoothstep(sub(u.threshold, soft), add(u.threshold, soft), H0)

            // Surface normal: metaball fill gradient (tilted) + 3D shape relief (front-depth gradient).
            const gradHx = div(sub(Hx, H0), EPS)
            const gradHy = div(sub(Hy, H0), EPS)
            const metaSlope = local(add(mul(u.bulge, 0.12), 0.03), 'metaSlope')
            const reliefGate = min(min(inside0, insideX), insideY)
            const shapeRelief = local(mul(mul(contain, 1.6), reliefGate), 'shapeRelief')
            const gradAx = clamp(div(sub(frame.surfX!.member('w'), s0.member('w')), EPS), -6, 6)
            const gradAy = clamp(div(sub(frame.surfY!.member('w'), s0.member('w')), EPS), -6, 6)
            const n = local(normalize(vec3(
                add(mul(mul(gradHx, -1), metaSlope), mul(gradAx, shapeRelief)),
                add(mul(mul(gradHy, -1), metaSlope), mul(gradAy, shapeRelief)),
                -1,
            )), 'n')

            // Dome height + optical thickness (dome + depth-into-shape).
            const domeZ = mul(sqrt(max(sub(H0, u.threshold), 0)), u.bulge)
            const shapeDepth = max(mul(s0.member('x'), -1), 0)
            const thick = local(clamp(add(domeZ, mul(mul(shapeDepth, contain), 1.2)), 0, 1), 'thick')

            // Wet shading.
            const kl = keyLightAt(direction(u.lightAngle, 'light'), -0.8)
            const ndl = local(clamp(dot(n, kl.member('L')), 0, 1), 'ndl')
            const ndh = local(clamp(dot(n, kl.member('H')), 0, 1), 'ndh')

            const specCore = mul(pow(ndh, 232), 1.7)
            const specHalo = mul(pow(ndh, 32), 0.25)
            const glint = mul(smoothstep(0.992, 0.999, ndh), 1.9)
            const fres = mul(pow(grazingFlat(n), 5), u.fresnel)
            const specRgb = mul(u.specColor.member('rgb'), mul(add(add(add(specCore, specHalo), glint), fres), u.wetness))

            // Subsurface (Beer–Lambert) + body diffuse + ambient.
            const transmit = exp(mul(mul(u.absorb, -1), sub(1, thick)))
            const gooRgb = local(u.gooColor.member('xyz'), 'gooRgb')
            const sss = mul(mul(gooRgb, mul(u.translucency, transmit)), add(mul(ndl, 0.5), 0.5))
            const body = mul(gooRgb, add(u.ambient, mul(ndl, 0.7)))
            const rgb = local(add(add(body, sss), specRgb), 'gooOut')

            // Far pixels skip the whole metaball evaluation (distance cull). `le` differs from the
            // original `!(distC > maxReach)` only on NaN inputs.
            return guarded(le(distC, maxReach), vec4(rgb, mul(coverage, u.gooColor.member('w'))), ZERO, 'goo')
        },
    }),
})

export default componentDefinition
