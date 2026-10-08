import type {Expr, GpuShaderDefinition} from "@coreroot/gpu/porters"
import {animatedTime} from "@coreroot/gpu/porters"
import {defineStd, p, uniformOf} from "@coreroot/std"
import {direction, directionFrame, surfaceOf} from "@coreroot/std/frames"
import {clothCoords, drapedFolds, foldSet, type DrapeOptions} from "@coreroot/std/paint/noise"
import {
    anisoSpecular, fdCurvature, fdSlope, grainNoise, lambert, lightVec3, neutralTone, nudgeNormal,
    sheenLobe, tiltNormal, viewRay, wardAlphas,
} from "@coreroot/std/paint/materials"
import {abs, add, clamp, div, local, mix, mul, neg, sub, vec2, vec3, vec4} from "@coreroot/std/math"
import {transformColor} from "@coreroot/utilities/transformations"
import {isMobileGpuViewport} from "@coreroot/utilities/device"

// Device tier, chosen once at build time (the Surface3D / sdf3d pattern; SSR + tests → desktop).
// The fold field is the dominant per-pixel cost — every tap re-sums the folds — so mobile takes
// the 3-tap slope (no cavity term) and two ripple octaves instead of three.
const MOBILE = isMobileGpuViewport()

// ── The Satin look (this data IS the look) ────────────────────────────────────────────────
// The ripple each fold's spine wanders by (freq along the drape, drift on the clock, amount
// across), the swell that merges and parts neighbouring ridges, the ridge profile, the thread
// grain, the finite-difference tap and the lighting gains. Tune here, not in the recipe.
const RIPPLE = [
    {freq: 3.1, speed: 0.35, amount: 0.045},
    {freq: 6.7, speed: -0.22, amount: 0.018},
    {freq: 11.3, speed: 0.5, amount: 0.007},
]
const RIPPLE_WAVES = MOBILE ? RIPPLE.slice(0, 2) : RIPPLE
const SWELL = {amount: 0.3, freq: 2.4, speed: 0.4}
const BREEZE_SWAY = 0.08
// Ridges are flat-topped faces meeting in tight creases (0 would be round bells), laid out
// with plenty of sideways jitter so no two gaps match.
const FOLD_CREASE = 0.7
const FOLD_SHARPNESS = 2
const FOLD_JITTER = 0.8
// How much cloth each flank consumes — the threads bunch up there as the surface tilts away.
const FOLD_FORESHORTEN = 0.25
const FD_EPS = 0.004
// Height-field slope → normal tilt; the fold depth prop scales the height, this sets the drama
// (0.3 puts the steepest flank near 60°).
const RELIEF_GAIN = 0.3
const CAVITY_GAIN = 0.0012
// Thread grain in cloth coordinates: short glints along the thread (first freq), fine across it.
const THREADS = {coarse: [14, 420] as [number, number], fine: [26, 900] as [number, number]}
const WEAVE_RELIEF = 0.1
const LIGHT = {ambient: 0.18, key: 0.9, fill: 0.45, elevation: -0.65, fillElevation: -0.45}
// The finish: the streak's gain across the gloss range, the broad bloom lobe behind it (rough,
// half the streak's anisotropy), and the velvet sheen gain.
const STREAK_GAIN: [number, number] = [0.25, 0.9]
const BLOOM = {roughness: 0.85, anisotropy: 0.5, gain: 0.35}
const SHEEN_GAIN = 0.6

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    highlightColor: Parameters<typeof transformColor>[0]
    foldCount: number
    foldDirection: number
    foldDepth: number
    foldWidth: number
    pinning: number
    seed: number
    speed: number
    breeze: number
    gloss: number
    anisotropy: number
    bloom: number
    sheen: number
    weave: number
    fiberAngle: number
    lightAngle: number
    fill: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Satin",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Flowing satin — a draped cloth of soft rippling folds lit like real fabric: a highlight streaked along the weave, a velvet sheen on the fold flanks and creases falling into shadow, swaying gently in a breeze",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        color: {
            default: '#c92c40',
            transform: transformColor,
            description: 'The dye of the cloth — deep colors show the sheen best',
            ui: { type: 'color', label: 'Color', group: 'Cloth' }
        },
        highlightColor: {
            default: '#c4667b',
            transform: transformColor,
            description: 'The color of the light caught by the weave — near white for silk, tinted toward the dye for a matte satin',
            ui: { type: 'color', label: 'Highlight', group: 'Cloth' }
        },
        foldCount: {
            default: 7,
            compileTime: true,
            description: 'How many ridges run across the cloth',
            ui: { type: 'range', min: 1, max: 12, step: 1, label: 'Folds', group: 'Folds' }
        },
        foldDirection: {
            default: 90,
            description: 'Direction the folds run, in degrees — 90 hangs them vertically like a curtain',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Direction', group: 'Folds' }
        },
        foldDepth: {
            default: 0.5,
            description: 'How deep the folds are — 0 is a flat sheet, 1 heavy gathered drapery',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Depth', group: 'Folds' }
        },
        foldWidth: {
            default: 0.14,
            description: 'Width of each ridge, relative to the canvas — narrow for fine pleats, wide for broad soft folds',
            ui: { type: ['range', 'map'], min: 0.04, max: 0.4, step: 0.01, label: 'Width', group: 'Folds' }
        },
        pinning: {
            default: 0.6,
            description: 'How much the cloth is gathered at the top edge — 0 is a loose drape, 1 a curtain pinned at the rail with the folds opening as they fall',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Pinning', group: 'Folds' }
        },
        seed: {
            default: 0,
            description: 'Random seed — a different layout of folds',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Folds' }
        },
        speed: {
            default: 1,
            description: 'How fast the cloth moves. 0 pauses.',
            ui: { type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Motion' }
        },
        breeze: {
            default: 0.4,
            description: 'Strength of the breeze — how far the folds ripple and sway',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Breeze', group: 'Motion' }
        },
        gloss: {
            default: 0.1,
            description: 'How shiny the weave is — 0 soft matte satin, 1 glossy silk with a tight bright streak',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Gloss', group: 'Finish' }
        },
        anisotropy: {
            default: 0.7,
            description: 'How much the highlight stretches along the threads — the long streak real satin shows',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Anisotropy', group: 'Finish' }
        },
        bloom: {
            default: 0.5,
            description: 'The broad soft glow around the highlight — light the weave scatters sideways. 0 leaves the crisp streak alone, 1 a wide satin bloom',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Bloom', group: 'Finish' }
        },
        sheen: {
            default: 0.5,
            description: 'The velvet glow on the flanks of each fold where the fibres catch the light edge-on',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Sheen', group: 'Finish' }
        },
        weave: {
            default: 0.1,
            description: 'Visibility of the thread texture — the fine grain that glints inside the highlight',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Weave', group: 'Finish' }
        },
        fiberAngle: {
            default: 90,
            description: 'Direction the threads run, in degrees — the highlight streaks along this axis. Usually the fold direction.',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Thread Angle', group: 'Finish' }
        },
        lightAngle: {
            default: 215,
            description: 'Direction of the key light, in degrees',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Light' }
        },
        fill: {
            default: 0.35,
            description: 'Strength of the fill light from the opposite side — lifts the shadowed flanks',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Fill', group: 'Light' }
        },
    },

    // Satin as a recipe: a set of folds laid across the drape direction → their height, slope and
    // curvature at this pixel → a normal, combed across by the thread grain → a key and a fill →
    // a highlight streaked along the threads, a broad bloom behind it, a velvet sheen on the
    // flanks → dyed, creases shadowed, tone-mapped. The constants above are the look; the words
    // are the language.
    paint: (params) => {
        const {uv, aspect} = surfaceOf(params)
        const u = (name: keyof ComponentProps): Expr => uniformOf(p(name), params)
        const t = local(animatedTime(params), 'satinTime')
        // The canvas as a centred, aspect-true plane: x spans ±aspect/2, y spans ±1/2, y down.
        const q = local(vec2(mul(sub(uv.member('x'), 0.5), aspect), sub(uv.member('y'), 0.5)), 'q')

        // ── The drape: ridges laid across the fold direction, rippling and swaying along it ──
        const drape = directionFrame(u('foldDirection'), 'drape')
        const count = Math.max(1, Math.min(12, Math.round((params.propValues.foldCount as number) ?? 7)))
        // The folds cover the canvas's extent across the drape, with a margin so the outer ridges
        // sit at the edges rather than inside them.
        const spread = local(mul(add(mul(abs(drape.perp.member('x')), aspect), abs(drape.perp.member('y'))), 1.1), 'spread')
        const folds = foldSet({count, seed: u('seed'), spread, width: u('foldWidth'), height: u('foldDepth'), jitter: FOLD_JITTER})
        const breeze = local(u('breeze'), 'breeze')
        const drapeOpts: DrapeOptions = {
            time: t,
            ripple: {waves: RIPPLE_WAVES, gain: mix(0.35, 1, breeze)},
            sway: {amount: mul(breeze, BREEZE_SWAY), rate: 0.7},
            swell: SWELL,
            pinning: local(u('pinning'), 'pinning'),
            sharpness: FOLD_SHARPNESS,
            crease: FOLD_CREASE,
        }
        // The drape frame's `along` runs 0 at the pinned edge to 1 at the far edge over the canvas's
        // whole extent in the fold direction, whatever the angle — so the pinning envelope's ends
        // sit on the canvas edges, never as a kink inside the view.
        const alongExtent = local(add(mul(abs(drape.tangent.member('x')), aspect), abs(drape.tangent.member('y'))), 'alongExtent')
        const onDrape = (at: Expr): {along: Expr; across: Expr} => {
            const c = drape.coordsOf(at)
            return {along: add(div(c.along, alongExtent), 0.5), across: c.across}
        }
        const cloth = (at: Expr): Expr => drapedFolds(onDrape(at), folds, drapeOpts)

        // ── Relief: the cloth's slope (and, on desktop, curvature) → the fold normal ──
        const curved = MOBILE ? null : fdCurvature(cloth, q, FD_EPS, 'cloth')
        const relief = curved ?? fdSlope(cloth, q, FD_EPS, 'cloth')
        const nFold = nudgeNormal(vec3(0, 0, -1), mul(relief.dx, -RELIEF_GAIN), mul(relief.dy, -RELIEF_GAIN), 'nFold')

        // ── The weave: threads drawn in the cloth's own coordinates, so they bend with the folds
        //    and bunch on the flanks, combing the normal across the thread direction ──
        const material = clothCoords(onDrape(q), folds, {...drapeOpts, foreshorten: FOLD_FORESHORTEN})
        const onCloth = local(add(mul(drape.tangent, material.along), mul(drape.perp, material.across)), 'onCloth')
        const fibre = directionFrame(u('fiberAngle'), 'fibre')
        const fc = fibre.coordsOf(onCloth)
        const threads = local(add(
            grainNoise(fc, {freq: THREADS.coarse}),
            mul(grainNoise(fc, {freq: THREADS.fine, offset: [3.1, 7.7]}), 0.5),
        ), 'threads')
        const n = tiltNormal(nFold, fibre, {across: mul(threads, mul(u('weave'), WEAVE_RELIEF))})

        // ── Lights: a key from the light angle, a softer fill from across, a little ambient ──
        const view = viewRay(params, {aspect})
        const keyDir = direction(u('lightAngle'), 'key')
        const L = lightVec3(keyDir, LIGHT.elevation, 'L')
        const Lfill = lightVec3(neg(keyDir), LIGHT.fillElevation, 'Lfill')
        const diffuse = local(add(
            add(LIGHT.ambient, mul(lambert(n, L, {wrap: 0.35}), LIGHT.key)),
            mul(lambert(n, Lfill, {wrap: 0.6}), mul(u('fill'), LIGHT.fill)),
        ), 'diffuse')
        // Creases fall into shadow (desktop only — the mobile tier skips the two curvature taps).
        const cavity: Expr | number = curved
            ? local(clamp(sub(1, mul(curved.curvature, CAVITY_GAIN)), 0.6, 1.1), 'cavity')
            : 1

        // ── The finish: a streak along the threads, a broad bloom behind it, a velvet sheen on the flanks ──
        const gloss = local(u('gloss'), 'gloss')
        const aniso = local(u('anisotropy'), 'aniso')
        const streak = anisoSpecular({
            normal: n, tangent: fibre.tangent, light: L, view,
            alphas: wardAlphas(sub(1, gloss), aniso),
            gain: mix(STREAK_GAIN[0], STREAK_GAIN[1], gloss),
        })
        const bloom = anisoSpecular({
            normal: n, tangent: fibre.tangent, light: L, view,
            alphas: wardAlphas(BLOOM.roughness, mul(aniso, BLOOM.anisotropy)),
            gain: mul(u('bloom'), BLOOM.gain),
        })
        const velvet = sheenLobe({normal: n, light: L, view, roughness: mix(0.7, 0.3, gloss), gain: mul(u('sheen'), SHEEN_GAIN)})

        // ── Dye + light ──
        const dye = u('color').member('rgb')
        const catchlight = u('highlightColor').member('rgb')
        const rgb = add(
            mul(dye, mul(diffuse, cavity)),
            mul(catchlight, mul(add(add(streak, bloom), velvet), cavity)),
        )
        return vec4(neutralTone(rgb), 1)
    },
})

export default componentDefinition
