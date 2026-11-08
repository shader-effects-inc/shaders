import type {Expr, GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p, schema, uniformOf} from "@coreroot/std"
import {surfaceOf} from "@coreroot/std/frames"
import {vignetteMask, glowSpot, glowAt} from "@coreroot/std/paint/light"
import {cpuHash01} from "@coreroot/std/signal"
import {
    add, clamp, div, exp, float, local, max, mix, mul, neg, smoothstep, splat3, sub, vec3, vec4,
} from "@coreroot/std/math"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"

const TAU = Math.PI * 2

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    brightness: number
    keyColor: Parameters<typeof transformColor>[0]
    keyIntensity: number
    keySoftness: number
    fillColor: Parameters<typeof transformColor>[0]
    fillIntensity: number
    fillSoftness: number
    fillAngle: number
    backColor: Parameters<typeof transformColor>[0]
    backIntensity: number
    backSoftness: number
    center: Parameters<typeof transformPosition>[0]
    lightTarget: number
    wallCurvature: number
    vignette: number
    ambientIntensity: number
    ambientSpeed: number
    seed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "StudioBackground",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Multi-light studio background with ambient motion.",
    // The four drifting ambient lights' per-frame state, computed on the CPU (the light-rig
    // part's onBeforeRender driver): orbit offsets from the (centerX, centerY − 0.2) anchor packed
    // pairwise, plus the squared glow radii. Pixel-invariant, so keeping them in-shader cost
    // 28 sin per pixel.
    extraFields: {
        ambPosA: { schema: schema.vec4f, initial: [0, 0, 0, 0] },
        ambPosB: { schema: schema.vec4f, initial: [0, 0, 0, 0] },
        ambSizeSq: { schema: schema.vec4f, initial: [0.01, 0.01, 0.01, 0.01] }
    },
    props: {
        color: {
            default: '#d8dbec',
            transform: transformColor,
            description: 'Base studio surface color',
            ui: { type: 'color', label: 'Surface Color', group: 'Colors' }
        },
        keyColor: {
            default: '#d5e4ea',
            transform: transformColor,
            description: 'Color of the overhead key light',
            ui: { type: 'color', label: 'Key Color', group: 'Key Light' }
        },
        keyIntensity: {
            default: 40,
            description: 'Intensity of the key light',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Key Light' }
        },
        keySoftness: {
            default: 50,
            description: 'How diffuse the key light is',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Softness', group: 'Key Light' }
        },
        fillColor: {
            default: '#d5e4ea',
            transform: transformColor,
            description: 'Color of the side fill lights',
            ui: { type: 'color', label: 'Fill Color', group: 'Fill Lights' }
        },
        fillIntensity: {
            default: 10,
            description: 'Intensity of the fill lights',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Fill Lights' }
        },
        fillSoftness: {
            default: 70,
            description: 'How diffuse the fill lights are',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Softness', group: 'Fill Lights' }
        },
        fillAngle: {
            default: 70,
            description: 'How far apart the fill lights are from center',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Spread', group: 'Fill Lights' }
        },
        backColor: {
            default: '#c8d4e8',
            transform: transformColor,
            description: 'Color of the upward back wash',
            ui: { type: 'color', label: 'Back Color', group: 'Back Light' }
        },
        backIntensity: {
            default: 20,
            description: 'Intensity of the back wash',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Intensity', group: 'Back Light' }
        },
        backSoftness: {
            default: 80,
            description: 'How diffuse the back wash is',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Softness', group: 'Back Light' }
        },
        brightness: {
            default: 20,
            description: 'Overall ambient light level',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Ambient', group: 'Lighting' }
        },
        vignette: {
            default: 0,
            description: 'Edge darkening',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Vignette', group: 'Lighting' }
        },
        center: {
            default: { x: 0.5, y: 0.8 },
            transform: transformPosition,
            description: 'Where the spotlight meets the floor',
            ui: { type: 'position', label: 'Stage Center', group: 'Scene' }
        },
        lightTarget: {
            default: 100,
            description: 'How far toward the floor vs wall the spotlights aim',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Light Depth', group: 'Scene' }
        },
        wallCurvature: {
            default: 10,
            description: 'How rounded the cove is',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Wall Curvature', group: 'Scene' }
        },
        ambientIntensity: {
            default: 50,
            description: 'Intensity of drifting ambient lights',
            ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Ambient Detail', group: 'Environment' }
        },
        ambientSpeed: {
            default: 2,
            description: 'Drift speed',
            ui: { type: 'range', min: -5, max: 5, step: 0.1, label: 'Speed', group: 'Environment' }
        },
        seed: {
            default: 0,
            description: 'Seed for ambient pattern',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Environment' }
        }
    },

    // The recipe: the fused light rig (key/fill/back/bounce over the cove geometry + the four
    // CPU-orbited ambient glows) → the vignette mask → the lit surface composite.
    // The StudioBackground look: a cove (seamless wall-meets-floor backdrop) lit by a
    // photographic rig — key + two fills from above, a back wash, a bounce off the crease,
    // and four slowly orbiting neutral glows — then vignetted onto the surface color.
    paint: (params) => {
        const {uniforms} = params
        const {uv, viewport} = surfaceOf(params)
        // Guarded aspect (a zero-height canvas frame must not render NaN).
        const aspect = local(div(viewport.member('x'), max(viewport.member('y'), 1e-6)), 'aspect')
        const u = (name: Parameters<typeof p>[0]) => uniformOf(p(name), params)

        // Prop conditioning: sliders are 0–100 ranges, the rig works in unit terms.
        const brightness = local(mul(u('brightness'), 0.02), 'sbBrightness')
        const keyInt = local(mul(u('keyIntensity'), 0.015), 'sbKeyInt')
        const fillInt = local(mul(u('fillIntensity'), 0.015), 'sbFillInt')
        const fillOffset = mul(u('fillAngle'), 0.005)
        const backInt = mul(u('backIntensity'), 0.015)
        const lightTarget = local(mul(u('lightTarget'), 0.01), 'sbTarget')
        const curvature = local(mul(u('wallCurvature'), 0.01), 'sbCurvature')
        // Per-light Gaussian decay (low softness = tight, high = wide).
        const keyDecay = mix(5.0, 0.8, mul(u('keySoftness'), 0.01))
        const fillDecay = mix(5.0, 0.8, mul(u('fillSoftness'), 0.01))
        const backDecay = local(mix(5.0, 0.8, mul(u('backSoftness'), 0.01)), 'sbBackDecay')

        // The cove frame (an authoring-time mixin): floor/wall transition, wall-normalized
        // heights, centered x, cone width and the squared wall/floor falloffs — bound once,
        // read by every light.
        const centerX = local(u('center').member('x'), 'coveCenterX')
        const centerY = local(sub(1, u('center').member('y')), 'coveCenterY')
        const transitionWidth = local(mix(0.015, 0.15, curvature), 'coveTransition')
        const isFloor = local(smoothstep(sub(centerY, transitionWidth), add(centerY, transitionWidth), uv.member('y')), 'coveIsFloor')
        const notFloor = local(sub(1, isFloor), 'coveNotFloor')
        const wallNormY = local(clamp(div(uv.member('y'), max(centerY, 1e-6)), 0, 1), 'coveWallY')
        const wallNormY2 = local(mul(wallNormY, wallNormY), 'coveWallY2')
        const floorNormY = clamp(div(sub(uv.member('y'), centerY), max(sub(1, centerY), 1e-6)), 0, 1)
        const uvXCentered = local(mul(sub(uv.member('x'), centerX), aspect), 'coveX')
        const coneWidth = local(mix(0.05, 0.45, wallNormY2), 'coveCone')
        const wallFallY = mul(sub(1, wallNormY), 1.2)
        const wallFallY2 = local(mul(wallFallY, wallFallY), 'coveWallFall')
        const floorPoolY = add(centerY, mul(sub(1, centerY), 0.3))
        const floorDy = mul(sub(uv.member('y'), floorPoolY), 4)
        const floorDy2 = local(mul(floorDy, floorDy), 'coveFloorFall')

        // The unlit base: wall/floor ambient gradient + the cove-crease glow.
        const ambient = mul(mix(mix(0.2, 0.4, wallNormY), mix(0.38, 0.18, floorNormY), isFloor), brightness)
        const curveDist = div(sub(uv.member('y'), centerY), add(transitionWidth, 0.001))
        const curveGlow = mul(mul(mul(exp(mul(mul(-0.5, curveDist), curveDist)), curvature), brightness), 0.1)
        const neutralBase = local(add(ambient, curveGlow), 'sbBase')

        // Wall/floor bias from lightTarget (0 = wall focus, 1 = floor focus).
        const wallBias = local(add(mul(sub(1, lightTarget), 1.4), 0.3), 'sbWallBias')
        const floorBias = local(add(mul(lightTarget, 1.4), 0.3), 'sbFloorBias')

        // A downward spotlight (an authoring-time mixin, shared by key + fills): a wall cone
        // narrowing toward the source and a pool where it lands on the floor.
        const downSpot = (sourceX: Expr, decay: Expr): Expr => {
            const beamX = mix(sourceX, centerX, wallNormY2)
            const wDx = div(mul(sub(uv.member('x'), beamX), aspect), coneWidth)
            const wD2 = add(mul(wDx, wDx), wallFallY2)
            const wall = mul(mul(exp(mul(neg(wD2), decay)), notFloor), wallBias)
            const poolX = mix(centerX, sourceX, 0.25)
            const fDx = mul(mul(sub(uv.member('x'), poolX), aspect), 1.4)
            const fD2 = add(mul(fDx, fDx), floorDy2)
            const pool = mul(mul(exp(mul(neg(fD2), mul(decay, 1.5))), isFloor), floorBias)
            return add(wall, pool)
        }

        // The rig, accumulated in the look's layering order.
        let light: Expr = local(vec3(neutralBase, neutralBase, neutralBase), 'sbLight0')
        // Key light (top center, straight down).
        light = add(light, mul(u('keyColor').member('rgb'), mul(local(downSpot(centerX, keyDecay), 'sbKey'), keyInt)))
        // Fill lights (top sides, angled inward).
        const fills = add(
            mul(local(downSpot(sub(centerX, fillOffset), fillDecay), 'sbFillL'), fillInt),
            mul(local(downSpot(add(centerX, fillOffset), fillDecay), 'sbFillR'), fillInt))
        light = add(light, mul(u('fillColor').member('rgb'), fills))
        // Back light: an upward wash — a floor pool + a wall wash, each an anisotropic glow.
        const backFloor = mul(glowSpot(mul(uvXCentered, 0.5), mul(sub(uv.member('y'), mix(1.0, centerY, 0.15)), 4), backDecay), isFloor)
        const backWall = mul(mul(glowSpot(mul(uvXCentered, 0.6), mul(sub(centerY, uv.member('y')), 3), mul(backDecay, 0.8)), notFloor), wallNormY2)
        light = add(light, mul(u('backColor').member('rgb'), mul(add(backFloor, backWall), backInt)))
        // Bounce off the crease from the key.
        const bounce = mul(mul(glowSpot(mul(uvXCentered, 0.7), mul(add(sub(uv.member('y'), centerY), 0.02), 10), float(2.5)), mul(keyInt, 0.12)), notFloor)
        light = add(light, splat3(bounce))
        // Four slowly orbiting neutral ambient glows (positions CPU-integrated per frame).
        const ambAnchorY = sub(centerY, 0.2)
        const orbits = add(add(add(
            glowAt({uv, aspect, x: add(centerX, uniforms.ambPosA.member('x')), y: add(ambAnchorY, uniforms.ambPosA.member('y')), sizeSq: uniforms.ambSizeSq.member('x')}),
            glowAt({uv, aspect, x: add(centerX, uniforms.ambPosA.member('z')), y: add(ambAnchorY, uniforms.ambPosA.member('w')), sizeSq: uniforms.ambSizeSq.member('y')})),
            glowAt({uv, aspect, x: add(centerX, uniforms.ambPosB.member('x')), y: add(ambAnchorY, uniforms.ambPosB.member('y')), sizeSq: uniforms.ambSizeSq.member('z')})),
            glowAt({uv, aspect, x: add(centerX, uniforms.ambPosB.member('z')), y: add(ambAnchorY, uniforms.ambPosB.member('w')), sizeSq: uniforms.ambSizeSq.member('w')}))
        light = add(light, splat3(mul(orbits, mul(mul(u('ambientIntensity'), 0.012), 0.25))))

        // CPU orbit driver: accumulate the ambient clock (deltaTime × ambientSpeed) and publish
        // the four lights' orbit offsets + squared radii each frame; the seed-hash constants
        // recompute only when `seed` changes. The clock restarts on recompose — a phase-only
        // shift of a slow neutral drift, invisible in practice.
        let ambTime = 0
        let lastSeed = Number.NaN
        let orbitConsts: {sizeSq: number; radiusX: number; radiusY: number; freqX: number; freqY: number; phaseX: number; phaseY: number}[] = []
        params.onBeforeRender((fp?: {deltaTime?: number}) => {
            ambTime += (fp?.deltaTime ?? 0) * ((params.getCpuValue('ambientSpeed') as number) ?? 2)
            const seed = (params.getCpuValue('seed') as number) ?? 0
            if (seed !== lastSeed) {
                lastSeed = seed
                orbitConsts = []
                for (let i = 0; i < 4; i++) {
                    const sd = seed + i * 73.1
                    const sizeVal = cpuHash01(sd + 311.0) * 0.12 + 0.1
                    orbitConsts.push({
                        sizeSq: sizeVal * sizeVal,
                        radiusX: cpuHash01(sd) * 0.5 + 0.2,
                        radiusY: cpuHash01(sd + 37.0) * 0.25 + 0.1,
                        freqX: cpuHash01(sd + 91.0) * 0.3 + 0.1,
                        freqY: cpuHash01(sd + 143.0) * 0.25 + 0.08,
                        phaseX: cpuHash01(sd + 200.0) * TAU,
                        phaseY: cpuHash01(sd + 257.0) * TAU,
                    })
                }
            }
            const off = orbitConsts.map((c) => [
                Math.sin(ambTime * c.freqX + c.phaseX) * c.radiusX,
                Math.cos(ambTime * c.freqY + c.phaseY) * c.radiusY,
            ])
            params.setExtraField('ambPosA', [off[0][0], off[0][1], off[1][0], off[1][1]])
            params.setExtraField('ambPosB', [off[2][0], off[2][1], off[3][0], off[3][1]])
            params.setExtraField('ambSizeSq', [orbitConsts[0].sizeSq, orbitConsts[1].sizeSq, orbitConsts[2].sizeSq, orbitConsts[3].sizeSq])
        })

        // Composite: clamp the accumulated light (headroom to 3), tint the surface color by it,
        // darken by the vignette, clamp to display range. Opaque.
        const vig = vignetteMask({strength: p('vignette'), scale: 0.025})(params)
        const lit = clamp(mul(mul(u('color').member('rgb'), clamp(light, splat3(0), splat3(3))), vig), splat3(0), splat3(1))
        return vec4(lit, 1)
    },
})

export default componentDefinition
