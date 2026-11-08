import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {call, floatE, ZERO} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {voronoiRegionField, cellLaneUVFor, crackGeom, crackRefractUV, crackShardCompose} from "@coreroot/std/effects/fracture"
import {edges, blend, toHalfFloat} from "@coreroot/gpu/kit"
import {transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    crackWidth: number
    intensity: number
    radius: number
    decay: number
    seed: number
    chromaticSplit: number
    refractionStrength: number
    shardLighting: number
    edges: string
}

const seededRandom = (seed: number): number => {
    const x = Math.sin(seed) * 10000
    return x - Math.floor(x)
}

const SHARD_COUNT = 16
const VORONOI_SIZE = 1024
// Per-cell data texture: r16float, 64 wide (16 cells × [posX, posY, dispX, dispY]), 1 tall.
const DATA_WIDTH = SHARD_COUNT * 4
const cellLaneUV = cellLaneUVFor(DATA_WIDTH)

// ── Host parts ───────────────────────────────────────────────────────────────────────────────
// The pipeline, as named parts: `shardSites` (deterministic cell centres) → `voronoiRegions`
// (GPU region field, re-marched on seed change) + `shardPhysics` (CPU spring integration) →
// `shardDataTexture` (the r16float row the fragment samples by cell ID) → the crack-render taps
// (crackGeom / crackRefractUV / crackShardCompose from std/effects/fracture).

/**
 * Deterministic shard sites for a seed: [posX, posY] per shard — the Voronoi cell centres.
 * Shared by `voronoiRegions` AND `shardPhysics`: the data texture's positions must match the
 * compute field's cell centres exactly, or cracks detach from their shards.
 */
const shardSites = (seed: number): Float32Array => {
    const positions = new Float32Array(SHARD_COUNT * 2)
    for (let i = 0; i < SHARD_COUNT; i++) {
        positions[i * 2] = seededRandom(seed + i * 2)
        positions[i * 2 + 1] = seededRandom(seed + i * 2 + 1)
    }
    return positions
}

/**
 * Recipe: shardPhysics — the per-frame CPU spring integration over the 16 shards. Each shard
 * carries a displacement that decays exponentially toward rest (`exp(-dt/decay)`) and receives a
 * cursor push (velocity-proportional, squared-falloff influence inside `radius`) plus a per-shard
 * random jitter, blended in at `FRICTION·dt`. Stays on the CPU deliberately: 16 shards × a handful
 * of ops is far below dispatch cost, and the cursor-velocity intermediates (clamped speed) are
 * shared by every shard — a serial host loop, not a kernel.
 */
function shardPhysics(getCpuValue: (key: string) => unknown) {
    const cellData = new Float32Array(SHARD_COUNT * 4) // posX, posY, randDirX, randDirY
    const displacementData = new Float32Array(SHARD_COUNT * 4)

    const generateCells = (seed: number): void => {
        const sites = shardSites(seed)
        for (let i = 0; i < SHARD_COUNT; i++) {
            cellData[i * 4] = sites[i * 2]
            cellData[i * 4 + 1] = sites[i * 2 + 1]
            cellData[i * 4 + 2] = seededRandom(seed + i * 3)
            cellData[i * 4 + 3] = seededRandom(seed + i * 3 + 1)
        }
    }

    let currentSeed = (getCpuValue('seed') as number) ?? 2
    generateCells(currentSeed)

    const FRICTION = 1
    let prevX = 0.5
    let prevY = 0.5

    return {
        cellData, displacementData,
        step(pointer: {x: number; y: number}, deltaTime: number, aspect: number): void {
            const dt = Math.min(deltaTime, 0.016)

            const newSeed = (getCpuValue('seed') as number) ?? 2
            if (newSeed !== currentSeed) {
                currentSeed = newSeed
                generateCells(currentSeed)
                displacementData.fill(0)
            }

            const intensity = (getCpuValue('intensity') as number) ?? 4
            const radius = (getCpuValue('radius') as number) ?? 0.4
            const decay = (getCpuValue('decay') as number) ?? 1

            let velX = dt > 0 ? (pointer.x - prevX) / dt : 0
            let velY = dt > 0 ? (pointer.y - prevY) / dt : 0
            let speed = Math.sqrt(velX * velX + velY * velY)
            const maxVelocity = 5 + intensity * 2
            if (speed > maxVelocity) {
                const scale = maxVelocity / speed
                velX *= scale
                velY *= scale
                speed = maxVelocity
            }

            for (let i = 0; i < SHARD_COUNT; i++) {
                const cellX = cellData[i * 4]
                const cellY = cellData[i * 4 + 1]
                const randomDirX = cellData[i * 4 + 2] - 0.5
                const randomDirY = cellData[i * 4 + 3] - 0.5
                const dx = aspect >= 1.0 ? (cellX - pointer.x) * aspect : cellX - pointer.x
                const dy = aspect >= 1.0 ? cellY - pointer.y : (cellY - pointer.y) / aspect
                const dist = Math.sqrt(dx * dx + dy * dy)

                let currentDx = displacementData[i * 4]
                let currentDy = displacementData[i * 4 + 1]
                const decayTime = Math.max(0.01, decay)
                const decayFactor = Math.exp(-dt / decayTime)
                currentDx *= decayFactor
                currentDy *= decayFactor

                let velocityDx = 0
                let velocityDy = 0
                if (dist < radius && speed > 0.01) {
                    const influence = Math.max(0, 1 - dist / radius)
                    const influenceCurve = influence * influence
                    const pushForce = influenceCurve * speed * intensity * dt * 0.5
                    velocityDx = velX * pushForce
                    velocityDy = velY * pushForce
                    const jitterForce = influenceCurve * speed * intensity * dt * 0.1
                    velocityDx += randomDirX * jitterForce
                    velocityDy += randomDirY * jitterForce
                }

                const lerpFactor = Math.min(1, FRICTION * dt)
                displacementData[i * 4] = currentDx + velocityDx * lerpFactor
                displacementData[i * 4 + 1] = currentDy + velocityDy * lerpFactor
            }

            prevX = pointer.x
            prevY = pointer.y
        },
    }
}

/**
 * Part: shardDataTexture — packs [posX, posY, dispX, dispY] per shard into the r16float row the
 * fragment samples by Voronoi cell ID (`cellDataUV`). `upload()` re-packs from the physics arrays
 * and writes the texture, once per frame.
 */
function shardDataTexture(params: GpuFragmentParams, physics: ReturnType<typeof shardPhysics>) {
    const {createDataTexture, registerMediaTexture, onCleanup} = params
    const dataMirror = new Float32Array(DATA_WIDTH)
    const texData = new Uint16Array(DATA_WIDTH)
    const packMirror = (): void => {
        for (let i = 0; i < SHARD_COUNT; i++) {
            dataMirror[i * 4] = physics.cellData[i * 4]
            dataMirror[i * 4 + 1] = physics.cellData[i * 4 + 1]
            dataMirror[i * 4 + 2] = physics.displacementData[i * 4]
            dataMirror[i * 4 + 3] = physics.displacementData[i * 4 + 1]
        }
        for (let i = 0; i < DATA_WIDTH; i++) texData[i] = toHalfFloat(dataMirror[i])
    }
    packMirror()

    const dataTex = createDataTexture({width: DATA_WIDTH, height: 1, format: 'r16float', data: texData, label: 'shatter-cells'})
    onCleanup(() => dataTex.destroy())
    const dataKit = registerMediaTexture(() => dataTex.texture)

    return {
        dataKit,
        upload(): void {
            packMirror()
            dataTex.write(texData)
        },
    }
}

// std custom simulation: a GPU Voronoi precompute + CPU per-frame shard physics feeding a
// hand-rolled RTT fragment (data-texture reads and per-channel refraction taps).
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Shatter",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Broken glass effect with tectonic plate displacement",
    requiresRTT: true,
    requiresChild: true,
    usesPointer: true,
    props: {
        crackWidth: {
            default: 1,
            description: 'Thickness of crack lines',
            ui: { type: ['range', 'map'], min: 0.5, max: 5, step: 0.1, label: 'Crack Width', group: 'Effect' }
        },
        intensity: {
            default: 4,
            description: 'How much shards shift',
            ui: { type: 'range', min: 0, max: 20, step: 1, label: 'Intensity', group: 'Effect' }
        },
        radius: {
            default: 0.4,
            description: 'Cursor influence radius',
            ui: { type: 'range', min: 0.1, max: 1, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        decay: {
            default: 1,
            description: 'How fast shards return to rest',
            ui: { type: 'range', min: 0.1, max: 10, step: 0.1, label: 'Decay', group: 'Effect' }
        },
        seed: {
            default: 2,
            description: 'Random seed for pattern',
            ui: { type: 'range', min: 0, max: 50, step: 1, label: 'Seed', group: 'Effect' }
        },
        chromaticSplit: {
            default: 1,
            description: 'RGB separation for prismatic glass effect',
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.1, label: 'Chromatic Split', group: 'Effect' }
        },
        refractionStrength: {
            default: 5,
            description: 'How much cracks bend/distort the underlying image',
            ui: { type: ['range', 'map'], min: 0, max: 10, step: 0.1, label: 'Refraction', group: 'Effect' }
        },
        shardLighting: {
            default: 0.1,
            description: 'Subtle lighting on tilted shards for 3D depth',
            ui: { type: 'range', min: 0, max: 0.5, step: 0.1, label: 'Shard Lighting', group: 'Effect' }
        },
        edges: {
            default: 'mirror',
            description: 'How to handle edges when displacement pushes content out of bounds',
            transform: transformEdges,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Transparent', value: 'transparent'},
                    {label: 'Mirror', value: 'mirror'},
                    {label: 'Wrap', value: 'wrap'}
                ],
                label: 'Edges',
                group: 'Effect'
            }
        }
    },

    // WebGPU compute: the Voronoi region field — precomputed once and re-dispatched only when
    // `seed` changes. Per-frame shard physics stays on CPU (16 shards) — see the fragment's
    // `shardPhysics` + `shardDataTexture` parts.
    compute: voronoiRegionField({
        count: SHARD_COUNT, size: VORONOI_SIZE, sites: shardSites,
        seedProp: 'seed', seedDefault: 2, output: 'voronoiField',
    }),

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {childNode, ctx, computeOutputs, propValues, convertToTexture} = params
        if (!childNode) return ZERO
        const child = convertToTexture(childNode)
        const voronoiField = computeOutputs?.voronoiField as KitTexture | undefined
        if (!voronoiField) {
            // GPU-free (no device) / no compute: passthrough (unpremultiplied).
            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [child.sample(ctx.uv)])
        }

        const {onBeforeRender, getCpuValue, dimensions} = params
        const edgeMode = (propValues.edges as number) ?? 0

        // Host pipeline: shardPhysics integrates displacements, shardDataTexture publishes them.
        const physics = shardPhysics(getCpuValue)
        const cells = shardDataTexture(params, physics)
        const dataKit = cells.dataKit
        onBeforeRender((fp: unknown) => {
            const {pointer, deltaTime} = fp as {pointer: {x: number; y: number}; deltaTime: number}
            const aspect = dimensions.width / Math.max(1, dimensions.height)
            physics.step(pointer, deltaTime, aspect)
            cells.upload()
        })

        // ── Crack render ──────────────────────────────────────────────────────────────────
        const vSample = voronoiField.sample(ctx.uv, 'nearestClamp')
        const nearestI = vSample.member('r')
        const secondI = vSample.member('g')
        const nearestClamp = 'nearestClamp' as const
        const dataAt = (idx: Expr, field: number): Expr =>
            dataKit.sample(call(cellLaneUV, 'cellLaneUV', [idx, floatE(field)]), nearestClamp).member('r')

        const geom = call(crackGeom, 'crackGeom', [
            ctx.uv,
            dataAt(nearestI, 0), dataAt(nearestI, 1),
            dataAt(secondI, 0), dataAt(secondI, 1),
            dataAt(nearestI, 2), dataAt(nearestI, 3),
            params.uniforms.crackWidth,
        ])
        const crackIntensity = geom.member('crackIntensity')
        const edgeNormal = geom.member('edgeNormal')
        const displacedUV = geom.member('displacedUV')
        const disp = geom.member('disp')

        const sample = (uv: Expr): Expr => edges.applyEdgeHandlingExpr(uv, (u) => child.sample(u), edgeMode)
        const normalColor = sample(displacedUV)
        const refractUV = (channel: number): Expr =>
            call(crackRefractUV, 'crackRefractUV', [
                displacedUV, edgeNormal, crackIntensity, params.uniforms.refractionStrength, params.uniforms.chromaticSplit, floatE(channel),
            ])
        const rFinal = sample(refractUV(1))
        const gFinal = sample(refractUV(0))
        const bFinal = sample(refractUV(-1))

        const composed = call(crackShardCompose, 'crackShardCompose', [
            normalColor, rFinal, gFinal, bFinal, crackIntensity, disp, params.uniforms.shardLighting,
        ])
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [composed])
    }},
})

export default componentDefinition
