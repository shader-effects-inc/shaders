import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {call, ZERO} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {clamp, dot, local, mix, mul, sub, vec3} from "@coreroot/std/math"
import {hostFieldTexture, hostGridProgram, hostStep} from "@coreroot/std/sim/grids"
import {edges, blend, sampling, toHalfFloat} from "@coreroot/gpu/kit"
import {transformEdges} from "@coreroot/utilities/transformations"

// Which channel of the child decides how far a pixel gets thrown (applied by the bridge → propValues
// carries the mapped number; the read below stays robust to a raw string too).
const transformThrowKey = (value: string): number => {
    const map: Record<string, number> = { luminance: 0, darkness: 1, red: 2, green: 3, blue: 4 }
    return map[value] ?? 0
}

const GRID = 128
const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)

export interface ComponentProps {
    throwKey: string
    strength: number
    keyInfluence: number
    radius: number
    friction: number
    momentum: number
    edges: string
}

/** `uv - flowVec·scale` — the throw displacement, as declaration-site algebra. */
const offsetUV = (uv: Expr, flowVec: Expr, scale: Expr | number): Expr => sub(uv, mul(flowVec, scale))

/**
 * Bicubic B-spline tap setup for the GRID×GRID flow texture — the kit primitive, instantiated for
 * this grid size and named as it always was so the emitted WGSL is unchanged. See
 * `kit/sampling.ts` for why a low-resolution displacement field needs a C1 reconstruction (short
 * version: hardware bilinear is C0, so the field kinks at every cell border and a sharp child edge
 * dragged across it scallops).
 */
const {Taps: PixelThrowBicubicTaps, setup: pixelThrowBicubicSetup} =
    sampling.bSplineTapSetup(GRID, {fnName: 'pixelThrowBicubicSetup', structName: 'PixelThrowBicubicTaps'})
export {PixelThrowBicubicTaps, pixelThrowBicubicSetup}

/**
 * The compile-time throw key for `keyMode` (0=luminance,1=darkness,2/3/4=r/g/b), then
 * `keyScale = mix(1, clamp(key,0,1), keyInfluence)` — how far this pixel is dragged along the flow.
 */
function keyScaleOf(keyMode: number, rgb: Expr, keyInfluence: Expr): Expr {
    const lum = vec3(0.299, 0.587, 0.114)
    const key = keyMode === 1 ? sub(1, dot(rgb, lum))   // darkness
        : keyMode === 2 ? rgb.member('r')               // red
        : keyMode === 3 ? rgb.member('g')               // green
        : keyMode === 4 ? rgb.member('b')               // blue
        : dot(rgb, lum)                                 // luminance
    return mix(1, clamp(key, 0, 1), keyInfluence)
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "PixelThrow",
    role: 'simulation',
    species: 'custom',
    category: "Interactive",
    description: "Throws pixels along the cursor's path like a fluid — brighter (or redder/greener/bluer) pixels are flung farther, then settle back with friction",
    requiresRTT: true,
    requiresChild: true,
    boundingBoxDeclaration: { aspectRatio: null },
    usesPointer: true,
    props: {
        throwKey: {
            default: "luminance",
            transform: transformThrowKey,
            compileTime: true,
            description: "Which pixel value decides how far a pixel is thrown",
            ui: {
                type: 'select',
                options: [
                    { label: 'Brightness', value: 'luminance' },
                    { label: 'Darkness', value: 'darkness' },
                    { label: 'Red', value: 'red' },
                    { label: 'Green', value: 'green' },
                    { label: 'Blue', value: 'blue' }
                ],
                label: 'Throw By',
                group: 'Effect'
            }
        },
        strength: {
            default: 0.25,
            description: "Maximum distance pixels can be thrown",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        keyInfluence: {
            default: 0.8,
            description: "How much the chosen value modulates throw distance (0 = all pixels thrown equally, 1 = only high-value pixels move)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Key Influence', group: 'Effect' }
        },
        radius: {
            default: 0.2,
            description: "Size of the cursor's throw brush",
            ui: { type: ['range', 'map'], min: 0.02, max: 0.6, step: 0.01, label: 'Radius', group: 'Effect' }
        },
        friction: {
            default: 0.3,
            description: "How quickly thrown pixels settle back to their origin (0 = float almost forever, 1 = snap back fast)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Friction', group: 'Effect' }
        },
        momentum: {
            default: 0.5,
            description: "How much the thrown flow keeps drifting and swirling after the cursor passes (fluidity)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Momentum', group: 'Effect' }
        },
        edges: {
            default: 'stretch',
            description: 'How to handle pixels thrown beyond the canvas edges',
            transform: transformEdges,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    { label: 'Stretch', value: 'stretch' },
                    { label: 'Transparent', value: 'transparent' },
                    { label: 'Mirror', value: 'mirror' },
                    { label: 'Wrap', value: 'wrap' }
                ],
                label: 'Edges',
                group: 'Effect'
            }
        }
    },

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {uniforms, ctx, childNode, convertToTexture, onBeforeRender, dimensions, propValues, getCpuValue} = params
        if (!childNode) return ZERO
        const child = convertToTexture(childNode)

        const edgeMode = (propValues.edges as number) ?? 0
        const rawKey = propValues.throwKey
        const keyMode = typeof rawKey === 'number' ? rawKey : transformThrowKey(String(rawKey ?? 'luminance'))

        // ── Persistent fluid throw-field (CPU grid sim, ChromaFlow-shaped) ────────────────────
        // Each cell holds a 2D displacement (UV units) packed into an rgba16float texture's .rg; the
        // cursor injects velocity, the field self-advects (momentum) and decays to zero (friction).
        // The loops ARE the effect's arithmetic — expressed as the std host frame program.
        const flow = new Float32Array(GRID * GRID * 2)
        const temp = new Float32Array(GRID * GRID * 2)
        const field = hostFieldTexture(params, {size: GRID, label: 'pixelthrow-flow'})
        const flowKit = field.texture

        let prevX = 0.5, prevY = 0.5
        let velX = 0, velY = 0
        let speed = 0
        let maxThrow = 0.125

        const sampleFlow = (gx: number, gy: number, ch: number): number => {
            const x0 = Math.floor(gx), y0 = Math.floor(gy)
            const x1 = x0 + 1, y1 = y0 + 1
            if (x0 < 0 || y0 < 0 || x1 >= GRID || y1 >= GRID) return 0
            const fx = gx - x0, fy = gy - y0
            const i00 = (y0 * GRID + x0) * 2 + ch
            const i01 = (y0 * GRID + x1) * 2 + ch
            const i10 = (y1 * GRID + x0) * 2 + ch
            const i11 = (y1 * GRID + x1) * 2 + ch
            return flow[i00] * (1 - fx) * (1 - fy) + flow[i01] * fx * (1 - fy)
                + flow[i10] * (1 - fx) * fy + flow[i11] * fx * fy
        }

        const aspectOf = () => Math.max(0.0001, dimensions.width / Math.max(1, dimensions.height))

        onBeforeRender(hostGridProgram({clampDt: 0.016, steps: [
            hostStep('dtGuard', (f) => (f.dt <= 0 ? 'skip' : undefined)),
            hostStep('cursorVelocity', (f) => {
                const rawVelX = (f.pointer.x - prevX) / f.dt
                const rawVelY = (f.pointer.y - prevY) / f.dt
                velX = velX * 0.8 + rawVelX * 0.2
                velY = velY * 0.8 + rawVelY * 0.2
                speed = Math.sqrt(velX * velX + velY * velY)
            }),
            hostStep('advect+decay', (f) => {
                const friction = num(getCpuValue('friction'), 0.3)
                const momentum = num(getCpuValue('momentum'), 0.5)
                maxThrow = num(getCpuValue('strength'), 0.25) * 0.5
                const decay = Math.max(0, 1 - friction * f.dt * 4)
                const advect = momentum * f.dt * 12

                for (let i = 0; i < GRID; i++) {
                    for (let j = 0; j < GRID; j++) {
                        const idx = (i * GRID + j) * 2
                        const srcX = j - flow[idx] * GRID * advect
                        const srcY = i - flow[idx + 1] * GRID * advect
                        temp[idx] = sampleFlow(srcX, srcY, 0) * decay
                        temp[idx + 1] = sampleFlow(srcX, srcY, 1) * decay
                    }
                }
            }),
            hostStep('inject(brush)', (f) => {
                if (speed <= 0.02) return
                const aspect = aspectOf()
                const r = Math.max(0.001, num(getCpuValue('radius'), 0.2))
                const r2 = r * r
                // Gaussian falloff SHIFTED to reach exactly zero at the brush edge: a raw
                // exp(-d²/r²) truncated at the radius leaves a 1/e (~37%) step, which the
                // GRID-cell lattice renders as a visible staircase around the brush.
                const EDGE = Math.exp(-1)
                const INV_EDGE = 1 / (1 - EDGE)
                const minI = Math.max(0, Math.floor((f.pointer.y - r) * GRID))
                const maxI = Math.min(GRID - 1, Math.ceil((f.pointer.y + r) * GRID))
                const minJ = Math.max(0, Math.floor((f.pointer.x - r) * GRID))
                const maxJ = Math.min(GRID - 1, Math.ceil((f.pointer.x + r) * GRID))
                for (let i = minI; i <= maxI; i++) {
                    for (let j = minJ; j <= maxJ; j++) {
                        const cellX = (j + 0.5) / GRID
                        const cellY = (i + 0.5) / GRID
                        const dx = aspect >= 1 ? (cellX - f.pointer.x) * aspect : (cellX - f.pointer.x)
                        const dy = aspect >= 1 ? (cellY - f.pointer.y) : (cellY - f.pointer.y) / aspect
                        const distSq = dx * dx + dy * dy
                        if (distSq > r2) continue
                        const influence = (Math.exp(-distSq / r2) - EDGE) * INV_EDGE
                        const idx = (i * GRID + j) * 2
                        temp[idx] += velX * influence * f.dt * 4
                        temp[idx + 1] += velY * influence * f.dt * 4
                    }
                }
            }),
            // Soft-saturate the displacement MAGNITUDE (tanh) instead of hard-clamping each
            // component: a fast swipe saturates the field in a frame or two, and a hard clamp
            // leaves a flat plateau whose kinked boundary reads as a staircase on the
            // GRID-cell lattice (the "stepping" along the trail).
            hostStep('saturate(tanh)', () => {
                for (let k = 0; k < GRID * GRID; k++) {
                    const fx = temp[k * 2]
                    const fy = temp[k * 2 + 1]
                    const len = Math.sqrt(fx * fx + fy * fy)
                    if (len > 1e-6) {
                        const s = (maxThrow * Math.tanh(len / maxThrow)) / len
                        temp[k * 2] = fx * s
                        temp[k * 2 + 1] = fy * s
                    }
                }
            }),
            hostStep('publish(halfFloat)', (f) => {
                flow.set(temp)
                // Pack RG displacement into the rgba16float texture's .rg (b/a unused).
                for (let k = 0; k < GRID * GRID; k++) {
                    field.texData[k * 4] = toHalfFloat(flow[k * 2])
                    field.texData[k * 4 + 1] = toHalfFloat(flow[k * 2 + 1])
                }
                field.upload()
                prevX = f.pointer.x
                prevY = f.pointer.y
            }),
        ]}))

        // ── Fragment: drag the child along the flow, scaled by the pixel's key ────
        const uv = ctx.uv
        // Bicubic B-spline flow sample (4 hardware-bilinear taps) — see pixelThrowBicubicSetup.
        const taps = call(pixelThrowBicubicSetup, 'pixelThrowBicubicSetup', [uv])
        const tapW = taps.member('w')
        // Bound to a local: the flow vector feeds both the probe and the final displacement, and
        // re-emitting it would double the four texture taps.
        const flowVec = local(flowKit.sample(taps.member('uvA'), 'linearClamp').member('xy').mul(tapW.member('x'))
            .add(flowKit.sample(taps.member('uvB'), 'linearClamp').member('xy').mul(tapW.member('y')))
            .add(flowKit.sample(taps.member('uvC'), 'linearClamp').member('xy').mul(tapW.member('z')))
            .add(flowKit.sample(taps.member('uvD'), 'linearClamp').member('xy').mul(tapW.member('w'))), 'flowVec')
        // Key from the (lightly pre-displaced) sample so the thrown distance reflects the pixel being dragged.
        const probeUV = offsetUV(uv, flowVec, 0.5)
        const probe = call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [child.sample(probeUV)])
        const keyScale = keyScaleOf(keyMode, probe.member('rgb'), uniforms.keyInfluence)
        const displacedUV = local(offsetUV(uv, flowVec, keyScale), 'displacedUV')
        const sampled = edges.applyEdgeHandlingExpr(displacedUV, (u) => child.sample(u), edgeMode)
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sampled])
    }}
})

export default componentDefinition
