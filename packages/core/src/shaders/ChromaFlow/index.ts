import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {call, flowDirectionColor} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {add, vec2} from "@coreroot/std/math"
import {hostFieldTexture, hostGridProgram, hostStep} from "@coreroot/std/sim/grids"
import {toHalfFloat} from "@coreroot/gpu/kit"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    baseColor: Parameters<typeof transformColor>[0]
    upColor: Parameters<typeof transformColor>[0]
    downColor: Parameters<typeof transformColor>[0]
    leftColor: Parameters<typeof transformColor>[0]
    rightColor: Parameters<typeof transformColor>[0]
    intensity: number
    radius: number
    momentum: number
}

const GRID_SIZE = 128
const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ChromaFlow",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Interactive",
    description: "Interactive liquid flow effect that follows your cursor",
    usesPointer: true,
    props: {
        baseColor: {
            default: "#0066ff",
            transform: transformColor,
            description: "Base liquid color",
            ui: { type: 'color', label: 'Base Color', group: 'Colors' }
        },
        upColor: {
            default: "#00ff00",
            transform: transformColor,
            description: "Color for upward movement",
            ui: { type: 'color', label: 'Up Color', group: 'Colors' }
        },
        downColor: {
            default: "#ff0000",
            transform: transformColor,
            description: "Color for downward movement",
            ui: { type: 'color', label: 'Down Color', group: 'Colors' }
        },
        leftColor: {
            default: "#0000ff",
            transform: transformColor,
            description: "Color for leftward movement",
            ui: { type: 'color', label: 'Left Color', group: 'Colors' }
        },
        rightColor: {
            default: "#ffff00",
            transform: transformColor,
            description: "Color for rightward movement",
            ui: { type: 'color', label: 'Right Color', group: 'Colors' }
        },
        intensity: {
            default: 1,
            description: 'Strength of the liquid effect',
            ui: { type: 'range', min: 0.5, max: 1.5, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        radius: {
            default: 3,
            description: 'Radius of the liquid effect',
            ui: { type: 'range', min: 0, max: 5, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        momentum: {
            default: 30,
            description: 'How much momentum colors retain in their flow direction',
            ui: { type: 'range', min: 10, max: 60, step: 1, label: 'Momentum', group: 'Effect' }
        }
    },

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {uniforms, ctx, uvContext, onBeforeRender, dimensions, getCpuValue} = params

        // Packed field: r,g = flow vector, b = liquid density (a unused). One rgba16float texture,
        // filterable (linearClamp) — a filtering sampler rejects r32float, so half-encode each texel
        // before upload (the std host-field part owns the texture + staging buffer).
        const fieldData = new Float32Array(GRID_SIZE * GRID_SIZE * 4)
        const tempFieldData = new Float32Array(GRID_SIZE * GRID_SIZE * 4)
        const field = hostFieldTexture(params, {size: GRID_SIZE, label: 'chromaflow-field'})
        const fieldKit = field.texture

        // Mouse tracking + running field max (idle-skip dirty flag) — pure CPU state (the fragment
        // reads only the field texture + color uniforms, so these need no GPU uniforms). This is a
        // CPU grid sim: the loops ARE the effect's arithmetic, expressed as the std frame program.
        let mouseVelX = 0, mouseVelY = 0
        let prevX = 0.5, prevY = 0.5
        let fieldMax = 0
        let aspect = 1
        let injecting = false
        let newMax = 0
        let intensity = 1
        let radius = 0.15

        onBeforeRender(hostGridProgram({clampDt: 0.016, steps: [
            hostStep('cursorVelocity', (f) => {
                aspect = dimensions.width / Math.max(1, dimensions.height)
                const velX = f.dt > 0 ? (f.pointer.x - prevX) / f.dt : 0
                const velY = f.dt > 0 ? (f.pointer.y - prevY) / f.dt : 0
                mouseVelX = mouseVelX * 0.85 + velX * 0.15
                mouseVelY = mouseVelY * 0.85 + velY * 0.15
                prevX = f.pointer.x
                prevY = f.pointer.y
                injecting = Math.abs(velX) + Math.abs(velY) > 0.01
            }),
            // Idle skip: both fields decay multiplicatively, so once everything is below epsilon and
            // the cursor is still, the sim step + upload are dead work.
            hostStep('idleSkip', () => (!injecting && fieldMax < 1e-4 ? 'skip' : undefined)),
            // Decay flow + advect liquid density into the temp buffer. Cell layout: [0]=flowX,
            // [1]=flowY, [2]=density.
            hostStep('advect+decay', (f) => {
                intensity = num(getCpuValue('intensity'), 1)
                radius = num(getCpuValue('radius'), 3) * 0.05
                const momentum = num(getCpuValue('momentum'), 30)

                const flowFadeRate = 1 - f.dt / Math.max(0.1, 1.0)
                const liquidFadeRate = 1 - f.dt
                const flowSpeed = momentum * 50 * f.dt
                newMax = 0

                for (let i = 0; i < GRID_SIZE; i++) {
                    for (let j = 0; j < GRID_SIZE; j++) {
                        const idx = (i * GRID_SIZE + j) * 4
                        const fx0 = fieldData[idx]
                        const fy0 = fieldData[idx + 1]
                        tempFieldData[idx] = fx0 * flowFadeRate
                        tempFieldData[idx + 1] = fy0 * flowFadeRate
                        tempFieldData[idx + 2] = fieldData[idx + 2] * liquidFadeRate

                        if (Math.abs(fx0) > 0.001 || Math.abs(fy0) > 0.001) {
                            const advectX = j - fx0 * flowSpeed
                            const advectY = i - fy0 * flowSpeed
                            const x0 = Math.floor(advectX)
                            const y0 = Math.floor(advectY)
                            const x1 = x0 + 1
                            const y1 = y0 + 1
                            if (x0 >= 0 && y0 >= 0 && x1 < GRID_SIZE && y1 < GRID_SIZE) {
                                const fx = advectX - x0
                                const fy = advectY - y0
                                const idx00 = (y0 * GRID_SIZE + x0) * 4
                                const idx01 = (y0 * GRID_SIZE + x1) * 4
                                const idx10 = (y1 * GRID_SIZE + x0) * 4
                                const idx11 = (y1 * GRID_SIZE + x1) * 4
                                const sampledLiquid =
                                    fieldData[idx00 + 2] * (1 - fx) * (1 - fy) +
                                    fieldData[idx01 + 2] * fx * (1 - fy) +
                                    fieldData[idx10 + 2] * (1 - fx) * fy +
                                    fieldData[idx11 + 2] * fx * fy
                                tempFieldData[idx + 2] = sampledLiquid * liquidFadeRate
                            }
                        }
                        const m = Math.max(Math.abs(tempFieldData[idx]), Math.abs(tempFieldData[idx + 1]), tempFieldData[idx + 2])
                        if (m > newMax) newMax = m
                    }
                }
            }),
            hostStep('inject(brush)', (f) => {
                if (!injecting) return
                const speed = Math.sqrt(mouseVelX * mouseVelX + mouseVelY * mouseVelY)
                const speedScale = Math.min(speed * speed * 20, 1.0)
                const effectiveRadius = radius * speedScale
                const maxDistSq = (effectiveRadius * 2) * (effectiveRadius * 2)
                const radSq = effectiveRadius * effectiveRadius
                const addScale = intensity * 100 * f.dt * 0.01
                const speedMultiplier = Math.min(speed * 10, 1.0)
                for (let i = 0; i < GRID_SIZE; i++) {
                    for (let j = 0; j < GRID_SIZE; j++) {
                        const cellX = (j + 0.5) / GRID_SIZE
                        const cellY = (i + 0.5) / GRID_SIZE
                        const dx = aspect >= 1.0 ? (cellX - f.pointer.x) * aspect : (cellX - f.pointer.x)
                        const dy = aspect >= 1.0 ? (cellY - f.pointer.y) : (cellY - f.pointer.y) / aspect
                        const distSq = dx * dx + dy * dy
                        if (distSq < maxDistSq) {
                            const idx = (i * GRID_SIZE + j) * 4
                            const influence = Math.exp(-distSq / radSq)
                            tempFieldData[idx] += mouseVelX * influence * addScale
                            tempFieldData[idx + 1] += mouseVelY * influence * addScale
                            tempFieldData[idx + 2] += influence * addScale * speedMultiplier
                            tempFieldData[idx] = Math.max(-1, Math.min(1, tempFieldData[idx]))
                            tempFieldData[idx + 1] = Math.max(-1, Math.min(1, tempFieldData[idx + 1]))
                            tempFieldData[idx + 2] = Math.max(0, Math.min(1, tempFieldData[idx + 2]))
                        }
                    }
                }
            }),
            hostStep('publish(halfFloat)', () => {
                fieldMax = injecting ? 1 : newMax
                fieldData.set(tempFieldData)
                for (let k = 0; k < field.texData.length; k++) field.texData[k] = toHalfFloat(fieldData[k])
                field.upload()
            }),
        ]}))

        // One 5-tap smoothing cross over the packed field (each tap serves both density .z and
        // flow .xy); the directional color mapping is the kit's flow-field display primitive.
        const uv = uvContext ?? ctx.uv
        const px = 1.0 / GRID_SIZE
        const clamp = 'linearClamp' as const
        const tap = (dx: number, dy: number): Expr => fieldKit.sample(add(uv, vec2(dx, dy)), clamp)
        const s0 = tap(0, 0)
        const s1 = tap(px, 0)
        const s2 = tap(0, px)
        const s3 = tap(-px, 0)
        const s4 = tap(0, -px)

        return call(flowDirectionColor, 'flowDirectionColor', [
            s0, s1, s2, s3, s4,
            uniforms.baseColor, uniforms.upColor, uniforms.downColor, uniforms.leftColor, uniforms.rightColor,
        ])
    }}
})

export default componentDefinition
