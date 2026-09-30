/**
 * Push fields the pointer stirs: a grid of displacement that follows the cursor and settles
 * back on its own, for warps. Each word is a whole simulation. Spread it into the definition
 * with `...sim.grids.gridSim(word)`, then read the field it publishes from a warp map
 * (`warps.gridCellDisplace` or `warps.liquidDisplace`) by its `output` name.
 *
 * `pointerSplatField` smears pixels along the cursor and lets them fade back.
 * `springLatticeField` is a cloth of springs that ripples and rings after a push. Both need
 * `usesPointer: true` on the definition, run only while there is motion to settle, and give
 * a displacement in uv per grid cell.
 */
// Maintainer: each noun is a complete grid-sim recipe that integrates a cursor-driven vec2
// displacement field on the GPU and publishes it as an RG texture the warp maps sample.
// pointerSplatField — a memoryless decaying field: per-cell dissipation + a Gaussian
// cursor-velocity splat. springLatticeField — a spring-mass cloth: structural + shear springs with
// return-to-rest, semi-implicit Euler, cursor impulse injection. The kernels and bind-group
// layouts are the machinery (hidden from the reference); definitions only bind props.
import type {GpuFragmentParams} from '../../gpu/contract'
import {createGuardedCompute, createStateBuffer} from '../../gpu/porters'
import {tgpu, d, std} from '../../gpu/kit/index'
import {op, trackedViewport, type GridSimConfig, type GridRoot} from '../sim/grids'
import type {PropRef} from '../values'

const STATE_FORMAT = 'rgba16float' as const

// ═══ Pointer-splat field ══════════════════════════════════════════════════════════════════════

/** The smallest grid `pointerSplatField` runs on: 8 cells per side. */
export const SPLAT_MIN_GRID = 8
/** The largest grid `pointerSplatField` runs on: 128 cells per side. */
export const SPLAT_MAX_GRID = 128
// The state buffer is fixed at the MAX cell count (module-scope layout) — the kernels index only
// the active gridSize² via a baked size literal, so the buffer never needs to resize.
const SPLAT_MAX_CELLS = SPLAT_MAX_GRID * SPLAT_MAX_GRID

/**
 * The grid size `pointerSplatField` really uses for a prop value: floored, then held to 8–128.
 *
 * Use it in the `gridSize` prop's recompile rule, so the simulation rebuilds only when the cell
 * count changes and not on every drag of the slider.
 *
 * @example
 * ```ts
 * gridSize: {default: 20, recompile: recompileWhen((prev, next) => clampSplatGridSize(prev as number) !== clampSplatGridSize(next as number))}
 * ```
 * @see pointerSplatField
 */
export const clampSplatGridSize = (v: number): number => Math.max(SPLAT_MIN_GRID, Math.min(SPLAT_MAX_GRID, Math.floor(v)))

// ── Compute bind-group layouts (per-cell vec4 = (dispX, dispY, 0, 0)). ───────────────────────
const splatUpdateLayout = tgpu.bindGroupLayout({
    buffer: {storage: d.arrayOf(d.vec4f, SPLAT_MAX_CELLS), access: 'mutable'},
    params: {uniform: d.struct({
        cursorX: d.f32, cursorY: d.f32, mouseVelX: d.f32, mouseVelY: d.f32,
        dt: d.f32, decay: d.f32, intensity: d.f32, radius: d.f32, aspect: d.f32,
    })},
})
const splatOutputLayout = tgpu.bindGroupLayout({
    buffer: {storage: d.arrayOf(d.vec4f, SPLAT_MAX_CELLS), access: 'readonly'},
    dispTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

/** @internal The splat update kernel factory behind `pointerSplatField`. */
// Grid size baked: per-cell dissipation + Gaussian cursor-velocity splat + clamp. No neighbour
// reads → single buffer, no ping-pong. Every cell updated.
export function makeSplatUpdateKernel(gridSize: number) {
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = splatUpdateLayout.$.params
        const idx = cy * d.u32(gridSize) + cx
        const cell = splatUpdateLayout.$.buffer[idx]
        let dx = cell.x * (1.0 - p.decay * p.dt)
        let dy = cell.y * (1.0 - p.decay * p.dt)
        const cellX = (d.f32(cx) + 0.5) / d.f32(gridSize)
        const cellY = (d.f32(cy) + 0.5) / d.f32(gridSize)
        const ddx = std.select(cellX - p.cursorX, (cellX - p.cursorX) * p.aspect, p.aspect >= 1.0)
        const ddy = std.select((cellY - p.cursorY) / p.aspect, cellY - p.cursorY, p.aspect >= 1.0)
        const distSq = ddx * ddx + ddy * ddy
        const radius2 = p.radius * 2.0
        if (distSq < radius2 * radius2 && std.abs(p.mouseVelX) + std.abs(p.mouseVelY) > 0.01) {
            const influence = std.exp((distSq / (p.radius * p.radius)) * -1.0)
            dx = dx + p.mouseVelX * influence * p.intensity * p.dt * 0.5
            dy = dy + p.mouseVelY * influence * p.intensity * p.dt * 0.5
        }
        dx = std.clamp(dx, -1.0, 1.0)
        dy = std.clamp(dy, -1.0, 1.0)
        splatUpdateLayout.$.buffer[idx] = d.vec4f(dx, dy, 0.0, 0.0)
    }).$name('pointerSplatUpdate')
}

/** @internal The splat output kernel factory behind `pointerSplatField`. */
// Grid size baked: copy the per-cell displacement into the RG texture.
export function makeSplatOutputKernel(gridSize: number) {
    return tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const idx = cy * d.u32(gridSize) + cx
        const cell = splatOutputLayout.$.buffer[idx]
        std.textureStore(splatOutputLayout.$.dispTex, d.vec2u(cx, cy), d.vec4f(cell.x, cell.y, 0.0, 1.0))
    }).$name('pointerSplatOutput')
}

/**
 * A field the cursor smears: cells near the pointer move with it, then fade back to rest.
 *
 * Spread it with `...sim.grids.gridSim(...)`. `gridSize` is the cells per side (8 to 128,
 * see `clampSplatGridSize`). `decay` (about 0 to 10) is how fast the smear fades; `intensity`
 * (about 0 to 5) how hard the cursor pushes; `radius` the brush size (1 is about 5% of the
 * canvas). The field is published under `output` for a warp map such as
 * `warps.gridCellDisplace`.
 *
 * @example
 * ```ts
 * ...sim.grids.gridSim(pointerSplatField({gridSize: p('gridSize'), decay: p('decay'), intensity: p('intensity'), radius: p('radius'), output: 'displacement'}))
 * ```
 * @tip Add `uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement` so the warp is a no-op before the field exists.
 * @see springLatticeField, clampSplatGridSize
 */
// Maintainer: a mouse-driven grid displacement field on a single vec4 state buffer
// (createStateBuffer; WebGPU zero-inits → flat), as the std grid pipeline: cursor-velocity
// smoothing → settle gate (settle time derived from decay) → params write → the in-place update
// pass (dissipate + splat + clamp, one fused kernel) → the output publish under `slots.output`.
// CHILD-INDEPENDENT (no bindInputs). The grid resolution is baked per-compose from
// `slots.gridSize` (kernel factories) — pair that prop with a `clampSplatGridSize` recompile rule.
export function pointerSplatField(slots: {
    gridSize: PropRef
    decay: PropRef
    intensity: PropRef
    radius: PropRef
    output: string
}): (params: GpuFragmentParams, root: GridRoot) => GridSimConfig {
    return (params, root) => {
        const {getCpuValue, registerComputeTexture, onCleanup} = params
        const gridSize = clampSplatGridSize((getCpuValue(slots.gridSize.name) as number) ?? 20)

        const buffer = createStateBuffer(root, d.vec4f, SPLAT_MAX_CELLS)
        const dispTex = root.createTexture({size: [gridSize, gridSize], format: STATE_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => dispTex.destroy())
        const displacement = registerComputeTexture(dispTex)

        const updateParams = root.createUniform(d.struct({
            cursorX: d.f32, cursorY: d.f32, mouseVelX: d.f32, mouseVelY: d.f32,
            dt: d.f32, decay: d.f32, intensity: d.f32, radius: d.f32, aspect: d.f32,
        }))

        const viewport = trackedViewport(params)

        const size: [number, number] = [gridSize, gridSize]
        const updateKernel = makeSplatUpdateKernel(gridSize)
        const outputKernel = makeSplatOutputKernel(gridSize)
        const updateBg = root.createBindGroup(splatUpdateLayout, {buffer, params: updateParams.buffer})
        const outputBg = root.createBindGroup(splatOutputLayout, {buffer, dispTex})
        const update = createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; updateKernel(cx, cy)}, {size, bindGroup: updateBg})
        const output = createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; outputKernel(cx, cy)}, {size, bindGroup: outputBg})

        let prevX = 0.5
        let prevY = 0.5
        let smoothVelX = 0
        let smoothVelY = 0
        let px = 0.5
        let py = 0.5
        let active = false

        return {
            outputs: {[slots.output]: displacement},
            clampDt: 0.016,
            stages: [
                op.host('cursorVelocity', (f) => {
                    px = f.frameParams.pointer?.x ?? 0.5
                    py = f.frameParams.pointer?.y ?? 0.5
                    const velX = f.dt > 0 ? (px - prevX) / f.dt : 0
                    const velY = f.dt > 0 ? (py - prevY) / f.dt : 0
                    smoothVelX = smoothVelX * 0.85 + velX * 0.15
                    smoothVelY = smoothVelY * 0.85 + velY * 0.15
                    prevX = px
                    prevY = py
                    active = Math.abs(velX) + Math.abs(velY) > 0.01
                }),
                // At-rest skip: settle time derived from decay so residual displacement fully decays.
                op.settle({
                    activeWhen: () => active,
                    settleMs: (f) => {
                        const decayVal = f.num(slots.decay.name, 3)
                        return decayVal > 0 ? Math.min(30000, (Math.log(1e-4) / Math.log(Math.max(1e-6, 1 - decayVal * 0.016))) * 16.67) : 30000
                    },
                }),
                op.values('brush+decay', (f) => {
                    const safe = viewport.safe(f.frameParams)
                    updateParams.write({
                        cursorX: px, cursorY: py, mouseVelX: smoothVelX, mouseVelY: smoothVelY,
                        dt: f.dt, decay: f.num(slots.decay.name, 3), intensity: f.num(slots.intensity.name, 1),
                        radius: f.num(slots.radius.name, 1) * 0.05, aspect: safe.width / safe.height,
                    })
                }),
                op.pass(update),
                op.publish(() => output),
            ],
        }
    }
}

// ═══ Spring-lattice field ═════════════════════════════════════════════════════════════════════

const LATTICE_GRID = 64
const LATTICE_CELLS = LATTICE_GRID * LATTICE_GRID
const LATTICE_MAX = LATTICE_GRID - 1

// ── Compute bind-group layouts. Per-cell vec4 = (dispX, dispY, velX, velY). ──────────────────
// Spring pass: reads the CURRENT field (readBuf) → writes the integrated field (writeBuf). A is
// always current, B is scratch (A→B→A each frame), so the two bind groups are fixed (no swap).
const latticeLayout = tgpu.bindGroupLayout({
    readBuf: {storage: d.arrayOf(d.vec4f, LATTICE_CELLS), access: 'readonly'},
    writeBuf: {storage: d.arrayOf(d.vec4f, LATTICE_CELLS), access: 'mutable'},
    params: {uniform: d.struct({
        cursorX: d.f32, cursorY: d.f32, dirX: d.f32, dirY: d.f32, clampedSpeed: d.f32,
        dt: d.f32, subDt: d.f32, stiffness: d.f32, dampFactor: d.f32, radius: d.f32, aspect: d.f32,
    })},
})
const latticeOutputLayout = tgpu.bindGroupLayout({
    srcBuf: {storage: d.arrayOf(d.vec4f, LATTICE_CELLS), access: 'readonly'},
    dispTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
})

/** @internal The spring-lattice step kernel behind `springLatticeField`. */
// 4 structural + 4 shear springs + return-to-rest, semi-implicit Euler, cursor impulse
// injection. 2D-dispatched with the interior border guard.
export const springLatticeKernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
    'use gpu'
    if (cx > d.u32(0) && cx < d.u32(LATTICE_MAX) && cy > d.u32(0) && cy < d.u32(LATTICE_MAX)) {
        const p = latticeLayout.$.params
        const gs = d.u32(LATTICE_GRID)
        const idx = cy * gs + cx
        const current = latticeLayout.$.readBuf[idx]
        const dispX = current.x
        const dispY = current.y
        const velX = current.z
        const velY = current.w

        const nL = latticeLayout.$.readBuf[idx - d.u32(1)]
        const nR = latticeLayout.$.readBuf[idx + d.u32(1)]
        const nU = latticeLayout.$.readBuf[idx - gs]
        const nD = latticeLayout.$.readBuf[idx + gs]
        const nUL = latticeLayout.$.readBuf[idx - gs - d.u32(1)]
        const nUR = latticeLayout.$.readBuf[idx - gs + d.u32(1)]
        const nDL = latticeLayout.$.readBuf[idx + gs - d.u32(1)]
        const nDR = latticeLayout.$.readBuf[idx + gs + d.u32(1)]

        // Structural springs + shear springs (35%) + return-to-rest.
        let fx = p.stiffness * (nL.x - dispX + (nR.x - dispX) + (nU.x - dispX) + (nD.x - dispX))
        let fy = p.stiffness * (nL.y - dispY + (nR.y - dispY) + (nU.y - dispY) + (nD.y - dispY))
        const shearK = p.stiffness * 0.35
        fx = fx + shearK * (nUL.x - dispX + (nUR.x - dispX) + (nDL.x - dispX) + (nDR.x - dispX))
        fy = fy + shearK * (nUL.y - dispY + (nUR.y - dispY) + (nDL.y - dispY) + (nDR.y - dispY))
        fx = fx - p.stiffness * 0.1 * dispX
        fy = fy - p.stiffness * 0.1 * dispY

        let newVelX = velX * p.dampFactor + fx * p.subDt
        let newVelY = velY * p.dampFactor + fy * p.subDt

        if (p.clampedSpeed > 0.01) {
            const cellX = (d.f32(cx) + 0.5) / d.f32(LATTICE_GRID)
            const cellY = (d.f32(cy) + 0.5) / d.f32(LATTICE_GRID)
            const ddx = std.select(cellX - p.cursorX, (cellX - p.cursorX) * p.aspect, p.aspect >= 1.0)
            const ddy = std.select((cellY - p.cursorY) / p.aspect, cellY - p.cursorY, p.aspect >= 1.0)
            const distSq = ddx * ddx + ddy * ddy
            const influenceRadius = p.radius * 3.0
            if (distSq < influenceRadius * influenceRadius) {
                const influence = std.exp((distSq / (p.radius * p.radius)) * -1.0)
                const forceMag = influence * p.clampedSpeed * p.dt * 2.0
                newVelX = newVelX + p.dirX * forceMag
                newVelY = newVelY + p.dirY * forceMag
            }
        }

        const newDispX = std.clamp(dispX + newVelX * p.subDt, -0.5, 0.5)
        const newDispY = std.clamp(dispY + newVelY * p.subDt, -0.5, 0.5)
        latticeLayout.$.writeBuf[idx] = d.vec4f(newDispX, newDispY, newVelX, newVelY)
    }
}).$name('springLatticeStep')

/** @internal The spring-lattice output kernel behind `springLatticeField`. */
// Copy the displacement (xy) of the current buffer into the RG displacement texture.
export const springLatticeOutputKernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
    'use gpu'
    const idx = cy * d.u32(LATTICE_GRID) + cx
    const cell = latticeOutputLayout.$.srcBuf[idx]
    std.textureStore(latticeOutputLayout.$.dispTex, d.vec2u(cx, cy), d.vec4f(cell.x, cell.y, 0.0, 1.0))
}).$name('springLatticeOutput')

/**
 * A cloth of springs the cursor pushes: it stretches, rings and settles like fabric.
 *
 * Spread it with `...sim.grids.gridSim(...)`. `stiffness` (about 1 to 30) is how rigid the cloth
 * is; `damping` (about 0 to 10) how quickly it stops moving; `radius` the push area (1 is about
 * 8% of the canvas). The grid is fixed at 64 cells per side. The field is published under
 * `output` for a warp map such as `warps.liquidDisplace`.
 *
 * @example
 * ```ts
 * ...sim.grids.gridSim(springLatticeField({stiffness: p('stiffness'), damping: p('damping'), radius: p('radius'), output: 'displacement'}))
 * ```
 * @tip Add `uvRemapIdentityWhen: ({computeOutputs}) => !computeOutputs?.displacement` so the warp is a no-op before the field exists.
 * @see pointerSplatField
 */
// Maintainer: a spring-mass cloth over two ping-pong vec4 state buffers (createStateBuffer;
// WebGPU zero-inits → the cloth starts at rest), as the std grid pipeline: cursor-impulse tracking
// → settle gate (5000/damping ms) → params write → 2 fixed substeps (A→B, B→A — A is always
// current, B is scratch, so the bind groups are fixed) → output publish from A under
// `slots.output`. CHILD-INDEPENDENT → no bindInputs. NB: no childNode guard — in the analytic
// uvRemap fold path the composer passes childNode=undefined (the child is folded, not sampled);
// the sim must still run there so `uvRemap` can read its displacement.
export function springLatticeField(slots: {
    stiffness: PropRef
    damping: PropRef
    radius: PropRef
    output: string
}): (params: GpuFragmentParams, root: GridRoot) => GridSimConfig {
    return (params, root) => {
        const {registerComputeTexture, onCleanup} = params

        const bufferA = createStateBuffer(root, d.vec4f, LATTICE_CELLS)
        const bufferB = createStateBuffer(root, d.vec4f, LATTICE_CELLS)

        const dispTex = root.createTexture({size: [LATTICE_GRID, LATTICE_GRID], format: STATE_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => dispTex.destroy())
        const displacement = registerComputeTexture(dispTex)

        const springParams = root.createUniform(d.struct({
            cursorX: d.f32, cursorY: d.f32, dirX: d.f32, dirY: d.f32, clampedSpeed: d.f32,
            dt: d.f32, subDt: d.f32, stiffness: d.f32, dampFactor: d.f32, radius: d.f32, aspect: d.f32,
        }))

        const viewport = trackedViewport(params)

        const size: [number, number] = [LATTICE_GRID, LATTICE_GRID]
        const springBgAB = root.createBindGroup(latticeLayout, {readBuf: bufferA, writeBuf: bufferB, params: springParams.buffer})
        const springBgBA = root.createBindGroup(latticeLayout, {readBuf: bufferB, writeBuf: bufferA, params: springParams.buffer})
        const spring = createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; springLatticeKernel(cx, cy)}, {size})
        const outputBgA = root.createBindGroup(latticeOutputLayout, {srcBuf: bufferA, dispTex})
        const output = createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; springLatticeOutputKernel(cx, cy)}, {size, bindGroup: outputBgA})

        let prevX = 0.5
        let prevY = 0.5
        let px = 0.5
        let py = 0.5
        let rawVelX = 0
        let rawVelY = 0
        let speed = 0
        let clampedSpeed = 0

        return {
            outputs: {[slots.output]: displacement},
            clampDt: 0.016,
            zeroDt: 'skip',
            stages: [
                op.host('cursorImpulse', (f) => {
                    px = f.frameParams.pointer?.x ?? 0.5
                    py = f.frameParams.pointer?.y ?? 0.5
                    rawVelX = (px - prevX) / f.dt
                    rawVelY = (py - prevY) / f.dt
                    speed = Math.sqrt(rawVelX * rawVelX + rawVelY * rawVelY)
                    clampedSpeed = Math.min(speed, 3.0)
                    prevX = px
                    prevY = py
                }),
                // At-rest skip: settle time derived from damping (stiff/low-damped keep running).
                op.settle({
                    activeWhen: () => clampedSpeed > 0.01,
                    settleMs: (f) => {
                        const damping = f.num(slots.damping.name, 3)
                        return damping > 0 ? Math.min(30000, 5000 / damping) : 30000
                    },
                }),
                op.values('springs+impulse', (f) => {
                    const safe = viewport.safe(f.frameParams)
                    const damping = f.num(slots.damping.name, 3)
                    const subDt = f.dt / 2
                    springParams.write({
                        cursorX: px, cursorY: py,
                        dirX: speed > 0.01 ? rawVelX / speed : 0,
                        dirY: speed > 0.01 ? rawVelY / speed : 0,
                        clampedSpeed, dt: f.dt, subDt,
                        stiffness: f.num(slots.stiffness.name, 3),
                        dampFactor: Math.max(0, Math.min(1, 1 - damping * subDt)),
                        radius: f.num(slots.radius.name, 1) * 0.08,
                        aspect: safe.width / safe.height,
                    })
                }),
                // 2 substeps (A→B, B→A) then output from A.
                op.pass(spring.with(springBgAB)),
                op.pass(spring.with(springBgBA)),
                op.publish(() => output),
            ],
        }
    }
}
