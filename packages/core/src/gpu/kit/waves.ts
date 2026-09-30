/**
 * Wave-field simulation — the GPU bodies + runtime harness behind std's
 * `simulate.grid({step: [op.wave, op.splat(pointer)], derive: {gradient}})`
 * (CursorRipples is the first consumer).
 *
 * Two-stage per the house solver-factory rule (kernels at module/factory scope, passes per instance):
 *  - {@link buildWaveFieldKernels} — memoized per resolution: the bind-group layouts + the two
 *    kernels ($name'd with the resolution key so two resolutions in one tree cannot collide).
 *  - {@link createWaveFieldSim} — the per-instance runtime object: ping-pong height-field state
 *    buffers, the displacement texture, pointer tracking (teleport-guarded), the dt clamp, and
 *    the at-rest skip whose settle window is DERIVED from the damping parameter (std
 *    `rest: {settlesWhen: 'derived-from-damping'}`).
 *
 * The wave step is the damped `avg − prev` wave equation and needs TWO history levels: the
 * propagate kernel reads the buffer it is about to overwrite as the t−2 state — this is what
 * std's `history: 2` declares.
 */
import {tgpu, d, std} from './index'
import type {GpuFragmentParams, KitTexture} from '../contract'
import {createGuardedCompute, createPingPongPair, createStateBuffer, type ComputeStep} from '../compute'
import {createPointerVelocityTracker} from './host/pointer'

// The Gaussian brush is truncated at 3σ, where a raw exp(−d²/σ²) still carries exp(−9) of its
// amplitude — a small step at the brush edge. Shifting and rescaling so the falloff reaches exactly
// zero at the cutoff removes it (the PixelThrow brush pattern, with the shift set by the wave
// brush's cutoff rather than PixelThrow's 1σ one).
const BRUSH_CUTOFF_SIGMAS = 3.0
const BRUSH_EDGE = Math.exp(-(BRUSH_CUTOFF_SIGMAS * BRUSH_CUTOFF_SIGMAS))
const BRUSH_EDGE_GAIN = 1 / (1 - BRUSH_EDGE)

/** Damping-prop → per-step damp factor curve (the wave op's damping semantic, 0–20 UI range). */
const DAMPING_SCALE = 0.004

const STATE_FORMAT = 'rgba16float' as const

export interface WaveFieldKernels {
    resolution: number
    propagateLayout: ReturnType<typeof buildPropagateLayout>
    gradientLayout: ReturnType<typeof buildGradientLayout>
    /** Callable inside a `'use gpu'` wrapper; resolvable via `tgpu.resolve` in tests. */
    propagateKernel: (cx: number, cy: number) => void
    gradientKernel: (cx: number, cy: number) => void
}

function buildPropagateLayout(cellCount: number) {
    return tgpu.bindGroupLayout({
        readBuf: {storage: d.arrayOf(d.f32, cellCount), access: 'readonly'},
        writeBuf: {storage: d.arrayOf(d.f32, cellCount), access: 'mutable'},
        params: {uniform: d.struct({
            cursorX: d.f32, cursorY: d.f32, cursorSpeed: d.f32, dt: d.f32, damping: d.f32, radius: d.f32, aspect: d.f32,
        })},
    })
}

function buildGradientLayout(cellCount: number) {
    return tgpu.bindGroupLayout({
        srcBuf: {storage: d.arrayOf(d.f32, cellCount), access: 'readonly'},
        dispTex: {storageTexture: d.textureStorage2d(STATE_FORMAT, 'write-only')},
    })
}

const kernelCache = new Map<number, WaveFieldKernels>()

/** Layouts + kernels for one grid resolution. Memoized; $name carries the resolution key. */
export function buildWaveFieldKernels(resolution: number): WaveFieldKernels {
    const cached = kernelCache.get(resolution)
    if (cached) return cached

    const cellCount = resolution * resolution
    const gridMax = resolution - 1
    const propagateLayout = buildPropagateLayout(cellCount)
    const gradientLayout = buildGradientLayout(cellCount)

    /**
     * Wave-equation propagation + cursor injection kernel (damped `avg − prev`; 2-history).
     * 2D-dispatched (cx=col, cy=row); the interior guard drops the 1-cell border.
     */
    const propagateKernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        if (cx > d.u32(0) && cx < d.u32(gridMax) && cy > d.u32(0) && cy < d.u32(gridMax)) {
            const p = propagateLayout.$.params
            const gs = d.u32(resolution)
            const idx = cy * gs + cx
            const left = propagateLayout.$.readBuf[idx - d.u32(1)]
            const right = propagateLayout.$.readBuf[idx + d.u32(1)]
            const up = propagateLayout.$.readBuf[idx - gs]
            const down = propagateLayout.$.readBuf[idx + gs]
            const avg = (left + right + up + down) * 0.5
            const prev = propagateLayout.$.writeBuf[idx]
            const dampFactor = 1.0 - p.damping * DAMPING_SCALE
            let next = (avg - prev) * dampFactor

            // Cursor injection: Gaussian force while the pointer is moving (one-frame delay vs a CPU sim).
            const cellX = (d.f32(cx) + 0.5) / d.f32(resolution)
            const cellY = (d.f32(cy) + 0.5) / d.f32(resolution)
            const ddx = std.select(cellX - p.cursorX, (cellX - p.cursorX) * p.aspect, p.aspect >= 1.0)
            const ddy = std.select((cellY - p.cursorY) / p.aspect, cellY - p.cursorY, p.aspect >= 1.0)
            const distSq = ddx * ddx + ddy * ddy
            const influenceRadius = p.radius * 3.0 // BRUSH_CUTOFF_SIGMAS (kept a literal so it folds as f32)
            if (distSq < influenceRadius * influenceRadius && p.cursorSpeed > 0.01) {
                const influence = (std.exp((distSq / (p.radius * p.radius)) * -1.0) - BRUSH_EDGE) * BRUSH_EDGE_GAIN
                next = next - influence * p.cursorSpeed * p.dt * 3.0
            }
            propagateLayout.$.writeBuf[idx] = next
        }
    }).$name(`waveFieldPropagate_${resolution}`)

    /** Central-difference gradient of the height field → RG displacement texel. */
    const gradientKernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        if (cx > d.u32(0) && cx < d.u32(gridMax) && cy > d.u32(0) && cy < d.u32(gridMax)) {
            const gs = d.u32(resolution)
            const idx = cy * gs + cx
            const right = gradientLayout.$.srcBuf[idx + d.u32(1)]
            const left = gradientLayout.$.srcBuf[idx - d.u32(1)]
            const down = gradientLayout.$.srcBuf[idx + gs]
            const up = gradientLayout.$.srcBuf[idx - gs]
            const dx = (right - left) * 0.5
            const dy = (down - up) * 0.5
            std.textureStore(gradientLayout.$.dispTex, d.vec2u(cx, cy), d.vec4f(dx, dy, 0.0, 1.0))
        }
    }).$name(`waveFieldGradient_${resolution}`)

    const built: WaveFieldKernels = {resolution, propagateLayout, gradientLayout, propagateKernel, gradientKernel}
    kernelCache.set(resolution, built)
    return built
}

export interface WaveFieldConfig {
    resolution: number
    /** Prop name read per frame as the damping value (std `op.wave({damping})`). */
    dampingProp: string
    /** Prop name read per frame as the brush radius (std `op.splat({radius})`). */
    radiusProp: string
    /** UI radius → field-space radius scale. */
    radiusScale: number
    /** Pointer-speed clamp (std `pointerSpeed({max})`). */
    speedMax: number
    /**
     * `'on'` (default): a sudden pointer jump — entering the canvas, returning from another tab —
     * injects nothing; motion is measured from the new spot next frame. `'off'`: every move counts,
     * including the jump (std `pointer({teleportGuard})`).
     */
    teleportGuard?: 'on' | 'off'
}

export interface WaveFieldSim {
    /** The RG displacement-gradient texture (a registered compute texture). */
    displacement: KitTexture
    getComputeNodes: (frameParams: unknown) => ComputeStep[] | null
}

/**
 * The per-instance wave-field runtime: two ping-pong height-field state buffers
 * (createStateBuffer; WebGPU zero-inits → the field starts flat), two dispatches/frame —
 * propagation (+ cursor injection) then the gradient pass writing the RG displacement texture.
 * Child-independent (a pure wave field), so there is no bindInputs. Returns null when the
 * composition has no GPU root (GPU-free resolve/tests fall back to zero displacement).
 */
export function createWaveFieldSim(params: GpuFragmentParams, config: WaveFieldConfig): WaveFieldSim | null {
    const {gpu, registerComputeTexture, getCpuValue, onCleanup, onResize, dimensions} = params
    const root = gpu?.root
    if (!root) return null

    const {resolution, dampingProp, radiusProp, radiusScale, speedMax, teleportGuard = 'on'} = config
    const {propagateLayout, gradientLayout, propagateKernel, gradientKernel} = buildWaveFieldKernels(resolution)
    const cellCount = resolution * resolution

    const bufferA = createStateBuffer(root, d.f32, cellCount)
    const bufferB = createStateBuffer(root, d.f32, cellCount)

    const dispTex = root.createTexture({size: [resolution, resolution], format: STATE_FORMAT}).$usage('storage', 'sampled')
    onCleanup(() => dispTex.destroy())
    const displacement = registerComputeTexture(dispTex)

    const propParams = root.createUniform(d.struct({
        cursorX: d.f32, cursorY: d.f32, cursorSpeed: d.f32, dt: d.f32, damping: d.f32, radius: d.f32, aspect: d.f32,
    }))

    let curW = Math.max(1, Math.round(dimensions.width))
    let curH = Math.max(1, Math.round(dimensions.height))
    onResize(({width, height}) => {
        curW = Math.max(1, Math.round(width))
        curH = Math.max(1, Math.round(height))
    })

    const size: [number, number] = [resolution, resolution]
    // Ping-pong the height field. Propagation reads one buffer and writes the other; the
    // gradient pass then reads whichever buffer was just WRITTEN — the two roles the pair
    // helper exists to keep straight.
    const field = createPingPongPair(bufferA, bufferB)
    const propGroup = field.groups((readBuf, writeBuf) =>
        root.createBindGroup(propagateLayout, {readBuf, writeBuf, params: propParams.buffer}))
    const gradGroup = field.perSide((srcBuf) => root.createBindGroup(gradientLayout, {srcBuf, dispTex}))
    const propagate = createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; propagateKernel(cx, cy)}, {size})
    const gradient = createGuardedCompute(root, (cx: number, cy: number) => {'use gpu'; gradientKernel(cx, cy)}, {size})

    // Shared pointer tracking: per-second velocity with the teleport guard applied, so a jump the
    // user never made (entering the canvas, a tab switch) injects nothing. `'off'` lifts the
    // jump threshold to infinity, so the tracker never classifies a move as a teleport.
    const pointer = createPointerVelocityTracker(teleportGuard === 'off' ? {teleportGuard: Number.POSITIVE_INFINITY} : {})
    let lastActiveTime = Date.now()

    return {
        displacement,
        getComputeNodes: (frameParams: unknown): ComputeStep[] | null => {
            const fp = frameParams as {pointer: {x: number; y: number}; deltaTime: number; dimensions?: {width: number; height: number}}
            const currentTime = Date.now()
            const dt = Math.min(fp.deltaTime ?? 0, 0.016)
            const safeW = Math.max(1, fp.dimensions?.width ?? curW)
            const safeH = Math.max(1, fp.dimensions?.height ?? curH)
            const aspect = safeW / safeH
            const damping = (getCpuValue(dampingProp) as number) ?? 10
            const radius = ((getCpuValue(radiusProp) as number) ?? 0.5) * radiusScale

            // velX/velY are zeroed on a teleport frame, so no impulse comes out of a jump.
            const move = pointer.update(fp.pointer, dt)
            const cursorSpeed = Math.min(Math.sqrt(move.velX * move.velX + move.velY * move.velY), speedMax)
            if (cursorSpeed > 0.01) lastActiveTime = currentTime

            // At-rest skip: once the pointer has been idle long enough for the field to decay away,
            // stop dispatching (settle time derived from the damping parameter — low damping keeps running).
            const dampFactor = 1 - damping * DAMPING_SCALE
            const settleTime = dampFactor >= 1 ? Infinity : Math.min(30000, (Math.log(1e-6) / Math.log(Math.max(dampFactor, 0.001))) * 16.67)
            if (currentTime - lastActiveTime > settleTime) return null

            propParams.write({cursorX: move.x, cursorY: move.y, cursorSpeed, dt, damping, radius, aspect})

            const nodes: ComputeStep[] = [propagate.with(propGroup()), gradient.with(gradGroup.written())]
            field.swap()
            return nodes
        },
    }
}
