/**
 * std/signal — Live inputs a simulation reads each frame: where the pointer is and how fast
 * it moves.
 *
 * You do not read these values yourself. Hand them to a simulation step such as `op.splat`,
 * which stirs its grid at the pointer with a strength that follows the pointer's speed. The
 * pointer position is in uv (0–1 across the canvas, y down). Speed is in uv per second.
 */
// Maintainer notes (not part of the reader-facing reference):
// - Signal nouns, first slice: the pointer signals a cursor-driven simulation needs. Pointer
//   signals carry POLICY as data — the teleport guard is a per-shader aesthetic decision
//   (ON for ripple/RD-class sims, OFF for smoke/trail-class), and the speed clamp names the
//   gating curve. `op.splat` passes `teleportGuard` through to the wave-field tracker (see lower.ts).

/**
 * Where the pointer is this frame, for a simulation step.
 *
 * `teleportGuard: 'on'` means a sudden jump (the cursor entering the canvas somewhere new)
 * adds no motion that frame.
 */
export interface PointerSignal {
    readonly kind: 'pointer'
    readonly teleportGuard: 'on' | 'off'
}

/**
 * The pointer's position, for a simulation step that reacts to the cursor.
 *
 * The teleport guard is on by default: when the cursor jumps a long way in one frame, the
 * jump adds no motion, so a cursor entering the canvas does not slam the simulation.
 *
 * @example
 * ```ts
 * op.splat({at: pointer({teleportGuard: 'on'}), amount: pointerSpeed({max: 2}), radius: p('radius')})
 * ```
 * @tip Keep the default `'on'` for cursor effects. Use `'off'` only when a jump should count as a stroke — a pointer that is set by code rather than a real cursor, for example.
 * @see pointerSpeed
 */
export function pointer(opts?: {teleportGuard?: 'on' | 'off'}): PointerSignal {
    return {kind: 'pointer', teleportGuard: opts?.teleportGuard ?? 'on'}
}

/**
 * How fast the pointer is moving, for a simulation step.
 *
 * Measured in uv per second and capped at `max`, so a fast flick cannot overdrive the
 * simulation.
 */
export interface PointerSpeedSignal {
    readonly kind: 'pointerSpeed'
    readonly max: number
}

/**
 * The pointer's speed, capped at `max`, as the strength of a simulation step.
 *
 * Speed is in uv per second. A still cursor gives 0, a flick across the canvas gives a
 * value near `max`. Default cap is 2.
 *
 * @example
 * ```ts
 * op.splat({at: pointer(), amount: pointerSpeed({max: 2}), radius: p('radius')})
 * ```
 * @see pointer
 */
export function pointerSpeed(opts?: {max?: number}): PointerSpeedSignal {
    return {kind: 'pointerSpeed', max: opts?.max ?? 2}
}

/** @internal */
export function driveUnitDirection(
    params: {
        onBeforeRender(cb: () => void): void
        getCpuValue(name: string): unknown
        setExtraField(name: string, value: number): void
    },
    props: {x: string; y: string; z: string},
    fields: {x: string; y: string; z: string},
): void {
    // Drive three extraFields with the CPU-normalized unit vector of three scalar props,
    // refreshed each frame — the standard way a direction prop reaches the GPU pre-normalized
    // (saves a per-fragment `normalize()`). Declare the fields on the definition's
    // `extraFields`; a zero-length input writes the zero vector. Blob is the consumer.
    params.onBeforeRender(() => {
        const x = params.getCpuValue(props.x) as number
        const y = params.getCpuValue(props.y) as number
        const z = params.getCpuValue(props.z) as number
        const len = Math.sqrt(x * x + y * y + z * z)
        if (len > 0) {
            params.setExtraField(fields.x, x / len)
            params.setExtraField(fields.y, y / len)
            params.setExtraField(fields.z, z / len)
        } else {
            params.setExtraField(fields.x, 0)
            params.setExtraField(fields.y, 0)
            params.setExtraField(fields.z, 0)
        }
    })
}

/**
 * A random but stable number, 0..1, from a seed, computed in JavaScript.
 *
 * Use it when a per-seed constant is worked out once on the CPU rather than at every pixel,
 * such as the size and drift of each light in a set.
 *
 * @example
 * ```ts
 * const radiusX = cpuHash01(seed) * 0.5 + 0.2
 * const radiusY = cpuHash01(seed + 37) * 0.25 + 0.1
 * ```
 * @see pointer
 */
export function cpuHash01(x: number): number {
    // CPU mirror of the classic in-shader `hash01` (`fract(sin(x·12.9898)·43758.5453)`) — for
    // pixel-invariant per-seed constants computed off the GPU (JS Math.sin at these argument
    // magnitudes is exact where GPU sin is driver-approximate).
    const v = Math.sin(x * 12.9898) * 43758.5453
    return v - Math.floor(v)
}
