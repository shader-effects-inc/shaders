/**
 * std/sim/agentFrame — the once-per-frame arithmetic an agent simulation writes into its values.
 *
 * Inside `agentSim`'s `frame` callback you turn props and the pointer into the numbers the GPU
 * steps read: a drag factor that does not depend on frame rate, the pointer in the shape's own
 * coordinates, a grid that fits the agent count, camera rows for a 3D view. These are those
 * recipes as plain functions. Nothing here touches the GPU.
 */
// Maintainer notes. Placement inversion (pointer → shape-local), rotation-delta → entrainment
// omega conjugation, grid fitting, camera rows, low-discrepancy dithers, and exponential
// decays. GPU parts live in `agentForces`/`agentRender`, dispatch in the `createAgentSystem`
// harness.
import {R3_ALPHA} from '../../gpu/scaffolds/agentSystem'
import {rotateVecCpu} from '../../gpu/kit/sdf3d'

// ── Exponential rates ───────────────────────────────────────────────────────────────────────

/**
 * The per-frame multiplier that decays a value at `rate` per second, whatever the frame rate.
 *
 * Write it into a `dragMul` value each frame. A rate of 1 halves a velocity in about 0.7 s.
 *
 * @example
 * ```ts
 * dragMul: agentFrame.expDecay(2.2 + damping * 6, dt)
 * ```
 * @see integrator
 */
// `exp(−rate·dt)`.
export function expDecay(rate: number, dt: number): number {
    return Math.exp(-rate * dt)
}

// ── Placement inversion ─────────────────────────────────────────────────────────────────────

/**
 * The pointer in the shape's own coordinates: undo the layer's center, scale and rotation so a
 * cursor force can act in the same space the swarm lives in (y up, isotropic).
 *
 * `pointerX`/`pointerY` and `centerX`/`centerYv` are in uv, `rotC`/`rotS` the cosine and sine
 * of the rotation, `aspect` the canvas width over height.
 *
 * @example
 * ```ts
 * const cursor = agentFrame.pointerToShapeLocal({pointerX, pointerY, centerX, centerYv, scale, rotC, rotS, aspect})
 * ```
 * @see force
 */
// The forward transform is `screen = center + R(rot)·(local·scale)/aspectLift`; this is its
// exact inverse, with x lifted by aspect so the local frame is isotropic.
export function pointerToShapeLocal(opts: {
    pointerX: number
    pointerY: number
    centerX: number
    centerYv: number
    scale: number
    rotC: number
    rotS: number
    aspect: number
}): {x: number; y: number} {
    const dxAc = (opts.pointerX - opts.centerX) * opts.aspect
    const dyv = opts.pointerY - opts.centerYv
    return {
        x: (dxAc * opts.rotC + dyv * opts.rotS) / opts.scale,
        y: -((dyv * opts.rotC - dxAc * opts.rotS) / opts.scale),
    }
}

// ── Entrainment omega (rotation-delta conjugation) ─────────────────────────────────────

/**
 * How fast a rotating shape's surface moves, from its rotation this frame and last frame, so a
 * swarm inside it can be dragged along. Angles in radians, `dt` in seconds. Null on the first
 * frame.
 *
 * @example
 * ```ts
 * om = agentFrame.omegaFromRotationDeltas(prevRot, next, dt) ?? om
 * ```
 * @see entrainmentFromOmega
 */
// Body-frame angular velocity from consecutive euler angles, for a field sampled at `R·p`:
// features move at `−(Rᵀω)×r`, so the euler rates are conjugated by Rᵀ and negated.
export function omegaFromRotationDeltas(
    prev: {x: number; y: number; z: number} | null,
    next: {x: number; y: number; z: number},
    dt: number,
): {x: number; y: number; z: number} | null {
    if (!prev) return null
    const wx = (next.x - prev.x) / dt
    const wy = (next.y - prev.y) / dt
    const wz = (next.z - prev.z) / dt
    const s = {
        cx: Math.cos(next.x), sx: Math.sin(next.x),
        cy: Math.cos(next.y), sy: Math.sin(next.y),
        cz: Math.cos(next.z), sz: Math.sin(next.z),
    }
    const ex = rotateVecCpu(1, 0, 0, s)
    const ey = rotateVecCpu(0, 1, 0, s)
    const ez = rotateVecCpu(0, 0, 1, s)
    return {
        x: -(ex.x * wx + ex.y * wy + ex.z * wz),
        y: -(ey.x * wx + ey.y * wy + ey.z * wz),
        z: -(ez.x * wx + ez.y * wy + ez.z * wz),
    }
}

/**
 * The values `force.containment` reads for a rotating shape: the rotation rate capped at
 * `cap`, and an `entrain` gain that grows with the rate up to `gainMax`.
 *
 * @example
 * ```ts
 * const {omegaX, omegaY, omegaZ, entrain} = agentFrame.entrainmentFromOmega(om, {cap: 10, gainRate: 2.5, gainMax: 5})
 * ```
 * @tip With a still shape the gain is exactly 0, so the entrainment force costs nothing to leave declared.
 * @see omegaFromRotationDeltas, force
 */
export function entrainmentFromOmega(om: {x: number; y: number; z: number}, opts: {cap: number; gainRate: number; gainMax: number}) {
    const omLen = Math.hypot(om.x, om.y, om.z)
    const omCap = omLen > opts.cap ? opts.cap / omLen : 1
    return {
        omegaX: om.x * omCap,
        omegaY: om.y * omCap,
        omegaZ: om.z * omCap,
        entrain: Math.min(1, omLen * opts.gainRate) * opts.gainMax,
    }
}

// ── Pointer-motion axis (the magnet's dipole heading) ───────────────────────────────────

/**
 * A unit direction that follows where the pointer has been travelling, smoothed frame to frame.
 * Feed `update(x, y)` the pointer each frame and write the result into an axis value.
 *
 * `smoothing` (0–1, default 0.1) is how fast the axis turns. A jump longer than `teleport`
 * (default 0.25, in the same units as the pointer) is the cursor re-entering the canvas, not a
 * stroke, and leaves the axis alone.
 *
 * @example
 * ```ts
 * const motionAxis = agentFrame.createMotionAxis({smoothing: 0.1, teleport: 0.25})
 * const axis = motionAxis.update(cursorX, cursorY)
 * ```
 * @see field
 */
export function createMotionAxis(opts?: {smoothing?: number; teleport?: number}) {
    const smoothing = opts?.smoothing ?? 0.1
    const teleport = opts?.teleport ?? 0.25
    let lastX = 0.5
    let lastY = 0.5
    let axisX = 1
    let axisY = 0
    return {
        update(x: number, y: number): {x: number; y: number} {
            const mvx = x - lastX
            const mvy = y - lastY
            const mvLen = Math.hypot(mvx, mvy)
            if (mvLen > 1e-3 && mvLen < teleport) {
                const nx = mvx / mvLen
                const ny = mvy / mvLen
                axisX += (nx - axisX) * smoothing
                axisY += (ny - axisY) * smoothing
                const al = Math.hypot(axisX, axisY) || 1
                axisX /= al
                axisY /= al
            }
            lastX = x
            lastY = y
            return {x: axisX, y: axisY}
        },
    }
}

// ── Grid fitting ────────────────────────────────────────────────────────────────────────

/**
 * A grid of about `count` roughly square cells over the canvas plus an `overscan` border on
 * every side, for agents that rest on a jittered grid. `domainX` is the canvas width in
 * heights (the aspect).
 *
 * Returns the column count and the cell size, which `field.jitteredGridHome` reads.
 *
 * @example
 * ```ts
 * const {cols, cellW, cellH} = agentFrame.fitJitteredGrid(count, domainX, 0.1)
 * ```
 * @see fitIsoGrid, field
 */
// Domain is `[−ov, domainX+ov] × [−ov, 1+ov]` — off-screen agents backfill pulled-in edges.
export function fitJitteredGrid(count: number, domainX: number, overscan: number) {
    const spanX = domainX + 2 * overscan
    const spanY = 1 + 2 * overscan
    const cols = Math.max(1, Math.round(Math.sqrt(count * (spanX / spanY))))
    const rows = Math.max(1, Math.ceil(count / cols))
    return {cols, cellW: spanX / cols, cellH: spanY / rows}
}

/**
 * A W×H grid of square cells holding about `count` agents at the canvas `aspect`, each side
 * clamped to `[min, max]`.
 *
 * @example
 * ```ts
 * const {w, h} = agentFrame.fitIsoGrid(count, aspect, 8, 160)
 * ```
 * @see fitJitteredGrid
 */
export function fitIsoGrid(count: number, aspect: number, min: number, max: number) {
    let w = Math.round(Math.sqrt(count * aspect))
    w = Math.min(Math.max(w, min), max)
    let h = Math.round(count / w)
    h = Math.min(Math.max(h, min), max)
    return {w, h}
}

// ── Camera rows ─────────────────────────────────────────────────────────────────────────

/**
 * The three rows of a camera rotation for a 3D view drawn on a y-down canvas. Angles in
 * radians. Write them into the values a projected splat reads.
 *
 * Positive angles orbit in the directions that feel natural in the editor.
 *
 * @example
 * ```ts
 * const rows = agentFrame.cameraRowsYDown(rotX * DEG_TO_RAD, rotY * DEG_TO_RAD, rotZ * DEG_TO_RAD)
 * ```
 * @see renderAgents
 */
// Rz·Ry·Rx, conjugated for screen space (y down): the y-up matrix with the off-diagonal y
// terms sign-flipped.
export function cameraRowsYDown(rx: number, ry: number, rz: number): {
    rowX: [number, number, number]
    rowY: [number, number, number]
    rowZ: [number, number, number]
} {
    const cx = Math.cos(rx), sx = Math.sin(rx)
    const cy = Math.cos(ry), sy = Math.sin(ry)
    const cz = Math.cos(rz), sz = Math.sin(rz)
    const m00 = cz * cy, m01 = cz * sy * sx - sz * cx, m02 = cz * sy * cx + sz * sx
    const m10 = sz * cy, m11 = sz * sy * sx + cz * cx, m12 = sz * sy * cx - cz * sx
    const m20 = -sy, m21 = cy * sx, m22 = cy * cx
    return {
        rowX: [m00, -m01, m02],
        rowY: [-m10, m11, -m12],
        rowZ: [m20, -m21, m22],
    }
}

// ── Low-discrepancy dither ──────────────────────────────────────────────────────────────

/**
 * A tiny offset for frame `frameIdx`, within one grid `cell`, that `force.pressure` reads to
 * keep its density grid from lining up with the shape. Increment the frame index each frame.
 *
 * @example
 * ```ts
 * const gridOff = agentFrame.r3SubCellOffset(frameIdx++, cellSize)
 * ```
 * @see force
 */
// The R3 sub-cell dither for a voxel lattice: frame `i`'s offset, scaled by one cell — evenly
// covers the sub-cell cube over frames, so a grid can never stand in coherent moiré with a
// boundary. (See `R3_ALPHA` on the harness.)
export function r3SubCellOffset(frameIdx: number, cell: number): {x: number; y: number; z: number} {
    return {
        x: ((frameIdx * R3_ALPHA[0]) % 1) * cell,
        y: ((frameIdx * R3_ALPHA[1]) % 1) * cell,
        z: ((frameIdx * R3_ALPHA[2]) % 1) * cell,
    }
}
