/**
 * The per-frame values a component's WGSL reads by name: `time`, `dt`, `frame`, `pointer`,
 * `pointerVelocity`, `pointerDown`, `idle`, plus the press log `events` reads. One tracker per
 * component instance, updated once per frame and shared by its inputs and its body.
 */
// Maintainer notes. The renderer hands `onBeforeRender` callbacks its reusable frame params
// (deltaTime, pointer, pointerDown, clicks, elapsed). The tracker is keyed on the component's
// fragment params object, which the composer passes to both the compute hook and the fragment
// builder, so both halves see one tracker. `update` is idempotent per frame (keyed on the
// renderer's elapsed clock), so a second caller in the same frame is a no-op.
import type {GpuFragmentParams} from '../../gpu/contract'
import type {RenderCallbackParams} from '../../types'

/** Longest frame step a simulation sees. Keeps a tab switch from blowing a field apart. */
export const MAX_DT = 1 / 20
/** A pointer jump longer than this (uv) in one frame is a teleport, not motion. */
const TELEPORT_UV = 0.25
/** How quickly the velocity estimate follows the pointer (per second). */
const VELOCITY_RESPONSE = 18

export interface FramePress {
    readonly x: number
    readonly y: number
    /** Seconds since the press. */
    readonly age: number
    /** A stable random value in [0, 1) for this press. */
    readonly seed: number
    /** A counter that increases by one for every press, so a hook can tell a new one. */
    readonly id: number
}

export interface FrameValues {
    time: number
    dt: number
    frame: number
    aspect: number
    viewport: [number, number]
    pointer: [number, number]
    pointerVelocity: [number, number]
    pointerDown: number
    idle: number
}

export interface FrameTracker {
    readonly values: Readonly<FrameValues>
    /** Presses still within `maxAge` seconds, newest last. */
    presses(maxAge: number): readonly FramePress[]
    /** Advance to this frame. Safe to call more than once per frame. */
    update(frame: RenderCallbackParams): void
}

const trackers = new WeakMap<object, FrameTracker>()

/** The tracker for a component instance (created on first use, updated before every render). */
export function frameTracker(params: GpuFragmentParams, animated: boolean): FrameTracker {
    let tracker = trackers.get(params)
    if (tracker) return tracker
    tracker = createFrameTracker(params, animated)
    trackers.set(params, tracker)
    const t = tracker
    params.onBeforeRender((frame) => t.update(frame))
    return tracker
}

function hash01(n: number): number {
    // A small integer hash → [0, 1). Deterministic, so a press keeps its seed for its whole life.
    let h = (n * 0x9e3779b1) >>> 0
    h ^= h >>> 15
    h = Math.imul(h, 0x85ebca77) >>> 0
    h ^= h >>> 13
    return (h >>> 0) / 4294967296
}

export function createFrameTracker(params: GpuFragmentParams, animated: boolean): FrameTracker {
    const values: FrameValues = {
        time: 0, dt: 0, frame: 0, aspect: 1,
        viewport: [params.dimensions.width, params.dimensions.height],
        pointer: [0.5, 0.5], pointerVelocity: [0, 0], pointerDown: 0, idle: 0,
    }
    let lastElapsed = Number.NaN
    let prevLayerTime = 0
    let havePrevLayerTime = false
    const target = {width: Math.max(1, params.dimensions.width), height: Math.max(1, params.dimensions.height)}
    params.onResize((size) => {
        target.width = Math.max(1, size.width)
        target.height = Math.max(1, size.height)
    })
    let ownClock = 0
    let havePointer = false
    let pressIds = new WeakMap<object, number>()
    let pressCount = 0
    let clicks: readonly {x: number; y: number; time: number}[] = []
    let elapsed = 0

    return {
        values,
        presses(maxAge) {
            const out: FramePress[] = []
            for (const c of clicks) {
                const age = elapsed - c.time
                if (age < 0 || age > maxAge) continue
                let id = pressIds.get(c)
                if (id === undefined) {
                    id = ++pressCount
                    pressIds.set(c, id)
                }
                out.push({x: c.x, y: c.y, age, seed: hash01(id), id})
            }
            return out
        },
        update(frame) {
            const now = frame.elapsed ?? Number.NaN
            if (Number.isFinite(now) && now === lastElapsed) return
            lastElapsed = now
            const dt = Math.min(Math.max(frame.deltaTime ?? 0, 0), MAX_DT)
            values.dt = dt
            values.frame += 1
            ownClock += dt
            if (animated) {
                const t = params.getCpuValue('_animTime')
                const layerTime = typeof t === 'number' ? t : ownClock
                // `dt` is the layer clock's step, so the Speed control scales (and at 0 pauses) a
                // simulation exactly as it does `time`. (Pointer velocity below stays in real
                // seconds: the pointer moves in real time whatever the layer's speed.)
                if (havePrevLayerTime) values.dt = Math.min(Math.max(layerTime - prevLayerTime, 0), MAX_DT)
                prevLayerTime = layerTime
                havePrevLayerTime = true
                values.time = layerTime
            } else {
                values.time = Number.isFinite(now) ? now : ownClock
            }
            // `viewport` is the render target's size (`target`, kept by onResize exactly as the
            // inputs' textures are), so a compute body's pixel maths matches the texture it
            // writes. `frame.dimensions` is the CSS size before DPR: wrong on a 2× screen and in
            // captures, which render at their own size.
            values.viewport = [target.width, target.height]
            values.aspect = target.width / target.height
            const px = frame.pointer?.x ?? 0.5
            const py = frame.pointer?.y ?? 0.5
            const onCanvas = px >= -0.5 && px <= 1.5 && py >= -0.5 && py <= 1.5
            if (onCanvas) {
                const dx = px - values.pointer[0]
                const dy = py - values.pointer[1]
                const jump = Math.hypot(dx, dy)
                if (!havePointer || jump > TELEPORT_UV || dt <= 0) {
                    values.pointerVelocity = [0, 0]
                } else {
                    const k = 1 - Math.exp(-VELOCITY_RESPONSE * dt)
                    values.pointerVelocity = [
                        values.pointerVelocity[0] + (dx / dt - values.pointerVelocity[0]) * k,
                        values.pointerVelocity[1] + (dy / dt - values.pointerVelocity[1]) * k,
                    ]
                }
                values.idle = jump > 1e-5 ? 0 : values.idle + dt
                values.pointer = [px, py]
                havePointer = true
            } else {
                // Off the canvas: the velocity decays to rest and the pointer keeps its last place.
                values.pointerVelocity = [values.pointerVelocity[0] * 0.8, values.pointerVelocity[1] * 0.8]
                values.idle += dt
            }
            values.pointerDown = frame.pointerDown ? 1 : 0
            if (frame.clicks !== clicks) {
                clicks = frame.clicks ?? []
            }
            elapsed = Number.isFinite(now) ? now : ownClock
            if (pressCount > 1e6) {
                pressIds = new WeakMap()
                pressCount = 0
            }
        },
    }
}
