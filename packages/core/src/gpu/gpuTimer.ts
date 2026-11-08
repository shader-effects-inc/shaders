/**
 * GPU pass timing: how long the GPU spent on one renderer's frame, from `timestamp-query`.
 *
 * Every pass the frame encodes (compute dispatches, render-to-texture passes, the final pass)
 * asks for a slot and writes a timestamp at its start and end. The frame's time is the sum of
 * those spans: the GPU's busy time on THIS renderer's work, not wall-clock time, so other
 * renderers sharing the device and the gaps between submissions don't count.
 *
 * Browsers quantize timestamps (Chromium to 100µs by default), so a single cheap pass can read
 * 0 or 0.1ms. Average over many frames.
 */
import type {TgpuQuerySet, TgpuRoot} from 'typegpu'

/** The `withTimestampWrites` options for one pass. */
export interface PassTimestampWrites {
    querySet: TgpuQuerySet<'timestamp'>
    beginningOfPassWriteIndex: number
    endOfPassWriteIndex: number
}

export interface GpuPassTimer {
    /** Timestamp writes for the next pass, or null once this frame's slots are used up. `label`
     *  names the pass in the profile (the component and pass it belongs to). */
    next(label?: string): PassTimestampWrites | null
}

/** One pass's GPU time in a frame. Passes sharing a label are summed. */
export interface PassTime {
    label: string
    ms: number
    /** How many passes share this label (a grid's steps, a blur's two halves). */
    count: number
}

export interface GpuFrameTimer extends GpuPassTimer {
    /** Start a frame: slots count from zero again. */
    begin(): void
    /** The frame's GPU time once its passes finish, or null if it timed no passes. */
    read(): Promise<{ms: number; passes: number; perPass: PassTime[]} | null>
    destroy(): void
}

/** Passes one frame can time; a frame with more leaves the rest untimed. */
const MAX_PASSES = 64

/** Whether this root's device can time passes. */
export function canTimeGpu(root: TgpuRoot): boolean {
    return root.enabledFeatures.has('timestamp-query')
}

export function createGpuFrameTimer(root: TgpuRoot): GpuFrameTimer {
    const querySet = root.createQuerySet('timestamp', MAX_PASSES * 2)
    let used = 0
    const labels: string[] = []
    return {
        begin() {
            used = 0
            labels.length = 0
        },
        next(label) {
            if (used >= MAX_PASSES) return null
            const slot = used++
            labels[slot] = label ?? 'unnamed pass'
            return {querySet, beginningOfPassWriteIndex: slot * 2, endOfPassWriteIndex: slot * 2 + 1}
        },
        async read() {
            const passes = used
            if (!passes) return null
            querySet.resolve()
            const stamps = await querySet.read()
            let ns = 0n
            const byLabel = new Map<string, bigint>()
            const counts = new Map<string, number>()
            for (let i = 0; i < passes; i++) {
                const start = stamps[i * 2]
                const end = stamps[i * 2 + 1]
                const span = start !== undefined && end !== undefined && end > start ? end - start : 0n
                ns += span
                const label = labels[i] ?? 'unnamed pass'
                byLabel.set(label, (byLabel.get(label) ?? 0n) + span)
                counts.set(label, (counts.get(label) ?? 0) + 1)
            }
            const perPass = [...byLabel].map(([label, t]) => ({label, ms: Number(t) / 1e6, count: counts.get(label) ?? 1}))
            return {ms: Number(ns) / 1e6, passes, perPass}
        },
        destroy() {
            querySet.destroy()
        },
    }
}

/**
 * The next timestamp slot as raw WebGPU `timestampWrites`, for a pass encoded straight on the
 * device (a raw compute or render pass). Undefined when not profiling or out of slots.
 */
export function rawTimestampWrites(root: TgpuRoot, timer: GpuPassTimer | undefined, label?: string): GPUComputePassTimestampWrites | undefined {
    const writes = timer?.next(label)
    if (!writes) return undefined
    return {
        querySet: root.unwrap(writes.querySet) as unknown as GPUQuerySet,
        beginningOfPassWriteIndex: writes.beginningOfPassWriteIndex,
        endOfPassWriteIndex: writes.endOfPassWriteIndex,
    }
}

/** A timer that names every pass taken through it `<label>` or `<label> · <pass>`. */
export function labelledTimer(timer: GpuPassTimer | undefined, label: string): GpuPassTimer | undefined {
    if (!timer) return undefined
    return {next: (pass) => timer.next(pass ? `${label} · ${pass}` : label)}
}
