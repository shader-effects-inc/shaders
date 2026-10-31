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
    /** Timestamp writes for the next pass, or null once this frame's slots are used up. */
    next(): PassTimestampWrites | null
}

export interface GpuFrameTimer extends GpuPassTimer {
    /** Start a frame: slots count from zero again. */
    begin(): void
    /** The frame's GPU time once its passes finish, or null if it timed no passes. */
    read(): Promise<{ms: number; passes: number} | null>
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
    return {
        begin() {
            used = 0
        },
        next() {
            if (used >= MAX_PASSES) return null
            const slot = used++
            return {querySet, beginningOfPassWriteIndex: slot * 2, endOfPassWriteIndex: slot * 2 + 1}
        },
        async read() {
            const passes = used
            if (!passes) return null
            querySet.resolve()
            const stamps = await querySet.read()
            let ns = 0n
            for (let i = 0; i < passes; i++) {
                const start = stamps[i * 2]
                const end = stamps[i * 2 + 1]
                if (start !== undefined && end !== undefined && end > start) ns += end - start
            }
            return {ms: Number(ns) / 1e6, passes}
        },
        destroy() {
            querySet.destroy()
        },
    }
}
