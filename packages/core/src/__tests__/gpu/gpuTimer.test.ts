import {describe, it, expect} from 'vitest'
import {canTimeGpu, createGpuFrameTimer} from '@coreroot/gpu/gpuTimer'
import {createComputeDispatcher, type ComputeStep} from '@coreroot/gpu/compute'

/**
 * GPU pass timing, GPU-free: a mocked root hands out a query set whose read returns the
 * timestamps the test chose, so slot bookkeeping and the per-frame sum are checked directly.
 */
function mockRoot(stamps: bigint[], features: string[] = ['timestamp-query']) {
    const resolved: number[] = []
    const querySet = {
        resolve: () => resolved.push(1),
        read: async () => stamps,
        destroy: () => {},
    }
    return {
        root: {enabledFeatures: new Set(features), createQuerySet: () => querySet} as never,
        resolved,
    }
}

describe('createGpuFrameTimer', () => {
    it('hands each pass its own timestamp pair and sums their spans', async () => {
        const {root, resolved} = mockRoot([1_000_000n, 3_000_000n, 3_500_000n, 4_000_000n])
        const timer = createGpuFrameTimer(root)
        timer.begin()
        const a = timer.next()!
        const b = timer.next()!
        expect([a.beginningOfPassWriteIndex, a.endOfPassWriteIndex]).toEqual([0, 1])
        expect([b.beginningOfPassWriteIndex, b.endOfPassWriteIndex]).toEqual([2, 3])
        // 2ms + 0.5ms of GPU work; the 0.5ms gap between the passes does not count.
        expect(await timer.read()).toEqual({ms: 2.5, passes: 2})
        expect(resolved).toHaveLength(1)
    })

    it('starts counting slots again each frame, and reads nothing for a frame with no passes', async () => {
        const {root} = mockRoot([0n, 1_000_000n])
        const timer = createGpuFrameTimer(root)
        timer.begin()
        timer.next()
        timer.begin()
        expect(timer.next()!.beginningOfPassWriteIndex).toBe(0)
        timer.begin()
        expect(await timer.read()).toBeNull()
    })

    it('stops handing out slots once a frame has used them all', () => {
        const {root} = mockRoot([])
        const timer = createGpuFrameTimer(root)
        timer.begin()
        let given = 0
        while (timer.next()) given++
        expect(given).toBe(64)
    })

    it('knows when the device cannot time passes', () => {
        expect(canTimeGpu(mockRoot([]).root)).toBe(true)
        expect(canTimeGpu(mockRoot([], []).root)).toBe(false)
    })
})

describe('timed compute dispatch', () => {
    it('gives each pipeline step a timestamp pair, and leaves thunks alone', () => {
        const {root} = mockRoot([])
        const timer = createGpuFrameTimer(root)
        timer.begin()
        const timed: number[] = []
        const step = {
            guarded: {} as never,
            with() { return this },
            withTimestampWrites(writes: {beginningOfPassWriteIndex: number}) {
                timed.push(writes.beginningOfPassWriteIndex)
                return this
            },
            dispatch() {},
            dispatchThreads() {},
        } as unknown as ComputeStep
        createComputeDispatcher({} as never).dispatch([step, () => {}, step], timer)
        expect(timed).toEqual([0, 2])
    })
})
