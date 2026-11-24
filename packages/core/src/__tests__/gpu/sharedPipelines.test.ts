import {describe, it, expect} from 'vitest'
import {acquireShared, registerShared, releaseShared, sharedEntryCount} from '@coreroot/gpu/sharedPipelines'

/** A compiled-looking pipeline: TypeGPU keeps its memo behind an internal symbol. */
function pipeline(catchall?: unknown) {
    const internal = Symbol('typegpu:test:$internal')
    return {[internal]: {core: {unwrap: () => ({catchall})}}} as never
}

const layout = {} as never

describe('sharedPipelines', () => {
    it('shares an entry once its owner has compiled, and counts references', async () => {
        const root = {} as never
        const own = registerShared(root, 'h1', layout, [{pipeline: pipeline(), reads: [], externalReads: []}], Promise.resolve())
        expect(own?.refs).toBe(1)
        const borrowed = acquireShared(root, 'h1')
        expect(borrowed).toBe(own)
        expect(own!.refs).toBe(2)
        expect(await own!.ready).toBe(true)
        releaseShared(root, own!)
        releaseShared(root, own!)
        expect(own!.refs).toBe(0)
        // Kept idle for a quick toggle back
        expect(sharedEntryCount(root)).toBe(1)
    })

    it('never shares a pipeline that binds a fixed resource', async () => {
        const root = {} as never
        const own = registerShared(root, 'h2', layout, [{pipeline: pipeline([0, {}]), reads: [], externalReads: []}], Promise.resolve())
        expect(await own!.ready).toBe(false)
        expect(acquireShared(root, 'h2')).toBeNull()
    })

    it('drops an entry whose compile failed', async () => {
        const root = {} as never
        const own = registerShared(root, 'h3', layout, [{pipeline: pipeline(), reads: [], externalReads: []}], Promise.reject(new Error('bad')))
        expect(await own!.ready).toBe(false)
        expect(sharedEntryCount(root)).toBe(0)
    })

    it('keeps separate caches per root and a single owner per key', () => {
        const a = {} as never
        const b = {} as never
        expect(registerShared(a, 'h', layout, [], null)).not.toBeNull()
        expect(registerShared(a, 'h', layout, [], null)).toBeNull()
        expect(acquireShared(b, 'h')).toBeNull()
    })

    it('forgets the oldest unused entries past the idle limit', () => {
        const root = {} as never
        const entries = Array.from({length: 12}, (_, i) => registerShared(root, `k${i}`, layout, [], null)!)
        for (const e of entries) releaseShared(root, e)
        expect(sharedEntryCount(root)).toBe(8)
        expect(acquireShared(root, 'k0')).toBeNull()
        expect(acquireShared(root, 'k11')).not.toBeNull()
    })
})
