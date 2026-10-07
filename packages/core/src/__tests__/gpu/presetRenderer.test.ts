import {describe, it, expect} from 'vitest'
import {createRendererFromJSON} from '@coreroot/presetRenderer'

/**
 * createRendererFromJSON — the GPU-free part of the headless preset renderer's surface. The
 * frame-locked path (`renderFrame({deltaSeconds})`) needs a device to draw; what is checked
 * here is the public shape and the input validation that runs before any GPU work.
 */
describe('createRendererFromJSON public API', () => {
    const r = createRendererFromJSON({components: []})
    for (const m of ['initialize', 'renderFrame', 'updatePreset', 'getGPUContext', 'dispose', 'setOnUnavailable', 'getFailureReason'] as const) {
        it(`exposes ${m}()`, () => {
            expect(typeof r[m]).toBe('function')
        })
    }
})

describe('renderFrame({deltaSeconds})', () => {
    it('rejects a non-finite delta before touching the renderer', async () => {
        const r = createRendererFromJSON({components: []})
        await expect(r.renderFrame({deltaSeconds: Number.NaN})).rejects.toThrow(TypeError)
        await expect(r.renderFrame({deltaSeconds: Number.POSITIVE_INFINITY})).rejects.toThrow(TypeError)
    })

    it('is a silent no-op before initialize() (real-time and frame-locked alike)', async () => {
        const r = createRendererFromJSON({components: []})
        await expect(r.renderFrame()).resolves.toBeUndefined()
        await expect(r.renderFrame({deltaSeconds: 1 / 60})).resolves.toBeUndefined()
        await expect(r.renderFrame({deltaSeconds: 0})).resolves.toBeUndefined()
    })
})
