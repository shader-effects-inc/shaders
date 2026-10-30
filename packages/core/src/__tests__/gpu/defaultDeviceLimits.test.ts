import {describe, it, expect, vi, beforeEach} from 'vitest'

/**
 * The shared default device asks for the adapter's own per-stage limits (sampled textures,
 * samplers, storage, uniform size) instead of the WebGPU defaults, so a large composition — a
 * pass reading 17+ textures — compiles. Should the raised request be refused, it falls back to
 * the defaults rather than failing. GPU-free: `typegpu` is mocked.
 */

const mocks = vi.hoisted(() => ({
    initCalls: [] as Array<{device?: {requiredLimits?: Record<string, number>}}>,
    rejectRaised: false,
}))

vi.mock('typegpu', () => ({
    default: {
        initFromDevice: ({device}: {device: unknown}) => ({device}),
        init: async (options: {device?: {requiredLimits?: Record<string, number>}}) => {
            mocks.initCalls.push(options)
            if (mocks.rejectRaised && Object.keys(options.device?.requiredLimits ?? {}).length) throw new Error('limit exceeded')
            return {device: {lost: new Promise(() => {})}}
        },
    },
}))

const {acquireRoot, __resetDefaultRoot} = await import('@coreroot/gpu/root')

beforeEach(() => {
    mocks.initCalls = []
    mocks.rejectRaised = false
    __resetDefaultRoot()
    ;(navigator as {gpu?: unknown}).gpu = {
        requestAdapter: async () => ({limits: {maxSampledTexturesPerShaderStage: 48, maxSamplersPerShaderStage: 16, maxTextureDimension2D: 16384}}),
    }
})

describe('default device limits', () => {
    it('requests the adapter maxima for the per-stage limits big compositions outgrow', async () => {
        await acquireRoot()
        expect(mocks.initCalls).toHaveLength(1)
        expect(mocks.initCalls[0]!.device?.requiredLimits).toEqual({maxSampledTexturesPerShaderStage: 48, maxSamplersPerShaderStage: 16})
    })

    it('falls back to the WebGPU defaults when the raised request is refused', async () => {
        mocks.rejectRaised = true
        const ctx = await acquireRoot()
        expect(ctx.device).toBeTruthy()
        expect(mocks.initCalls.map(c => c.device?.requiredLimits)).toEqual([{maxSampledTexturesPerShaderStage: 48, maxSamplersPerShaderStage: 16}, {}])
    })
})
