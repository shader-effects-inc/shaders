import {describe, it, expect} from 'vitest'
import {createRendererFromJSON} from '@coreroot/presetRenderer'
import {shaderRendererGPU} from '@coreroot/gpu/index'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import GaborNoise from '@coreroot/shaders/GaborNoise/index'
import {RootContainer} from './noiseHarness'

/**
 * createRendererFromJSON + the frame-locked clock behind `renderFrame({deltaSeconds})`.
 *
 * Drawing needs a WebGPU device, which vitest does not have, so rendered pixels are not
 * compared here (the parity harness does that). What IS checked, through the core renderer's
 * __testing step helpers, is the property that makes offline output deterministic: after
 * locking, every clock is the exact sum of the deltas passed in — independent of what rendered
 * before the lock and of wall-clock time between steps.
 */
const GN = GaborNoise as GpuShaderDefinition
const meta = (m: Partial<NodeMetadata> = {}): NodeMetadata => ({blendMode: 'normal', opacity: undefined, ...m}) as NodeMetadata
const u = (value: unknown) => ({value})
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** A ready, device-free core renderer with one animated node, after `preLockFrames` wall-clock-ish frames. */
function lockedRenderer(preLockFrames: number) {
    const r = shaderRendererGPU()
    r.__testing.setTestReady({width: 800, height: 600})
    r.registerNode('root', RootContainer.fragment, null, meta(), {}, RootContainer)
    r.registerNode('gn', GN.fragment, 'root', meta(), {speed: u(1)} as never, GN)
    for (let i = 0; i < preLockFrames; i++) {
        r.__testing.stepMouseDrivers(0.016)
        r.__testing.stepAnimatedTime(0.016)
    }
    r.setFrameLocked(true)
    return r
}
const clocks = (r: ReturnType<typeof shaderRendererGPU>) => ({
    time: r.__testing.getFrameDiagnostics().globalElapsedTime,
    anim: r.__testing.getAnimatedTimeValue('gn')!,
})

describe('createRendererFromJSON public API', () => {
    const r = createRendererFromJSON({components: []})
    for (const m of ['initialize', 'renderFrame', 'updatePreset', 'getGPUContext', 'dispose', 'setOnUnavailable', 'getFailureReason'] as const) {
        it(`exposes ${m}()`, () => {
            expect(typeof r[m]).toBe('function')
        })
    }
})

describe('renderFrame({deltaSeconds}) input handling', () => {
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

describe('frame-locked clock (core setFrameLocked)', () => {
    it('locking restarts every clock at 0, whatever rendered before', () => {
        const fresh = lockedRenderer(0)
        const warmed = lockedRenderer(45)
        expect(clocks(fresh)).toEqual({time: 0, anim: 0})
        expect(clocks(warmed)).toEqual({time: 0, anim: 0})
    })

    it('identical delta sequences give identical clocks regardless of pre-lock history', () => {
        const a = lockedRenderer(0)
        const b = lockedRenderer(45)
        const deltas = [1 / 60, 1 / 60, 1 / 30, 0, 2, 1 / 60]
        for (const dt of deltas) {
            for (const r of [a, b]) {
                r.__testing.stepMouseDrivers(dt)
                r.__testing.stepAnimatedTime(dt)
            }
        }
        expect(clocks(a)).toEqual(clocks(b))
        const total = deltas.reduce((s, d) => s + d, 0)
        expect(clocks(a).time).toBeCloseTo(total, 12)
        expect(clocks(a).anim).toBeCloseTo(total, 12)
    })

    it('the locked `time` clock ignores wall-clock gaps and any shared time origin', async () => {
        const r = lockedRenderer(0)
        r.setTimeOrigin(performance.now() - 5000) // would read as ~5s if wall-clock derivation ran
        r.__testing.stepMouseDrivers(1 / 60)
        await sleep(25)
        r.__testing.stepMouseDrivers(1 / 60)
        expect(clocks(r).time).toBeCloseTo(2 / 60, 12)
    })

    it('renderSyntheticFrame rejects a non-finite delta without touching the clocks', async () => {
        const r = lockedRenderer(0)
        r.__testing.stepMouseDrivers(1 / 60)
        r.__testing.stepAnimatedTime(1 / 60)
        const before = clocks(r)
        await expect(r.renderSyntheticFrame(Number.NaN)).rejects.toThrow(TypeError)
        await expect(r.renderSyntheticFrame(Number.NEGATIVE_INFINITY)).rejects.toThrow(TypeError)
        expect(clocks(r)).toEqual(before)
    })

    it('stepping to a time matches jumping straight there', () => {
        const stepped = lockedRenderer(0)
        const jumped = lockedRenderer(0)
        for (let i = 0; i < 120; i++) {
            stepped.__testing.stepMouseDrivers(1 / 60)
            stepped.__testing.stepAnimatedTime(1 / 60)
        }
        jumped.__testing.stepMouseDrivers(2)
        jumped.__testing.stepAnimatedTime(2)
        expect(clocks(stepped).time).toBeCloseTo(clocks(jumped).time, 10)
        expect(clocks(stepped).anim).toBeCloseTo(clocks(jumped).anim, 10)
    })
})
