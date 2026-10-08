import {describe, it, expect, vi} from 'vitest'
import {
    applySpring,
    applyEasing,
    applyBounceEase,
    clampToTextureCap,
    deriveElapsedTime,
    frameGate,
    runFrameSequence,
    createFrameLoop,
    OFF_SCREEN_FPS_INTERVAL,
    MIN_FRAME_INTERVAL,
    type FrameSequence,
} from '@coreroot/gpu/frame'

/**
 * B5b frame.ts tests — the PURE frame-layer helpers, golden-tested against the values the old
 * renderer produced (spring physics, easing, texture-cap clamp, shared-clock time, throttle
 * gate) plus the frame-order orchestrator and the RAF loop factory. All GPU-free.
 */

describe('applySpring (verbatim port of renderer.ts:416)', () => {
    it('returns the target immediately when smoothing and momentum are both 0', () => {
        expect(applySpring(0.2, 5, 0.9, 0, 0, 0.016)).toEqual([0.9, 0])
    })

    it('matches hand-computed dynamics for smoothing=0.5 (stiffness 20)', () => {
        // stiffness = 200 * 0.01^0.5 = 20; force = 20; newVel = 20*0.016 = 0.32; pos = 0.32*0.016.
        const [pos, vel] = applySpring(0, 0, 1, 0.5, 0, 0.016)
        expect(vel).toBeCloseTo(0.32, 6)
        expect(pos).toBeCloseTo(0.00512, 6)
    })

    it('applies momentum damping (smoothing=0, momentum=0.5, stiffness 200)', () => {
        // stiffness = 200; damping = 2*sqrt(200)*(1-0.425); force = 200; newVel = 200*0.016 = 3.2.
        const [pos, vel] = applySpring(0, 0, 1, 0, 0.5, 0.016)
        expect(vel).toBeCloseTo(3.2, 6)
        expect(pos).toBeCloseTo(0.0512, 6)
    })

    it('stays stable at the clamped 0.1s delta (1 FPS off-screen throttle / return from a hidden tab)', () => {
        // Un-substepped semi-implicit Euler diverges here (sign flip + growth every step). Sub-stepping
        // must keep it converging on the target instead — the "shape spun like crazy" regression.
        let pos = 0.3, vel = 0
        for (let i = 0; i < 30; i++) [pos, vel] = applySpring(pos, vel, 0.5, 0.1, 0, 0.1)
        expect(pos).toBeCloseTo(0.5, 3)
        expect(Math.abs(vel)).toBeLessThan(0.01)
    })

    it('a normal 60 FPS delta takes exactly one step (behaviour unchanged)', () => {
        const k = 200 * Math.pow(0.01, 0.3), dmp = 2 * Math.sqrt(k)
        const v = 0.1 + (k * (1 - 0.2) - dmp * 0.1) * 0.016
        expect(applySpring(0.2, 0.1, 1, 0.3, 0, 0.016)).toEqual([0.2 + v * 0.016, v])
    })
})

describe('applyEasing / applyBounceEase (verbatim port of renderer.ts:3797/:3805)', () => {
    it('linear is the identity', () => {
        expect(applyEasing(0.3, 'linear')).toBe(0.3)
    })
    it('sine (default) endpoints + midpoint', () => {
        expect(applyEasing(0, 'sine')).toBeCloseTo(0, 12)
        expect(applyEasing(0.5, 'sine')).toBeCloseTo(0.5, 12)
        expect(applyEasing(1, 'sine')).toBeCloseTo(1, 12)
        // Unknown easing falls back to sine.
        expect(applyEasing(0.5, 'nonsense')).toBeCloseTo(0.5, 12)
    })
    it('quad ease-in/out', () => {
        expect(applyEasing(0.25, 'quad')).toBeCloseTo(0.125, 12)
        expect(applyEasing(0.75, 'quad')).toBeCloseTo(0.875, 12)
    })
    it('expo endpoints + midpoint', () => {
        expect(applyEasing(0, 'expo')).toBe(0)
        expect(applyEasing(1, 'expo')).toBe(1)
        expect(applyEasing(0.5, 'expo')).toBeCloseTo(0.5, 12)
    })
    it('bounce matches the multi-segment curve', () => {
        expect(applyBounceEase(0.5)).toBeCloseTo(0.765625, 6)
        expect(applyEasing(0.5, 'bounce')).toBeCloseTo(0.765625, 6)
    })
})

describe('clampToTextureCap (port of renderer.ts:655)', () => {
    const env = {maxTextureDim: 8192, pixelRatio: 2, viewportWidth: 1920, viewportHeight: 1080}

    it('leaves a small request untouched', () => {
        expect(clampToTextureCap(800, 600, env)).toEqual({width: 800, height: 600})
    })
    it('clamps to the viewport, preserving aspect ratio', () => {
        // capH = 1080 binds; scale = 1080/3000 = 0.36 → 4000*0.36 = 1440.
        expect(clampToTextureCap(4000, 3000, env)).toEqual({width: 1440, height: 1080})
    })
    it('clamps to the GPU max-texture-dim cap (pr=1)', () => {
        // gpuCssCap = floor(4096/1) - 1 = 4095; scale = 4095/5000.
        const out = clampToTextureCap(5000, 5000, {maxTextureDim: 4096, pixelRatio: 1, viewportWidth: 10000, viewportHeight: 10000})
        expect(out).toEqual({width: 4095, height: 4095})
    })
})

describe('deriveElapsedTime (shared-clock, port of renderer.ts:3829)', () => {
    it('accumulates delta with no shared origin', () => {
        expect(deriveElapsedTime(null, 1.0, 0.016, 999999)).toBeCloseTo(1.016, 9)
    })
    it('derives from the shared wall-clock origin', () => {
        expect(deriveElapsedTime(1000, 5, 0.016, 3000)).toBeCloseTo(2.0, 9)
    })
})

describe('frameGate (throttle port of renderer.ts:4108)', () => {
    it('always renders on the first frame (lastRenderTime 0), delta clamped to 0.016', () => {
        const r = frameGate(1000, {lastRenderTime: 0, isVisible: true, forceFullFrameRate: false})
        expect(r.render).toBe(true)
        expect(r.deltaTime).toBeCloseTo(0.016, 9)
    })
    it('caps on-screen frames at ~60fps with jitter tolerance', () => {
        const now = 1000
        expect(frameGate(now, {lastRenderTime: now - (MIN_FRAME_INTERVAL - 1), isVisible: true, forceFullFrameRate: false}).render).toBe(false)
        expect(frameGate(now, {lastRenderTime: now - (MIN_FRAME_INTERVAL + 1), isVisible: true, forceFullFrameRate: false}).render).toBe(true)
    })
    it('throttles off-screen frames to 1fps', () => {
        const now = 5000
        expect(frameGate(now, {lastRenderTime: now - 500, isVisible: false, forceFullFrameRate: false}).render).toBe(false)
        expect(frameGate(now, {lastRenderTime: now - (OFF_SCREEN_FPS_INTERVAL + 1), isVisible: false, forceFullFrameRate: false}).render).toBe(true)
    })
    it('forceFullFrameRate bypasses the off-screen throttle (but keeps the 60fps cap)', () => {
        const now = 5000
        expect(frameGate(now, {lastRenderTime: now - 100, isVisible: false, forceFullFrameRate: true}).render).toBe(true)
        expect(frameGate(now, {lastRenderTime: now - 5, isVisible: false, forceFullFrameRate: true}).render).toBe(false)
    })
    it('clamps a long delta to 0.1s', () => {
        const r = frameGate(10000, {lastRenderTime: 5000, isVisible: true, forceFullFrameRate: false})
        expect(r.render).toBe(true)
        expect(r.deltaTime).toBe(0.1)
    })
    it('keeps every frame of a 60Hz display when given its frame timestamps', () => {
        // RAF timestamps follow the refresh, so they arrive a refresh interval apart. Panels report
        // timings either side of 60Hz (59.94, 60.10), and none of them may lose a frame to the cap.
        for (const hz of [59.94, 60, 60.1]) {
            let lastRenderTime = 0
            let rendered = 0
            for (let i = 1; i <= 600; i++) {
                const now = 1000 + (i * 1000) / hz
                if (frameGate(now, {lastRenderTime, isVisible: true, forceFullFrameRate: false}).render) {
                    lastRenderTime = now
                    rendered++
                }
            }
            expect(rendered).toBe(600)
        }
    })
    it('keeps the 60fps cap on high-refresh displays (every other frame at 120Hz)', () => {
        let lastRenderTime = 0
        let rendered = 0
        for (let i = 1; i <= 240; i++) {
            const now = 1000 + (i * 1000) / 120
            if (frameGate(now, {lastRenderTime, isVisible: true, forceFullFrameRate: false}).render) {
                lastRenderTime = now
                rendered++
            }
        }
        expect(rendered).toBe(120)
    })
})

describe('runFrameSequence (canonical frame order)', () => {
    function recorder(ensure: boolean) {
        const calls: string[] = []
        const seq: FrameSequence = {
            updateDrivers: () => calls.push('drivers'),
            ensureComposition: () => {
                calls.push('ensure')
                return ensure
            },
            beforeRender: () => calls.push('before'),
            flush: () => calls.push('flush'),
            render: () => calls.push('render'),
            markReady: () => calls.push('markReady'),
            afterRender: () => calls.push('after'),
        }
        return {calls, seq}
    }

    it('runs steps in wiring-contract order: drivers → ensure → before → flush → render → markReady → after', () => {
        const {calls, seq} = recorder(true)
        runFrameSequence(seq)
        expect(calls).toEqual(['drivers', 'ensure', 'before', 'flush', 'render', 'markReady', 'after'])
    })
    it('short-circuits after ensureComposition returns false (nothing composed)', () => {
        const {calls, seq} = recorder(false)
        runFrameSequence(seq)
        expect(calls).toEqual(['drivers', 'ensure'])
    })
    it('drivers always run before flush (uniform patches reflect this frame)', () => {
        const {calls, seq} = recorder(true)
        runFrameSequence(seq)
        expect(calls.indexOf('drivers')).toBeLessThan(calls.indexOf('flush'))
        expect(calls.indexOf('flush')).toBeLessThan(calls.indexOf('render'))
        expect(calls.indexOf('render')).toBeLessThan(calls.indexOf('markReady'))
    })
})

describe('createFrameLoop', () => {
    it('starts, ticks via requestAnimationFrame, and stops idempotently', () => {
        const scheduled: FrameRequestCallback[] = []
        const rafSpy = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((cb) => {
            scheduled.push(cb)
            return scheduled.length
        })
        const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {})
        let ticks = 0
        const loop = createFrameLoop(() => ticks++)

        expect(loop.running).toBe(false)
        loop.start()
        expect(loop.running).toBe(true)
        loop.start() // idempotent — no second schedule
        expect(scheduled.length).toBe(1)

        // Drive one frame: tick fires, then reschedules.
        scheduled[0](performance.now())
        expect(ticks).toBe(1)
        expect(scheduled.length).toBe(2)

        loop.stop()
        expect(loop.running).toBe(false)
        expect(cancelSpy).toHaveBeenCalled()

        rafSpy.mockRestore()
        cancelSpy.mockRestore()
    })

    it("passes each frame's RAF timestamp to tick (the clock frameGate caps against)", () => {
        const scheduled: FrameRequestCallback[] = []
        const rafSpy = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((cb) => {
            scheduled.push(cb)
            return scheduled.length
        })
        const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {})
        const times: number[] = []
        const loop = createFrameLoop((frameTime) => times.push(frameTime))

        loop.start()
        scheduled[0](1000)
        scheduled[1](1016.667)
        expect(times).toEqual([1000, 1016.667])

        loop.stop()
        rafSpy.mockRestore()
        cancelSpy.mockRestore()
    })
})
