/**
 * Device-loss recovery + crash-loop guard (gpu/recovery.ts). GPU-free: the guard is a pure
 * state machine and the wrapper is exercised against a fake renderer.
 */
import {describe, expect, it, vi} from 'vitest'
import {createCrashGuard, withGpuRecovery, type RecoverableRenderer} from '../../gpu/recovery'

function clock(start = 1_000) {
    let t = start
    return {now: () => t, advance: (ms: number) => { t += ms }}
}

describe('createCrashGuard — device loss', () => {
    it('rebuilds on the first losses and disables the user-WGSL suspects on the third within 30 s', () => {
        const c = clock()
        const warn = vi.fn()
        const guard = createCrashGuard({now: c.now, warn})
        guard.noteLayer('glow', {name: 'MyGlow'})
        expect(guard.deviceLost(['glow'])).toEqual({action: 'rebuild', disabled: []})
        c.advance(5_000)
        expect(guard.deviceLost(['glow'])).toEqual({action: 'rebuild', disabled: []})
        c.advance(5_000)
        expect(guard.deviceLost(['glow'])).toEqual({action: 'rebuild', disabled: [{id: 'glow', name: 'MyGlow'}]})
        expect(guard.isDisabled('glow')).toBe(true)
        expect(warn).toHaveBeenCalledTimes(1)
        expect(warn.mock.calls[0][0]).toContain('"MyGlow" (id glow)')
    })

    it('does not accumulate losses spread wider than the window', () => {
        const c = clock()
        const guard = createCrashGuard({now: c.now, warn: () => {}})
        for (let i = 0; i < 10; i++) {
            expect(guard.deviceLost(['glow']).action).toBe('rebuild')
            c.advance(16_000) // two losses never share a 30 s window with a third
        }
        expect(guard.isDisabled('glow')).toBe(false)
    })

    it('gives up when a composition with no user WGSL keeps losing the device', () => {
        const c = clock()
        const guard = createCrashGuard({now: c.now, warn: () => {}})
        expect(guard.deviceLost([]).action).toBe('rebuild')
        expect(guard.deviceLost([]).action).toBe('rebuild')
        expect(guard.deviceLost([])).toEqual({action: 'give-up'})
    })

    it('counts per composition: once the suspect is cut, the remaining tree gets its own budget', () => {
        const c = clock()
        const guard = createCrashGuard({now: c.now, warn: () => {}})
        guard.deviceLost(['bad'])
        guard.deviceLost(['bad'])
        expect(guard.deviceLost(['bad']).action).toBe('rebuild') // 'bad' disabled
        // The rest of the tree (no suspects left) starts at zero strikes.
        expect(guard.deviceLost(['bad']).action).toBe('rebuild')
        expect(guard.deviceLost([]).action).toBe('rebuild')
        expect(guard.deviceLost([]).action).toBe('give-up')
    })

    it('honours custom maxStrikes / windowMs', () => {
        const c = clock()
        const guard = createCrashGuard({now: c.now, warn: () => {}, maxStrikes: 2, windowMs: 1_000})
        guard.deviceLost(['a'])
        c.advance(1_500)
        expect(guard.deviceLost(['a']).action).toBe('rebuild')
        expect(guard.isDisabled('a')).toBe(false)
        guard.deviceLost(['a'])
        expect(guard.isDisabled('a')).toBe(true)
    })

    it('names every suspect in a single warning and falls back to the id without a name', () => {
        const warn = vi.fn()
        const guard = createCrashGuard({warn})
        guard.noteLayer('b', {name: 'Bloom'})
        for (let i = 0; i < 3; i++) guard.deviceLost(['b', 'a', 'a'])
        expect(warn).toHaveBeenCalledTimes(1)
        expect(warn.mock.calls[0][0]).toContain('layers "a", "Bloom" (id b)')
    })
})

describe('createCrashGuard — validation failures', () => {
    it('disables the suspects at once, warns once, and is idempotent', () => {
        const warn = vi.fn()
        const guard = createCrashGuard({warn})
        guard.noteLayer('custom', {name: 'Ripple'})
        expect(guard.validationFailed(['custom'])).toEqual([{id: 'custom', name: 'Ripple'}])
        expect(guard.validationFailed(['custom'])).toEqual([])
        expect(warn).toHaveBeenCalledTimes(1)
        expect(warn.mock.calls[0][0]).toContain('failed GPU validation')
    })

    it('disables nothing (and stays quiet) when the composition carried no user WGSL', () => {
        const warn = vi.fn()
        const guard = createCrashGuard({warn})
        expect(guard.validationFailed([])).toEqual([])
        expect(warn).not.toHaveBeenCalled()
    })

    it('re-enables a layer only when it is re-registered with a different definition', () => {
        const guard = createCrashGuard({warn: () => {}})
        const broken = {name: 'Ripple'}
        guard.noteLayer('custom', broken)
        guard.validationFailed(['custom'])
        guard.noteLayer('custom', broken) // reset-signal re-registration: same definition
        expect(guard.isDisabled('custom')).toBe(true)
        guard.noteLayer('custom', {name: 'Ripple'}) // the author fixed the body
        expect(guard.isDisabled('custom')).toBe(false)
        expect(guard.disabledLayers()).toEqual([])
    })
})

// ── Wrapper ───────────────────────────────────────────────────────────────────────────

interface FakeNode {metadata: Record<string, unknown> | null}

function fakeRenderer() {
    const nodes = new Map<string, FakeNode>()
    let onLost: ((reason: string) => void) | null = null
    let onBroken: ((ids: string[], message: string) => void) | null = null
    let suspects: string[] = []
    const r = {
        initialize: vi.fn(async (_o: unknown) => {}),
        cleanup: vi.fn(() => { nodes.clear(); onLost = null }),
        registerNode: vi.fn((id: string, fragment: unknown, _parent: string | null, metadata: Record<string, unknown> | null) => {
            if (fragment === null) nodes.delete(id)
            else nodes.set(id, {metadata})
        }),
        updateNodeMetadata: vi.fn((id: string, metadata: Record<string, unknown>) => {
            const n = nodes.get(id)
            if (n) n.metadata = {...(n.metadata ?? {}), ...metadata}
        }),
        setOnDeviceLost: (cb: ((reason: string) => void) | null) => { onLost = cb },
        setOnCompositionBroken: (cb: ((ids: string[], message: string) => void) | null) => { onBroken = cb },
        getCustomWgslNodeIds: () => suspects,
        startAnimation: () => 'still here',
    }
    return {
        r,
        nodes,
        setSuspects: (ids: string[]) => { suspects = ids },
        loseDevice: (reason = 'gpu crash') => onLost?.(reason),
        breakComposition: (ids: string[]) => onBroken?.(ids, 'error: bad wgsl'),
        armed: () => onLost !== null,
    }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))
const def = (name: string) => ({name, fragment: () => null})

describe('withGpuRecovery', () => {
    it('keeps the rest of the renderer surface and arms on initialize', async () => {
        const f = fakeRenderer()
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {rebuild: () => {}})
        expect(r.startAnimation()).toBe('still here')
        expect(f.armed()).toBe(false)
        await r.initialize({})
        expect(f.armed()).toBe(true)
        expect(f.r.initialize).toHaveBeenCalledTimes(1)
    })

    it('cleans up and rebuilds on device loss, then re-arms on the next initialize', async () => {
        const f = fakeRenderer()
        const rebuild = vi.fn(async () => { await r.initialize({}) })
        const onDeviceLost = vi.fn()
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {rebuild, onDeviceLost, guard: createCrashGuard({warn: () => {}})})
        await r.initialize({})
        f.loseDevice('boom')
        expect(onDeviceLost).toHaveBeenCalledWith('boom')
        expect(f.r.cleanup).toHaveBeenCalledTimes(1)
        expect(rebuild).toHaveBeenCalledTimes(1)
        await flush()
        expect(f.armed()).toBe(true)
    })

    it('disables a crash-looping custom layer on re-registration and keeps it hidden through metadata updates', async () => {
        const f = fakeRenderer()
        const custom = def('Ripple')
        const register = () => {
            r.registerNode('root', () => null, null, null, {} as never, def('Root') as never)
            r.registerNode('bg', () => null, 'root', {visible: true}, {} as never, def('Gradient') as never)
            r.registerNode('custom', () => null, 'root', {opacity: 1}, {} as never, custom as never)
        }
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {
            rebuild: async () => { await r.initialize({}); register() },
            guard: createCrashGuard({warn: () => {}}),
        })
        await r.initialize({})
        register()
        f.setSuspects(['custom'])
        for (let i = 0; i < 3; i++) {
            f.loseDevice()
            await flush()
        }
        expect(f.nodes.get('custom')?.metadata).toMatchObject({visible: false, opacity: 1})
        expect(f.nodes.get('bg')?.metadata).toMatchObject({visible: true})
        // A host pushing the layer's own visibility can't switch it back on.
        r.updateNodeMetadata('custom', {visible: true} as never)
        expect(f.nodes.get('custom')?.metadata?.visible).toBe(false)
        // Unrelated metadata updates pass through untouched.
        r.updateNodeMetadata('custom', {opacity: 0.5} as never)
        expect(f.nodes.get('custom')?.metadata).toMatchObject({visible: false, opacity: 0.5})
    })

    it('hides the layers a broken composition carried', async () => {
        const f = fakeRenderer()
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {rebuild: () => {}, guard: createCrashGuard({warn: () => {}})})
        await r.initialize({})
        r.registerNode('custom', () => null, null, {}, {} as never, def('Ripple') as never)
        f.breakComposition(['custom'])
        expect(f.nodes.get('custom')?.metadata?.visible).toBe(false)
        // The author fixes the body: a new definition registers visible again.
        r.registerNode('custom', () => null, null, {}, {} as never, def('Ripple') as never)
        expect(f.nodes.get('custom')?.metadata?.visible).toBeUndefined()
    })

    it('gives up once (no rebuild) when a library-only composition keeps losing the device', async () => {
        const f = fakeRenderer()
        const onGiveUp = vi.fn()
        const rebuild = vi.fn(async () => { await r.initialize({}) })
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {rebuild, onGiveUp, guard: createCrashGuard({warn: () => {}})})
        await r.initialize({})
        for (let i = 0; i < 3; i++) {
            f.loseDevice()
            await flush()
        }
        expect(rebuild).toHaveBeenCalledTimes(2)
        expect(onGiveUp).toHaveBeenCalledTimes(1)
        expect(onGiveUp).toHaveBeenCalledWith('device-lost')
        f.loseDevice() // late events after giving up are ignored
        expect(onGiveUp).toHaveBeenCalledTimes(1)
    })

    it('defers a loss that lands mid-rebuild until the rebuild settles', async () => {
        const f = fakeRenderer()
        let release!: () => void
        const rebuild = vi.fn(() => new Promise<void>((resolve) => { release = resolve }))
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {rebuild, guard: createCrashGuard({warn: () => {}})})
        await r.initialize({})
        const lost = f.r.setOnDeviceLost
        let handler: ((reason: string) => void) | null = null
        f.r.setOnDeviceLost = (cb) => { handler = cb; lost(cb) }
        await r.initialize({}) // capture the armed handler
        f.loseDevice()
        expect(rebuild).toHaveBeenCalledTimes(1)
        handler!('second loss during rebuild')
        expect(rebuild).toHaveBeenCalledTimes(1)
        release()
        await flush()
        expect(rebuild).toHaveBeenCalledTimes(2)
    })

    it('ignores losses after disposeRecovery', async () => {
        const f = fakeRenderer()
        const rebuild = vi.fn()
        const r = withGpuRecovery(f.r as typeof f.r & RecoverableRenderer, {rebuild})
        await r.initialize({})
        let handler: ((reason: string) => void) | null = null
        const lost = f.r.setOnDeviceLost
        f.r.setOnDeviceLost = (cb) => { if (cb) handler = cb; lost(cb) }
        await r.initialize({})
        r.disposeRecovery()
        handler!('late')
        expect(rebuild).not.toHaveBeenCalled()
        expect(f.armed()).toBe(false)
    })
})
