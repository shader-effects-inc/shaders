/**
 * Device-loss recovery + crash-loop guard, shared by every host (js createShader and the
 * React / Vue / Svelte / Solid `<Shader>` roots).
 *
 * Two pieces:
 *
 *   - {@link createCrashGuard} — the pure, GPU-free state machine. It decides, per device
 *     loss, whether to rebuild, rebuild with some layers switched off, or give up; and which
 *     layers a composition that failed GPU validation should lose. It remembers disabled
 *     layers across rebuilds.
 *
 *   - {@link withGpuRecovery} — wraps a renderer so a host gets all of it by swapping one
 *     constructor call. The wrapped renderer arms the device-loss / broken-composition hooks
 *     on every `initialize()` (`cleanup()` drops them), and forces `visible: false` on
 *     disabled layers at `registerNode` / `updateNodeMetadata` so a host re-registering its
 *     tree after a reset can never switch a bad layer back on.
 *
 * The rules:
 *
 *   - Device loss: rebuild on a fresh device straight away. Losses are counted per
 *     composition (keyed by the user-WGSL layers it carried) inside a sliding window; the
 *     `maxStrikes`-th loss within `windowMs` (default 3 in 30 s) stops retrying that
 *     composition. If it carried user-WGSL layers, those are the suspects: they are disabled
 *     and the rest of the tree is rebuilt without them. If it carried none there is nothing
 *     to cut out, so the host gives up (transparent canvas, `onUnavailable`), exactly like the
 *     JS package's old rebuild budget.
 *   - GPU validation failure: the renderer already refuses to redraw a broken composition
 *     until its structure changes, and the compiler's verdict on the same WGSL never changes,
 *     so a retry can only fail identically. The user-WGSL layers it carried are disabled at
 *     once and the rest draws.
 *   - A disabled layer comes back only when it is re-registered with a DIFFERENT definition
 *     (an author fixing the body). Strike history is kept, so a layer that is still bad is
 *     cut again on its next failure instead of earning a fresh budget.
 *
 * Each disable logs ONE `console.warn` naming the layer(s); nothing else is printed.
 */

import {debugError, type GpuFailureReason} from './support'

export interface CrashGuardOptions {
    /** Losses of the same composition inside `windowMs` that stop retrying it. Default 3. */
    maxStrikes?: number
    /** Sliding window for counting losses, in milliseconds. Default 30 000. */
    windowMs?: number
    /** Clock (testing seam). Default `Date.now`. */
    now?: () => number
    /** Where the single "layer disabled" line goes. Default `console.warn`. */
    warn?: (message: string) => void
}

/** A layer the guard knows about: its registered id and a human name for the warning. */
export interface GuardedLayer {
    id: string
    name: string
}

export type DeviceLossVerdict =
    /** Rebuild now. `disabled` lists layers switched off by this loss (usually empty). */
    | {action: 'rebuild'; disabled: GuardedLayer[]}
    /** Stop retrying: the same composition keeps losing the device and has nothing to cut out. */
    | {action: 'give-up'}

export interface CrashGuard {
    /** Record a (re-)registration. A new definition for a disabled id switches it back on. */
    noteLayer(id: string, definition: unknown): void
    isDisabled(id: string): boolean
    disabledLayers(): GuardedLayer[]
    /** The device was lost while drawing a composition carrying `suspectIds` (user-WGSL layers). */
    deviceLost(suspectIds: readonly string[]): DeviceLossVerdict
    /** A composition carrying `suspectIds` failed GPU validation. Returns the layers it disables. */
    validationFailed(suspectIds: readonly string[]): GuardedLayer[]
}

export const DEFAULT_MAX_STRIKES = 3
export const DEFAULT_STRIKE_WINDOW_MS = 30_000

function definitionName(definition: unknown): string | null {
    const name = (definition as {name?: unknown} | null | undefined)?.name
    return typeof name === 'string' && name ? name : null
}

export function createCrashGuard(options: CrashGuardOptions = {}): CrashGuard {
    const maxStrikes = Math.max(1, options.maxStrikes ?? DEFAULT_MAX_STRIKES)
    const windowMs = options.windowMs ?? DEFAULT_STRIKE_WINDOW_MS
    const now = options.now ?? Date.now
    const warn = options.warn ?? ((message: string) => console.warn(message))

    /** Last definition seen per id, so a re-registration can tell "same layer" from "fixed layer". */
    const definitions = new Map<string, unknown>()
    /** Disabled id → the definition it was disabled under. */
    const disabled = new Map<string, unknown>()
    /** Composition key → timestamps of its device losses (pruned to the window). */
    const losses = new Map<string, number[]>()

    const layerOf = (id: string): GuardedLayer => ({id, name: definitionName(definitions.get(id)) ?? id})

    const describe = (layers: GuardedLayer[]): string =>
        layers.map((l) => (l.name === l.id ? `"${l.id}"` : `"${l.name}" (id ${l.id})`)).join(', ')

    const disable = (ids: readonly string[], why: string): GuardedLayer[] => {
        const fresh = ids.filter((id) => !disabled.has(id))
        for (const id of fresh) disabled.set(id, definitions.get(id))
        const layers = fresh.map(layerOf)
        if (layers.length) {
            const noun = layers.length === 1 ? 'layer' : 'layers'
            warn(`[shaders] Disabled ${noun} ${describe(layers)}: ${why}. Rendering the rest without it.`)
        }
        return layers
    }

    return {
        noteLayer(id, definition) {
            definitions.set(id, definition)
            if (disabled.has(id) && disabled.get(id) !== definition) disabled.delete(id)
        },

        isDisabled: (id) => disabled.has(id),

        disabledLayers: () => [...disabled.keys()].map(layerOf),

        deviceLost(suspectIds) {
            const suspects = [...new Set(suspectIds)].filter((id) => !disabled.has(id)).sort()
            const key = suspects.join('\u0000')
            const t = now()
            const history = (losses.get(key) ?? []).filter((at) => t - at < windowMs)
            history.push(t)
            losses.set(key, history)
            if (history.length < maxStrikes) return {action: 'rebuild', disabled: []}
            if (!suspects.length) return {action: 'give-up'}
            const seconds = Math.round(windowMs / 1000)
            const cut = disable(suspects, `the GPU device was lost ${history.length} times in ${seconds}s while it was drawing`)
            return {action: 'rebuild', disabled: cut}
        },

        validationFailed(suspectIds) {
            return disable([...new Set(suspectIds)], 'its WGSL failed GPU validation')
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// Renderer wrapper
// ═══════════════════════════════════════════════════════════════════════════════════════

/** The slice of `shaderRendererGPU()` the wrapper touches (structural, so tests can fake it). */
export interface RecoverableRenderer {
    initialize(options: never): Promise<void>
    cleanup(): void
    registerNode(id: string, fragment: unknown, parentId: string | null, metadata: unknown, ...rest: never[]): void
    updateNodeMetadata(id: string, metadata: never): void
    setOnDeviceLost(callback: ((reason: string) => void) | null): void
    setOnCompositionBroken?(callback: ((nodeIds: string[], message: string) => void) | null): void
    getCustomWgslNodeIds?(): string[]
}

export interface GpuRecoveryOptions {
    /**
     * The device was lost and the renderer has been `cleanup()`-ed: initialize it again and
     * re-register the tree (the host's existing colorSpace / toneMapping reset path). May be
     * async; a loss that lands while it runs is handled once it settles.
     */
    rebuild: () => void | Promise<void>
    /**
     * Retrying is pointless (the same composition keeps losing the device). The renderer has
     * been cleaned up; treat it like `onUnavailable('device-lost')`. Called at most once.
     */
    onGiveUp?: (reason: GpuFailureReason) => void
    /** Told about every loss before the verdict (the JS package forwards it to `onError`). */
    onDeviceLost?: (reason: string) => void
    /** Share one guard across renderer instances (the JS package builds a new one per loss). */
    guard?: CrashGuard
}

export interface GpuRecoveryControls {
    /** Stop reacting to losses (the host is unmounting). Safe to call more than once. */
    disposeRecovery(): void
    /** The guard holding disabled layers and strike history. */
    readonly crashGuard: CrashGuard
}

/**
 * Wrap a renderer with device-loss recovery and the crash-loop guard. Returns a renderer
 * (same shape, same methods) the host uses everywhere it used the raw one, plus
 * `disposeRecovery()` for unmount.
 */
export function withGpuRecovery<R extends RecoverableRenderer>(renderer: R, options: GpuRecoveryOptions): R & GpuRecoveryControls {
    const guard = options.guard ?? createCrashGuard()
    let disposed = false
    let gaveUp = false
    let rebuilding = false
    /** A loss reported while a rebuild was in flight — handled once it settles. */
    let pendingLoss: {reason: string; suspects: string[]} | null = null

    const hide = (id: string): void => {
        try {
            renderer.updateNodeMetadata(id, {visible: false} as never)
        } catch (e) {
            debugError('[gpu] disabling a layer failed:', e)
        }
    }

    const handleLoss = (reason: string, suspects: string[]): void => {
        if (disposed || gaveUp) return
        if (rebuilding) {
            pendingLoss = {reason, suspects}
            return
        }
        try {
            options.onDeviceLost?.(reason)
        } catch (e) {
            debugError('[gpu] onDeviceLost callback threw:', e)
        }
        const verdict = guard.deviceLost(suspects)
        try {
            renderer.cleanup()
        } catch (e) {
            debugError('[gpu] cleanup after device loss threw:', e)
        }
        if (verdict.action === 'give-up') {
            gaveUp = true
            try {
                options.onGiveUp?.('device-lost')
            } catch (e) {
                debugError('[gpu] onGiveUp callback threw:', e)
            }
            return
        }
        rebuilding = true
        let result: void | Promise<void>
        try {
            result = options.rebuild()
        } catch (e) {
            result = Promise.reject(e)
        }
        void Promise.resolve(result)
            .catch((e) => debugError('[gpu] rebuild after device loss failed:', e))
            .finally(() => {
                rebuilding = false
                const next = pendingLoss
                pendingLoss = null
                if (next) handleLoss(next.reason, next.suspects)
            })
    }

    const arm = (): void => {
        if (disposed) return
        renderer.setOnDeviceLost((reason) => {
            // Read the suspects before anything tears the composition down.
            let suspects: string[] = []
            try {
                suspects = renderer.getCustomWgslNodeIds?.() ?? []
            } catch {
                /* no composition to blame */
            }
            handleLoss(reason, suspects)
        })
        renderer.setOnCompositionBroken?.((nodeIds) => {
            if (disposed) return
            for (const layer of guard.validationFailed(nodeIds)) hide(layer.id)
        })
    }

    const wrapped: R & GpuRecoveryControls = Object.assign({}, renderer, {
        initialize: async (initOptions: never): Promise<void> => {
            // Arm BEFORE initializing (initialize keeps the hooks; only cleanup drops them) so a
            // device that dies the moment we get it is still routed to recovery.
            arm()
            await renderer.initialize(initOptions)
            arm()
        },
        registerNode: (id: string, fragment: unknown, parentId: string | null, metadata: unknown, ...rest: never[]): void => {
            if (fragment !== null) {
                const definition = rest[1]
                if (definition !== undefined) guard.noteLayer(id, definition)
                if (guard.isDisabled(id)) metadata = {...((metadata as object | null) ?? {}), visible: false}
            }
            renderer.registerNode(id, fragment, parentId, metadata, ...rest)
        },
        updateNodeMetadata: (id: string, metadata: never): void => {
            const md = metadata as Record<string, unknown> | null
            if (md && 'visible' in md && guard.isDisabled(id)) {
                renderer.updateNodeMetadata(id, {...md, visible: false} as never)
                return
            }
            renderer.updateNodeMetadata(id, metadata)
        },
        disposeRecovery: (): void => {
            disposed = true
            pendingLoss = null
            try {
                renderer.setOnDeviceLost(null)
                renderer.setOnCompositionBroken?.(null)
            } catch {
                /* ignore */
            }
        },
        crashGuard: guard,
    }) as R & GpuRecoveryControls
    return wrapped
}
