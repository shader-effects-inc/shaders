/**
 * Shared compile: renderers on one device reuse each other's render pipelines when their
 * compositions have the same structure.
 *
 * A composition's structural hash (pipelineCache.ts) covers everything that changes its
 * generated code; colors, speeds and slider values live in its uniform buffer and aren't part
 * of it. So two renderers whose compositions share a hash (a duplicated shader, color
 * variations made by duplicating) generate identical WGSL, and the second one can draw with the
 * first one's pipelines, binding its OWN buffers and textures. That skips the main-thread WGSL
 * generation and the GPU compile for every pass.
 *
 * What makes it work: a pipeline only knows its bind-group LAYOUTS, never a buffer. The second
 * renderer creates its bind groups (uniforms, per-pass textures, external textures) from the
 * first one's layout objects (TypeGPU matches bind groups to a pipeline by layout identity), so
 * each renderer still reads its own data.
 *
 * What isn't shared: the uniform buffer, RTT/media textures, compute pipelines and their state,
 * and every callback; each renderer composes its own IR for those. A pipeline whose code binds a
 * resource directly (TypeGPU's automatic "catchall" bind group: a fixed buffer or texture baked
 * into the pipeline) would make the second renderer read the first one's resource, so such a
 * pipeline is never shared.
 *
 * Entries are kept per `TgpuRoot` (all editor tiles share one), reference-counted by the
 * compositions using them. An entry nobody uses stays in a small LRU, so toggling a layer off and
 * on stays instant, then is dropped. Shared pipelines are never modified: a structural edit gives
 * the edited renderer a new hash and a new entry, and the others keep theirs. GPU pipelines have
 * no destroy; dropping the entry lets them be collected. A lost device brings a new root, and
 * with it an empty cache.
 */
import type {TgpuBindGroupLayout, TgpuRenderPipeline, TgpuRoot} from 'typegpu'

/** One shared pass: its pipeline and the layouts a borrower builds its bind groups from. */
export interface SharedPass {
    pipeline: TgpuRenderPipeline
    /** The texture keys the pass samples, in binding order (a borrower must match exactly). */
    reads: readonly string[]
    externalReads: readonly string[]
    textureLayout?: unknown
    externalLayout?: unknown
}

export interface SharedEntry {
    key: string
    /** The group-0 layout the pipelines were built with; a borrower binds its buffer through it. */
    uniformLayout: TgpuBindGroupLayout
    /** RTT passes in order, then the final pass. */
    passes: SharedPass[]
    /**
     * Resolves once the owner's pipelines have compiled: true when they can be shared, false
     * when they failed or bind a fixed resource. Borrowers that joined while it compiled wait
     * on it and rebuild on their own when it says false.
     */
    ready: Promise<boolean>
    /** The settled verdict of `ready` (undefined while compiling). */
    shareable?: boolean
    refs: number
    lastUsed: number
}

/** Unused entries kept per root before they're dropped. */
const IDLE_ENTRIES = 8

const byRoot = new WeakMap<object, Map<string, SharedEntry>>()
let clock = 0

function entriesOf(root: TgpuRoot): Map<string, SharedEntry> {
    let map = byRoot.get(root)
    if (!map) {
        map = new Map()
        byRoot.set(root, map)
    }
    return map
}

/** A usable entry for `key`, with a reference taken, or null (none yet, or not shareable). */
export function acquireShared(root: TgpuRoot, key: string): SharedEntry | null {
    const entry = byRoot.get(root)?.get(key)
    if (!entry || entry.shareable === false) return null
    entry.refs++
    entry.lastUsed = ++clock
    return entry
}

/**
 * Offer a just-built composition's pipelines for others to share. `compiled` is the owner's own
 * background compile. Returns the entry with the owner's reference taken, or null if an entry
 * already exists for the key.
 */
export function registerShared(
    root: TgpuRoot,
    key: string,
    uniformLayout: TgpuBindGroupLayout,
    passes: SharedPass[],
    compiled: Promise<void> | null,
): SharedEntry | null {
    const map = entriesOf(root)
    if (map.has(key)) return null
    const entry: SharedEntry = {key, uniformLayout, passes, ready: Promise.resolve(false), refs: 1, lastUsed: ++clock}
    entry.ready = (compiled ?? Promise.resolve()).then(
        () => passes.every((p) => !hasCatchall(p.pipeline)),
        () => false,
    ).then((ok) => {
        entry.shareable = ok
        // A failed or unshareable entry is useless to the next builder: let it build its own
        if (!ok && map.get(key) === entry) map.delete(key)
        return ok
    })
    map.set(key, entry)
    return entry
}

/** Drop a composition's reference. Unused entries past the idle limit are forgotten, oldest first. */
export function releaseShared(root: TgpuRoot, entry: SharedEntry): void {
    entry.refs = Math.max(0, entry.refs - 1)
    entry.lastUsed = ++clock
    const map = byRoot.get(root)
    if (!map) return
    const idle = [...map.values()].filter((e) => e.refs === 0).sort((a, b) => a.lastUsed - b.lastUsed)
    for (let i = 0; i < idle.length - IDLE_ENTRIES; i++) map.delete(idle[i].key)
}

/** Entries cached for `root` (tests and diagnostics). */
export function sharedEntryCount(root: TgpuRoot): number {
    return byRoot.get(root)?.size ?? 0
}

/**
 * Whether a COMPILED pipeline binds a resource directly (TypeGPU's automatic catchall group).
 * TypeGPU doesn't export its internal symbol, so it's found by description. Unknown means
 * "don't share".
 */
function hasCatchall(pipeline: TgpuRenderPipeline): boolean {
    try {
        const internal = Object.getOwnPropertySymbols(pipeline).find((s) => s.description?.endsWith(':$internal'))
        if (!internal) return true
        const core = (pipeline as unknown as Record<symbol, {core?: {unwrap?: () => {catchall?: unknown}}}>)[internal]?.core
        if (typeof core?.unwrap !== 'function') return true
        // Safe: the pipeline has compiled, so this returns the memo without building anything
        return core.unwrap().catchall !== undefined && core.unwrap().catchall !== null
    } catch {
        return true
    }
}
