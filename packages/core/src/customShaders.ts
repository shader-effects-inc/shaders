/**
 * Runtime shader registry — user-defined components.
 *
 * The build-time registry (`shaderRegistry.ts`) lists the library's shaders. Components
 * authored in an application (`defineShader(...)`) live here instead, registered at runtime
 * so every name-keyed surface can find them: `createShader` / `createRendererFromJSON`
 * presets (`type: 'Halo'`), code export, and hosts like the design editor.
 *
 * The framework `<CustomShader src={…}>` components register their definition on mount, so
 * an app never has to call {@link registerShader} itself unless it renders from preset JSON.
 * Registration is idempotent for the same definition object; a DIFFERENT definition under a
 * name already taken replaces it (hot reload, live editing) and notifies subscribers.
 */
import type {GpuShaderDefinition} from './gpu/contract'

const registry = new Map<string, GpuShaderDefinition>()
const listeners = new Set<(name: string, definition: GpuShaderDefinition | null) => void>()

/**
 * Register a user-defined shader under its `name`. Returns the definition so it can be used
 * inline: `export const Halo = registerShader(defineShader({...}))`.
 */
export function registerShader<T extends GpuShaderDefinition<any>>(definition: T): T {
    const name = definition?.name
    if (typeof name !== 'string' || name.trim() === '') {
        throw new Error('[Shaders] registerShader: the definition needs a non-empty `name`')
    }
    if (typeof definition.fragment !== 'function') {
        throw new Error(`[Shaders] registerShader("${name}"): not a shader definition (missing fragment) — did you pass the result of defineShader()?`)
    }
    const previous = registry.get(name)
    if (previous === definition) return definition
    registry.set(name, definition)
    for (const listener of listeners) listener(name, definition)
    return definition
}

/** Remove a user-defined shader by name. No-op for names that were never registered. */
export function unregisterShader(name: string): void {
    if (!registry.delete(name)) return
    for (const listener of listeners) listener(name, null)
}

/** A user-defined shader by name, or `undefined`. Library shaders are NOT in this registry. */
export function getRegisteredShader(name: string): GpuShaderDefinition | undefined {
    return registry.get(name)
}

/** Every user-defined shader, in registration order. */
export function getRegisteredShaders(): GpuShaderDefinition[] {
    return [...registry.values()]
}

/**
 * Subscribe to registrations and removals (a component picker, a live editor). Returns the
 * unsubscribe function.
 */
export function onShaderRegistered(listener: (name: string, definition: GpuShaderDefinition | null) => void): () => void {
    listeners.add(listener)
    return () => {
        listeners.delete(listener)
    }
}
