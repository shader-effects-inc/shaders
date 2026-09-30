/**
 * The runtime registry of user-defined shaders: how a name in preset JSON finds a `defineShader` result.
 *
 * The library's own shaders are known at build time. Yours are registered at runtime, so
 * anything that looks a shader up by name (`createShader` with `type: 'Halo'`, code export,
 * an editor) can find them. `<CustomShader src={…}>` registers its definition on mount, so
 * you only call `registerShader` yourself when rendering from preset JSON without passing
 * `components`.
 */
// Maintainer notes: the build-time registry is `shaderRegistry.ts`. Registration is
// idempotent for the same definition object; a DIFFERENT definition under a name already
// taken replaces it (hot reload, live editing) and notifies subscribers.
import type {GpuShaderDefinition} from './gpu/contract'

const registry = new Map<string, GpuShaderDefinition>()
const listeners = new Set<(name: string, definition: GpuShaderDefinition | null) => void>()

/**
 * Make a user-defined shader findable by its `name`, so preset JSON can reference it as `type: 'Halo'`.
 *
 * Returns the definition, so it can wrap `defineShader` inline. Registering the same object
 * again does nothing. A different definition under a taken name replaces it (live editing,
 * hot reload) and notifies `onShaderRegistered` listeners. `<CustomShader>` calls this for
 * you on mount; `createShader(canvas, preset, {components: [Halo]})` is the other way to
 * make a name resolvable without registering globally.
 *
 * @example
 * ```ts
 * import {createShader} from 'shaders/js'
 * import {Halo} from './halo'
 *
 * registerShader(Halo)
 * const shader = await createShader(canvas, {components: [{type: 'Halo', props: {bands: 6}}]})
 * ```
 * @see defineShader, unregisterShader, getRegisteredShader, onShaderRegistered
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

/**
 * Forget a user-defined shader by name.
 *
 * Does nothing for a name that was never registered. Listeners from `onShaderRegistered` are
 * told with a `null` definition.
 *
 * @example
 * ```ts
 * unregisterShader('Halo')
 * ```
 * @see registerShader
 */
export function unregisterShader(name: string): void {
    if (!registry.delete(name)) return
    for (const listener of listeners) listener(name, null)
}

/**
 * Look up a user-defined shader by name.
 *
 * Returns `undefined` for unknown names. The library's own shaders are not in this registry.
 *
 * @example
 * ```ts
 * const definition = getRegisteredShader('Halo')
 * ```
 * @see getRegisteredShaders, registerShader
 */
export function getRegisteredShader(name: string): GpuShaderDefinition | undefined {
    return registry.get(name)
}

/**
 * Every user-defined shader currently registered, oldest first.
 *
 * @example
 * ```ts
 * const names = getRegisteredShaders().map((d) => d.name)
 * ```
 * @see getRegisteredShader, onShaderRegistered
 */
export function getRegisteredShaders(): GpuShaderDefinition[] {
    return [...registry.values()]
}

/**
 * Be told when a user-defined shader is registered, replaced or removed.
 *
 * The listener gets the name and the new definition, or `null` on removal. Returns a function
 * that unsubscribes. Useful for a component picker or a live editor that lists custom shaders.
 *
 * @example
 * ```ts
 * const stop = onShaderRegistered((name, definition) => refreshPicker())
 * ```
 * @see registerShader, getRegisteredShaders
 */
export function onShaderRegistered(listener: (name: string, definition: GpuShaderDefinition | null) => void): () => void {
    listeners.add(listener)
    return () => {
        listeners.delete(listener)
    }
}
