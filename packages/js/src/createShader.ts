import { shaderRendererGPU, createGpuUniformsMap, rootPassthrough, resolveBoundingBox, debugError } from 'shaders-core'
import { getAllShaders } from 'shaders-core/registry'
import { getRegisteredShader } from 'shaders-core'
import { isExternalUser, startTelemetry } from 'shaders-core/telemetry'
import type { GpuShaderDefinition, GpuUniformsMap, NodeMetadata, PropDriver, GpuFailureReason } from 'shaders-core'
import type { BlendMode } from 'shaders-core'
import type { PresetConfig, ComponentConfig } from 'shaders-core'
import type { ShaderOptions, ShaderInstance } from './types'

declare const __SHADERS_VERSION__: string

interface NodeEntry {
  componentDef: GpuShaderDefinition
  uniforms: GpuUniformsMap
  nodeId: string
  component: ComponentConfig
  currentMaps: Record<string, PropDriver>
}

function isPropDriver(value: unknown): value is PropDriver {
  if (typeof value !== 'object' || value === null || !('type' in value)) return false
  const maybe = value as { type?: unknown }
  return maybe.type === 'map' || maybe.type === 'mouse' ||
    maybe.type === 'mouse-position' || maybe.type === 'auto-animate'
}

const METADATA_PROPS = new Set(['opacity', 'blendMode', 'visible', 'transform', 'boundingBox', 'maskSource', 'maskType', 'flow', 'absolute'])

// Renderer-backed state for a single live build. On device loss we throw the
// whole thing away and assemble a new one — the public ShaderInstance methods
// dereference `current` on every call, so callers keep their original
// references and don't have to care that we swapped renderers underneath.
interface LiveState {
  renderer: ReturnType<typeof shaderRendererGPU>
  nodeEntryMap: Map<string, NodeEntry>
  telemetryCollector: { start: () => void; stop: () => void } | null
  telemetryStartTimeout: number | null
}

/**
 * Create a shader instance from preset JSON.
 * Auto-starts animation loop — no manual render needed.
 *
 * @example
 * ```ts
 * const shader = await createShader(canvas, {
 *   components: [
 *     { type: 'Circle', id: 'c1', props: { color: '#ff0000', radius: 0.5 } }
 *   ]
 * })
 *
 * // Update at runtime
 * shader.update('c1', { radius: 0.8 })
 *
 * // Cleanup
 * shader.destroy()
 * ```
 */
export async function createShader(
  canvas: HTMLCanvasElement,
  preset: PresetConfig,
  options?: ShaderOptions
): Promise<ShaderInstance> {
  // Component registry is constant for the lifetime of the instance — no need
  // to rebuild on recovery. Deprecated (renamed) shader names alias the same
  // definition so old `type` strings keep working.
  const componentRegistry = new Map<string, GpuShaderDefinition>()
  getAllShaders().forEach(shader => {
    componentRegistry.set(shader.definition.name, shader.definition as GpuShaderDefinition)
    for (const oldName of (shader.definition as { deprecatedNames?: string[] }).deprecatedNames ?? []) {
      componentRegistry.set(oldName, shader.definition as GpuShaderDefinition)
    }
  })
  // User-defined components (`defineShader` results) this preset references by `type`.
  // Anything registered globally via `registerShader` is found at lookup time as a fallback.
  for (const custom of options?.components ?? []) {
    componentRegistry.set(custom.name, custom)
  }

  // Ensure the canvas has explicit CSS dimensions before initializing.
  //
  // When the canvas has no CSS sizing, its layout size is driven by its width/height
  // attributes. Three.js's setSize() changes those attributes (to apply devicePixelRatio),
  // which changes the layout size, which fires the ResizeObserver again — an infinite loop.
  // Locking the CSS dimensions breaks that cycle while leaving the pixel buffer free to
  // be scaled correctly by Three.js.
  //
  // We only set a dimension if the inline style for that dimension is absent.
  if (!canvas.style.width || !canvas.style.height) {
    const rect = canvas.getBoundingClientRect()
    const w = rect.width > 0 ? rect.width : canvas.width
    const h = rect.height > 0 ? rect.height : canvas.height
    if (w > 0 && !canvas.style.width) canvas.style.width = `${w}px`
    if (h > 0 && !canvas.style.height) canvas.style.height = `${h}px`
  }

  // Shallow-clone the preset into a mutable shadow we own. `update()` writes
  // runtime changes back into this shadow's component.props so a device-loss
  // rebuild reads the *current* state (e.g. a slider value the user dragged
  // to 0.8) instead of the value the partner originally passed in. The
  // partner's preset object is left untouched — they can keep their own
  // reference frozen / shared without surprise mutations.
  function cloneComponent(c: ComponentConfig): ComponentConfig {
    return {
      ...c,
      props: c.props ? { ...c.props } : undefined,
      children: c.children?.map(cloneComponent),
    }
  }
  const livePreset: PresetConfig = {
    ...preset,
    components: preset.components.map(cloneComponent),
  }

  let current: LiveState | null = null
  let destroyed = false
  let rebuilding = false

  /**
   * Device-loss rebuilds are bounded. A tab whose GPU was evicted recovers on the first
   * attempt; a machine whose driver dies the moment we touch it would otherwise
   * lose-rebuild-lose forever, pinning a core and freezing the page — exactly the failure
   * mode reported on partially-implemented WebGPU backends. After the budget is spent we
   * stay down (transparent canvas) and report `'unrecoverable'` once.
   *
   * The budget covers a BURST, not the page's whole lifetime: a loss that arrives long
   * after the previous one is a fresh incident (a browser reclaiming a backgrounded tab's
   * device, say) and gets a full budget again. Only losses landing back-to-back — the tight
   * loop that actually freezes a page — accumulate.
   */
  const MAX_DEVICE_LOSS_REBUILDS = 3
  const DEVICE_LOSS_BURST_WINDOW_MS = 60_000
  let rebuildAttempts = 0
  let lastDeviceLossAt = 0
  /**
   * A permanent stop: either the GPU is unusable or the rebuild budget is spent. Holds the
   * reason so `getFailureReason()` can report it, and doubles as the "have we stopped?" flag.
   */
  let unavailable: string | null = null

  /** Hand a reason to the caller's onError without ever letting it break us. */
  function notify(reason: string): void {
    try { options?.onError?.(reason) } catch (e) {
      debugError('[createShader] onError callback threw:', e)
    }
  }

  /**
   * Latch a terminal condition: stop for good and tell the caller exactly once.
   *
   * The guard comes FIRST so the stored reason and the delivered reason can never diverge.
   * Several paths can fire in sequence — the rebuild budget reporting `'unrecoverable'`, then
   * the renderer's own `onUnavailable` arriving with `'device-lost'` — and assigning before
   * the guard would leave `getFailureReason()` naming a reason the caller was never given.
   */
  let reportedTerminal = false
  function goDown(reason: string): void {
    if (reportedTerminal) return
    reportedTerminal = true
    unavailable = reason
    notify(reason)
  }

  /**
   * Recursively register a component and its children against a freshly-built
   * renderer + node map. Pure function over the inputs — no shared state with
   * other builds — so device-loss recovery can run it from scratch safely.
   */
  function registerComponent(
    renderer: ReturnType<typeof shaderRendererGPU>,
    nodeEntryMap: Map<string, NodeEntry>,
    component: ComponentConfig,
    parentId: string,
    renderOrder: number
  ): void {
    const componentDef = componentRegistry.get(component.type) ?? getRegisteredShader(component.type)
    if (!componentDef) {
      console.warn(`[createShader] Unknown component type: ${component.type}`)
      return
    }

    const nodeId = component.id || `${component.type}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`

    // Separate PropDriver (map) values from static prop values.
    // createGpuUniformsMap expects static values only — PropDrivers go to metadata.maps
    // so the renderer can drive them dynamically at render time.
    const mapsFromProps: Record<string, PropDriver> = {}
    for (const [key, val] of Object.entries(component.props || {})) {
      if (!METADATA_PROPS.has(key) && Object.prototype.hasOwnProperty.call(componentDef.props, key) && isPropDriver(val)) {
        mapsFromProps[key] = val
      }
    }

    // Build a complete props object (user values merged over defaults) and create uniforms.
    // PropDriver values are replaced with the prop's static default so createGpuUniformsMap
    // receives valid uniform values — the actual dynamic driving happens via metadata.maps.
    const completeProps = Object.fromEntries(
      Object.entries(componentDef.props).map(([key, propConfig]) => {
        const val = component.props?.[key] !== undefined ? component.props[key] : propConfig.default
        return [key, isPropDriver(val) ? propConfig.default : val]
      })
    )
    const uniforms = createGpuUniformsMap(componentDef, completeProps, nodeId)

    const metadata: NodeMetadata = {
      blendMode: (component.props?.blendMode || 'normal') as BlendMode,
      opacity: component.props?.opacity,
      visible: component.props?.visible,
      renderOrder,
      id: component.id,
      maps: Object.keys(mapsFromProps).length > 0 ? { ...mapsFromProps } : undefined,
      mask: component.props?.maskSource ? {
        source: component.props.maskSource,
        type: component.props.maskType || 'alpha'
      } : undefined,
      // Merge partial transform configs with defaults so the renderer never receives
      // undefined field values (which would produce null-typed TSL uniforms)
      transform: component.props?.transform ? {
        offsetX: 0, offsetY: 0, rotation: 0, scale: 1,
        anchorX: 0.5, anchorY: 0.5, edges: 'transparent' as const,
        ...component.props.transform
      } : undefined,
      // Same merge for bounding boxes — the renderer dereferences every dimension
      boundingBox: resolveBoundingBox(component.props?.boundingBox),
      // Group flow layout + per-child escape hatch — the renderer's layout pass positions
      // in-flow children each resize, so a JS-hosted embed stays responsive, not baked.
      flow: component.props?.flow,
      absolute: component.props?.absolute
    }

    renderer.registerNode(
      nodeId,
      componentDef.fragment,
      parentId,
      metadata,
      uniforms,
      componentDef
    )

    nodeEntryMap.set(nodeId, {
      componentDef,
      uniforms,
      nodeId,
      component,
      currentMaps: { ...mapsFromProps }
    })

    component.children?.forEach((child, index) => {
      registerComponent(renderer, nodeEntryMap, child, nodeId, index)
    })
  }

  async function build(): Promise<LiveState> {
    const renderer = shaderRendererGPU()
    const nodeEntryMap = new Map<string, NodeEntry>()

    if (options?.onReady) {
      renderer.setOnReady(options.onReady)
    }

    // Subscribe BEFORE initialize so a start-up failure (no WebGPU, no adapter, device
    // refused) is caught on the very first attempt. The renderer has already released
    // everything by the time this fires — a canvas that never drew is simply transparent —
    // so all we do is latch it so nothing retries, and tell the caller once. Nothing is
    // logged: an embedded background effect that can't run must not pollute the host page's
    // console.
    renderer.setOnUnavailable((reason: GpuFailureReason) => {
      if (destroyed) return
      goDown(reason)
    })

    // Initialize renderer — observe the canvas itself (not its parent).
    // Framework packages control the canvas CSS (100%/100% inside a wrapper) so observing
    // the parent is reliable there. Here the user owns the canvas and the parent may be
    // arbitrarily larger, so we observe the canvas directly for correct dimensions.
    await renderer.initialize({
      canvas,
      resizeTarget: canvas,
      enablePerformanceTracking: options?.enablePerformanceTracking || false,
      colorSpace: options?.colorSpace,
      toneMapping: options?.toneMapping,
      observeElement: options?.observeElement,
      gpu: options?.gpu
    })

    // Build the state holder up front (with the still-empty nodeEntryMap)
    // and wire device-loss recovery *before* any node registration / graph
    // construction. Without this, a device.lost microtask that resolves
    // between initialize() returning and our wiring call lands while
    // onDeviceLostCallback is still null — the renderer sets its
    // `deviceLostFired` latch and the event is dropped forever for this
    // instance. The state holder gets populated as registerComponent runs
    // below (state.nodeEntryMap is the same Map by reference).
    const state: LiveState = {
      renderer,
      nodeEntryMap,
      telemetryCollector: null,
      telemetryStartTimeout: null,
    }

    // Wire device-loss recovery. The renderer detects the loss (WebGPU
    // device.lost / WebGL contextlost) and calls us; we tear down + rebuild.
    // Guarded against re-entry so a flurry of loss events (or a loss during
    // the rebuild itself) doesn't kick off multiple parallel rebuilds.
    renderer.setOnDeviceLost((reason: string) => {
      // Stale-callback guard: skip only when there *is* a live state and it
      // isn't this one. `current === null` is the in-rebuild gap; we don't
      // need to skip there because `rebuilding` already covers re-entry.
      if (destroyed || rebuilding || unavailable) return
      if (current !== null && current !== state) return
      rebuilding = true
      notify(reason)

      // Spend one unit of the rebuild budget. Past it, tear down and stay down: a device
      // that dies immediately on every attempt would otherwise loop forever.
      const now = Date.now()
      if (now - lastDeviceLossAt > DEVICE_LOSS_BURST_WINDOW_MS) rebuildAttempts = 0
      lastDeviceLossAt = now
      if (++rebuildAttempts > MAX_DEVICE_LOSS_REBUILDS) {
        teardownState(state)
        current = null
        rebuilding = false
        goDown('unrecoverable')
        return
      }

      void (async () => {
        try {
          teardownState(state)
          // Null out before the await so pause/update/resume calls during the
          // rebuild window become safe no-ops instead of poking a dead
          // renderer. Once the new build resolves we swap to it.
          current = null
          if (destroyed) return
          const next = await build()
          if (destroyed) {
            teardownState(next)
            return
          }
          current = next
        } catch (rebuildErr) {
          debugError('[createShader] rebuild after device loss failed:', rebuildErr)
          goDown('rebuild_failed')
        } finally {
          rebuilding = false
        }
      })()
    })

    // Graph construction runs AFTER setOnDeviceLost so any device.lost event
    // that races initialize's return is still caught (see the rationale on
    // state construction above).
    // The GPU renderer requires a component definition on every node — rootPassthrough supplies
    // the shared "composite children / transparent when empty" root (replacing the v1 vec4 fn).
    const rootId = 'shader-root'
    renderer.registerNode(
      rootId,
      rootPassthrough.fragment,
      null,
      null,
      {},
      rootPassthrough
    )

    livePreset.components.forEach((component, index) => {
      registerComponent(renderer, nodeEntryMap, component, rootId, index)
    })

    // Telemetry — bound to *this* renderer instance. A rebuild gets a fresh
    // collector against the new renderer so we don't keep polling a dead one.
    if (isExternalUser()) {
      // Bound the "has it started drawing yet?" poll. On a browser that can't run WebGPU
      // fps never leaves 0, and an unbounded 500ms recursion would tick for the life of the
      // page against a renderer that will never produce a frame.
      const MAX_TELEMETRY_POLLS = 40 // 40 × 500ms = 20s
      let telemetryPolls = 0
      const checkRendering = () => {
        // If the instance was destroyed (or this state was superseded by a
        // rebuild) before telemetry could start, bail rather than booting a
        // collector against a stale renderer.
        if (destroyed || unavailable || current !== state || ++telemetryPolls > MAX_TELEMETRY_POLLS) {
          state.telemetryStartTimeout = null
          return
        }
        const stats = renderer.getPerformanceStats()
        if (stats.fps > 0) {
          const version = typeof __SHADERS_VERSION__ !== 'undefined' ? __SHADERS_VERSION__ : 'unknown'
          state.telemetryCollector = startTelemetry(
            renderer,
            version,
            options?.disableTelemetry || false,
            false
          )
          if (state.telemetryCollector) {
            state.telemetryCollector.start()
          }
          state.telemetryStartTimeout = null
        } else {
          state.telemetryStartTimeout = setTimeout(checkRendering, 500) as unknown as number
        }
      }
      state.telemetryStartTimeout = setTimeout(checkRendering, 500) as unknown as number
    }

    return state
  }

  function teardownState(state: LiveState): void {
    if (state.telemetryCollector) {
      try { state.telemetryCollector.stop() } catch { /* noop */ }
      state.telemetryCollector = null
    }
    if (state.telemetryStartTimeout !== null) {
      clearTimeout(state.telemetryStartTimeout)
      state.telemetryStartTimeout = null
    }
    try { state.renderer.cleanup() } catch { /* noop */ }
  }

  current = await build()

  function update(componentId: string, props: Record<string, any>): void {
    if (!current) return
    const { renderer, nodeEntryMap } = current
    const entry = nodeEntryMap.get(componentId)
    if (!entry) {
      console.warn(`[createShader] Component ID not found: ${componentId}`)
      return
    }

    // Mirror runtime updates back into the live preset's component.props so
    // a device-loss rebuild reads the *current* state instead of replaying
    // the original preset. `livePreset` is our own clone (see top of
    // createShader) so this mutation never touches the partner's input.
    if (!entry.component.props) entry.component.props = {}
    const live = entry.component.props as Record<string, unknown>

    for (const [key, value] of Object.entries(props)) {
      if (METADATA_PROPS.has(key)) {
        if (key === 'maskSource') {
          renderer.updateNodeMetadata(entry.nodeId, {
            mask: value ? {
              source: value,
              type: props.maskType || live.maskType || 'alpha'
            } : undefined
          })
          live.maskSource = value
          // maskType in the same call must be persisted too; the next branch
          // continues and never gets a chance to.
          if ('maskType' in props) live.maskType = props.maskType
        } else if (key === 'maskType') {
          // maskType is handled together with maskSource above
          continue
        } else if (key === 'transform') {
          renderer.updateNodeMetadata(entry.nodeId, { transform: value })
          live.transform = value
        } else if (key === 'boundingBox') {
          renderer.updateNodeMetadata(entry.nodeId, { boundingBox: resolveBoundingBox(value) })
          live.boundingBox = value
        } else {
          renderer.updateNodeMetadata(entry.nodeId, { [key]: value })
          live[key] = value
        }
      } else {
        if (isPropDriver(value)) {
          if (!Object.prototype.hasOwnProperty.call(entry.componentDef.props, key)) {
            console.warn(`[createShader] Ignoring PropDriver for unknown prop "${key}" on ${entry.component.type}`)
          } else {
            entry.currentMaps[key] = value
            renderer.updateNodeMetadata(entry.nodeId, { maps: { ...entry.currentMaps } })
            // PropDrivers are stored as the prop value itself; registerComponent
            // detects them via isPropDriver() on rebuild and re-routes them to
            // metadata.maps the same way.
            live[key] = value
          }
        } else {
          if (key in entry.currentMaps) {
            delete entry.currentMaps[key]
            renderer.updateNodeMetadata(entry.nodeId, {
              maps: Object.keys(entry.currentMaps).length > 0 ? { ...entry.currentMaps } : undefined
            })
          }
          if (Object.prototype.hasOwnProperty.call(entry.componentDef.props, key)) {
            renderer.updateUniformValue(entry.nodeId, key, value)
            live[key] = value
          }
        }
      }
    }
  }

  function resize(width?: number, height?: number): void {
    const rect = canvas.getBoundingClientRect()
    const w = width ?? (rect.width > 0 ? rect.width : canvas.width)
    const h = height ?? (rect.height > 0 ? rect.height : canvas.height)
    if (w > 0 && h > 0) {
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      if (current) current.renderer.resize(w, h)
    }
  }

  function destroy(): void {
    destroyed = true
    if (current) {
      teardownState(current)
      current = null
    }
  }

  function pause(): void {
    if (current) current.renderer.stopAnimation()
  }

  function resume(): void {
    if (current) current.renderer.startAnimation()
  }

  return { update, resize, pause, resume, destroy, getFailureReason: () => unavailable }
}
