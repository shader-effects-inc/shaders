import { shaderRendererGPU } from './gpu/index'
import { rootPassthrough } from './gpu/porters'
import { createGpuUniformsMap } from './gpu/uniformBridge'
import { debugWarn } from './gpu/support'
import { getAllShaders } from './shaderRegistry'
import { getRegisteredShader } from './customShaders'
import type { GpuShaderDefinition } from './gpu/contract'
import type { NodeMetadata } from './types'
import type { BlendMode } from './types'

/**
 * Component configuration from JSON preset
 */
export interface ComponentConfig {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ComponentConfig[]
}

/**
 * Preset configuration structure
 */
export interface PresetConfig {
  components: ComponentConfig[]
  structureVersion?: number
}

/**
 * Options for creating a preset renderer
 */
export interface PresetRendererOptions {
  enablePerformanceTracking?: boolean
  /** User-defined components (`defineShader` results) this preset may reference by `type`. */
  components?: GpuShaderDefinition[]
  // Adopt an existing device instead of requesting one (shared-device integrations).
  gpu?: {
    device: GPUDevice
    adapter: GPUAdapter
  }
  context?: WebGLRenderingContext | WebGL2RenderingContext
  // Force full frame rate even when canvas is offscreen (for partner integrations)
  forceFullFrameRate?: boolean
}

/**
 * Options for {@link createRendererFromJSON}'s `renderFrame()`.
 */
export interface RenderFrameOptions {
  /**
   * Await `queue.onSubmittedWorkDone()` before resolving (default `true`). See `renderFrame`
   * for when it is safe to skip.
   */
  waitForGpu?: boolean
  /**
   * Advance the shader clocks by exactly this many seconds instead of the wall-clock time
   * since the last frame. Switches this renderer to frame-locked rendering for deterministic
   * offline output; `0` repaints without advancing time. See `renderFrame`.
   */
  deltaSeconds?: number
}

/**
 * GPU context information for sharing with other renderers
 */
export interface GPUContext {
  type: 'webgpu' | 'webgl'
  device?: GPUDevice
  adapter?: GPUAdapter
  context?: WebGLRenderingContext | WebGL2RenderingContext
  canvas: HTMLCanvasElement
}

/**
 * Create a renderer from preset JSON configuration
 * Provides manual render control for integration with third-party tools
 *
 * @example
 * ```typescript
 * const preset = {
 *   components: [
 *     {
 *       type: 'RadialGradient',
 *       props: {
 *         color1: '#ff0000',
 *         color2: '#0000ff'
 *       }
 *     }
 *   ]
 * }
 *
 * const renderer = createRendererFromJSON(preset)
 * await renderer.initialize(canvas)
 *
 * // Manual render loop
 * function animate() {
 *   await renderer.renderFrame()
 *   requestAnimationFrame(animate)
 * }
 * animate()
 * ```
 */
export function createRendererFromJSON(
  preset: PresetConfig,
  options?: PresetRendererOptions
) {
  const coreRenderer = shaderRendererGPU()
  const componentRegistry = new Map<string, GpuShaderDefinition>()
  const nodeIdMap = new Map<string, string>()
  // Captured at initialize() — getGPUContext() reports it (the GPU renderer's getInternalRenderer()
  // returns {device, adapter, root}, not the canvas, so we hold onto it here).
  let presetCanvas: HTMLCanvasElement | null = null
  // Frame-locked rendering (renderFrame({deltaSeconds})): once set, every clock in the core
  // renderer is the exact sum of the deltas passed in, and only this path may draw.
  let frameLocked = false

  // Build component registry from core. Every shader is a GpuShaderDefinition.
  const allShaders = getAllShaders()
  allShaders.forEach(shader => {
    componentRegistry.set(shader.definition.name, shader.definition as GpuShaderDefinition)
  })
  // User-defined components passed for this renderer (a `defineShader` result). Anything
  // registered globally via `registerShader` is found at lookup time as a fallback.
  for (const custom of options?.components ?? []) {
    componentRegistry.set(custom.name, custom)
  }

  /**
   * Initialize renderer on canvas
   * Does NOT start animation loop - partner controls rendering
   */
  async function initialize(canvas: HTMLCanvasElement): Promise<void> {
    presetCanvas = canvas
    await coreRenderer.initialize({
      canvas,
      enablePerformanceTracking: options?.enablePerformanceTracking || false,
      gpu: options?.gpu,
      forceFullFrameRate: options?.forceFullFrameRate || false
    })

    // Register nodes from preset
    registerPresetNodes()

    // Stop automatic animation - partner controls rendering
    coreRenderer.stopAnimation()
  }

  /**
   * Manually render a single frame. Called by the host in its own render loop.
   *
   * **Real time (default).** Each frame advances the shader clocks by the wall-clock time since
   * the previous one, so a `requestAnimationFrame` loop plays back at natural speed.
   *
   * **Frame-locked (`deltaSeconds`).** The clocks advance by exactly `deltaSeconds`, whatever
   * the wall clock says. This is the path for rendering to video (Remotion, Puppeteer capture,
   * any offline pipeline): a frame becomes a pure function of the deltas rendered before it,
   * so the same frame rendered in two tabs is pixel-identical and stepping to frame N matches
   * jumping straight there. The first locked call hands the clock to the host for the life of
   * this renderer: the clocks restart at 0, the renderer's own on-demand repaints (after
   * `updatePreset`, a resize, a pointer move, an async asset load) are dropped from then on,
   * and a plain `renderFrame()` throws, since each would leak a wall-clock delta into the
   * clocks. Call `renderFrame({deltaSeconds: 0})` to repaint without advancing time.
   *
   * Simulation components (CursorRipples, DataMosh, Voxels and other ping-pong compute
   * effects) depend on the step sequence by design: a jump of 1s is not the same as sixty
   * steps of 1/60s. Step those at the output frame rate.
   *
   * **GPU fence.** By default this waits for the GPU to finish the frame before resolving, so
   * the canvas is guaranteed complete when you composite or read it. That wait is ~1ms on
   * Chromium but **~104ms on Firefox 152**, which caps a per-frame render loop at roughly
   * 10fps there. Pass `{waitForGpu: false}` if your loop reads the canvas via `toBlob()` /
   * `convertToBlob()` (that readback synchronises with the GPU itself), or if you sample the
   * canvas as a texture on the SAME device, where queue ordering already covers you. Keep the
   * default if you hand the canvas to a different device, draw it into a 2D canvas with
   * `drawImage`, or otherwise assume the frame has landed.
   */
  async function renderFrame(options?: RenderFrameOptions): Promise<void> {
    const delta = options?.deltaSeconds
    if (delta !== undefined && !Number.isFinite(delta)) {
      throw new TypeError(`[presetRenderer] renderFrame: deltaSeconds must be a finite number, got ${delta}`)
    }
    // Check if renderer is ready before rendering
    if (!coreRenderer.isInitialized()) {
      debugWarn('[presetRenderer] Renderer not yet initialized, skipping frame')
      return
    }
    if (delta === undefined) {
      if (frameLocked) {
        throw new Error(
          '[presetRenderer] renderFrame: this renderer is frame-locked (a previous call passed deltaSeconds); ' +
            'pass deltaSeconds on every call, or {deltaSeconds: 0} to repaint without advancing time'
        )
      }
      await coreRenderer.renderAndWait(options)
      return
    }
    if (!frameLocked) {
      frameLocked = true
      coreRenderer.setFrameLocked(true)
    }
    await coreRenderer.renderSyntheticFrame(delta, {waitForGpu: options?.waitForGpu})
  }

  /**
   * Update preset dynamically
   * Replaces all nodes with new preset configuration
   */
  function updatePreset(newPreset: PresetConfig): void {
    // Clear existing nodes
    nodeIdMap.forEach((_, nodeId) => {
      coreRenderer.removeNode(nodeId)
    })
    nodeIdMap.clear()

    // Re-register with new preset
    preset = newPreset
    registerPresetNodes()
  }

  /**
   * Get the GPU context, for host engines that want to share this renderer's device
   * Allows sharing the GPU device or WebGL context with other renderers
   */
  function getGPUContext(): GPUContext {
    // WebGPU-only now — the renderer's getInternalRenderer() returns { device, adapter?, root }.
    const internal = coreRenderer.getInternalRenderer()
    return {
      type: 'webgpu',
      device: internal?.device,
      adapter: internal?.adapter,
      canvas: presetCanvas as HTMLCanvasElement
    }
  }

  /**
   * Cleanup and dispose of all resources
   */
  function dispose(): void {
    coreRenderer.cleanup()
    frameLocked = false
  }

  /**
   * Register all nodes from preset
   * Creates a hierarchical node structure matching the preset
   */
  function registerPresetNodes(): void {
    // Register root. The GPU renderer requires a component definition on every node —
    // rootPassthrough supplies the shared "composite children / transparent when empty" root.
    const rootId = 'preset-root'
    coreRenderer.registerNode(
      rootId,
      rootPassthrough.fragment,
      null,
      null,
      {},
      rootPassthrough
    )

    // Register components
    preset.components.forEach((component, index) => {
      registerComponent(component, rootId, index)
    })
  }

  /**
   * Recursively register component and children
   */
  function registerComponent(
    component: ComponentConfig,
    parentId: string,
    renderOrder: number
  ): void {
    const componentDef = componentRegistry.get(component.type) ?? getRegisteredShader(component.type)
    if (!componentDef) {
      console.warn(`[createRendererFromJSON] Unknown component type: ${component.type}`)
      return
    }

    const nodeId = component.id || `${component.type}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
    nodeIdMap.set(nodeId, component.type)

    // Build complete props (user values merged over defaults) and create uniforms via the
    // canonical createGpuUniformsMap — handles string uniforms, null-safety, and compileTime flags.
    const completeProps = Object.fromEntries(
      Object.entries(componentDef.props).map(([key, propConfig]) => [
        key,
        component.props?.[key] !== undefined ? component.props[key] : propConfig.default
      ])
    )
    const uniforms = createGpuUniformsMap(componentDef, completeProps, nodeId)

    // Extract universal props
    const metadata: NodeMetadata = {
      blendMode: (component.props?.blendMode || 'normal') as BlendMode,
      opacity: component.props?.opacity,
      visible: component.props?.visible,
      renderOrder,
      id: component.id,
      mask: component.props?.maskSource ? {
        source: component.props.maskSource,
        type: component.props.maskType || 'alpha'
      } : undefined,
      transform: component.props?.transform ? {
        offsetX: 0, offsetY: 0, rotation: 0, scale: 1,
        anchorX: 0.5, anchorY: 0.5, edges: 'transparent' as const,
        ...component.props.transform
      } : undefined
    }

    // Register node
    coreRenderer.registerNode(
      nodeId,
      componentDef.fragment,
      parentId,
      metadata,
      uniforms,
      componentDef
    )

    // Register children
    component.children?.forEach((child, index) => {
      registerComponent(child, nodeId, index)
    })
  }

  return {
    initialize,
    renderFrame,
    updatePreset,
    getGPUContext,
    dispose,
    /**
     * Subscribe to "this environment can never run WebGPU" (or the GPU proved unusable at
     * runtime). Fires at most once; by then every GPU resource has been released and the
     * frame loop stopped. `initialize()` resolves normally in that case and
     * `renderFrame()` becomes a no-op, so a partner render loop keeps running safely —
     * subscribe here if you want to swap in a static fallback instead.
     */
    setOnUnavailable: coreRenderer.setOnUnavailable,
    /** Why the renderer gave up, or `null` while it is healthy. */
    getFailureReason: coreRenderer.getFailureReason
  }
}
