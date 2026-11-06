<script lang="ts">
  import { setContext, onMount, onDestroy } from 'svelte';
  import { shaderRendererGPU, rootPassthrough, debugWarn } from 'shaders-core';
  import type { UniformsMap, BlendMode, NodeMetadata, GpuFailureReason } from 'shaders-core';
  import { isExternalUser, startTelemetry } from 'shaders-core/telemetry';
  import { setColorSpaceMode } from 'shaders-core/utilities/transformations';

  declare const __SHADERS_VERSION__: string;

  interface Props {
    disableTelemetry?: boolean;
    colorSpace?: 'p3-linear' | 'srgb';
    toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal';
    isPreview?: boolean;
    onready?: () => void;
    /**
     * This browser/GPU cannot run the shader, so the canvas will stay transparent for good.
     * Fires at most once, and nothing is written to the console — render your own static
     * fallback (a gradient, an image) from this callback if you want one.
     */
    onunavailable?: (reason: GpuFailureReason) => void;
    children?: import('svelte').Snippet;
    [key: string]: any;
  }

  const { disableTelemetry = false, colorSpace = 'p3-linear', toneMapping = 'linear', isPreview = false, onready, onunavailable, children, ...rest }: Props = $props();

  // DOM references to the container and canvas
  let containerRef: HTMLDivElement;
  let canvasRef: HTMLCanvasElement;

  // SSR guard — renderer and children are deferred until mount
  let mounted = $state(false);

  // Bumped after every renderer (re-)initialization so mounted child components can
  // re-register: cleanup() clears the node registry and initializeRenderer() only
  // re-registers the root node (mirrors the Vue adapter's `shaderRendererResetSignal`).
  let rendererResetSignal = $state(0);

  // Unique ID for this root component
  const rootId = 'shader-root-' + Math.random().toString(36).substring(7);

  // Renderer instance — created in onMount (deferred for SSR compatibility)
  let rendererInstance: ReturnType<typeof shaderRendererGPU> | null = null;

  // Terminal GPU failure (no WebGPU, no adapter, device refused/lost, GPU unusable). The
  // renderer has already released everything — a canvas that never drew is transparent — so
  // we latch it here so none of the re-entry points below (visibility observer, colorSpace /
  // toneMapping effects) try again, and hand it to the host for a static fallback.
  let gpuUnavailable: GpuFailureReason | null = null;

  // Telemetry collector reference for cleanup
  let telemetryCollector: any = null;
  let telemetryStartTimeout: number | null = null;
  let shouldSendTelemetry: boolean | null = null;

  // Set context for child components - provide parent ID
  setContext('shaderParentId', rootId);

  // Provide color space to child components (function getter for reactivity)
  setContext('shaderColorSpace', () => colorSpace);

  // Provide the renderer reset signal so child components can re-register after
  // cleanup() + initializeRenderer() (function getter so child $effects stay reactive)
  setContext('shaderRendererResetSignal', () => rendererResetSignal);

  // Provide node registration function for child components
  // Callbacks capture `rendererInstance` by reference — they see the live value after onMount sets it
  setContext('shaderNodeRegister', (id: string, fragmentNodeFunc: any, parentId: string | null, metadata: any, uniforms: any = null, componentDef: any = null, domCanvas?: HTMLCanvasElement) => {
    if (!rendererInstance) return
    // If fragmentNodeFunc is null, the component is being unmounted, so remove it
    if (fragmentNodeFunc === null) {
      rendererInstance.removeNode(id);
    } else {
      rendererInstance.registerNode(id, fragmentNodeFunc, parentId, metadata, uniforms, componentDef, domCanvas);
    }
  });

  // Provide optimized uniform update function
  setContext('shaderUniformUpdate', (nodeId: string, uniformName: string, value: any) => {
    if (!rendererInstance) return
    rendererInstance.updateUniformValue(nodeId, uniformName, value);
  });

  // Provide optimized metadata update function
  setContext('shaderMetadataUpdate', (nodeId: string, metadata: any) => {
    if (!rendererInstance) return
    rendererInstance.updateNodeMetadata(nodeId, metadata);
  });

  // Track visibility state
  let wasVisible = false;
  let visibilityObserver: IntersectionObserver | null = null;

  // Wait for active rendering before starting telemetry collection.
  // Bounded: on a browser that can't run WebGPU fps never leaves 0, and an unbounded 500ms
  // recursion would tick for the life of the page against a renderer that will never draw.
  const MAX_TELEMETRY_POLLS = 40; // 40 × 500ms = 20s
  const startTelemetryWhenReady = () => {
    let polls = 0;
    const checkRendering = () => {
      if (!rendererInstance) return
      if (gpuUnavailable || ++polls > MAX_TELEMETRY_POLLS) {
        telemetryStartTimeout = null;
        return;
      }
      const stats = rendererInstance.getPerformanceStats();
      if (stats.fps > 0) {
        const version = typeof __SHADERS_VERSION__ !== 'undefined' ? __SHADERS_VERSION__ : 'unknown';
        telemetryCollector = startTelemetry(
          rendererInstance,
          version,
          disableTelemetry,
          isPreview
        );
        if (telemetryCollector) {
          telemetryCollector.start();
        }
        telemetryStartTimeout = null;
      } else {
        telemetryStartTimeout = window.setTimeout(checkRendering, 500);
      }
    };

    telemetryStartTimeout = window.setTimeout(checkRendering, 500);
  };

  // Function to initialize renderer with visibility check.
  //
  // Never rejects. `initialize()` resolves even when WebGPU is unavailable (it reports via
  // setOnUnavailable and leaves the canvas transparent), and the remaining steps are guarded
  // so a failure here can never surface as an unhandled promise rejection.
  const initializeRenderer = async () => {
    if (!canvasRef || !rendererInstance || gpuUnavailable) return;

    try {
      // Check if renderer is already initialized to avoid double initialization
      if (!rendererInstance.isInitialized()) {
        await rendererInstance.initialize({
          canvas: canvasRef,
          colorSpace: colorSpace,
          toneMapping: toneMapping,
        });
      }

      // The renderer reports unavailability asynchronously (via a microtask), so re-check
      // synchronously here too — nothing below is worth doing against a renderer that has
      // already given up.
      if (rendererInstance.getFailureReason()) return;

      // Register the root node. The GPU renderer requires a component definition on every node —
      // rootPassthrough supplies the shared "composite children / transparent when empty" root
      // (replacing the v1 bare vec4 fn).
      rendererInstance.registerNode(
        rootId,
        rootPassthrough.fragment,
        null, // No parent (this is the root)
        null, // No metadata to pass
        {},
        rootPassthrough
      );

      // cleanup() clears child nodes; mounted layers must register again after initialization.
      rendererResetSignal++;

      // Compute sampling decision once (includes random sampling roll)
      if (shouldSendTelemetry === null) {
        shouldSendTelemetry = isExternalUser();
      }

      if (shouldSendTelemetry && !telemetryCollector) {
        startTelemetryWhenReady();
      }

    } catch (err) {
      debugWarn('[Shaders] renderer initialization failed:', err);
    }
  };

  // Setup visibility observer for container hide/show detection
  const setupVisibilityObserver = () => {
    if (!containerRef || visibilityObserver) return;

    visibilityObserver = new IntersectionObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;

      const rect = containerRef?.getBoundingClientRect();
      const isCurrentlyVisible = entry.isIntersecting && rect && rect.width > 0 && rect.height > 0;
      
      if (isCurrentlyVisible && !wasVisible) {
        // Canvas became visible - resume animation
        if (rendererInstance?.isInitialized()) {
          rendererInstance.startAnimation();
          // Start telemetry if conditions are met and not already started
          if (shouldSendTelemetry && !telemetryCollector && !telemetryStartTimeout) {
            startTelemetryWhenReady();
          }
        } else {
          // First time visible, need to initialize
          void initializeRenderer();
        }
        wasVisible = true;
      } else if (!isCurrentlyVisible && wasVisible) {
        // Canvas became hidden - pause animation but keep renderer alive
        rendererInstance?.stopAnimation();
        wasVisible = false;
      }
    }, { threshold: 0 });

    visibilityObserver.observe(containerRef);
  };

  // Initialize renderer on mount - with visibility awareness
  onMount(async () => {
    // Create the renderer instance (deferred from module scope for SSR compatibility)
    rendererInstance = shaderRendererGPU();
    rendererInstance.setOnReady(() => onready?.());
    rendererInstance.setOnUnavailable((reason: GpuFailureReason) => {
      gpuUnavailable = reason;
      try {
        onunavailable?.(reason);
      } catch (err) {
        debugWarn('[Shaders] onunavailable callback threw:', err);
      }
    });
    mounted = true;

    // Wait for Svelte to render the canvas + children before proceeding
    await new Promise((resolve) => requestAnimationFrame(resolve));

    if (!canvasRef || !containerRef) {
      debugWarn('[Shaders] canvas or container ref is null in Shader onMount');
      return;
    }

    // Check if container is visible on mount
    const rect = containerRef.getBoundingClientRect();
    const isVisible = rect.width > 0 && rect.height > 0;

    if (isVisible) {
      // Container is visible, initialize immediately
      await initializeRenderer();
      wasVisible = true;
    } else {
      // Container is hidden, set up observer for when it becomes visible
      wasVisible = false;
    }

    // Always set up observer to handle show/hide cycles
    setupVisibilityObserver();
  });

  // Cleanup on component destroy
  onDestroy(() => {
    // Stop telemetry collection if active
    if (telemetryCollector) {
      telemetryCollector.stop();
      telemetryCollector = null;
    }

    // Clear telemetry start timeout if pending
    if (telemetryStartTimeout !== null) {
      clearTimeout(telemetryStartTimeout);
      telemetryStartTimeout = null;
    }

    // Clean up visibility observer
    if (visibilityObserver) {
      visibilityObserver.disconnect();
      visibilityObserver = null;
    }

    // Only cleanup if renderer was initialized
    if (rendererInstance?.isInitialized()) {
      try {
        rendererInstance.cleanup();
      } catch (err) {
        debugWarn('[Shaders] error during cleanup:', err);
      }
    }
  });

  // Watch for colorSpace changes and update the renderer
  // Skip the initial execution since colorSpace is set during initialization
  let isFirstEffect = true;
  $effect(() => {
    // Access colorSpace to make it reactive
    const currentColorSpace = colorSpace;
    
    if (isFirstEffect) {
      isFirstEffect = false;
      return;
    }
    
    if (rendererInstance?.isInitialized()) {
      // Update the global color space mode
      setColorSpaceMode(currentColorSpace);

      // Force re-initialization to apply color space changes
      rendererInstance.cleanup();
      void initializeRenderer();
    }
  });

  // Watch for toneMapping changes — re-initialize to apply the new mode
  let isFirstToneMappingEffect = true;
  $effect(() => {
    const currentToneMapping = toneMapping;

    if (isFirstToneMappingEffect) {
      isFirstToneMappingEffect = false;
      return;
    }

    if (rendererInstance?.isInitialized()) {
      rendererInstance.cleanup();
      void initializeRenderer();
    }
  });

</script>

<div class="shader" bind:this={containerRef} {...rest}>
  {#if mounted}
    <canvas
      data-renderer="shaders"
      bind:this={canvasRef}
      style="width: 100%; height: 100%; display: block;"
    ></canvas>
    {@render children?.()}
  {/if}
</div>