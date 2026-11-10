<script setup lang="ts">
import {onMounted, ref, shallowRef, onBeforeUnmount, provide, inject, watch, computed, useId, toValue, type Ref} from 'vue'
import {shaderRendererGPU, rootPassthrough, debugWarn, withGpuRecovery} from 'shaders-core'
import type {GpuFailureReason} from 'shaders-core'
import {isExternalUser, startTelemetry} from 'shaders-core/telemetry'
import {setColorSpaceMode} from 'shaders-core/utilities/transformations'

declare const __SHADERS_VERSION__: string

// A shared WebGPU device + adapter. When provided (via prop or inject), this
// Shader uses it instead of creating its own device — letting many Shaders on a
// single page (e.g. the Projects multi-tile canvas) share one GPU device.
interface SharedGpu {
  device: GPUDevice
  adapter: GPUAdapter
}

interface Props {
  disableTelemetry?: boolean
  enablePerformanceTracking?: boolean
  colorSpace?: 'p3-linear' | 'srgb'
  toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal'
  isPreview?: boolean
  gpu?: SharedGpu | null
  // Shared wall-clock time origin (a performance.now() timestamp). Pass the same
  // value to several Shaders so their time-based animations stay in sync.
  timeOrigin?: number | null
  // Keep the drawing buffer no larger than the browser window (the default). Hosts that
  // show the canvas through their own zoom and manage resolution pass false.
  capToViewport?: boolean | null
}

const props = withDefaults(defineProps<Props>(), {
  disableTelemetry: false,
  enablePerformanceTracking: false,
  colorSpace: 'p3-linear',
  toneMapping: 'linear',
  isPreview: false,
  gpu: null,
  timeOrigin: null,
  capToViewport: null
})

// A parent may provide one shared device to a whole subtree. An explicit `gpu`
// prop takes precedence over the injected one. Falsy when no sharing is in play
// (every existing standalone/npm use), in which case the renderer makes its own
// device exactly as before.
const injectedGpu = inject<SharedGpu | null>('shaderGpuDevice', null)
const resolveGpu = (): SharedGpu | null => props.gpu ?? injectedGpu ?? null

// A parent may provide one shared time origin to a subtree; an explicit prop wins.
const injectedTimeOrigin = inject<number | null>('shaderTimeOrigin', null)
const resolveTimeOrigin = (): number | null => props.timeOrigin ?? injectedTimeOrigin ?? null

// A parent may turn the window-size buffer cap off for a subtree (an infinite canvas that
// zooms its tiles and sets their resolution itself); an explicit prop wins.
const injectedCapToViewport = inject<boolean | null>('shaderCapToViewport', null)
const resolveCapToViewport = (): boolean => props.capToViewport ?? injectedCapToViewport ?? true

// An optional custom IntersectionObserver root. When a parent provides one (e.g.
// the Projects infinite canvas providing its viewport container), on-screen
// visibility — and thus the auto pause/resume below — is measured against THAT
// element's box instead of the browser viewport. This lets tiles panned outside
// the canvas region pause even while they remain within the browser viewport.
// Falsy (every standalone use) → observe against the viewport exactly as before.
const injectedViewportRoot = inject<Ref<Element | null> | Element | null>('shaderViewportRoot', null)
const resolveViewportRoot = (): Element | null => (toValue(injectedViewportRoot) as Element | null) ?? null

const emit = defineEmits<{
  ready: []
  /**
   * This browser/GPU cannot run the shader, so the canvas will stay transparent for good.
   * Fires at most once, and nothing is written to the console — render your own static
   * fallback (a gradient, an image) from this event if you want one.
   */
  unavailable: [reason: GpuFailureReason]
}>()

// Provide color space to child components so they can set the global before creating uniforms
provide('shaderColorSpace', computed(() => props.colorSpace))

const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)

// Unique ID for this root component
const rootId = ref('shader-root-' + useId().replace(/[^a-zA-Z0-9_]/g, ''))

// Terminal GPU failure (no WebGPU, no adapter, device refused/lost, GPU unusable). The
// renderer has already released everything — a canvas that never drew is transparent — so we
// latch it here so none of the re-entry points below (visibility observer, colorSpace /
// toneMapping watchers) try again, and surface it as an event for hosts wanting a fallback.
const gpuUnavailable = ref<GpuFailureReason | null>(null)
const handleUnavailable = (reason: GpuFailureReason) => {
  gpuUnavailable.value = reason
  emit('unavailable', reason)
}

// Reactive signal to trigger child component re-registration
const rendererResetSignal = ref(0)

// Renderer instance, wrapped with device-loss recovery (shared with every host, see core
// gpu/recovery.ts): a lost device is cleaned up and re-initialized through the same path a
// toneMapping change takes, and a layer that keeps crashing the GPU is switched off.
const rendererInstance = shallowRef(withGpuRecovery(shaderRendererGPU(), {
  rebuild: () => initializeRenderer().then(() => {
    rendererResetSignal.value++
  }),
  onGiveUp: handleUnavailable
}))

// Wire up onReady callback to emit event
rendererInstance.value.setOnReady(() => emit('ready'))
rendererInstance.value.setOnUnavailable(handleUnavailable)

// Telemetry collector reference for cleanup
let telemetryCollector: any = null
let telemetryStartTimeout: number | null = null
let shouldSendTelemetry: boolean | null = null

// Provide our ID to children
provide('shaderParentId', rootId.value)

// Provide the reset signal so children can watch for renderer resets
provide('shaderRendererResetSignal', rendererResetSignal)

// Provide all registration and update functions for child components
provide('shaderNodeRegister', (id: string, fragmentNodeFunc: any, parentId: string | null, metadata: any, uniforms: any = null, componentDefinition: any = null, domElement: HTMLElement | undefined = undefined) => {
  // If fragmentNodeFunc is null, the component is being unmounted, so remove it
  if (fragmentNodeFunc === null) {
    rendererInstance.value.removeNode(id)
  } else {
    rendererInstance.value.registerNode(id, fragmentNodeFunc, parentId, metadata, uniforms, componentDefinition, domElement)
  }
})

// Provide optimized uniform update function
provide('shaderUniformUpdate', (nodeId: string, uniformName: string, value: any) => {
  rendererInstance.value.updateUniformValue(nodeId, uniformName, value)
})

// Provide optimized metadata update function
provide('shaderMetadataUpdate', (nodeId: string, metadata: any) => {
  rendererInstance.value.updateNodeMetadata(nodeId, metadata)
})

// Function to fully initialize renderer and register root node.
//
// Never rejects. `initialize()` resolves even when WebGPU is unavailable (it reports via
// setOnUnavailable and leaves the canvas transparent), and every remaining step is guarded
// so a failure here can never surface as an unhandled promise rejection in the host app.
const initializeRenderer = async () => {
  if (!canvasRef.value || gpuUnavailable.value) return

  try {
    // Check if renderer is already initialized to avoid double initialization
    if (!rendererInstance.value.isInitialized()) {
      const sharedGpu = resolveGpu()
      const sharedTimeOrigin = resolveTimeOrigin()
      await rendererInstance.value.initialize({
        canvas: canvasRef.value,
        enablePerformanceTracking: props.enablePerformanceTracking,
        colorSpace: props.colorSpace,
        toneMapping: props.toneMapping,
        capToViewport: resolveCapToViewport(),
        ...(sharedGpu ? {gpu: sharedGpu} : {}),
        ...(sharedTimeOrigin != null ? {timeOrigin: sharedTimeOrigin} : {})
      })
    }

    // The renderer reports unavailability asynchronously (via a microtask), so re-check
    // synchronously here too: there is no point registering nodes or booting telemetry
    // against a renderer that has already given up.
    if (rendererInstance.value.getFailureReason()) return

    // Register the root node (this may be called multiple times, renderer should handle it).
    // The GPU renderer requires a component definition on every node — rootPassthrough supplies the
    // shared "composite children / transparent when empty" root (replacing the v1 bare vec4 fn).
    rendererInstance.value.registerNode(
        rootId.value,
        rootPassthrough.fragment,
        null, // No parent (this is the root)
        null, // No metadata to pass
        {},
        rootPassthrough
    )

    // Compute sampling decision once (includes random sampling roll)
    if (shouldSendTelemetry === null) {
      shouldSendTelemetry = isExternalUser()
    }

    if (shouldSendTelemetry && !telemetryCollector) {
      startTelemetryWhenReady()
    }
  } catch (error) {
    debugWarn('[Shaders] renderer initialization failed:', error)
  }
}

// Wait for active rendering before starting telemetry collection.
// Bounded: on a browser that can't run WebGPU fps never leaves 0, and an unbounded 500ms
// recursion would tick for the life of the page against a renderer that will never draw.
const MAX_TELEMETRY_POLLS = 40 // 40 × 500ms = 20s
const startTelemetryWhenReady = () => {
  let polls = 0
  const checkRendering = () => {
    if (gpuUnavailable.value || ++polls > MAX_TELEMETRY_POLLS) {
      telemetryStartTimeout = null
      return
    }
    const stats = rendererInstance.value.getPerformanceStats()
    if (stats.fps > 0) {
      const version = typeof __SHADERS_VERSION__ !== 'undefined' ? __SHADERS_VERSION__ : 'unknown'
      telemetryCollector = startTelemetry(
        rendererInstance.value,
        version,
        props.disableTelemetry,
        props.isPreview
      )
      if (telemetryCollector) {
        telemetryCollector.start()
      }
      telemetryStartTimeout = null
    } else {
      telemetryStartTimeout = setTimeout(checkRendering, 500) as unknown as number
    }
  }

  telemetryStartTimeout = setTimeout(checkRendering, 500) as unknown as number
}

// Initialize renderer on mount
// Track visibility state
let wasVisible = false
let visibilityObserver: IntersectionObserver | null = null

// Set by an explicit pause() call from a parent. While true the renderer is kept
// paused regardless of on-screen visibility — the observer won't auto-resume it
// until resume() clears the flag. Lets consumers force-pause (JS-package parity)
// without the visibility observer fighting them.
let externallyPaused = false

const setupVisibilityObserver = () => {
  if (!containerRef.value || visibilityObserver) return

  visibilityObserver = new IntersectionObserver((entries) => {
    const entry = entries[0]
    if (!entry) return

    const isCurrentlyVisible = entry.isIntersecting

    if (isCurrentlyVisible && !wasVisible) {
      // Canvas became visible - resume animation (unless explicitly paused)
      if (rendererInstance.value.isInitialized()) {
        if (!externallyPaused) rendererInstance.value.startAnimation()
        // Start telemetry if conditions are met and not already started
        if (shouldSendTelemetry && !telemetryCollector && !telemetryStartTimeout) {
          startTelemetryWhenReady()
        }
      } else if (!externallyPaused && !gpuUnavailable.value) {
        // First time visible, need to initialize
        void initializeRenderer().then(() => {
          // Increment the reset signal to trigger child component re-registration
          rendererResetSignal.value++
        })
      }
      wasVisible = true
    } else if (!isCurrentlyVisible && wasVisible) {
      // Canvas became hidden - pause animation but keep renderer alive
      rendererInstance.value.stopAnimation()
      wasVisible = false
    }
  }, { root: resolveViewportRoot(), rootMargin: '128px', threshold: 0 })

  visibilityObserver.observe(containerRef.value)
}

onMounted(async () => {
  if (canvasRef.value && containerRef.value) {
    // Force cleanup in case of HMR or stale renderer state (dev mode only)
    if (import.meta.env.DEV && rendererInstance.value.isInitialized()) {
      rendererInstance.value.cleanup()
    }

    // Check if container is visible
    const rect = containerRef.value.getBoundingClientRect()
    const isVisible = rect.width > 0 && rect.height > 0

    if (isVisible) {
      // Canvas is visible, initialize immediately
      await initializeRenderer()
      // Increment signal for initial registration (though children should already be registering)
      rendererResetSignal.value++
      wasVisible = true
    } else {
      // Canvas is hidden, set up observer for when it becomes visible
      wasVisible = false
    }

    // Always set up observer to handle show/hide cycles
    setupVisibilityObserver()
  }
})

// Watch for colorSpace changes and update the renderer
// Skip the initial execution since colorSpace is set during initialization
let isFirstRun = true
watch(() => props.colorSpace, (newColorSpace) => {
  if (isFirstRun) {
    isFirstRun = false
    return
  }

  if (rendererInstance.value.isInitialized()) {
    // Update the global color space mode
    setColorSpaceMode(newColorSpace)

    // Increment reset signal to trigger child component re-registration
    // This will cause all components to re-register and re-transform their colors
    rendererResetSignal.value++
  }
})

// Watch for toneMapping changes — re-initialize to apply the new mode.
// (No first-run skip: a non-immediate watch never fires at mount, so a guard here would
// swallow the user's FIRST genuine change — the reported "takes two changes to apply" bug.)
watch(() => props.toneMapping, () => {
  if (rendererInstance.value.isInitialized()) {
    rendererInstance.value.cleanup()
    void initializeRenderer().then(() => {
      rendererResetSignal.value++
    })
  }
})


// Keep the renderer's shared time origin in sync if the prop/inject changes.
watch(() => resolveTimeOrigin(), (origin) => {
  if (rendererInstance.value.isInitialized()) {
    rendererInstance.value.setTimeOrigin?.(origin ?? null)
  }
})

// Capture options for full-quality image export
type CaptureFormat = 'png' | 'jpeg' | 'webp'

interface CaptureImageOptions {
  format?: CaptureFormat
  scale?: number
  quality?: number
  maxWidth?: number
}

// Force a fresh frame before reading pixels from the canvas.
// On WebGPU the swap chain texture is invalidated after present, so we must
// render-and-wait inside the same task that performs the readback. Errors
// are warned-only, not rethrown — a transient "renderer not ready" during
// autosave would otherwise surface as a black-image failure even though the
// canvas may still hold a valid prior frame.
const renderForCapture = async (): Promise<void> => {
  try {
    await rendererInstance.value.renderAndWait()
  } catch (error) {
    debugWarn('[Shaders] Failed to force render before capture:', error)
  }
}

// Snapshot the source canvas via toBlob('image/png'), then decode and resize.
// On WebGPU/WebGL canvases, toBlob/toDataURL are the only readback paths that
// reliably force a GPU→CPU pixel copy regardless of swap chain / drawing
// buffer presentation state. drawImage(canvas) and createImageBitmap(canvas)
// both go through the canvas "snapshot" path which can read a presented (or
// recycled) frame — that's why our previous attempts produced intermittent
// black images. The PNG round-trip costs a few ms; correctness wins.
const canvasToBlob = async (
  source: HTMLCanvasElement,
  targetWidth: number,
  targetHeight: number,
  mimeType: string,
  quality?: number
): Promise<Blob> => {
  const sourceBlob: Blob = await new Promise((resolve, reject) => {
    source.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('Failed to read source canvas'))
    }, 'image/png')
  })

  const needsReencode = mimeType !== 'image/png'
    || targetWidth !== source.width
    || targetHeight !== source.height
  if (!needsReencode) {
    return sourceBlob
  }

  const bitmap = await createImageBitmap(sourceBlob)
  try {
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = targetWidth
    tempCanvas.height = targetHeight
    const ctx = tempCanvas.getContext('2d')
    if (!ctx) {
      throw new Error('Failed to get canvas context')
    }
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight)
    return await new Promise<Blob>((resolve, reject) => {
      tempCanvas.toBlob((blob) => {
        if (blob) resolve(blob)
        else reject(new Error('Failed to encode image'))
      }, mimeType, quality)
    })
  } finally {
    bitmap.close()
  }
}

// Full-quality image export. Defaults to PNG at the canvas's native backing
// resolution. Pass `scale` > 1 to upscale a 2D draw of the source frame.
const captureImage = async (options: CaptureImageOptions = {}): Promise<Blob> => {
  if (!canvasRef.value) {
    throw new Error('Canvas not available')
  }

  await renderForCapture()

  const source = canvasRef.value
  const format = options.format ?? 'png'
  const scale = Math.max(0.1, options.scale ?? 1)
  const targetWidth = Math.round(source.width * scale)
  const targetHeight = Math.round(source.height * scale)
  const mimeType = format === 'png' ? 'image/png'
    : format === 'webp' ? 'image/webp'
    : 'image/jpeg'
  const quality = format === 'png' ? undefined : (options.quality ?? 0.95)

  return canvasToBlob(source, targetWidth, targetHeight, mimeType, quality)
}

// Thumbnail-friendly capture for autosave/version snapshots. Caps at maxWidth,
// JPEG at 85% to keep upload payloads small.
//
// The editor's infinite canvas shrinks the GPU backing buffer when zoomed out
// (setResolutionScale), so a screenshot taken then would inherit that tiny
// resolution. If the live buffer is smaller than the thumbnail target, render
// a frame at the needed pixel ratio via the recording-resolution machinery,
// capture, then restore the steady-state resolution.
const captureScreenshot = async (maxWidth = 1200): Promise<Blob> => {
  if (!canvasRef.value) {
    throw new Error('Canvas not available')
  }

  const cssWidth = canvasRef.value.clientWidth
  const fullRatio = Math.min((typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1, 2)
  const targetBackingWidth = Math.min(maxWidth, Math.round(cssWidth * fullRatio))
  let restoreResolution: (() => Promise<void>) | null = null

  try {
    if (cssWidth > 0 && canvasRef.value.width < targetBackingWidth) {
      restoreResolution = await beginRecordingResolution(targetBackingWidth / cssWidth)
    }

    await renderForCapture()

    const source = canvasRef.value
    const sourceWidth = source.width
    const sourceHeight = source.height

    let targetWidth = sourceWidth
    let targetHeight = sourceHeight
    if (sourceWidth > maxWidth) {
      targetWidth = maxWidth
      targetHeight = Math.round((sourceHeight / sourceWidth) * maxWidth)
    }

    return await canvasToBlob(source, targetWidth, targetHeight, 'image/jpeg', 0.85)
  } finally {
    if (restoreResolution) await restoreResolution()
  }
}

// Expose performance stats method
const getPerformanceStats = () => {
  return rendererInstance.value.getPerformanceStats()
}

/** GPU time per frame of this shader as it renders now (null when the device can't time passes). */
const profileGpu = (options?: { frames?: number }) => rendererInstance.value.profileGpu(options)

// Live resolved value of a mouse-position driver (for UI readback — e.g. showing the
// dynamically-driven X/Y in an editor). Returns null when the prop has no active driver.
const getLiveDriverValue = (nodeId: string, propName: string) => {
  return rendererInstance.value.getLiveDriverValue(nodeId, propName)
}

// Live resolved scalar value of a range driver (mouse / auto-animate) for UI readback. Returns null
// for map drivers (per-pixel) or when the prop has no active scalar driver.
const getLiveScalarValue = (nodeId: string, propName: string) => {
  return rendererInstance.value.getLiveScalarValue(nodeId, propName)
}

// Recording helper — temporarily override the renderer's pixel ratio so the
// canvas backing store renders at the desired output resolution. Returns a
// restore function the caller invokes when recording stops.
//
// Video codecs (H.264/VP9 4:2:0 chroma subsampling) require even width/height —
// an odd backing buffer produces a corrupted/unplayable export. Video recording
// always passes an integer pixelRatio (1 or 2 — see RecordingResolution), so
// evenifying the CSS-pixel input is sufficient to guarantee an even backing
// buffer. captureScreenshot passes fractional ratios, but JPEG stills have no
// evenness requirement.
const evenify = (n: number): number => {
  const rounded = Math.round(n)
  return rounded % 2 === 0 ? rounded : rounded + 1
}

const beginRecordingResolution = async (pixelRatio: number): Promise<() => Promise<void>> => {
  if (!canvasRef.value || !rendererInstance.value.beginRecordingResolution) {
    return async () => {}
  }
  // The canvas fills its (transform-scaled) frame wrapper at 100%, so clientWidth/
  // Height report the frame's CSS-pixel layout size (unaffected by the zoom
  // transform). That's the logical size we record at; pixelRatio scales it to the
  // chosen output resolution. Routing through the core renderer keeps currentWidth,
  // the camera frustum, px-unit uniforms and the resize callbacks all consistent —
  // the auto-resizing child render targets (getSize × pixelRatio) and the shaders'
  // input dimensions (currentWidth × pixelRatio) then agree, so Glow / ReflectivePlane
  // / blurs render correctly at the recording resolution.
  const cssWidth = evenify(canvasRef.value.clientWidth)
  const cssHeight = evenify(canvasRef.value.clientHeight)

  // Off-screen throttle would freeze captureStream at 1 FPS if the canvas
  // scrolls out of view. Force full frame rate while recording, but capture
  // the previous flag so embedders that initialised with forceFullFrameRate
  // (e.g. hidden-canvas integrations) keep their original setting on restore.
  const previousForce = rendererInstance.value.setForceFullFrameRate?.(true) ?? false
  let restoreResolution: (() => void) | null = null
  try {
    restoreResolution = rendererInstance.value.beginRecordingResolution(cssWidth, cssHeight, pixelRatio)
    await renderForCapture()
    // A resize reconfigures the WebGPU swap chain; the very next frame can still land on
    // a not-yet-settled buffer, which showed up as a black first frame in exports. A second
    // forced render after the reconfigure reliably lands on a valid, presentable frame.
    await renderForCapture()
  } catch (error) {
    // Roll back whatever was applied so a failed setup doesn't leave the
    // renderer stuck at the recording resolution / forced frame rate.
    try {
      restoreResolution?.()
    } finally {
      rendererInstance.value.setForceFullFrameRate?.(previousForce)
    }
    throw error
  }

  return async () => {
    try {
      restoreResolution?.()
      await renderForCapture()
    } finally {
      rendererInstance.value.setForceFullFrameRate?.(previousForce)
    }
  }
}

// Steady-state resolution scale (Projects multi-tile canvas). Shrinks the GPU
// backing buffer by `scale` (pixel-ratio lever) while keeping the authored frame
// size pixel-identical. scale === 1 restores device-native resolution. No-op when
// the renderer isn't initialised yet.
const setResolutionScale = (scale: number): void => {
  rendererInstance.value.setResolutionScale?.(scale)
}

// Cap the on-screen frame rate (null restores the default 60 FPS). Pairs with
// setResolutionScale as the multi-tile canvas's per-tile performance levers:
// resolution scales with on-screen SIZE, frame rate with how much motion a tile
// that small can even show.
const setFrameRateCap = (fps: number | null): void => {
  rendererInstance.value.setFrameRateCap?.(fps)
}

// Explicitly pause/resume the render loop (parity with the JS package's
// createShader().pause()/resume()). Pausing stops per-frame GPU work while
// keeping the renderer — and its compiled shader graph — fully alive, so resume
// is instant with no recompile. While paused the visibility observer won't
// auto-resume; resume() re-starts only if the canvas is currently on-screen.
const pause = (): void => {
  externallyPaused = true
  if (rendererInstance.value.isInitialized()) rendererInstance.value.stopAnimation()
}
const resume = (): void => {
  externallyPaused = false
  if (rendererInstance.value.isInitialized() && wasVisible) rendererInstance.value.startAnimation()
}

// Advances the renderer's clock by exactly `deltaSeconds` and renders a single
// frame, independent of wall-clock time. Used by the frame-locked video export
// path so exported frame count is deterministic (duration * fps) regardless of
// actual render throughput.
//
// `waitForGpu: false` skips the per-frame GPU fence, which costs ~104ms per call on Firefox
// 152 (vs ~1ms on Chromium) and dominates a long export. Only safe when the caller reads the
// canvas via toBlob() immediately after — see the core renderer's note. captureFrameBitmap()
// below does exactly that, so the export loop pairs the two.
const renderSyntheticFrame = async (deltaSeconds: number, options?: {waitForGpu?: boolean}): Promise<void> => {
  await rendererInstance.value.renderSyntheticFrame(deltaSeconds, options)
}

// Captures the canvas's current pixels as an ImageBitmap for video export. Goes through
// a toBlob() round-trip rather than handing the raw canvas element to mediabunny/WebCodecs
// — constructing a VideoFrame directly from a WebGPU canvas is susceptible to the same
// presented/recycled-frame staleness bug as drawImage/createImageBitmap(canvas), which
// showed up as corrupted or jittery exports. The ImageBitmap here is decoded from an
// already-copied blob, so it's a fully realized, independent pixel buffer with no such risk.
//
// Deliberately bypasses canvasToBlob (which always PNG-encodes first, then re-encodes if a
// different format is requested — a double compression pass). This is a per-frame hot path
// during recording, feeding into an already-lossy video codec, so we go straight to a single
// JPEG encode: PNG's deflate compression is dramatically slower than JPEG's DCT-based encode
// for typical shader output, and that cost was the main contributor to recording slowdown.
// `transparent: true` (opt-in from the recording dialog) switches to PNG instead, since JPEG
// has no alpha channel — the export loop only pays PNG's slower deflate cost when the user
// has actually asked to preserve a transparent background.
const captureFrameBitmap = async (quality = 0.92, transparent = false): Promise<ImageBitmap> => {
  if (!canvasRef.value) {
    throw new Error('Canvas not available')
  }
  const source = canvasRef.value
  const blob: Blob = await new Promise((resolve, reject) => {
    source.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('Failed to read source canvas'))
    }, transparent ? 'image/png' : 'image/jpeg', quality)
  })
  return await createImageBitmap(blob)
}

// Expose capture + recording helpers to parent components
defineExpose({
  captureImage,
  captureScreenshot,
  getPerformanceStats,
  profileGpu,
  getLiveDriverValue,
  getLiveScalarValue,
  beginRecordingResolution,
  setResolutionScale,
  setFrameRateCap,
  pause,
  resume,
  getCanvas: () => canvasRef.value,
  /** Non-null once the GPU has permanently failed; the canvas stays transparent. */
  getFailureReason: () => gpuUnavailable.value,
  renderSyntheticFrame,
  /** Drive the pointer from a script (captures that show a click or a drag); see the core renderer. */
  beginScriptedInput: () => rendererInstance.value.beginScriptedInput(),
  /** Why the current structure isn't drawing (a layer that failed to build or compile), or null. */
  getCompositionError: () => rendererInstance.value?.getCompositionError?.() ?? null,
  captureFrameBitmap,
  startAnimation: () => rendererInstance.value.startAnimation(),
  stopAnimation: () => rendererInstance.value.stopAnimation()
})

onBeforeUnmount(() => {
  // Stop telemetry collection if active
  if (telemetryCollector) {
    telemetryCollector.stop()
    telemetryCollector = null
  }

  // Clear telemetry start timeout if pending
  if (telemetryStartTimeout !== null) {
    clearTimeout(telemetryStartTimeout)
    telemetryStartTimeout = null
  }

  // Clean up visibility observer
  if (visibilityObserver) {
    visibilityObserver.disconnect()
    visibilityObserver = null
  }

  rendererInstance.value.disposeRecovery()
  rendererInstance.value.cleanup()
})
</script>

<template>
  <div class="shader" ref="containerRef" v-bind="$attrs">
    <canvas
      data-renderer="shaders"
      ref="canvasRef"
      style="width: 100%; height: 100%; display: block;"
    >
      <slot></slot>
    </canvas>
  </div>
</template>