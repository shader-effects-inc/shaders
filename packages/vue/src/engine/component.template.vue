<script setup lang="ts">
import {watch, toValue, computed, inject, provide, onBeforeUnmount, onMounted, ref, effectScope, useId} from 'vue'
import {
  createGpuUniformsMap,
  resolveBoundingBox,
  type GpuUniformsMap,
  type BlendMode,
  type NodeMetadata,
  type PropConfig,
  type MaskConfig,
  type PropDriver,
  type TransformConfig,
  type BoundingBoxConfig,
  type BoundingBoxInput,
  type LayoutConfig
} from 'shaders-core'
import {setColorSpaceMode} from 'shaders-core/utilities/transformations'

// @ts-ignore
import {componentDefinition, type ComponentProps} from 'shaders-core/__SHADER_NAME__'

// Warn once per page load if this is an experimental component
let _experimentalWarnedOnce = false
if ((componentDefinition as any).experimental && !_experimentalWarnedOnce) {
  _experimentalWarnedOnce = true
  const _e = (componentDefinition as any).experimental
  console.info(`%c⚠ [Shaders] ${componentDefinition.name} is experimental: ${_e.message}`, 'color: #f59e0b; font-weight: bold')
}

/**
 * Define component props including blend mode, opacity, visibility, masking, and transformation
 */
interface ExtendedComponentProps extends Partial<ComponentProps> {
__MAPPABLE_PROP_DECLS__  blendMode?: BlendMode;
  opacity?: number;
  visible?: boolean;
  id?: string;
  maskSource?: string;
  maskType?: string;
  renderOrder?: number;
  transform?: Partial<TransformConfig>;
  boundingBox?: BoundingBoxInput;
  flow?: LayoutConfig;
  absolute?: boolean;
}

function isPropDriver(value: unknown): value is PropDriver {
  return typeof value === 'object' && value !== null && 'type' in value &&
    ((value as any).type === 'map' || (value as any).type === 'mouse' || (value as any).type === 'mouse-position' || (value as any).type === 'auto-animate')
}

/**
 * Define the Vue component props and their default values from the shader definition
 * This creates the standard reactive Vue props with proper defaults for the component
 */
const props = withDefaults(defineProps<ExtendedComponentProps>(), {
  blendMode: 'normal',
  visible: true,
  // opacity intentionally has no default - handled by renderer
  // transform intentionally has no default - handled by effectiveTransform computed
  ...Object.entries(componentDefinition.props).reduce(
      (acc, [key, config]) => {
        acc[key] = (config as unknown as PropConfig<typeof config>).default
        return acc
      },
      {} as Record<string, any>
  )
})

/**
 * Use the provided ID or generate a unique identifier for this component instance
 */
const instanceId = (props.id ? props.id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || useId().replace(/[^a-zA-Z0-9_]/g, '')

/**
 * Provide our unique identifier to child components
 */
provide('shaderParentId', instanceId)

/**
 * Default transform configuration (optimized for zero overhead)
 */
const DEFAULT_TRANSFORM: TransformConfig = {
  offsetX: 0,
  offsetY: 0,
  rotation: 0,
  scale: 1,
  anchorX: 0.5,
  anchorY: 0.5,
  edges: 'transparent'
}

/**
 * Computed transform that merges user-provided values with defaults
 */
const effectiveTransform = computed<TransformConfig>(() => ({
  ...DEFAULT_TRANSFORM,
  ...props.transform
}))

/**
 * Effective bounding box — partial configs merge with full-frame defaults,
 * absent stays undefined so the bbox render path is not activated
 */
const effectiveBoundingBox = computed<BoundingBoxConfig | undefined>(() => resolveBoundingBox(props.boundingBox))

/**
 * Creates a non-reactive object containing only props that differ from defaults
 * This optimization prevents unnecessary GPU uniform updates for unchanged values
 * Special props like blendMode and opacity are handled separately
 */
const shaderReadyProps = computed(() => {
  let baseProps = {
    ...Object.entries(componentDefinition.props).reduce(
        (acc, [key, config]) => {
          acc[key] = (config as unknown as PropConfig<typeof config>).default
          return acc
        }, {} as Record<string, any>
    )
  }
  for (const key in props) {
    if (key !== 'blendMode' && key !== 'opacity' && key !== 'visible' &&
        key !== 'id' && key !== 'maskSource' && key !== 'maskType' && key !== 'renderOrder' &&
        key !== 'transform' && key !== 'boundingBox' && key !== 'flow' && key !== 'absolute') {
      const val = toValue((props as any)[key])
      if (isPropDriver(val)) continue // PropDrivers go to metadata.maps, not uniforms
      if (val !== (Object.entries(componentDefinition.props).reduce(
            (acc, [key, config]) => {
              acc[key] = (config as unknown as PropConfig<typeof config>).default
              return acc
            }, {} as Record<string, any>
        ) as any)[key]) {
        (baseProps as any)[key] = val
      }
    }
  }
  return baseProps
})

// Collect PropDriver values from shader props into the maps metadata structure
const mapsFromProps = computed(() => {
  const maps: Record<string, PropDriver> = {}
  for (const key of Object.keys(componentDefinition.props)) {
    const val = (props as any)[key]
    if (isPropDriver(val)) maps[key] = val as PropDriver
  }
  return Object.keys(maps).length > 0 ? maps : undefined
})

/**
 * Get the color space from the root Shader component.
 * Used to set the global color space mode before creating uniforms.
 */
const shaderColorSpace = inject<{value: 'p3-linear' | 'srgb'}>('shaderColorSpace')

/**
 * Creates the GPU uniform values map using only the changed props
 * Set the global color space mode before creating uniforms so colors are transformed correctly
 */
if (shaderColorSpace) {
  setColorSpaceMode(shaderColorSpace.value)
}
const uniforms: GpuUniformsMap = createGpuUniformsMap(componentDefinition, shaderReadyProps.value, instanceId)

/**
 * Provide the ID of the parent down to all child components
 */
const parentId = inject<string>('shaderParentId')
if (parentId === undefined) {
  throw new Error('Shader components must be used inside an <Shader> component or another shader component')
}

/**
 * Get the node registration function (provided by the root Shader component).
 * Calling this registers this component into the shader node graph / registry
 */
const parentRegister = inject<(id: string, fragmentNodeFunc: any, parentId: string | null, metadata: NodeMetadata | null, uniforms: GpuUniformsMap | null, componentDefinition: any, domCanvas?: HTMLCanvasElement) => void>('shaderNodeRegister')
if (parentRegister === undefined) {
  throw new Error('Shader components must be used inside an <Shader> component or another shader component')
}

/**
 * Get the uniform update function (provided by the root Shader component).
 * Calling this updates the value of a uniform in the shader node graph / registry
 */
const parentUniformUpdate = inject<(nodeId: string, uniformName: string, value: any) => void>('shaderUniformUpdate')
if (parentUniformUpdate === undefined) {
  throw new Error('Shader components require shaderUniformUpdate from parent')
}

/**
 * Get the metadata update function (provided by the root Shader component).
 * Calling this updates the metadata associated with this node in the shader node graph / registry
 */
const parentMetadataUpdate = inject<(nodeId: string, metadata: NodeMetadata) => void>('shaderMetadataUpdate')
if (parentMetadataUpdate === undefined) {
  throw new Error('Shader components require shaderMetadataUpdate from parent')
}

/**
 * Get the renderer reset signal (provided by the root Shader component).
 * When this changes, it means the renderer has been reinitialized and we need to re-register
 */
const rendererResetSignal = inject<{value: number}>('shaderRendererResetSignal')

/**
 * DOM marker ref for determining render order from template position
 */
const orderMarker = ref<HTMLSpanElement | null>(null)

/**
 * capturesDOM — canvas layoutsubtree portal for HTMLInCanvas-style shaders
 */
const isCapturesDOM = !!(componentDefinition as any).capturesDOM
const captureCanvas = ref<HTMLCanvasElement | null>(null)
const captureW = ref(typeof window !== 'undefined' ? Math.round(window.innerWidth * Math.min(window.devicePixelRatio, 2)) : 0)
const captureH = ref(typeof window !== 'undefined' ? Math.round(window.innerHeight * Math.min(window.devicePixelRatio, 2)) : 0)
const captureStyle = { position: 'fixed' as const, inset: '0', width: '100vw', height: '100vh', zIndex: '-9999' }

if (isCapturesDOM) {
  const onWinResize = () => {
    const d = Math.min(window.devicePixelRatio, 2)
    captureW.value = Math.round(window.innerWidth * d)
    captureH.value = Math.round(window.innerHeight * d)
  }
  onMounted(() => {
    // Set correct dimensions on client after SSR
    const d = Math.min(window.devicePixelRatio, 2)
    captureW.value = Math.round(window.innerWidth * d)
    captureH.value = Math.round(window.innerHeight * d)
    window.addEventListener('resize', onWinResize)
  })
  onBeforeUnmount(() => { window.removeEventListener('resize', onWinResize) })
}

/**
 * Stores the DOM-detected render order for use across re-registrations
 * Only used when the user hasn't explicitly set renderOrder
 */
let detectedRenderOrder: number | undefined = undefined

/**
 * Register this component with the renderer
 */
const registerWithRenderer = () => {
  parentRegister(
      instanceId,
      componentDefinition.fragment,
      parentId,
      {
        blendMode: props.blendMode,
        opacity: props.opacity,
        visible: props.visible === false ? false : true,
        id: props.id,
        mask: props.maskSource ? {
          source: props.maskSource,
          type: props.maskType || 'alpha'
        } as MaskConfig : undefined,
        maps: mapsFromProps.value,
        renderOrder: props.renderOrder ?? detectedRenderOrder,
        transform: effectiveTransform.value,
        boundingBox: effectiveBoundingBox.value,
        flow: props.flow,
        absolute: props.absolute
      },
      uniforms,
      componentDefinition,
      isCapturesDOM ? (captureCanvas.value ?? undefined) : undefined
  )
}

/**
 * On mount, detect this component's position among shader siblings in the DOM.
 * This determines the actual template order for correct composition sequencing.
 */
onMounted(() => {
  if (props.renderOrder === undefined && orderMarker.value) {
    const parent = orderMarker.value.parentElement
    if (parent) {
      const siblings = parent.querySelectorAll(':scope > [data-shader-id]')
      const position = Array.from(siblings).indexOf(orderMarker.value)
      if (position >= 0) {
        detectedRenderOrder = position
        parentMetadataUpdate(instanceId, { renderOrder: position } as NodeMetadata)
      }
    }
  }
})

/**
 * Setup uniform watchers with reactive tracking.
 * Create a dedicated effect scope to ensure these watchers stay active and can be cleaned up.
 * Transforms all prop values to uniform equivalent before updating
 */
const setupUniformWatchers = () => {
  const scope = effectScope()
  scope.run(() => {
    Object.entries(uniforms).forEach(([propName, entry]) => {
      watch(() => (props as any)[propName], (newValue) => {
        // GPU bridge entries carry `.value` (no v1 `.uniform` node). colorStops expands into extra
        // array fields (colorsArray/…) that aren't component props — their `props[name]` never
        // changes, so their watcher never fires. Forward the raw value; renderer applies the transform.
        if (entry && entry.value !== undefined && !isPropDriver(newValue)) {
          // PropDriver values go to metadata.maps, not uniforms
          parentUniformUpdate(instanceId, propName, newValue)
        }
      }, {deep: true})
    })
    watch(() => [props.blendMode, props.opacity, props.visible, props.id, props.maskSource, props.maskType, props.renderOrder, props.transform, props.boundingBox, props.flow, props.absolute, mapsFromProps.value],
        ([blendMode, opacity, visible]) => {
          parentMetadataUpdate(instanceId, {
            blendMode,
            opacity,
            visible: visible === false ? false : true,
            id: props.id,
            mask: props.maskSource ? {
              source: props.maskSource,
              type: props.maskType || 'alpha'
            } : undefined,
            maps: mapsFromProps.value,
            renderOrder: props.renderOrder ?? detectedRenderOrder,
            transform: effectiveTransform.value,
            boundingBox: effectiveBoundingBox.value,
            flow: props.flow,
            absolute: props.absolute
          } as NodeMetadata)
        }, {deep: true})
    
    // Watch for renderer resets and register when they happen
    if (rendererResetSignal) {
      watch(() => rendererResetSignal.value, (newValue) => {
        if (newValue > 0) {
          registerWithRenderer()
        }
      }, { immediate: true })
    }
  })
  onBeforeUnmount(() => {
    scope.stop()
  })
}

// For capturesDOM shaders, defer watcher setup to onMounted so the canvas ref
// is available when the renderer reset signal fires and registerWithRenderer() runs.
if (isCapturesDOM) {
  onMounted(() => { setupUniformWatchers() })
} else {
  setupUniformWatchers()
}

// Clean up node from registry when component is unmounted
onBeforeUnmount(() => {
  parentRegister!(instanceId, null, null, null, null, null)
})
</script>

<template>
  <span ref="orderMarker" style="display:contents" :data-shader-id="instanceId">
    <slot v-if="!isCapturesDOM"></slot>
  </span>
  <Teleport v-if="isCapturesDOM" to="body">
    <canvas ref="captureCanvas" v-bind="{'layoutsubtree': ''}"
            :width="captureW" :height="captureH" :style="captureStyle">
      <slot></slot>
    </canvas>
  </Teleport>
</template>