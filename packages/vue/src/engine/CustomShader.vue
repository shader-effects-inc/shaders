<script setup lang="ts" generic="T extends Record<string, any> = Record<string, any>">
/**
 * <CustomShader :src="definition" …props />
 *
 * Mounts a user-defined shader (a `defineShader` result from `shaders/std`) as a node in the
 * shader tree — the same registration, uniform bridge, metadata and child slot the generated
 * library components use, with the definition supplied at runtime instead of imported at
 * build time. Shader props are passed as ordinary attributes (`:radius="0.5"`), typed from
 * the definition's props; the layer props (blendMode, opacity, transform, …) are declared.
 *
 * Swapping `src` for a new definition object re-registers the node in place (children stay
 * attached); the definition's `revision` makes the renderer recompose.
 */
import {watch, computed, inject, provide, onBeforeUnmount, onMounted, onUpdated, ref, effectScope, useId, useAttrs} from 'vue'
import {
  createGpuUniformsMap,
  resolveBoundingBox,
  registerShader,
  type GpuShaderDefinition,
  type GpuUniformsMap,
  type BlendMode,
  type NodeMetadata,
  type PropConfig,
  type MaskConfig,
  type PropDriver,
  type TransformConfig,
  type BoundingBoxConfig,
  type LayoutConfig
} from 'shaders-core'
import {setColorSpaceMode} from 'shaders-core/utilities/transformations'

defineOptions({name: 'CustomShader', inheritAttrs: false})

interface LayerProps {
  /** The shader definition (`defineShader({...})`). */
  src: GpuShaderDefinition<T>
  blendMode?: BlendMode
  opacity?: number
  visible?: boolean
  id?: string
  maskSource?: string
  maskType?: string
  renderOrder?: number
  transform?: Partial<TransformConfig>
  boundingBox?: Partial<BoundingBoxConfig>
  flow?: LayoutConfig
  absolute?: boolean
}

const props = withDefaults(defineProps<LayerProps>(), {
  blendMode: 'normal',
  visible: true
})

/** Shader props arrive as fallthrough attributes (any casing the template used). */
const attrs = useAttrs()

function isPropDriver(value: unknown): value is PropDriver {
  return typeof value === 'object' && value !== null && 'type' in value &&
    ((value as any).type === 'map' || (value as any).type === 'mouse' || (value as any).type === 'mouse-position' || (value as any).type === 'auto-animate')
}

const camelize = (key: string) => key.replace(/-(\w)/g, (_, c: string) => c.toUpperCase())

/** The definition's props → their current authored values (attrs over defaults), drivers included. */
function readShaderProps(definition: GpuShaderDefinition<T>): Record<string, unknown> {
  const authored: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(attrs)) authored[camelize(key)] = value
  const out: Record<string, unknown> = {}
  for (const [key, config] of Object.entries(definition.props as Record<string, PropConfig<unknown>>)) {
    const value = authored[key]
    out[key] = value === undefined ? config.default : value
  }
  return out
}

/** Uniform seed: authored values with drivers stripped (drivers ride in metadata.maps). */
function uniformValues(values: Record<string, unknown>, definition: GpuShaderDefinition<T>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(values)) {
    out[key] = isPropDriver(value) ? (definition.props as Record<string, PropConfig<unknown>>)[key]?.default : value
  }
  return out
}

function mapsOf(values: Record<string, unknown>): Record<string, PropDriver> | undefined {
  const maps: Record<string, PropDriver> = {}
  for (const [key, value] of Object.entries(values)) if (isPropDriver(value)) maps[key] = value
  return Object.keys(maps).length > 0 ? maps : undefined
}

const instanceId = (props.id ? props.id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || useId().replace(/[^a-zA-Z0-9_]/g, '')
provide('shaderParentId', instanceId)

const DEFAULT_TRANSFORM: TransformConfig = {
  offsetX: 0,
  offsetY: 0,
  rotation: 0,
  scale: 1,
  anchorX: 0.5,
  anchorY: 0.5,
  edges: 'transparent'
}
const effectiveTransform = computed<TransformConfig>(() => ({...DEFAULT_TRANSFORM, ...props.transform}))
const effectiveBoundingBox = computed<BoundingBoxConfig | undefined>(() => resolveBoundingBox(props.boundingBox))

const shaderColorSpace = inject<{value: 'p3-linear' | 'srgb'}>('shaderColorSpace')
const parentId = inject<string>('shaderParentId')
if (parentId === undefined) {
  throw new Error('<CustomShader> must be used inside a <Shader> component or another shader component')
}
const parentRegister = inject<(id: string, fragmentNodeFunc: any, parentId: string | null, metadata: NodeMetadata | null, uniforms: GpuUniformsMap | null, componentDefinition: any, domCanvas?: HTMLCanvasElement) => void>('shaderNodeRegister')
const parentUniformUpdate = inject<(nodeId: string, uniformName: string, value: any) => void>('shaderUniformUpdate')
const parentMetadataUpdate = inject<(nodeId: string, metadata: NodeMetadata) => void>('shaderMetadataUpdate')
if (!parentRegister || !parentUniformUpdate || !parentMetadataUpdate) {
  throw new Error('<CustomShader> must be used inside a <Shader> component or another shader component')
}
const rendererResetSignal = inject<{value: number}>('shaderRendererResetSignal')

// ── Live state ──────────────────────────────────────────────────────────────────────────

let definition: GpuShaderDefinition<T> = props.src
let currentValues = readShaderProps(definition)
let uniforms: GpuUniformsMap = buildUniforms()

function buildUniforms(): GpuUniformsMap {
  if (shaderColorSpace) setColorSpaceMode(shaderColorSpace.value)
  return createGpuUniformsMap(definition as any, uniformValues(currentValues, definition), instanceId)
}

const orderMarker = ref<HTMLSpanElement | null>(null)
let detectedRenderOrder: number | undefined = undefined

function metadata(): NodeMetadata {
  return {
    blendMode: props.blendMode,
    opacity: props.opacity,
    visible: props.visible === false ? false : true,
    id: props.id,
    mask: props.maskSource ? {source: props.maskSource, type: props.maskType || 'alpha'} as MaskConfig : undefined,
    maps: mapsOf(currentValues),
    renderOrder: props.renderOrder ?? detectedRenderOrder,
    transform: effectiveTransform.value,
    boundingBox: effectiveBoundingBox.value,
    flow: props.flow,
    absolute: props.absolute
  } as NodeMetadata
}

const registerWithRenderer = () => {
  registerShader(definition)
  parentRegister!(instanceId, definition.fragment, parentId, metadata(), uniforms, definition)
}

onMounted(() => {
  if (props.renderOrder === undefined && orderMarker.value) {
    const parent = orderMarker.value.parentElement
    if (parent) {
      const siblings = parent.querySelectorAll(':scope > [data-shader-id]')
      const position = Array.from(siblings).indexOf(orderMarker.value)
      if (position >= 0) {
        detectedRenderOrder = position
        parentMetadataUpdate!(instanceId, {renderOrder: position} as NodeMetadata)
      }
    }
  }
})

/**
 * Fallthrough attrs are not reactive, so shader-prop changes are picked up after each
 * re-render: diff against the last pushed values and forward only what changed.
 */
onUpdated(() => {
  const next = readShaderProps(definition)
  let mapsChanged = false
  for (const [key, value] of Object.entries(next)) {
    const prev = currentValues[key]
    if (value === prev) continue
    if (isPropDriver(value) || isPropDriver(prev)) mapsChanged = true
    else if (uniforms[key] && uniforms[key].value !== undefined) parentUniformUpdate!(instanceId, key, value)
  }
  currentValues = next
  if (mapsChanged) parentMetadataUpdate!(instanceId, metadata())
})

const scope = effectScope()
scope.run(() => {
  // A new definition object → new uniform map, re-register the same node id (children stay).
  watch(() => props.src, (next) => {
    if (!next || next === definition) return
    definition = next
    currentValues = readShaderProps(definition)
    uniforms = buildUniforms()
    registerWithRenderer()
  })

  watch(() => [props.blendMode, props.opacity, props.visible, props.id, props.maskSource, props.maskType, props.renderOrder, props.transform, props.boundingBox, props.flow, props.absolute],
    () => parentMetadataUpdate!(instanceId, metadata()), {deep: true})

  if (rendererResetSignal) {
    watch(() => rendererResetSignal.value, (value) => {
      if (value > 0) registerWithRenderer()
    }, {immediate: true})
  }
})

onBeforeUnmount(() => {
  scope.stop()
  parentRegister!(instanceId, null, null, null, null, null)
})
</script>

<template>
  <span ref="orderMarker" style="display:contents" :data-shader-id="instanceId">
    <slot></slot>
  </span>
</template>
