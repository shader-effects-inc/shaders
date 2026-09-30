/**
 * Shaders © Shader Effects, Inc.
 *
 * Released under the MIT License — see the LICENSE file at the root of the repository.
 */

// WebGPU renderer (TypeGPU) — the runtime path the framework engines, js createShader, and
// presetRenderer register against.
export { shaderRendererGPU } from './gpu/index'
// Every renderer shares ONE lazily-created GPU device per page (see gpu/root.ts). Nothing
// needs to opt in; this is the escape hatch for hosts that want to reclaim it explicitly.
export { destroyDefaultGpuDevice } from './gpu/root'
export { createGpuUniformsMap } from './gpu/uniformBridge'
export { rootPassthrough } from './gpu/porters'
export type { GpuUniformsMap, GpuUniformInput } from './gpu/index'
export type { GpuShaderDefinition } from './gpu/contract'

// WebGPU availability + the quiet-by-default diagnostics channel. Shaders is WebGPU-only,
// so hosts need a way to ask "can this browser run me?" before committing to a layout, and
// a way to turn our (otherwise silent) internal logging back on while debugging.
export {
  isWebGPUSupported,
  getWebGPUSupport,
  setShadersDebug,
  isShadersDebug,
  GpuUnavailableError,
  isGpuUnavailableError,
  // @internal — the channel the framework packages log through. Silent unless debugging
  // is switched on; not part of the supported surface for application code.
  debugWarn,
  debugError
} from './gpu/support'
export type { GpuFailureReason, WebGPUSupportInfo } from './gpu/support'

// Runtime registry for user-defined components (`defineShader` + `<CustomShader>`), so
// name-keyed surfaces (presets, export, hosts) can find them next to the library's shaders.
export {registerShader, unregisterShader, getRegisteredShader, getRegisteredShaders, onShaderRegistered} from './customShaders'
export {authorError, hasCustomWgsl} from './gpu/support'

// Preset Renderer
export { createRendererFromJSON } from './presetRenderer'
export type {
  PresetConfig,
  ComponentConfig,
  PresetRendererOptions,
  GPUContext
} from './presetRenderer'

// Utilities
export { resolveMaskDependencies } from './maskResolution'
export { screenUVToBoxLocal, boxLocalToScreenUV } from './utilities/uvTransform'
export type { BoxLocalGeometry } from './utilities/uvTransform'
export { isDimensionalValue } from './utilities/dimensionalProps'
export { SHAPE3D_TYPES, SHAPE3D_DEFAULTS, SVG3D_TYPES, SVG3D_DEFAULTS, is3dShapeType, isSvg3dShapeType, isVolumetricShapeType, shape3dBoundingRadius } from './utilities/sdf3d'
export type { Shape3DType, Svg3DType } from './utilities/sdf3d'
export { resolveStops, sortStops, MAX_COLOR_STOPS } from './utilities/colorStops'
export type { ColorStop } from './utilities/colorStops'
export { resolveBoundingBox, DEFAULT_BOUNDING_BOX } from './utilities/boundingBox'
export { registerNaturalSize, getNaturalSize, onNaturalSizeChange } from './utilities/naturalSize'
export type { NaturalSize } from './utilities/naturalSize'
export { measureNode } from './utilities/measure'
export type { MeasureInput, MeasuredSize } from './utilities/measure'
export { computeLayout, computeTreeLayout, resolveLayoutConfig, layoutBlockSize, measureChild, anchorOffsetForBlockTopLeft, computeBoundsShift, computeGroupBlockRects } from './utilities/layout'
export { measureText, measureTextBlock, wrapLines } from './utilities/textMeasure'
export type { TextMeasureResult, TextBlockMeasure, TextMeasureSpec } from './utilities/textMeasure'
export type { LayoutItem, LayoutPlacement, LayoutNodeView, LayoutWrite, ResolvedLayout, GroupBlockRect } from './utilities/layout'

// Types
export type {UniformsMap, PropConfig, NodeMetadata, MaskConfig, MapConfig, MapChannel, PropDriver, MousePositionConfig, MouseMapConfig, AutoAnimateConfig, TransformConfig, ComponentDefinition, ToneMappingMode, BoundingBoxConfig, BoundingBoxDimension, BoundingBoxOrigin, DimensionalValue, BoundingBoxDeclaration, BoundingBoxAxisBinding, BoundingBoxBindingType, BoundingBoxPropBindings, LayoutConfig} from './types'
export type {BlendMode} from './types'
export type {PerformanceStats} from './performanceTracker'

// Note: Shader Registry is available at 'shaders-core/registry' to avoid bundling all shaders

