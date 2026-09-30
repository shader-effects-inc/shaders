/**
 * `@coreroot/std` — the standard authoring layer for shader definitions.
 *
 * A shader definition authored on std is declarative data built from these constructors;
 * `defineStd` lowers it onto the engine scaffolds and kit primitives and returns an
 * ordinary `GpuShaderDefinition` — registry generation, framework components, editor
 * metadata and presets are all untouched. GPU math lives in the kit behind the nouns; the
 * per-species L1 tiers (`pointwise`, `gather`, a warp's map fn, a generator's paint
 * builder) carry blessed bespoke bodies where the vocabulary does not reach yet.
 */
export {defineStd, defineShader} from './lower'
export {wgsl, WgslBody, isWgslBody, scanIdentifiers, wgslTypeForProp} from './wgsl'
export type {WgslSpec, WgslInput, WgslType, WgslHost} from './wgsl'
export {tintToward, pointwise, gather, displaceBy, paintThrough} from './filter'
export {radialMask} from './mask'
export {simulate, op, GridSim} from './sim'
export {pointer, pointerSpeed} from './signal'
export {p, Scalar} from './values'
export {crosses, recompileWhen, isZero, isValue, allOf, identityWhenever} from './slots'
export {ctx, pointwiseOp, resolveArg, resolveScalar, uniformOf} from './invoke'

// WGSL data schemas for `extraFields` declarations (the one GPU-typed surface a
// definition legitimately carries).
export {d as schema} from '../gpu/kit/index'
export type {ArgSpec, CtxToken} from './invoke'

// Warp map combinators (coordinate-space partial application and hard piecewise selection).
export {lerpToIdentity, selectMap} from '../gpu/scaffolds/uvRemapShader'
export type {UvMap, UvMapResult, UvMapSource, UvRemapHookParams} from '../gpu/scaffolds/uvRemapShader'
export type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
export type {RttFilterParams} from '../gpu/scaffolds/rttFilter'
export type {SdfShapeShaderSpec, SdfShapeBounds} from '../gpu/scaffolds/sdfShape'

export type {
    StdDefinition,
    StdDefinitionBase,
    StdPointwiseFilterDefinition,
    StdGatherFilterDefinition,
    StdWarpDefinition,
    StdShapeDefinition,
    StdGeneratorDefinition,
    StdCustomDefinition,
    StdProps,
    StdPropConfig,
    PointwiseEffect,
    GatherEffect,
    RecompileRule,
    IdentityRule,
} from './types'
export type {TintTowardEffect, DisplaceByEffect} from './filter'
export type {PropRef, ScalarInput, ScalarSource} from './values'
export type {GridSimConfig, SimOutputRef, GridStepOp, GridDeriveOp} from './sim'
export type {PointerSignal, PointerSpeedSignal} from './signal'

// ── Vocabulary namespaces (public alpha) ────────────────────────────────────────────────
// The words a definition composes with. Grouped so an author can browse them from one import:
// `import {math, paint, shape} from 'shaders/std'`.
export * as math from './math'
export * as shape from './shape'
export * as paint from './paint/fields'
export * as gradients from './paint/gradients'
export * as patterns from './paint/patterns'
export * as light from './paint/light'
export * as noise from './paint/noise'
export * as materials from './paint/materials'
export * as volume from './paint/volume'
export * as compose from './paint/compose'
export * as figures from './paint/figures'
export * as media from './paint/media'
export * as radiance from './paint/radiance'
export * as voxels from './paint/voxels'
export * as frames from './frames'
export * as warps from './warps'
export * as motion from './motion'
export * as mask from './mask'
export * as signal from './signal'
export * as effects from './effects/index'
export * as sim from './sim/index'
export {layered, layers} from './paint/compose'
export {paintFrame, resolveArgIn} from './invoke'

// List (array) props: declare with `listPropConfig`, read with `listOf` + `accumulate`.
export {listOf, accumulate} from './lists'
export {listPropConfig} from '../utilities/listProps'
export type {ListPropSpec} from '../utilities/listProps'

// The per-node animated clock (declared with `animatedTime: {speed: 'prop'}`), for paints that
// drive their own motion rather than reading a word that already does.
export {animatedTime} from '../gpu/porters'
// The standard multi-stop palette prop (`stops`), read by `paint.standardPalette` next to
// `colorA` / `colorB` / `colorSpace`.
export {colorStopsPropConfig} from '../utilities/colorStops'
export type {ColorStop} from '../utilities/colorStops'

// Prop transforms — the `transform:` values a prop config names (colors, positions, enums).
export {
    transformColor,
    transformPosition,
    transformColorSpace,
    colorSpaceOptions,
    transformBoolean,
    transformAngle,
    transformEdges,
} from '../utilities/transformations'

// The runtime registry for user-defined components (what `<CustomShader>` registers into).
export {registerShader, unregisterShader, getRegisteredShader, getRegisteredShaders, onShaderRegistered} from '../customShaders'

export type {GpuShaderDefinition as ShaderDefinition, GpuShaderDefinition, GpuFragmentParams, Expr} from '../gpu/contract'
export type {PropConfig, PropUIConfig, ComponentProps} from '../types'
export type {StdWgslFilterDefinition} from './types'
