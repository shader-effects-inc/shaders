/**
 * The shape of a definition object: what `defineShader` accepts, one type per kind of shader.
 *
 * Every definition has a `name`, `props` and optional metadata. The field that carries the
 * per-pixel work picks the type: `paint:` (a generator), `effect:` (a filter), `map:` (a
 * distortion), `shape:` (a 2D shape) or `gpu:` (a raw builder). A prop config is the usual
 * one (`default`, `transform`, `ui`, `compileTime`) with a `recompile:` rule in place of a
 * hand-written predicate.
 */
// Maintainer notes: a shader definition authored on std is declarative data built from the
// constructors in this package; `defineStd` (see `lower.ts`) lowers it onto the engine
// scaffolds and kit primitives and returns an ordinary `GpuShaderDefinition`, so every
// downstream surface (registry generation, framework components, editor metadata, presets)
// is untouched. Two rules this file enforces: recompile behavior is DECLARED, never a
// function (`StdPropConfig` omits `compileTimeWhen` and offers `recompile:` rules, see
// `slots.ts`); and the role/species determine the engine mechanics (composer flags, alpha
// discipline, identity-bypass shape) — authors state intent, never plumbing.
import type {ComponentProps, PropConfig, BoundingBoxDeclaration} from '../types'
import type {Expr, GpuComputeNode, GpuFragmentParams, GpuMapSampleUVs, GpuUvRemap} from '../gpu/contract'
import type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import type {RttFilterParams} from '../gpu/scaffolds/rttFilter'
import type {UvMapSource, UvRemapEdgeSource, UvRemapHookParams} from '../gpu/scaffolds/uvRemapShader'
import type {SdfShapeShaderSpec} from '../gpu/scaffolds/sdfShape'
import type {WgslBody} from './wgsl'

// ── Slot classification ─────────────────────────────────────────────────────────────────

/**
 * When a prop change rebuilds the shader: `crosses(value)` at a threshold, or `recompileWhen(fn)` by your own test.
 */
export type RecompileRule =
    | {readonly kind: 'crosses'; readonly value: number}
    // The loose fn signature mirrors PropConfig.compileTimeWhen.
    | {readonly kind: 'custom'; readonly predicate: (prev: any, next: any) => boolean}

/**
 * One prop of a definition: `default`, optional `transform`, `ui`, `description`, `compileTime`, and a `recompile:` rule.
 */
// The existing `PropConfig` surface with the function-valued `compileTimeWhen` escape hatch
// replaced by the declarative `recompile` rule.
export type StdPropConfig<V> = Omit<PropConfig<V>, 'compileTimeWhen'> & {
    recompile?: RecompileRule
}

/** The `props` object of a definition: one `StdPropConfig` per prop name. */
export type StdProps<T extends ComponentProps> = {[K in keyof T]: StdPropConfig<T[K]>}

// ── Identity rules ──────────────────────────────────────────────────────────────────────

/**
 * When a filter does nothing and can be skipped: built with `isZero`, `isValue`, `allOf` or `identityWhenever`.
 */
// Lowered to the filter scaffolds' `FilterIdentity`, which owns the map-driver bypass refusal
// — a prop the rule reads that carries a map driver disables the bypass automatically.
export type IdentityRule =
    | {readonly kind: 'isZero'; readonly prop: string}
    | {readonly kind: 'isValue'; readonly prop: string; readonly value: unknown}
    | {readonly kind: 'allOf'; readonly rules: IdentityRule[]}
    | {readonly kind: 'custom'; readonly props: string[]; readonly when: (values: Record<string, unknown>) => boolean}

// ── Filter effects ──────────────────────────────────────────────────────────────────────

/**
 * A filter that edits the layer inside one pixel at a time: what the color words (`saturate`, `hueRotate`, …) and `pointwise({build})` return.
 *
 * Give either `body` (a compiled GPU function taking the child's color first, then `args`)
 * or `build` (a function of the params that returns the new color from `params.childNode`).
 * The child arrives in straight alpha and there is no extra render pass.
 */
// `hint` lands in the emitted WGSL and is part of the shader's snapshot contract. A `body`
// picker reading compile-time `propValues` serves shaders with a body PAIR (Tint's variants).
export interface PointwiseEffect {
    readonly kind: 'pointwise'
    readonly body?: {fn: unknown; hint: string} | ((propValues: Record<string, unknown>) => {fn: unknown; hint: string})
    /** Build the new color yourself from `params.childNode`. Give exactly one of `body` or `build`. */
    readonly build?: (params: FilterParams) => Expr
    /** The values that follow the child's color as `body`'s arguments. */
    readonly args?: (params: FilterParams) => Expr[]
    /** Post-process the result (a color-space mix, a palette lookup). */
    readonly compose?: (result: Expr, params: FilterParams) => Expr
    /** Runs once per build, only when the filter is not skipped (for per-frame driven values). */
    readonly setup?: (params: FilterParams) => void
}

/**
 * A filter that reads the layer inside anywhere, not just at this pixel: blurs, halftones, displacements.
 *
 * The child is rendered to a texture first (`params.texture`, premultiplied) and
 * `params.sampleStraight(uv)` reads it back in straight alpha. Return premultiplied color
 * and it is unpremultiplied for you; say `resultAlpha: 'straight'` when you already did.
 */
// Identity bypasses to a centre sample — never to the raw child (the species' alpha
// discipline, owned by `defineRttFilter`, invisible to authors).
export interface GatherEffect {
    readonly kind: 'gather'
    /** Build the new color from the child texture. */
    readonly build: (params: RttFilterParams) => Expr
    readonly resultAlpha?: 'premultiplied' | 'straight'
    readonly setup?: (params: RttFilterParams) => void
}

/** A filter that moves the layer inside by a simulation's vector field. See `displaceBy`. */
export interface DisplaceByEffectRef {
    readonly kind: 'displaceBy'
}

// ── Definitions ─────────────────────────────────────────────────────────────────────────

/**
 * The fields every definition shares: `name`, `props`, optional `description` and `category`, the `animatedTime` clock, and engine flags.
 *
 * `animatedTime: {speed: 'speed'}` gives the layer its own clock scaled by that prop.
 * `usesPointer` asks for mouse tracking, `acceptsOptionalChild` lets a generator take a
 * nested layer, `extraFields` declares per-frame values you compute on the CPU. Most
 * definitions need only the first three fields.
 */
// The optional flags are the declarative half of the engine contract (passthrough to
// `GpuShaderDefinition`); species may derive some of them (a filter never declares
// `requiresChild`), while the custom tier states them explicitly.
export interface StdDefinitionBase<T extends ComponentProps> {
    name: string
    category?: string
    description?: string
    deprecatedNames?: string[]
    boundingBoxDeclaration?: BoundingBoxDeclaration
    props: StdProps<T>

    // Declarative engine contract (passthrough).
    usesPointer?: boolean
    acceptsUVContext?: boolean
    acceptsOptionalChild?: boolean
    blendWithChildren?: boolean
    capturesDOM?: boolean
    wantsBoundsParams?: boolean
    providesUVContextViaCompute?: boolean
    naturalSizeKey?: {fromProp: string} | {fixed: string}
    experimental?: ComponentProps
    animatedTime?: {speed: string}
    extraAnimatedTimes?: Record<string, string>
    extraFields?: Record<string, {schema: import('typegpu/data').AnyWgslData; initial: number | number[]}>

    /** A compute pass that runs before the pixel work (simulation state, a prepass). */
    compute?: GpuComputeNode
    /** Where a per-prop map driver samples its map (cell-centre sampling, for grids). */
    mapSampleUVs?: GpuMapSampleUVs
}

/**
 * A filter that edits the layer inside one pixel at a time: `effect:` holds a color word, `tintToward`, `pointwise({…})` or a `wgsl` body that reads `child`.
 *
 * `role` and `species` are optional; the `effect:` field already says filter, and its kind
 * says pointwise. `identityWhen` skips the filter while a rule holds. `missingChildMessage`
 * is shown when nothing is nested inside; omit it to fail silently with a transparent output.
 */
export interface StdPointwiseFilterDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'filter'
    species?: 'pointwise'
    effect: PointwiseEffect | import('./filter').TintTowardEffect | WgslBody
    /** Skip the filter while this rule holds (`isZero`, `isValue`, `allOf`, `identityWhenever`). */
    identityWhen?: IdentityRule
    /** Logged when nothing is nested inside. Omit to stay silent (the output is transparent). */
    missingChildMessage?: string
}

/**
 * A filter that reads the layer inside anywhere: `effect:` holds `gather({…})`, `displaceBy(…)` or a `wgsl` body that samples `childTexture`.
 *
 * Same optional fields as the pointwise form. The child gets its own render pass so the
 * filter can read neighbouring pixels.
 */
export interface StdGatherFilterDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'filter'
    species?: 'gather'
    effect: GatherEffect | import('./filter').DisplaceByEffect | WgslBody
    identityWhen?: IdentityRule
    missingChildMessage?: string
}

/**
 * A filter whose `effect:` is a `wgsl` body with no `species` declared: gather when the body samples `childTexture`, pointwise otherwise.
 *
 * Declare `species` to override what the body implies.
 */
export interface StdWgslFilterDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'filter'
    species?: undefined
    effect: WgslBody
    identityWhen?: IdentityRule
    missingChildMessage?: string
}

/**
 * A distortion: `map:` is one function from the pixel's coordinate to where in the layer inside to read.
 *
 * The map returns the source uv, or `{uv, coverage}` when parts of the output should be
 * transparent. `edges` says what happens past the layer's edge (`'prop'` reads an `edges`
 * prop, a number fixes a mode, `'none'` skips it). `resample` picks the filter used to read
 * the layer (`'catmullRom'` by default, `'bilinear'` for blocky or many-tap maps).
 */
// The lowering emits both engine paths from the one map — the render-to-texture fragment
// and the analytic fold — so they cannot drift.
export interface StdWarpDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'warp'
    map: UvMapSource
    /** Past the edge: `'prop'` (an `edges` prop, default), a fixed mode number, or `'none'`. */
    edges?: UvRemapEdgeSource
    /** How the layer is read back: `'catmullRom'` (default, sharp when magnified) or `'bilinear'`. */
    resample?: 'catmullRom' | 'bilinear'
    /** Treat the map as identity while this returns true (a texture it needs may not exist yet). */
    uvRemapIdentityWhen?: (params: UvRemapHookParams) => boolean
    missingChildMessage?: string
}

/** A 2D shape: `shape:` gives the distance function and the shape's own props; fill, stroke, softness and bounds come for free. */
export interface StdShapeDefinition {
    role?: 'shape'
    name: string
    description?: string
    category?: string
    shape: Omit<SdfShapeShaderSpec, 'name' | 'description' | 'category'>
}

/**
 * A generator: `paint:` produces a color from coordinates and needs nothing nested inside.
 *
 * `paint` is either a std composition (`rampOver(…)`, a builder function of the params) or
 * a `wgsl` body.
 */
export interface StdGeneratorDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'generator'
    /** A std composition or builder, or a `wgsl` body, returning the pixel's color. */
    paint: ((params: GpuFragmentParams) => Expr) | WgslBody
}

/**
 * A raw builder: `gpu:` holds a fragment function (and optionally a `uvRemap`) for effects the words cannot say yet.
 *
 * `role` here is only a label. Set `requiresChild` and `requiresRTT` yourself.
 */
// The custom tier — simulations with bespoke render paths, materials, media, structural
// nodes. The declarative half stays declared on the base; the GPU half is quarantined in
// `gpu:`.
export interface StdCustomDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    /** A label for the kind of thing this is. Nothing reads it. */
    role?: 'simulation' | 'shapeEffect' | 'media' | 'structural' | 'filter' | 'generator' | 'overlay'
    species?: 'custom'
    requiresRTT?: boolean
    requiresChild?: boolean
    gpu: {
        fragment: (params: GpuFragmentParams) => Expr
        uvRemap?: GpuUvRemap
    }
}

/**
 * What `defineShader` accepts: one of the definition shapes, told apart by which field carries the per-pixel work.
 *
 * `paint:` is a generator, `effect:` a filter (pointwise or gather from the effect), `map:` a
 * distortion, `shape:` a 2D shape, `gpu:` a raw builder. `role` and `species` may be
 * declared but must agree with the field; a contradiction throws when the definition is
 * built.
 */
export type StdDefinition<T extends ComponentProps> =
    | StdPointwiseFilterDefinition<T>
    | StdGatherFilterDefinition<T>
    | StdWgslFilterDefinition<T>
    | StdWarpDefinition<T>
    | StdShapeDefinition
    | StdGeneratorDefinition<T>
    | StdCustomDefinition<T>
