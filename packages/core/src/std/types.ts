/**
 * std — core types for the standard authoring layer.
 *
 * A shader definition authored on std is declarative data built from the constructors in
 * this package; `defineStd` (see `lower.ts`) lowers it onto the engine scaffolds and kit
 * primitives and returns an ordinary `GpuShaderDefinition`, so every downstream surface
 * (registry generation, framework components, editor metadata, presets) is untouched.
 *
 * Two rules this file enforces:
 *  - Recompile behavior is DECLARED, never a function: `StdPropConfig` omits
 *    `compileTimeWhen` and offers `recompile:` rules instead (see `slots.ts`).
 *  - The role/species determine the engine mechanics (composer flags, alpha discipline,
 *    identity-bypass shape); authors state intent, never plumbing.
 */
import type {ComponentProps, PropConfig, BoundingBoxDeclaration} from '../types'
import type {Expr, GpuComputeNode, GpuFragmentParams, GpuMapSampleUVs, GpuUvRemap} from '../gpu/contract'
import type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import type {RttFilterParams} from '../gpu/scaffolds/rttFilter'
import type {UvMapSource, UvRemapEdgeSource, UvRemapHookParams} from '../gpu/scaffolds/uvRemapShader'
import type {SdfShapeShaderSpec} from '../gpu/scaffolds/sdfShape'
import type {WgslBody} from './wgsl'

// ── Slot classification ─────────────────────────────────────────────────────────────────

/**
 * A declarative recompile rule — the std replacement for a `compileTimeWhen` function.
 * `crosses(v)` recompiles only when the prop value crosses `v` (identity ↔ effect
 * boundaries that add or remove whole code paths); `recompileWhen(fn)` is the escape for
 * a genuinely bespoke predicate.
 */
export type RecompileRule =
    | {readonly kind: 'crosses'; readonly value: number}
    // The loose fn signature mirrors PropConfig.compileTimeWhen.
    | {readonly kind: 'custom'; readonly predicate: (prev: any, next: any) => boolean}

/**
 * A std prop config: the existing `PropConfig` surface (defaults, transforms, ui — the
 * public surface, unchanged) with the function-valued `compileTimeWhen` escape hatch
 * replaced by the declarative `recompile` rule.
 */
export type StdPropConfig<V> = Omit<PropConfig<V>, 'compileTimeWhen'> & {
    recompile?: RecompileRule
}

export type StdProps<T extends ComponentProps> = {[K in keyof T]: StdPropConfig<T[K]>}

// ── Identity rules ──────────────────────────────────────────────────────────────────────

/**
 * A declarative no-op condition. Lowered to the filter scaffolds' `FilterIdentity`, which
 * owns the map-driver bypass refusal — a prop the rule reads that carries a map driver
 * disables the bypass automatically.
 */
export type IdentityRule =
    | {readonly kind: 'isZero'; readonly prop: string}
    | {readonly kind: 'isValue'; readonly prop: string; readonly value: unknown}
    | {readonly kind: 'allOf'; readonly rules: IdentityRule[]}
    | {readonly kind: 'custom'; readonly props: string[]; readonly when: (values: Record<string, unknown>) => boolean}

// ── Filter effects ──────────────────────────────────────────────────────────────────────

/**
 * A pointwise filter effect: a pure fn(color, …) over the composed child — straight
 * alpha, no render-to-texture. The body is a blessed `'use gpu'` fn; `hint` lands in the
 * emitted WGSL and is part of the shader's snapshot contract.
 */
export interface PointwiseEffect {
    readonly kind: 'pointwise'
    readonly body?: {fn: unknown; hint: string} | ((propValues: Record<string, unknown>) => {fn: unknown; hint: string})
    /** Pure-Expr alternative to `body`: build the filtered color directly (`params.childNode`
     *  is the composed child). Exactly one of `body`/`build` must be given. */
    readonly build?: (params: FilterParams) => Expr
    /** Body args AFTER the child color (always first). */
    readonly args?: (params: FilterParams) => Expr[]
    /** Builder-level tail on the body's result (compile-time color-space mixes etc.). */
    readonly compose?: (result: Expr, params: FilterParams) => Expr
    /** Per-composition side effects (extraFields drivers). Non-identity path only. */
    readonly setup?: (params: FilterParams) => void
}

/**
 * A gather filter effect: samples the child's render-to-texture (neighbour taps, screens,
 * dithers). Taps are premultiplied; the species appends the unpremultiply tail unless the
 * build already returns straight alpha. Identity bypasses to a centre sample — never to
 * the raw child (the species' alpha discipline, owned here, invisible to authors).
 */
export interface GatherEffect {
    readonly kind: 'gather'
    /** Build the filtered color from the child texture (`params.texture` / `sampleStraight`). */
    readonly build: (params: RttFilterParams) => Expr
    readonly resultAlpha?: 'premultiplied' | 'straight'
    readonly setup?: (params: RttFilterParams) => void
}

/** See `filter.ts` — displace the child's sample coordinates by a simulation field. */
export interface DisplaceByEffectRef {
    readonly kind: 'displaceBy'
}

// ── Definitions ─────────────────────────────────────────────────────────────────────────

/**
 * Fields shared by every std definition. The optional flags are the declarative half of
 * the engine contract; species may derive some of them (a filter never declares
 * `requiresChild`), while the custom tier states them explicitly.
 */
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

    /** A compute pass (simulation state, prepasses). Part of the L1 tier. */
    compute?: GpuComputeNode
    /** Custom per-prop map-driver sample UVs (cell-centre sampling). */
    mapSampleUVs?: GpuMapSampleUVs
}

/**
 * A pointwise filter — inline color filter over the composed child. `role`/`species` are
 * optional: `effect:` says filter, and the effect's kind says pointwise.
 */
export interface StdPointwiseFilterDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'filter'
    species?: 'pointwise'
    effect: PointwiseEffect | import('./filter').TintTowardEffect | WgslBody
    /** Declared no-op condition; the lowering owns bypass + driver guards. */
    identityWhen?: IdentityRule
    /** User-facing missing-child message; omit for silent (returns transparent). */
    missingChildMessage?: string
}

/** A gather filter — samples the child's render-to-texture. `role`/`species` optional (inferred). */
export interface StdGatherFilterDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'filter'
    species?: 'gather'
    effect: GatherEffect | import('./filter').DisplaceByEffect | WgslBody
    identityWhen?: IdentityRule
    missingChildMessage?: string
}

/**
 * role 'filter' with a raw `wgsl` body and NO declared species — the species is inferred
 * from the body: `gather` when it samples `childTexture`, `pointwise` otherwise. Declare
 * `species` explicitly to override the inference.
 */
export interface StdWgslFilterDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'filter'
    species?: undefined
    effect: WgslBody
    identityWhen?: IdentityRule
    missingChildMessage?: string
}

/**
 * role 'warp' — a distortion expressed as ONE coordinate map (screen UV → source UV +
 * coverage). The lowering emits both engine paths from it — the render-to-texture
 * fragment and the analytic fold — so they cannot drift.
 */
export interface StdWarpDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'warp'
    map: UvMapSource
    /** Edge handling: `'prop'` (an `edges` prop, default) · a fixed mode · `'none'`. */
    edges?: UvRemapEdgeSource
    /** Reconstruction filter on the texture path. Default `'catmullRom'`. */
    resample?: 'catmullRom' | 'bilinear'
    /** Analytic-path identity bail-out (a compute texture that may not exist yet). */
    uvRemapIdentityWhen?: (params: UvRemapHookParams) => boolean
    missingChildMessage?: string
}

/** role 'shape' — an analytic 2D SDF shape (fill + stroke + soften + bounds). */
export interface StdShapeDefinition {
    role?: 'shape'
    name: string
    description?: string
    category?: string
    shape: Omit<SdfShapeShaderSpec, 'name' | 'description' | 'category'>
}

/**
 * role 'generator' — a source that paints from coordinates (no child required). `paint`
 * is the composition builder: the L1 tier for generators whose look is a blessed body fn.
 */
export interface StdGeneratorDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    role?: 'generator'
    /** The composition builder, or a raw `wgsl` body returning the pixel color. */
    paint: ((params: GpuFragmentParams) => Expr) | WgslBody
}

/**
 * The custom tier — the escape hatch for effects whose composition cannot yet be said in
 * vocabulary (simulations with bespoke render paths, materials, media, structural nodes).
 * The declarative half stays declared on the base; the GPU half is quarantined in `gpu:`.
 */
export interface StdCustomDefinition<T extends ComponentProps> extends StdDefinitionBase<T> {
    /** A label for the kind of thing this is (documentation; nothing downstream reads it). */
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
 * The std definition union — one member per role/species. The role is INFERRED from the
 * field that carries the GPU half: `paint:` → generator, `effect:` → filter (species from the
 * effect), `map:` → warp, `shape:` → shape, `gpu:` → custom. `role`/`species` may still be
 * declared; a declaration that contradicts the shape is an error at definition time.
 */
export type StdDefinition<T extends ComponentProps> =
    | StdPointwiseFilterDefinition<T>
    | StdGatherFilterDefinition<T>
    | StdWgslFilterDefinition<T>
    | StdWarpDefinition<T>
    | StdShapeDefinition
    | StdGeneratorDefinition<T>
    | StdCustomDefinition<T>
