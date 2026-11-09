/**
 * `defineShader` — turn a plain definition object into a shader component.
 *
 * You describe the shader as data: a `name`, its `props`, an optional `animatedTime` clock,
 * and ONE field carrying the per-pixel work — `paint:` for a generator, `effect:` for a
 * filter over the layer inside it, `map:` for a distortion, `shape:` for a 2D shape, `gpu:`
 * for a raw builder. Blend modes, opacity, masks, transforms, prop drivers and export are
 * handled by the engine. The result mounts as `<CustomShader src={…}>` in any framework, or
 * by name in preset JSON.
 */
// Maintainer notes — the lowering: StdDefinition → GpuShaderDefinition.
//
// Noun effects lower to kit primitives (the normative GPU bodies) wired through the engine
// scaffolds; the L1 tiers pass their blessed bodies straight through, emitting the same calls
// under the same hints so a port leaves the compiled WGSL untouched. Either way the output is
// an ordinary `GpuShaderDefinition` — every downstream surface (registry, generated
// components, editor metadata, presets) is unchanged.
import type {ComponentProps, PropConfig} from '../types'
import type {GpuShaderDefinition} from '../gpu/contract'
// Through the kit barrel: colorMixing and colorStops import each other, and the barrel fixes their
// evaluation order.
import {colorMixing} from '../gpu/kit/index'
import {definePointwiseFilter, type FilterIdentity} from '../gpu/scaffolds/pointwiseFilter'
import {defineRttFilter} from '../gpu/scaffolds/rttFilter'
import {uvRemapShader} from '../gpu/scaffolds/uvRemapShader'
import {defineSdfShapeShader} from '../gpu/scaffolds/sdfShape'
import type {
    IdentityRule,
    StdCustomDefinition,
    StdDefinition,
    StdGatherFilterDefinition,
    StdGeneratorDefinition,
    StdPointwiseFilterDefinition,
    StdPropConfig,
    StdProps,
    StdShapeDefinition,
    StdSurfaceDefinition,
    StdWarpDefinition,
    StdWgslFilterDefinition,
    StdWgslShapeDefinition,
} from './types'
import {resolveScalar, uniformOf} from './invoke'
import {builtinExtraFields, BUILTIN_FIELDS, isWgslBody, lowerWgsl, lowerWgslFunction, scanIdentifiers, type WgslBody, type WgslHostDefinition} from './wgsl'
import {animatedTime} from '../gpu/porters'
import {Expr} from '../gpu/contract'
import type {UvMapSource} from '../gpu/scaffolds/uvRemapShader'
import {codeIn, inputsCompute, planInputs, type InputPlan} from './inputs/graph'
import {LOWER_DISPLACE, type DisplaceLowering} from './displaceSymbol'
import {identifiersIn} from './wgslSource'
import {Scalar} from './values'
import {shapedSurface} from './paint/shapedSurface'
import {shapeEffectBoundingBoxDeclaration} from '../utilities/shapeEffectBounds'
import {colorSpaceOptions, selectIndex, transformBool01, transformColor, transformColorSpace, transformPosition} from '../utilities/transformations'
import {listPropTransform} from '../utilities/listProps'
import {colorStopsTransform} from '../utilities/colorStops'

// ── Props ───────────────────────────────────────────────────────────────────────────────

/** `recompile:` rules → the equivalent `compileTimeWhen` predicate. */
function lowerProps<T extends ComponentProps>(props: StdProps<T>): GpuShaderDefinition<T>['props'] {
    const lowered: Record<string, PropConfig<unknown>> = {}
    for (const key of Object.keys(props) as (keyof T & string)[]) {
        const {recompile, ...rest} = props[key] as StdPropConfig<unknown>
        const config: PropConfig<unknown> = {...rest}
        if (recompile?.kind === 'crosses') {
            const boundary = recompile.value
            config.compileTimeWhen = (prev: unknown, next: unknown) =>
                (prev === boundary) !== (next === boundary)
        } else if (recompile?.kind === 'custom') {
            config.compileTimeWhen = recompile.predicate
        }
        lowered[key] = config
    }
    return lowered as GpuShaderDefinition<T>['props']
}

// ── Identity rules ──────────────────────────────────────────────────────────────────────

interface LoweredIdentity {
    props: string[]
    when: (values: Record<string, unknown>) => boolean
}

function lowerIdentityRule(
    rule: IdentityRule,
    defaults: Record<string, unknown>,
): LoweredIdentity {
    switch (rule.kind) {
        case 'isZero':
        case 'isValue': {
            const target = rule.kind === 'isZero' ? 0 : rule.value
            const declared = defaults[rule.prop]
            return {
                props: [rule.prop],
                when: (values) => (values[rule.prop] ?? declared) === target,
            }
        }
        case 'allOf': {
            const lowered = rule.rules.map((r) => lowerIdentityRule(r, defaults))
            return {
                props: [...new Set(lowered.flatMap((l) => l.props))],
                when: (values) => lowered.every((l) => l.when(values)),
            }
        }
        case 'custom':
            return {props: rule.props, when: rule.when}
    }
}

/**
 * `identityWhen:` → the scaffolds' `FilterIdentity`. The scaffold owns the map-driver
 * bypass refusal; unset values fall back to each prop's declared default.
 */
function lowerIdentity<T extends ComponentProps>(
    rule: IdentityRule | undefined,
    props: StdProps<T>,
): FilterIdentity | undefined {
    if (!rule) return undefined
    const defaults: Record<string, unknown> = {}
    for (const [key, config] of Object.entries(props as Record<string, StdPropConfig<unknown>>)) {
        defaults[key] = config.default
    }
    return lowerIdentityRule(rule, defaults)
}

// ── Species lowerings ───────────────────────────────────────────────────────────────────

function lowerPointwiseFilter<T extends ComponentProps>(definition: StdPointwiseFilterDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, effect, identityWhen, missingChildMessage, props, ...meta} = definition
    const shared = {
        ...meta,
        props: lowerProps(props),
        identity: lowerIdentity(identityWhen, props),
        missingChildMessage,
    }
    if (effect.kind === 'tintToward') {
        const {color, amount} = effect
        return definePointwiseFilter<T>({
            ...shared,
            body: {fn: colorMixing.mixToward, hint: 'mixToward'},
            args: (params) => [uniformOf(color, params), resolveScalar(amount, params)],
        })
    }
    if (isWgslBody(effect)) {
        return definePointwiseFilter<T>({
            ...shared,
            build: lowerWgsl(effect, 'pointwise', wgslHost(definition)),
        })
    }
    const {kind: _kind, ...effectConfig} = effect
    return definePointwiseFilter<T>({...shared, ...effectConfig})
}

/** The inputs and helpers of the definition being lowered (set around `lowerStd`, which is synchronous). */
let lowering: {plan: InputPlan; helpers: string | null} | null = null

/** What a `wgsl` body needs from its definition to bind props, the time clock, inputs and helpers. */
function wgslHost<T extends ComponentProps>(definition: {name: string; props: StdProps<T>; animatedTime?: {speed: string}}): WgslHostDefinition {
    return {
        name: definition.name,
        props: definition.props as unknown as Record<string, PropConfig<unknown>>,
        animatedTime: definition.animatedTime,
        ...(lowering ? {inputs: lowering.plan, helpers: lowering.helpers} : {}),
    }
}


function lowerGatherFilter<T extends ComponentProps>(definition: StdGatherFilterDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, effect, identityWhen, missingChildMessage, props, ...meta} = definition
    if (effect.kind === 'displaceBy') {
        // The lowering travels with the word (std/displaceBy.ts), so the wave-field runtime is only
        // bundled into apps that use `displaceBy`.
        const lower = (effect as unknown as Record<symbol, unknown>)[LOWER_DISPLACE] as DisplaceLowering
        return lower(definition, {lowerProps, lowerIdentity}) as GpuShaderDefinition<T>
    }
    if (isWgslBody(effect)) {
        return defineRttFilter<T>({
            ...meta,
            props: lowerProps(props),
            identity: lowerIdentity(identityWhen, props),
            missingChildMessage,
            build: lowerWgsl(effect, 'gather', wgslHost(definition)),
            // The body's own spec, else the definition's `alpha` (what an agent-format file writes).
            resultAlpha: effect.spec.alpha ?? (definition as {alpha?: 'premultiplied' | 'straight'}).alpha ?? 'premultiplied',
        })
    }
    const {kind: _kind, ...effectConfig} = effect
    return defineRttFilter<T>({
        ...meta,
        props: lowerProps(props),
        identity: lowerIdentity(identityWhen, props),
        missingChildMessage,
        ...effectConfig,
    })
}

function lowerWarp<T extends ComponentProps>(definition: StdWarpDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, map: mapField, edges, resample, uvRemapIdentityWhen, missingChildMessage, props, ...meta} = definition
    const map = isWgslBody(mapField) ? wgslMap(mapField, definition) : mapField
    return {
        ...meta,
        props: lowerProps(props),
        requiresRTT: true,
        requiresChild: true,
        ...uvRemapShader(map, {
            edges,
            resample,
            uvRemapIdentityWhen,
            requireChildMessage: missingChildMessage,
        }),
    } as GpuShaderDefinition<T>
}

function lowerShape<T extends ComponentProps>(definition: StdShapeDefinition | StdWgslShapeDefinition<T>): GpuShaderDefinition<T> {
    if (isWgslBody(definition.shape)) return lowerWgslShape(definition as StdWgslShapeDefinition<T>)
    const {role: _role, name, description, category, shape} = definition as StdShapeDefinition
    return defineSdfShapeShader<T>({name, description, category, ...shape})
}

/** Read a prop for a scalar-slot body: positions back in uv space (the stored y is flipped). */
function propValue(props: Record<string, PropConfig<unknown>>, uniforms: Record<string, Expr>, name: string): Expr | null {
    const accessor = uniforms[name]
    if (!accessor) return null
    if (props[name]?.transform !== (transformPosition as unknown)) return accessor
    return new Expr((ctx) => `vec2f(${accessor.member('x')._emit(ctx)}, 1.0 - ${accessor.member('y')._emit(ctx)})`)
}

/** A `shape:` body (signed distance at `p`) on the engine's 2D-shape scaffold. */
function lowerWgslShape<T extends ComponentProps>(definition: StdWgslShapeDefinition<T>): GpuShaderDefinition<T> {
    const {name, description, category, props, bounds, rotatable} = definition
    const host = wgslHost(definition)
    const distance = lowerWgslFunction(definition.shape, host, {name: 'shape', args: [{name: 'p', type: 'vec2f'}], ret: 'f32'})
    const shapeProps = props as unknown as Record<string, PropConfig<unknown>>
    return defineSdfShapeShader<T>({
        name,
        description,
        category,
        distance: ({x, y}, u) => distance(
            [new Expr((ctx) => `vec2f(${x._emit(ctx)}, ${y._emit(ctx)})`)],
            (n) => (n === 'time' || n === 'aspect' ? null : shapeProps[n] ? u(n) : null),
        ),
        colorDescription: 'Fill color of the shape',
        centerDescription: 'Center position of the shape',
        bounds: bounds ?? {custom: {aspectRatio: null}},
        shapeProps: shapeProps as never,
        ...(rotatable === false ? {rotatable: false} : {}),
    })
}

/** A `map:` body (the uv to read at) as a UV map: it composes analytically with other warps. */
function wgslMap<T extends ComponentProps>(body: WgslBody, definition: StdWarpDefinition<T>): UvMapSource {
    const host = wgslHost(definition)
    const fn = lowerWgslFunction(body, host, {name: 'map', args: [{name: 'uv', type: 'vec2f'}], ret: 'vec2f'})
    const props = definition.props as unknown as Record<string, PropConfig<unknown>>
    return (hook) => ({
        map: (uv, aspect) => fn([uv], (n) => {
            if (n === 'aspect') return aspect
            if (n === 'time') return definition.animatedTime ? animatedTime(hook as never) : null
            return propValue(props, hook.uniforms, n)
        }),
    })
}

function lowerGenerator<T extends ComponentProps>(definition: StdGeneratorDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, paint, props, ...meta} = definition
    if (isWgslBody(paint)) {
        // A wgsl body already reads the distorted UV when a parent supplies one; accepting UV
        // context by default is what makes that happen inside a library distortion, and lets a
        // bounding box re-fit the paint to the box rather than clip it.
        return {
            acceptsUVContext: true,
            boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
            ...meta,
            props: lowerProps(props),
            fragment: lowerWgsl(paint, 'generator', wgslHost(definition)),
        }
    }
    return {
        ...meta,
        props: lowerProps(props),
        fragment: paint,
    }
}

/** The props every shape material has; the editor's shape picker writes `shape` (an SVG's `svgUrl` rides in it) and `shapeType`. */
const SURFACE_PROPS: Record<string, StdPropConfig<unknown>> = {
    shape: {
        default: JSON.stringify({type: 'circleSDF', radius: 0.35}),
        description: 'The shape the material covers: a flat shape, an SVG or a 3D shape',
        ui: {type: 'shape', label: 'Shape', group: 'Shape'},
    },
    origin: {
        default: 'center',
        description: 'Reference edge the center position is measured from',
        ui: {type: 'origin', label: 'Origin', group: 'Position'},
    },
    center: {
        default: {x: 0.5, y: 0.5},
        transform: transformPosition as (value: unknown) => unknown,
        description: 'Center of the shape',
        ui: {type: 'position', label: 'Center', group: 'Position', units: ['%', 'px']},
    },
    scale: {
        default: 1,
        description: 'Size of the shape (1 = default)',
        ui: {type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position'},
    },
    rotation: {
        default: 0,
        description: 'Rotation of the shape in degrees',
        ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position'},
    },
    shapeSdfUrl: {
        default: '',
        compileTime: true,
        description: 'Legacy: URL of a pre-generated SDF .bin for an SVG shape. SVG shapes are generated in real time from the svgUrl in shape; this is only the fallback for older presets',
    },
    shapeType: {
        default: '',
        compileTime: true,
        description: 'The active shape type; set by the shape picker',
    },
}

function lowerSurface<T extends ComponentProps>(definition: StdSurfaceDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, surface, props, ...meta} = definition
    const readsChild = surface.identifiers.has('child') || surface.identifiers.has('childTexture')
    const build = lowerWgsl(surface, 'surface', wgslHost(definition))
    const material = shapedSurface({
        stencil: 'centre',
        ...(readsChild ? {child: 'required' as const} : {}),
        surface: (frame, params) => build(params, frame),
    })
    return {
        boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
        ...meta,
        props: lowerProps(props),
        extraFields: {...material.extraFields, ...(meta.extraFields ?? {})},
        compute: material.compute,
        fragment: material.gpu.fragment,
        ...(readsChild ? {requiresRTT: true, requiresChild: true} : {}),
    } as GpuShaderDefinition<T>
}

/** The `wgsl` body a definition carries, if any. */
function wgslBodyOf(definition: Record<string, unknown>): WgslBody | null {
    for (const field of ['paint', 'effect', 'surface']) {
        if (isWgslBody(definition[field])) return definition[field] as WgslBody
    }
    return null
}

/**
 * A prop with no `transform` gets the one its control implies, so a definition written as plain
 * data (an agent's file, the stored IR) needs no transform imports: a colour control parses
 * colours, a position control positions, a checkbox becomes 0/1, a select its option index.
 */
function withInferredTransforms<T extends ComponentProps>(definition: StdDefinition<T>): StdDefinition<T> {
    const record = definition as unknown as Record<string, unknown>
    // Only definitions written in WGSL (the agent format): library shaders state their transforms,
    // and some read a select's string value on purpose.
    const wgslWritten = ['paint', 'effect', 'surface', 'map', 'shape'].some((f) => isWgslBody(record[f]))
    if (!record.props || !wgslWritten) return definition
    let changed = false
    const props: Record<string, StdPropConfig<unknown>> = {...(record.props as Record<string, StdPropConfig<unknown>>)}
    for (const [name, config] of Object.entries(props)) {
        if (!config || config.transform) continue
        const type = Array.isArray(config.ui?.type) ? config.ui?.type[0] : config.ui?.type
        let transform: ((v: unknown) => unknown) | undefined
        if (type === 'color') transform = transformColor as (v: unknown) => unknown
        else if (type === 'position') transform = transformPosition as (v: unknown) => unknown
        else if (type === 'checkbox') transform = transformBool01
        else if (type === 'gradient-stops') transform = colorStopsTransform as (v: unknown) => unknown
        else if (type === 'list') transform = listPropTransform as (v: unknown) => unknown
        else if (type === 'select') {
            const options = (config.ui as {options?: {value: unknown}[]} | undefined)?.options ?? []
            const values = options.map((o) => o.value)
            if (values.length && values.every((v) => typeof v === 'string') && typeof config.default === 'string') {
                transform = selectIndex(values as string[]) as (v: unknown) => unknown
            }
        }
        if (!transform) continue
        props[name] = {...config, transform}
        changed = true
    }
    return changed ? ({...definition, props} as StdDefinition<T>) : definition
}

/**
 * A WGSL definition that reads `time` anywhere (a slot, its helpers, an input's code) and declares
 * no clock gets the layer's own clock and a Speed prop, like the envelope does. Without it `time`
 * would be the page clock, which captures and the Speed control can't drive.
 */
function withImpliedClock<T extends ComponentProps>(definition: StdDefinition<T>, extraCode: readonly string[] = []): StdDefinition<T> {
    const record = definition as unknown as Record<string, unknown>
    if (record.animatedTime) return definition
    const bodies = ['paint', 'effect', 'surface', 'map', 'shape'].map((f) => record[f]).filter(isWgslBody)
    if (!bodies.length) return definition
    const readsTime = [...bodies.map((b) => b.spec.body), ...extraCode].some((code) => scanIdentifiers(code).has('time'))
    if (!readsTime) return definition
    const props = {...(record.props as Record<string, StdPropConfig<unknown>>)}
    props.speed ??= {default: 1, description: 'How fast it moves (0 holds it still)', ui: {type: 'range', min: 0, max: 5, step: 0.01, label: 'Speed', group: 'Animation'}}
    return {...definition, props, animatedTime: {speed: 'speed'}} as StdDefinition<T>
}

/**
 * The props a `wgsl` definition implies but needn't spell out: a shape material's shape and
 * placement props, a Color Space select for a gradient prop (its stops mix in that space), and
 * a Loop toggle for each video prop.
 */
function withImpliedProps<T extends ComponentProps>(input: StdDefinition<T>, extraCode: readonly string[] = []): StdDefinition<T> {
    const definition = withImpliedClock(withInferredTransforms(input), extraCode)
    const record = definition as unknown as Record<string, unknown>
    if (!wgslBodyOf(record) || !('props' in definition)) return definition
    const props = {...(definition.props as Record<string, StdPropConfig<unknown>>)}
    const before = Object.keys(props).length
    if ('surface' in record) {
        for (const [name, config] of Object.entries(SURFACE_PROPS)) props[name] ??= config
    }
    const configs = Object.entries(props)
    if (!props.colorSpace && configs.some(([, config]) => config.transform === (colorStopsTransform as unknown))) {
        props.colorSpace = {
            default: 'linear',
            transform: transformColorSpace as (value: unknown) => unknown,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'},
        }
    }
    for (const [name, config] of configs) {
        const type = Array.isArray(config.ui?.type) ? config.ui?.type[0] : config.ui?.type
        if (type === 'video-upload' && !props[`${name}Loop`]) {
            props[`${name}Loop`] = {default: true, transform: transformBool01, description: 'Loop the video', ui: {type: 'checkbox', label: 'Loop', group: config.ui?.group}}
        }
    }
    return Object.keys(props).length === before ? definition : {...definition, props} as StdDefinition<T>
}

function lowerCustom<T extends ComponentProps>(definition: StdCustomDefinition<T>): GpuShaderDefinition<T> {
    const {role: _role, species: _species, gpu, props, ...meta} = definition
    return {
        ...meta,
        props: lowerProps(props),
        fragment: gpu.fragment,
        ...(gpu.uvRemap ? {uvRemap: gpu.uvRemap} : {}),
    }
}

// ── Entry ───────────────────────────────────────────────────────────────────────────────

/**
 * A small stable fingerprint (FNV-1a) of the `wgsl` bodies a definition carries, so a
 * live-edited body under an unchanged name still recomposes (see `GpuShaderDefinition.revision`).
 */
function wgslRevision(definition: StdDefinition<unknown & ComponentProps>, extra?: {inputs?: unknown; helpers: string | null}): string | undefined {
    const bodies: string[] = []
    if (extra) bodies.push(JSON.stringify(extra.inputs ?? {}), extra.helpers ?? '')
    const {paint, effect, surface} = definition as {paint?: unknown; effect?: unknown; surface?: unknown}
    for (const candidate of [paint, effect, surface]) {
        if (isWgslBody(candidate)) bodies.push(candidate.spec.body, serializeInputs(candidate.spec.inputs ?? {}), candidate.spec.alpha ?? '')
    }
    if (bodies.length === 0) return undefined
    let h = 0x811c9dc5
    for (const ch of bodies.join('\u0000')) {
        h ^= ch.charCodeAt(0)
        h = Math.imul(h, 0x01000193) >>> 0
    }
    return h.toString(16)
}

/**
 * A stable text form of a body's explicit inputs — each binding's value AND declared type —
 * so changing what a name is bound to (`k: 4` → `k: 5`, `t: ctx.time` → `t: p('phase')`)
 * yields a new revision. Keys are sorted; a signal graph serializes by kind (its build fn
 * has no stable text, and signals are created per definition anyway).
 */
function serializeInputs(inputs: Record<string, unknown>): string {
    const scalarNode = (node: unknown): string => {
        if (typeof node !== 'object' || node === null) return String(node)
        const n = node as {kind: string; name?: string; a?: unknown; b?: unknown; center?: {name: string}; radius?: {name: string}; falloff?: {name: string}}
        switch (n.kind) {
            case 'prop': return `p:${n.name}`
            case 'ctx': return `c:${n.name}`
            case 'mul': return `mul(${scalarNode(n.a)},${scalarNode(n.b)})`
            case 'radialMask': return `radialMask(${n.center?.name},${n.radius?.name},${n.falloff?.name})`
            default: return n.kind
        }
    }
    const spec = (value: unknown): string => {
        if (typeof value === 'number') return `n:${value}`
        if (value instanceof Scalar) return `s:${scalarNode(value.node)}`
        return scalarNode(value)
    }
    return Object.keys(inputs).sort().map((key) => {
        const input = inputs[key]
        const typed = input !== null && typeof input === 'object' && 'value' in (input as object) && 'type' in (input as object)
        return typed
            ? `${key}=${spec((input as {value: unknown}).value)}:${(input as {type: string}).type}`
            : `${key}=${spec(input)}`
    }).join(',')
}

// ── Prop-name validation ────────────────────────────────────────────────────────────────

/**
 * Names every framework component owns as LAYER props: a shader prop spelled the same could
 * never be set (the component consumes it first), so it is rejected at definition time.
 */
const LAYER_PROP_NAMES = new Set([
    'blendMode', 'opacity', 'visible', 'id', 'maskSource', 'maskType', 'renderOrder',
    'transform', 'boundingBox', 'flow', 'absolute', 'children', 'ref', 'key',
    // How <CustomShader> receives the definition itself.
    'src',
])

/** Names the renderer registers as synthetic per-node fields (never authored). */
const SYNTHETIC_PROP_NAMES = new Set(['_animTime', '_opacity'])
const SYNTHETIC_PROP_PREFIXES = ['_pad', '_bbox_', '_map_', '_childBounds_', '_animTime_']

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/

function validatePropNames(name: string, props: Record<string, unknown>): void {
    for (const prop of Object.keys(props)) {
        if (!IDENTIFIER.test(prop)) {
            throw new Error(`defineShader("${name}"): prop '${prop}' is not a valid identifier (letters, digits and _ only, not starting with a digit)`)
        }
        if (LAYER_PROP_NAMES.has(prop)) {
            throw new Error(`defineShader("${name}"): '${prop}' is a layer prop every component already has (blend mode, opacity, layout, …) — rename the shader prop`)
        }
        if (SYNTHETIC_PROP_NAMES.has(prop) || SYNTHETIC_PROP_PREFIXES.some((prefix) => prop.startsWith(prefix))) {
            throw new Error(`defineShader("${name}"): '${prop}' collides with a field the renderer manages itself — rename the shader prop`)
        }
    }
}

// Lowers a std definition to the engine contract: validates prop names, dispatches on the
// inferred role, and stamps a `revision` fingerprint when the definition carries `wgsl` bodies.
function defineShaderImpl<T extends ComponentProps>(input: StdDefinition<T>): GpuShaderDefinition<T> {
    const {inputs, helpers, ...rest} = input as StdDefinition<T> & {inputs?: Record<string, unknown>; helpers?: WgslBody | string}
    const helpersText = helpers === undefined ? null : isWgslBody(helpers) ? helpers.spec.body : String(helpers)
    const plan = inputs || helpersText ? planInputs(inputs) : null
    // Helpers and input code count toward "reads time" (a simulation step that reads time
    // needs the layer's clock as much as the slot does).
    const extraCode = [...(helpersText ? [helpersText] : []), ...(plan?.inputs.map((i) => codeIn(i.node)) ?? [])]
    const definition = withImpliedProps(rest as StdDefinition<T>, extraCode)
    if ('props' in definition && definition.props) validatePropNames(definition.name, definition.props as Record<string, unknown>)
    if (plan?.invalid.length) {
        throw new Error(`defineShader("${definition.name}"): inputs ${plan.invalid.map((n) => `'${n}'`).join(', ')} must be input words (blur, previousFrame, pass, …) or library functions from shaders/std/wgsl`)
    }
    let lowered: GpuShaderDefinition<T>
    lowering = plan ? {plan, helpers: helpersText} : null
    try {
        lowered = lowerStd(definition)
    } finally {
        lowering = null
    }
    // Built-ins a WGSL body reads (`dt`, `frame`, `idle`, …) need their frame fields whether or not
    // the component has inputs: an old envelope with no inputs that mentions `dt` must still compose.
    const wgslWritten = ['paint', 'effect', 'surface', 'map', 'shape'].some((f) => isWgslBody((definition as unknown as Record<string, unknown>)[f]))
    if (plan || wgslWritten) lowered = withInputs(lowered, definition as StdDefinition<ComponentProps>, plan ?? planInputs(undefined), helpersText)
    const revision = wgslRevision(definition as StdDefinition<ComponentProps>, plan ? {inputs, helpers: helpersText} : undefined)
    return revision ? {...lowered, revision} : lowered
}

/** The WGSL a definition carries: its bodies, its helpers and the code inside its inputs. */
function definitionCode(definition: Record<string, unknown>, helpers: string | null, plan: InputPlan): string[] {
    const parts: string[] = []
    for (const field of ['paint', 'effect', 'surface', 'map', 'shape']) {
        const body = definition[field]
        if (isWgslBody(body)) parts.push(body.spec.body)
    }
    if (helpers) parts.push(helpers)
    // Each input's WGSL as real text. (It was JSON.stringify(node), which puts a whole simulation
    // on one line: a `//` comment then blanked everything after it, so a step that read `pointer`
    // after a comment silently lost mouse tracking.)
    for (const i of plan.inputs) parts.push(codeIn(i.node))
    return parts
}

/** Every name the definition's WGSL reads. Each piece is scanned on its own, so a comment in one
 *  can never hide names in another. */
function definitionIdentifiers(definition: Record<string, unknown>, helpers: string | null, plan: InputPlan): Set<string> {
    const ids = new Set<string>()
    for (const piece of definitionCode(definition, helpers, plan)) for (const id of identifiersIn(piece)) ids.add(id)
    return ids
}

/** Attach the inputs' compute hook, the frame fields built-ins read, and the flags inputs imply. */
function withInputs<T extends ComponentProps>(
    lowered: GpuShaderDefinition<T>,
    definition: StdDefinition<ComponentProps>,
    plan: InputPlan,
    helpers: string | null,
): GpuShaderDefinition<T> {
    const ids = definitionIdentifiers(definition as unknown as Record<string, unknown>, helpers, plan)
    const builtins = BUILTIN_FIELDS.filter((b) => ids.has(b))
    const pointerish = ['pointer', 'pointerVelocity', 'pointerDown', 'idle', 'pointerOver'].some((n) => ids.has(n))
        || plan.inputs.some((i) => i.node.kind === 'events' || i.node.kind === 'pointerTrail' || i.node.readsPointer)
    const out: GpuShaderDefinition<T> = {...lowered}
    if (plan.inputs.length) {
        out.compute = inputsCompute(plan, {
            name: definition.name,
            props: lowered.props as unknown as Record<string, PropConfig<unknown>>,
            animated: !!(definition as {animatedTime?: unknown}).animatedTime,
            helpers,
        }, lowered.compute)
    }
    if (builtins.length) out.extraFields = {...builtinExtraFields(builtins), ...(lowered.extraFields ?? {})} as GpuShaderDefinition<T>['extraFields']
    if (pointerish) out.usesPointer = true
    if (plan.readsChild && !lowered.requiresChild) {
        out.requiresRTT = true
        out.requiresChild = true
    }
    return out
}

type InferredRole = 'shape' | 'warp' | 'custom' | 'generator' | 'filter' | 'surface'

/**
 * The role a definition has, from the field carrying its GPU half. A declared `role` may
 * restate it (the library's shaders do) but cannot contradict it: the field IS the role.
 */
function inferRole(definition: Record<string, unknown>): InferredRole {
    const inferred: InferredRole | null =
        'shape' in definition ? 'shape'
        : 'map' in definition ? 'warp'
        : 'gpu' in definition ? 'custom'
        : 'paint' in definition ? 'generator'
        : 'effect' in definition ? 'filter'
        : 'surface' in definition ? 'surface'
        : null
    const name = String(definition.name ?? '?')
    if (!inferred) {
        throw new Error(`defineShader("${name}"): give it a GPU half — paint: (generator), effect: (filter), map: (warp), shape: (shape), surface: (a material on a shape) or gpu: (custom)`)
    }
    const declared = definition.role as string | undefined
    // Custom-tier roles are labels (simulation, media, …) and can sit on a `gpu:` definition.
    if (declared && inferred !== 'custom' && declared !== inferred) {
        throw new Error(`defineShader("${name}"): role '${declared}' contradicts its ${FIELD_FOR_ROLE[inferred]} field (a ${inferred})`)
    }
    return inferred
}

const FIELD_FOR_ROLE: Record<InferredRole, string> = {shape: 'shape:', warp: 'map:', custom: 'gpu:', generator: 'paint:', filter: 'effect:', surface: 'surface:'}

/** The species a filter effect implies; a declared species must agree unless the effect is raw WGSL. */
function inferSpecies<T extends ComponentProps>(
    definition: StdPointwiseFilterDefinition<T> | StdGatherFilterDefinition<T> | StdWgslFilterDefinition<T>,
): 'pointwise' | 'gather' {
    const {effect, species} = definition
    if (isWgslBody(effect)) return species ?? (effect.samplesChild ? 'gather' : 'pointwise')
    const implied: 'pointwise' | 'gather' = effect.kind === 'gather' || effect.kind === 'displaceBy' ? 'gather' : 'pointwise'
    if (species && species !== implied) {
        throw new Error(`defineShader("${definition.name}"): species '${species}' contradicts its '${effect.kind}' effect (${implied})`)
    }
    return implied
}

function lowerStd<T extends ComponentProps>(definition: StdDefinition<T>): GpuShaderDefinition<T> {
    switch (inferRole(definition as unknown as Record<string, unknown>)) {
        case 'shape':
            return lowerShape(definition as StdShapeDefinition | StdWgslShapeDefinition<T>)
        case 'warp':
            return lowerWarp(definition as StdWarpDefinition<T>)
        case 'custom':
            return lowerCustom(definition as StdCustomDefinition<T>)
        case 'generator':
            return lowerGenerator(definition as StdGeneratorDefinition<T>)
        case 'surface':
            return lowerSurface(definition as StdSurfaceDefinition<T>)
        case 'filter': {
            const filter = definition as StdPointwiseFilterDefinition<T> | StdGatherFilterDefinition<T> | StdWgslFilterDefinition<T>
            return inferSpecies(filter) === 'gather'
                ? lowerGatherFilter({...filter, species: 'gather'} as StdGatherFilterDefinition<T>)
                : lowerPointwiseFilter({...filter, species: 'pointwise'} as StdPointwiseFilterDefinition<T>)
        }
    }
}

/**
 * Define a shader component from a plain object: its props, and one field that says what it draws.
 *
 * The field carrying the per-pixel work decides what kind of component you get. `paint:`
 * makes a generator: it paints from coordinates and needs nothing inside it. `effect:` makes
 * a filter over the layer inside it (the child). A `wgsl` body that only reads `child` edits
 * one pixel at a time; one that samples `childTexture` reads neighbours through its own
 * render pass. `map:` makes a distortion from one coordinate function, `shape:` a 2D shape
 * with fill and stroke, `gpu:` a raw builder. Give exactly one. `role` and `species` are
 * optional and may only restate what the field already says.
 *
 * `props` are the component's attributes. Each has a `default`, an optional `transform`
 * (`transformColor`, `transformPosition`, …), optional `ui` metadata for editors, and
 * `compileTime: true` when a change should rebuild the shader (or `recompile: crosses(0)` to
 * rebuild only at a threshold). A prop name must be a plain identifier and must not be one
 * of the layer props every component already has — `blendMode`, `opacity`, `visible`, `id`,
 * `maskSource`, `maskType`, `renderOrder`, `transform`, `boundingBox`, `flow`, `absolute`,
 * `children`, `ref`, `key`, `src` — nor a name the renderer manages itself (`_animTime`,
 * `_opacity`, or anything starting `_pad`, `_bbox_`, `_map_`). Those throw when the
 * definition is built. `animatedTime: {speed: 'speed'}` gives the layer its own clock, scaled by that prop
 * (0 pauses it); `time` in a `wgsl` body and `animatedTime(params)` in a builder read it.
 *
 * The returned definition is what you hand to `<CustomShader src={Halo}>` in React, Vue,
 * Svelte or Solid, to `registerShader`, or to `createShader(canvas, preset, {components:
 * [Halo]})` so preset JSON can name it by `type`.
 *
 * @example
 * ```ts
 * import {defineShader, wgsl, transformColor, transformPosition} from 'shaders/std'
 *
 * export const Halo = defineShader({
 *   name: 'Halo',
 *   animatedTime: {speed: 'speed'},
 *   props: {
 *     inner: {default: '#ffd166', transform: transformColor},
 *     outer: {default: '#0b132b', transform: transformColor},
 *     center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
 *     radius: {default: 0.6},
 *     bands: {default: 4},
 *     speed: {default: 1},
 *   },
 *   paint: wgsl`
 *     let d = length((uv - center) * vec2f(aspect, 1.0)) / radius;
 *     let wave = 0.5 + 0.5 * cos(d * bands * 6.2831853 - time * 2.0);
 *     return vec4f(mix(outer.rgb, inner.rgb, wave * (1.0 - smoothstep(0.7, 1.0, d))), 1.0);
 *   `,
 * })
 * ```
 * @example
 * ```ts
 * // A filter composed from std words, skipped entirely while `intensity` is 0.
 * export const Vignette = defineShader({
 *   name: 'Vignette',
 *   props: {
 *     color: {default: '#000000', transform: transformColor},
 *     center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
 *     radius: {default: 0.5},
 *     falloff: {default: 0.5},
 *     intensity: {default: 1, recompile: crosses(0)},
 *   },
 *   effect: tintToward(p('color'), {amount: radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity'))}),
 *   identityWhen: isZero('intensity'),
 * })
 * ```
 * @tip A prop the engine never sends to the GPU (a URL string, a shape object, a list) cannot be read in a `wgsl` body; reference it and WGSL reports the name as undefined.
 * @see wgsl, registerShader, p, crosses, isZero, listOf
 */
// Role inference (`inferRole`), species inference (`inferSpecies`) and prop-name validation
// (`validatePropNames`) live above.
export const defineShader: typeof defineShaderImpl = defineShaderImpl

/**
 * The earlier name of `defineShader`. Same function; kept so code written against it still builds.
 *
 * @deprecated Use `defineShader`. `defineStd` will be removed in the next major version.
 */
export const defineStd: typeof defineShaderImpl = defineShaderImpl
