/**
 * std/document — the `wgsl` envelope: a raw `wgsl` body plus a props manifest, as data.
 *
 *   validateWgslEnvelope(envelope)   static checks — never throws
 *   wgslComponentData(envelope)      → the `defineShader` object it stands for, as data
 *   defineWgslEnvelope(envelope)     → an ordinary GpuShaderDefinition
 *
 * The body is the same one `defineShader({paint: wgsl`…`})` takes, so the binding rules are
 * the `wgsl` escape hatch's (see `../wgsl.ts` and `../wgslResources.ts`). Prop kinds are the
 * library's own controls by name (`range`, `select`, `image-upload`, `gradient-stops`, `shape`…).
 * What the document cannot say, the body and props decide: a `shape` prop makes a material on
 * that shape (`surface:`), reading `child` makes a per-pixel filter, sampling `childTexture` a
 * filter with its own render pass, neither a generator; reading `time` adds a `speed` prop and
 * the layer's own clock.
 */
import type {GpuShaderDefinition} from '../../gpu/contract'
import type {ComponentProps} from '../../types'
import {defineShader} from '../lower'
import {scanIdentifiers, wgsl} from '../wgsl'
import {propConfigsFromData, type GraphPropConfigData} from './define'
import {STD_GRAPH_LIMITS, StdGraphError, type WgslEnvelope, type WgslPropKind, type WgslPropSpec} from './types'

const MAX_BODY = 12000
const MAX_LIST_ITEMS = 16
const MAX_STOPS = 8

/** Names the body owns; a prop spelled the same would never be bound. */
const BODY_NAMES = new Set([
    'uv', 'time', 'aspect', 'viewport', 'pointer', 'child', 'childTexture', 'childSampler', 'viewportSize', 'logicalViewportSize',
    'shapeDistance', 'shapeSlope', 'shapeUV', 'shapeDepth', 'shapeField', 'shapeAt',
])

/** Every layer has these, or the renderer reads them by name. */
const RESERVED_PROPS = new Set(['colorSpace', 'opacity', 'blendMode', 'visible', 'transform', 'boundingBox', 'maskSource', 'maskType', 'origin', 'shapeSdfUrl', 'shapeType'])

/** A shape material places its shape with these itself. */
const SHAPE_PLACEMENT = new Set(['center', 'scale', 'rotation'])

const KINDS: readonly WgslPropKind[] = ['range', 'number', 'checkbox', 'color', 'position', 'select', 'gradient-stops', 'list', 'image-upload', 'video-upload', 'layer', 'shape']

/**
 * The earlier kind names, still read so components saved with them keep working: each is the
 * same control as its library name (`angle` a 0–360 range in degrees).
 */
const KIND_ALIASES: Record<string, (spec: WgslPropSpec) => WgslPropSpec> = {
    float: (spec) => ({...spec, kind: 'range'}),
    angle: (spec) => ({...spec, kind: 'range', min: spec.min ?? 0, max: spec.max ?? 360, step: spec.step ?? 1}),
    bool: (spec) => ({...spec, kind: 'checkbox'}),
}

/** A manifest with the earlier kind names read as their library names. */
function resolveAliases(props: Record<string, WgslPropSpec>): Record<string, WgslPropSpec> {
    let out: Record<string, WgslPropSpec> | null = null
    for (const [name, spec] of Object.entries(props)) {
        const alias = spec && KIND_ALIASES[spec.kind as string]
        if (alias) (out ??= {...props})[name] = alias(spec)
    }
    return out ?? props
}

/** Kinds people reach for that have another name here, or no place in a custom component. */
const KIND_HINTS: Record<string, string> = {
    boolean: 'use "checkbox"',
    image: 'use "image-upload"',
    video: 'use "video-upload"',
    gradient: 'use "gradient-stops"',
    map: 'every "range" prop can already be driven by a map, the mouse or auto-animate',
    origin: 'it comes with a "shape" prop, along with center, scale and rotation',
    shape3d: 'use "shape": its picker includes the 3D shapes',
    text: 'text is not available inside a custom component yet; put a Text layer inside or above it',
    'font-family': 'text is not available inside a custom component yet; put a Text layer inside or above it',
    'font-weight': 'text is not available inside a custom component yet; put a Text layer inside or above it',
}

const HEX = /^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/

export interface WgslEnvelopeValidation {
    ok: boolean
    error?: string
    /** `generator` paints from coordinates; `filter` reads the layer inside it; `surface` shades a shape. */
    role?: 'generator' | 'filter' | 'surface'
    /** The body reads `time`, so the definition carries `speed` + the layer clock. */
    usesTime?: boolean
}

/** The option values of a select spec. */
function optionValues(spec: WgslPropSpec): string[] {
    return (spec.options ?? []).map((option) => (typeof option === 'string' ? option : option.value))
}

/** Names a prop gives the body besides its own (resource textures, list lanes…). */
function derivedNames(name: string, spec: WgslPropSpec): string[] {
    switch (spec.kind) {
        case 'image-upload': case 'video-upload': case 'layer':
            return [`${name}Texture`, `${name}Sampler`, `${name}_viewport`]
        case 'gradient-stops':
            return [`${name}_stopColors`, `${name}_stopPositions`, `${name}_stopConverted`, `${name}_stopCount`]
        case 'list':
            return [`${name}Count`, ...Object.keys(spec.item ?? {}).map((field) => `${name}_${field}`)]
        default:
            return []
    }
}

function checkProp(name: string, spec: WgslPropSpec, hasShape: boolean): void {
    if (!/^[a-z][A-Za-z0-9]*$/.test(name)) throw new StdGraphError(`prop "${name}" must be camelCase (letters/digits, starting lowercase)`)
    if (RESERVED_PROPS.has(name)) throw new StdGraphError(`prop "${name}" is reserved (every layer has it, or the renderer reads it)`)
    if (hasShape && SHAPE_PLACEMENT.has(name)) throw new StdGraphError(`prop "${name}" comes with the shape prop (it places the shape) — leave it out or rename yours`)
    if (BODY_NAMES.has(name)) throw new StdGraphError(`prop "${name}" collides with a name the body is given — rename it`)
    if (!spec || typeof spec !== 'object') throw new StdGraphError(`prop "${name}" needs a spec: {"kind": …, "default": …}`)
    if (!KINDS.includes(spec.kind)) {
        const hint = KIND_HINTS[spec.kind as string]
        throw new StdGraphError(`prop "${name}": kind "${String(spec.kind)}" isn't available${hint ? `: ${hint}` : ''}. Kinds: ${KINDS.join(', ')}`)
    }
    if (name === 'speed' && spec.kind !== 'range') throw new StdGraphError(`prop "speed" must be a range (it drives the clock)`)
    if (name === 'shape' && spec.kind !== 'shape') throw new StdGraphError(`prop "shape" is the shape picker: give it kind "shape"`)
    const dflt = spec.default
    switch (spec.kind) {
        case 'range':
            if (typeof dflt !== 'number' || !Number.isFinite(dflt)) throw new StdGraphError(`prop "${name}": default must be a number`)
            if (typeof spec.min !== 'number' || typeof spec.max !== 'number') throw new StdGraphError(`prop "${name}": range props need min and max`)
            if (spec.min >= spec.max) throw new StdGraphError(`prop "${name}": min must be below max`)
            break
        case 'number':
            if (typeof dflt !== 'number' || !Number.isFinite(dflt)) throw new StdGraphError(`prop "${name}": default must be a number`)
            break
        case 'checkbox':
            if (typeof dflt !== 'boolean') throw new StdGraphError(`prop "${name}": default must be true/false`)
            break
        case 'color':
            if (typeof dflt !== 'string' || !HEX.test(dflt)) throw new StdGraphError(`prop "${name}": default must be "#rrggbb"`)
            break
        case 'position': {
            const pos = dflt as {x?: unknown; y?: unknown} | null
            if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number') throw new StdGraphError(`prop "${name}": default must be {x, y}`)
            break
        }
        case 'select': {
            const values = optionValues(spec)
            if (!values.length || values.some((value) => typeof value !== 'string' || !value)) throw new StdGraphError(`prop "${name}": select props need "options": a list of strings or {value, label}`)
            if (dflt !== undefined && !values.includes(dflt as string)) throw new StdGraphError(`prop "${name}": default must be one of the option values (${values.join(', ')})`)
            break
        }
        case 'gradient-stops': {
            const stops = dflt as Array<{color?: unknown; position?: unknown}> | undefined
            if (!Array.isArray(stops) || stops.length < 2 || stops.length > MAX_STOPS) throw new StdGraphError(`prop "${name}": default must be 2–${MAX_STOPS} stops: [{"color": "#rrggbb", "position": 0}, …]`)
            if (stops.some((stop) => typeof stop?.color !== 'string' || !HEX.test(stop.color) || typeof stop.position !== 'number' || stop.position < 0 || stop.position > 1)) {
                throw new StdGraphError(`prop "${name}": each stop is {"color": "#rrggbb", "position": 0–1}`)
            }
            break
        }
        case 'list': {
            const fields = Object.entries(spec.item ?? {})
            if (!fields.length) throw new StdGraphError(`prop "${name}": list props need "item": each item's fields, e.g. {"at": {"kind": "position", "default": {"x": 0.5, "y": 0.5}}}`)
            for (const [field, config] of fields) {
                if (!/^[a-z][A-Za-z0-9]*$/.test(field)) throw new StdGraphError(`prop "${name}": item field "${field}" must be camelCase`)
                if (!['position', 'color', 'number', 'boolean'].includes(config?.kind)) throw new StdGraphError(`prop "${name}": item field "${field}" kind must be position, color, number or boolean`)
            }
            const max = spec.maxItems
            if (typeof max !== 'number' || !Number.isInteger(max) || max < 1 || max > MAX_LIST_ITEMS) throw new StdGraphError(`prop "${name}": list props need "maxItems", 1–${MAX_LIST_ITEMS}`)
            if (!Array.isArray(dflt) || dflt.length > max) throw new StdGraphError(`prop "${name}": default must be a list of at most ${max} items`)
            break
        }
        case 'image-upload': case 'video-upload':
            if (typeof dflt !== 'string') throw new StdGraphError(`prop "${name}": default must be a URL ("" for none)`)
            if (spec.fit !== undefined && !['cover', 'contain', 'fill'].includes(spec.fit)) throw new StdGraphError(`prop "${name}": fit must be cover, contain or fill`)
            break
        case 'layer':
            if (dflt !== undefined && dflt !== '') throw new StdGraphError(`prop "${name}": a layer prop starts empty (default ""); the user picks the layer`)
            break
        case 'shape': {
            const shape = dflt as {type?: unknown} | undefined
            if (dflt !== undefined && (typeof shape !== 'object' || shape === null || typeof shape.type !== 'string')) throw new StdGraphError(`prop "${name}": default must be a shape config such as {"type": "starSDF", "sides": 5, "radius": 0.35}`)
            break
        }
    }
}

/** Every problem with a props manifest, one message each. */
function propErrors(props: Record<string, WgslPropSpec>): string[] {
    const errors: string[] = []
    const names = Object.keys(props)
    if (names.length > STD_GRAPH_LIMITS.maxProps) errors.push(`${names.length} props declared (max ${STD_GRAPH_LIMITS.maxProps})`)
    const taken = new Map<string, string>()
    const hasShape = Object.values(props).some((spec) => spec?.kind === 'shape')
    for (const [name, spec] of Object.entries(props)) {
        try {
            checkProp(name, spec, hasShape)
        } catch (err) {
            errors.push(err instanceof Error ? err.message : String(err))
            continue
        }
        for (const derived of derivedNames(name, spec)) {
            if (props[derived] || taken.has(derived)) errors.push(`prop "${name}" gives the body "${derived}", which another prop already uses — rename one`)
            taken.set(derived, name)
        }
    }
    if (Object.values(props).filter((spec) => spec?.kind === 'gradient-stops').length > 1) errors.push('one gradient-stops prop per component')
    if (Object.entries(props).some(([name, spec]) => spec?.kind === 'shape' && name !== 'shape')) errors.push('the shape prop must be named "shape"')
    return errors
}

/** At most this many problems are listed; the rest are counted. */
const MAX_REPORTED_ERRORS = 10

function summarize(errors: string[]): string {
    if (errors.length === 1) return errors[0]!
    const listed = errors.slice(0, MAX_REPORTED_ERRORS).map((e) => `- ${e}`)
    if (errors.length > MAX_REPORTED_ERRORS) listed.push(`- … and ${errors.length - MAX_REPORTED_ERRORS} more`)
    return `${errors.length} problems:\n${listed.join('\n')}`
}

/** Static check of a `wgsl` envelope. Never throws. */
export function validateWgslEnvelope(envelope: unknown): WgslEnvelopeValidation {
    try {
        const env = envelope as WgslEnvelope
        if (!env || typeof env !== 'object') throw new StdGraphError('envelope is not an object')
        if (env.kind !== 'wgsl') throw new StdGraphError('kind must be "wgsl"')
        if (typeof env.name !== 'string' || !env.name.trim()) throw new StdGraphError('name is required')
        if (typeof env.body !== 'string' || !env.body.trim()) throw new StdGraphError('body is required: WGSL statements that return a vec4f colour')
        if (env.body.length > MAX_BODY) throw new StdGraphError(`body is ${env.body.length} chars (max ${MAX_BODY})`)
        if (!/\breturn\b/.test(env.body)) throw new StdGraphError('body must `return` a vec4f colour')
        const props = resolveAliases(env.props ?? {})
        const errors = propErrors(props)
        if (errors.length) throw new StdGraphError(summarize(errors))
        const identifiers = scanIdentifiers(env.body)
        const hasShape = Object.values(props).some((spec) => spec.kind === 'shape')
        if (!hasShape && ['shapeUV', 'shapeDepth', 'shapeField', 'shapeAt'].some((name) => identifiers.has(name))) {
            throw new StdGraphError('the body reads the shape (shapeUV / shapeDepth / shapeField / shapeAt) but there is no shape: add a prop "shape" of kind "shape"')
        }
        const role = hasShape ? 'surface' : identifiers.has('child') || identifiers.has('childTexture') ? 'filter' : 'generator'
        return {ok: true, role, usesTime: identifiers.has('time')}
    } catch (err) {
        return {ok: false, error: err instanceof Error ? err.message : String(err)}
    }
}

// ── Props → std prop configs, as data ─────────────────────────────────────────────────────

function labelOf(name: string, spec: WgslPropSpec): string {
    return spec.label ?? name.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase())
}

function niceStep(min: number, max: number): number {
    const span = Math.abs(max - min)
    if (span <= 1) return 0.01
    if (span <= 10) return 0.1
    if (span <= 100) return 1
    return Math.pow(10, Math.floor(Math.log10(span)) - 2)
}

/** One prop spec as the library's prop config, transforms named. */
function propData(name: string, spec: WgslPropSpec): GraphPropConfigData {
    const label = labelOf(name, spec)
    const base = {description: spec.description ?? label}
    const ui = (extra: Record<string, unknown>) => ({...extra, label, group: spec.group ?? (spec.kind === 'color' || spec.kind === 'gradient-stops' ? 'Colors' : 'Effect')})
    switch (spec.kind) {
        case 'range':
            return {...base, default: spec.default, ui: ui({type: ['range', 'map'], min: spec.min, max: spec.max, step: spec.step ?? niceStep(spec.min!, spec.max!)})}
        case 'number':
            return {...base, default: spec.default, ui: ui({type: 'number', ...(spec.min !== undefined ? {min: spec.min} : {}), ...(spec.max !== undefined ? {max: spec.max} : {}), ...(spec.step !== undefined ? {step: spec.step} : {})})}
        case 'checkbox':
            return {...base, default: spec.default, transform: 'transformBool01', ui: ui({type: 'checkbox'})}
        case 'color':
            return {...base, default: spec.default, transform: 'transformColor', ui: ui({type: 'color'})}
        case 'position':
            return {...base, default: spec.default, transform: 'transformPosition', ui: {type: 'position', label, group: spec.group ?? 'Position'}}
        case 'select': {
            const options = (spec.options ?? []).map((option) => (typeof option === 'string'
                ? {label: option.replace(/(^|[-_ ])(\w)/g, (_, sep: string, c: string) => (sep ? ' ' : '') + c.toUpperCase()), value: option}
                : {label: option.label ?? option.value, value: option.value}))
            const values = options.map((option) => option.value)
            return {...base, default: spec.default ?? values[0], transform: {selectIndex: values}, ui: ui({type: 'select', options})}
        }
        case 'gradient-stops':
            return {...base, default: spec.default, transform: 'colorStopsTransform', ui: ui({type: 'gradient-stops'})}
        case 'list':
            return {
                ...base,
                default: spec.default,
                transform: 'listPropTransform',
                ui: ui({type: 'list', item: spec.item, maxItems: spec.maxItems, minItems: spec.minItems ?? 0, ...(spec.itemLabel ? {itemLabel: spec.itemLabel} : {})}),
            }
        case 'image-upload': case 'video-upload':
            return {...base, default: spec.default, ui: ui({type: spec.kind, ...(spec.fit && spec.fit !== 'cover' ? {fit: spec.fit} : {})})}
        case 'layer':
            return {...base, default: '', compileTime: true, ui: ui({type: 'layer'})}
        case 'shape':
            return {
                description: spec.description ?? 'The shape the material covers: a flat shape, an SVG or a 3D shape',
                default: JSON.stringify(spec.default ?? {type: 'circleSDF', radius: 0.35}),
                ui: {type: 'shape', label: spec.label ?? 'Shape', group: spec.group ?? 'Shape'},
            }
    }
}

/**
 * Expand a manifest to prop configs as data. A body that reads the clock gains a `speed` prop
 * (unless it declares one). The props a gradient, video or shape implies (Color Space, Loop,
 * placement) are added by `defineShader` itself, so they're not written here.
 */
export function wgslPropData(props: Record<string, WgslPropSpec>, options: {usesTime: boolean}): Record<string, GraphPropConfigData> {
    const out: Record<string, GraphPropConfigData> = {}
    for (const [name, spec] of Object.entries(resolveAliases(props))) out[name] = propData(name, spec)
    if (options.usesTime && !out.speed) {
        out.speed = {default: 1, description: 'Animation speed', ui: {type: 'range', min: 0, max: 5, step: 0.05, label: 'Speed', group: 'Animation'}}
    }
    return out
}

/**
 * The `defineShader` object a `wgsl` envelope stands for, as plain data: which field carries
 * the body, the clock, and the expanded prop configs with their transforms named. The editor
 * builds the live definition from it (`defineWgslEnvelope`) and code export writes it out as
 * source (`wgslComponentSource`), so the two can't drift. Throws StdGraphError on an invalid
 * envelope.
 */
export interface WgslComponentData {
    name: string
    description?: string
    /** `surface` with a shape prop; `effect` when the body reads the layer inside it; else `paint`. */
    field: 'paint' | 'effect' | 'surface'
    /** Present when the body reads `time`: the layer's own clock, scaled by the `speed` prop. */
    animatedTime?: {speed: 'speed'}
    props: Record<string, GraphPropConfigData>
    body: string
}

export function wgslComponentData(envelope: WgslEnvelope): WgslComponentData {
    const check = validateWgslEnvelope(envelope)
    if (!check.ok) throw new StdGraphError(check.error!)
    const usesTime = !!check.usesTime
    return {
        name: envelope.name,
        ...(envelope.description ? {description: envelope.description} : {}),
        field: check.role === 'surface' ? 'surface' : check.role === 'filter' ? 'effect' : 'paint',
        ...(usesTime ? {animatedTime: {speed: 'speed' as const}} : {}),
        props: wgslPropData(envelope.props ?? {}, {usesTime}),
        body: envelope.body,
    }
}

/** Lower a `wgsl` envelope to an ordinary definition. Throws StdGraphError on an invalid envelope. */
export function defineWgslEnvelope(envelope: WgslEnvelope): GpuShaderDefinition {
    const {field, body, props, description, ...meta} = wgslComponentData(envelope)
    return defineShader<ComponentProps>({
        category: 'Custom',
        description: description ?? meta.name,
        ...meta,
        props: propConfigsFromData(props) as never,
        [field]: wgsl(body),
    } as never)
}
