/**
 * What a `wgsl` body gets for the props that aren't a single number, colour or position: an
 * image, a video, another layer, a gradient or a list. The library's own components read the
 * same props through the same host machinery (media loaders, layer textures, packed stop and
 * list arrays); this module hands them to a body by name.
 *
 * For a prop `logo` (`ui.type` `image-upload`, `video-upload` or `layer`) the body gets:
 *   - `logo`: the picture at this pixel (`vec4f`, linear, straight alpha; transparent outside an
 *     image or video, which is fitted to the layer by `ui.fit`);
 *   - `logo(p)`: the same at any `uv`;
 *   - `logoTexture` + `logoSampler`: the raw texture (sRGB-encoded for images, premultiplied
 *     for a layer; a `texture_external` for video).
 * A gradient prop `ramp` (`gradient-stops`) is called: `ramp(t)` → `vec4f`, mixed in the
 * component's colour space. A list prop `dots` gives `dots_<field>` (`array<vec4f, maxItems>`,
 * one lane per item) and `dotsCount`.
 */
// Maintainer notes: a callable (`logo(p)`) is a text rewrite. In WGSL a value can't be called,
// so `logo(` unambiguously means the sampler; the body text has it replaced by a call to a
// helper fn that takes the prop's texture / arrays as leading arguments, linked into the body's
// `tgpu.fn` through `$uses`. Helpers are string-form fns built once per variant (fit, colour
// space). Textures and samplers reach the body as function arguments, so one lowered body
// serves every layer of a component.
import {tgpu, d, media} from '../gpu/kit/index'
import {call, expr} from '../gpu/composer'
import type {Expr, GpuFragmentParams, KitTexture} from '../gpu/contract'
import {gradientStopsInSpace, gradientStopsInSpaceVariants} from '../gpu/kit/colorStops'
import {sRGBToP3, oklabToRgb, oklchToOklab, hslToRgb, hsvToRgb, labToRgb, lchToLab} from '../gpu/kit/colorMixing'
import {createSwappableMediaTexture, createUrlSourceLoader, createVideoElementSource, decodeImageSource} from '../gpu/kit/host/mediaLifecycle'
import {MAX_COLOR_STOPS} from '../gpu/kit/colorStops'
import {colorStopsTransform} from '../utilities/colorStops'
import {listFieldName, listCountName, listSpecOf, listPropTransform} from '../utilities/listProps'
import {transformColorSpace} from '../utilities/transformations'
import type {PropConfig} from '../types'

/** A WGSL type a binding can have. Arrays carry their element type and length. */
export type ResourceWgslType = 'f32' | 'vec2f' | 'vec4f' | 'texture_2d<f32>' | 'texture_external' | 'sampler' | `array<vec4f, ${number}>` | `array<vec3f, ${number}>`

export interface ResourceBinding {
    name: string
    type: ResourceWgslType
    value: (params: GpuFragmentParams) => Expr
}

/** A prop the body calls like a function: `name(args…)` becomes `alias(leading…, args…)`. */
export interface ResourceCallable {
    name: string
    /** The identifier the call is rewritten to, linked to `helper` through `$uses`. */
    alias: string
    /** Binding names passed ahead of the body's own arguments. */
    leading: string[]
    /** The helper fn for this composition, and a key naming its variant. */
    helper: (params: GpuFragmentParams) => {key: string; fn: unknown}
}

export interface ResourcePlan {
    bindings: ResourceBinding[]
    callables: ResourceCallable[]
    /** Names bound only when the body uses them bare (not called): the value at this pixel. */
    bareValues: string[]
}

/** How the body's `uv` and `viewport` are read in this host. */
export interface HostCoords {
    uv: (params: GpuFragmentParams) => Expr
    viewport: (params: GpuFragmentParams) => Expr
}

const FIT_MODES = {cover: 0, contain: 1, fill: 2} as const
type Fit = keyof typeof FIT_MODES

// ── One resource per prop per composition ─────────────────────────────────────────────────

const perComposition = new WeakMap<object, Map<string, unknown>>()

function once<T>(params: GpuFragmentParams, key: string, make: () => T): T {
    let cache = perComposition.get(params)
    if (!cache) perComposition.set(params, (cache = new Map()))
    if (!cache.has(key)) cache.set(key, make())
    return cache.get(key) as T
}

export function imageTexture(params: GpuFragmentParams, prop: string): KitTexture {
    return once(params, `image:${prop}`, () => {
        const tex = createSwappableMediaTexture(params, {label: prop})
        createUrlSourceLoader(params, {
            prop,
            async load(url, loadCtx) {
                const decoded = await decodeImageSource(url)
                if (loadCtx.isDisposed()) {
                    decoded.close()
                    return
                }
                loadCtx.commit()
                tex.ensureSize(decoded.width, decoded.height)
                tex.write(decoded.source)
                decoded.close()
            },
        })
        return tex.kit
    })
}

function videoTexture(params: GpuFragmentParams, prop: string, loopProp: string | null): KitTexture {
    return once(params, `video:${prop}`, () => {
        const source = createVideoElementSource(params, {
            source: {kind: 'url', prop},
            ...(loopProp ? {loop: {prop: loopProp}} : {}),
            metadataTimeoutMs: 10000,
        })
        return params.registerExternalTexture(source.getSource)
    })
}

/** The chosen layer's texture, or a transparent placeholder while no layer is chosen. */
export function layerTexture(params: GpuFragmentParams, prop: string): KitTexture {
    return once(params, `layer:${prop}`, () => params.getLayerTexture(prop) ?? createSwappableMediaTexture(params, {label: `${prop} (no layer)`, mipmaps: false}).kit)
}

// ── Helpers the rewritten calls go to ─────────────────────────────────────────────────────

const helpers = new Map<string, unknown>()

function helper(key: string, make: () => unknown): {key: string; fn: unknown} {
    if (!helpers.has(key)) helpers.set(key, make())
    return {key, fn: helpers.get(key)}
}

type StringFn = (impl: string) => {$uses: (deps: Record<string, unknown>) => unknown}
const shell = (args: unknown[], ret: unknown) => (tgpu.fn as unknown as (a: unknown[], r: unknown) => StringFn)(args, ret)

/**
 * A fitted image or video sample at `p`, linear with straight alpha, transparent outside.
 * Images read the mip level that matches how large the picture is drawn (an explicit level, so
 * the call is legal anywhere in the body, after early returns too).
 */
function mediaHelper(fit: Fit, external: boolean) {
    return helper(`media:${fit}:${external ? 'ext' : 'tex'}`, () => {
        const scale = fit === 'fill' ? 'vec2f(1.0)' : `fitScale(dims, viewport)`
        const sample = external
            ? 'textureSampleBaseClampToEdge(tex, samp, q)'
            : 'textureSampleLevel(tex, samp, q, log2(max(1.0, dims.x / max(viewport.x * s.x, 1.0))))'
        const deps: Record<string, unknown> = {sampleUV: media.sampleUV, alphaCut: media.alphaCutLinear}
        if (fit !== 'fill') deps.fitScale = fit === 'cover' ? media.scaleCover : media.scaleContain
        return shell([external ? d.textureExternal() : d.texture2d(d.f32), d.sampler(), d.vec2f, d.vec2f], d.vec4f)(`(tex: ${external ? 'texture_external' : 'texture_2d<f32>'}, samp: sampler, viewport: vec2f, p: vec2f) -> vec4f {
  let dims = vec2f(textureDimensions(tex));
  let s = ${scale};
  let q = sampleUV(p, s);
  return alphaCut(${sample}, q);
}`).$uses(deps)
    })
}

/** A layer's colour at `p`, unpremultiplied to straight alpha. */
function layerHelper() {
    return helper('layer', () => shell([d.texture2d(d.f32), d.sampler(), d.vec2f], d.vec4f)(`(tex: texture_2d<f32>, samp: sampler, p: vec2f) -> vec4f {
  let c = textureSampleLevel(tex, samp, p, 0.0);
  return select(vec4f(c.rgb / c.a, c.a), vec4f(0.0), c.a <= 0.0);
}`).$uses({}))
}

/** Back-conversion from each colour space's working space to linear P3 (mode 0 is P3 already). */
const BACK_CONVERSIONS: Record<number, string> = {
    0: 's.conv',
    1: 'sRGBToP3(oklabToRgb(oklchToOklab(s.conv)))',
    2: 'sRGBToP3(oklabToRgb(s.conv))',
    3: 'sRGBToP3(hslToRgb(s.conv))',
    4: 'sRGBToP3(hsvToRgb(s.conv))',
    5: 'sRGBToP3(labToRgb(lchToLab(s.conv)))',
}

// The stops loop is the per-mode variant: cylindrical spaces take the shortest hue arc.
function gradientHelper(mode: number) {
    const back = BACK_CONVERSIONS[mode] ?? BACK_CONVERSIONS[0]!
    return helper(`gradient:${mode}`, () => shell(
        [d.arrayOf(d.vec4f, MAX_COLOR_STOPS), d.arrayOf(d.vec4f, MAX_COLOR_STOPS / 4), d.arrayOf(d.vec3f, MAX_COLOR_STOPS), d.f32, d.f32],
        d.vec4f,
    )(`(colors: array<vec4f, ${MAX_COLOR_STOPS}>, positions: array<vec4f, ${MAX_COLOR_STOPS / 4}>, converted: array<vec3f, ${MAX_COLOR_STOPS}>, count: f32, t: f32) -> vec4f {
  let s = stops(t, colors, positions, converted, count);
  return vec4f(${back}, s.alpha);
}`).$uses({stops: gradientStopsInSpaceVariants[mode as keyof typeof gradientStopsInSpaceVariants] ?? gradientStopsInSpace, sRGBToP3, oklabToRgb, oklchToOklab, hslToRgb, hsvToRgb, labToRgb, lchToLab}))
}

/** The compile-time colour-space mode, from the transformed number or the raw option. */
function colorSpaceMode(params: GpuFragmentParams): number {
    const raw = params.propValues.colorSpace
    if (typeof raw === 'number') return raw
    return typeof raw === 'string' ? Number(transformColorSpace(raw as never)) || 0 : 0
}

const SAMPLER = () => expr('samp.$.linearClamp')

/** A premultiplied texture's colour at `uv`, straight alpha (how a body's `child` reads an RTT). */
export function sampleStraight(texture: KitTexture, uv: Expr): Expr {
    return call(layerHelper().fn, 'sampleStraight', [texture.accessor(), SAMPLER(), uv])
}

// ── Plans ─────────────────────────────────────────────────────────────────────────────────

/** True when a prop config is one this module binds (rather than a plain uniform). */
export function isResourceProp(config: PropConfig<unknown>): boolean {
    const type = Array.isArray(config.ui?.type) ? config.ui?.type[0] : config.ui?.type
    return type === 'image-upload' || type === 'video-upload' || type === 'layer'
        || config.transform === (colorStopsTransform as unknown) || config.transform === (listPropTransform as unknown)
}

/** The bindings and callables a resource prop gives a body; null for a plain prop. */
export function resourcePlan(name: string, config: PropConfig<unknown>, props: Record<string, PropConfig<unknown>>, coords: HostCoords): ResourcePlan | null {
    const type = Array.isArray(config.ui?.type) ? config.ui?.type[0] : config.ui?.type
    const alias = `shaders_${name}`

    if (type === 'image-upload' || type === 'video-upload' || type === 'layer') {
        const fit: Fit = (config.ui?.fit as Fit | undefined) ?? 'cover'
        const external = type === 'video-upload'
        const loopProp = props[`${name}Loop`] ? `${name}Loop` : null
        const texture = (params: GpuFragmentParams) =>
            type === 'image-upload' ? imageTexture(params, name) : type === 'video-upload' ? videoTexture(params, name, loopProp) : layerTexture(params, name)
        const sampleWith = type === 'layer'
            ? (params: GpuFragmentParams, p: Expr) => call(layerHelper().fn, 'layerSample', [texture(params).accessor(), SAMPLER(), p])
            : (params: GpuFragmentParams, p: Expr) => call(mediaHelper(fit, external).fn, 'mediaSample', [texture(params).accessor(), SAMPLER(), coords.viewport(params), p])
        return {
            bindings: [
                {name, type: 'vec4f', value: (params) => sampleWith(params, coords.uv(params))},
                {name: `${name}Texture`, type: external ? 'texture_external' : 'texture_2d<f32>', value: (params) => texture(params).accessor()},
                {name: `${name}Sampler`, type: 'sampler', value: SAMPLER},
                {name: `${name}_viewport`, type: 'vec2f', value: coords.viewport},
            ],
            callables: [{
                name,
                alias,
                leading: type === 'layer' ? [`${name}Texture`, `${name}Sampler`] : [`${name}Texture`, `${name}Sampler`, `${name}_viewport`],
                helper: () => (type === 'layer' ? layerHelper() : mediaHelper(fit, external)),
            }],
            bareValues: [name],
        }
    }

    if (config.transform === (colorStopsTransform as unknown)) {
        // The bridge packs a component's one stops prop into fixed-name uniforms.
        const uniform = (key: string) => (params: GpuFragmentParams) => params.uniforms[key]
        return {
            bindings: [
                {name: `${name}_stopColors`, type: `array<vec4f, ${MAX_COLOR_STOPS}>`, value: uniform('colorsArray')},
                {name: `${name}_stopPositions`, type: `array<vec4f, ${MAX_COLOR_STOPS / 4}>`, value: uniform('positionsArray')},
                {name: `${name}_stopConverted`, type: `array<vec3f, ${MAX_COLOR_STOPS}>`, value: uniform('convertedColorsArray')},
                {name: `${name}_stopCount`, type: 'f32', value: uniform('stopCount')},
            ],
            callables: [{
                name,
                alias,
                leading: [`${name}_stopColors`, `${name}_stopPositions`, `${name}_stopConverted`, `${name}_stopCount`],
                helper: (params) => gradientHelper(colorSpaceMode(params)),
            }],
            bareValues: [],
        }
    }

    if (config.transform === (listPropTransform as unknown)) {
        const spec = listSpecOf(config)
        if (!spec) return null
        return {
            bindings: [
                ...Object.keys(spec.item).map((field): ResourceBinding => ({
                    name: listFieldName(name, field),
                    type: `array<vec4f, ${spec.maxItems}>`,
                    value: (params) => params.uniforms[listFieldName(name, field)],
                })),
                {name: listCountName(name), type: 'f32', value: (params) => params.uniforms[listCountName(name)]},
            ],
            callables: [],
            bareValues: [],
        }
    }

    return null
}

/** The typegpu schema for a binding type. */
export function arraySchema(type: string): unknown | null {
    const match = /^array<(vec4f|vec3f), (\d+)>$/.exec(type)
    if (!match) return null
    return d.arrayOf(match[1] === 'vec4f' ? d.vec4f : d.vec3f, Number(match[2]))
}
