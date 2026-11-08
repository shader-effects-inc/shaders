/**
 * std/document — the op catalog: every node type a graph may use, each one a std word with a
 * typed signature and a `build` that calls the real function. The catalog is the whole
 * contract: the validator types a graph against `inputs`/`output`, the interpreter runs
 * `build`, and `renderStdGraphManifest` (define.ts) prints it for the AI composer. Adding a
 * word to std makes it available here by adding one entry — nothing else changes.
 *
 * Levels, mirroring STD_LANGUAGE_GUIDE §1: coordinate words, geometry, masks, motion, noise,
 * the field pipeline (composition-time `Field` objects sampled via `sample`/`ramp`), light,
 * colour, compose. The algebra (`add`, `mul`, …) is seasoning — the manifest says so.
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {colorMixing} from '../../gpu/kit/index'
import {Scalar, type PropRef} from '../values'
import {uniformOf, type ArgSpec} from '../invoke'
import * as m from '../math'
import {centredFrame, direction, directionFrame, segmentFrame, sectorFold} from '../frames'
import * as shape from '../shape'
import {crispFill, crispStroke, softBand, softDisc} from '../mask'
import {cycle, cycleSeed, easeIn, easeInOut, easeOut, oscillate, phase, pulseTrain} from '../motion'
import {
    colorLadder3, colorSpaceModeOf, dist, flowField, hueWheel, noiseField, pair,
    pulse, rings, scaledVolume, scatterField, threshold, tiles, tone, warped, warpedPoint, wrapRamp, foldRamp,
    type Field, type Palette, type FieldScalar,
} from '../paint/fields'
import {
    cellBorders, cellDistances, cellFill, curlSpeed, erosionRidges, evolving, gaborGrains, noiseTone,
    organicWaves, scaledPlane, seededPlane, unitized, waves, wavyLine, waveletBands, type WaveTerm,
} from '../paint/noise'
import {domeNormal, emitterFalloff, glowSpot, rayBands, shine} from '../paint/light'
import {
    cellNoiseAt, cosineRainbow, exposureTone, fdSlope, hashNoise, inverseSquare,
    lambert, pointStars, schlickFresnel, sensorGrain, surfaceNoiseAt, twinkle, valueNoise, volumeNoiseAt,
} from '../paint/materials'
import {dithered, emissiveAlpha, layers as layersWord, type Layer} from '../paint/compose'
import {sineStroke, strokedSegment} from '../paint/figures'
import {turbulentMedium, volumeMarch} from '../paint/volume'
import type {GraphPropKind, GraphValueType} from './types'
import {StdGraphError} from './types'

// ── Values ──────────────────────────────────────────────────────────────────────────────

export type NumericType = 'f32' | 'vec2' | 'vec3' | 'vec4'

/** A resolved node value at build time. */
export type Val =
    | {type: NumericType; expr: Expr; ref?: PropRef; propKind?: GraphPropKind}
    | {type: 'literal'; value: number}
    | {type: 'enum'; value: string}
    | {type: 'data'; value: unknown}
    | {type: 'field'; field: Field; valueType?: NumericType}
    | {type: 'palette'; palette: Palette}
    | {type: 'schedule'; value: WaveTerm[]}
    | {type: 'record'; fields: Record<string, Val>}

/** The static type of a value, as the validator sees it (a literal types as f32 or const). */
export type StaticType =
    | {type: GraphValueType | 'literal' | 'enum' | 'data'; ref?: true; propKind?: GraphPropKind; fields?: Record<string, StaticType>; valueType?: NumericType}

/** What an input accepts. */
export type InputType =
    | NumericType
    | 'num' // any numeric (f32 or vector)
    | 'vec' // vec2 | vec3 | vec4
    | 'const' // a JS number literal (unrolled counts, exponents)
    | 'enum'
    | 'data'
    | 'field'
    | 'palette'
    | 'schedule'
    | 'prop:float' // a declared float/angle/bool prop (bound by name)
    | 'prop:color'
    | 'prop:position'

export interface InputSpec {
    type: InputType | InputType[]
    optional?: boolean
    doc: string
    /** `enum` inputs: the allowed strings. */
    values?: string[]
}

export type OutputRule =
    | GraphValueType
    | 'broadcast' // the widest numeric input type (all vectors must agree)
    | {record: Record<string, GraphValueType>}
    | ((inputs: Record<string, StaticType | undefined>) => GraphValueType | {record: Record<string, GraphValueType>})

export interface BuildCtx {
    params: GpuFragmentParams
    /** The composed UV (`uvContext ?? ctx.uv`), y = 0 at the top. */
    uv: Expr
    aspect: Expr
    viewport: Expr
    /** The node clock (`animatedTime`, scaled by the `speed` prop). */
    time: () => Expr
    /** The colour-space mode of the component's `colorSpace` prop (compile-time). */
    colorSpaceMode: number
    /** Resolve a `"$ref"` inside a `data` argument (layer lists). */
    resolveRef: (ref: string) => Val
}

export interface OpDef {
    section: string
    doc: string
    inputs: Record<string, InputSpec>
    output: OutputRule
    build: (args: Record<string, Val | undefined>, ctx: BuildCtx) => Val
    /** Control ops carry a body; handled by the interpreter, documented here. */
    control?: 'repeat' | 'fieldOf' | 'march' | 'wgsl'
}

// ── Coercion helpers ────────────────────────────────────────────────────────────────────

const isNumeric = (t: string): t is NumericType => t === 'f32' || t === 'vec2' || t === 'vec3' || t === 'vec4'

/** A numeric (GPU-expression) value. */
export type NumVal = Extract<Val, {expr: Expr}>
export const isNumVal = (v: Val | undefined): v is NumVal => !!v && isNumeric(v.type)

export function num(v: Val | undefined, name = 'value'): Expr {
    if (!v) throw new StdGraphError(`missing input "${name}"`)
    if (v.type === 'literal') return floatE(v.value)
    if (isNumVal(v)) return v.expr
    throw new StdGraphError(`input "${name}" is not numeric (${v.type})`)
}

function numOr(v: Val | undefined, fallback: number): Expr {
    return v ? num(v) : floatE(fallback)
}

export function konst(v: Val | undefined, name = 'value'): number {
    if (!v) throw new StdGraphError(`missing input "${name}"`)
    if (v.type === 'literal') return v.value
    throw new StdGraphError(`input "${name}" must be a number literal`)
}

function konstOr(v: Val | undefined, fallback: number): number {
    return v ? konst(v) : fallback
}

function enumOf(v: Val | undefined, fallback?: string): string {
    if (!v) {
        if (fallback !== undefined) return fallback
        throw new StdGraphError('missing enum input')
    }
    if (v.type === 'enum') return v.value
    throw new StdGraphError(`expected an enum string, got ${v.type}`)
}

function fieldOf(v: Val | undefined, name = 'field'): Field {
    if (!v || v.type !== 'field') throw new StdGraphError(`input "${name}" must be a field`)
    return v.field
}

function paletteOf(v: Val | undefined): Palette {
    if (!v || v.type !== 'palette') throw new StdGraphError('expected a palette')
    return v.palette
}

function propRef(v: Val | undefined, name: string): PropRef {
    if (!isNumVal(v) || !v.ref) throw new StdGraphError(`input "${name}" must be a prop reference ("@name")`)
    return v.ref
}

/** Hand an arbitrary expression to a std slot typed `ArgSpec` / `FieldScalar` (a signal Scalar). */
function asSpec(v: Val | undefined, fallback?: number): ArgSpec {
    if (!v) {
        if (fallback === undefined) throw new StdGraphError('missing slot value')
        return fallback
    }
    if (v.type === 'literal') return v.value
    if (isNumVal(v)) {
        if (v.ref) return v.ref
        const e = v.expr
        return new Scalar({kind: 'signal', build: () => e})
    }
    throw new StdGraphError(`slot value is not numeric (${v.type})`)
}

const f32 = (expr: Expr): Val => ({type: 'f32', expr})
const vec2v = (expr: Expr): Val => ({type: 'vec2', expr})
const vec3v = (expr: Expr): Val => ({type: 'vec3', expr})
const vec4v = (expr: Expr): Val => ({type: 'vec4', expr})
const fieldV = (field: Field): Val => ({type: 'field', field})
const paletteV = (palette: Palette): Val => ({type: 'palette', palette})
const record = (fields: Record<string, Val>): Val => ({type: 'record', fields})

/** The numeric type of a value (literals are f32). */
function typeOf(v: Val | undefined): NumericType {
    if (!v) return 'f32'
    if (v.type === 'literal') return 'f32'
    if (isNumeric(v.type)) return v.type
    throw new StdGraphError(`value is not numeric (${v.type})`)
}

/** Output type of a broadcast op given its numeric inputs. */
function broadcastType(vals: (Val | undefined)[]): NumericType {
    let out: NumericType = 'f32'
    for (const v of vals) {
        const t = typeOf(v)
        if (t !== 'f32') out = t
    }
    return out
}

const same = (v: Val | undefined, expr: Expr): Val => ({type: typeOf(v), expr})
const bcast = (vals: (Val | undefined)[], expr: Expr): Val => ({type: broadcastType(vals), expr})

// ── Input shorthands ────────────────────────────────────────────────────────────────────

const N = (doc: string, optional = false): InputSpec => ({type: 'num', doc, optional})
const F = (doc: string, optional = false): InputSpec => ({type: 'f32', doc, optional})
const V2 = (doc: string, optional = false): InputSpec => ({type: 'vec2', doc, optional})
const V3 = (doc: string, optional = false): InputSpec => ({type: 'vec3', doc, optional})
const V4 = (doc: string, optional = false): InputSpec => ({type: 'vec4', doc, optional})
const VEC = (doc: string): InputSpec => ({type: 'vec', doc})
const C = (doc: string, optional = false): InputSpec => ({type: 'const', doc, optional})
const FIELD = (doc: string): InputSpec => ({type: 'field', doc})
const PCOLOR = (doc: string): InputSpec => ({type: 'prop:color', doc})
const PPOS = (doc: string, optional = false): InputSpec => ({type: 'prop:position', doc, optional})
const PFLOAT = (doc: string, optional = false): InputSpec => ({type: 'prop:float', doc, optional})
const ENUM = (values: string[], doc: string, optional = false): InputSpec => ({type: 'enum', values, doc, optional})

const unary = (section: string, doc: string, fn: (x: Expr) => Expr): OpDef => ({
    section, doc, inputs: {x: N('')}, output: 'broadcast',
    build: (a) => same(a.x, fn(num(a.x, 'x'))),
})
const binary = (section: string, doc: string, fn: (a: Expr, b: Expr) => Expr): OpDef => ({
    section, doc, inputs: {a: N(''), b: N('')}, output: 'broadcast',
    build: (a) => bcast([a.a, a.b], fn(num(a.a, 'a'), num(a.b, 'b'))),
})

// ── The catalog ─────────────────────────────────────────────────────────────────────────

export const CATALOG: Record<string, OpDef> = {
    // ── Algebra (seasoning — thresholds, ramps, gains over what the words hand you) ──
    add: binary('algebra', 'a + b', m.add),
    sub: binary('algebra', 'a − b', m.sub),
    mul: binary('algebra', 'a × b (scalar × vector broadcasts)', m.mul),
    div: binary('algebra', 'a ÷ b', m.div),
    min: binary('algebra', 'min(a, b)', m.min),
    max: binary('algebra', 'max(a, b)', m.max),
    pow: binary('algebra', 'a ^ b (a should be ≥ 0)', (a, b) => m.pow(m.max(a, 0), b)),
    step: binary('algebra', '0 below edge a, 1 at/above: step(a, b) = b ≥ a ? 1 : 0', m.step),
    distance: {
        section: 'algebra', doc: 'Euclidean distance between two points/vectors',
        inputs: {a: VEC(''), b: VEC('')}, output: 'f32',
        build: (a) => f32(m.distance(num(a.a, 'a'), num(a.b, 'b'))),
    },
    dot: {
        section: 'algebra', doc: 'dot product', inputs: {a: VEC(''), b: VEC('')}, output: 'f32',
        build: (a) => f32(m.dot(num(a.a, 'a'), num(a.b, 'b'))),
    },
    length: {section: 'algebra', doc: 'vector length', inputs: {v: VEC('')}, output: 'f32', build: (a) => f32(m.length(num(a.v, 'v')))},
    normalize: {section: 'algebra', doc: 'unit vector', inputs: {v: VEC('')}, output: (i) => (i.v?.type as GraphValueType) ?? 'vec2', build: (a) => same(a.v, m.normalize(num(a.v, 'v')))},
    neg: unary('algebra', '−x', m.neg),
    abs: unary('algebra', '|x|', m.abs),
    floor: unary('algebra', 'floor(x)', m.floor),
    fract: unary('algebra', 'fract(x) — the repeating 0..1 sawtooth', m.fract),
    sqrt: unary('algebra', '√x (x clamped ≥ 0)', (x) => m.sqrt(m.max(x, 0))),
    exp: unary('algebra', 'e^x', m.exp),
    sin: unary('algebra', 'sin(x) (radians)', m.sin),
    cos: unary('algebra', 'cos(x) (radians)', m.cos),
    saturate: unary('algebra', 'clamp(x, 0, 1)', (x) => m.clamp(x, 0, 1)),
    oneMinus: unary('algebra', '1 − x', (x) => m.sub(1, x)),
    atan2: {section: 'algebra', doc: 'atan2(y, x) in radians (−π..π)', inputs: {y: F(''), x: F('')}, output: 'f32', build: (a) => f32(m.atan2(num(a.y, 'y'), num(a.x, 'x')))},
    clamp: {
        section: 'algebra', doc: 'clamp(x, lo, hi)', inputs: {x: N(''), lo: N(''), hi: N('')}, output: 'broadcast',
        build: (a) => same(a.x, m.clamp(num(a.x, 'x'), num(a.lo, 'lo'), num(a.hi, 'hi'))),
    },
    mix: {
        section: 'algebra', doc: 'linear blend a→b by t (t may be a scalar or a matching vector)',
        inputs: {a: N(''), b: N(''), t: N('0 = a, 1 = b')}, output: 'broadcast',
        build: (a) => bcast([a.a, a.b], m.mix(num(a.a, 'a'), num(a.b, 'b'), num(a.t, 't'))),
    },
    smoothstep: {
        section: 'algebra', doc: 'smooth 0→1 ramp as x rises from e0 to e1 — THE soft threshold',
        inputs: {e0: N(''), e1: N(''), x: N('')}, output: 'broadcast',
        build: (a) => same(a.x, m.smoothstep(num(a.e0, 'e0'), num(a.e1, 'e1'), num(a.x, 'x'))),
    },
    remap: {
        section: 'algebra', doc: 'map x from [from0, from1] onto [to0, to1] (unclamped)',
        inputs: {x: N(''), from0: F(''), from1: F(''), to0: N(''), to1: N('')}, output: 'broadcast',
        build: (a) => {
            const x = num(a.x, 'x')
            const t = m.div(m.sub(x, num(a.from0, 'from0')), m.sub(num(a.from1, 'from1'), num(a.from0, 'from0')))
            return bcast([a.x, a.to0, a.to1], m.add(num(a.to0, 't0'), m.mul(t, m.sub(num(a.to1, 't1'), num(a.to0, 't0')))))
        },
    },
    ifLess: {
        section: 'algebra', doc: 'a < b ? then : else',
        inputs: {a: F(''), b: F(''), then: N(''), else: N('')}, output: 'broadcast',
        build: (a) => bcast([a.then, a.else], m.select(m.lt(num(a.a, 'a'), num(a.b, 'b')), num(a.then, 'then'), num(a.else, 'else'))),
    },
    gaussian: {
        section: 'algebra', doc: 'exp(−(d/width)²) — a soft bell over a distance (glow cores, falloffs)',
        inputs: {d: N('distance'), width: N('where the bell has fallen to 1/e')}, output: 'broadcast',
        build: (a) => same(a.d, m.gaussBell(m.div(num(a.d, 'd'), num(a.width, 'width')), 1)),
    },

    // ── Vectors ──
    vec2: {section: 'vectors', doc: 'vec2(x, y)', inputs: {x: F(''), y: F('')}, output: 'vec2', build: (a) => vec2v(m.vec2(num(a.x, 'x'), num(a.y, 'y')))},
    vec3: {section: 'vectors', doc: 'vec3(x, y, z)', inputs: {x: F(''), y: F(''), z: F('')}, output: 'vec3', build: (a) => vec3v(m.vec3(num(a.x, 'x'), num(a.y, 'y'), num(a.z, 'z')))},
    vec4: {section: 'vectors', doc: 'vec4(x, y, z, w)', inputs: {x: F(''), y: F(''), z: F(''), w: F('')}, output: 'vec4', build: (a) => vec4v(m.vec4(num(a.x, 'x'), num(a.y, 'y'), num(a.z, 'z'), num(a.w, 'w')))},
    splat3: {section: 'vectors', doc: 'vec3(x, x, x) — a grey / a scalar as rgb', inputs: {x: F('')}, output: 'vec3', build: (a) => vec3v(m.splat3(num(a.x, 'x')))},
    rgba: {section: 'vectors', doc: 'rgb + alpha → vec4 (straight alpha; alpha 1 = opaque)', inputs: {rgb: V3(''), a: F('')}, output: 'vec4', build: (a) => vec4v(m.vec4(num(a.rgb, 'rgb'), num(a.a, 'a')))},
    rgb: {section: 'vectors', doc: 'vec3 from three scalars (0..1 each)', inputs: {r: F(''), g: F(''), b: F('')}, output: 'vec3', build: (a) => vec3v(m.vec3(num(a.r, 'r'), num(a.g, 'g'), num(a.b, 'b')))},
    xyPlusZ: {section: 'vectors', doc: 'lift a vec2 to vec3 with a z (noise3 domains: xy + time)', inputs: {xy: V2(''), z: F('')}, output: 'vec3', build: (a) => vec3v(m.vec3(num(a.xy, 'xy').member('x'), num(a.xy, 'xy').member('y'), num(a.z, 'z')))},
    rotate2: {
        section: 'vectors', doc: 'rotate a 2D point about the origin by angle (degrees)',
        inputs: {p: V2(''), angle: F('degrees')}, output: 'vec2',
        build: (a) => {
            const rad = m.mul(num(a.angle, 'angle'), 0.017453292519943295)
            return vec2v(m.rotate2(num(a.p, 'p'), m.cos(rad), m.sin(rad)))
        },
    },

    // ── Coordinates & frames ──
    aspectUV: {
        section: 'coordinates', doc: 'the pixel in aspect-corrected screen space: (uv.x·aspect, uv.y). Square units — use for distances/shapes',
        inputs: {}, output: 'vec2',
        build: (_a, ctx) => vec2v(m.vec2(m.mul(ctx.uv.member('x'), ctx.aspect), ctx.uv.member('y'))),
    },
    centered: {
        section: 'coordinates', doc: 'aspect-corrected frame around a position prop (or the canvas centre): `delta` = pixel − centre (vec2), `dist` = its length. The opening move of any radial look',
        inputs: {center: PPOS('a position prop; default = canvas centre', true)},
        output: {record: {delta: 'vec2', dist: 'f32'}},
        build: (a, ctx) => {
            if (a.center) {
                const fr = centredFrame({center: propRef(a.center, 'center')})(ctx.params, {uv: ctx.uv, aspect: ctx.aspect})
                return record({delta: vec2v(fr.delta), dist: f32(fr.dist)})
            }
            const delta = m.local(m.vec2(m.mul(m.sub(ctx.uv.member('x'), 0.5), ctx.aspect), m.sub(ctx.uv.member('y'), 0.5)), 'centred')
            return record({delta: vec2v(delta), dist: f32(m.local(m.length(delta), 'centredDist'))})
        },
    },
    placed: {
        section: 'coordinates', doc: 'a position prop in the same space as `aspectUV` (x·aspect, y) — compare it to aspectUV directly',
        inputs: {position: PPOS('')}, output: 'vec2',
        build: (a, ctx) => {
            const pos = uniformOf(propRef(a.position, 'position'), ctx.params)
            return vec2v(m.vec2(m.mul(pos.member('x'), ctx.aspect), m.sub(1, pos.member('y'))))
        },
    },
    direction: {section: 'coordinates', doc: 'degrees → unit vector (cos θ, sin θ)', inputs: {angle: F('degrees')}, output: 'vec2', build: (a) => vec2v(direction(num(a.angle, 'angle')))},
    directionFrame: {
        section: 'coordinates', doc: 'project a point into a rotated basis: `along`/`across` the direction (brush grain, streaks, weave, travel axis), plus `tangent`/`perp` unit vectors',
        inputs: {angle: F('degrees'), p: V2('the point (usually aspectUV or a centered delta)')},
        output: {record: {along: 'f32', across: 'f32', tangent: 'vec2', perp: 'vec2'}},
        build: (a) => {
            const fr = directionFrame(num(a.angle, 'angle'))
            const c = fr.coordsOf(num(a.p, 'p'))
            return record({along: f32(c.along), across: f32(c.across), tangent: vec2v(fr.tangent), perp: vec2v(fr.perp)})
        },
    },
    segment: {
        section: 'coordinates', doc: 'frame along the segment between two position props: `along` (0 at from), signed `across`, and the segment `length`',
        inputs: {from: PPOS(''), to: PPOS('')},
        output: {record: {along: 'f32', across: 'f32', length: 'f32'}},
        build: (a, ctx) => {
            const fr = segmentFrame({from: propRef(a.from, 'from'), to: propRef(a.to, 'to')})(ctx.params, {uv: ctx.uv, aspect: ctx.aspect})
            return record({along: f32(fr.along), across: f32(fr.across), length: f32(fr.length)})
        },
    },
    polar: {
        section: 'coordinates', doc: 'polar coordinates of a vector: `radius`, `angle` (radians −π..π), `turns` (angle as 0..1)',
        inputs: {v: V2('usually a centered delta')},
        output: {record: {radius: 'f32', angle: 'f32', turns: 'f32'}},
        build: (a) => {
            const v = num(a.v, 'v')
            const angle = m.local(m.atan2(v.member('y'), v.member('x')), 'polarA')
            return record({
                radius: f32(m.local(m.length(v), 'polarR')),
                angle: f32(angle),
                turns: f32(m.fract(m.add(m.div(angle, 6.283185307179586), 0.5))),
            })
        },
    },
    sectorFold: {
        section: 'coordinates', doc: 'kaleidoscope fold: mirror a point into one angular sector of `sectors` around (0.5, 0.5)',
        inputs: {p: V2('a uv-space point'), sectors: F('how many sectors')}, output: 'vec2',
        build: (a) => {
            const p = num(a.p, 'p')
            return vec2v(sectorFold(p.member('x'), p.member('y'), m.div(6.283185307179586, num(a.sectors, 'sectors'))))
        },
    },

    // ── Geometry (signed distance fields; negative inside) ──
    sdfCircle: {section: 'geometry', doc: 'circle SDF at p (p relative to the centre)', inputs: {p: V2(''), radius: F('')}, output: 'f32', build: (a) => f32(shape.circle(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.radius, 'radius')))},
    sdfEllipse: {section: 'geometry', doc: 'ellipse SDF', inputs: {p: V2(''), rx: F(''), ry: F('')}, output: 'f32', build: (a) => f32(shape.ellipse(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.rx, 'rx'), num(a.ry, 'ry')))},
    sdfRoundedRect: {section: 'geometry', doc: 'rounded rectangle SDF (full width/height)', inputs: {p: V2(''), width: F(''), height: F(''), rounding: F('')}, output: 'f32', build: (a) => f32(shape.roundedRect(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.width, 'width'), num(a.height, 'height'), num(a.rounding, 'rounding')))},
    sdfPolygon: {section: 'geometry', doc: 'regular polygon SDF', inputs: {p: V2(''), radius: F(''), sides: F('')}, output: 'f32', build: (a) => f32(shape.polygon(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.radius, 'radius'), num(a.sides, 'sides')))},
    sdfStar: {section: 'geometry', doc: 'star SDF', inputs: {p: V2(''), radius: F(''), points: F(''), innerRatio: F('0..1')}, output: 'f32', build: (a) => f32(shape.star(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.radius, 'radius'), num(a.points, 'points'), num(a.innerRatio, 'innerRatio')))},
    sdfHeart: {section: 'geometry', doc: 'heart SDF', inputs: {p: V2(''), radius: F('')}, output: 'f32', build: (a) => f32(shape.heart(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.radius, 'radius')))},
    sdfRing: {section: 'geometry', doc: 'ring (annulus) SDF', inputs: {p: V2(''), radius: F(''), thickness: F('')}, output: 'f32', build: (a) => f32(shape.ring(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.radius, 'radius'), num(a.thickness, 'thickness')))},
    sdfCross: {section: 'geometry', doc: 'plus/cross SDF', inputs: {p: V2(''), size: F(''), thickness: F(''), rounding: F('')}, output: 'f32', build: (a) => f32(shape.cross(num(a.p, 'p').member('x'), num(a.p, 'p').member('y'), num(a.size, 'size'), num(a.thickness, 'thickness'), num(a.rounding, 'rounding')))},
    sdfLine: {
        section: 'geometry', doc: 'unsigned distance from p to the segment a→b (draw a line with `stroke`)',
        inputs: {p: V2(''), a: V2(''), b: V2('')}, output: 'f32',
        build: (a) => {
            const p = num(a.p, 'p'), A = num(a.a, 'a'), B = num(a.b, 'b')
            const ab = m.local(m.sub(B, A), 'ab')
            const t = m.clamp(m.div(m.dot(m.sub(p, A), ab), m.max(m.dot(ab, ab), 1e-6)), 0, 1)
            return f32(m.length(m.sub(p, m.add(A, m.mul(ab, t)))))
        },
    },

    // ── Masks (coverage 0..1 from distances) ──
    fill: {section: 'masks', doc: 'fill an SDF: 1 inside, 0 outside, feathered over `footprint` (≈ 0.003 for crisp AA, larger for a soft edge)', inputs: {distance: F(''), footprint: N('')}, output: 'f32', build: (a) => f32(crispFill({distance: num(a.distance, 'distance'), footprint: num(a.footprint, 'footprint')}))},
    stroke: {section: 'masks', doc: 'outline an SDF / draw a curve: a band of `width` on the zero contour', inputs: {distance: F(''), width: N(''), footprint: N('')}, output: 'f32', build: (a) => f32(crispStroke({distance: num(a.distance, 'distance'), width: num(a.width, 'width'), footprint: num(a.footprint, 'footprint')}))},
    softBand: {
        section: 'masks', doc: 'asymmetric soft band around a signed distance: fades in across [belowFrom, belowTo], out across [aboveFrom, aboveTo]. Horizons, curtains, beam falloffs',
        inputs: {distance: F('signed distance from the line'), belowFrom: N(''), belowTo: N(''), aboveFrom: N('', true), aboveTo: N('')}, output: 'f32',
        build: (a) => f32(softBand({distance: num(a.distance, 'distance'), below: {from: num(a.belowFrom, 'belowFrom'), to: num(a.belowTo, 'belowTo')}, above: {from: a.aboveFrom ? num(a.aboveFrom) : undefined, to: num(a.aboveTo, 'aboveTo')}})),
    },
    softDisc: {section: 'masks', doc: 'disc coverage with a soft, curved edge (blobs, orbs, glow dots)', inputs: {dist: F(''), radius: N(''), width: N('edge half-width'), curve: N('edge gamma; 1 = linear', true)}, output: 'f32', build: (a) => f32(softDisc({dist: num(a.dist, 'dist'), radius: num(a.radius, 'radius'), softness: {width: num(a.width, 'width'), curve: numOr(a.curve, 1)}}))},
    vignette: {
        section: 'masks', doc: 'darkening toward the corners: 1 at centre → 0 at the edges (strength 0..1, power shapes the curve)',
        inputs: {strength: N(''), power: N('', true)}, output: 'f32',
        build: (a, ctx) => {
            const d = m.length(m.mul(m.sub(ctx.uv, m.vec2(0.5, 0.5)), 1.4142))
            return f32(m.sub(1, m.mul(m.pow(m.clamp(d, 0, 1), numOr(a.power, 2)), num(a.strength, 'strength'))))
        },
    },

    // ── Motion (time is the node clock in seconds × speed) ──
    phase: {section: 'motion', doc: 'position within a repeating cycle, 0..1', inputs: {t: F(''), period: N('seconds')}, output: 'f32', build: (a) => f32(phase(num(a.t, 't'), num(a.period, 'period')))},
    cycle: {section: 'motion', doc: 'a repeating cycle: `progress` 0..1 and integer `index` — feed index to cycleSeed for a random-but-stable value per pass', inputs: {t: F(''), period: N('')}, output: {record: {progress: 'f32', index: 'f32'}}, build: (a) => { const c = cycle(num(a.t, 't'), num(a.period, 'period')); return record({progress: f32(c.progress), index: f32(c.index)}) }},
    cycleSeed: {section: 'motion', doc: 'random-but-stable 0..1 per cycle index (salt for independent draws)', inputs: {index: F(''), salt: C('', true)}, output: 'f32', build: (a) => f32(cycleSeed(num(a.index, 'index'), konstOr(a.salt, 0)))},
    pulseTrain: {section: 'motion', doc: 'on/off rhythm: 1 for the first `duty` of each period, else 0; `soften` eases edges', inputs: {t: F(''), period: N(''), duty: N('0..1'), soften: C('', true)}, output: 'f32', build: (a) => f32(pulseTrain(num(a.t, 't'), {period: num(a.period, 'period'), duty: num(a.duty, 'duty'), soften: konstOr(a.soften, 0)}))},
    oscillate: {section: 'motion', doc: 'sine sweep between min and max at rate cycles/second', inputs: {t: F(''), rate: C(''), min: N('', true), max: N('', true), offset: C('phase offset', true)}, output: 'f32', build: (a) => f32(oscillate(num(a.t, 't'), {rate: konst(a.rate, 'rate'), min: a.min ? num(a.min) : 0, max: a.max ? num(a.max) : 1, offset: konstOr(a.offset, 0)}))},
    easeIn: {section: 'motion', doc: 'x² (unit in → unit out)', inputs: {x: F('')}, output: 'f32', build: (a) => f32(easeIn(num(a.x, 'x')))},
    easeOut: {section: 'motion', doc: '1 − (1 − x)²', inputs: {x: F('')}, output: 'f32', build: (a) => f32(easeOut(num(a.x, 'x')))},
    easeInOut: {section: 'motion', doc: 'smoothstep(0, 1, x)', inputs: {x: F('')}, output: 'f32', build: (a) => f32(easeInOut(num(a.x, 'x')))},

    // ── Noise (expression-level; sample at any coordinate you build) ──
    noise2: {section: 'noise', doc: 'smooth signed noise −1..1 at a 2D point (the surface grain basis). Scale p for frequency, add time to an axis to drift', inputs: {p: V2('')}, output: 'f32', build: (a) => f32(surfaceNoiseAt(num(a.p, 'p')))},
    noise3: {section: 'noise', doc: 'smooth signed noise −1..1 at a 3D point — put time in z so the pattern MORPHS in place instead of sliding', inputs: {p: V3('')}, output: 'f32', build: (a) => f32(volumeNoiseAt(num(a.p, 'p')))},
    valueNoise: {section: 'noise', doc: 'soft blocky value noise 0..1 at a 2D point', inputs: {p: V2('')}, output: 'f32', build: (a) => f32(valueNoise(num(a.p, 'p').member('x'), num(a.p, 'p').member('y')))},
    cellNoise: {section: 'noise', doc: 'worley/cellular distance at a 2D point (0 at cell centres, rising toward borders) — scales, cracks, cells', inputs: {p: V2('')}, output: 'f32', build: (a) => f32(cellNoiseAt(num(a.p, 'p')))},
    hash: {section: 'noise', doc: 'white noise 0..1 at a 2D point (grain, sparkle, per-pixel jitter). Not smooth', inputs: {p: V2('')}, output: 'f32', build: (a) => f32(hashNoise(num(a.p, 'p').member('x'), num(a.p, 'p').member('y')))},
    hash1: {section: 'noise', doc: 'white noise 0..1 from one number — a random value per cell or per index (floor a coordinate first: a different brightness, size or phase per cell)', inputs: {x: F('')}, output: 'f32', build: (a) => f32(hashNoise(num(a.x, 'x'), 7.31))},
    fbm2: {
        section: 'noise', doc: 'fractal noise −1..1 at a 2D point: `octaves` layers of noise2, each `lacunarity`× finer and `gain`× fainter (clouds, terrain, water height)',
        inputs: {p: V2(''), octaves: C('1..6, default 4', true), lacunarity: C('default 2', true), gain: C('default 0.5', true)}, output: 'f32',
        build: (a) => {
            const p = num(a.p, 'p')
            const octaves = Math.max(1, Math.min(6, Math.round(konstOr(a.octaves, 4))))
            const lac = konstOr(a.lacunarity, 2), gain = konstOr(a.gain, 0.5)
            let acc: Expr | undefined, amp = 1, freq = 1, total = 0
            for (let i = 0; i < octaves; i++) {
                const term = m.mul(surfaceNoiseAt(m.add(m.mul(p, freq), m.vec2(i * 17.3, i * 31.7))), amp)
                acc = acc ? m.add(acc, term) : term
                total += amp
                amp *= gain
                freq *= lac
            }
            return f32(m.div(acc!, total))
        },
    },
    fbm3: {
        section: 'noise', doc: 'fractal noise −1..1 at a 3D point (xy + time in z morphs in place)',
        inputs: {p: V3(''), octaves: C('1..6, default 4', true), lacunarity: C('default 2', true), gain: C('default 0.5', true)}, output: 'f32',
        build: (a) => {
            const p = num(a.p, 'p')
            const octaves = Math.max(1, Math.min(6, Math.round(konstOr(a.octaves, 4))))
            const lac = konstOr(a.lacunarity, 2), gain = konstOr(a.gain, 0.5)
            let acc: Expr | undefined, amp = 1, freq = 1, total = 0
            for (let i = 0; i < octaves; i++) {
                const term = m.mul(volumeNoiseAt(m.add(m.mul(p, freq), m.vec3(i * 17.3, i * 31.7, i * 7.1))), amp)
                acc = acc ? m.add(acc, term) : term
                total += amp
                amp *= gain
                freq *= lac
            }
            return f32(m.div(acc!, total))
        },
    },
    turbulence: {
        section: 'noise', doc: 'a turbulent medium at a 3D point: domain-warped 3-octave density (gas, smoke, clouds, inclusions) plus its free `mid`/`fine` detail octaves',
        inputs: {p: V3(''), frequency: F(''), drift: F('animation phase (e.g. time × 0.2)', true), seed: F('', true), billow: F('warp strength, ~0.3', true)},
        output: {record: {density: 'f32', mid: 'f32', fine: 'f32'}},
        build: (a) => {
            const p = num(a.p, 'p')
            const r = turbulentMedium({x: p.member('x'), y: p.member('y'), z: p.member('z')}, {
                frequency: num(a.frequency, 'frequency'),
                drift: a.drift ? num(a.drift) : undefined,
                seed: a.seed ? num(a.seed) : undefined,
                billow: a.billow ? num(a.billow) : undefined,
            })
            return record({density: f32(r.density), mid: f32(r.mid), fine: f32(r.fine)})
        },
    },
    warpedPoint: {section: 'noise', doc: 'the domain-warped coordinate itself (two IQ warp levels): feed to noise/shapes for flowing, marbled, plasma-like structure', inputs: {at: V3('a 3D point, usually xyPlusZ(aspectUV·k, time)'), time: F(''), amount: N('0..1 warp strength')}, output: 'vec3', build: (a) => vec3v(warpedPoint({at: num(a.at, 'at'), time: num(a.time, 'time'), amount: num(a.amount, 'amount')}))},
    waves: {
        section: 'noise', doc: 'overlapping traveling sines — the organic wobble engine: Σ weight·Π sin(x·kx + y·ky + t·kt + phase). Signed. `schedule` is data: [{waves:[{x,y,t,phase}], weight}] or an organicWaves node (organicWaves schedules use x, y AND t — pass all three)',
        inputs: {x: F(''), y: F('', true), t: F('', true), schedule: {type: ['schedule', 'data'], doc: 'the wave schedule'}}, output: 'f32',
        build: (a) => {
            const sched = a.schedule?.type === 'schedule' ? a.schedule.value : (a.schedule?.type === 'data' ? a.schedule.value as WaveTerm[] : undefined)
            if (!Array.isArray(sched)) throw new StdGraphError('waves: schedule must be a list of {waves, weight} terms')
            return f32(waves({x: num(a.x, 'x'), y: a.y ? num(a.y) : undefined, t: a.t ? num(a.t) : undefined}, sched))
        },
    },
    organicWaves: {
        section: 'noise', doc: 'generate a `waves` schedule from intent: base frequency, detail layers (each ~1.8× finer, half weight), drift speed, seed. Deterministic',
        inputs: {frequency: C(''), detail: C('1..5', true), drift: C('', true), seed: C('', true)}, output: 'schedule',
        build: (a) => ({type: 'schedule', value: organicWaves({frequency: konst(a.frequency, 'frequency'), detail: konstOr(a.detail, 3), drift: konstOr(a.drift, 0.5), seed: konstOr(a.seed, 1)})}),
    },
    wavyLine: {
        section: 'noise', doc: 'the offset of an animated organic line at coordinate `along`: a stack of traveling sines. `waves` is data: [{freq, speed, amount}]. Water horizons, curtain paths, cloth edges',
        inputs: {along: F(''), time: F(''), waves: {type: 'data', doc: '[{freq, speed, amount}, …]'}}, output: 'f32',
        build: (a) => {
            const list = a.waves?.type === 'data' ? a.waves.value : undefined
            if (!Array.isArray(list)) throw new StdGraphError('wavyLine: waves must be a list of {freq, speed, amount}')
            return f32(wavyLine({along: num(a.along, 'along'), time: num(a.time, 'time'), waves: list as {freq: number; speed: number; amount: number}[]}))
        },
    },
    scatterField: {
        section: 'noise', doc: 'smooth field blending `count` drifting anchor values across the canvas (the mesh-gradient primitive); softness = crispness of cells',
        inputs: {at: V2('aspectUV'), count: N('1..8'), seed: N(''), drift: N('0..1'), time: F(''), softness: N('~2')}, output: 'f32',
        build: (a, ctx) => f32(scatterField({at: num(a.at, 'at'), count: num(a.count, 'count'), seed: num(a.seed, 'seed'), drift: num(a.drift, 'drift'), aspect: ctx.aspect, time: num(a.time, 'time'), softness: num(a.softness, 'softness')})),
    },
    pointStars: {
        section: 'noise', doc: 'a plane of sparse point glints (stars, glitter, dust): `glint` intensity and the slow `variation` field for tinting',
        inputs: {p: V2('scaled coordinate; cells ≈ 1 unit'), density: C('0..1 keep fraction, default 0.4', true), size: C('point radius, default 0.02', true), gain: N('', true), twinkleRate: C('', true), time: F('', true)},
        output: {record: {glint: 'f32', variation: 'f32'}},
        build: (a) => {
            const keep = Math.max(0.01, Math.min(1, konstOr(a.density, 0.4)))
            const size = konstOr(a.size, 0.02)
            const time = a.time ? num(a.time) : undefined
            const rate = konstOr(a.twinkleRate, 2)
            const r = pointStars(num(a.p, 'p'), {
                variationFreq: 0.13, radius: [size * size, size * size * 2], keep: [1 - keep, 1 - keep + 0.15],
                brightness: [0.6, 1.2], gain: numOr(a.gain, 1),
                shimmer: time ? (v) => twinkle(time, v, {amount: floatE(0.5), rate, spread: 40}) : undefined,
            })
            return record({glint: f32(r.glint), variation: f32(r.variation)})
        },
    },

    // ── Fields (composition-time pipeline: build a Field, then `sample`/`ramp`/`normalFrom` it) ──
    linearField: {section: 'fields', doc: 'projection 0..1 along from→to (position props), rotated by angle (degrees)', inputs: {from: PPOS(''), to: PPOS(''), angle: N('degrees', true)}, output: 'field', build: (a) => fieldV(dist.linear({from: propRef(a.from, 'from'), to: propRef(a.to, 'to'), angle: asSpec(a.angle, 0)}))},
    radialField: {section: 'fields', doc: 'distance from a centre prop normalized by radius (0 centre → 1 at radius); aspect stretches, skew rotates', inputs: {center: PPOS(''), radius: N(''), aspect: N('1 = round', true), skew: N('degrees', true)}, output: 'field', build: (a) => fieldV(dist.radial({center: propRef(a.center, 'center'), radius: asSpec(a.radius), aspect: asSpec(a.aspect, 1), skew: asSpec(a.skew, 0)}))},
    conicField: {section: 'fields', doc: 'angular sweep 0..1 around a centre prop, shifted by rotation (degrees)', inputs: {center: PPOS(''), rotation: N('', true)}, output: 'field', build: (a) => fieldV(dist.conic({center: propRef(a.center, 'center'), rotation: asSpec(a.rotation, 0)}))},
    diamondField: {section: 'fields', doc: 'diamond→square distance from a centre prop (roundness 0 = diamond, 1 = square)', inputs: {center: PPOS(''), size: N(''), rotation: N('', true), roundness: N('', true)}, output: 'field', build: (a) => fieldV(dist.diamond({center: propRef(a.center, 'center'), size: asSpec(a.size), rotation: asSpec(a.rotation, 0), roundness: asSpec(a.roundness, 0)}))},
    noiseField: {section: 'fields', doc: 'a 3D noise basis as a field (0..1; mx3signed is −1..1) — lift it onto the screen with `volume`', inputs: {basis: ENUM(['mx3', 'perlin', 'value', 'mx3signed'], 'noise basis')}, output: 'field', build: (a) => fieldV(noiseField(enumOf(a.basis) as 'mx3'))},
    volume: {section: 'fields', doc: 'lift a 3D-coordinate field onto the screen (aspect-corrected slab), zoomed exponentially by scale', inputs: {field: FIELD(''), scale: N('')}, output: 'field', build: (a) => fieldV(scaledVolume(fieldOf(a.field), {scale: asSpec(a.scale)}))},
    warpedField: {section: 'fields', doc: 'domain-warp a field over time — smooth noise becomes flowing plasma', inputs: {field: FIELD(''), amount: N(''), timeScale: C('', true)}, output: 'field', build: (a) => fieldV(warped(fieldOf(a.field), {amount: asSpec(a.amount), timeScale: a.timeScale ? konst(a.timeScale) : undefined}))},
    evolving: {section: 'fields', doc: 'walk a 2D field\'s third axis with time so it morphs in place (rate scales the evolution)', inputs: {field: FIELD(''), rate: C('')}, output: 'field', build: (a) => fieldV(evolving(fieldOf(a.field), {rate: konst(a.rate, 'rate')}))},
    seededPlane: {section: 'fields', doc: 'exponential-scale noise framing: aspect-correct, scale by exp(scale prop), offset by seed prop', inputs: {field: FIELD(''), scale: PFLOAT(''), seed: PFLOAT('')}, output: 'field', build: (a) => fieldV(seededPlane(fieldOf(a.field), {scale: propRef(a.scale, 'scale'), seed: propRef(a.seed, 'seed')}))},
    scaledPlane: {section: 'fields', doc: 'linear-scale framing: aspect-correct and multiply by a scale prop', inputs: {field: FIELD(''), scale: PFLOAT('')}, output: 'field', build: (a) => fieldV(scaledPlane(fieldOf(a.field), {scale: propRef(a.scale, 'scale')}))},
    rings: {section: 'fields', doc: 'concentric repetition of a field (count rings)', inputs: {field: FIELD(''), count: N('')}, output: 'field', build: (a) => fieldV(rings(fieldOf(a.field), asSpec(a.count)))},
    tiles: {section: 'fields', doc: 'wrapped repetition of a field (count cycles)', inputs: {field: FIELD(''), count: N('')}, output: 'field', build: (a) => fieldV(tiles(fieldOf(a.field), asSpec(a.count)))},
    tone: {section: 'fields', doc: 'tone-shape a unit field: glow spreads, contrast about mid-grey, balance 0..100 shifts', inputs: {field: FIELD(''), glow: N(''), contrast: N(''), balance: N('')}, output: 'field', build: (a) => fieldV(tone(fieldOf(a.field), {glow: asSpec(a.glow), contrast: asSpec(a.contrast), balance: asSpec(a.balance), invert: true}))},
    thresholdField: {section: 'fields', doc: 'soft threshold of a signed field: to 0..1 then smoothstep low→high', inputs: {field: FIELD(''), low: C(''), high: C(''), bias: N('', true)}, output: 'field', build: (a) => fieldV(threshold(fieldOf(a.field), {low: konst(a.low, 'low'), high: konst(a.high, 'high'), bias: a.bias ? asSpec(a.bias) as FieldScalar : undefined}))},
    flowField: {section: 'fields', doc: 'the layered trig flow pattern (nested sin/cos domain warps), signed, animated', inputs: {detail: N('')}, output: 'field', build: (a) => fieldV(flowField({detail: asSpec(a.detail)}))},
    pulseField: {section: 'fields', doc: 'brightness pulse from a field: sin(t·speed + field·span)·depth + 1', inputs: {field: FIELD(''), speed: C(''), span: C(''), depth: C('')}, output: 'field', build: (a) => fieldV(pulse(fieldOf(a.field), {speed: konst(a.speed), span: konst(a.span), depth: konst(a.depth)}))},
    unitized: {section: 'fields', doc: 'signed −1..1 field → 0..1', inputs: {field: FIELD('')}, output: 'field', build: (a) => fieldV(unitized(fieldOf(a.field)))},
    noiseTone: {section: 'fields', doc: 'the standard noise tone tail: contrast + balance props, inverted so colorA reads low', inputs: {field: FIELD(''), contrast: PFLOAT('', true), balance: PFLOAT('', true)}, output: 'field', build: (a) => fieldV(noiseTone(fieldOf(a.field), {contrast: a.contrast ? propRef(a.contrast, 'contrast') : undefined, balance: a.balance ? propRef(a.balance, 'balance') : undefined}))},
    gaborGrains: {section: 'fields', doc: 'oriented sine grains (frequency prop), phase-animated, signed — over a scaledPlane/seededPlane coordinate', inputs: {frequency: PFLOAT('')}, output: 'field', build: (a) => fieldV(gaborGrains({frequency: propRef(a.frequency, 'frequency')}))},
    waveletBands: {section: 'fields', doc: 'rotating banded wavelets (detail prop = octave ratio), animated, signed', inputs: {detail: PFLOAT('')}, output: 'field', build: (a) => fieldV(waveletBands({detail: propRef(a.detail, 'detail')}))},
    erosionRidges: {section: 'fields', doc: 'branching hydraulic-erosion ridge height, static, signed (sample over a plane coordinate)', inputs: {}, output: 'field', build: () => fieldV(erosionRidges())},
    curlSpeed: {section: 'fields', doc: 'swirling curl-flow speed 0..1, morphing at rate × clock', inputs: {rate: C('')}, output: 'field', build: (a) => fieldV(curlSpeed({rate: konst(a.rate, 'rate')}))},
    cellDistances: {section: 'fields', doc: 'nearest-two voronoi distances vec2(d1, d2) over the aspect plane at a scale prop, drifting on the clock', inputs: {scale: PFLOAT(''), seed: PFLOAT('')}, output: 'field', build: (a) => fieldV(cellDistances({scale: propRef(a.scale, 'scale'), seed: propRef(a.seed, 'seed')}))},
    cellFill: {section: 'fields', doc: 'F1/F2 fill gradient of cellDistances; edgeIntensity prop = how far the edge colour reaches in', inputs: {cells: FIELD(''), edgeIntensity: PFLOAT('')}, output: 'field', build: (a) => fieldV(cellFill(fieldOf(a.cells, 'cells'), {edgeIntensity: propRef(a.edgeIntensity, 'edgeIntensity')}))},
    cellBorders: {section: 'fields', doc: 'cell boundary mask of cellDistances (0 on borders, 1 inside)', inputs: {cells: FIELD(''), softness: PFLOAT(''), scale: PFLOAT('')}, output: 'field', build: (a) => fieldV(cellBorders(fieldOf(a.cells, 'cells'), {softness: propRef(a.softness, 'softness'), scale: propRef(a.scale, 'scale')}))},
    sample: {
        section: 'fields', doc: 'evaluate a field at a coordinate → its value (f32 for library fields; whatever a fieldOf body yields — a vec3 colour function sampled twice is how a sky gets reflected). Library fields want `uv`; a fieldOf field wants whatever its body treats `p` as',
        inputs: {field: FIELD(''), at: {type: ['vec2', 'vec3'], doc: 'the coordinate'}},
        output: (i) => (i.field?.valueType ?? 'f32'),
        build: (a, ctx) => ({type: (a.field?.type === 'field' ? a.field.valueType : undefined) ?? 'f32', expr: fieldOf(a.field)(ctx.params, num(a.at, 'at'))}),
    },
    slope: {
        section: 'fields', doc: 'finite-difference slope of a field at a 2D point: `value`, `dx`, `dy` — true derivatives (change per unit of `at`, already divided by eps), so a field with fine detail gives large values: scale them down before building a normal. The relief-to-normal bridge (use with a `fieldOf` height field)',
        inputs: {field: FIELD(''), at: V2(''), eps: C('step, ~0.01')}, output: {record: {value: 'f32', dx: 'f32', dy: 'f32'}},
        build: (a, ctx) => {
            const field = fieldOf(a.field)
            const r = fdSlope((q) => field(ctx.params, q), num(a.at, 'at'), konst(a.eps, 'eps'))
            return record({value: f32(r.value), dx: f32(r.dx), dy: f32(r.dy)})
        },
    },
    normalFrom: {
        section: 'fields', doc: 'surface normal (vec3, +z toward the viewer) of a height field at a 2D point; strength scales the relief. Light it with `shine`/`lambert`',
        inputs: {field: FIELD('a fieldOf height field'), at: V2(''), eps: C('~0.01'), strength: N('')}, output: 'vec3',
        build: (a, ctx) => {
            const field = fieldOf(a.field)
            const r = fdSlope((q) => field(ctx.params, q), num(a.at, 'at'), konst(a.eps, 'eps'), 'hn')
            const s = num(a.strength, 'strength')
            return vec3v(m.normalize(m.vec3(m.neg(m.mul(r.dx, s)), m.neg(m.mul(r.dy, s)), 1)))
        },
    },

    // ── Colour & palettes ──
    pair: {section: 'color', doc: 'two colour props mixed by t in the component\'s colour space (perceptual when the user picks OKLCh etc.)', inputs: {a: PCOLOR(''), b: PCOLOR(''), t: F('0..1')}, output: 'vec4', build: (a, ctx) => vec4v(pair(propRef(a.a, 'a'), propRef(a.b, 'b'), {kind: 'prop', name: 'colorSpace'})(num(a.t, 't'), ctx.params))},
    palette: {section: 'color', doc: 'a reusable two-colour palette object for `ramp`', inputs: {a: PCOLOR(''), b: PCOLOR('')}, output: 'palette', build: (a) => paletteV(pair(propRef(a.a, 'a'), propRef(a.b, 'b'), {kind: 'prop', name: 'colorSpace'}))},
    ramp: {section: 'color', doc: 'sample a field at uv, clamp to 0..1, look it up in a palette → vec4', inputs: {field: FIELD(''), palette: {type: 'palette', doc: ''}}, output: 'vec4', build: (a, ctx) => vec4v(paletteOf(a.palette)(m.clamp(fieldOf(a.field)(ctx.params, ctx.uv), 0, 1), ctx.params))},
    ladder3: {section: 'color', doc: 'three colour props: a→b by t1, then →c by t2, in the colour space', inputs: {a: PCOLOR(''), b: PCOLOR(''), c: PCOLOR(''), t1: F(''), t2: F('')}, output: 'vec4', build: (a, ctx) => vec4v(colorLadder3({a: propRef(a.a, 'a'), b: propRef(a.b, 'b'), c: propRef(a.c, 'c'), space: {kind: 'prop', name: 'colorSpace'}})(num(a.t1, 't1'), num(a.t2, 't2'), ctx.params))},
    mixColors: {
        section: 'color', doc: 'mix two vec4 colours by t in the component\'s colour space',
        inputs: {a: V4(''), b: V4(''), t: F('')}, output: 'vec4',
        build: (a, ctx) => {
            const variant = colorMixing.mixColorsVariants[ctx.colorSpaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
            return vec4v(call(variant, 'mixColors', [num(a.a, 'a'), num(a.b, 'b'), num(a.t, 't')]))
        },
    },
    hueWheel: {section: 'color', doc: 'cyclic hue 0..1 (wraps) → rgb rainbow', inputs: {h: F('')}, output: 'vec3', build: (a) => vec3v(hueWheel(num(a.h, 'h')))},
    cosineRainbow: {section: 'color', doc: 'IQ cosine palette: smooth rgb ramp over t with per-channel phases [r, g, b] (data)', inputs: {t: F(''), phases: {type: 'data', doc: '[0, 0.33, 0.67] etc.', optional: true}}, output: 'vec3', build: (a) => { const ph = (a.phases?.type === 'data' && Array.isArray(a.phases.value) ? a.phases.value : [0, 0.33, 0.67]) as [number, number, number]; return vec3v(cosineRainbow(num(a.t, 't'), ph)) }},
    wrapRamp: {section: 'color', doc: 'traverse a 0..1 ramp through `cycles` half-cycles as t rises, folding at the ends (seam-free; sub-pixel bands fade)', inputs: {t: F(''), cycles: N('')}, output: 'f32', build: (a) => f32(wrapRamp({t: num(a.t, 't'), cycles: num(a.cycles, 'cycles')}))},
    foldRamp: {section: 'color', doc: 'fold any real t back into 0..1 as a triangle wave (0→1→0 every 2 units)', inputs: {t: F('')}, output: 'f32', build: (a) => f32(foldRamp(num(a.t, 't')))},
    exposureTone: {section: 'color', doc: 'photographic exposure 1 − e^(−rgb): light piles up toward white instead of clipping (the HDR→display step for emissive looks)', inputs: {rgb: V3('')}, output: 'vec3', build: (a) => vec3v(exposureTone(num(a.rgb, 'rgb')))},
    luminance: {section: 'color', doc: 'perceived brightness of an rgb', inputs: {rgb: V3('')}, output: 'f32', build: (a) => f32(m.dot(num(a.rgb, 'rgb'), m.vec3(0.2126, 0.7152, 0.0722)))},

    // ── Light (2.5D, ONE convention: normals AND light vectors face +z, toward the viewer) ──
    keyLight: {
        section: 'light', doc: 'a key light from an in-plane direction (degrees; 0 = from the right, 90 = from the bottom, 270 = from the top) and `elevation` (how far toward the viewer it sits: 0.15 a low raking sun … 1 straight overhead): `L` unit light vector (its xy points TOWARD the light, in uv with y down: angle 270 gives L.y < 0), `H` half vector toward the viewer',
        inputs: {angle: F('degrees'), elevation: N('0..1, default 0.6', true)}, output: {record: {L: 'vec3', H: 'vec3'}},
        build: (a) => {
            const dir = direction(num(a.angle, 'angle'))
            // Tolerant of the materials-layer habit of negative elevations: the sign is the convention, not the intent.
            const L = m.local(m.normalize(m.vec3(dir.member('x'), dir.member('y'), m.max(m.abs(numOr(a.elevation, 0.6)), 0.05))), 'keyL')
            const H = m.local(m.normalize(m.add(L, m.vec3(0, 0, 1))), 'keyH')
            return record({L: vec3v(L), H: vec3v(H)})
        },
    },
    pointLight: {
        section: 'light', doc: 'a light at a point above the frame: `position` (uv, y down) and `height` above the surface (uv units, default 0.25; low = raking). For the surface point `at` (usually uv): `L` unit light vector, `H` half vector toward the viewer, `distance` to the light (aspect-correct) — attenuate with `inverseSquare` or `emitterFalloff`',
        inputs: {at: V2('the surface point, usually uv'), position: V2('the light, e.g. "@lightPos"'), height: N('default 0.25', true)}, output: {record: {L: 'vec3', H: 'vec3', distance: 'f32'}},
        build: (a, ctx) => {
            const offset = m.sub(num(a.position, 'position'), num(a.at, 'at'))
            const toLight = m.local(m.vec3(m.mul(offset.member('x'), ctx.aspect), offset.member('y'), m.max(numOr(a.height, 0.25), 0.001)), 'ptL')
            const L = m.local(m.normalize(toLight), 'ptLn')
            const H = m.local(m.normalize(m.add(L, m.vec3(0, 0, 1))), 'ptH')
            return record({L: vec3v(L), H: vec3v(H), distance: f32(m.length(toLight))})
        },
    },
    lambert: {section: 'light', doc: 'diffuse response of a normal to a light vector (wrap 0 = hemisphere, 1 = light reaches all round)', inputs: {normal: V3(''), light: V3(''), wrap: N('', true)}, output: 'f32', build: (a) => f32(lambert(num(a.normal, 'normal'), num(a.light, 'light'), {wrap: a.wrap ? num(a.wrap) : undefined}))},
    shine: {section: 'light', doc: 'specular lobe of a light against a normal (viewer along +z); gloss = tightness (8 soft … 200 sharp). Raw lobe — scale by intensity', inputs: {normal: V3(''), light: V3(''), gloss: N('')}, output: 'f32', build: (a) => f32(shine({normal: num(a.normal, 'normal'), light: num(a.light, 'light'), gloss: num(a.gloss, 'gloss')}))},
    domeNormal: {section: 'light', doc: 'a fake bulging surface: normal field of a disc treated as a dome (puffy shapes). Light it with shine/lambert', inputs: {delta: V2('centered delta'), dist: F(''), radius: N(''), tilt: C('', true), drop: C('', true)}, output: 'vec3', build: (a) => vec3v(domeNormal({delta: num(a.delta, 'delta'), dist: num(a.dist, 'dist'), radius: num(a.radius, 'radius'), bulge: {tilt: a.tilt ? konst(a.tilt) : undefined, drop: a.drop ? konst(a.drop) : undefined}}))},
    grazing: {section: 'light', doc: 'grazing-view factor of a +z normal (0 facing the viewer → 1 edge-on) — the fresnel input', inputs: {normal: V3('')}, output: 'f32', build: (a) => f32(m.sub(1, m.clamp(num(a.normal, 'normal').member('z'), 0, 1)))},
    fresnel: {section: 'light', doc: 'Schlick fresnel reflectance r0 + grazing^power·gain (glass: r0 0.04, gain 0.96)', inputs: {grazing: F(''), r0: C('', true), gain: N(''), power: C('', true)}, output: 'f32', build: (a) => f32(schlickFresnel(num(a.grazing, 'grazing'), {r0: a.r0 ? konst(a.r0) : undefined, gain: num(a.gain, 'gain'), power: a.power ? konst(a.power) : undefined}))},
    glowSpot: {section: 'light', doc: 'gaussian glow at offset (du, dv) from a point; sharpness = falloff', inputs: {du: F(''), dv: F(''), sharpness: F('')}, output: 'f32', build: (a) => f32(glowSpot(num(a.du, 'du'), num(a.dv, 'dv'), num(a.sharpness, 'sharpness')))},
    emitterFalloff: {section: 'light', doc: 'real light falloff with distance d from an emitter: (radius/(radius+d))^power — 1 at the source, long tail', inputs: {d: F(''), radius: N(''), power: N('1 line emitter … 2 point emitter')}, output: 'f32', build: (a) => f32(emitterFalloff(num(a.d, 'd'), {radius: num(a.radius, 'radius'), power: num(a.power, 'power')}))},
    inverseSquare: {section: 'light', doc: 'point-light attenuation: 1 at the source, half at range', inputs: {distance: F(''), range: N('')}, output: 'f32', build: (a) => f32(inverseSquare(num(a.distance, 'distance'), num(a.range, 'range')))},
    rayBands: {section: 'light', doc: 'banded shimmer along an axis: traveling sine lifted to 0..1 and sharpened (godrays, curtains, fabric; noise-warp `along` for organic rays)', inputs: {along: F(''), frequency: N(''), time: F(''), speed: N(''), sharpness: N(''), phase: N('', true)}, output: 'f32', build: (a) => f32(rayBands({along: num(a.along, 'along'), frequency: num(a.frequency, 'frequency'), time: num(a.time, 'time'), speed: num(a.speed, 'speed'), sharpness: num(a.sharpness, 'sharpness'), phase: a.phase ? num(a.phase) : undefined}))},
    sensorGrain: {section: 'light', doc: 'multiplicative film grain on an rgb, re-rolled by the clock (the photon-noise floor of photographic looks)', inputs: {rgb: V3(''), amount: N('0..0.2'), time: F('', true)}, output: 'vec3', build: (a, ctx) => vec3v(sensorGrain(num(a.rgb, 'rgb'), {amount: num(a.amount, 'amount'), pixel: m.mul(ctx.uv, ctx.viewport), clock: a.time ? num(a.time) : 0}))},

    // ── Compose (stacking light and paint) ──
    layers: {
        section: 'compose', doc: 'composite a stack of rgb light contributions, first at the back: additive by default, `blend: "screen"` soft-saturates; `opacity` scales; `behind` is a transmittance the layer is seen through. `items` is data: [{paint: "$node", opacity?, blend?, behind?}] — paints may be vec3 rgb, f32 (grey) or vec4 (rgb used). Returns vec3 rgb; finish with emissiveAlpha or rgba',
        inputs: {items: {type: 'data', doc: '[{paint: "$id", opacity?: number|"$id", blend?: "add"|"screen", behind?: "$id"}]'}}, output: 'vec3',
        build: (a, ctx) => {
            const items = a.items?.type === 'data' ? a.items.value : undefined
            if (!Array.isArray(items) || !items.length) throw new StdGraphError('layers: items must be a non-empty list')
            const asRgb = (v: Val): Expr => {
                const t = typeOf(v)
                const e = num(v)
                return t === 'f32' ? m.splat3(e) : t === 'vec4' ? e.member('rgb') : t === 'vec2' ? m.vec3(e.member('x'), e.member('y'), 0) : e
            }
            const entries: Layer[] = items.map((raw, i) => {
                const item = raw as {paint?: string; opacity?: number | string; blend?: 'add' | 'screen'; behind?: string}
                if (typeof item?.paint !== 'string') throw new StdGraphError(`layers: item ${i} needs a "paint" reference`)
                const entry: Exclude<Layer, Expr> = {paint: asRgb(ctx.resolveRef(item.paint))}
                if (item.opacity !== undefined) entry.opacity = typeof item.opacity === 'number' ? item.opacity : num(ctx.resolveRef(item.opacity))
                if (item.blend) entry.blend = item.blend
                if (item.behind) entry.behind = num(ctx.resolveRef(item.behind))
                return entry
            })
            return vec3v(layersWord(entries))
        },
    },
    emissiveAlpha: {section: 'compose', doc: 'light → straight-alpha rgba: alpha = the peak channel, colour = light ÷ alpha. THE output step for glows/flares drawn over transparency', inputs: {rgb: V3('')}, output: 'vec4', build: (a) => vec4v(emissiveAlpha(num(a.rgb, 'rgb')))},
    dithered: {section: 'compose', doc: 'the always-on ±0.002 output dither on a final rgba (hides banding on long soft ramps). Apply to the COLOUR last', inputs: {color: V4('')}, output: 'vec4', build: (a, ctx) => vec4v(dithered(num(a.color, 'color'), ctx.params))},
    over: {
        section: 'compose', doc: 'alpha-composite `top` over `bottom` (both straight-alpha vec4)',
        inputs: {top: V4(''), bottom: V4('')}, output: 'vec4',
        build: (a) => {
            const t = m.local(num(a.top, 'top'), 'ovT'), b = m.local(num(a.bottom, 'bottom'), 'ovB')
            const ta = t.member('a'), ba = b.member('a')
            const outA = m.local(m.add(ta, m.mul(ba, m.sub(1, ta))), 'ovA')
            const rgb = m.div(m.add(m.mul(t.member('rgb'), ta), m.mul(m.mul(b.member('rgb'), ba), m.sub(1, ta))), m.max(outA, 1e-5))
            return vec4v(m.vec4(rgb, outA))
        },
    },

    // ── Figures (complete drawn paints) ──
    strokedSegment: {section: 'figures', doc: 'a stroked line between two position props: width (canvas-height units), style 0 solid/1 dashed/2 dotted, caps 0 square/1 round → vec4 in the colour prop', inputs: {from: PPOS(''), to: PPOS(''), color: PCOLOR(''), width: N(''), style: N('', true), dashLength: N('', true), gapLength: N('', true), capStart: N('', true), capEnd: N('', true)}, output: 'vec4', build: (a, ctx) => vec4v(strokedSegment({from: propRef(a.from, 'from'), to: propRef(a.to, 'to'), color: propRef(a.color, 'color'), width: asSpec(a.width), style: asSpec(a.style, 0), dashLength: asSpec(a.dashLength, 0.05), gapLength: asSpec(a.gapLength, 0.03), capStart: asSpec(a.capStart, 1), capEnd: asSpec(a.capEnd, 1)})(ctx.params))},
    sineStroke: {section: 'figures', doc: 'a traveling sine-wave stroke centred on a position prop: angle (degrees), frequency (cycles across), amplitude, thickness, softness → vec4 in the colour prop. Animated', inputs: {position: PPOS(''), color: PCOLOR(''), angle: N(''), frequency: N(''), amplitude: N(''), thickness: N(''), softness: N('')}, output: 'vec4', build: (a, ctx) => vec4v(sineStroke({position: propRef(a.position, 'position'), color: propRef(a.color, 'color'), angle: asSpec(a.angle), frequency: asSpec(a.frequency), amplitude: asSpec(a.amplitude), thickness: asSpec(a.thickness), softness: asSpec(a.softness)})(ctx.params))},

    // ── Control (bodies; handled by the interpreter) ──
    repeat: {
        section: 'control', doc: 'unroll `body` `count` times with "i" = 0,1,2… bound; the `result` node of each pass is combined by `reduce` (add | max | screen). "N things": stars, ripples, streaks, layered echoes. Body nodes may read outer nodes',
        inputs: {count: C('≤ 24'), reduce: ENUM(['add', 'max', 'screen'], '', true)}, output: 'f32',
        build: () => { throw new StdGraphError('repeat is interpreted, not built') },
        control: 'repeat',
    },
    fieldOf: {
        section: 'control', doc: 'define a FIELD from a body: inside, "p" (vec2) is the coordinate being sampled; `result` is what the field yields — an f32 (a height field: take its normal with normalFrom/slope) or a vec3/vec4 (a colour function: a sky you can `sample` at the real and the reflected coordinate)',
        inputs: {}, output: 'field',
        build: () => { throw new StdGraphError('fieldOf is interpreted, not built') },
        control: 'fieldOf',
    },
    march: {
        section: 'control', doc: 'VOLUMETRIC ray march — the word for clouds, fog, smoke, nebulae, murky water: `steps` samples front to back along a ray; the body sees "z" (depth = entry + length·t), "t" (0..1) and "i", and yields the step\'s emitted light (`result`, vec3 or f32) and absorption (`absorb`, f32 or vec3 per channel). Returns `color` (accumulated light) and `transmittance` (what survives — multiply whatever lies BEHIND by it). Build a 3D sample point inside the body from the screen position and z, e.g. xyPlusZ(aspectUV·k, z), and read density with turbulence/fbm3. Banding: pass a `jitter` (hash of uv, 0..1/steps) rather than more steps',
        inputs: {steps: C('≤ 32; 8–16 is plenty'), entry: N('depth where the ray enters (often 0)'), length: N('ray length in your units'), jitter: F('per-pixel step offset, e.g. hash(uv·viewport)/steps', true), emissionGain: N('per-step emission scale, typically k/steps', true)},
        output: {record: {color: 'vec3', transmittance: 'vec3'}},
        build: () => { throw new StdGraphError('march is interpreted, not built') },
        control: 'march',
    },
}

// ── Escape hatch (interpreted; gated by the validator's allowWgsl option) ──
CATALOG.wgsl = {
    section: 'escape', doc: 'HAND-ROLL a part no word covers: a pure WGSL function of its `inputs`, returning `type`. `code` is the function BODY (statements; must `return`); WGSL builtins, `let`/`var`, loops and `if` are fine; NO textures, uniforms, globals, `fn`, `struct` or attributes. Declare `name` (the concept, e.g. "groundPlane") and `why` (which catalog words you tried and what they lacked) — hand-rolled parts are logged and reviewed for promotion into the vocabulary, and are UNTESTED for performance, so keep them small and reach for them last',
    inputs: {
        inputs: {type: 'data', doc: '{paramName: "$ref" | "@prop" | number, …} — become the function parameters, in this order'},
        type: ENUM(['f32', 'vec2', 'vec3', 'vec4'], 'the return type'),
        code: {type: 'enum', doc: 'the WGSL function body, e.g. "var acc = 0.0; for (var k = 0; k < 4; k++) { acc += sin(x * f32(k)); } return acc * 0.25;"'},
        name: {type: 'enum', doc: 'concept name, camelCase'},
        why: {type: 'enum', doc: 'one sentence: what you tried first and why it did not fit'},
    },
    output: (i) => ((i.type as {value?: string} | undefined)?.value as GraphValueType | undefined) ?? 'f32',
    build: () => { throw new StdGraphError('wgsl is interpreted, not built') },
    control: 'wgsl',
}

// Re-exports the interpreter needs.
export {colorSpaceModeOf, uniformOf, animatedTime, volumeMarch}
