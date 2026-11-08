/**
 * std/document — shader definitions written as DATA: JSON documents (envelopes) an editor or an
 * agent can author, store and send, lowered at runtime to ordinary engine definitions.
 *
 * Two kinds:
 *
 * - `std-graph` — a generator composed from the std vocabulary: a props manifest plus an SSA
 *   list of nodes, each node one WORD (`std/math`, `std/frames`, `std/shape`, `std/mask`,
 *   `std/motion`, `std/paint/*`) with named inputs. The interpreter (`define.ts`) walks the
 *   list at composition time and calls the real std functions through the catalog
 *   (`catalog.ts`). One escape node (`wgsl`) hand-rolls a pure function no word covers.
 * - `wgsl` — a raw `wgsl` body (a generator, or a filter when it reads the layer inside)
 *   with the same props manifest (`wgslEnvelope.ts`).
 *
 * Node argument syntax (`GraphArg`):
 *   42 / 0.5            number literal (an f32, or a JS constant where the input says `const`)
 *   "$id"               the value of an earlier node; "$id.x" / "$id.rgb" swizzles a vector,
 *                       "$id.dist" reads a field of a record-valued node
 *   "@prop"             a declared prop (float/angle/bool → f32, color → vec4, position → vec2);
 *                       "@color.rgb" swizzles
 *   "uv" "aspect" "viewport" "time" "pointer"   system values ("uv.x" etc. swizzle)
 *   "p"                 inside a `fieldOf` body: the coordinate being sampled (vec2)
 *   "i"                 inside a `repeat` body: the iteration index (0, 1, 2 …)
 *   "add" / "screen"    a string where the input is an `enum`
 *   {…} / […]           plain data where the input is `data` (wave schedules, layer lists)
 */

export type GraphArg = number | string | boolean | null | GraphArg[] | {[key: string]: GraphArg}

export interface GraphNode {
    /** Unique within its scope; letters/digits/underscore. Referenced as `"$id"`. */
    id: string
    /** A catalog op name. */
    op: string
    args?: Record<string, GraphArg>
    /** `repeat` / `fieldOf` / `march` only: the nodes evaluated per iteration / per sample / per step. */
    body?: GraphNode[]
    /** `repeat` / `fieldOf` / `march` only: the body node whose value is the result (`"$id"`).
     *  For `march` this is the step's EMITTED light (vec3, or f32 grey). */
    result?: string
    /** `march` only: the body node giving the step's absorption (f32, or vec3 per channel). */
    absorb?: string
}

export type GraphPropKind = 'float' | 'angle' | 'color' | 'position' | 'bool'

export interface GraphPropSpec {
    kind: GraphPropKind
    /** float/angle: a number · color: "#rrggbb" · position: {x, y} in 0..1 (y = 0 top) · bool. */
    default: unknown
    min?: number
    max?: number
    step?: number
    label?: string
    group?: string
    description?: string
}

export interface StdGraphEnvelope {
    kind: 'std-graph'
    name: string
    description?: string
    props?: Record<string, GraphPropSpec>
    nodes: GraphNode[]
    /** The node whose value is the pixel colour — a vec4 (straight alpha) or a vec3 (alpha 1). */
    output: string
}

/**
 * The prop kinds a `wgsl` envelope declares: the library's own controls, by the same names.
 * What a body receives for each is in `../wgslResources.ts` and `wgslEnvelope.ts`.
 */
export type WgslPropKind =
    | 'range' | 'number' | 'checkbox' | 'color' | 'position' | 'select'
    | 'gradient-stops' | 'list' | 'image-upload' | 'video-upload' | 'layer' | 'shape'

export interface WgslPropSpec {
    kind: WgslPropKind
    /**
     * range/number: a number · checkbox: true/false · color: "#rrggbb" · position: {x, y} ·
     * select: one of the option values · gradient-stops: [{color, position}, …] (2–8) ·
     * list: the items · image-upload/video-upload: a URL ("" for none) · layer: "" · shape: a
     * shape config such as {type: "starSDF", sides: 5}.
     */
    default?: unknown
    min?: number
    max?: number
    step?: number
    label?: string
    group?: string
    description?: string
    /** select: the choices, as values or {value, label}. */
    options?: Array<string | {value: string; label?: string}>
    /** list: each item's fields, as the library's list props declare them. */
    item?: Record<string, {kind: 'position' | 'color' | 'number' | 'boolean'; default: unknown; label?: string; min?: number; max?: number; step?: number}>
    /** list: the most items (up to 16). */
    maxItems?: number
    minItems?: number
    itemLabel?: string
    /** image-upload / video-upload: how the picture fits the layer (default cover). */
    fit?: 'cover' | 'contain' | 'fill'
}

export interface WgslEnvelope {
    kind: 'wgsl'
    name: string
    description?: string
    props?: Record<string, WgslPropSpec>
    /**
     * The body of the per-pixel function: WGSL statements that `return` a vec4f colour
     * (straight alpha). Props are bound by name; so are `uv`, `time`, `aspect`, `viewport`,
     * `pointer`, and — in a filter — `child` (the layer inside at this pixel) or
     * `childTexture` + `childSampler` (sample it anywhere; premultiplied).
     */
    body: string
}

/** Any shader written as data. */
export type ShaderEnvelope = StdGraphEnvelope | WgslEnvelope

export const STD_GRAPH_LIMITS = {
    maxNodes: 96,
    maxProps: 12,
    maxRepeat: 24,
    maxMarchSteps: 32,
    maxBody: 48,
} as const

/** The value kinds a node can hold. Numeric kinds are GPU expressions; the rest are composition-time objects. */
export type GraphValueType = 'f32' | 'vec2' | 'vec3' | 'vec4' | 'field' | 'palette' | 'schedule' | 'record'

export class StdGraphError extends Error {
    constructor(message: string, readonly nodeId?: string) {
        super(nodeId ? `node "${nodeId}": ${message}` : message)
        this.name = 'StdGraphError'
    }
}
