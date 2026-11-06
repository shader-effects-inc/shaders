/**
 * The set of blend modes used when compositing a layer over its siblings. Mirrors the keys of
 * the kit's `blendModes` map (`gpu/kit/blend.ts`); kept as an explicit union here so the core
 * type surface carries no dependency on the GPU kit.
 */
export type BlendMode =
    | 'normal'
    | 'normal-oklch'
    | 'normal-oklab'
    | 'multiply'
    | 'screen'
    | 'linearDodge'
    | 'overlay'
    | 'difference'
    | 'colorDodge'
    | 'exclusion'
    | 'color'
    | 'luminosity'
    | 'darken'
    | 'lighten'
    | 'colorBurn'
    | 'linearBurn'
    | 'softLight'
    | 'hardLight'
    | 'hue'
    | 'saturation'

/**
 * Layer transform configuration (offset / rotation / scale / anchor + edge handling).
 * Carried on NodeMetadata.transform; the design editor reads/writes it verbatim.
 */
export interface TransformConfig {
    offsetX: number
    offsetY: number
    rotation: number  // in degrees
    scale: number
    anchorX: number   // 0-1 normalized
    anchorY: number   // 0-1 normalized
    edges: 'stretch' | 'transparent' | 'mirror' | 'wrap'
}

export type ToneMappingMode = 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal'

export interface UniformDefinition {
    uniform: any  // uniform node type
    transform?: (value: any) => any
    compileTime?: boolean
    compileTimeWhen?: (previousValue: any, newValue: any) => boolean
    _lastCompiledValue?: any
    /**
     * Present on shape position/size props that carry a unit (px/uv) and/or are origin-relative.
     * Stores the raw (unresolved) prop value so the renderer can re-resolve px → UV against live
     * canvas dimensions on resize and when the `origin` prop changes. See dimensionalProps.ts.
     */
    _rawDimensional?: any
    /**
     * Present on a color-stops (array) prop. The flat, fixed-MAX-size `uniformArray`s holding
     * the active stops — `colorsArray` packs rgba floats (4 per stop), `positionsArray` one
     * float per stop. `stopCount` is the JS unroll bound read by `fragmentNode` at build time
     * (>1 → multi-stop path, ≤1 → legacy two-color path). See utilities/colorStops.ts.
     */
    colorsArray?: any
    positionsArray?: any
    stopCount?: number
    /**
     * CPU-preconverted working-space stop colors (3 floats per stop) and the colorSpaceMode
     * they were converted for. Refilled by writeStops on stop edits (using the cached mode)
     * and refreshed by mixColorStops at fragmentNode build time — a colorSpace change
     * rebuilds the fragment WITHOUT recreating the uniforms map, so the build-time refresh
     * is what keeps this array from silently holding the previous mode's values.
     */
    convertedColorsArray?: any
    convertedColorSpaceMode?: number
    /**
     * Present when a driver resolves to a GPU-side expression (e.g. mouse driver).
     * Returns the current CPU-side scalar value so compute shaders can read it without
     * needing to evaluate the GPU expression. Updated every frame via driver state.
     */
    getCpuValue?: () => number
    /**
     * Present when a prop-map driver is active. Gives compute shaders access to the
     * source texture and remapping uniforms so they can sample per-pixel values via
     * textureLoad rather than relying on the fragment-only screenUV.
     */
    mapComputeInfo?: {
        sourceTexture: any   // texture node (RTT from convertToTexture of the source layer)
        channel: MapChannel  // which component to extract
        inputMin: any        // uniform node
        inputMax: any        // uniform node
        outputMin: any       // uniform node
        outputMax: any       // uniform node
        curve: any           // uniform node
    }
}

/**
 * Defines types of masks that can be applied to components
 */
export type MaskType = 'alpha' | 'alphaInverted' | 'luminance' | 'luminanceInverted'

/**
 * Which channel of a source layer to extract as the driving scalar for a prop map
 */
export type MapChannel = 'alpha' | 'alphaInverted' | 'luminance' | 'luminanceInverted'

/**
 * Configuration for driving a numeric prop from the visual output of another layer.
 * Pass this directly as a prop value instead of a static number to enable prop mapping.
 *
 * @example
 * // Vue
 * <DotGrid :dot-size="{ type: 'map', source: 'my-layer', channel: 'alpha', inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 0.8 }" />
 * // React
 * <DotGrid dotSize={{ type: 'map', source: 'my-layer', channel: 'alpha', inputMin: 0, inputMax: 1, outputMin: 0, outputMax: 0.8 }} />
 */
export interface MapConfig {
    /**
     * Discriminant — always 'map'. Identifies this as a PropDriver of type map.
     */
    type: 'map'

    /**
     * Element ID of the source component whose visual output drives this prop
     */
    source: string

    /**
     * Which channel to extract as the driving scalar
     */
    channel: MapChannel

    /**
     * Source values at/below this → treated as 0
     */
    inputMin: number

    /**
     * Source values at/above this → treated as 1
     */
    inputMax: number

    /**
     * Output value when input=0
     */
    outputMin: number | DimensionalValue

    /**
     * Output value when input=1
     */
    outputMax: number | DimensionalValue

    /**
     * Power curve applied after normalisation. Range -1 to +1, default 0 (linear).
     * Negative values push the response toward outputMin; positive toward outputMax.
     * Converted to a power exponent via: pow(2, -curve * 2)
     */
    curve?: number
}

/**
 * Configuration for driving a position prop from the mouse cursor.
 * Pass this directly as a prop value instead of a static {x, y} to track the cursor.
 *
 * @example
 * // Vue
 * <LensFlare :center="{ type: 'mouse-position', smoothing: 0.1 }" />
 * // React
 * <LensFlare center={{ type: 'mouse-position', smoothing: 0.1 }} />
 */
export interface MousePositionConfig {
    /**
     * Discriminant — always 'mouse-position'.
     */
    type: 'mouse-position'

    /**
     * X-axis behaviour. 'mouse' (default) tracks pointer X. A number pins it to that value (0-1).
     */
    x?: 'mouse' | number

    /**
     * Y-axis behaviour. 'mouse' (default) tracks pointer Y. A number pins it to that value (0-1).
     */
    y?: 'mouse' | number

    /**
     * Invert the X axis — pointer moving right drives the position left.
     */
    invertX?: boolean

    /**
     * Invert the Y axis — pointer moving down drives the position up.
     */
    invertY?: boolean

    /**
     * Lag behind cursor (0-1). 0 = instant tracking, 1 = very slow follow.
     */
    smoothing?: number

    /**
     * Overshoot/bounce amount (0-1). 0 = no bounce, values near 1 = springy.
     */
    momentum?: number

    /**
     * Scales how far the position follows the mouse, relative to the origin point (originX, originY),
     * defaulting to viewport center (0.5, 0.5).
     * - 0: stuck at origin regardless of mouse position
     * - 1: full 1:1 tracking (default)
     * - 2: exaggerated — travels twice as far as the mouse from the origin
     */
    reach?: number

    /**
     * X coordinate of the reach origin — the point displacement scales from.
     * Defaults to 0.5 (viewport center). Same 0–1 range as static position x.
     */
    originX?: number

    /**
     * Y coordinate of the reach origin — the point displacement scales from.
     * Defaults to 0.5 (viewport center). Same 0–1 range as static position y.
     */
    originY?: number
}

/**
 * Configuration for driving a numeric prop from mouse position along one axis.
 * Pass this directly as a prop value instead of a static number to map pointer movement.
 *
 * @example
 * // Vue
 * <DotGrid :dot-size="{ type: 'mouse', axis: 'x', outputMin: 0.1, outputMax: 0.8 }" />
 * // React
 * <DotGrid dotSize={{ type: 'mouse', axis: 'x', outputMin: 0.1, outputMax: 0.8 }} />
 */
export interface MouseMapConfig {
    /**
     * Discriminant — always 'mouse'.
     */
    type: 'mouse'

    /**
     * Which pointer axis drives this prop.
     */
    axis: 'x' | 'y'

    /**
     * Output value when axis position is 0 (left / top).
     */
    outputMin: number | DimensionalValue

    /**
     * Output value when axis position is 1 (right / bottom).
     */
    outputMax: number | DimensionalValue

    /**
     * Power curve applied after normalisation. Range -1 to +1, default 0 (linear).
     */
    curve?: number

    /**
     * Lag behind cursor (0-1). 0 = instant tracking, 1 = very slow follow.
     */
    smoothing?: number

    /**
     * Overshoot/bounce amount (0-1). 0 = no bounce, values near 1 = springy.
     */
    momentum?: number
}

/**
 * A discriminated union of all supported prop driver types.
 * Pass a PropDriver as a prop value to drive that prop dynamically instead of a static value.
 */
/**
 * Configuration for automatically animating a numeric prop over time.
 * Pass this directly as a prop value to drive it with continuous motion.
 *
 * @example
 * // Vue — ping-pong dot size
 * <DotGrid :dot-size="{ type: 'auto-animate', mode: 'ping-pong', outputMin: 0.1, outputMax: 0.8 }" />
 * // React — continuous rotation loop
 * <Spiral scale={{ type: 'auto-animate', mode: 'loop', outputMin: 0, outputMax: 360, speed: 0.5 }} />
 */
export interface AutoAnimateConfig {
    /**
     * Discriminant — always 'auto-animate'.
     */
    type: 'auto-animate'

    /**
     * Animation mode.
     * - 'ping-pong': oscillates between outputMin and outputMax
     * - 'loop': advances from outputMin to outputMax then wraps (good for rotation)
     */
    mode: 'ping-pong' | 'loop'

    /**
     * Output value at the start/bottom of the animation range.
     */
    outputMin: number | DimensionalValue

    /**
     * Output value at the end/top of the animation range.
     */
    outputMax: number | DimensionalValue

    /**
     * Animation speed in cycles per second. Default 1.0.
     * Negative values reverse loop direction.
     */
    speed?: number

    /**
     * Easing curve. Only applies to 'ping-pong' mode.
     * - 'sine': smooth cosine ease in/out (default)
     * - 'linear': constant-speed triangle wave
     * - 'quad': quadratic ease in/out
     * - 'expo': exponential ease in/out (very dramatic)
     * - 'bounce': springy bounce at each end
     */
    easing?: 'sine' | 'linear' | 'quad' | 'expo' | 'bounce'

    /**
     * @deprecated Use `easing` instead.
     */
    waveform?: 'sine' | 'linear'
}

export type PropDriver = MapConfig | MousePositionConfig | MouseMapConfig | AutoAnimateConfig

/**
 * Mask source configuration
 */
export interface MaskConfig {
    /**
     * Element ID reference to use as a mask (ID of another component)
     */
    source: string;
    
    /**
     * How the mask should be applied:
     * - 'alpha': Uses the alpha channel of the mask
     * - 'alphaInverted': Uses the inverted alpha channel of the mask
     * - 'luminance': Uses the luminance (brightness) of the mask
     * - 'luminanceInverted': Uses the inverted luminance of the mask
     */
    type: MaskType;
}

/**
 * A single dimensional value with its own unit, used in BoundingBoxConfig. `uv` is a fraction
 * of the canvas on that axis (the legacy unit); every other unit is a CSS length (see
 * utilities/units.ts) — `%` measures the containing box (the canvas until frames exist).
 */
export interface BoundingBoxDimension {
    value: number
    unit: DimensionalUnit
}

/**
 * A bounding-box length as authored: a `BoundingBoxDimension`, a CSS-like string (`'50%'`,
 * `'2rem'`), or a bare number — which, as in CSS and React's `style` prop, means PIXELS.
 */
export type BoundingBoxLength = number | DimensionalString | Partial<BoundingBoxDimension>

/**
 * Reference edge a bounding box / shape position is measured from. The horizontal and
 * vertical anchors are independent — each axis is start / center / end:
 * - horizontal: 'left' = from the left edge, 'right' = from the right edge,
 *   'center' (no left/right in the name) = signed offset from the horizontal center.
 * - vertical: 'top' = from the top edge, 'bottom' = from the bottom edge,
 *   'center' (no top/bottom in the name) = signed offset from the vertical center.
 * 'top-left' (default) measures x/y from the left/top edges (Y-down); 'center' centers
 * both axes (x/y become signed offsets from the viewport center).
 */
export type BoundingBoxOrigin =
    | 'top-left'
    | 'top-center'
    | 'top-right'
    | 'center-left'
    | 'center'
    | 'center-right'
    | 'bottom-left'
    | 'bottom-center'
    | 'bottom-right'

/**
 * Units a dimensional value may carry (see utilities/units.ts):
 * - 'px'                          CSS pixels
 * - '%'                           the containing box (the canvas until frames exist); axis-tied props only
 * - 'cw' / 'ch' / 'cmin' / 'cmax' 1% of the canvas width / height / smaller / larger side
 * - 'vw' / 'vh' / 'vmin' / 'vmax' 1% of the browser viewport
 * - 'rem'                         the root font size
 * - 'uv'                          the legacy unit a plain number already means
 */
export type DimensionalUnit = 'px' | '%' | 'cw' | 'ch' | 'cmin' | 'cmax' | 'vw' | 'vh' | 'vmin' | 'vmax' | 'rem' | 'uv'

/**
 * A CSS-like length string accepted anywhere a DimensionalValue is: `'12px'`, `'50%'`, `'40cw'`,
 * `'100vh'`, `'2rem'`. Normalized to the object form when it's resolved.
 */
export type DimensionalString = `${number}${DimensionalUnit}`

/**
 * A single dimensional value usable on shape position/size props.
 * Position props are `{ x, y }` where each axis may be a plain number (UV, legacy),
 * a DimensionalValue, or a length string. Size props (radius/width/height) may be any of those.
 * A plain number always keeps its legacy UV meaning, so existing presets are unaffected.
 */
export interface DimensionalValue {
    value: number
    unit: DimensionalUnit
}

/**
 * Bounding box configuration for a shader layer.
 * Positional and size properties each carry their own unit so a power user
 * can mix absolute pixels and relative UV values on the same layer.
 *
 * x/y are the box's TOP-LEFT corner, measured in the frame named by `origin`.
 * rotation has no unit (degrees); the box rotates about its geometric center.
 */
export interface BoundingBoxConfig {
    /** X position of the top-left corner, in the `origin` reference frame (Y-down). */
    x: BoundingBoxDimension
    /** Y position of the top-left corner, in the `origin` reference frame (Y-down). */
    y: BoundingBoxDimension
    /** Width of the bounding box. */
    width: BoundingBoxDimension
    /** Height of the bounding box. */
    height: BoundingBoxDimension
    /** Reference edge x/y are measured from. Default 'top-left'. */
    origin?: BoundingBoxOrigin
    /** Rotation in degrees, around the geometric center of the box. */
    rotation: number
    /** Corner radius for the clip region (generic filter path only, e.g. Dither). */
    cornerRadius?: BoundingBoxDimension
    /**
     * Design-Editor-only hint: when true, the overlay locks resize handles and the W/H panel
     * inputs to the current box aspect ratio (e.g. an image's natural AR). The renderer ignores
     * this — it carries no rendering meaning, only the editor's resize constraint.
     */
    lockAspect?: boolean
}

/**
 * A bounding box as authored on the `boundingBox` prop: every key optional (omitted keys fall back
 * to the full-frame defaults) and every length in any accepted form. `resolveBoundingBox`
 * normalizes it to a complete BoundingBoxConfig.
 *
 * @example { x: '10%', y: 24, width: '50vw', height: '300px', origin: 'bottom-right', cornerRadius: '1rem' }
 */
export interface BoundingBoxInput extends Omit<Partial<BoundingBoxConfig>, 'x' | 'y' | 'width' | 'height' | 'cornerRadius'> {
    x?: BoundingBoxLength
    y?: BoundingBoxLength
    width?: BoundingBoxLength
    height?: BoundingBoxLength
    cornerRadius?: BoundingBoxLength
}

/**
 * How a single prop value maps to/from an overlay pixel dimension.
 * Using string discriminants keeps this serializable (no functions in core types).
 *
 * - 'position-x'          : prop is {x,y}; reads/writes prop.x in UV 0-1 (x * canvasWidth = pixels)
 * - 'position-y'          : prop is {x,y}; reads/writes prop.y in UV 0-1 (y * canvasHeight = pixels)
 * - 'half-canvas-height'  : prop is a half-extent in canvas-height units
 *                           pixels = prop * 2 * canvasHeight  (RoundedRect width/height, Star radius)
 * - 'canvas-height'       : prop maps directly to full visual size in canvas-height units
 *                           pixels = prop * canvasHeight  (Circle radius, which the shader scales by 0.5)
 * - 'degrees'             : prop is a number in degrees; passes through unchanged
 */
export type BoundingBoxBindingType = 'position-x' | 'position-y' | 'half-canvas-height' | 'canvas-height' | 'degrees'

export interface BoundingBoxAxisBinding {
    /** The prop key in the component's props (e.g. 'center', 'radius', 'rotation') */
    prop: string
    /** Conversion formula between the prop value and overlay pixel coordinates */
    as: BoundingBoxBindingType
}

/**
 * Declares which props drive which axes of the bounding box overlay.
 * When propBindings is present, the overlay reads/writes these props directly
 * and NO `boundingBox` metadata is sent to the renderer.
 * Axes that are omitted are not shown in the overlay (e.g., omit rotation for Circle).
 */
export interface BoundingBoxPropBindings {
    x?: BoundingBoxAxisBinding
    y?: BoundingBoxAxisBinding
    width?: BoundingBoxAxisBinding
    height?: BoundingBoxAxisBinding
    rotation?: BoundingBoxAxisBinding
}

/**
 * Opt-in declaration on a ComponentDefinition that enables the bounding box system
 * for that shader. Only shaders with this declaration show bbox handles in the Design Editor.
 */
export interface BoundingBoxDeclaration {
    /**
     * Direct prop bindings for the overlay.
     * When set, dragging bbox handles updates these props directly — no renderer metadata needed.
     * When absent (e.g., Dither), the generic clip-mask path is used and `boundingBox` metadata
     * is sent to the renderer.
     */
    propBindings?: BoundingBoxPropBindings

    /**
     * For the generic (non-prop-binding) path: expected visual aspect ratio (width:height).
     * null = freeform. When non-null the Design Editor constrains resize handles.
     * Not needed when propBindings is set (aspect ratio is implied by the bindings).
     */
    aspectRatio?: number | null

    /**
     * Generators only: when true, an active bounding box treats the box as a RESIZED CANVAS
     * rather than a crop. The renderer feeds the generator a box-local UV plus the box's pixel
     * dimensions (via `effectiveViewportSize`) so its content re-derives to the box (cover/contain
     * re-fit, spiral centre lands at box centre) instead of being cropped. There is no clip mode
     * for these — users mask if they want a crop. The generator's `fragmentNode` must consume
     * `uvContext ?? screenUV` and `effectiveViewportSize ?? viewportSize` for this to take effect.
     */
    supportsResizeFit?: boolean

    /**
     * Composites (Group) only: when true, an active bounding box RESAMPLES the node's composited
     * result into the box like an image — the renderer RTTs the composite and samples it through
     * the box→content UV map (so the full-canvas composite scales/positions/rotates into the box),
     * then clips to the box (with corner radius). This is how a Group behaves like an ImageTexture:
     * a viewport over its children, with px x/y/w/h, origin, corner-radius clip and aspect-lock.
     * Unlike `supportsResizeFit` (which re-derives a GENERATOR via uvContext), this transforms an
     * already-composed subtree. Identity (zero RTT) until the box is non-default.
     */
    boxResamplesContent?: boolean

    /**
     * computeBounds shapes only: when true, the Design Editor's resize handles move W and H
     * independently instead of locking to the content aspect ratio. Set it when `writeBounds`
     * can absorb each axis into its own prop (e.g. Trapezoid width vs height). Without it a
     * computeBounds shape resizes aspect-locked (Blob, Ring, Text).
     */
    freeResize?: boolean

    /**
     * For complex shapes where bounding box geometry depends on multiple props
     * (e.g. Blob: size + deformation + softness). When set, overrides propBindings
     * width/height calculation in the Design Editor overlay.
     *
     * Returns bbox geometry in canvas pixels. centerXPx/centerYPx use Y-down convention.
     */
    computeBounds?: (
        props: Record<string, any>,
        canvasWidth: number,
        canvasHeight: number
    ) => { centerXPx: number; centerYPx: number; widthPx: number; heightPx: number; rotationDeg: number }

    /**
     * Optional softness handle — lets the overlay render a "soft zone" ring and a
     * draggable handle that controls an edge-softness prop (e.g. Blob softness).
     * toPx: prop value → pixel width of the soft zone extending beyond the bbox edge.
     * fromPx: pixel width → prop value (used when the handle is dragged).
     */
    softnessBinding?: {
        prop: string
        toPx: (propValue: number, canvasWidth: number, canvasHeight: number) => number
        fromPx: (px: number, canvasWidth: number, canvasHeight: number) => number
    }

    /**
     * Inverse of computeBounds — converts overlay drag result back to prop updates.
     * Returns a partial props object with only the props that changed.
     * Used alongside computeBounds for multi-prop shapes.
     */
    writeBounds?: (
        bounds: { centerXPx: number; centerYPx: number; widthPx: number; heightPx: number; rotationDeg: number },
        currentProps: Record<string, any>,
        canvasWidth: number,
        canvasHeight: number
    ) => Record<string, any>
}

/**
 * Node metadata including blend information and masking
 */
export interface NodeMetadata {
    /**
     * Blend mode to use when compositing this element with siblings
     */
    blendMode: BlendMode;

    /**
     * Opacity to apply during blending
     */
    opacity: number | undefined;

    /**
     * Whether this node is visible (defaults to true)
     * Hidden nodes (visible=false) are completely excluded from composition
     */
    visible?: boolean;

    /**
     * Optional unique ID that can be referenced by other components
     */
    id?: string;

    /**
     * Optional mask configuration
     */
    mask?: MaskConfig;

    /**
     * Optional prop map configurations — keyed by prop name
     * When set, the renderer replaces that prop's static uniform with a live GPU expression
     * derived from the per-pixel output of the source layer
     */
    maps?: Record<string, PropDriver>

    /**
     * Position among siblings for rendering order (higher renders on top)
     */
    renderOrder: number;

    /**
     * Optional transformation configuration
     * Only creates RTT boundary if values differ from defaults
     */
    transform?: TransformConfig;

    /**
     * Optional bounding box configuration.
     * When set, the renderer uses this for positioning/clipping instead of the legacy transform.
     * Only effective on shaders whose ComponentDefinition declares boundingBoxDeclaration.
     */
    boundingBox?: BoundingBoxConfig;

    /**
     * Optional flow-layout configuration (Groups only). Named `flow` — NOT `layout`/`position`,
     * which collide with existing shader props (ObjectTracker.layout, SineWave/LightLeak/
     * Prism.position). When mode is 'column'/'row' the renderer
     * measures the group's in-flow children (measureNode), stacks them along the main axis with
     * `gap`, aligns on the cross axis, anchors the resulting block on the canvas, and writes each
     * child's position through its native prop channel — no RTT, no resample, recompile-free.
     * Layers without measurable content (filters, full-frame generators) are OUT OF FLOW: they
     * keep their own placement and don't contribute to the block. Default: no layout (mode 'none').
     */
    flow?: LayoutConfig;

    /**
     * Per-child layout escape hatch (CSS `position: absolute` semantics): true opts this node
     * out of an ancestor group's flow — it keeps positioning itself while siblings stack around
     * it. Ignored outside a flow-layout group.
     */
    absolute?: boolean;
}

/**
 * Flex-like layout on a Group. Mirrors the CSS vocabulary developers already have:
 * mode ≈ flex-direction, gap ≈ gap, align ≈ align-items (no 'stretch' until text wrapping
 * exists — nothing can reflow to fill), anchor+anchorOffset place the content-derived block
 * on the canvas (a px offset from the named edge is resize-stable).
 */
export interface LayoutConfig {
    mode: 'none' | 'column' | 'row';
    /** Space between in-flow children. Number = px; a DimensionalValue / length string may carry any unit (uv and % measure the main axis). */
    gap?: number | DimensionalValue | DimensionalString;
    /** Cross-axis alignment of each child within the block. Default 'center'. */
    align?: 'start' | 'center' | 'end';
    /** Where the block pins on the canvas. Default 'center'. */
    anchor?: BoundingBoxOrigin;
    /** Resize-stable px offset from the anchor (same gap-inset convention as position props). */
    anchorOffset?: { x?: number; y?: number };
}

export interface UniformsMap {
    [key: string]: UniformDefinition
}

/**
 * UI control types for props
 */
export type PropUIType = 'text' | 'number' | 'range' | 'color' | 'checkbox' | 'select' | 'origin' | 'position' | 'image-upload' | 'video-upload' | 'map' | 'shape' | 'shape3d' | 'gradient-stops' | 'font-family' | 'font-weight' | 'layer' | 'list'

/**
 * The value kinds a list item's field may hold. Each kind has ONE packing into a vec4 uniform
 * lane (see `utilities/listProps`): position → (x, 1−y, 0, 0) in the transformPosition
 * convention, color → linear rgba, number → (v, 0, 0, 0), boolean → (±1, 0, 0, 0).
 */
export type ListItemFieldKind = 'position' | 'color' | 'number' | 'boolean'

/** One field of a list prop's item: its kind, default, and the scalar UI hints the editor renders it with. */
export interface ListItemFieldConfig {
    kind: ListItemFieldKind
    default: unknown
    label?: string
    description?: string
    min?: number
    max?: number
    step?: number
}

/**
 * Condition for conditional prop visibility
 * Key is the prop name to check, value is the expected value (or array of acceptable values)
 */
export type PropCondition = Record<string, any | any[]>

/**
 * UI metadata for prop controls
 */
export interface PropUIConfig {
    /**
     * The type of UI control to render.
     * Can be a single type string or an array when multiple modes are supported
     * (e.g. ['range', 'map'] renders a dual-mode range/map control).
     */
    type?: PropUIType | PropUIType[]

    /**
     * Minimum value (for number/range inputs)
     */
    min?: number

    /**
     * Maximum value (for number/range inputs)
     */
    max?: number

    /**
     * Step size (for number/range inputs)
     */
    step?: number

    /**
     * Options for select inputs
     */
    options?: Array<{ label: string; value: any }>

    /**
     * Display label (defaults to prop key if not provided)
     */
    label?: string

    /**
     * Conditional visibility - prop is only shown when condition is met
     * Example: { colorMode: 'custom' } - only show when colorMode equals 'custom'
     * Example: { colorMode: ['custom', 'gradient'] } - show when colorMode is either value
     */
    condition?: PropCondition

    /**
     * Group name for organizing props in the settings panel.
     * Props with the same group are displayed together under a section header.
     */
    group?: string

    /**
     * Available unit options for this prop (e.g. ['uv', 'px']).
     * When provided, the settings panel shows a unit toggle chip next to the input.
     * The active unit is stored as a companion flat prop `${propName}_unit`.
     * For 'px' mode, values are displayed as `raw * canvasHeight` and converted on input.
     */
    units?: string[]

    /**
     * Hides the prop from the Design Editor settings panel while keeping it fully
     * functional (presets, code export, MCP). Used for props edited through a dedicated
     * canvas interaction instead — e.g. Text's `text` prop is edited inline on the canvas.
     */
    hidden?: boolean

    /**
     * Marks a scalar distance prop as dimensional (px-capable) with a single line.
     * The value names the px → UV conversion used both by core resolution and by the
     * editor's unit chip (which derives `units: ['%', 'px']` and the conversion axis from it):
     *   - 'canvas-height'      : px = value / canvasHeight        (display: raw * canvasHeight)
     *   - 'half-canvas-height' : px = value / (2 * canvasHeight)  (display: raw * 2 * canvasHeight)
     *   - 'canvas-width'       : px = value / canvasWidth         (display: raw * canvasWidth)
     * A stored plain number always means the original UV unit; only a `{ value, unit: 'px' }`
     * object carries pixels (see DimensionalValue), so existing presets stay byte-identical.
     *
     * The 'count-*' variants are INVERSE: the prop's plain number is a COUNT (e.g. cells across
     * the canvas), and a px value sets the size of one unit, resolving to `count = dimension / px`.
     * This lets a count-style prop (Checkerboard `cells`) toggle to an absolute px cell size while
     * staying resize-stable. Like the forward variants, a plain number is left untouched, so
     * existing presets keep their exact count-based visual output.
     *   - 'count-canvas-height' : count = canvasHeight / px  (px = canvasHeight / count)
     */
    dimensional?: 'canvas-height' | 'half-canvas-height' | 'canvas-width' | 'count-canvas-height'

    /**
     * Which side of the containing box a `%` value on this dimensional prop measures against
     * (`'x'` → width, `'y'` → height). Props with no axis (a radius, a stroke) don't accept `%`,
     * matching CSS. Defaults: `'x'` for the 'canvas-width' conversion, none otherwise.
     */
    dimensionalAxis?: 'x' | 'y'

    /**
     * LIST props (`type: 'list'`): the item's fields, the item cap (the fixed size of the packed
     * uniform arrays), the minimum item count the editor allows, and the per-row label stem
     * ("Light" → "Light 1", "Light 2", …). See `utilities/listProps`.
     */
    item?: Record<string, ListItemFieldConfig>
    maxItems?: number
    minItems?: number
    itemLabel?: string
}

export type PropConfig<T> = {
    default: T
    transform?: (value: T) => any
    compileTime?: boolean
    compileTimeWhen?: (previousValue: T, newValue: T) => boolean
    description?: string
    ui?: PropUIConfig
}

export type ComponentProps = Record<string, any>

/**
 * Function that can be registered to clean up resources when a node is removed
 */
export type CleanupCallback = () => void;

/**
 * Parameters passed to render callbacks
 */
export interface RenderCallbackParams {
  /**
   * Time in seconds since the last frame
   */
  deltaTime: number;
  
  /**
   * Normalized mouse/touch position (0-1)
   */
  pointer: {
    x: number;
    y: number;
  };
  
  /**
   * Whether the pointer is currently active (mouse down or touch active)
   */
  pointerActive: boolean;
  
  /**
   * Current dimensions of the canvas
   */
  dimensions: {
    width: number;
    height: number;
  };
}

/**
 * Function that is called during the render process
 */
export type RenderCallback = (params: RenderCallbackParams) => void;

/**
 * Parameters passed to resize callbacks
 */
export interface ResizeCallbackParams {
  /**
   * New width of the canvas
   */
  width: number;

  /**
   * New height of the canvas
   */
  height: number;
}

/**
 * Function that is called when the canvas resizes
 */
export type ResizeCallback = (params: ResizeCallbackParams) => void;

/**
 * Parameters for fragment node function
 */
export interface FragmentNodeParams {

    /**
     * The parsed uniform data from props
     */
    uniforms: UniformsMap

    /**
     * A vec4 node representing the combined children (if any) of this node
     */
    childNode?: unknown
    
    /**
     * Register a callback to be executed when the node is removed
     * This is useful for cleaning up resources like event listeners
     */
    onCleanup: (callback: CleanupCallback) => void
    
    /**
     * Register a callback to be executed before each render frame
     * This is useful for preparing data or updating uniforms before rendering
     */
    onBeforeRender: (callback: RenderCallback) => void
    
    /**
     * Register a callback to be executed after each render frame
     * This is useful for post-render operations or preparing for the next frame
     */
    onAfterRender: (callback: RenderCallback) => void

    /**
     * Register a callback to be executed when the canvas resizes
     * This is useful for resizing render targets in RTT effects
     */
    onResize: (callback: ResizeCallback) => void

    /**
     * Reference to the canvas element used for rendering
     * This is useful for accessing canvas properties like width and height
     * as well as establish resize observers for RTT effects
     */
    canvas: HTMLCanvasElement
    
    /**
     * Current renderer dimensions (managed by the renderer)
     * Use these instead of canvas.clientWidth/clientHeight to avoid issues with hidden canvases
     */
    dimensions: {
        width: number
        height: number
    }

    /**
     * The logical (authored-frame) resolution as a live uniform (vec2, device-independent).
     *
     * Unlike the built-in `viewportSize`, which tracks the actual GPU backing buffer and
     * therefore shrinks when a host scales render resolution (e.g. the infinite-canvas
     * zoom-out path), this always reflects the unclamped authored frame size. Use it for
     * resolution-dependent effects that should keep a constant cell/feature count across
     * zoom — sizing a pixel grid against this makes the effect scale as a true zoom rather
     * than re-quantizing to the reduced buffer. Falls back to `viewportSize` if absent.
     *
     * @example
     * ```ts
     * fragmentNode: ({ logicalViewportSize }) => {
     *     const res = logicalViewportSize ?? viewportSize
     *     const coord = floor(screenUV.mul(res).div(pixelSize))
     * }
     * ```
     */
    logicalViewportSize?: unknown

    /**
     * Reference to the renderer instance
     * This is useful for manual render target management in RTT effects
     */
    renderer: any

    /**
     * Reference to the <canvas layoutsubtree> element used to capture DOM content.
     * Only populated for shaders with capturesDOM: true in their ComponentDefinition.
     * The element's direct children are available for copyElementImageToTexture.
     */
    domCanvas?: HTMLCanvasElement

    /**
     * Outputs from the compute shader pass, if this component defines a computeNode
     * and the renderer is running in WebGPU mode. Undefined when WebGL fallback is active.
     */
    computeOutputs?: Record<string, any>

    /**
     * Composed UV from a parent distortion's UV-propagation pipeline.
     *
     * When a generator with `acceptsUVContext: true` is the immediate child of a parent
     * that performs UV propagation (e.g. `providesUVContextViaCompute: true`), the renderer
     * supplies the parent's per-pixel UV as this `uvContext` node. The generator should use
     * `uvContext ?? screenUV` for UV-based math, so its own analytical AA (smoothstep + fwidth)
     * operates at the final canvas resolution against the distorted UV — preserving line
     * crispness that an RTT bilinear sample would lose.
     *
     * @example
     * ```ts
     * fragmentNode: ({ uniforms, uvContext }) => {
     *     const uv = uvContext ?? screenUV
     *     // ... rest of the shader unchanged
     * }
     * ```
     *
     */
    uvContext?: unknown

    /**
     * Pre-computed bounding-box clip mask node (float: 1 inside bbox, 0 outside).
     * Only supplied when the node has an active bbox AND requiresRTT: true AND is on
     * the filter clip path (not the generator UV-transform path).
     *
     * Shaders that do expensive per-pixel work (texture sampling, heavy ALU) can use
     * this for a GPU-level early exit before the expensive code runs:
     *
     * @example
     * ```ts
     * fragmentNode: ({ uniforms, childNode, bboxMask }) => {
     *     return Fn(() => {
     *         if (bboxMask) Discard(bboxMask.lessThanEqual(float(0)))
     *         // ... expensive texture sampling and computation ...
     *     })()
     * }
     * ```
     */
    bboxMask?: unknown

    /**
     * Effective viewport size (device pixels) the generator should treat as "the canvas".
     *
     * Normally a generator derives its layout aspect from the real `viewportSize`. When a
     * generator with an active bounding box is in `fit: 'resize'` mode, the renderer supplies
     * the BOX's pixel dimensions here so the generator re-derives exactly as it would on a real
     * canvas resize — `cover`/`contain` re-fit, the spiral centre lands at the box centre, a
     * circle stays a circle. Generators should read aspect/pixel-fit from this:
     *
     * @example
     * ```ts
     * fragmentNode: ({ uniforms, effectiveViewportSize }) => {
     *     const vp = effectiveViewportSize ?? viewportSize
     *     const aspect = vp.x.div(vp.y)
     *     // ... rest unchanged. Screen-space derivatives (fwidth) must keep using the real
     *     // viewportSize — only LAYOUT aspect/pixel-fit uses the effective size.
     * }
     * ```
     */
    effectiveViewportSize?: unknown

    /**
     * Live bounds of this node's first resolvable direct child, as uniform nodes.
     * Only supplied to shaders that declare `wantsBoundsParams: true` (e.g. Repeater),
     * and only when the child's bounds are knowable:
     *   - the child has an explicit active `boundingBox` → its live boundingBoxUniforms
     *     are forwarded directly (auto-updates as the box is dragged), or
     *   - the child's definition exposes bbox geometry (propBindings / computeBounds,
     *     i.e. shapes and shape effects) → the renderer resolves bounds CPU-side and
     *     keeps the uniforms fresh on child prop updates and canvas resizes.
     * All values are canvas-UV, centerY in Y-down convention (0 = top), rotation in
     * degrees — identical to the bounding box uniform conventions. Undefined when no
     * child bounds are resolvable (shaders should fall back to fullscreen).
     */
    childBoundsParams?: BoundsParams

    /**
     * This node's OWN active bounding box as uniform nodes (same shape and
     * conventions as childBoundsParams). Only supplied to shaders that declare
     * `wantsBoundsParams: true` and have an active `boundingBox`. Lets a shader use
     * its own box as a layout/placement area (e.g. Repeater fills the box with its
     * grid) in addition to the standard clip mask the renderer applies on top.
     */
    ownBoundsParams?: BoundsParams
}

/**
 * UV-space rectangle bounds as uniform nodes, used by childBoundsParams /
 * ownBoundsParams. centerY uses the renderer's Y-down convention (0 = top);
 * halfWidth/halfHeight are canvas-UV fractions; rotation is in degrees.
 */
export interface BoundsParams {
    centerX: unknown
    centerY: unknown
    halfWidth: unknown
    halfHeight: unknown
    rotation: unknown
}

/**
 * Return value from a component's computeNode function.
 */
export interface ComputeNodeResult {
    /**
     * Textures or buffers produced by the compute pass, passed to fragmentNode via computeOutputs.
     */
    outputs: Record<string, any>

    /**
     * Called each frame before compute dispatch. Should update uniforms and return
     * the list of compute nodes to dispatch this frame, or null to skip.
     */
    getComputeNodes: (params: RenderCallbackParams) => any[] | null
}

/**
 * The definition of a component within the Shader library.
 */
export interface ComponentDefinition<T extends ComponentProps = ComponentProps> {

    /**
     * The name of the component. Will be automatically prefixed with "ombre", such as "OmbreSpiral".
     */
    name: string

    /**
     * Former public names of this shader — see GpuShaderDefinition.deprecatedNames.
     */
    deprecatedNames?: string[]

    /**
     * The category this shader belongs to for organization in UI
     */
    category?: string

    /**
     * A brief description of what this shader does
     */
    description?: string

    /**
     * Whether this shader requires render-to-texture (RTT) for proper operation.
     * Defaults to false. Set to true for effects that need to sample from child textures.
     */
    requiresRTT?: boolean

    /**
     * Whether this shader requires a child component to function.
     * Defaults to false. Set to true for filter effects that operate on child content.
     * Generative effects that create their own visuals should leave this false/undefined.
     */
    requiresChild?: boolean

    /**
     * When true, the filter composites over its child content using Porter-Duff 'normal'
     * blending rather than `mix()` replacement. Use for stylization filters (Dither, Halftone,
     * ContourLines) whose transparent output pixels mean "show the original content through here."
     * Distortions (Surface3D, Ripple) should leave this false — their transparent areas
     * should stay transparent, not reveal the undistorted original.
     * @default false
     */
    blendWithChildren?: boolean

    /**
     * Whether this shader reads pointer position directly (via onBeforeRender / compute callbacks).
     * When true, global mouse listeners are registered even without mouse-type prop drivers.
     */
    usesPointer?: boolean

    /**
     * Whether this shader (a generator) accepts a composed UV context from a UV-propagating
     * parent. When true, the renderer may route the generator through a non-RTT path where
     * the parent's distortion contributes a per-pixel UV expression, and the generator's
     * `fragmentNode` receives it via `params.uvContext`.
     *
     * The generator must use `params.uvContext ?? screenUV` for its UV-based math.
     *
     * Falls back to the RTT path when the parent / siblings have any of: mask, opacity<1,
     * non-normal blendMode, layer transform, or the generator is not the immediate child
     * of an opted-in distortion.
     *
     *
     * @default false
     */
    acceptsUVContext?: boolean

    /**
     * Whether this distortion exposes its per-pixel UV transformation via a compute
     * pass that writes UV-map storage textures. The renderer reads these in the fragment
     * stage, applies edge handling, and supplies the result as `uvContext` to a child
     * generator that has `acceptsUVContext: true`.
     *
     * Used by heavy distortions (raymarching, multi-step iteration) where inlining the
     * transform into the generator's fragment shader would blow up compile time or
     * per-pixel cost. Cheaper math distortions should use `uvTransformNode` instead (TBD).
     *
     * Component's `computeNode` must populate `outputs.uvMaskMap` (RGBA: u, v, hit*mask, _)
     * and may also expose `outputs.litMap` for lighting.
     *
     * @default false
     */
    providesUVContextViaCompute?: boolean

    /**
     * Whether this shader captures live DOM/HTML content as a texture.
     * When true, the framework template wraps slot children in a <canvas layoutsubtree>
     * and passes it as `domCanvas` in FragmentNodeParams.
     * Requires Chrome Canary with chrome://flags/#canvas-draw-element enabled.
     */
    capturesDOM?: boolean

    /**
     * Whether this shader wants `childBoundsParams` / `ownBoundsParams` in its
     * FragmentNodeParams (see those fields for semantics). Opt-in so the renderer only
     * does the CPU-side child-bounds resolution work for shaders that consume it.
     * @default false
     */
    wantsBoundsParams?: boolean

    /**
     * Optional declaration that this shader supports the bounding box system.
     * When present, the Design Editor shows bbox overlay handles for this layer.
     * Only shaders that opt into the bounding box system declare this — all others behave as before.
     */
    boundingBoxDeclaration?: BoundingBoxDeclaration

    /**
     * Marks this component as experimental/preview.
     * Surfaces a warning tooltip in the design editor layer stack,
     * a callout block in generated documentation, and a console.info on first mount.
     */
    experimental?: {
        /** Short message shown in the layer stack tooltip and console.info */
        message: string
        /** Longer description for the generated docs callout (falls back to message if omitted) */
        docs?: string
        /** Optional link shown in the docs callout only */
        docsLink?: {
            url: string
            label: string
        }
    }

    /**
     * The props that can be passed to the component by the end-user.
     */
    props: {
        [K in keyof T]: PropConfig<T[K]>
    }

    /**
     * A function that returns the fragment node for this component.
     */
    fragmentNode: (params: FragmentNodeParams) => unknown

    /**
     * Optional pure-analytic UV bend — the fast path for distortions.
     *
     * A distortion that exposes `uvRemap` declares that its effect can be expressed
     * as a pure coordinate transformation: given an incoming UV (the coordinate the
     * pixel below it is being asked for) it returns the UV to look up one layer down,
     * plus an updated coverage mask. NO sampling of the child happens here — the
     * child/generator is sampled exactly ONCE downstream, at the fully-composed UV.
     *
     * This lets the renderer fold a run of distortions into one combined coordinate
     * expression and feed it straight into a generator (`acceptsUVContext: true`),
     * skipping the per-distortion render-to-texture passes entirely (zero resamples).
     *
     * Contract:
     * - `uv`     incoming coordinate. The renderer has already `.toVar()`'d it, and
     *            re-`.toVar()`s the result after each step, so the composed expression
     *            stays linear in chain depth (no per-use-site duplication of the tree).
     * - `mask`   incoming coverage (1 = visible). Multiply it for transparent-edge
     *            handling or alpha-bounded effects.
     * - `uniforms` the component's effective uniforms (same shape as fragmentNode).
     * - `aspect` viewport aspect (width/height) for aspect-correct math.
     *
     * Edge modes bake into the remap here: clamp/mirror/wrap via `applyEdgeToUV`,
     * `transparent` by multiplying `mask` (see `composeEdgeRemap`). Keep `fragmentNode`
     * as the fallback — the renderer uses the RTT path whenever the run is ineligible.
     *
     * Presence of `uvRemap` is the opt-in; no separate boolean is needed.
     */
    uvRemap?: (ctx: {
        uv: unknown
        mask: unknown
        uniforms: UniformsMap
        aspect: unknown
        /**
         * Register a per-frame callback for this distortion (wired to its node by
         * the renderer). Animated distortions use it via `createAnimatedTime` so
         * their motion keeps running on the analytic fast path.
         */
        onBeforeRender: (callback: RenderCallback) => void
        /**
         * Outputs from this distortion's own compute pass (if it defines `computeNode`
         * and the renderer is in WebGPU mode). Texture-reading distortions (GridDistortion,
         * Liquify) read their precomputed displacement StorageTexture from here and offset
         * the UV accordingly. Undefined when the distortion has no compute pass. The remap
         * must sample only its OWN compute outputs — never the child/generator.
         */
        computeOutputs?: Record<string, any>
    }) => { uv: unknown; mask: unknown }

    /**
     * Optional GPU compute pass. Called once at composition time when the renderer
     * is running in WebGPU mode. Returns compute nodes to dispatch each frame and
     * output textures/buffers that are forwarded to fragmentNode via computeOutputs.
     *
     * When the renderer falls back to WebGL, this function is never called and
     * computeOutputs will be undefined — the fragmentNode should implement a CPU
     * fallback for that path.
     */
    computeNode?: (params: FragmentNodeParams) => ComputeNodeResult | null

    /**
     * Optional function returning per-prop custom sample UVs for prop maps.
     * Shaders with cell-based patterns (e.g. DotGrid, Grid) should implement this
     * to return a cell-center UV for each mapped prop, preventing per-fragment
     * variation within a single cell from causing visual clipping artifacts.
     * Returned nodes are used instead of screenUV when sampling the map texture.
     */
    mapSampleUVs?: (uniforms: UniformsMap) => Record<string, any>
}
