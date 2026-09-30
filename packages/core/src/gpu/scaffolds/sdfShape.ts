/**
 * `defineSdfShapeShader` — the definition-level scaffold for the analytic 2D shape fleet.
 *
 * ## What the fleet actually shares
 *
 * Fifteen shape shaders (Circle, Ellipse, RoundedRect, Polygon, Star, Heart, Cross, Crescent,
 * Vesica, Teardrop, Arc, Ring, Flower, Parallelogram, Trapezoid) are the same shader three times
 * over:
 *
 * 1. **Props.** `origin`, `color`, `center`, 1–3 shape props, `rotation`, the four-prop stroke
 *    block, `colorSpace`. Only the shape props and a handful of descriptions vary.
 * 2. **Body.** `shapeLocalCoords` → one SDF call → `strokeMaskFromSdf`.
 * 3. **Fragment.** resolve `uvContext ?? ctx.uv` → call the body → `sdf.fillStrokeColor`.
 *
 * This factory owns (1) and (3), plus the declarative `boundingBoxDeclaration`.
 *
 * ## The body is algebra: `distance`
 *
 * A shape file declares its geometry as one expression — `distance: (local, u) => Expr` — over the
 * shape-local frame the factory provides (aspect-corrected, centered, rotated). The factory builds
 * the frame with `shapeLocalCoords`, evaluates the distance expression, and masks it with
 * `strokeMaskFromSdf`. The geometry vocabulary (`circle`, `heart`, `star`, …) lives in
 * `@coreroot/std/shape`; specifics like Circle's `radius · 0.5` edge or Arc's degrees→half-angle
 * are plain algebra at the declaration site.
 *
 * ## Rotation-free shapes
 *
 * Circle and Ring are rotationally symmetric: `rotatable: false` declares no `rotation` prop,
 * binds no rotation box axis, and the frame is built with rotation pinned to zero.
 *
 * ## Prop order is load-bearing
 *
 * `props` key order is not cosmetic: `createGpuUniformsMap` walks it to build the node's uniform
 * struct, so reordering props reorders WGSL struct members. It also drives the Design Editor panel
 * order via the generated `shaderMetadata.ts`. The factory emits the fleet-canonical order (which
 * fourteen shaders already use verbatim); Circle interleaves `center` differently and passes
 * `propOrder` to keep its exact sequence.
 */
import type {Expr, GpuFragmentParams, GpuShaderDefinition} from '../contract'
import {asLocal, call, expr} from '../composer'
import {sdf} from '../kit'
import type {
    BoundingBoxBindingType,
    BoundingBoxDeclaration,
    BoundingBoxPropBindings,
    ComponentProps,
    PropConfig,
} from '../../types'
import {transformColor} from '../../utilities/transformations'
import {
    centerPropConfig,
    originProp,
    shapeColorSpaceProp,
    shapeStrokeProps,
} from '../../utilities/propConfigs'

/**
 * Any prop config, whatever the prop's value type. Shape props are heterogeneous by nature (a
 * number, a `DimensionalValue`, a color string), and the factory only ever moves them around, so
 * the value type is deliberately unconstrained here — the shader's own `ComponentProps` interface is
 * where the per-prop types are stated.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyPropConfig = PropConfig<any>

/**
 * How the shape's bounding box is declared.
 *
 * `size` is the declarative form thirteen shapes use: x/y bind `center`, width/height bind the
 * named size props, and rotation binds `rotation` unless `rotatable: false`. `as` defaults to
 * `'half-canvas-height'` (the prop is a half-extent); Circle passes `'canvas-height'` because its
 * `radius` names the full visual size and the shader halves it internally.
 *
 * `custom` is the escape hatch for the two shapes whose box cannot be read off a single prop: Ring
 * (outer edge = midline radius + half-band thickness) and Trapezoid (visual width = the WIDER of
 * its two edges). Those pass a complete `BoundingBoxDeclaration` — `computeBounds` / `writeBounds`
 * are per-shape geometry, not something a factory can infer.
 */
export type SdfShapeBounds =
    | {size: {width: string; height: string; as?: BoundingBoxBindingType}; custom?: never}
    | {custom: BoundingBoxDeclaration; size?: never}

export interface SdfShapeShaderSpec {
    name: string
    description?: string
    /** Defaults to `'Shapes'`. */
    category?: string

    /**
     * The shape's geometry, as algebra over the shape-local frame (aspect-corrected, centered,
     * rotated — y grows downward like the SDF kit). `u(name)` reads a prop uniform. Returns the
     * signed distance; the factory applies the fill/stroke/softness mask.
     */
    distance: (local: {x: Expr; y: Expr}, u: (name: string) => Expr) => Expr

    /** The `color` prop's description ("Fill color of the star"). */
    colorDescription: string
    /** The `center` prop's description ("Center position of the star"). */
    centerDescription: string
    /** Circle labels its center prop 'Center Position'; the rest use the factory default. */
    centerLabel?: string

    /** The 1–3 props that make this shape itself, in declaration order. */
    shapeProps: Record<string, AnyPropConfig>

    /** `false` for the rotationally symmetric shapes (Circle, Ring): no rotation prop, no rotation box axis. */
    rotatable?: boolean
    /** Cross explains the 45° case; everything else uses the default wording. */
    rotationDescription?: string

    bounds: SdfShapeBounds

    /** Passed straight to `shapeStrokeProps` (Ring's wording + max, Circle's wording). */
    stroke?: Parameters<typeof shapeStrokeProps>[0]
    /** Circle's `colorSpace` description mentions soft edges. */
    colorSpaceDescription?: string

    /**
     * Prop configs that REPLACE a factory-built one, keyed by prop name. Spread last, so a replaced
     * prop keeps its position in the key order. Circle uses this for its widened, map-driveable
     * `softness` / `strokeThickness`.
     */
    propOverrides?: Record<string, AnyPropConfig>
    /**
     * Explicit prop key order, when the shader's current order is not the fleet-canonical one.
     * Must list every prop exactly once — a mismatch throws, because a silently dropped prop would
     * change the uniform struct.
     */
    propOrder?: string[]
}

/** The rotation prop, byte-identical across the twelve rotatable shapes bar Cross's wording. */
function rotationProp(description: string): PropConfig<number> {
    return {
        default: 0,
        description,
        ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Shape'},
    }
}

/** The fill-color prop. `#ffffff` fleet-wide; only the description names the shape. */
function colorProp(description: string): PropConfig<Parameters<typeof transformColor>[0]> {
    return {
        default: '#ffffff',
        transform: transformColor,
        description,
        ui: {type: 'color', label: 'Color', group: 'Colors'},
    }
}

function buildBounds(bounds: SdfShapeBounds, rotatable: boolean): BoundingBoxDeclaration {
    if (bounds.custom) return bounds.custom
    const as = bounds.size.as ?? 'half-canvas-height'
    const propBindings: BoundingBoxPropBindings = {
        x: {prop: 'center', as: 'position-x'},
        y: {prop: 'center', as: 'position-y'},
        width: {prop: bounds.size.width, as},
        height: {prop: bounds.size.height, as},
    }
    // A circle/ring looks the same at any angle, so the overlay shows no rotation handle.
    if (rotatable) propBindings.rotation = {prop: 'rotation', as: 'degrees'}
    return {propBindings}
}

function orderProps(
    props: Record<string, AnyPropConfig>,
    order: string[] | undefined,
    shaderName: string,
): Record<string, AnyPropConfig> {
    if (!order) return props
    const keys = Object.keys(props)
    const missing = keys.filter((k) => !order.includes(k))
    const unknown = order.filter((k) => !(k in props))
    if (missing.length || unknown.length) {
        throw new Error(
            `defineSdfShapeShader(${shaderName}): propOrder must list every prop exactly once ` +
                `(missing: ${missing.join(', ') || 'none'}; unknown: ${unknown.join(', ') || 'none'})`,
        )
    }
    const out: Record<string, AnyPropConfig> = {}
    for (const key of order) out[key] = props[key]
    return out
}

/**
 * Build a complete analytic-2D-shape shader definition from its shape props, its mask body fn and
 * its box declaration.
 *
 * The emitted definition is byte-identical to the hand-written one for all fifteen migrated shapes
 * (Gate A): same props in the same order, same `boundingBoxDeclaration`, and a fragment that emits
 * the same `call` sequence.
 */
export function defineSdfShapeShader<T extends ComponentProps = ComponentProps>(
    spec: SdfShapeShaderSpec,
): GpuShaderDefinition<T> {
    const rotatable = spec.rotatable !== false
    const stroke = shapeStrokeProps(spec.stroke)

    const props: Record<string, AnyPropConfig> = {
        origin: originProp() as AnyPropConfig,
        color: colorProp(spec.colorDescription) as AnyPropConfig,
        center: centerPropConfig(spec.centerDescription, {
            units: ['%', 'px'],
            ...(spec.centerLabel ? {label: spec.centerLabel} : {}),
        }) as AnyPropConfig,
        ...spec.shapeProps,
        ...(rotatable
            ? {rotation: rotationProp(spec.rotationDescription ?? 'Rotation in degrees') as AnyPropConfig}
            : {}),
        softness: stroke.softness as AnyPropConfig,
        strokeThickness: stroke.strokeThickness as AnyPropConfig,
        strokeColor: stroke.strokeColor as AnyPropConfig,
        strokePosition: stroke.strokePosition as AnyPropConfig,
        colorSpace: shapeColorSpaceProp(
            spec.colorSpaceDescription ? {description: spec.colorSpaceDescription} : undefined,
        ) as AnyPropConfig,
        // Spread last so a replacement keeps the position its factory-built original held.
        ...spec.propOverrides,
    }

    return {
        name: spec.name,
        category: spec.category ?? 'Shapes',
        description: spec.description,
        acceptsUVContext: true,
        boundingBoxDeclaration: buildBounds(spec.bounds, rotatable),
        props: orderProps(props, spec.propOrder, spec.name) as GpuShaderDefinition<T>['props'],

        // Generator fragment. Standalone → ctx.uv; wrapped by a UV-propagating parent → the
        // composed uvContext. `colorSpace` is compile-time (propValues), so it selects the
        // mixColors variant in JS and only that variant's math is emitted.
        fragment: (params: GpuFragmentParams): Expr => {
            const {uniforms, ctx, propValues} = params
            const uv = params.uvContext ?? ctx.uv
            const colorSpaceMode = (propValues.colorSpace as number) ?? 0

            const mask = asLocal(sdfShapeMask(spec, uniforms, uv, ctx.aspect), 'shapeMask')
            return sdf.fillStrokeColor(mask, uniforms.color, uniforms.strokeColor, colorSpaceMode)
        },
    }
}

/** Shape-local frame → the spec's distance algebra → the shared fill/stroke mask. */
function sdfShapeMask(
    spec: SdfShapeShaderSpec,
    uniforms: Record<string, Expr>,
    uv: Expr,
    aspect: Expr,
): Expr {
    const rotation = uniforms.rotation ?? expr('0f')
    const frame = asLocal(
        call(sdf.shapeLocalCoords, 'shapeLocalCoords', [uniforms.center, rotation, uv, aspect]),
        'shapeLocal',
    )
    const u = (name: string): Expr => {
        const value = uniforms[name]
        if (!value) throw new Error(`defineSdfShapeShader(${spec.name}): distance reads unknown prop '${name}'`)
        return value
    }
    const dist = spec.distance({x: frame.member('x'), y: frame.member('y')}, u)
    return call(sdf.strokeMaskFromSdf, 'strokeMaskFromSdf', [
        dist, uniforms.softness, uniforms.strokeThickness, uniforms.strokePosition,
    ])
}
