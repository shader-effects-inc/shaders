/**
 * Layer transform math.
 *
 * Split by execution stage:
 *   - CPU (plain TS): `needsTransformation`, `boundingBoxToUVParams`, `screenUVToBoxLocal`,
 *     `boxLocalToScreenUV` and their interfaces. No GPU dependency — used by the renderer's
 *     mouse-position driver and the Design Editor's handles.
 *   - GPU (`'use gpu'` tgpu fns): `applyUVTransform`, `applyInverseUVTransform`,
 *     `applyRectangularClipMask`, `boundingBoxToGeneratorUVContext`. Uniform values and the screen
 *     UV arrive as explicit fn args (the composer wires them); the GPU fns that operate on the
 *     screen UV take it as their first argument. `PI` inlines as `Math.PI`.
 */
import {tgpu, d, std} from './index'
import type {BoundingBoxConfig} from '../../types'
import {boxCornerRadiusUV, boxLengthUV} from '../../utilities/boundingBox'

/** Configuration for UV transformations. */
export interface TransformConfig {
    offsetX: number
    offsetY: number
    rotation: number  // in degrees
    scale: number
    anchorX: number   // 0-1 normalized
    anchorY: number   // 0-1 normalized
    edges: 'stretch' | 'transparent' | 'mirror' | 'wrap'
}

/**
 * Checks if transformation config requires actual transformation. Returns true only if
 * values differ from defaults. Critical for performance — default transforms have ZERO
 * overhead. (CPU.)
 */
export const needsTransformation = (transform: TransformConfig | undefined): boolean => {
    if (!transform) return false

    return (
        transform.offsetX !== 0 ||
        transform.offsetY !== 0 ||
        transform.rotation !== 0 ||
        transform.scale !== 1 ||
        transform.anchorX !== 0.5 ||
        transform.anchorY !== 0.5
    )
}

/**
 * Applies 2D transformations to UV coordinates (translation, rotation, scale, custom anchor,
 * aspect-ratio aware). Transform order: center at anchor → scale → aspect-normalize → rotate
 * in square space → un-normalize → offset → un-center.
 */
export const applyUVTransform = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.vec2f,
)((uv, offsetX, offsetY, rotation, scale, anchorX, anchorY, aspectRatio) => {
    'use gpu'
    const rotationRad = rotation * Math.PI / 180.0
    const cosAngle = std.cos(rotationRad)
    const sinAngle = std.sin(rotationRad)

    // Step 1: center at anchor; Step 2: scale
    const centeredX = (uv.x - anchorX) / scale
    const centeredY = (uv.y - anchorY) / scale

    // Step 3: normalize to square space (stretch X by aspect)
    const squareX = centeredX * aspectRatio
    const squareY = centeredY

    // Step 4: rotate in square space
    const rotatedSquareX = squareX * cosAngle - squareY * sinAngle
    const rotatedSquareY = squareX * sinAngle + squareY * cosAngle

    // Step 5: back to UV space; Step 6: offset; Step 7: un-center
    const finalX = rotatedSquareX / aspectRatio - offsetX + anchorX
    const finalY = rotatedSquareY - offsetY + anchorY
    return d.vec2f(finalX, finalY)
})

/**
 * Applies the INVERSE of `applyUVTransform` (reverse order, inverted operations). Used in UV
 * context propagation to apply view transformations without RTT. Uniform values are explicit args.
 */
export const applyInverseUVTransform = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.vec2f,
)((uv, offsetX, offsetY, rotation, scale, anchorX, anchorY, aspectRatio) => {
    'use gpu'
    // Step 1: center at anchor; Step 2: offset inversely (add)
    const centeredX = (uv.x - anchorX) + offsetX
    const centeredY = (uv.y - anchorY) + offsetY

    // Step 3: normalize to square space
    const squareX = centeredX * aspectRatio
    const squareY = centeredY

    // Step 4: inverse rotation (negate angle)
    const rotationRad = -(rotation * Math.PI / 180.0)
    const cosAngle = std.cos(rotationRad)
    const sinAngle = std.sin(rotationRad)
    const rotatedSquareX = squareX * cosAngle - squareY * sinAngle
    const rotatedSquareY = squareX * sinAngle + squareY * cosAngle

    // Step 5: back from square space; Step 6: inverse scale (multiply); Step 7: un-center
    const scaledX = (rotatedSquareX / aspectRatio) * scale
    const scaledY = rotatedSquareY * scale
    return d.vec2f(scaledX + anchorX, scaledY + anchorY)
})

/**
 * Pre-computed UV-space geometry derived from a BoundingBoxConfig. All values in UV space
 * (0-1 fractions), Y-down convention.
 */
export interface BoundingBoxUVParams {
    centerX: number
    centerY: number
    halfWidthUV: number
    halfHeightUV: number
    cornerRadiusUV: number
    rotation: number
}

/**
 * Converts a BoundingBoxConfig to UV-space geometry params for the renderer. Flips Y so the
 * output is ready for GPU use. (CPU.)
 */
export const boundingBoxToUVParams = (
    bbox: BoundingBoxConfig,
    canvasWidth: number,
    canvasHeight: number,
): BoundingBoxUVParams => {
    // Any unit → fraction of the canvas on its axis (`uv` already is; `%` measures the containing
    // box on the same axis — the canvas until frames exist). x/y stay insets from the origin edge.
    const xUV = boxLengthUV(bbox.x, 'x', canvasWidth, canvasHeight)
    const yUV = boxLengthUV(bbox.y, 'y', canvasWidth, canvasHeight)
    const widthUV = boxLengthUV(bbox.width, 'x', canvasWidth, canvasHeight)
    const heightUV = boxLengthUV(bbox.height, 'y', canvasWidth, canvasHeight)

    const resolveTopLeft = (vUV: number, sizeUV: number, edge: 'start' | 'end' | 'center') =>
        edge === 'end' ? 1 - vUV - sizeUV
        : edge === 'center' ? 0.5 + vUV - sizeUV / 2
        : vUV

    const origin = bbox.origin ?? 'top-left'
    const ox = origin.includes('left') ? 'start' : origin.includes('right') ? 'end' : 'center'
    const oy = origin.includes('top') ? 'start' : origin.includes('bottom') ? 'end' : 'center'
    const leftUV = resolveTopLeft(xUV, widthUV, ox)
    const topUV = resolveTopLeft(yUV, heightUV, oy)

    const centerXUV_down = leftUV + widthUV / 2
    const centerYUV_down = topUV + heightUV / 2

    const cornerRadiusUV = boxCornerRadiusUV(bbox.cornerRadius, canvasWidth, canvasHeight)

    return {
        centerX: centerXUV_down,
        centerY: centerYUV_down,
        halfWidthUV: widthUV / 2,
        halfHeightUV: heightUV / 2,
        cornerRadiusUV,
        rotation: bbox.rotation,
    }
}

/**
 * Returns 1 inside the rotated bounding-box rectangle, 0 outside (rounded-rect SDF).
 * Rotation is aspect-corrected. The screen UV is the explicit `uv` arg.
 *
 * `aaFeather` is the HALF-WIDTH of the coverage ramp, in the same square space the SDF is
 * measured in (x aspect-corrected, y = canvas-height UV) — so half a device pixel is
 * `0.5 / viewportHeightPx`. The ramp is centred on the true edge, which means a pixel whose
 * centre sits half a pixel inside reads a full 1.0 (no darkened border row on a box whose edge
 * lands on the canvas edge) while a partially covered edge pixel gets fractional alpha instead
 * of a binary one. `aaFeather <= 0` selects the legacy binary step exactly — including 1.0 ON the
 * edge (`sdf == 0`), which the ramp reports as half coverage. Only worth asking for when the caller
 * genuinely wants a binary mask, since a hard edge visibly stair-steps as soon as anything
 * downstream (a distortion, a layer transform) magnifies it.
 */
export const applyRectangularClipMask = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((uv, centerX, centerY, halfWidth, halfHeight, cornerRadius, rotation, aspectRatio, aaFeather) => {
    'use gpu'
    const centeredX = uv.x - centerX
    const centeredY = uv.y - centerY

    const rotRad = -(rotation * Math.PI / 180.0)
    const cosA = std.cos(rotRad)
    const sinA = std.sin(rotRad)

    const sqX = centeredX * aspectRatio
    const sqY = centeredY
    const rotatedX = sqX * cosA - sqY * sinA
    const rotatedY = sqX * sinA + sqY * cosA

    const halfWidthSq = halfWidth * aspectRatio
    const halfHeightSq = halfHeight
    const innerHalfW = halfWidthSq - cornerRadius
    const innerHalfH = halfHeightSq - cornerRadius
    const qx = std.abs(rotatedX) - innerHalfW
    const qy = std.abs(rotatedY) - innerHalfH
    const qClamped = d.vec2f(std.max(qx, 0.0), std.max(qy, 0.0))
    const sdf = std.length(qClamped) + std.min(std.max(qx, qy), 0.0) - cornerRadius
    // The ramp: floor the feather so smoothstep never gets edge0 == edge1 (undefined).
    const feather = std.max(aaFeather, 1e-7)
    const ramp = 1.0 - std.smoothstep(-feather, feather, sdf)
    // A floored feather is NOT the legacy step: at sdf == 0 the ramp sits at 0.5, while the binary
    // mask counts the edge as inside. That is a whole row of pixels wherever an axis-aligned edge
    // lands on pixel centres, so a caller asking for a binary mask gets one. d.f32 wrappers: bare
    // integer-valued literals in a select transpile to i32.
    const binary = std.select(d.f32(0), d.f32(1), sdf <= 0.0)
    return std.select(ramp, binary, aaFeather <= 0.0)
})

/**
 * Maps screen UV into the bounding box's local [0,1] space for the generator "resize" fit
 * mode (non-uniform fill of both axes, box centre → local 0.5,0.5, rotation applied).
 * Reduces to identity at a full-canvas box. The screen UV is the explicit `uv` arg.
 */
export const boundingBoxToGeneratorUVContext = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.vec2f,
)((uv, centerX, centerY, halfWidth, halfHeight, rotation, aspectRatio) => {
    'use gpu'
    const centeredX = uv.x - centerX
    const centeredY = uv.y - centerY

    const rotRad = -(rotation * Math.PI / 180.0)
    const cosA = std.cos(rotRad)
    const sinA = std.sin(rotRad)
    const sqX = centeredX * aspectRatio
    const sqY = centeredY
    const rotatedX = sqX * cosA - sqY * sinA
    const rotatedY = sqX * sinA + sqY * cosA

    const halfWidthSq = std.max(halfWidth * aspectRatio, 1e-6)
    const halfHeightSq = std.max(halfHeight, 1e-6)
    const localX = rotatedX / (halfWidthSq * 2.0) + 0.5
    const localY = rotatedY / (halfHeightSq * 2.0) + 0.5
    return d.vec2f(localX, localY)
})

/**
 * Box geometry for the CPU screen↔box-local mapping. UV fractions, renderer Y-down;
 * rotation in degrees; aspectRatio = canvas width / height.
 */
export interface BoxLocalGeometry {
    centerX: number
    centerY: number
    halfWidthUV: number
    halfHeightUV: number
    rotationDeg: number
    aspectRatio: number
}

/**
 * CPU/JS equivalent of `boundingBoxToGeneratorUVContext`, in plain numbers. Used by the mouse
 * driver and Design Editor handles so GPU content, mouse tracking, and drag handles agree.
 * Exact inverse of `boxLocalToScreenUV`. (CPU.)
 */
export const screenUVToBoxLocal = (sx: number, sy: number, g: BoxLocalGeometry): {x: number; y: number} => {
    const th = g.rotationDeg * Math.PI / 180
    const c = Math.cos(th)
    const s = Math.sin(th)
    const sqx = (sx - g.centerX) * g.aspectRatio
    const sqy = (sy - g.centerY)
    const rx = sqx * c + sqy * s
    const ry = sqx * (-s) + sqy * c
    const hwSq = Math.max(g.halfWidthUV * g.aspectRatio, 1e-6)
    const hhSq = Math.max(g.halfHeightUV, 1e-6)
    return {x: rx / (2 * hwSq) + 0.5, y: ry / (2 * hhSq) + 0.5}
}

/** Box-local fraction [0,1] (Y-down) → screen-UV point (Y-down). Inverse of screenUVToBoxLocal. */
export const boxLocalToScreenUV = (bx: number, by: number, g: BoxLocalGeometry): {x: number; y: number} => {
    const th = g.rotationDeg * Math.PI / 180
    const c = Math.cos(th)
    const s = Math.sin(th)
    const lx = (bx - 0.5) * 2 * (g.halfWidthUV * g.aspectRatio)
    const ly = (by - 0.5) * 2 * g.halfHeightUV
    const sqx = lx * c - ly * s
    const sqy = lx * s + ly * c
    return {x: sqx / g.aspectRatio + g.centerX, y: sqy + g.centerY}
}
