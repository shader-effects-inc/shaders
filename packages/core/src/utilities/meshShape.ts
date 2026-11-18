// Is this shape a 3D model, and where does it come from?
//
// A model shape is `{ type: 'mesh3D', meshUrl, size, rotX, rotY, rotZ }` in the `shape` prop (the
// shape picker also sets `shapeType: 'mesh3D'`). The engine draws the model's nearest and farthest
// surfaces into the shape effects' volumetric field each time the shape changes
// (`gpu/kit/meshField.ts`) — nothing is pre-generated or stored besides the .glb itself.

/** The shape type of an uploaded 3D model. */
export const MESH3D_TYPES = ['mesh3D'] as const
export type Mesh3DType = typeof MESH3D_TYPES[number]

export function isMesh3dShapeType(shapeType: unknown): shapeType is Mesh3DType {
    return shapeType === 'mesh3D'
}

/**
 * Defaults for a model shape's sub-props (stored in the shape JSON; shared by the field, bounds
 * and UI). `size` is the radius of the model's bounding sphere in field units (0.5 = half the
 * shape box), so any model fills the same space whatever units it was made in.
 */
export const MESH3D_DEFAULTS: Record<string, number> = {
    size: 0.42, rotX: 0, rotY: 0, rotZ: 0,
}

/** The largest `size` the field can hold (the march's camera plane sits 1.2 field units out). */
export const MESH3D_MAX_SIZE = 1

const parseCache = new Map<string, Record<string, unknown>>()

function shapeConfig(shape: unknown): Record<string, unknown> {
    if (shape && typeof shape === 'object') return shape as Record<string, unknown>
    if (typeof shape !== 'string' || !shape) return {}
    let cfg = parseCache.get(shape)
    if (!cfg) {
        try { cfg = JSON.parse(shape) as Record<string, unknown> } catch { cfg = {} }
        if (parseCache.size >= 64) parseCache.clear()
        parseCache.set(shape, cfg)
    }
    return cfg
}

/** The model URL a `shape` prop (JSON string or object) carries, or ''. */
export function meshUrlOfShape(shape: unknown): string {
    const url = shapeConfig(shape).meshUrl
    return typeof url === 'string' ? url : ''
}

/** True when the shape is a 3D model (a `mesh3D` with a `meshUrl`). */
export function isMeshShape(shape: unknown): boolean {
    const cfg = shapeConfig(shape)
    return cfg.type === 'mesh3D' && typeof cfg.meshUrl === 'string' && cfg.meshUrl !== ''
}
