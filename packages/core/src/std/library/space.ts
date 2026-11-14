/**
 * Space: rotating, polar mapping and tiling the plane into cells. Angles are radians,
 * counter-clockwise, 0 along +x. Tilings return a struct with the point in cell-local
 * coordinates and a cell id that is unique per cell (feed it to cellHash for a random value).
 */
import {libFn} from '../wgslLibrary'
import {hash21, hash33} from './noise'

export const rotate2 = libFn({
    name: 'rotate2',
    family: 'space',
    signature: 'rotate2(p: vec2f, angle: f32) -> vec2f  // angle in radians, counter-clockwise',
    doc: 'Rotate a point about the origin. To spin a shape, rotate the sample point the opposite way: sdBox(rotate2(p, -angle), b).',
    code: `
fn rotate2(p: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(c * p.x - s * p.y, s * p.x + c * p.y);
}
`,
})

export const eulerRotate = libFn({
    name: 'eulerRotate',
    family: 'space',
    signature: 'eulerRotate(v: vec3f, degrees: vec3f) -> vec3f  // view space -> object space',
    doc: 'Rotate a 3D vector into an object\'s own space by Euler angles in degrees (x, y, z), in the order the library\'s 3D shapes use (Rz·Rx·Ry). Turn a view ray into object space with it: rd = eulerRotate(rd, vec3f(tiltX, tiltY, spin)). This is the convention the editor\'s rotation gizmo (rotationGizmo) drives; take object-space normals back to view space with eulerRotateInverse.',
    code: `
fn eulerRotate(v: vec3f, degrees: vec3f) -> vec3f {
    let a = radians(degrees);
    let cx = cos(a.x); let sx = sin(a.x);
    let cy = cos(a.y); let sy = sin(a.y);
    let cz = cos(a.z); let sz = sin(a.z);
    let x1 = v.x * cy + v.z * sy;
    let z1 = v.z * cy - v.x * sy;
    let y2 = v.y * cx - z1 * sx;
    let z2 = v.y * sx + z1 * cx;
    return vec3f(x1 * cz - y2 * sz, x1 * sz + y2 * cz, z2);
}
`,
})

export const eulerRotateInverse = libFn({
    name: 'eulerRotateInverse',
    family: 'space',
    signature: 'eulerRotateInverse(v: vec3f, degrees: vec3f) -> vec3f  // object space -> view space',
    doc: 'Undo eulerRotate: take an object-space vector (a normal, a reflection) back to view space, where the camera and studio lights live.',
    code: `
fn eulerRotateInverse(v: vec3f, degrees: vec3f) -> vec3f {
    let a = radians(degrees);
    let cx = cos(a.x); let sx = sin(a.x);
    let cy = cos(a.y); let sy = sin(a.y);
    let cz = cos(a.z); let sz = sin(a.z);
    let x1 = v.x * cz + v.y * sz;
    let y1 = v.y * cz - v.x * sz;
    let y2 = y1 * cx + v.z * sx;
    let z2 = v.z * cx - y1 * sx;
    return vec3f(x1 * cy - z2 * sy, y2, x1 * sy + z2 * cy);
}
`,
})

export const toPolar = libFn({
    name: 'toPolar',
    family: 'space',
    signature: 'toPolar(p: vec2f) -> vec2f  // (angle -pi..pi, radius >= 0)',
    doc: 'Cartesian to polar: x is the angle (radians from +x, counter-clockwise), y the distance from the origin.',
    code: `
fn toPolar(p: vec2f) -> vec2f {
    return vec2f(atan2(p.y, p.x), length(p));
}
`,
})

export const fromPolar = libFn({
    name: 'fromPolar',
    family: 'space',
    signature: 'fromPolar(polar: vec2f) -> vec2f  // (angle radians, radius) in',
    doc: 'Polar (angle, radius) back to Cartesian. The inverse of toPolar.',
    code: `
fn fromPolar(polar: vec2f) -> vec2f {
    return polar.y * vec2f(cos(polar.x), sin(polar.x));
}
`,
})

export const logPolar = libFn({
    name: 'logPolar',
    family: 'space',
    signature: 'logPolar(p: vec2f) -> vec2f  // (log radius, angle -pi..pi)',
    doc: 'Log-polar coordinates: x is log(radius), y the angle. Patterns tiled in this space repeat at every scale toward the centre (Droste zooms, spiral tunnels); scrolling x zooms forever.',
    code: `
fn logPolar(p: vec2f) -> vec2f {
    return vec2f(log(max(length(p), 1e-6)), atan2(p.y, p.x));
}
`,
})

export const cellHash = libFn({
    name: 'cellHash',
    family: 'space',
    signature: 'cellHash(id: vec2f) -> f32  // 0..1',
    doc: 'A stable random value for a cell id from any tiling here (hexTile, triTile, radialTile, voronoiCell, or floor(p)).',
    deps: [hash21],
    code: `
fn cellHash(id: vec2f) -> f32 {
    return hash21(id);
}
`,
})

export const hexTile = libFn({
    name: 'hexTile',
    family: 'space',
    signature: 'hexTile(p: vec2f) -> HexCell { local: vec2f /* from the cell centre */, id: vec2f /* the cell centre */, edge: f32 /* distance to the cell edge, 0..0.5 */ }',
    doc: 'Tile the plane with pointy-top hexagons, centres one unit apart. local is p relative to its hexagon\'s centre, id that centre, edge the distance to the nearest side (0 on the border, 0.5 at the centre): aaStep(0.05, cell.edge) draws the cells with gaps.',
    code: `
struct HexCell { local: vec2f, id: vec2f, edge: f32 }
fn hexTile(p: vec2f) -> HexCell {
    let s = vec2f(1.0, 1.7320508);
    let ca = (floor(p / s) + 0.5) * s;
    let cb = floor(p / s + 0.5) * s;
    let la = p - ca;
    let lb = p - cb;
    let useA = dot(la, la) < dot(lb, lb);
    let l = select(lb, la, useA);
    let a = abs(l);
    let edge = 0.5 - max(a.x, a.x * 0.5 + a.y * 0.8660254);
    return HexCell(l, select(cb, ca, useA), edge);
}
`,
})

export const triTile = libFn({
    name: 'triTile',
    family: 'space',
    signature: 'triTile(p: vec2f) -> TriCell { local: vec2f /* from the triangle centre */, id: vec2f /* whole numbers, unique */, edge: f32 /* distance to the nearest side, 0..0.289 */ }',
    doc: 'Tile the plane with equilateral triangles of side 1, alternating up and down. local is p relative to its triangle\'s centre, edge the distance to the nearest side; id.x is odd for the upward-pointing triangles.',
    code: `
struct TriCell { local: vec2f, id: vec2f, edge: f32 }
fn triTile(p: vec2f) -> TriCell {
    let sk = vec2f(p.x - p.y * 0.57735027, p.y * 1.15470054);
    let cell = floor(sk);
    let f = sk - cell;
    let up = f.x + f.y > 1.0;
    let cen = cell + select(vec2f(1.0 / 3.0), vec2f(2.0 / 3.0), up);
    let cp = vec2f(cen.x + cen.y * 0.5, cen.y * 0.8660254);
    let bary = select(vec3f(f, 1.0 - f.x - f.y), vec3f(1.0 - f, f.x + f.y - 1.0), up);
    let edge = min(bary.x, min(bary.y, bary.z)) * 0.8660254;
    return TriCell(p - cp, vec2f(cell.x * 2.0 + select(0.0, 1.0, up), cell.y), edge);
}
`,
})

export const radialTile = libFn({
    name: 'radialTile',
    family: 'space',
    signature: 'radialTile(p: vec2f, sectors: f32, rings: f32) -> RadialCell { local: vec2f /* 0..1 across the sector, 0..1 outward */, id: vec2f /* (sector, ring) */, folded: vec2f /* p turned so its sector faces +x */ }',
    doc: 'Cut the plane around the origin into pie sectors and concentric rings (rings per unit of radius). local is the position inside the cell, id its (sector, ring) index, and folded is p rotated so every sector looks the same: draw in folded for kaleidoscopes and radial repeats.',
    code: `
struct RadialCell { local: vec2f, id: vec2f, folded: vec2f }
fn radialTile(p: vec2f, sectors: f32, rings: f32) -> RadialCell {
    let n = max(floor(sectors), 1.0);
    let a = fract(atan2(p.y, p.x) * 0.15915494) * n;
    let si = min(floor(a), n - 1.0);
    let rr = length(p) * rings;
    let ri = floor(rr);
    let mid = (si + 0.5) * 6.28318531 / n;
    let c = cos(mid);
    let s = sin(mid);
    let folded = vec2f(c * p.x + s * p.y, c * p.y - s * p.x);
    return RadialCell(vec2f(a - si, rr - ri), vec2f(si, ri), folded);
}
`,
})

export const voronoiCell = libFn({
    name: 'voronoiCell',
    family: 'space',
    signature: 'voronoiCell(p: vec2f, seed: f32) -> VoronoiCell { id: vec2f, center: vec2f, edgeDist: f32 /* exact distance to the cell border */, dist: f32 /* distance to the centre */ }',
    doc: 'The Voronoi cell (about one random point per unit square) that p falls in: its id, its centre point, the exact distance to its border (for crisp, even-width cracks and mosaics), and the distance to its centre. seed picks a different layout.',
    deps: [hash33],
    code: `
struct VoronoiCell { id: vec2f, center: vec2f, edgeDist: f32, dist: f32 }
fn voronoiCell(p: vec2f, seed: f32) -> VoronoiCell {
    let ip = floor(p);
    let fp = p - ip;
    var mg = vec2f(0.0);
    var mr = vec2f(0.0);
    var md = 8.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let g = vec2f(f32(i), f32(j));
            let r = g + hash33(vec3f(ip + g, seed)).xy - fp;
            let dd = dot(r, r);
            if (dd < md) {
                md = dd;
                mr = r;
                mg = g;
            }
        }
    }
    var ed = 8.0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let g = mg + vec2f(f32(i), f32(j));
            let r = g + hash33(vec3f(ip + g, seed)).xy - fp;
            let dr = r - mr;
            if (dot(dr, dr) > 1e-5) {
                ed = min(ed, dot(0.5 * (mr + r), normalize(dr)));
            }
        }
    }
    return VoronoiCell(ip + mg, p + mr, ed, sqrt(md));
}
`,
})
