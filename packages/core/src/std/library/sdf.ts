/**
 * 2D signed distance functions: negative inside, positive outside, zero on the outline, in the
 * units of p. Point first, shape centred on the origin (move it with p - centre, turn it with
 * rotate2). Fill with aaFill(d), outline with aaLine(d, width). "Exact" means the true Euclidean
 * distance; "bound" means it never overestimates (safe for marching, offsets are approximate).
 */
// Maintainer notes. Formulas are Inigo Quilez's (iquilezles.org/articles/distfunctions2d), with
// GLSL mod() written as x - y * floor(x / y) (WGSL % truncates) and atan(y, x) as atan2.
import {libFn} from '../wgslLibrary'

export const sdCircle = libFn({
    name: 'sdCircle',
    family: 'sdf',
    signature: 'sdCircle(p: vec2f, r: f32) -> f32  // exact',
    doc: 'Signed distance to a circle of radius r.',
    code: `
fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}
`,
})

export const sdBox = libFn({
    name: 'sdBox',
    family: 'sdf',
    signature: 'sdBox(p: vec2f, b: vec2f) -> f32  // exact; b = half width, half height',
    doc: 'Signed distance to an axis-aligned rectangle whose half-size is b (so a 2 x 1 box is b = vec2f(1.0, 0.5)).',
    code: `
fn sdBox(p: vec2f, b: vec2f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}
`,
})

export const sdRoundBox = libFn({
    name: 'sdRoundBox',
    family: 'sdf',
    signature: 'sdRoundBox(p: vec2f, b: vec2f, r: f32) -> f32  // exact; b = half size, r = corner radius',
    doc: 'Signed distance to a rectangle of half-size b with corners rounded by r. The outer size stays b; r is clamped to fit.',
    code: `
fn sdRoundBox(p: vec2f, b: vec2f, r: f32) -> f32 {
    let rr = clamp(r, 0.0, min(b.x, b.y));
    let q = abs(p) - b + rr;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}
`,
})

export const sdSegment = libFn({
    name: 'sdSegment',
    family: 'sdf',
    signature: 'sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32  // exact, unsigned (>= 0)',
    doc: 'Distance to the line segment from a to b. Draw a line with aaLine(sdSegment(p, a, b), width).',
    code: `
fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
    return length(pa - ba * h);
}
`,
})

export const sdCapsule = libFn({
    name: 'sdCapsule',
    family: 'sdf',
    signature: 'sdCapsule(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32  // exact',
    doc: 'Signed distance to a capsule: the segment a to b thickened by radius r, with round ends.',
    deps: [sdSegment],
    code: `
fn sdCapsule(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 {
    return sdSegment(p, a, b) - r;
}
`,
})

export const sdPolygon = libFn({
    name: 'sdPolygon',
    family: 'sdf',
    signature: 'sdPolygon(p: vec2f, r: f32, n: f32) -> f32  // exact; r = centre-to-corner radius, n sides (>= 3, whole)',
    doc: 'Signed distance to a regular polygon with n sides (3 triangle, 6 hexagon, ...), with a corner pointing up (+y).',
    code: `
fn sdPolygon(p: vec2f, r: f32, n: f32) -> f32 {
    let an = 3.14159265 / max(floor(n), 3.0);
    let acs = vec2f(cos(an), sin(an));
    let ang = atan2(p.x, p.y);
    let bn = ang - 2.0 * an * floor(ang / (2.0 * an)) - an;
    var q = length(p) * vec2f(cos(bn), abs(sin(bn)));
    q -= r * acs;
    q.y += clamp(-q.y, 0.0, r * acs.y);
    return length(q) * sign(q.x);
}
`,
})

export const sdStar = libFn({
    name: 'sdStar',
    family: 'sdf',
    signature: 'sdStar(p: vec2f, r: f32, n: f32, m: f32) -> f32  // exact; r = tip radius, n points, m 2 (polygon) .. n (sharpest)',
    doc: 'Signed distance to an n-pointed star with a point up (+y). m sets the sharpness: 2 is a plain polygon, larger is spikier, n gives spikes that reach the centre.',
    code: `
fn sdStar(p: vec2f, r: f32, n: f32, m: f32) -> f32 {
    let pts = max(floor(n), 2.0);
    let an = 3.14159265 / pts;
    let en = 3.14159265 / clamp(m, 2.0, pts);
    let acs = vec2f(cos(an), sin(an));
    let ecs = vec2f(cos(en), sin(en));
    let ang = atan2(p.x, p.y);
    let bn = ang - 2.0 * an * floor(ang / (2.0 * an)) - an;
    var q = length(p) * vec2f(cos(bn), abs(sin(bn)));
    q -= r * acs;
    q += ecs * clamp(-dot(q, ecs), 0.0, r * acs.y / ecs.y);
    return length(q) * sign(q.x);
}
`,
})

export const sdStarPolygon = libFn({
    name: 'sdStarPolygon',
    family: 'sdf',
    signature: 'sdStarPolygon(p: vec2f, r: f32, n: f32, skip: f32) -> f32  // exact; n points, skip 1 .. (n-1)/2 (whole)',
    doc: 'Signed distance to the outline of the star polygon {n/skip}: n corners, each joined to the corner skip steps away. {5/2} is the pentagram, {6/2} the hexagram outline, skip 1 a plain polygon.',
    deps: [sdStar],
    code: `
fn sdStarPolygon(p: vec2f, r: f32, n: f32, skip: f32) -> f32 {
    let pts = max(floor(n), 3.0);
    let k = clamp(floor(skip), 1.0, max(floor((pts - 1.0) * 0.5), 1.0));
    let en = 1.57079633 - 3.14159265 * (k - 1.0) / pts;
    return sdStar(p, r, pts, 3.14159265 / en);
}
`,
})

export const sdGear = libFn({
    name: 'sdGear',
    family: 'sdf',
    signature: 'sdGear(p: vec2f, r: f32, teeth: f32, toothDepth: f32) -> f32  // approximate; r = mid radius, teeth whole',
    doc: 'Signed distance (approximate, accurate near the outline) to a gear: a disc of radius r with flat-topped teeth reaching toothDepth / 2 above and below it. Good for fills, outlines and anti-aliasing; offsets far from the outline are only roughly even.',
    code: `
fn sdGear(p: vec2f, r: f32, teeth: f32, toothDepth: f32) -> f32 {
    let n = max(floor(teeth), 1.0);
    let rad = length(p);
    let profile = clamp(2.0 * cos(n * atan2(p.y, p.x)), -1.0, 1.0) * 0.5;
    let edgeR = r + toothDepth * profile;
    let slope = n * toothDepth / max(rad, max(r - toothDepth * 0.5, 1e-4));
    return (rad - edgeR) / sqrt(1.0 + slope * slope);
}
`,
})

export const sdArc = libFn({
    name: 'sdArc',
    family: 'sdf',
    signature: 'sdArc(p: vec2f, sc: vec2f, ra: f32, rb: f32) -> f32  // exact; sc = vec2f(sin(a), cos(a)) of the half-opening a',
    doc: 'Signed distance to an arc of radius ra and thickness 2 * rb with round caps, centred on +y and opening a radians to each side (sc = vec2f(sin(a), cos(a)); a = pi gives a full ring).',
    code: `
fn sdArc(p: vec2f, sc: vec2f, ra: f32, rb: f32) -> f32 {
    let q = vec2f(abs(p.x), p.y);
    let capDist = length(q - sc * ra);
    let ringDist = abs(length(q) - ra);
    return select(ringDist, capDist, sc.y * q.x > sc.x * q.y) - rb;
}
`,
})

export const sdBezier = libFn({
    name: 'sdBezier',
    family: 'sdf',
    signature: 'sdBezier(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> f32  // exact, unsigned (>= 0)',
    doc: 'Distance to a quadratic Bezier curve from a to c, pulled toward the control point b. Draw it with aaLine; chain several for paths.',
    code: `
fn sdBezier(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> f32 {
    let e1 = b - a;
    let e2 = a - 2.0 * b + c;
    let bb = dot(e2, e2);
    if (bb < 1e-10) {
        let pa = p - a;
        let ca = c - a;
        let h = clamp(dot(pa, ca) / max(dot(ca, ca), 1e-12), 0.0, 1.0);
        return length(pa - ca * h);
    }
    let e3 = e1 * 2.0;
    let dp = a - p;
    let kk = 1.0 / bb;
    let kx = kk * dot(e1, e2);
    let ky = kk * (2.0 * dot(e1, e1) + dot(dp, e2)) / 3.0;
    let kz = kk * dot(dp, e1);
    let pp = ky - kx * kx;
    let p3 = pp * pp * pp;
    let qq = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    var h = qq * qq + 4.0 * p3;
    var res = 0.0;
    if (h >= 0.0) {
        h = sqrt(h);
        let x = (vec2f(h, -h) - qq) * 0.5;
        let uv = sign(x) * pow(max(abs(x), vec2f(1e-20)), vec2f(1.0 / 3.0));
        let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        let v = dp + (e3 + e2 * t) * t;
        res = dot(v, v);
    } else {
        let z = sqrt(-pp);
        let ang = acos(clamp(qq / (pp * z * 2.0), -1.0, 1.0)) / 3.0;
        let m = cos(ang);
        let n = sin(ang) * 1.732050808;
        let t = clamp(vec2f(m + m, -n - m) * z - kx, vec2f(0.0), vec2f(1.0));
        let v1 = dp + (e3 + e2 * t.x) * t.x;
        let v2 = dp + (e3 + e2 * t.y) * t.y;
        res = min(dot(v1, v1), dot(v2, v2));
    }
    return sqrt(res);
}
`,
})

export const smoothUnion = libFn({
    name: 'smoothUnion',
    family: 'sdf',
    signature: 'smoothUnion(a: f32, b: f32, k: f32) -> f32  // bound; k = blend radius (0 = sharp)',
    doc: 'Merge two shapes (min of their distances) with a smooth fillet of about k where they meet, like blobs of liquid joining.',
    code: `
fn smoothUnion(a: f32, b: f32, k: f32) -> f32 {
    let kk = max(k, 1e-6);
    let h = max(kk - abs(a - b), 0.0) / kk;
    return min(a, b) - h * h * kk * 0.25;
}
`,
})

export const smoothSubtract = libFn({
    name: 'smoothSubtract',
    family: 'sdf',
    signature: 'smoothSubtract(a: f32, b: f32, k: f32) -> f32  // bound; a minus b, k = blend radius',
    doc: 'Cut shape b out of shape a, rounding the cut edge by about k.',
    code: `
fn smoothSubtract(a: f32, b: f32, k: f32) -> f32 {
    let kk = max(k, 1e-6);
    let h = max(kk - abs(a + b), 0.0) / kk;
    return max(a, -b) + h * h * kk * 0.25;
}
`,
})

export const smoothIntersect = libFn({
    name: 'smoothIntersect',
    family: 'sdf',
    signature: 'smoothIntersect(a: f32, b: f32, k: f32) -> f32  // bound; k = blend radius',
    doc: 'Keep only where shapes a and b overlap, rounding the meeting edges by about k.',
    code: `
fn smoothIntersect(a: f32, b: f32, k: f32) -> f32 {
    let kk = max(k, 1e-6);
    let h = max(kk - abs(a - b), 0.0) / kk;
    return max(a, b) + h * h * kk * 0.25;
}
`,
})

export const onion = libFn({
    name: 'onion',
    family: 'sdf',
    signature: 'onion(d: f32, thickness: f32) -> f32  // exact if d is',
    doc: 'Turn any shape into a hollow outline shape of total width thickness, centred on the original edge. Apply twice for concentric rims.',
    code: `
fn onion(d: f32, thickness: f32) -> f32 {
    return abs(d) - thickness * 0.5;
}
`,
})
