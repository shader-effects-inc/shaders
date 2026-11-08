/**
 * Hashes and noise. Hashes are integer (PCG / lookup3) and work on the exact bits of their input,
 * so they stay random far from the origin, unlike the `fract(sin(x) * 43758.5)` kind.
 *
 * Ranges, unified: hashes return 0..1 (1 excluded). Gradient noise (`noise2`, `noise3`, `fbm2`,
 * `fbm3`, the `.value` of `noise2d` / `noise3d`) returns about -1..1, centred on 0.
 */
// Maintainer notes. noise2/noise3 are the MaterialX `mx_perlin_noise_float` (Bob Jenkins lookup3
// lattice hash, quintic fade, 0.6616 / 0.9820 scale), the same field the kit's mxNoiseFloat2/3
// draw, so a library-built noise matches the built-in Noise components. noise2d/noise3d are that
// exact field with analytic derivatives (value matches noise2/noise3 bit for bit in exact
// arithmetic). blueDither is the kit's hilbertBlue12.
import {libFn} from '../wgslLibrary'

// ── Hashes ────────────────────────────────────────────────────────────────────────────────

export const hash11 = libFn({
    name: 'hash11',
    family: 'noise',
    signature: 'hash11(p: f32) -> f32  // 0..1',
    doc: 'A random number from one value, the same every frame for the same input. Use whole numbers (a floor()-ed cell or an index) for one value per cell.',
    code: `
fn _hash11_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash11(p: f32) -> f32 {
    let bits = bitcast<u32>(select(p, 0.0, p == 0.0));
    return f32(_hash11_pcg(bits) >> 8u) * (1.0 / 16777216.0);
}
`,
})

export const hash21 = libFn({
    name: 'hash21',
    family: 'noise',
    signature: 'hash21(p: vec2f) -> f32  // 0..1',
    doc: 'A random number from a 2D point, e.g. one value per grid cell with hash21(floor(uv * 10.0)).',
    code: `
fn _hash21_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash21(p: vec2f) -> f32 {
    let bits = bitcast<vec2u>(select(p, vec2f(0.0), p == vec2f(0.0)));
    return f32(_hash21_pcg(bits.x + _hash21_pcg(bits.y)) >> 8u) * (1.0 / 16777216.0);
}
`,
})

export const hash22 = libFn({
    name: 'hash22',
    family: 'noise',
    signature: 'hash22(p: vec2f) -> vec2f  // each 0..1',
    doc: 'Two independent random numbers from a 2D point, e.g. a random offset per cell.',
    code: `
fn _hash22_pcg2d(u: vec2u) -> vec2u {
    var v = u * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    return v;
}
fn hash22(p: vec2f) -> vec2f {
    let bits = bitcast<vec2u>(select(p, vec2f(0.0), p == vec2f(0.0)));
    return vec2f(_hash22_pcg2d(bits) >> vec2u(8u)) * (1.0 / 16777216.0);
}
`,
})

export const hash33 = libFn({
    name: 'hash33',
    family: 'noise',
    signature: 'hash33(p: vec3f) -> vec3f  // each 0..1',
    doc: 'Three independent random numbers from a 3D point, e.g. a random colour per cell, or a per-cell value that also depends on a seed in z.',
    code: `
fn _hash33_pcg3d(u: vec3u) -> vec3u {
    var v = u * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3u(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}
fn hash33(p: vec3f) -> vec3f {
    let bits = bitcast<vec3u>(select(p, vec3f(0.0), p == vec3f(0.0)));
    return vec3f(_hash33_pcg3d(bits) >> vec3u(8u)) * (1.0 / 16777216.0);
}
`,
})

export const latticeHash2 = libFn({
    name: 'latticeHash2',
    family: 'noise',
    signature: 'latticeHash2(cell: vec2i) -> u32  // full 32-bit hash',
    doc: 'A 32-bit hash of an integer grid cell (Bob Jenkins lookup3, as MaterialX uses). Take bits from it for several random values per cell: f32(h & 0xffffu) / 65536.0.',
    code: `
fn _latticeHash2_rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}
fn _latticeHash2_final(a0: u32, b0: u32, c0: u32) -> u32 {
    var a = a0;
    var b = b0;
    var c = c0;
    c = c ^ b;
    c = c - _latticeHash2_rotl(b, 14u);
    a = a ^ c;
    a = a - _latticeHash2_rotl(c, 11u);
    b = b ^ a;
    b = b - _latticeHash2_rotl(a, 25u);
    c = c ^ b;
    c = c - _latticeHash2_rotl(b, 16u);
    a = a ^ c;
    a = a - _latticeHash2_rotl(c, 4u);
    b = b ^ a;
    b = b - _latticeHash2_rotl(a, 14u);
    c = c ^ b;
    c = c - _latticeHash2_rotl(b, 24u);
    return c;
}
fn latticeHash2(cell: vec2i) -> u32 {
    let seed = 0xdeadbeefu + (2u << 2u) + 13u;
    return _latticeHash2_final(seed + bitcast<u32>(cell.x), seed + bitcast<u32>(cell.y), seed);
}
`,
})

export const latticeHash3 = libFn({
    name: 'latticeHash3',
    family: 'noise',
    signature: 'latticeHash3(cell: vec3i) -> u32  // full 32-bit hash',
    doc: 'A 32-bit hash of an integer 3D grid cell (Bob Jenkins lookup3, as MaterialX uses).',
    code: `
fn _latticeHash3_rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}
fn _latticeHash3_final(a0: u32, b0: u32, c0: u32) -> u32 {
    var a = a0;
    var b = b0;
    var c = c0;
    c = c ^ b;
    c = c - _latticeHash3_rotl(b, 14u);
    a = a ^ c;
    a = a - _latticeHash3_rotl(c, 11u);
    b = b ^ a;
    b = b - _latticeHash3_rotl(a, 25u);
    c = c ^ b;
    c = c - _latticeHash3_rotl(b, 16u);
    a = a ^ c;
    a = a - _latticeHash3_rotl(c, 4u);
    b = b ^ a;
    b = b - _latticeHash3_rotl(a, 14u);
    c = c ^ b;
    c = c - _latticeHash3_rotl(b, 24u);
    return c;
}
fn latticeHash3(cell: vec3i) -> u32 {
    let seed = 0xdeadbeefu + (3u << 2u) + 13u;
    return _latticeHash3_final(seed + bitcast<u32>(cell.x), seed + bitcast<u32>(cell.y), seed + bitcast<u32>(cell.z));
}
`,
})

// ── Gradient noise ────────────────────────────────────────────────────────────────────────

export const noise2 = libFn({
    name: 'noise2',
    family: 'noise',
    signature: 'noise2(p: vec2f) -> f32  // about -1..1, mean 0',
    doc: 'Smooth Perlin gradient noise with one feature per unit of p (the MaterialX field the Noise components use). Scale p for size; add time to p to drift it.',
    deps: [latticeHash2],
    code: `
fn _noise2_grad(h: u32, f: vec2f) -> f32 {
    let hh = h & 7u;
    let u = select(f.y, f.x, hh < 4u);
    let v = 2.0 * select(f.x, f.y, hh < 4u);
    return select(u, -u, (hh & 1u) != 0u) + select(v, -v, (hh & 2u) != 0u);
}
fn noise2(p: vec2f) -> f32 {
    let fl = floor(p);
    let i = vec2i(fl);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let va = _noise2_grad(latticeHash2(i), f);
    let vb = _noise2_grad(latticeHash2(i + vec2i(1, 0)), f - vec2f(1.0, 0.0));
    let vc = _noise2_grad(latticeHash2(i + vec2i(0, 1)), f - vec2f(0.0, 1.0));
    let vd = _noise2_grad(latticeHash2(i + vec2i(1, 1)), f - vec2f(1.0, 1.0));
    return 0.6616 * mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}
`,
})

export const noise2d = libFn({
    name: 'noise2d',
    family: 'noise',
    signature: 'noise2d(p: vec2f) -> NoiseD { value: f32 /* about -1..1 */, grad: vec2f /* d value / d p */ }',
    doc: 'noise2 plus its exact slope, computed together for little more than the cost of one sample. Use grad for lighting a noise height, warping along the noise, or flow (see curl2).',
    deps: [latticeHash2],
    code: `
struct NoiseD { value: f32, grad: vec2f }
fn _noise2d_g(h: u32) -> vec2f {
    let hh = h & 7u;
    let s1 = select(1.0, -1.0, (hh & 1u) != 0u);
    let s2 = select(2.0, -2.0, (hh & 2u) != 0u);
    return select(vec2f(s2, s1), vec2f(s1, s2), hh < 4u);
}
fn noise2d(p: vec2f) -> NoiseD {
    let fl = floor(p);
    let i = vec2i(fl);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = _noise2d_g(latticeHash2(i));
    let gb = _noise2d_g(latticeHash2(i + vec2i(1, 0)));
    let gc = _noise2d_g(latticeHash2(i + vec2i(0, 1)));
    let gd = _noise2d_g(latticeHash2(i + vec2i(1, 1)));
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2f(1.0, 0.0));
    let vc = dot(gc, f - vec2f(0.0, 1.0));
    let vd = dot(gd, f - vec2f(1.0, 1.0));
    let k = va - vb - vc + vd;
    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
    let grad = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (vec2f(u.y, u.x) * k + vec2f(vb, vc) - va);
    return NoiseD(0.6616 * value, 0.6616 * grad);
}
`,
})

export const noise3 = libFn({
    name: 'noise3',
    family: 'noise',
    signature: 'noise3(p: vec3f) -> f32  // about -1..1, mean 0',
    doc: 'Smooth 3D Perlin gradient noise (MaterialX). Pass vec3f(uv * scale, time * speed) for a 2D pattern that morphs in place instead of sliding.',
    deps: [latticeHash3],
    code: `
fn _noise3_grad(h: u32, f: vec3f) -> f32 {
    let hh = h & 15u;
    let u = select(f.y, f.x, hh < 8u);
    let v = select(select(f.z, f.x, hh == 12u || hh == 14u), f.y, hh < 4u);
    return select(u, -u, (hh & 1u) != 0u) + select(v, -v, (hh & 2u) != 0u);
}
fn noise3(p: vec3f) -> f32 {
    let fl = floor(p);
    let i = vec3i(fl);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let v0 = _noise3_grad(latticeHash3(i), f);
    let v1 = _noise3_grad(latticeHash3(i + vec3i(1, 0, 0)), f - vec3f(1.0, 0.0, 0.0));
    let v2 = _noise3_grad(latticeHash3(i + vec3i(0, 1, 0)), f - vec3f(0.0, 1.0, 0.0));
    let v3 = _noise3_grad(latticeHash3(i + vec3i(1, 1, 0)), f - vec3f(1.0, 1.0, 0.0));
    let v4 = _noise3_grad(latticeHash3(i + vec3i(0, 0, 1)), f - vec3f(0.0, 0.0, 1.0));
    let v5 = _noise3_grad(latticeHash3(i + vec3i(1, 0, 1)), f - vec3f(1.0, 0.0, 1.0));
    let v6 = _noise3_grad(latticeHash3(i + vec3i(0, 1, 1)), f - vec3f(0.0, 1.0, 1.0));
    let v7 = _noise3_grad(latticeHash3(i + vec3i(1, 1, 1)), f - vec3f(1.0, 1.0, 1.0));
    let z0 = mix(mix(v0, v1, u.x), mix(v2, v3, u.x), u.y);
    let z1 = mix(mix(v4, v5, u.x), mix(v6, v7, u.x), u.y);
    return 0.982 * mix(z0, z1, u.z);
}
`,
})

export const noise3d = libFn({
    name: 'noise3d',
    family: 'noise',
    signature: 'noise3d(p: vec3f) -> NoiseD3 { value: f32 /* about -1..1 */, grad: vec3f /* d value / d p */ }',
    doc: 'noise3 plus its exact 3D slope. The building block for curl3, and for lighting or warping a 3D noise.',
    deps: [latticeHash3],
    code: `
struct NoiseD3 { value: f32, grad: vec3f }
fn _noise3d_g(h: u32) -> vec3f {
    let hh = h & 15u;
    let eu = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), hh < 8u);
    let ez = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), hh == 12u || hh == 14u);
    let ev = select(ez, vec3f(0.0, 1.0, 0.0), hh < 4u);
    return eu * select(1.0, -1.0, (hh & 1u) != 0u) + ev * select(1.0, -1.0, (hh & 2u) != 0u);
}
fn noise3d(p: vec3f) -> NoiseD3 {
    let fl = floor(p);
    let i = vec3i(fl);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    var value = 0.0;
    var grad = vec3f(0.0);
    for (var k = 0u; k < 8u; k++) {
        let o = vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
        let g = _noise3d_g(latticeHash3(i + vec3i(o)));
        let n = dot(g, f - o);
        let w3 = mix(1.0 - u, u, o);
        let dw3 = du * (2.0 * o - 1.0);
        let w = w3.x * w3.y * w3.z;
        value += w * n;
        grad += w * g + n * vec3f(dw3.x * w3.y * w3.z, w3.x * dw3.y * w3.z, w3.x * w3.y * dw3.z);
    }
    return NoiseD3(0.982 * value, 0.982 * grad);
}
`,
})

export const fbm2 = libFn({
    name: 'fbm2',
    family: 'noise',
    signature: 'fbm2(p: vec2f, octaves: i32) -> f32  // about -1..1 (most values within -0.6..0.6); octaves 1..12',
    doc: 'Fractal noise: octaves of noise2, each twice as fine and half as strong, rotated so no grid shows. More octaves add finer detail (4 to 6 is typical).',
    deps: [noise2],
    code: `
fn fbm2(p: vec2f, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var q = p;
    let count = clamp(octaves, 1, 12);
    for (var i = 0; i < count; i++) {
        sum += amp * noise2(q);
        norm += amp;
        amp *= 0.5;
        q = vec2f(1.6 * q.x - 1.2 * q.y, 1.2 * q.x + 1.6 * q.y) + vec2f(5.31, 1.73);
    }
    return sum / norm;
}
`,
})

export const fbm3 = libFn({
    name: 'fbm3',
    family: 'noise',
    signature: 'fbm3(p: vec3f, octaves: i32) -> f32  // about -1..1 (most values within -0.6..0.6); octaves 1..12',
    doc: 'Fractal 3D noise: octaves of noise3, each twice as fine and half as strong, rotated between octaves. Use z = time for clouds and smoke that evolve in place.',
    deps: [noise3],
    code: `
fn fbm3(p: vec3f, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var q = p;
    let count = clamp(octaves, 1, 12);
    let rot = mat3x3f(0.0, 0.8, 0.6, -0.8, 0.36, -0.48, -0.6, -0.48, 0.64);
    for (var i = 0; i < count; i++) {
        sum += amp * noise3(q);
        norm += amp;
        amp *= 0.5;
        q = rot * q * 2.0 + vec3f(5.31, 1.73, 3.17);
    }
    return sum / norm;
}
`,
})

// ── Cellular ──────────────────────────────────────────────────────────────────────────────

export const worley2 = libFn({
    name: 'worley2',
    family: 'noise',
    signature: 'worley2(p: vec2f) -> Worley { f1: f32 /* 0..~1.1 */, f2: f32 /* >= f1 */, id: vec2f /* nearest cell */ }',
    doc: 'Cellular (Worley) noise: distance to the nearest random feature point (f1), to the second nearest (f2), and the integer cell of the nearest. f2 - f1 is 0 on the cell borders; hash21(id) gives each cell its own random value.',
    deps: [latticeHash2],
    code: `
struct Worley { f1: f32, f2: f32, id: vec2f }
fn worley2(p: vec2f) -> Worley {
    let fl = floor(p);
    let ip = vec2i(fl);
    let fp = p - fl;
    var d1 = 8.0;
    var d2 = 8.0;
    var near = vec2f(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let g = vec2i(i, j);
            let h = latticeHash2(ip + g);
            let o = vec2f(f32(h & 0xffffu), f32(h >> 16u)) * (1.0 / 65536.0);
            let r = vec2f(g) + o - fp;
            let dd = dot(r, r);
            if (dd < d1) {
                d2 = d1;
                d1 = dd;
                near = fl + vec2f(g);
            } else if (dd < d2) {
                d2 = dd;
            }
        }
    }
    return Worley(sqrt(d1), sqrt(d2), near);
}
`,
})

// ── Flow ──────────────────────────────────────────────────────────────────────────────────

export const curl2 = libFn({
    name: 'curl2',
    family: 'noise',
    signature: 'curl2(p: vec2f) -> vec2f  // unnormalised, length typically 0..3',
    doc: 'A swirling, divergence-free flow direction from noise2: things moved along it circulate without bunching up or thinning out. Use it to advect particles or warp uv; normalize() it if you only want the direction.',
    deps: [noise2d],
    code: `
fn curl2(p: vec2f) -> vec2f {
    let n = noise2d(p);
    return vec2f(n.grad.y, -n.grad.x);
}
`,
})

export const curl3 = libFn({
    name: 'curl3',
    family: 'noise',
    signature: 'curl3(p: vec3f) -> vec3f  // unnormalised, length typically 0..4',
    doc: 'A divergence-free 3D flow from three noise3 fields (Bridson curl noise). For a 2D flow that evolves over time, use curl3(vec3f(uv, time)).xy.',
    deps: [noise3d],
    code: `
fn curl3(p: vec3f) -> vec3f {
    let a = noise3d(p).grad;
    let b = noise3d(p + vec3f(31.416, -47.853, 12.793)).grad;
    let c = noise3d(p + vec3f(-233.145, -113.408, -185.31)).grad;
    return vec3f(c.y - b.z, a.z - c.x, b.x - a.y);
}
`,
})

// ── Dither ────────────────────────────────────────────────────────────────────────────────

export const blueDither = libFn({
    name: 'blueDither',
    family: 'noise',
    signature: 'blueDither(fragCoord: vec2f) -> f32  // 0..1, evenly spread',
    doc: 'Blue-noise-like dither threshold per pixel (a Hilbert curve walked with the golden ratio): neighbouring pixels always differ, so banding breaks up without visible grain. Pass the pixel position; tiles every 512 pixels.',
    code: `
fn _blueDither_hilbert(p: vec2i) -> i32 {
    var idx = 0;
    var px = p.x;
    var py = p.y;
    for (var s = 256; s > 0; s = s >> 1u) {
        let rx = select(0, 1, (px & s) > 0);
        let ry = select(0, 1, (py & s) > 0);
        idx += s * s * ((rx << 1u) | (rx ^ ry));
        let flipY = 1 - ry;
        let swap = ((px ^ py) * flipY) ^ ((s - 1) * (rx & flipY));
        px = px ^ swap;
        py = py ^ swap;
    }
    return idx;
}
fn blueDither(fragCoord: vec2f) -> f32 {
    let idx = bitcast<u32>(_blueDither_hilbert(vec2i(floor(fragCoord))) & 262143);
    // fract(idx * golden ratio) in exact integer arithmetic (2654435769 = 2^32 / phi)
    return f32((idx * 2654435769u) >> 8u) * (1.0 / 16777216.0);
}
`,
})
