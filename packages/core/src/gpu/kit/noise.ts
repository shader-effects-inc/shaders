/**
 * MaterialX noise (the AcademySoftwareFoundation/MaterialX algorithm): the MaterialX
 * permutation/hash (Bob Jenkins lookup3), gradients, and constants. Backs the 18 noise shaders
 * (16 perlin `mx_noise_float`, 2 worley `mx_worley_noise_float`).
 *
 * PUBLIC surface (what shaders call — see the "Public" section):
 *   - `mxNoiseFloat2(p: vec2f)` / `mxNoiseFloat3(p: vec3f)` — the `mx_noise_float(texcoord)` form
 *     with amplitude=1, pivot=0 (the only form our shaders use). It is exactly
 *     `mx_perlin_noise_float(p)`. If a shader ever needs the amplitude/pivot form, apply
 *     `.mul(amp).add(pivot)` at the call site.
 *   - `mxWorleyNoiseFloat2(p: vec2f, jitter: f32)` / `mxWorleyNoiseFloat3(p: vec3f, jitter)`
 *     — the `mx_worley_noise_float(texcoord, jitter)` form, which fixes `metric = 1` (squared
 *     euclidean F1). Our shaders only ever use the 2D form.
 *
 * Call forms: `mx_noise_float(vec2(...))` → `mxNoiseFloat2(...)`,
 * `mx_noise_float(vec3(...))` → `mxNoiseFloat3(...)`,
 * `mx_worley_noise_float(vec2(...), j)` → `mxWorleyNoiseFloat2(..., j)`.
 *
 * Equivalence testing note: the integer hash is u32 arithmetic with WGSL wrapping/logical
 * semantics. Shifts use `>>>` (typegpu deprecated `>>` on u32); its CPU dual still runs as JS, so
 * CPU-executing the hash would NOT match the GPU for values with the high bit set. The hash
 * is therefore verified by the resolve gate (transpiles to correct u32 WGSL), and the pure-float
 * helpers (fade, bilerp, trilerp, scale) are verified by CPU golden values.
 */
import {tgpu, d, std} from './index'

// ─── low-level integer hash (Bob Jenkins lookup3) ────────────────────────────

/** `mx_rotl32`: rotate-left a u32 by k bits. `(x << k) | (x >>> (32 - k))`. */
export const mxRotl32 = tgpu.fn([d.u32, d.i32], d.u32)((x, k) => {
    'use gpu'
    return (x << d.u32(k)) | (x >>> d.u32(d.i32(32) - k))
})

/** `mx_bjfinal`: final mixing of three u32 accumulators → one u32 hash. */
export const mxBjfinal = tgpu.fn([d.u32, d.u32, d.u32], d.u32)((a0, b0, c0) => {
    'use gpu'
    let a = a0
    let b = b0
    let c = c0
    c = c ^ b
    c = c - mxRotl32(b, d.i32(14))
    a = a ^ c
    a = a - mxRotl32(c, d.i32(11))
    b = b ^ a
    b = b - mxRotl32(a, d.i32(25))
    c = c ^ b
    c = c - mxRotl32(b, d.i32(16))
    a = a ^ c
    a = a - mxRotl32(c, d.i32(4))
    b = b ^ a
    b = b - mxRotl32(a, d.i32(14))
    c = c ^ b
    c = c - mxRotl32(b, d.i32(24))
    return c
})

// Seed base: `uint(int(0xdeadbeef)) + (len << 2) + 13`. 0xdeadbeef is 3735928559 (a valid u32).
const DEADBEEF = 0xdeadbeef

/** `mx_hash_int` (2 ints). */
export const mxHashInt1 = tgpu.fn([d.i32, d.i32], d.u32)((x, y) => {
    'use gpu'
    const len = d.u32(2)
    const seed = d.u32(DEADBEEF) + (len << d.u32(2)) + d.u32(13)
    const a = seed + d.u32(x)
    const b = seed + d.u32(y)
    return mxBjfinal(a, b, seed)
})

/** `mx_hash_int` (3 ints). */
export const mxHashInt2 = tgpu.fn([d.i32, d.i32, d.i32], d.u32)((x, y, z) => {
    'use gpu'
    const len = d.u32(3)
    const seed = d.u32(DEADBEEF) + (len << d.u32(2)) + d.u32(13)
    const a = seed + d.u32(x)
    const b = seed + d.u32(y)
    const c = seed + d.u32(z)
    return mxBjfinal(a, b, c)
})

/** `mx_hash_int` (4 ints) — includes the `mx_bjmix` round (inlined; no by-ref args in TGSL). */
export const mxHashInt3 = tgpu.fn([d.i32, d.i32, d.i32, d.i32], d.u32)((x, y, z, xx) => {
    'use gpu'
    const len = d.u32(4)
    const seed = d.u32(DEADBEEF) + (len << d.u32(2)) + d.u32(13)
    let a = seed + d.u32(x)
    let b = seed + d.u32(y)
    let c = seed + d.u32(z)
    // mx_bjmix(a, b, c) — inlined (no by-ref args in TGSL)
    a = a - c
    a = a ^ mxRotl32(c, d.i32(4))
    c = c + b
    b = b - a
    b = b ^ mxRotl32(a, d.i32(6))
    a = a + c
    c = c - b
    c = c ^ mxRotl32(b, d.i32(8))
    b = b + a
    a = a - c
    a = a ^ mxRotl32(c, d.i32(16))
    c = c + b
    b = b - a
    b = b ^ mxRotl32(a, d.i32(19))
    a = a + c
    c = c - b
    c = c ^ mxRotl32(b, d.i32(4))
    b = b + a
    a = a + d.u32(xx)
    return mxBjfinal(a, b, c)
})

/** `mx_bits_to_01`: map a u32 to [0,1] by dividing by 0xffffffff. */
export const mxBitsTo01 = tgpu.fn([d.u32], d.f32)((bits) => {
    'use gpu'
    return d.f32(bits) / d.f32(4294967295)
})

// ─── perlin gradients + interpolation ────────────────────────────────────────

/** `mx_fade`: quintic smootherstep `t^3 (t (6t - 15) + 10)`. */
export const mxFade = tgpu.fn([d.f32], d.f32)((t) => {
    'use gpu'
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
})

/** `mx_gradient_float` (2D): hashed gradient dot for perlin. */
export const mxGradientFloat2 = tgpu.fn([d.u32, d.f32, d.f32], d.f32)((hash, x, y) => {
    'use gpu'
    const h = hash & d.u32(7)
    const u = std.select(y, x, h < d.u32(4))
    const v = 2.0 * std.select(x, y, h < d.u32(4))
    const uu = std.select(u, -u, (h & d.u32(1)) !== d.u32(0))
    const vv = std.select(v, -v, (h & d.u32(2)) !== d.u32(0))
    return uu + vv
})

/** `mx_gradient_float` (3D): hashed gradient dot for perlin. */
export const mxGradientFloat3 = tgpu.fn([d.u32, d.f32, d.f32, d.f32], d.f32)((hash, x, y, z) => {
    'use gpu'
    const h = hash & d.u32(15)
    const u = std.select(y, x, h < d.u32(8))
    const inner = std.select(z, x, (h === d.u32(12)) || (h === d.u32(14)))
    const v = std.select(inner, y, h < d.u32(4))
    const uu = std.select(u, -u, (h & d.u32(1)) !== d.u32(0))
    const vv = std.select(v, -v, (h & d.u32(2)) !== d.u32(0))
    return uu + vv
})

/** `mx_gradient_scale2d`: 0.6616 · v. */
export const mxGradientScale2d = tgpu.fn([d.f32], d.f32)((v) => {
    'use gpu'
    return 0.6616 * v
})

/** `mx_gradient_scale3d`: 0.9820 · v. */
export const mxGradientScale3d = tgpu.fn([d.f32], d.f32)((v) => {
    'use gpu'
    return 0.982 * v
})

/** `mx_bilerp` (float). */
export const mxBilerp = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((v0, v1, v2, v3, s, t) => {
    'use gpu'
    const s1 = 1.0 - s
    return (1.0 - t) * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s)
})

/** `mx_trilerp` (float). */
export const mxTrilerp = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((v0, v1, v2, v3, v4, v5, v6, v7, s, t, r) => {
    'use gpu'
    const s1 = 1.0 - s
    const t1 = 1.0 - t
    const r1 = 1.0 - r
    return r1 * (t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s))
        + r * (t1 * (v4 * s1 + v5 * s) + t * (v6 * s1 + v7 * s))
})

// ─── perlin noise (float) ────────────────────────────────────────────────────

/** `mx_perlin_noise_float` (2D). */
export const mxPerlinNoiseFloat2 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const X = d.i32(std.floor(p.x))
    const Y = d.i32(std.floor(p.y))
    const fx = p.x - d.f32(X)
    const fy = p.y - d.f32(Y)
    const u = mxFade(fx)
    const v = mxFade(fy)
    const result = mxBilerp(
        mxGradientFloat2(mxHashInt1(X, Y), fx, fy),
        mxGradientFloat2(mxHashInt1(X + d.i32(1), Y), fx - 1.0, fy),
        mxGradientFloat2(mxHashInt1(X, Y + d.i32(1)), fx, fy - 1.0),
        mxGradientFloat2(mxHashInt1(X + d.i32(1), Y + d.i32(1)), fx - 1.0, fy - 1.0),
        u, v,
    )
    return mxGradientScale2d(result)
})

/** `mx_perlin_noise_float` (3D). */
export const mxPerlinNoiseFloat3 = tgpu.fn([d.vec3f], d.f32)((p) => {
    'use gpu'
    const X = d.i32(std.floor(p.x))
    const Y = d.i32(std.floor(p.y))
    const Z = d.i32(std.floor(p.z))
    const fx = p.x - d.f32(X)
    const fy = p.y - d.f32(Y)
    const fz = p.z - d.f32(Z)
    const u = mxFade(fx)
    const v = mxFade(fy)
    const w = mxFade(fz)
    const result = mxTrilerp(
        mxGradientFloat3(mxHashInt2(X, Y, Z), fx, fy, fz),
        mxGradientFloat3(mxHashInt2(X + d.i32(1), Y, Z), fx - 1.0, fy, fz),
        mxGradientFloat3(mxHashInt2(X, Y + d.i32(1), Z), fx, fy - 1.0, fz),
        mxGradientFloat3(mxHashInt2(X + d.i32(1), Y + d.i32(1), Z), fx - 1.0, fy - 1.0, fz),
        mxGradientFloat3(mxHashInt2(X, Y, Z + d.i32(1)), fx, fy, fz - 1.0),
        mxGradientFloat3(mxHashInt2(X + d.i32(1), Y, Z + d.i32(1)), fx - 1.0, fy, fz - 1.0),
        mxGradientFloat3(mxHashInt2(X, Y + d.i32(1), Z + d.i32(1)), fx, fy - 1.0, fz - 1.0),
        mxGradientFloat3(mxHashInt2(X + d.i32(1), Y + d.i32(1), Z + d.i32(1)), fx - 1.0, fy - 1.0, fz - 1.0),
        u, v, w,
    )
    return mxGradientScale3d(result)
})

// ─── cell noise (for worley jitter offsets) ──────────────────────────────────

/** `mx_cell_noise_vec3` (2D input). */
export const mxCellNoiseVec3_2d = tgpu.fn([d.vec2f], d.vec3f)((p) => {
    'use gpu'
    const ix = d.i32(std.floor(p.x))
    const iy = d.i32(std.floor(p.y))
    return d.vec3f(
        mxBitsTo01(mxHashInt2(ix, iy, d.i32(0))),
        mxBitsTo01(mxHashInt2(ix, iy, d.i32(1))),
        mxBitsTo01(mxHashInt2(ix, iy, d.i32(2))),
    )
})

/** `mx_cell_noise_vec3` (3D input). */
export const mxCellNoiseVec3_3d = tgpu.fn([d.vec3f], d.vec3f)((p) => {
    'use gpu'
    const ix = d.i32(std.floor(p.x))
    const iy = d.i32(std.floor(p.y))
    const iz = d.i32(std.floor(p.z))
    return d.vec3f(
        mxBitsTo01(mxHashInt3(ix, iy, iz, d.i32(0))),
        mxBitsTo01(mxHashInt3(ix, iy, iz, d.i32(1))),
        mxBitsTo01(mxHashInt3(ix, iy, iz, d.i32(2))),
    )
})

// ─── worley noise (float) ────────────────────────────────────────────────────

/** `mx_worley_distance` (2D). metric: 2=manhattan, 3=chebyshev, else squared euclidean. */
export const mxWorleyDistance2d = tgpu.fn(
    [d.vec2f, d.i32, d.i32, d.i32, d.i32, d.f32, d.i32],
    d.f32,
)((p, x, y, xoff, yoff, jitter, metric) => {
    'use gpu'
    const tmp = mxCellNoiseVec3_2d(d.vec2f(d.f32(x + xoff), d.f32(y + yoff)))
    let off = d.vec2f(tmp.x, tmp.y)
    off = std.add(std.mul(std.sub(off, 0.5), jitter), 0.5)
    const cellpos = std.add(d.vec2f(d.f32(x), d.f32(y)), off)
    const diff = std.sub(cellpos, p)
    if (metric === d.i32(2)) {
        return std.abs(diff.x) + std.abs(diff.y)
    }
    if (metric === d.i32(3)) {
        return std.max(std.abs(diff.x), std.abs(diff.y))
    }
    return std.dot(diff, diff)
})

/** `mx_worley_distance` (3D). */
export const mxWorleyDistance3d = tgpu.fn(
    [d.vec3f, d.i32, d.i32, d.i32, d.i32, d.i32, d.i32, d.f32, d.i32],
    d.f32,
)((p, x, y, z, xoff, yoff, zoff, jitter, metric) => {
    'use gpu'
    let off = mxCellNoiseVec3_3d(d.vec3f(d.f32(x + xoff), d.f32(y + yoff), d.f32(z + zoff)))
    off = std.add(std.mul(std.sub(off, 0.5), jitter), 0.5)
    const cellpos = std.add(d.vec3f(d.f32(x), d.f32(y), d.f32(z)), off)
    const diff = std.sub(cellpos, p)
    if (metric === d.i32(2)) {
        return std.abs(diff.x) + std.abs(diff.y) + std.abs(diff.z)
    }
    if (metric === d.i32(3)) {
        return std.max(std.max(std.abs(diff.x), std.abs(diff.y)), std.abs(diff.z))
    }
    return std.dot(diff, diff)
})

/** `mx_worley_noise_float` (2D). F1 nearest-cell distance; sqrt only for metric 0 (euclidean). */
export const mxWorleyNoiseFloat2 = tgpu.fn([d.vec2f, d.f32, d.i32], d.f32)((p, jitter, metric) => {
    'use gpu'
    const X = d.i32(std.floor(p.x))
    const Y = d.i32(std.floor(p.y))
    const localpos = d.vec2f(p.x - d.f32(X), p.y - d.f32(Y))
    let sqdist = d.f32(1e6)
    for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
            const dist = mxWorleyDistance2d(localpos, x, y, X, Y, jitter, metric)
            sqdist = std.min(sqdist, dist)
        }
    }
    if (metric === d.i32(0)) {
        sqdist = std.sqrt(sqdist)
    }
    return sqdist
})

/** `mx_worley_noise_float` (3D). */
export const mxWorleyNoiseFloat3 = tgpu.fn([d.vec3f, d.f32, d.i32], d.f32)((p, jitter, metric) => {
    'use gpu'
    const X = d.i32(std.floor(p.x))
    const Y = d.i32(std.floor(p.y))
    const Z = d.i32(std.floor(p.z))
    const localpos = d.vec3f(p.x - d.f32(X), p.y - d.f32(Y), p.z - d.f32(Z))
    let sqdist = d.f32(1e6)
    for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
            for (let z = -1; z <= 1; z++) {
                const dist = mxWorleyDistance3d(localpos, x, y, z, X, Y, Z, jitter, metric)
                sqdist = std.min(sqdist, dist)
            }
        }
    }
    if (metric === d.i32(0)) {
        sqdist = std.sqrt(sqdist)
    }
    return sqdist
})

// ─── Public (`mx_noise_float` / `mx_worley_noise_float` equivalents) ──────

/** `mx_noise_float(p: vec2)` with amplitude=1, pivot=0 → `mx_perlin_noise_float(p)`. */
export const mxNoiseFloat2 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    return mxPerlinNoiseFloat2(p)
})

/** `mx_noise_float(p: vec3)` with amplitude=1, pivot=0 → `mx_perlin_noise_float(p)`. */
export const mxNoiseFloat3 = tgpu.fn([d.vec3f], d.f32)((p) => {
    'use gpu'
    return mxPerlinNoiseFloat3(p)
})

/** `mx_worley_noise_float(p: vec2, jitter)` — metric fixed to 1 (squared euclidean F1). */
export const mxWorleyNoiseFloat2Pub = tgpu.fn([d.vec2f, d.f32], d.f32)((p, jitter) => {
    'use gpu'
    return mxWorleyNoiseFloat2(p, jitter, d.i32(1))
})

/** `mx_worley_noise_float(p: vec3, jitter)` — metric fixed to 1. */
export const mxWorleyNoiseFloat3Pub = tgpu.fn([d.vec3f, d.f32], d.f32)((p, jitter) => {
    'use gpu'
    return mxWorleyNoiseFloat3(p, jitter, d.i32(1))
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// Fi-hash noise family (@lumiey, MIT).
// ═══════════════════════════════════════════════════════════════════════════════════════
//
// These are the @lumiey lattice noises the DESIGN-EDITOR noise textures actually use
// (PerlinNoise → perlin13, BlockNoise → value13, BlueNoise → blue12, plus GaborNoise,
// CurlNoise, WaveletNoise, ErosionNoise, Paper, Scratches, Stone, Wool, LiquidMetal…) — a
// DIFFERENT hash/algorithm from the MaterialX perlin/worley above, so it lives alongside
// (not in place of) it. Each `float(...)` literal keeps its f32 rounding (e.g. `141421356.0`
// resolves to `141421360`, `4294967295.0` to `4294967296f`).
//
// EQUIVALENCE TESTING: the Fi hashes bit-cast float coords to u32 (`floatBitsToUint`). typegpu
// 0.11.9 std exposes only the u32→f32/i32 bitcast direction, so the reverse is provided below as
// raw-WGSL `tgpu.fn`s (the documented `TgpuFnShell` string-implementation form). A raw-WGSL fn has
// NO CPU DualFn, so anything transitively calling a hash is GPU-only — verified by the resolve gate,
// never CPU-golden-tested. The pure-float members (perm4/value13/wavelet12 and the perlin
// fade/interp arithmetic) run on CPU and ARE golden-tested.

// ─── floatBitsToUint (raw-WGSL; typegpu std lacks the f32→u32 bitcast) ─────────

const floatBitsToUint1 = tgpu.fn([d.f32], d.u32)('(x: f32) -> u32 { return bitcast<u32>(x); }')
const floatBitsToUint2 = tgpu.fn([d.vec2f], d.vec2u)('(v: vec2f) -> vec2u { return bitcast<vec2u>(v); }')
const floatBitsToUint3 = tgpu.fn([d.vec3f], d.vec3u)('(v: vec3f) -> vec3u { return bitcast<vec3u>(v); }')

// float(~0u) — the max uint, used to normalise hashes into [0, 1]. `d.f32(4294967295)` resolves
// to `4294967296f` in f32.
const U32 = d.f32(4294967295)

// ─── Fi hashes ─────────────────────────────────────────────────────────────
// floatBitsToUint reinterprets the float bit pattern; uint multiplies wrap mod 2^32 (WGSL u32
// semantics, matching GLSL), so these are integer-exact.

/** hash11 — float → float in [0, 1]. */
export const hash11 = tgpu.fn([d.f32], d.f32)((p) => {
    'use gpu'
    const u = floatBitsToUint1(p * 3141592653.0)
    return d.f32(u * u * d.u32(3141592653)) / U32
})

/** hash12 — vec2 → float in [0, 1]. */
export const hash12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const u = floatBitsToUint2(p.mul(d.vec2f(141421356.0, 2718281828.0)))
    return d.f32((u.x ^ u.y) * d.u32(3141592653)) / U32
})

/** hash22 — vec2 → vec2 in [0, 1]². */
export const hash22 = tgpu.fn([d.vec2f], d.vec2f)((p) => {
    'use gpu'
    const u = floatBitsToUint2(p.mul(d.vec2f(141421356.0, 2718281828.0)))
    const h = u.x ^ u.y
    const prod = std.mul(d.vec2u(h, h), d.vec2u(3141592653, 1618033988))
    return d.vec2f(d.f32(prod.x), d.f32(prod.y)).div(U32)
})

/** hash32 — vec2 → vec3 in [0, 1]³. */
export const hash32 = tgpu.fn([d.vec2f], d.vec3f)((p) => {
    'use gpu'
    const u = floatBitsToUint2(p.mul(d.vec2f(141421356.0, 2718281828.0)))
    const h = u.x ^ u.y
    const prod = std.mul(d.vec3u(h, h, h), d.vec3u(1732050807, 2645751311, 3316624790))
    return d.vec3f(d.f32(prod.x), d.f32(prod.y), d.f32(prod.z)).div(U32)
})

/** hash13 — vec3 → float in [0, 1]. */
export const hash13 = tgpu.fn([d.vec3f], d.f32)((p) => {
    'use gpu'
    const u = floatBitsToUint3(p.mul(d.vec3f(141421356.0, 2718281828.0, 1618033988.0)))
    return d.f32((u.x ^ u.y ^ u.z) * d.u32(3141592653)) / U32
})

/** hash33 — vec3 → vec3 in [0, 1]³. */
export const hash33 = tgpu.fn([d.vec3f], d.vec3f)((p) => {
    'use gpu'
    const u = floatBitsToUint3(p.mul(d.vec3f(141421356.0, 2718281828.0, 1618033988.0)))
    const h = u.x ^ u.y ^ u.z
    const prod = std.mul(d.vec3u(h, h, h), d.vec3u(1732050807, 2645751311, 3316624790))
    return d.vec3f(d.f32(prod.x), d.f32(prod.y), d.f32(prod.z)).div(U32)
})

// ─── 2D value noise ──────────────────────────────────────────────────────────

/** value12 — smoothstep-interpolated value noise, vec2 → [0, 1]. */
export const value12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const i = std.floor(p)
    const f0 = p.sub(i)
    const f = f0.mul(f0).mul(std.sub(3.0, f0.mul(2.0))) // f *= f*(3-2f)
    return std.mix(
        std.mix(hash12(i), hash12(i.add(d.vec2f(1.0, 0.0))), f.x),
        std.mix(hash12(i.add(d.vec2f(0.0, 1.0))), hash12(i.add(d.vec2f(1.0, 1.0))), f.x),
        f.y,
    )
})

// ─── 2D Perlin (gradient) noise ───────────────────────────────────────────────

/** Hashed gradient dot for perlin12: dot(normalize(hash22(i+o) - 0.5), f - o). */
const perlinGrad2 = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32], d.f32)((i, f, ox, oy) => {
    'use gpu'
    const o = d.vec2f(ox, oy)
    return std.dot(std.normalize(hash22(i.add(o)).sub(0.5)), f.sub(o))
})

/** perlin12 — gradient noise, vec2 → roughly [0, 1] (×0.7 + 0.5). */
export const perlin12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const i = std.floor(p)
    const f = p.sub(i)
    const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0))
    const a = perlinGrad2(i, f, 0.0, 0.0)
    const b = perlinGrad2(i, f, 1.0, 0.0)
    const c = perlinGrad2(i, f, 0.0, 1.0)
    const dd = perlinGrad2(i, f, 1.0, 1.0)
    return std.mix(std.mix(a, b, u.x), std.mix(c, dd, u.x), u.y) * 0.7 + 0.5
})

/**
 * perlin12d — Perlin value + analytic gradient, vec2 → vec3 (value, dx, dy).
 * The gradient is used by Stone / Erosion / LiquidMetal / relief to warp / carve the field.
 */
export const perlin12d = tgpu.fn([d.vec2f], d.vec3f)((p) => {
    'use gpu'
    const i = std.floor(p)
    const f = std.fract(p)
    const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0))
    const du = f.mul(f).mul(f.mul(f.sub(2.0)).add(1.0)).mul(30.0)
    const ga = hash22(i.add(d.vec2f(0.0, 0.0))).mul(2.0).sub(1.0)
    const gb = hash22(i.add(d.vec2f(1.0, 0.0))).mul(2.0).sub(1.0)
    const gc = hash22(i.add(d.vec2f(0.0, 1.0))).mul(2.0).sub(1.0)
    const gd = hash22(i.add(d.vec2f(1.0, 1.0))).mul(2.0).sub(1.0)
    const va = std.dot(ga, f.sub(d.vec2f(0.0, 0.0)))
    const vb = std.dot(gb, f.sub(d.vec2f(1.0, 0.0)))
    const vc = std.dot(gc, f.sub(d.vec2f(0.0, 1.0)))
    const vd = std.dot(gd, f.sub(d.vec2f(1.0, 1.0)))
    const k = va - vb - vc + vd
    const value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k
    const deriv = ga
        .add(gb.sub(ga).mul(u.x))
        .add(gc.sub(ga).mul(u.y))
        .add(ga.sub(gb).sub(gc).add(gd).mul(u.x * u.y))
        .add(du.mul(d.vec2f(u.y, u.x).mul(k).add(d.vec2f(vb, vc)).sub(va)))
    return d.vec3f(value, deriv.x, deriv.y)
})

// ─── 3D value & Perlin (for in-place morphing via z = time) ───────────────────

/** perm — vec4 helper for value13 (Quilez-style permutation). Pure float. */
export const perm4 = tgpu.fn([d.vec4f], d.vec4f)((x) => {
    'use gpu'
    const xx = x.mul(x.mul(34.0).add(1.0))
    return xx.sub(std.floor(xx.div(289.0)).mul(289.0))
})

/** value13 — 3D value noise, vec3 → [0, 1]. Pure float (no bitcast). */
export const value13 = tgpu.fn([d.vec3f], d.f32)((p) => {
    'use gpu'
    const a = std.floor(p)
    const d0 = p.sub(a)
    const sd = d0.mul(d0).mul(std.sub(3.0, d0.mul(2.0)))
    const b = d.vec4f(a.x, a.x, a.y, a.y).add(d.vec4f(0.0, 1.0, 0.0, 1.0))
    const k1 = perm4(d.vec4f(b.x, b.y, b.x, b.y))
    const k2 = perm4(d.vec4f(k1.x, k1.y, k1.x, k1.y).add(d.vec4f(b.z, b.z, b.w, b.w))).add(d.vec4f(a.z, a.z, a.z, a.z))
    const k3 = perm4(k2)
    const k4 = perm4(k2.add(1.0))
    const o1 = std.fract(k3.mul(0.02439024))
    const o2 = std.fract(k4.mul(0.02439024))
    const o3 = std.mix(o1, o2, d.vec4f(sd.z))
    const o4 = std.mix(d.vec2f(o3.x, o3.z), d.vec2f(o3.y, o3.w), d.vec2f(sd.x))
    return std.mix(o4.x, o4.y, sd.y)
})

/** Hashed gradient dot for perlin13: dot(f - o, normalize(hash33(i+o) - 0.5)). */
const perlinGrad3 = tgpu.fn([d.vec3f, d.vec3f, d.f32, d.f32, d.f32], d.f32)((i, f, ox, oy, oz) => {
    'use gpu'
    const o = d.vec3f(ox, oy, oz)
    return std.dot(f.sub(o), std.normalize(hash33(i.add(o)).sub(0.5)))
})

/** perlin13 — 3D gradient noise, vec3 → roughly [0, 1]. */
export const perlin13 = tgpu.fn([d.vec3f], d.f32)((p) => {
    'use gpu'
    const i = std.floor(p)
    const f = p.sub(i)
    const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0))
    const a0 = perlinGrad3(i, f, 0.0, 0.0, 0.0)
    const b0 = perlinGrad3(i, f, 1.0, 0.0, 0.0)
    const c0 = perlinGrad3(i, f, 0.0, 1.0, 0.0)
    const d0 = perlinGrad3(i, f, 1.0, 1.0, 0.0)
    const a1 = perlinGrad3(i, f, 0.0, 0.0, 1.0)
    const b1 = perlinGrad3(i, f, 1.0, 0.0, 1.0)
    const c1 = perlinGrad3(i, f, 0.0, 1.0, 1.0)
    const d1 = perlinGrad3(i, f, 1.0, 1.0, 1.0)
    const z0 = std.mix(std.mix(a0, b0, u.x), std.mix(c0, d0, u.x), u.y)
    const z1 = std.mix(std.mix(a1, b1, u.x), std.mix(c1, d1, u.x), u.y)
    return std.mix(z0, z1, u.z) * 0.7 + 0.5
})

// ─── Gabor noise ──────────────────────────────────────────────────────────────

/** One oriented sine grain for gabor12: sin(freq·dot(p, hash22(i+o)) + phase). */
const gaborGrain = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32], d.f32)((p, i, freq, phase, ox, oy) => {
    'use gpu'
    return std.sin(freq * std.dot(p, hash22(i.add(d.vec2f(ox, oy)))) + phase)
})

/**
 * gabor12 — bilinearly-blended oriented sine grains, vec2 → [-1, 1]. `freq` is the in-cell wave
 * frequency (default 8); `phase` shifts the waves along their per-cell orientation.
 */
export const gabor12 = tgpu.fn([d.vec2f, d.f32, d.f32], d.f32)((p, freq, phase) => {
    'use gpu'
    const i = std.floor(p)
    const f0 = p.sub(i)
    const f = f0.mul(f0).mul(std.sub(3.0, f0.mul(2.0)))
    return std.mix(
        std.mix(gaborGrain(p, i, freq, phase, 0.0, 0.0), gaborGrain(p, i, freq, phase, 1.0, 0.0), f.x),
        std.mix(gaborGrain(p, i, freq, phase, 0.0, 1.0), gaborGrain(p, i, freq, phase, 1.0, 1.0), f.x),
        f.y,
    )
})

// ─── Curl noise (from 2D Perlin) ──────────────────────────────────────────────

/** curl22 — divergence-free flow field, vec2 → vec2. */
export const curl22 = tgpu.fn([d.vec2f], d.vec2f)((p) => {
    'use gpu'
    const a = d.vec2f(perlin12(p.add(d.vec2f(0.1, 0.0))), perlin12(p.add(d.vec2f(0.0, 0.1))))
    const b = d.vec2f(perlin12(p.sub(d.vec2f(0.1, 0.0))), perlin12(p.sub(d.vec2f(0.0, 0.1))))
    return a.sub(b).div(0.1).mul(0.5)
})

/**
 * curl22z — curl field that evolves in place along a third (time) axis, vec2 → vec2. Same as
 * curl22 but from 3D Perlin so animating `z` morphs the flow rather than sliding it.
 */
export const curl22z = tgpu.fn([d.vec2f, d.f32], d.vec2f)((p, z) => {
    'use gpu'
    const pax = p.add(d.vec2f(0.1, 0.0))
    const pay = p.add(d.vec2f(0.0, 0.1))
    const pbx = p.sub(d.vec2f(0.1, 0.0))
    const pby = p.sub(d.vec2f(0.0, 0.1))
    const a = d.vec2f(perlin13(d.vec3f(pax.x, pax.y, z)), perlin13(d.vec3f(pay.x, pay.y, z)))
    const b = d.vec2f(perlin13(d.vec3f(pbx.x, pbx.y, z)), perlin13(d.vec3f(pby.x, pby.y, z)))
    return a.sub(b).div(0.1).mul(0.5)
})

// ─── Wavelet noise ────────────────────────────────────────────────────────────

/**
 * wavelet12 — rotating banded wavelets, vec2 → roughly [-1, 1]. `phase` animates the bands;
 * `scale` is the per-octave frequency ratio. Pure float (fract-based hash, no bitcast).
 */
export const wavelet12 = tgpu.fn([d.vec2f, d.f32, d.f32], d.f32)((p, phase, scale) => {
    'use gpu'
    let dsum = d.f32(0)
    let s = d.f32(1)
    let m = d.f32(0)
    let pp = d.vec2f(p.x, p.y)
    for (let idx = 0; idx < 4; idx++) {
        const q0 = pp.mul(s)
        let g = std.fract(std.floor(q0).mul(d.vec2f(123.34, 233.53)))
        g = g.add(std.dot(g, g.add(23.234)))
        const a = std.fract(g.x * g.y) * 1000.0
        const qc = std.fract(q0).sub(0.5)
        const ca = std.cos(a)
        const sa = std.sin(a)
        // qc * mat2(cos a, -sin a, sin a, cos a)  (row-vector × matrix)
        const q = d.vec2f(qc.x * ca - qc.y * sa, qc.x * sa + qc.y * ca)
        dsum = dsum + std.sin(q.x * 10.0 + phase) * std.smoothstep(0.25, 0.0, std.dot(q, q)) / s
        // p * mat2(0.54, -0.84, 0.84, 0.54) + idx
        pp = d.vec2f(pp.x * 0.54 - pp.y * 0.84, pp.x * 0.84 + pp.y * 0.54).add(d.f32(idx))
        m = m + 1.0 / s
        s = s * scale
    }
    return dsum / m
})

// ─── Blue noise (spatial high-pass of white noise) ────────────────────────────

/** blue12 — pixel-space blue noise, vec2(floored pixel coords) → ~[0, 1]. */
export const blue12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    let v = d.f32(0)
    // v += hash12(p + vec2((k%3)-1, floor(k/3)-1)) over the 3×3 neighbourhood. Offsets are derived
    // with float floor (TGSL `/` is always float division — integer index math is unavailable).
    for (let k = 0; k < 9; k++) {
        const fk = d.f32(k)
        const row = std.floor(fk / 3.0) // 0,0,0,1,1,1,2,2,2
        const col = fk - row * 3.0 // 0,1,2,0,1,2,0,1,2
        v = v + hash12(p.add(d.vec2f(col - 1.0, row - 1.0)))
    }
    return 0.9 * (1.125 * hash12(p) - v / 8.0) + 0.5
})

// ─── Hilbert blue noise (ordered, Hilbert-curve + golden ratio) ───────────────

/** Hilbert-curve index of integer pixel `p` over a 512×512 grid. */
const hilbertEncode512 = tgpu.fn([d.vec2i], d.i32)((p) => {
    'use gpu'
    let i = d.i32(0)
    let px = p.x
    let py = p.y
    for (let s = 256; s > 0; s = s >> 1) {
        const S = d.i32(s)
        const rx = std.select(d.i32(0), d.i32(1), (px & S) > 0)
        const ry = std.select(d.i32(0), d.i32(1), (py & S) > 0)
        const dq = (rx << 1) | (rx ^ ry)
        i = i + S * S * dq
        // rotate the sub-quadrant: p ^= (p.x^p.y)*(1-ry) ^ (s-1)*(rx & (1-ry))
        const oneMinusRy = d.i32(1) - ry
        const rhs = ((px ^ py) * oneMinusRy) ^ (d.i32(s - 1) * (rx & oneMinusRy))
        px = px ^ rhs
        py = py ^ rhs
    }
    return i
})

/** hilbertBlue12 — vec2(floored pixel coords) → [0, 1] ordered blue noise. */
export const hilbertBlue12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const idx = hilbertEncode512(d.vec2i(d.i32(p.x), d.i32(p.y))) & d.i32(262143) // % 262144
    return std.fract(0.6180339887498948482 * d.f32(idx))
})

// ─── Scratches ──────────────────────────────────────────────────────────────

/**
 * One scratch streak through cell `uv`; `f` is the antialias sharpness, `time` flickers the
 * streak in place, `thickness` widens the bright core (it divides the line falloff).
 */
const scratchOne = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32], d.f32)((uv, f, time, thickness) => {
    'use gpu'
    const seed0 = std.floor(uv)
    const lo = uv.sub(seed0)
    const sx = std.floor(std.sin(seed0.x * 51024.0) * 3104.0)
    const sy = std.floor(std.sin(seed0.y * 1324.0) * 554.0)
    const ss = sx + sy
    let u2 = lo.mul(2.0).sub(1.0)
    // rotate: uv*cos(ss) + vec2(-uv.y, uv.x)*sin(ss)
    u2 = u2.mul(std.cos(ss)).add(d.vec2f(-u2.y, u2.x).mul(std.sin(ss)))
    u2 = u2.add(std.sin(sx - sy))
    u2 = u2.mul(0.5).add(0.5)
    const WAVYNESS = 0.2
    const sVal = (std.sin(sx + u2.y * Math.PI) + std.sin(sy + u2.y * Math.PI)) * WAVYNESS
    let x = std.abs(u2.x - 0.5 + sVal)
    // Dividing the falloff by thickness widens the bright core of the streak.
    x = 0.5 - x * (f / thickness)
    x = std.smoothstep(-2.0, std.fwidth(x) * 1.5 + 16.0, x) * 12.0
    // Per-cell intensity pulse — phase from the cell hash so streaks flicker independently.
    const pulse = std.sin(time + hash12(seed0) * 6.2831853) * 0.5 + 0.5
    return x * u2.y * pulse
})

/** scratches12 — overlaid hairline scratches, vec2 → [0, ~12] (typically used with clamp). */
export const scratches12 = tgpu.fn([d.vec2f, d.f32, d.f32], d.f32)((uv, time, thickness) => {
    'use gpu'
    let scratches = d.f32(0)
    let p = d.vec2f(uv.x, uv.y)
    const f = 1.0 / std.length(std.fwidth(uv))
    for (let i = 0; i < 8; i++) {
        scratches = std.max(scratches, scratchOne(p, f, time, thickness))
        // p * mat2(1, 0.7, -0.7, 1) - 12.31
        p = d.vec2f(p.x + p.y * 0.7, p.x * -0.7 + p.y).sub(12.31)
    }
    return scratches
})

// ─── fBm builders for the composite textures ──────────────────────────────────

/** fractal Perlin sum, vec2 → [0, 1]. */
export const fbmPerlin = tgpu.fn([d.vec2f, d.i32], d.f32)((p, octaves) => {
    'use gpu'
    let s = d.f32(0)
    let m = d.f32(0)
    let a = d.f32(1)
    let pp = d.vec2f(p.x, p.y)
    for (let i = 0; i < octaves; i++) {
        s = s + a * perlin12(pp)
        m = m + a
        a = a * 0.5
        pp = pp.mul(2.0)
    }
    return s / m
})

// ─── Stone ────────────────────────────────────────────────────────────────────

/** Raw (un-normalised) sum of Perlin-with-derivatives, vec2 → vec3 (6 octaves). */
const fbmStone = tgpu.fn([d.vec2f], d.vec3f)((p) => {
    'use gpu'
    let s = d.vec3f(0.0, 0.0, 0.0)
    let a = d.f32(1)
    let pp = d.vec2f(p.x, p.y)
    for (let i = 0; i < 6; i++) {
        s = s.add(perlin12d(pp).mul(a))
        a = a * 0.5
        pp = pp.mul(2.0)
    }
    return s
})

/** stone12 — derivative-warped fractal noise, vec2 → [0, 1]. */
export const stone12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const fs = fbmStone(p)
    return fbmPerlin(p.add(d.vec2f(fs.y, fs.z).mul(0.4)), d.i32(6))
})

// ─── Wool ───────────────────────────────────────────────────────────────────

/** Fractal sum of the curl field, vec2 → vec2 (6 octaves). */
const fbmWool = tgpu.fn([d.vec2f], d.vec2f)((p) => {
    'use gpu'
    let s = d.vec2f(0.0, 0.0)
    let m = d.f32(0)
    let a = d.f32(1)
    let pp = d.vec2f(p.x, p.y)
    for (let i = 0; i < 6; i++) {
        s = s.add(curl22(pp).mul(a))
        m = m + a
        a = a * 0.5
        pp = pp.mul(2.0)
    }
    return s.div(m)
})

/** wool12 — interwoven fibres, vec2 → [0, ~1]. */
export const wool12 = tgpu.fn([d.vec2f], d.f32)((p) => {
    'use gpu'
    const n = fbmWool(p)
    return std.max(std.abs(n.x), std.abs(n.y))
})

// ─── Erosion ──────────────────────────────────────────────────────────────────

/** Eroded gully contribution at `p` flowing along `slope`, → vec3 (height, slope.xy). */
const gullies = tgpu.fn([d.vec2f, d.vec2f], d.vec3f)((p, slope) => {
    'use gpu'
    const sideDir = d.vec2f(-slope.y, slope.x).mul(Math.PI)
    const id = std.floor(p)
    const pp = p.sub(id)
    let heightSlope = d.vec2f(0.0, 0.0)
    let wSum = d.f32(0)
    for (let x = -1; x <= 2; x++) {
        for (let y = -1; y <= 2; y++) {
            const off = d.vec2f(d.f32(x), d.f32(y))
            const c = pp.sub(off).sub(hash22(id.add(off))).add(0.5)
            const dist2 = std.dot(c, c)
            const w = std.max(d.f32(0), std.exp(dist2 * -2.0) - 0.01111)
            wSum = wSum + w
            const t = std.dot(c, sideDir)
            heightSlope = heightSlope.add(d.vec2f(std.cos(t), -std.sin(t)).mul(w))
        }
    }
    return d.vec3f(heightSlope.x, heightSlope.y * sideDir.x, heightSlope.y * sideDir.y).div(wSum)
})

/** erosion12 — hydraulic-erosion-style ridges, vec2 → vec3 (height in .x). */
export const erosion12 = tgpu.fn([d.vec2f], d.vec3f)((p) => {
    'use gpu'
    let nd = perlin12d(p)
    let strength = d.f32(0.25)
    let freq = d.f32(8)
    let total = d.f32(1)
    for (let i = 0; i < 4; i++) {
        const grad = d.vec2f(nd.y, nd.z)
        const len2 = std.max(std.dot(grad, grad), d.f32(1e-4)) // guard pow(0, -0.25) → ∞
        const slope = grad.mul(std.pow(len2, d.f32(0.5 * (0.5 - 1.0))))
        nd = nd.add(gullies(p.mul(freq), slope).mul(strength).mul(d.vec3f(1.0, freq, freq)))
        total = total + strength
        strength = strength * 0.5
        freq = freq * 2.0
    }
    return nd.div(total)
})

// ─── Paper ────────────────────────────────────────────────────────────────────

/** Fractal sum of clamped curl, vec2 → vec2 (used by paper grain). */
const fbmPaper = tgpu.fn([d.vec2f, d.i32], d.vec2f)((p, octaves) => {
    'use gpu'
    let s = d.vec2f(0.0, 0.0)
    let m = d.f32(0)
    let a = d.f32(1)
    let pp = d.vec2f(p.x, p.y)
    for (let i = 0; i < octaves; i++) {
        s = s.add(std.clamp(curl22(pp).mul(0.5).add(0.5), d.vec2f(0.0, 0.0), d.vec2f(1.0, 1.0)).mul(a))
        m = m + a
        a = a * 0.8
        pp = pp.mul(2.0)
    }
    return s.div(m)
})

/** paper12 — fibrous paper grain, vec2 → roughly [0.4, 1]. `octaves` defaults to 10. */
export const paper12 = tgpu.fn([d.vec2f, d.i32], d.f32)((p, octaves) => {
    'use gpu'
    return std.length(fbmPaper(p, octaves)) / 1.414 * 0.6 + 0.4
})
