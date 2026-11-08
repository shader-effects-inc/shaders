/**
 * Anti-aliasing: masks that stay crisp up close and fade to the right average grey far away.
 *
 * Every function here measures its own pixel footprint with `dpdx`/`dpdy`, so they work in
 * fragment bodies only, and must be called where all pixels of a quad run the same code (not
 * inside an `if` that depends on the pixel). Pass the CONTINUOUS coordinate or distance: the
 * footprint of a value that was already `fract`-ed or `abs`-ed jumps at every seam.
 */
// Maintainer notes. WGSL ports of `gpu/kit/aa.ts` (Quilez analytical filters + footprint-widened
// smoothsteps). Falling ramps are written `1 - smoothstep(lo, hi, x)`, never `smoothstep(hi, lo,
// x)` (undefined in WGSL when edge0 > edge1).
import {libFn} from '../wgslLibrary'

export const footprint = libFn({
    name: 'footprint',
    family: 'aa',
    signature: 'footprint(p: vec2f) -> vec2f  // per-axis size of one pixel in p units, > 0',
    doc: 'How much p changes across one pixel, per axis (the larger of its x and y screen derivatives, never 0). Divide by it to turn p distances into pixels.',
    code: `
fn footprint(p: vec2f) -> vec2f {
    let dx = abs(dpdx(p));
    let dy = abs(dpdy(p));
    return max(max(dx, dy), vec2f(1e-5));
}
`,
})

export const aaStep = libFn({
    name: 'aaStep',
    family: 'aa',
    signature: 'aaStep(edge: f32, x: f32) -> f32  // 0..1',
    doc: 'An anti-aliased step(edge, x): 0 below the edge, 1 above, with a one-pixel ramp measured from x itself.',
    code: `
fn aaStep(edge: f32, x: f32) -> f32 {
    let fw = max(max(abs(dpdx(x)), abs(dpdy(x))), 1e-5);
    return clamp((x - edge) / fw + 0.5, 0.0, 1.0);
}
`,
})

export const aaFill = libFn({
    name: 'aaFill',
    family: 'aa',
    signature: 'aaFill(sd: f32) -> f32  // 0..1, 1 inside (sd < 0)',
    doc: 'Coverage of the inside of a signed distance (negative inside), with a one-pixel anti-aliased edge. The standard way to fill any sd* shape.',
    code: `
fn aaFill(sd: f32) -> f32 {
    let fw = max(max(abs(dpdx(sd)), abs(dpdy(sd))), 1e-5);
    return clamp(0.5 - sd / fw, 0.0, 1.0);
}
`,
})

export const aaLine = libFn({
    name: 'aaLine',
    family: 'aa',
    signature: 'aaLine(dist: f32, width: f32) -> f32  // 0..1',
    doc: 'Coverage of a line of total width `width` (in dist units) centred where dist = 0, e.g. an outline from a signed distance. Lines thinner than a pixel stay one pixel wide and fade by their true coverage instead of breaking up.',
    code: `
fn aaLine(dist: f32, width: f32) -> f32 {
    let fw = max(max(abs(dpdx(dist)), abs(dpdy(dist))), 1e-5);
    let half = max(width, fw) * 0.5;
    let cov = 1.0 - smoothstep(half - fw * 0.5, half + fw * 0.5, abs(dist));
    return cov * clamp(width / fw, 0.0, 1.0);
}
`,
})

export const aaDisc = libFn({
    name: 'aaDisc',
    family: 'aa',
    signature: 'aaDisc(dist: f32, radius: f32) -> f32  // 0..1, 1 inside',
    doc: 'Coverage of a disc, given the distance from its centre. The soft edge sits just inside the radius so small dots stay their size and only feather.',
    code: `
fn aaDisc(dist: f32, radius: f32) -> f32 {
    let fw = max(max(abs(dpdx(dist)), abs(dpdy(dist))), 1e-5);
    return 1.0 - smoothstep(radius - fw * 0.5, radius, dist);
}
`,
})

export const aaChecker = libFn({
    name: 'aaChecker',
    family: 'aa',
    signature: 'aaChecker(p: vec2f) -> f32  // 0..1; 0 on the cell at floor(p) = (0, 0)',
    doc: 'A checkerboard with one cell per unit of p, exactly box-filtered over the pixel: crisp up close, fading to a flat 0.5 when the cells get smaller than a pixel (no moire).',
    deps: [footprint],
    code: `
fn aaChecker(p: vec2f) -> f32 {
    let w = footprint(p);
    let lo = abs(fract((p - w * 0.5) * 0.5) - 0.5);
    let hi = abs(fract((p + w * 0.5) * 0.5) - 0.5);
    let i = (lo - hi) * 2.0 / w;
    return clamp(0.5 - i.x * i.y * 0.5, 0.0, 1.0);
}
`,
})

export const bandMask = libFn({
    name: 'bandMask',
    family: 'aa',
    signature: 'bandMask(x: f32, lo: f32, hi: f32) -> f32  // 0..1; x in 0..1',
    doc: 'A band across a unit cell: 1 in the middle, 0 within lo of either edge, ramping between lo and hi. Multiply an x band by a y band for an inset rectangle (bricks, tiles). Needs lo < hi.',
    code: `
fn bandMask(x: f32, lo: f32, hi: f32) -> f32 {
    return smoothstep(lo, hi, x) * smoothstep(lo, hi, 1.0 - x);
}
`,
})
