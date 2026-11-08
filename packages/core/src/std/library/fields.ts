/**
 * Fields: turning a scalar value (a distance, a noise, a height) into bands, contour lines and
 * masks. The line functions filter over the pixel exactly (the Quilez box filter, applied before
 * the repeat), so dense lines fade to their average instead of shimmering. They use derivatives:
 * fragment bodies only, called outside pixel-dependent branches.
 */
import {libFn} from '../wgslLibrary'

export const bands = libFn({
    name: 'bands',
    family: 'fields',
    signature: 'bands(d: f32, spacing: f32, width: f32) -> f32  // 0..1; spacing and width in d units',
    doc: 'Repeating anti-aliased lines along any distance or value: one line every spacing, width wide, centred on 0, spacing, 2 * spacing, ... Feed a shape\'s distance for rings that echo its outline (ripples, topographic rings, halftone rings).',
    code: `
fn bands(d: f32, spacing: f32, width: f32) -> f32 {
    let sp = max(spacing, 1e-6);
    let duty = clamp(width / sp, 1e-5, 1.0);
    let x = d / sp + duty * 0.5;
    let w = max(max(abs(dpdx(x)), abs(dpdy(x))), 1e-5);
    let n = 1.0 / duty;
    let a = x + w * 0.5;
    let b = x - w * 0.5;
    let cov = (floor(a) + min(fract(a) * n, 1.0) - floor(b) - min(fract(b) * n, 1.0)) / (n * w);
    return clamp(cov, 0.0, 1.0) * step(1e-6, width);
}
`,
})

export const isolines = libFn({
    name: 'isolines',
    family: 'fields',
    signature: 'isolines(f: f32, interval: f32, width: f32) -> f32  // 0..1; width in PIXELS',
    doc: 'Contour lines of any smooth field (noise, height, a distance) every interval, each a constant width in pixels however steep the field is, like an elevation map.',
    code: `
fn isolines(f: f32, interval: f32, width: f32) -> f32 {
    let x0 = f / max(interval, 1e-6);
    let w = max(max(abs(dpdx(x0)), abs(dpdy(x0))), 1e-5);
    let duty = clamp(width * w, 1e-5, 1.0);
    let x = x0 + duty * 0.5;
    let n = 1.0 / duty;
    let a = x + w * 0.5;
    let b = x - w * 0.5;
    let cov = (floor(a) + min(fract(a) * n, 1.0) - floor(b) - min(fract(b) * n, 1.0)) / (n * w);
    return clamp(cov, 0.0, 1.0) * step(1e-6, width);
}
`,
})

export const quantize = libFn({
    name: 'quantize',
    family: 'fields',
    signature: 'quantize(f: f32, steps: f32) -> f32  // snaps down to multiples of 1 / steps',
    doc: 'Snap a value down to a staircase with steps levels per unit (0..1 in, 0, 1/steps, ... out). For terraces, stepped gradients and stepped animation.',
    code: `
fn quantize(f: f32, steps: f32) -> f32 {
    let n = max(steps, 1.0);
    return floor(f * n) / n;
}
`,
})

export const proximity = libFn({
    name: 'proximity',
    family: 'fields',
    signature: 'proximity(p: vec2f, center: vec2f, radius: f32) -> f32  // 1 at center, 0 at radius and beyond',
    doc: 'A smooth falloff around a point: 1 at center, easing to 0 at radius. For hover glows, magnets and soft spotlights (pass the mouse as center).',
    code: `
fn proximity(p: vec2f, center: vec2f, radius: f32) -> f32 {
    return 1.0 - smoothstep(0.0, max(radius, 1e-6), distance(p, center));
}
`,
})

export const relief = libFn({
    name: 'relief',
    family: 'fields',
    signature: 'relief(height: f32, slope: vec2f, lightDir: vec3f) -> f32  // 0..1 shade; slope = d height / d p; lightDir toward the light, z out of the screen',
    doc: 'Shade a height field as if lit from lightDir (like an embossed or hill-shaded map): slope is the height\'s gradient (e.g. noise2d(p).grad), and height 0..1 darkens low areas slightly.',
    code: `
fn relief(height: f32, slope: vec2f, lightDir: vec3f) -> f32 {
    let n = normalize(vec3f(-slope, 1.0));
    let diffuse = max(dot(n, normalize(lightDir)), 0.0);
    return diffuse * mix(0.75, 1.0, clamp(height, 0.0, 1.0));
}
`,
})
