/**
 * Colour. Colours in the engine are LINEAR light in Display P3 primaries (P3-linear) unless a
 * function says otherwise: that is what `child`, colour props and textures hand a body, and what
 * a body returns. Every function here takes and returns P3-linear, so they chain without
 * conversions. OKLab/OKLCh values are the standard ones (L 0..1, chroma about 0..0.4).
 */
// Maintainer notes. Conversions are ported from `gpu/kit/colorMixing.ts` (P3 <-> sRGB matrices,
// Ottosson OKLab, the shortest-arc / achromatic hue mix) with two hardening changes: the inverse
// cube uses x*x*x (pow is NaN for negative bases) and the cube root guards abs() away from 0.
import {libFn} from '../wgslLibrary'

export const p3ToSrgb = libFn({
    name: 'p3ToSrgb',
    family: 'colour',
    signature: 'p3ToSrgb(rgb: vec3f) -> vec3f  // P3-linear in, sRGB-linear out',
    doc: 'Re-express a linear P3 colour with sRGB primaries (both linear). Vivid P3 colours come out with components below 0 or above 1.',
    code: `
fn p3ToSrgb(rgb: vec3f) -> vec3f {
    return vec3f(
        rgb.x * 1.2249401 - rgb.y * 0.2249404,
        rgb.x * -0.0420569 + rgb.y * 1.0420571,
        rgb.x * -0.0196376 - rgb.y * 0.0786361 + rgb.z * 1.0982735,
    );
}
`,
})

export const srgbToP3 = libFn({
    name: 'srgbToP3',
    family: 'colour',
    signature: 'srgbToP3(rgb: vec3f) -> vec3f  // sRGB-linear in, P3-linear out',
    doc: 'Re-express a linear sRGB colour with P3 primaries (both linear). Use it on colours written as linear sRGB (e.g. decoded hex values) before returning them.',
    code: `
fn srgbToP3(rgb: vec3f) -> vec3f {
    return vec3f(
        rgb.x * 0.8224621 + rgb.y * 0.177538,
        rgb.x * 0.0331941 + rgb.y * 0.9668058,
        rgb.x * 0.0170826 + rgb.y * 0.0723974 + rgb.z * 0.9105199,
    );
}
`,
})

export const emissiveAlpha = libFn({
    name: 'emissiveAlpha',
    family: 'colour',
    signature: 'emissiveAlpha(rgb: vec3f) -> vec4f  // straight-alpha rgba',
    doc: 'Turn light into a colour with alpha so it composites over transparency: alpha is the brightest channel, the colour is scaled up to match. Over black it looks exactly like the light; over anything else like a glowing tint. The last step of a glow, spill, flare or star field. Not for opaque surfaces.',
    code: `
fn emissiveAlpha(rgb: vec3f) -> vec4f {
    let a = clamp(max(max(rgb.x, rgb.y), rgb.z), 0.0, 1.0);
    return vec4f(rgb / max(a, 0.0001), a);
}
`,
})

export const luminance = libFn({
    name: 'luminance',
    family: 'colour',
    signature: 'luminance(rgb: vec3f) -> f32  // P3-linear in; 0..1 for in-range colours',
    doc: 'Relative luminance (CIE Y) of a P3-linear colour: how bright it looks, linear in light. Use it for greyscale, masks from brightness, or keeping brightness while changing colour.',
    code: `
fn luminance(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.2289746, 0.6917385, 0.0792869));
}
`,
})

export const toOklab = libFn({
    name: 'toOklab',
    family: 'colour',
    signature: 'toOklab(rgb: vec3f) -> vec3f  // P3-linear in; (L 0..1, a, b about -0.4..0.4)',
    doc: 'Convert to OKLab, a perceptual space: L is lightness, a is green to red, b is blue to yellow. Equal steps look equal, so mix, blur or shift colours here.',
    deps: [p3ToSrgb],
    code: `
fn _toOklab_cbrt(x: vec3f) -> vec3f {
    return sign(x) * pow(max(abs(x), vec3f(1e-12)), vec3f(1.0 / 3.0));
}
fn toOklab(rgb: vec3f) -> vec3f {
    let c = p3ToSrgb(rgb);
    let lms = vec3f(
        c.x * 0.4122214708 + c.y * 0.5363325363 + c.z * 0.0514459929,
        c.x * 0.2119034982 + c.y * 0.6806995451 + c.z * 0.1073969566,
        c.x * 0.0883024619 + c.y * 0.2817188376 + c.z * 0.6299787005,
    );
    let q = _toOklab_cbrt(lms);
    return vec3f(
        q.x * 0.2104542553 + q.y * 0.793617785 - q.z * 0.0040720468,
        q.x * 1.9779984951 - q.y * 2.428592205 + q.z * 0.4505937099,
        q.x * 0.0259040371 + q.y * 0.7827717662 - q.z * 0.808675766,
    );
}
`,
})

export const fromOklab = libFn({
    name: 'fromOklab',
    family: 'colour',
    signature: 'fromOklab(lab: vec3f) -> vec3f  // (L, a, b) in; P3-linear out',
    doc: 'Convert an OKLab colour back to P3-linear. The inverse of toOklab.',
    deps: [srgbToP3],
    code: `
fn fromOklab(lab: vec3f) -> vec3f {
    let q = vec3f(
        lab.x + lab.y * 0.3963377774 + lab.z * 0.2158037573,
        lab.x - lab.y * 0.1055613458 - lab.z * 0.0638541728,
        lab.x - lab.y * 0.0894841775 - lab.z * 1.291485548,
    );
    let lms = q * q * q;
    return srgbToP3(vec3f(
        lms.x * 4.0767416621 - lms.y * 3.3077115913 + lms.z * 0.2309699292,
        lms.x * -1.2684380046 + lms.y * 2.6097574011 - lms.z * 0.3413193965,
        lms.x * -0.0041960863 - lms.y * 0.7034186147 + lms.z * 1.707614701,
    ));
}
`,
})

export const toOklch = libFn({
    name: 'toOklch',
    family: 'colour',
    signature: 'toOklch(rgb: vec3f) -> vec3f  // P3-linear in; (L 0..1, C about 0..0.4, h radians -pi..pi)',
    doc: 'Convert to OKLCh: lightness, chroma (colourfulness) and hue angle in radians. Shift .z to rotate hue, scale .y to change saturation.',
    deps: [toOklab],
    code: `
fn toOklch(rgb: vec3f) -> vec3f {
    let lab = toOklab(rgb);
    return vec3f(lab.x, length(lab.yz), atan2(lab.z, lab.y));
}
`,
})

export const fromOklch = libFn({
    name: 'fromOklch',
    family: 'colour',
    signature: 'fromOklch(lch: vec3f) -> vec3f  // (L, C, h radians) in; P3-linear out',
    doc: 'Convert an OKLCh colour (hue in radians, any value) back to P3-linear. The inverse of toOklch.',
    deps: [fromOklab],
    code: `
fn fromOklch(lch: vec3f) -> vec3f {
    return fromOklab(vec3f(lch.x, lch.y * cos(lch.z), lch.y * sin(lch.z)));
}
`,
})

export const mixOklab = libFn({
    name: 'mixOklab',
    family: 'colour',
    signature: 'mixOklab(a: vec4f, b: vec4f, t: f32) -> vec4f  // P3-linear rgba, t 0..1',
    doc: 'Blend two colours in OKLab (no muddy midpoints), weighted by alpha so a transparent end never tints the result. t = 0 gives a, 1 gives b; alpha mixes linearly.',
    deps: [toOklab, fromOklab],
    code: `
fn mixOklab(a: vec4f, b: vec4f, t: f32) -> vec4f {
    let wa = a.w * (1.0 - t);
    let wb = b.w * t;
    let total = wa + wb;
    let lab = (toOklab(a.xyz) * wa + toOklab(b.xyz) * wb) / max(total, 0.001);
    return vec4f(fromOklab(lab), total);
}
`,
})

export const mixOklch = libFn({
    name: 'mixOklch',
    family: 'colour',
    signature: 'mixOklch(a: vec4f, b: vec4f, t: f32) -> vec4f  // P3-linear rgba, t 0..1',
    doc: 'Blend two colours in OKLCh, going round the hue wheel the short way, so the midpoint stays vivid. A grey end takes the other end\'s hue (no stray tints); alpha-weighted like mixOklab.',
    deps: [toOklch, fromOklch],
    code: `
fn mixOklch(a: vec4f, b: vec4f, t: f32) -> vec4f {
    let wa = a.w * (1.0 - t);
    let wb = b.w * t;
    let total = wa + wb;
    let sw = max(total, 0.001);
    let la = toOklch(a.xyz);
    let lb = toOklch(b.xyz);
    let ha = select(la.z, lb.z, la.y < 0.001);
    let hb = select(lb.z, ha, lb.y < 0.001);
    let diff = hb - ha;
    let wrapped = diff - floor((diff + 3.14159265) * 0.15915494) * 6.28318531;
    let lc = (la.xy * wa + lb.xy * wb) / sw;
    return vec4f(fromOklch(vec3f(lc, ha + wrapped * (wb / sw))), total);
}
`,
})

export const tonemap = libFn({
    name: 'tonemap',
    family: 'colour',
    signature: 'tonemap(rgb: vec3f) -> vec3f  // P3-linear HDR in (any brightness >= 0); P3-linear 0..1 out',
    doc: 'Compress bright HDR light into displayable 0..1 with the AgX curve (Blender / Filament): highlights roll off and desaturate toward white instead of clipping to flat primaries.',
    deps: [p3ToSrgb, srgbToP3],
    code: `
fn _tonemap_contrast(x: vec3f) -> vec3f {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn tonemap(rgb: vec3f) -> vec3f {
    let toRec2020 = mat3x3f(0.6274, 0.0691, 0.0164, 0.3293, 0.9195, 0.088, 0.0433, 0.0113, 0.8956);
    let fromRec2020 = mat3x3f(1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187);
    let inset = mat3x3f(
        0.856627153315983, 0.137318972929847, 0.11189821299995,
        0.0951212405381588, 0.761241990602591, 0.0767994186031903,
        0.0482516061458583, 0.101439036467562, 0.811302368396859,
    );
    let outset = mat3x3f(
        1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
        -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
        -0.016493938717834573, -0.016493938717834257, 1.2519364065950405,
    );
    var c = inset * (toRec2020 * p3ToSrgb(rgb));
    c = max(c, vec3f(1e-10));
    c = clamp((log2(c) + 12.47393) / (4.026069 + 12.47393), vec3f(0.0), vec3f(1.0));
    c = outset * _tonemap_contrast(c);
    c = pow(max(c, vec3f(0.0)), vec3f(2.2));
    return srgbToP3(clamp(fromRec2020 * c, vec3f(0.0), vec3f(1.0)));
}
`,
})

export const hueRotate = libFn({
    name: 'hueRotate',
    family: 'colour',
    signature: 'hueRotate(rgb: vec3f, angle: f32) -> vec3f  // P3-linear; angle in radians',
    doc: 'Turn the hue by an angle (radians, positive goes red to yellow to green) in OKLab, keeping lightness and colourfulness steady.',
    deps: [toOklab, fromOklab],
    code: `
fn hueRotate(rgb: vec3f, angle: f32) -> vec3f {
    let lab = toOklab(rgb);
    let c = cos(angle);
    let s = sin(angle);
    return fromOklab(vec3f(lab.x, c * lab.y - s * lab.z, s * lab.y + c * lab.z));
}
`,
})

export const saturation = libFn({
    name: 'saturation',
    family: 'colour',
    signature: 'saturation(rgb: vec3f, amount: f32) -> vec3f  // P3-linear; amount 0 grey, 1 unchanged, >1 boosted',
    doc: 'Scale colourfulness around the colour\'s own luminance: 0 makes it grey at the same brightness, 1 leaves it, 2 doubles it. (Named saturation because WGSL already has saturate = clamp 0..1.)',
    deps: [luminance],
    code: `
fn saturation(rgb: vec3f, amount: f32) -> vec3f {
    return mix(vec3f(luminance(rgb)), rgb, amount);
}
`,
})

export const exposure = libFn({
    name: 'exposure',
    family: 'colour',
    signature: 'exposure(rgb: vec3f, stops: f32) -> vec3f  // linear; +1 stop doubles the light',
    doc: 'Brighten or darken like a camera: each stop doubles (+1) or halves (-1) the light. Works on linear colour, so it can push past 1 (follow with tonemap).',
    code: `
fn exposure(rgb: vec3f, stops: f32) -> vec3f {
    return rgb * exp2(stops);
}
`,
})

export const posterize = libFn({
    name: 'posterize',
    family: 'colour',
    signature: 'posterize(rgb: vec3f, steps: f32) -> vec3f  // P3-linear; steps >= 2 levels per channel',
    doc: 'Reduce each channel to a few flat levels (including black and full), spaced evenly to the eye rather than in linear light, for a screen-print or retro look.',
    code: `
fn posterize(rgb: vec3f, steps: f32) -> vec3f {
    let n = max(floor(steps), 2.0);
    let e = pow(max(rgb, vec3f(1e-10)), vec3f(1.0 / 2.2));
    let q = min(floor(e * n) / (n - 1.0), vec3f(1.0));
    return pow(max(q, vec3f(1e-10)), vec3f(2.2));
}
`,
})

export const paletteQuantize = libFn({
    name: 'paletteQuantize',
    family: 'colour',
    signature: 'paletteQuantize(rgb: vec3f, palette: array<vec3f, 8>, count: i32) -> vec3f  // P3-linear; uses palette[0..count-1], count 1..8',
    doc: 'Snap a colour to the perceptually nearest of up to 8 palette colours (nearest in OKLab). Unused slots can hold anything; pair with blueDither for dithered palettes.',
    deps: [toOklab],
    code: `
fn paletteQuantize(rgb: vec3f, palette: array<vec3f, 8>, count: i32) -> vec3f {
    var pal = palette;
    let lab = toOklab(rgb);
    let n = clamp(count, 1, 8);
    var best = pal[0];
    var bestDist = 1e30;
    for (var i = 0; i < n; i++) {
        let c = pal[i];
        let dl = toOklab(c) - lab;
        let dd = dot(dl, dl);
        if (dd < bestDist) {
            bestDist = dd;
            best = c;
        }
    }
    return best;
}
`,
})
