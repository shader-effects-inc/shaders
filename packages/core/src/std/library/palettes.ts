/**
 * Palettes: map a 0..1 value to a colour. Outputs are P3-linear like every engine colour.
 */
// Maintainer notes. falseColor / blackbody ramps are authored as 5 sRGB-encoded stops, interpolated
// in the encoded space (perceptually even), then decoded with the exact sRGB curve and re-expressed
// in P3 primaries. Stop values are part of the public behaviour: changing them is a new name.
import {libFn} from '../wgslLibrary'
import {srgbToP3} from './colour'

export const falseColor = libFn({
    name: 'falseColor',
    family: 'palettes',
    signature: 'falseColor(t: f32, palette: i32) -> vec3f  // t 0..1 (clamped); palette 0 THERMAL, 1 IRON, 2 NIGHT_VISION, 3 XRAY, 4 HYPSOMETRIC, 5 BLUEPRINT; P3-linear out',
    doc: 'Colour a 0..1 value with a scientific / imaging ramp: 0 THERMAL (navy, cyan, green, orange, white), 1 IRON (black, purple, red, orange, white-hot), 2 NIGHT_VISION (phosphor greens), 3 XRAY (black to cool white), 4 HYPSOMETRIC (map elevation: green, tan, brown, snow), 5 BLUEPRINT (deep to pale blue).',
    deps: [srgbToP3],
    code: `
fn _falseColor_decode(c: vec3f) -> vec3f {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}
fn _falseColor_ramp(x: f32, s0: vec3f, s1: vec3f, s2: vec3f, s3: vec3f, s4: vec3f) -> vec3f {
    let k = clamp(x, 0.0, 1.0) * 4.0;
    var c = mix(s0, s1, clamp(k, 0.0, 1.0));
    c = mix(c, s2, clamp(k - 1.0, 0.0, 1.0));
    c = mix(c, s3, clamp(k - 2.0, 0.0, 1.0));
    return mix(c, s4, clamp(k - 3.0, 0.0, 1.0));
}
fn falseColor(t: f32, palette: i32) -> vec3f {
    var c: vec3f;
    switch clamp(palette, 0, 5) {
        case 1: {
            c = _falseColor_ramp(t, vec3f(0.0, 0.0, 0.0), vec3f(0.27, 0.0, 0.45), vec3f(0.8, 0.1, 0.35), vec3f(1.0, 0.6, 0.0), vec3f(1.0, 1.0, 0.85));
        }
        case 2: {
            c = _falseColor_ramp(t, vec3f(0.0, 0.02, 0.0), vec3f(0.04, 0.2, 0.03), vec3f(0.15, 0.55, 0.1), vec3f(0.45, 0.9, 0.35), vec3f(0.85, 1.0, 0.75));
        }
        case 3: {
            c = _falseColor_ramp(t, vec3f(0.0, 0.01, 0.03), vec3f(0.1, 0.16, 0.24), vec3f(0.38, 0.48, 0.58), vec3f(0.74, 0.82, 0.88), vec3f(0.97, 0.99, 1.0));
        }
        case 4: {
            c = _falseColor_ramp(t, vec3f(0.13, 0.45, 0.25), vec3f(0.55, 0.7, 0.35), vec3f(0.9, 0.82, 0.5), vec3f(0.6, 0.42, 0.28), vec3f(0.97, 0.97, 0.97));
        }
        case 5: {
            c = _falseColor_ramp(t, vec3f(0.02, 0.1, 0.3), vec3f(0.06, 0.24, 0.55), vec3f(0.25, 0.5, 0.8), vec3f(0.65, 0.82, 0.95), vec3f(0.97, 0.99, 1.0));
        }
        default: {
            c = _falseColor_ramp(t, vec3f(0.02, 0.02, 0.2), vec3f(0.0, 0.6, 0.95), vec3f(0.3, 0.9, 0.2), vec3f(1.0, 0.55, 0.0), vec3f(1.0, 1.0, 0.95));
        }
    }
    return srgbToP3(_falseColor_decode(c));
}
`,
})

export const blackbody = libFn({
    name: 'blackbody',
    family: 'palettes',
    signature: 'blackbody(t: f32) -> vec3f  // t 0..1 (clamped): cold black, red, orange, yellow, white-hot; P3-linear out',
    doc: 'The colour of something heating up: black at 0, through deep red and orange to white-hot at 1. For fire, embers, lava and heat maps; multiply by an intensity for glow.',
    deps: [srgbToP3],
    code: `
fn _blackbody_decode(c: vec3f) -> vec3f {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}
fn blackbody(t: f32) -> vec3f {
    let k = clamp(t, 0.0, 1.0) * 4.0;
    var c = mix(vec3f(0.0, 0.0, 0.0), vec3f(0.55, 0.04, 0.0), clamp(k, 0.0, 1.0));
    c = mix(c, vec3f(0.95, 0.3, 0.02), clamp(k - 1.0, 0.0, 1.0));
    c = mix(c, vec3f(1.0, 0.72, 0.3), clamp(k - 2.0, 0.0, 1.0));
    c = mix(c, vec3f(1.0, 0.97, 0.9), clamp(k - 3.0, 0.0, 1.0));
    return srgbToP3(_blackbody_decode(c));
}
`,
})

export const cosinePalette = libFn({
    name: 'cosinePalette',
    family: 'palettes',
    signature: 'cosinePalette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f  // a + b * cos(2pi * (c * t + d)); colour as given',
    doc: 'Inigo Quilez\'s cosine palette: a is the mean colour, b the swing, c how many cycles over t = 0..1, d the phase per channel. Used as is (treat a and b as linear colour), so it can leave 0..1.',
    code: `
fn cosinePalette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
    return a + b * cos(6.28318531 * (c * t + d));
}
`,
})
