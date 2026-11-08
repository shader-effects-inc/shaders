/**
 * Light and volume: diffuse light, fog, scattering and waves. Same direction convention as the
 * surface family: z toward the viewer, light directions point toward the light.
 */
import {libFn} from '../wgslLibrary'

export const directionalLight = libFn({
    name: 'directionalLight',
    family: 'light',
    signature: 'directionalLight(n: vec3f, dir: vec3f, color: vec3f) -> vec3f  // linear light',
    doc: 'Diffuse (Lambert) light from a distant source like the sun: color scaled by how directly the surface faces dir (toward the light), 0 when it faces away. Add an ambient term for the shadow side.',
    code: `
fn directionalLight(n: vec3f, dir: vec3f, color: vec3f) -> vec3f {
    return color * max(dot(normalize(n), normalize(dir)), 0.0);
}
`,
})

export const fog = libFn({
    name: 'fog',
    family: 'light',
    signature: 'fog(color: vec3f, fogColor: vec3f, depth: f32, density: f32) -> vec3f  // linear',
    doc: 'Fade a colour into fogColor with distance (exponential fog): at depth 1 / density about two thirds of the way. Use for depth haze, mist and underwater falloff.',
    code: `
fn fog(color: vec3f, fogColor: vec3f, depth: f32, density: f32) -> vec3f {
    return mix(fogColor, color, exp(-max(depth * density, 0.0)));
}
`,
})

export const phaseHG = libFn({
    name: 'phaseHG',
    family: 'light',
    signature: 'phaseHG(cosTheta: f32, g: f32) -> f32  // >= 0, integrates to 1 over the sphere; g -1..1',
    doc: 'Henyey-Greenstein phase: how much light a hazy medium scatters toward the viewer, given the cosine between the light and view directions. g > 0 favours forward scattering (the bright halo around a sun behind clouds), g < 0 backward, 0 even.',
    code: `
fn phaseHG(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-6);
    return (1.0 - g2) / (12.5663706 * denom * sqrt(denom));
}
`,
})

export const gerstner = libFn({
    name: 'gerstner',
    family: 'light',
    signature: 'gerstner(p: vec2f, t: f32, dir: vec2f, steepness: f32, wavelength: f32) -> vec3f  // (x, y) sideways shift, z height',
    doc: 'One Gerstner ocean wave at point p and time t (seconds, deep-water speed): the offset to add to the surface, with sharp crests and wide troughs. steepness 0..1 (1 is the sharpest crest); sum a few with different dir and wavelength for water.',
    code: `
fn gerstner(p: vec2f, t: f32, dir: vec2f, steepness: f32, wavelength: f32) -> vec3f {
    let k = 6.28318531 / max(wavelength, 1e-4);
    let speed = sqrt(9.8 / k);
    let dn = normalize(dir);
    let f = k * (dot(dn, p) - speed * t);
    let a = clamp(steepness, 0.0, 1.0) / k;
    let c = cos(f);
    return vec3f(dn * (a * c), a * sin(f));
}
`,
})
