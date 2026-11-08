/**
 * Surface: normals and reflectance for shading flat layers as if they had relief. Directions use
 * one convention: x right, y up, z out of the screen toward the viewer; light directions point
 * from the surface TOWARD the light. Vectors need not be normalised.
 */
import {libFn} from '../wgslLibrary'

export const normalFromHeight = libFn({
    name: 'normalFromHeight',
    family: 'surface',
    signature: 'normalFromHeight(slope: vec2f, strength: f32) -> vec3f  // unit normal, z toward the viewer',
    doc: 'The surface normal of a height field from its slope (gradient, e.g. noise2d(p).grad). strength scales how tall the bumps look; 0 is flat (0, 0, 1).',
    code: `
fn normalFromHeight(slope: vec2f, strength: f32) -> vec3f {
    return normalize(vec3f(-slope * strength, 1.0));
}
`,
})

export const fresnel = libFn({
    name: 'fresnel',
    family: 'surface',
    signature: 'fresnel(cosTheta: f32, f0: f32) -> f32  // f0..1; cosTheta = dot(normal, viewDir)',
    doc: 'Schlick\'s Fresnel: how much light a surface reflects at this viewing angle, from f0 head-on (0.04 for glass and plastic, 0.5 to 1 for metals) up to 1 at grazing edges. Brightens rims.',
    code: `
fn fresnel(cosTheta: f32, f0: f32) -> f32 {
    let m = 1.0 - clamp(cosTheta, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (1.0 - f0) * m2 * m2 * m;
}
`,
})

export const specular = libFn({
    name: 'specular',
    family: 'surface',
    signature: 'specular(n: vec3f, l: vec3f, v: vec3f, roughness: f32) -> f32  // >= 0 (can exceed 1); roughness 0..1',
    doc: 'The highlight a light makes on a glossy surface: GGX microfacet distribution with Smith visibility, already multiplied by N.L. n normal, l toward the light, v toward the viewer (vec3f(0.0, 0.0, 1.0) for flat layers). Multiply by fresnel(...) and the light colour.',
    code: `
fn specular(n: vec3f, l: vec3f, v: vec3f, roughness: f32) -> f32 {
    let nn = normalize(n);
    let ll = normalize(l);
    let vv = normalize(v);
    let h = normalize(ll + vv + vec3f(0.0, 0.0, 1e-6));
    let ndl = max(dot(nn, ll), 0.0);
    let ndv = max(dot(nn, vv), 1e-4);
    let ndh = max(dot(nn, h), 0.0);
    let a = max(roughness * roughness, 1e-3);
    let a2 = a * a;
    let dn = ndh * ndh * (a2 - 1.0) + 1.0;
    let distribution = a2 / (3.14159265 * dn * dn);
    let k = a * 0.5;
    let vis = 1.0 / ((ndl * (1.0 - k) + k) * (ndv * (1.0 - k) + k));
    return distribution * vis * 0.25 * ndl;
}
`,
})

export const envGradient = libFn({
    name: 'envGradient',
    family: 'surface',
    signature: 'envGradient(dir: vec3f, horizon: vec3f, zenith: vec3f, ground: vec3f) -> vec3f  // y up',
    doc: 'A simple sky to reflect: colour seen along dir, blending from horizon (dir level) up to zenith (straight up, +y) and down to ground. Pass reflect(-viewDir, normal) for a cheap chrome or glossy look.',
    code: `
fn envGradient(dir: vec3f, horizon: vec3f, zenith: vec3f, ground: vec3f) -> vec3f {
    let y = normalize(dir).y;
    let sky = mix(horizon, zenith, sqrt(max(y, 0.0)));
    let below = mix(horizon, ground, sqrt(max(-y, 0.0)));
    return select(below, sky, y >= 0.0);
}
`,
})
