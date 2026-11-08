/**
 * Paths: patterns along a curve, driven by a distance travelled along it.
 */
import {libFn} from '../wgslLibrary'

export const dashAlong = libFn({
    name: 'dashAlong',
    family: 'paths',
    signature: 'dashAlong(t: f32, dash: f32, gap: f32) -> f32  // 0..1; t = distance along the path, dash and gap in t units',
    doc: 'A dashed pattern along a path: 1 on the dashes, 0 in the gaps, exactly anti-aliased (dashes far smaller than a pixel fade to their average). The first dash starts at t = 0; subtract time from t to march the dashes. Multiply by aaLine for a dashed stroke. Fragment bodies only.',
    code: `
fn dashAlong(t: f32, dash: f32, gap: f32) -> f32 {
    let period = max(dash + gap, 1e-6);
    let duty = clamp(dash / period, 1e-5, 1.0);
    let x = t / period;
    let w = max(max(abs(dpdx(x)), abs(dpdy(x))), 1e-5);
    let n = 1.0 / duty;
    let a = x + w * 0.5;
    let b = x - w * 0.5;
    let cov = (floor(a) + min(fract(a) * n, 1.0) - floor(b) - min(fract(b) * n, 1.0)) / (n * w);
    return clamp(cov, 0.0, 1.0) * step(1e-6, dash);
}
`,
})
