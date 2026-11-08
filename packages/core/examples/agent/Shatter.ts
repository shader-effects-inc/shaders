// A plane of glass that shatters and reforms, differently every time. Features: a pure per-pixel
// effect (no compute), library functions returning structs (`cycle`), helpers.
import {defineShader, wgsl} from 'shaders/std'
import {cycle, easeInOut, hash22, rotate2} from 'shaders/std/wgsl'

export default defineShader({
    name: 'Shatter',
    description: 'Glass that shatters into shards and reforms, with new timing and motion every cycle',
    props: {
        period: {default: 6, ui: {type: 'range', min: 2, max: 12, step: 0.1, label: 'Cycle (s)'}},
        shards: {default: 7, ui: {type: 'range', min: 3, max: 20, step: 1, label: 'Shards'}},
        spread: {default: 0.25, ui: {type: 'range', min: 0, max: 0.6, step: 0.01, label: 'Spread'}},
    },
    inputs: {cycle, easeInOut, hash22, rotate2},
    helpers: wgsl`
        // The Voronoi cell (by lattice id) nearest to p, and how close p is to its edge.
        fn nearestCell(p: vec2f, seed: f32) -> vec3f {
            let base = floor(p);
            var best = vec2f(0.0);
            var d1 = 1e9;
            var d2 = 1e9;
            for (var j = -1; j <= 1; j++) {
                for (var i = -1; i <= 1; i++) {
                    let q = base + vec2f(f32(i), f32(j));
                    let d = length(p - (q + hash22(q + seed)));
                    if (d < d1) { d2 = d1; d1 = d; best = q; } else if (d < d2) { d2 = d; }
                }
            }
            return vec3f(best, d2 - d1);
        }
    `,
    effect: wgsl`
        let c = cycle(time, period);
        let k = easeInOut(sin(c.phase * 3.14159265));
        let scale = vec2f(aspect, 1.0) * shards;
        let gp = uv * scale;
        let base = floor(gp);
        var out = vec4f(0.0);
        var nearest = 1e9;
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let q = base + vec2f(f32(i), f32(j));
                let fp = q + hash22(q + c.seed);
                let rnd = hash22(q * 1.7 + c.seed);
                let away = normalize(fp - scale * 0.5 + vec2f(1e-4));
                let offset = away * spread * shards * k * (0.4 + rnd.y);
                let angle = (rnd.x - 0.5) * 2.5 * k;
                let local = rotate2(gp - fp - offset, -angle) + fp;
                let cell = nearestCell(local, c.seed);
                if (all(cell.xy == q) && rnd.y < nearest) {
                    nearest = rnd.y;
                    let tilt = vec2f(cos(angle), sin(angle)) * k * 0.02;
                    let px = textureSampleLevel(childTexture, childSampler, local / scale + tilt, 0.0);
                    let edge = 1.0 - smoothstep(0.0, 0.06, cell.z);
                    out = vec4f(px.rgb + edge * 0.35 * k, px.a);
                }
            }
        }
        return out;
    `,
})
