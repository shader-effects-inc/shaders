// Tens of thousands of stars on spiral arms, seen from just above the disc while the camera
// orbits slowly, with half-resolution nebula wisps drifting past. Features: a formula-driven
// particle field (no state), a half-resolution pass, helpers reading props, library functions.
import {defineShader, p, pass, simulate, wgsl} from 'shaders/std'
import {hash11, fbm3, blackbody} from 'shaders/std/wgsl'

export default defineShader({
    name: 'GalaxyDrift',
    description: 'Stars on spiral arms seen from just above the disc, with nebula wisps drifting past',
    animatedTime: {speed: 'speed'},
    props: {
        stars: {default: 50000, ui: {type: 'range', min: 5000, max: 100000, step: 1000, label: 'Stars'}},
        twist: {default: 3, ui: {type: 'range', min: 0, max: 8, step: 0.1, label: 'Arm twist'}},
        nebula: {default: 0.6, ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Nebula'}},
        speed: {default: 1, ui: {type: 'range', min: 0, max: 3, step: 0.01, label: 'Speed'}},
    },
    helpers: wgsl`
        // Camera space: x right, y up, z forward. The camera orbits the galaxy and looks down a little.
        fn camera(t: f32) -> mat3x3f {
            let a = t * 0.03;
            let tilt = 0.42;
            let ry = mat3x3f(vec3f(cos(a), 0.0, -sin(a)), vec3f(0.0, 1.0, 0.0), vec3f(sin(a), 0.0, cos(a)));
            let rx = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, cos(tilt), sin(tilt)), vec3f(0.0, -sin(tilt), cos(tilt)));
            return rx * ry;
        }
        fn starPos(i: u32, t: f32) -> vec3f {
            let fi = f32(i);
            let r = 0.05 + pow(hash11(fi * 1.31), 0.7) * 1.2;
            let arm = floor(hash11(fi * 3.7) * 3.0);
            let spread = (hash11(fi * 5.1) - 0.5) * 0.6;
            let a = arm * 2.0944 + twist * log(r + 0.05) + spread - t * 0.08 / max(r, 0.1);
            let h = (hash11(fi * 9.3) - 0.5) * 0.04 * (1.2 - r);
            return vec3f(cos(a) * r, h, sin(a) * r);
        }
        fn toView(q: vec3f, t: f32) -> vec3f {
            return camera(t) * q + vec3f(0.0, 0.25, 1.6);
        }
    `,
    inputs: {
        hash11, fbm3, blackbody,
        starfield: simulate.particles({
            count: p('stars'),
            max: 100000,
            mobile: {max: 12000},
            splat: 'soft',
            draw: wgsl`
                let v = toView(starPos(index, time), time);
                if (v.z <= 0.05) { return Point(vec2f(-1.0), 0.0, vec4f(0.0)); }
                let s = v.xy / v.z * 1.4;
                let at = vec2f(0.5 + s.x * 0.5 / aspect, 0.5 - s.y * 0.5);
                let heat = 0.35 + hash11(f32(index) * 7.7) * 0.65;
                return Point(at, max(2.5 / v.z * targetSize.y / 1080.0, 1.0), vec4f(blackbody(heat) * (0.7 / v.z), 1.0));
            `,
        }),
        clouds: pass(wgsl`
            let rd = normalize(vec3f((uv.x - 0.5) * 2.0 * aspect / 1.4, (0.5 - uv.y) * 2.0 / 1.4, 1.0));
            let back = transpose(camera(time));
            var glow = vec3f(0.0);
            var trans = 1.0;
            for (var s = 0; s < 40; s++) {
                let dist = 0.3 + f32(s) * 0.06;
                let w = back * (rd * dist - vec3f(0.0, 0.25, 1.6));
                let shape = exp(-abs(w.y) * 12.0) * (1.0 - smoothstep(0.2, 1.4, length(w.xz)));
                let density = max(fbm3(w * 3.0 + vec3f(0.0, 0.0, time * 0.02), 4) - 0.05, 0.0) * shape * nebula;
                let tint = mix(vec3f(0.55, 0.25, 0.85), vec3f(1.0, 0.45, 0.35), smoothstep(0.0, 1.0, length(w.xz)));
                glow += trans * density * tint * 0.08;
                trans *= exp(-density * 0.1);
            }
            return vec4f(glow, trans);
        `, {scale: 0.5}),
    },
    paint: wgsl`
        let c = clouds(uv);
        return vec4f(starfield(uv).rgb * c.a + c.rgb, 1.0);
    `,
})
