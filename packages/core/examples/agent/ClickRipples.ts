// Every click sends out a ring; rapid clicks overlap. Features: `events` (a queue of presses).
import {defineShader, events, wgsl} from 'shaders/std'

export default defineShader({
    name: 'ClickRipples',
    description: 'Each click sends a ring outward; rapid clicks overlap',
    props: {
        color: {default: '#ffffff', ui: {type: 'color', label: 'Color'}},
        speed: {default: 0.5, ui: {type: 'range', min: 0.1, max: 2, step: 0.01, label: 'Ring speed'}},
    },
    inputs: {clicks: events('click', {max: 8, maxAge: 3})},
    paint: wgsl`
        var c = 0.0;
        for (var i = 0; i < 8; i++) {
            let e = clicks(i);
            if (e.z < 0.0) { continue; }
            let d = length((uv - e.xy) * vec2f(aspect, 1.0));
            let ring = abs(d - e.z * speed);
            c += (1.0 - smoothstep(0.0, 0.006, ring)) * exp(-e.z * 1.5);
        }
        return vec4f(color.rgb, clamp(c, 0.0, 1.0));
    `,
})
