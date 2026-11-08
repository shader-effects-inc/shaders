// Drops of paint falling through water, mixing as they go and pooling at the bottom. Features:
// the fluid solve with author-written forces and drops, OKLab dye, a solid floor that drains.
import {defineShader, simulate, wgsl} from 'shaders/std'
import {hash11} from 'shaders/std/wgsl'

export default defineShader({
    name: 'InkFall',
    description: 'Drops of paint sinking through water, blending as they meet and pooling at the bottom',
    props: {
        colorA: {default: '#2b6cff', ui: {type: 'color', label: 'Ink 1'}},
        colorB: {default: '#ff4f7b', ui: {type: 'color', label: 'Ink 2'}},
        colorC: {default: '#ffc83d', ui: {type: 'color', label: 'Ink 3'}},
        gravity: {default: 0.6, ui: {type: 'range', min: 0, max: 2, step: 0.01, label: 'Weight'}},
        drops: {default: 1.5, ui: {type: 'range', min: 0.2, max: 4, step: 0.1, label: 'Drops per second'}},
    },
    inputs: {
        hash11,
        ink: simulate.fluid({
            velocity: 192,
            dye: {res: 384, space: 'oklab'},
            boundary: {floor: 'solid', walls: 'solid'},
            drain: {floor: 0.04},
            forces: wgsl`return vec2f(0.0, gravity * min(dye.a, 1.0) * 0.6);`,
            inject: wgsl`
                let period = 1.0 / max(drops, 0.01);
                let id = floor(time / period);
                let at = vec2f(0.15 + hash11(id) * 0.7, 0.08);
                let pick = hash11(id + 7.0);
                let tint = select(select(colorC, colorB, pick < 0.66), colorA, pick < 0.33);
                let d = distance(cellUV * vec2f(aspect, 1.0), at * vec2f(aspect, 1.0));
                let k = (1.0 - smoothstep(0.0, 0.035, d)) * select(0.0, 1.0, isNew(period));
                return vec4f(tint.rgb, 1.0) * k;
            `,
        }),
    },
    paint: wgsl`
        let c = ink(uv);
        return vec4f(mix(vec3f(1.0), c.rgb, clamp(c.a, 0.0, 1.0)), 1.0);
    `,
})
