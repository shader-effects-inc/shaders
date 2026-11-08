// Rings rippling outward from a point, bending the layers beneath. Features: the `map` slot (a
// coordinate warp that composes analytically with other warps), with its own clock.
import {defineShader, wgsl} from 'shaders/std'

export default defineShader({
    name: 'RippleWarp',
    description: 'Rings ripple outward from a point and bend the layers beneath',
    props: {
        center: {default: {x: 0.5, y: 0.5}, ui: {type: 'position', label: 'Center'}},
        amount: {default: 0.015, ui: {type: 'range', min: 0, max: 0.06, step: 0.001, label: 'Amount'}},
        rings: {default: 30, ui: {type: 'range', min: 5, max: 80, step: 1, label: 'Rings'}},
    },
    map: wgsl`
        let d = (uv - center) * vec2f(aspect, 1.0);
        let r = length(d);
        let wave = sin(r * rings - time * 3.0) * amount * smoothstep(0.8, 0.0, r);
        return uv + normalize(d + vec2f(1e-5)) * wave / vec2f(aspect, 1.0);
    `,
})
