// The pointer warms the image it passes over, and the heat fades. Features: `pointerTrail`, the
// WGSL library's palettes in a pointwise effect.
import {defineShader, pointerTrail, wgsl} from 'shaders/std'
import {blackbody} from 'shaders/std/wgsl'

export default defineShader({
    name: 'CursorHeat',
    description: 'The pointer leaves glowing heat on the image that fades away',
    props: {
        strength: {default: 0.8, ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Strength'}},
    },
    inputs: {
        blackbody,
        trail: pointerTrail({decay: 0.7, radius: 0.05}),
    },
    effect: wgsl`
        let h = trail(uv).x * strength;
        return vec4f(child.rgb + blackbody(h) * h, child.a);
    `,
})
