// Sharp in a band across the middle, dreamy and glowing above and below it. Features: engine
// passes as inputs (`blur`, `bloom` of the layer beneath), a pointwise effect reading `child`.
import {defineShader, p, child, blur, bloom, wgsl} from 'shaders/std'

export default defineShader({
    name: 'DreamyFocus',
    description: 'Sharp in a band, soft and glowing outside it',
    props: {
        softness: {default: 40, ui: {type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Softness'}},
        focus: {default: 0.5, ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Focus height'}},
        glow: {default: 0.6, ui: {type: 'range', min: 0, max: 2, step: 0.01, label: 'Glow'}},
    },
    inputs: {
        soft: blur(child, {radius: p('softness')}),
        halo: bloom(child, {threshold: 0.7, radius: 40}),
    },
    effect: wgsl`
        let band = smoothstep(0.05, 0.3, abs(uv.y - focus));
        let base = mix(child, soft(uv), band);
        return vec4f(base.rgb + halo(uv).rgb * glow, base.a);
    `,
})
