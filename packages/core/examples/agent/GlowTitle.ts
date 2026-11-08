// A title with a soft neon glow. Features: `text` as an input, and an input of an input
// (`blur` of the text).
import {defineShader, blur, p, text, wgsl} from 'shaders/std'

const title = text(p('label'), {font: '700 Inter, system-ui, sans-serif', size: 0.22})

export default defineShader({
    name: 'GlowTitle',
    description: 'A title with a soft neon glow',
    props: {
        label: {default: 'Hello', ui: {type: 'text', label: 'Text'}},
        color: {default: '#ff3d9a', ui: {type: 'color', label: 'Color'}},
        glow: {default: 24, ui: {type: 'range', min: 0, max: 80, step: 1, label: 'Glow'}},
    },
    inputs: {
        title,
        halo: blur(title, {radius: p('glow')}),
    },
    paint: wgsl`
        let core = title(uv).a;
        let soft = halo(uv).a;
        return vec4f(color.rgb * (soft * 1.5 + core), clamp(core + soft, 0.0, 1.0));
    `,
})
