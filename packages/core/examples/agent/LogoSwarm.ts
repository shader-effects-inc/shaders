// Thousands of particles hold the shape of a word, scatter from the pointer and drift back.
// Features: `text` as an input, `samplePoints` to spawn particles inside it, forces as data.
import {defineShader, p, samplePoints, simulate, text, wgsl} from 'shaders/std'

const word = text(p('label'), {font: '800 Inter, system-ui, sans-serif', size: 0.28})

export default defineShader({
    name: 'LogoSwarm',
    description: 'Particles hold the shape of a word, scatter from the pointer and drift back',
    props: {
        label: {default: 'SHADERS', ui: {type: 'text', label: 'Text'}},
        color: {default: '#9ad1ff', ui: {type: 'color', label: 'Color'}},
    },
    inputs: {
        word,
        dust: simulate.particles({
            count: 30000,
            mobile: {max: 10000},
            spawn: samplePoints(word),
            forces: [{spring: 3}, {cursor: 1.5, radius: 0.12}, {drag: 0.7}],
            draw: wgsl`return Point(p.pos, 2.0, vec4f(color.rgb, 1.0));`,
        }),
    },
    paint: wgsl`
        let d = dust(uv);
        return vec4f(d.rgb, min(d.a, 1.0));
    `,
})
