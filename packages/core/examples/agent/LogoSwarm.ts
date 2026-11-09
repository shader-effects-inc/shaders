// Thousands of particles hold the shape of a word, scatter from the pointer and drift back.
// Features: `text` as an input (it brings the Text layer's typography controls, an ink-hugging
// bounding box and double-click editing), `samplePoints` to spawn particles inside it, and forces
// driven by props.
import {defineShader, p, samplePoints, simulate, text, wgsl} from 'shaders/std'

const word = text(p('label'), {font: '800 Inter', size: 0.28})

export default defineShader({
    name: 'LogoSwarm',
    description: 'Particles hold the shape of a word, scatter from the pointer and drift back',
    animatedTime: {speed: 'speed'},
    props: {
        label: {default: 'SHADERS', ui: {type: 'text', label: 'Text', group: 'Content'}},
        color: {default: '#9ad1ff', ui: {type: 'color', label: 'Color', group: 'Colors'}},
        pointSize: {default: 3, description: 'Particle size, in pixels at 1080p', ui: {type: 'range', min: 1, max: 6, step: 0.1, label: 'Point size', group: 'Particles'}},
        brightness: {default: 2.5, description: 'How brightly each particle glows; overlapping particles add up', ui: {type: 'range', min: 0.5, max: 6, step: 0.01, label: 'Brightness', group: 'Particles'}},
        count: {default: 30000, description: 'How many particles make up the word', ui: {type: 'range', min: 5000, max: 60000, step: 1000, label: 'Particles', group: 'Particles'}},
        speed: {default: 1, description: 'How fast everything moves (0 freezes the swarm)', ui: {type: 'range', min: 0, max: 3, step: 0.01, label: 'Speed', group: 'Motion'}},
        pull: {default: 3, description: 'How strongly each particle is pulled back to its place in the word', ui: {type: 'range', min: 0.5, max: 10, step: 0.1, label: 'Pull back', group: 'Motion'}},
        settling: {default: 0.7, description: 'How quickly motion dies away: low overshoots and wobbles, high glides straight home', ui: {type: 'range', min: 0, max: 3, step: 0.01, label: 'Settling', group: 'Motion'}},
        wander: {default: 0.02, description: 'Random jiggle on every particle, so the word shimmers instead of sitting still', ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Wander', group: 'Motion'}},
        pushStrength: {default: 1.5, description: 'How hard the pointer pushes particles away', ui: {type: 'range', min: 0, max: 5, step: 0.01, label: 'Push strength', group: 'Pointer'}},
        pushRadius: {default: 0.12, description: 'How far the pointer reaches, as a fraction of the frame', ui: {type: 'range', min: 0.02, max: 0.4, step: 0.005, label: 'Push radius', group: 'Pointer'}},
    },
    inputs: {
        word,
        dust: simulate.particles({
            count: p('count'),
            max: 60000,
            mobile: {max: 10000},
            spawn: samplePoints(word),
            forces: [
                {spring: p('pull')},
                {cursor: p('pushStrength'), radius: p('pushRadius')},
                {wander: p('wander')},
                {drag: p('settling')},
            ],
            // Sized against the particle texture, so the swarm looks the same at any frame size.
            draw: wgsl`return Point(p.pos, max(pointSize * targetSize.y / 1080.0, 1.0), vec4f(color.rgb * brightness, 1.0));`,
        }),
    },
    paint: wgsl`
        let d = dust(uv);
        return vec4f(d.rgb, min(d.a, 1.0));
    `,
})
