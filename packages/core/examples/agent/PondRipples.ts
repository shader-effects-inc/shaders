// The pointer stirs a pond and the layers beneath ripple. Features: `simulate.grid` with built-in
// steps read as an input, a derived output (`field.slope`), a gather effect.
import {defineShader, op, p, pointer, pointerSpeed, simulate, wgsl} from 'shaders/std'

export default defineShader({
    name: 'PondRipples',
    description: 'The pointer stirs ripples that bend the layers beneath',
    props: {
        decay: {default: 6, ui: {type: 'range', min: 0, max: 20, step: 0.1, label: 'Decay'}},
        radius: {default: 0.5, ui: {type: 'range', min: 0.1, max: 1, step: 0.1, label: 'Brush'}},
        intensity: {default: 0.04, ui: {type: 'range', min: 0, max: 0.2, step: 0.001, label: 'Intensity'}},
    },
    inputs: {
        field: simulate.grid({
            resolution: 192,
            history: 2,
            step: [op.wave({damping: p('decay')}), op.splat({at: pointer(), amount: pointerSpeed({max: 2}), radius: p('radius')})],
            derive: {slope: op.gradient()},
        }),
    },
    effect: wgsl`
        let s = field.slope(uv).xy;
        return textureSampleLevel(childTexture, childSampler, uv + s * intensity, 0.0);
    `,
})
