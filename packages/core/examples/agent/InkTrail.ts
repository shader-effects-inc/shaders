// The pointer leaves a fading ink trail. Features: `previousFrame` (this layer's own last frame).
import {defineShader, previousFrame, wgsl} from 'shaders/std'

export default defineShader({
    name: 'InkTrail',
    description: 'The pointer draws a soft ink trail that fades away',
    props: {
        ink: {default: '#ff5a1f', ui: {type: 'color', label: 'Ink'}},
        fade: {default: 0.96, ui: {type: 'range', min: 0.8, max: 0.999, step: 0.001, label: 'Persistence'}},
        size: {default: 0.03, ui: {type: 'range', min: 0.005, max: 0.1, step: 0.001, label: 'Brush'}},
    },
    inputs: {last: previousFrame()},
    paint: wgsl`
        let d = length((uv - pointer) * vec2f(aspect, 1.0));
        let stroke = (1.0 - smoothstep(size * 0.5, size, d)) * min(length(pointerVelocity) * 4.0, 1.0);
        let before = last(uv) * fade;
        return max(before, vec4f(ink.rgb, 1.0) * stroke);
    `,
})
