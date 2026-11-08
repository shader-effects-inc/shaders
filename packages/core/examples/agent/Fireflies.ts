// Fireflies that wander, blink and drift toward the pointer. Features: the raw compute tier: a
// `struct`, a `buffer` of them, a seeding `kernel` (once) and a moving `kernel` (every frame), and a
// half-resolution `pass` that reads the buffer to draw them. (`simulate.particles` is the usual way
// to draw many points; this shows the raw pieces for anything it does not cover.)
import {buffer, defineShader, kernel, pass, struct, wgsl} from 'shaders/std'
import {hash22, noise2} from 'shaders/std/wgsl'

const Fly = struct('Fly', {pos: 'vec2f', vel: 'vec2f', glow: 'f32'})

export default defineShader({
    name: 'Fireflies',
    description: 'Fireflies wander, blink, and drift toward the pointer',
    props: {
        pull: {default: 0.5, ui: {type: 'range', min: 0, max: 2, step: 0.01, label: 'Pull'}},
    },
    inputs: {
        hash22, noise2,
        flies: buffer(Fly, 64),
        seed: kernel(wgsl`
            let r = hash22(vec2f(f32(index), 3.7));
            flies[index] = Fly(r, vec2f(0.0), hash22(r).x);
        `, {over: 'flies', when: 'init'}),
        move: kernel(wgsl`
            var f = flies[index];
            let wander = vec2f(noise2(f.pos * 3.0 + time * 0.2), noise2(f.pos * 3.0 - time * 0.2));
            f.vel = f.vel * 0.96 + (wander * 0.3 + (pointer - f.pos) * pull * 0.2) * dt;
            f.pos = clamp(f.pos + f.vel * dt, vec2f(0.0), vec2f(1.0));
            f.glow = 0.5 + 0.5 * sin(time * 3.0 + f32(index));
            flies[index] = f;
        `, {over: 'flies'}),
        lights: pass(wgsl`
            var c = 0.0;
            for (var i = 0u; i < 64u; i++) {
                let f = flies[i];
                let d = length((uv - f.pos) * vec2f(aspect, 1.0));
                c += f.glow * 0.0004 / (d * d + 0.0004);
            }
            return vec4f(vec3f(1.0, 0.85, 0.4) * c, 1.0);
        `, {scale: 0.5}),
    },
    paint: wgsl`return lights(uv);`,
})
