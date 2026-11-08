// A cloth curtain over the layers beneath, billowing in a breeze and pushed by the pointer.
// Features: a grid simulation with Verlet motion (readPrev is last frame) and Jacobi
// constraint iterations, drawn as a sheet with `rasterize`, shading the layer beneath.
import {defineShader, op, rasterize, simulate, wgsl} from 'shaders/std'
import {noise3, fresnel} from 'shaders/std/wgsl'

// Each cell: .xy position in uv, .z depth (0 near, 1 far), .w 1 when pinned (top row).
const cloth = simulate.grid({
    resolution: [48, 36],
    history: 2,
    precision: 'full',
    init: wgsl`return vec4f(0.1 + cellUV.x * 0.8, 0.02 + cellUV.y * 0.9, 0.5, select(0.0, 1.0, cell.y == 0));`,
    step: [
        op.kernel(wgsl`
            let c = read(cell);
            if (c.w > 0.5) { return c; }
            let before = readPrev(cell);
            let breeze = wind * noise3(vec3f(c.xy * 3.0, time * 0.3));
            var push = vec3f(0.0);
            let toPointer = c.xy - pointer;
            let d = length(toPointer * vec2f(aspect, 1.0));
            if (d < 0.12) { push = vec3f(normalize(toPointer + vec2f(1e-5)) * 0.4, 0.6) * (1.0 - d / 0.12) * length(pointerVelocity); }
            let accel = vec3f(breeze * 0.3, 0.9, breeze) + push * 40.0;
            let next = c.xyz + (c.xyz - before.xyz) * 0.985 + accel * dt * dt;
            return vec4f(next, c.w);
        `),
        op.kernel(wgsl`
            let c = read(cell);
            if (c.w > 0.5) { return c; }
            let restX = 0.8 / f32(gridSize.x - 1);
            let restY = 0.9 / f32(gridSize.y - 1);
            var shift = vec3f(0.0);
            var n = 0.0;
            var dirs = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
            for (var k = 0; k < 4; k++) {
                let o = dirs[k];
                let other = cell + o;
                if (other.x < 0 || other.y < 0 || other.x >= gridSize.x || other.y >= gridSize.y) { continue; }
                let rest = select(restY, restX, o.y == 0);
                let delta = read(other).xyz - c.xyz;
                let len = max(length(delta), 1e-5);
                shift += delta * (1.0 - rest / len) * 0.5;
                n += 1.0;
            }
            return vec4f(c.xyz + shift / max(n, 1.0), c.w);
        `, {iterations: 12}),
    ],
})

export default defineShader({
    name: 'Curtain',
    description: 'A cloth curtain over the layers beneath, billowing in a breeze and pushed by the pointer',
    props: {
        wind: {default: 0.4, ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Wind'}},
        sheen: {default: 0.3, ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Sheen'}},
    },
    inputs: {
        noise3, fresnel, cloth,
        sheet: rasterize(cloth),
    },
    effect: wgsl`
        let s = sheet(uv);
        if (s.w < 0.5) { return vec4f(0.0); }
        let n = sheet.normal(uv).xyz;
        let base = textureSampleLevel(childTexture, childSampler, s.xy, 0.0);
        let lit = 0.55 + 0.45 * max(dot(n, normalize(vec3f(-0.3, 0.5, 0.8))), 0.0);
        let rim = fresnel(abs(n.z), 0.04) * sheen;
        return vec4f(base.rgb / max(base.a, 0.0001) * lit + rim, 1.0);
    `,
})
