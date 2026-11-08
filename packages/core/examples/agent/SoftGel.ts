// A soft gel material on whichever shape the user picks (flat, SVG or 3D). Features: the
// `surface` slot with `shapeDistance`, `shapeSlope` and the library's anti-aliasing.
import {defineShader, wgsl} from 'shaders/std'
import {aaFill, fresnel, normalFromHeight} from 'shaders/std/wgsl'

export default defineShader({
    name: 'SoftGel',
    description: 'A soft, glossy gel on any shape',
    props: {
        tint: {default: '#7aa8ff', ui: {type: 'color', label: 'Tint'}},
        gloss: {default: 0.6, ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Gloss'}},
    },
    inputs: {aaFill, fresnel, normalFromHeight},
    surface: wgsl`
        let n = normalFromHeight(-shapeSlope, 0.6);
        let light = normalize(vec3f(-0.4, 0.6, 0.7));
        let diffuse = 0.45 + 0.55 * max(dot(n, light), 0.0);
        let spec = pow(max(dot(reflect(-light, n), vec3f(0.0, 0.0, 1.0)), 0.0), 40.0) * gloss;
        let rim = fresnel(n.z, 0.04) * 0.5;
        let rgb = tint.rgb * diffuse + vec3f(spec + rim);
        return vec4f(rgb, aaFill(shapeDistance));
    `,
})
