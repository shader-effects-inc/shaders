// A 2D shape written as a signed distance. Features: the `shape` slot (fill, stroke, softness,
// position, rotation and the bounding box come from the engine) and a library SDF.
import {defineShader, wgsl} from 'shaders/std'
import {sdRoundBox} from 'shaders/std/wgsl'

export default defineShader({
    name: 'Pill',
    description: 'A rounded pill shape',
    props: {
        width: {default: 0.4, ui: {type: 'range', min: 0.05, max: 1, step: 0.01, label: 'Width', group: 'Shape'}},
        height: {default: 0.15, ui: {type: 'range', min: 0.05, max: 1, step: 0.01, label: 'Height', group: 'Shape'}},
    },
    inputs: {sdRoundBox},
    shape: wgsl`return sdRoundBox(p, vec2f(width, height), height);`,
    bounds: {size: {width: 'width', height: 'height'}},
})
