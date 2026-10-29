/**
 * std/graph — worked examples, pinned by the resolve gate (stdGraph.test.ts). The ocean shows
 * the house method: coordinates → height field → normal → light → colour → compose. Keep them
 * honest: every op here must exist in the catalog with these exact inputs.
 */
import type {StdGraphEnvelope} from '@coreroot/std/graph/types'

/** A photoreal-leaning ocean: perspective-warped fractal height field, lit by a key light. */
export const OCEAN_EXAMPLE: StdGraphEnvelope = {
    kind: 'std-graph',
    name: 'Ocean Surface',
    description: 'A rolling ocean seen from the shore: perspective waves, sun glints, fresnel sky reflection, foam on the crests.',
    props: {
        skyColor: {kind: 'color', default: '#5c8fc9', label: 'Sky'},
        waterColor: {kind: 'color', default: '#0b3d5c', label: 'Water'},
        sunColor: {kind: 'color', default: '#ffe9c4', label: 'Sun'},
        horizon: {kind: 'float', default: 0.42, min: 0.2, max: 0.8, label: 'Horizon'},
        waveHeight: {kind: 'float', default: 0.6, min: 0, max: 2, label: 'Wave Height'},
        choppiness: {kind: 'float', default: 1.6, min: 0.5, max: 6, label: 'Choppiness'},
        sunAngle: {kind: 'angle', default: 250, label: 'Sun Angle'},
    },
    nodes: [
        // Where we are relative to the horizon line (y = 0 is the top of the frame).
        {id: 'd', op: 'sub', args: {a: 'uv.y', b: '@horizon'}},
        {id: 'depth', op: 'add', args: {a: '$d', b: 0.03}},
        {id: 'isWater', op: 'smoothstep', args: {e0: -0.003, e1: 0.003, x: '$d'}},
        // Perspective: divide screen offsets by distance below the horizon so waves shrink toward it.
        {id: 'cx', op: 'sub', args: {a: 'uv.x', b: 0.5}},
        {id: 'cxa', op: 'mul', args: {a: '$cx', b: 'aspect'}},
        {id: 'wx', op: 'div', args: {a: '$cxa', b: '$depth'}},
        {id: 'wz', op: 'div', args: {a: 1, b: '$depth'}},
        {id: 'travel', op: 'mul', args: {a: 'time', b: 0.7}},
        {id: 'wzT', op: 'add', args: {a: '$wz', b: '$travel'}},
        {id: 'q', op: 'vec2', args: {x: '$wx', y: '$wzT'}},
        {id: 'qs', op: 'mul', args: {a: '$q', b: '@choppiness'}},
        // The water height as a FIELD so we can take its slope for a normal.
        {
            id: 'height', op: 'fieldOf', body: [
                {id: 'big', op: 'fbm2', args: {p: 'p', octaves: 5, gain: 0.55}},
                {id: 'p2', op: 'mul', args: {a: 'p', b: 3.7}},
                {id: 'ripple', op: 'noise2', args: {p: '$p2'}},
                {id: 'rippleS', op: 'mul', args: {a: '$ripple', b: 0.06}},
                {id: 'h', op: 'add', args: {a: '$big', b: '$rippleS'}},
            ], result: '$h',
        },
        {id: 'strength', op: 'mul', args: {a: '@waveHeight', b: 1.6}},
        {id: 'n', op: 'normalFrom', args: {field: '$height', at: '$qs', eps: 0.03, strength: '$strength'}},
        {id: 'h', op: 'sample', args: {field: '$height', at: '$qs'}},
        // Light: a low sun.
        {id: 'sun', op: 'keyLight', args: {angle: '@sunAngle', elevation: 0.35}},
        {id: 'diffuse', op: 'lambert', args: {normal: '$n', light: '$sun.L', wrap: 0.6}},
        {id: 'spec', op: 'shine', args: {normal: '$n', light: '$sun.L', gloss: 90}},
        {id: 'g', op: 'grazing', args: {normal: '$n'}},
        {id: 'fr', op: 'fresnel', args: {grazing: '$g', r0: 0.03, gain: 0.9, power: 4}},
        // Colour: deep water lifted by diffuse, sky reflected at grazing angles, sun glints on top.
        {id: 'lit', op: 'add', args: {a: 0.3, b: '$diffuse'}},
        {id: 'litS', op: 'mul', args: {a: '$lit', b: 0.7}},
        {id: 'body', op: 'mul', args: {a: '@waterColor.rgb', b: '$litS'}},
        {id: 'reflect', op: 'mix', args: {a: '$body', b: '@skyColor.rgb', t: '$fr'}},
        {id: 'glintRgb', op: 'mul', args: {a: '@sunColor.rgb', b: '$spec'}},
        {id: 'glint', op: 'mul', args: {a: '$glintRgb', b: 1.4}},
        {id: 'water0', op: 'add', args: {a: '$reflect', b: '$glint'}},
        // Foam on the crests, fading with distance.
        {id: 'crest', op: 'smoothstep', args: {e0: 0.42, e1: 0.7, x: '$h'}},
        {id: 'near', op: 'smoothstep', args: {e0: 0.02, e1: 0.25, x: '$d'}},
        {id: 'foam', op: 'mul', args: {a: '$crest', b: '$near'}},
        {id: 'foamAmt', op: 'mul', args: {a: '$foam', b: 0.8}},
        {id: 'foamRgb', op: 'splat3', args: {x: 0.92}},
        {id: 'water1', op: 'mix', args: {a: '$water0', b: '$foamRgb', t: '$foamAmt'}},
        // Atmosphere: water fades into the horizon haze.
        {id: 'haze', op: 'smoothstep', args: {e0: 0.12, e1: 0, x: '$d'}},
        {id: 'hazeRgb', op: 'mul', args: {a: '@skyColor.rgb', b: 1.15}},
        {id: 'water', op: 'mix', args: {a: '$water1', b: '$hazeRgb', t: '$haze'}},
        // Sky: darker overhead, brighter at the horizon.
        {id: 'skyT', op: 'div', args: {a: 'uv.y', b: '@horizon'}},
        {id: 'skyTc', op: 'saturate', args: {x: '$skyT'}},
        {id: 'skyTop', op: 'mul', args: {a: '@skyColor.rgb', b: 0.55}},
        {id: 'skyBottom', op: 'add', args: {a: '$hazeRgb', b: 0.05}},
        {id: 'sky', op: 'mix', args: {a: '$skyTop', b: '$skyBottom', t: '$skyTc'}},
        // Compose and tone.
        {id: 'rgb', op: 'mix', args: {a: '$sky', b: '$water', t: '$isWater'}},
        {id: 'toned', op: 'exposureTone', args: {rgb: '$rgb'}},
        {id: 'out', op: 'rgba', args: {rgb: '$toned', a: 1}},
    ],
    output: '$out',
}

/** A small second example: a pulsing orb with a starfield behind it (repeat + masks + light). */
export const ORB_EXAMPLE: StdGraphEnvelope = {
    kind: 'std-graph',
    name: 'Pulsing Orb',
    props: {
        center: {kind: 'position', default: {x: 0.5, y: 0.5}},
        glow: {kind: 'color', default: '#ff7a3d'},
        core: {kind: 'color', default: '#fff3d6'},
        radius: {kind: 'float', default: 0.22, min: 0.05, max: 0.6},
    },
    nodes: [
        {id: 'c', op: 'centered', args: {center: '@center'}},
        {id: 'pulse', op: 'oscillate', args: {t: 'time', rate: 0.5, min: 0.9, max: 1.1}},
        {id: 'r', op: 'mul', args: {a: '@radius', b: '$pulse'}},
        {id: 'disc', op: 'softDisc', args: {dist: '$c.dist', radius: '$r', width: 0.01, curve: 1}},
        {id: 'n', op: 'domeNormal', args: {delta: '$c.delta', dist: '$c.dist', radius: '$r'}},
        {id: 'light', op: 'keyLight', args: {angle: 225, elevation: 0.7}},
        {id: 'spec', op: 'shine', args: {normal: '$n', light: '$light.L', gloss: 40}},
        {id: 'diff', op: 'lambert', args: {normal: '$n', light: '$light.L', wrap: 0.4}},
        {id: 'halo', op: 'emitterFalloff', args: {d: '$c.dist', radius: '$r', power: 2}},
        {id: 'haloRgb', op: 'mul', args: {a: '@glow.rgb', b: '$halo'}},
        {id: 'bodyRgb', op: 'mix', args: {a: '@glow.rgb', b: '@core.rgb', t: '$diff'}},
        {id: 'specRgb', op: 'splat3', args: {x: '$spec'}},
        {id: 'lit', op: 'add', args: {a: '$bodyRgb', b: '$specRgb'}},
        {id: 'stars', op: 'repeat', args: {count: 3, reduce: 'add'}, body: [
            {id: 'scale', op: 'add', args: {a: 12, b: 'i'}},
            {id: 'sp', op: 'aspectUV'},
            {id: 'spScaled', op: 'mul', args: {a: '$sp', b: '$scale'}},
            {id: 'st', op: 'pointStars', args: {p: '$spScaled', density: 0.25, size: 0.03, time: 'time', twinkleRate: 1.5}},
        ], result: '$st.glint'},
        {id: 'starsRgb', op: 'splat3', args: {x: '$stars'}},
        {id: 'scene', op: 'layers', args: {items: [{paint: '$starsRgb', opacity: 0.6}, {paint: '$haloRgb', opacity: 0.8}]}},
        {id: 'sceneAlpha', op: 'emissiveAlpha', args: {rgb: '$scene'}},
        {id: 'body', op: 'rgba', args: {rgb: '$lit', a: '$disc'}},
        {id: 'out', op: 'over', args: {top: '$body', bottom: '$sceneAlpha'}},
    ],
    output: '$out',
}

/** Volumetric cloud bank over a sky: `march` through a turbulent medium, typed `fieldOf` sky sampled twice. */
export const CLOUDS_EXAMPLE: StdGraphEnvelope = {
    kind: 'std-graph',
    name: 'Cloud Bank',
    props: {
        skyColor: {kind: 'color', default: '#3e6fb5'},
        cloudColor: {kind: 'color', default: '#fff1e0'},
        shadowColor: {kind: 'color', default: '#6d7fa3'},
        coverage: {kind: 'float', default: 0.5, min: 0, max: 1},
    },
    nodes: [
        // A reusable sky colour function (vec3 fieldOf), sampled twice below.
        {id: 'sky', op: 'fieldOf', body: [
            {id: 'top', op: 'mul', args: {a: '@skyColor.rgb', b: 0.6}},
            {id: 'bot', op: 'add', args: {a: '@skyColor.rgb', b: 0.25}},
            {id: 'c', op: 'mix', args: {a: '$top', b: '$bot', t: 'p.y'}},
        ], result: '$c'},
        {id: 'skyHere', op: 'sample', args: {field: '$sky', at: 'uv'}},
        {id: 'flipUv', op: 'vec2', args: {x: 'uv.x', y: 'uv.y'}},
        {id: 'skyAgain', op: 'sample', args: {field: '$sky', at: '$flipUv'}},
        {id: 'base', op: 'mix', args: {a: '$skyHere', b: '$skyAgain', t: 0.5}},
        // The march: a slab of turbulent medium, lit warm where thin and cool where dense.
        {id: 'auv', op: 'aspectUV'},
        {id: 'drift', op: 'mul', args: {a: 'time', b: 0.15}},
        {id: 'cov', op: 'mul', args: {a: '@coverage', b: 0.5}},
        {id: 'thresh', op: 'sub', args: {a: 0.45, b: '$cov'}},
        {id: 'px', op: 'mul', args: {a: 'uv', b: 'viewport'}},
        {id: 'jit', op: 'hash', args: {p: '$px'}},
        {id: 'jitter', op: 'mul', args: {a: '$jit', b: 0.1}},
        {id: 'cloud', op: 'march', args: {steps: 10, entry: 0, length: 1, jitter: '$jitter', emissionGain: 0.6}, body: [
            {id: 'q', op: 'xyPlusZ', args: {xy: '$auv', z: 'z'}},
            {id: 'qs', op: 'mul', args: {a: '$q', b: 1.6}},
            {id: 'med', op: 'turbulence', args: {p: '$qs', frequency: 1.2, drift: '$drift', billow: 0.35}},
            {id: 'dens0', op: 'sub', args: {a: '$med.density', b: '$thresh'}},
            {id: 'dens', op: 'smoothstep', args: {e0: 0, e1: 0.35, x: '$dens0'}},
            {id: 'lit', op: 'mix', args: {a: '@shadowColor.rgb', b: '@cloudColor.rgb', t: 't'}},
            {id: 'emit', op: 'mul', args: {a: '$lit', b: '$dens'}},
            {id: 'absorb', op: 'mul', args: {a: '$dens', b: 2.2}},
        ], result: '$emit', absorb: '$absorb'},
        {id: 'behind', op: 'mul', args: {a: '$base', b: '$cloud.transmittance'}},
        {id: 'rgb', op: 'add', args: {a: '$behind', b: '$cloud.color'}},
        {id: 'toned', op: 'exposureTone', args: {rgb: '$rgb'}},
        {id: 'out', op: 'rgba', args: {rgb: '$toned', a: 1}},
    ],
    output: '$out',
}
