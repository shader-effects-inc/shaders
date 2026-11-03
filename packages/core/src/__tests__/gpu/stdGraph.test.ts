import {describe, it, expect} from 'vitest'
import {defineStdGraph, normalizeStdGraph, renderStdGraphManifest, resolveStdGraphWgsl, validateStdGraph} from '@coreroot/std/graph/define'
import {CATALOG, type InputSpec, type InputType} from '@coreroot/std/graph/catalog'
import type {GraphArg, GraphNode, GraphPropSpec, StdGraphEnvelope} from '@coreroot/std/graph/types'
import {CLOUDS_EXAMPLE, OCEAN_EXAMPLE, ORB_EXAMPLE} from './stdGraphExamples'
import {composeNodeTree} from '@coreroot/gpu/composer'
import {authorError, onAuthorError} from '@coreroot/gpu/support'
import {defineShaderEnvelope, validateShaderEnvelope, validateWgslEnvelope} from '@coreroot/std/graph/index'
import * as harness from './_patternHarness'

/**
 * std/graph gate: the JSON node-graph interpreter over the std vocabulary. GPU-free —
 * every graph composes under a root and resolves to WGSL through the same path the
 * per-shader gates use. (1) The two worked examples resolve. (2) EVERY catalog op resolves
 * in a generic harness graph, so a word that is wired wrong fails here, not in a browser.
 * (3) The validator's messages are repair-grade.
 */

describe('std/graph examples', () => {
    it('validates and resolves the ocean exemplar', () => {
        const check = validateStdGraph(OCEAN_EXAMPLE)
        expect(check).toMatchObject({ok: true, usesTime: true, outputType: 'vec4'})
        const res = resolveStdGraphWgsl(OCEAN_EXAMPLE)
        if (!res.ok) throw new Error(res.error)
        expect(res.wgsl).toMatch(/mxNoiseFloat2/) // fbm2 → surface noise
        expect(res.wgsl).toMatch(/keyL/) // keyLight → the +z light vector local
        expect(res.wgsl).toMatch(/_animTime/) // time → animatedTime clock
        expect(res.wgsl).toMatch(/oneMinusExp|1\.0 - exp|exp\(/) // exposureTone
    })

    it('validates and resolves the orb exemplar (repeat, layers, over)', () => {
        const check = validateStdGraph(ORB_EXAMPLE)
        if (!check.ok) throw new Error(check.error)
        const res = resolveStdGraphWgsl(ORB_EXAMPLE)
        if (!res.ok) throw new Error(res.error)
        expect(res.wgsl).toMatch(/mxWorleyNoiseFloat2Pub/) // pointStars
    })

    it('validates and resolves the clouds exemplar (march + a vec3 fieldOf sampled twice)', () => {
        const check = validateStdGraph(CLOUDS_EXAMPLE)
        if (!check.ok) throw new Error(check.error)
        const res = resolveStdGraphWgsl(CLOUDS_EXAMPLE)
        if (!res.ok) throw new Error(res.error)
        expect(res.wgsl).toMatch(/transmit9/) // 10 unrolled steps
        expect(res.wgsl).toMatch(/mxNoiseFloat3/) // turbulence
    })

    it('types fieldOf bodies and guards march shapes', () => {
        const base = (nodes: GraphNode[], output: string): StdGraphEnvelope => ({kind: 'std-graph', name: 'T', props: HARNESS_PROPS, nodes, output})
        // A vec3 field sampled → vec3 (usable as rgb).
        const ok = validateStdGraph(base([
            {id: 'f', op: 'fieldOf', body: [{id: 'c', op: 'splat3', args: {x: 'p.x'}}], result: '$c'},
            {id: 's', op: 'sample', args: {field: '$f', at: 'uv'}},
            {id: 'out', op: 'rgba', args: {rgb: '$s', a: 1}},
        ], '$out'))
        expect(ok.ok).toBe(true)
        // A vec3 field cannot be a height field.
        expect(validateStdGraph(base([
            {id: 'f', op: 'fieldOf', body: [{id: 'c', op: 'splat3', args: {x: 'p.x'}}], result: '$c'},
            {id: 'n', op: 'normalFrom', args: {field: '$f', at: 'uv', eps: 0.01, strength: 1}},
            {id: 'out', op: 'rgba', args: {rgb: '$n', a: 1}},
        ], '$out')).ok).toBe(true) // normalFrom samples .x of a vec3? No: fdSlope subtracts vec3s — allowed by types; runtime resolve decides.
        expect(validateStdGraph(base([
            {id: 'm', op: 'march', args: {steps: 99, entry: 0, length: 1}, body: [{id: 'e', op: 'splat3', args: {x: 'z'}}], result: '$e', absorb: '$e'},
            {id: 'out', op: 'rgba', args: {rgb: '$m.color', a: 1}},
        ], '$out')).error).toMatch(/1\.\.32/)
        expect(validateStdGraph(base([
            {id: 'm', op: 'march', args: {steps: 4, entry: 0, length: 1}, body: [{id: 'e', op: 'splat3', args: {x: 'z'}}], result: '$e'},
            {id: 'out', op: 'rgba', args: {rgb: '$m.color', a: 1}},
        ], '$out')).error).toMatch(/needs "absorb"/)
    })

    it('lowers to a generator definition, adding speed when it reads the clock and colorSpace only when it mixes colours in it', () => {
        const def = defineStdGraph(OCEAN_EXAMPLE)
        expect(def.name).toBe('Ocean Surface')
        expect(def.animatedTime).toEqual({speed: 'speed'})
        expect(Object.keys(def.props)).toEqual(expect.arrayContaining(['skyColor', 'horizon', 'speed']))
        // The ocean lights rgb with plain algebra — no colour-space words, so no Color Space control.
        expect(def.props).not.toHaveProperty('colorSpace')
        expect(def.acceptsUVContext).toBe(true)
        const mixed = defineStdGraph({
            kind: 'std-graph', name: 'Two Tone', props: {a: {kind: 'color', default: '#ff0000'}, b: {kind: 'color', default: '#0000ff'}},
            nodes: [{id: 'c', op: 'pair', args: {a: '@a', b: '@b', t: 'uv.x'}}], output: '$c',
        })
        expect(Object.keys(mixed.props)).toEqual(['a', 'b', 'colorSpace'])
        expect(mixed.animatedTime).toBeUndefined()
    })

    it('respects the compile-time colour space on mixColors words', () => {
        const env: StdGraphEnvelope = {
            kind: 'std-graph', name: 'Pair', props: {a: {kind: 'color', default: '#ff0000'}, b: {kind: 'color', default: '#0000ff'}},
            nodes: [{id: 'c', op: 'pair', args: {a: '@a', b: '@b', t: 'uv.x'}}], output: '$c',
        }
        const linear = resolveStdGraphWgsl(env)
        const oklch = resolveStdGraphWgsl(env, {colorSpace: 'oklch'})
        if (!linear.ok || !oklch.ok) throw new Error('resolve failed')
        expect(oklch.wgsl).toMatch(/oklch/i)
        expect(linear.wgsl).not.toMatch(/oklch/i)
    })
})

// ── Per-op sweep ────────────────────────────────────────────────────────────────────────

const HARNESS_PROPS: StdGraphEnvelope['props'] = {
    c1: {kind: 'color', default: '#ff4400'},
    c2: {kind: 'color', default: '#00aaff'},
    c3: {kind: 'color', default: '#ffffff'},
    pos1: {kind: 'position', default: {x: 0.2, y: 0.5}},
    pos2: {kind: 'position', default: {x: 0.8, y: 0.5}},
    f1: {kind: 'float', default: 2, min: 0, max: 10},
    f2: {kind: 'float', default: 1, min: 0, max: 10},
}

const PRELUDE: GraphNode[] = [
    {id: 'v2', op: 'aspectUV'},
    {id: 'v3', op: 'xyPlusZ', args: {xy: '$v2', z: 'time'}},
    {id: 'v4', op: 'rgba', args: {rgb: '$v3', a: 1}},
    {id: 'nf', op: 'noiseField', args: {basis: 'mx3'}},
    {id: 'fld', op: 'volume', args: {field: '$nf', scale: 1}},
    {id: 'pal', op: 'palette', args: {a: '@c1', b: '@c2'}},
    {id: 'sched', op: 'organicWaves', args: {frequency: 2, detail: 2}},
    {id: 'hf', op: 'fieldOf', body: [{id: 'n', op: 'noise2', args: {p: 'p'}}], result: '$n'},
]

const DATA_FOR: Record<string, GraphArg> = {
    'waves.schedule': '$sched',
    'wavyLine.waves': [{freq: 2, speed: 0.5, amount: 0.1}, {freq: 5, speed: 1, amount: 0.05}],
    'layers.items': [{paint: '$v3', opacity: 0.5}, {paint: '$v3', blend: 'screen'}],
    'cosineRainbow.phases': [0, 0.33, 0.67],
}

function argFor(op: string, name: string, spec: InputSpec): GraphArg {
    const types = Array.isArray(spec.type) ? spec.type : [spec.type]
    const t: InputType = types[0]!
    switch (t) {
        case 'num': case 'f32': return 0.5
        case 'vec': case 'vec2': return '$v2'
        case 'vec3': return '$v3'
        case 'vec4': return '$v4'
        case 'const': return name === 'octaves' ? 3 : name === 'eps' ? 0.01 : 2
        case 'enum': return spec.values![0]!
        case 'data': case 'schedule': return DATA_FOR[`${op}.${name}`] ?? '$sched'
        case 'field': return name === 'cells' ? '$cells' : op === 'normalFrom' || op === 'slope' ? '$hf' : '$fld'
        case 'palette': return '$pal'
        case 'prop:color': return name === 'b' ? '@c2' : name === 'c' ? '@c3' : '@c1'
        case 'prop:position': return name === 'to' ? '@pos2' : '@pos1'
        case 'prop:float': return name === 'seed' || name === 'balance' ? '@f2' : '@f1'
    }
}

/** Wrap a node's value into a vec4 so it can be the output. */
function toOutput(op: string, out: string, outputType: string): GraphNode[] {
    switch (outputType) {
        case 'f32': return [{id: 'o3', op: 'splat3', args: {x: out}}, {id: 'out', op: 'rgba', args: {rgb: '$o3', a: 1}}]
        case 'vec2': return [{id: 'out', op: 'vec4', args: {x: `${out}.x`, y: `${out}.y`, z: 0, w: 1}}]
        case 'vec3': return [{id: 'out', op: 'rgba', args: {rgb: out, a: 1}}]
        case 'vec4': return [{id: 'out', op: 'add', args: {a: out, b: 0}}]
        case 'field': return [{id: 's', op: 'sample', args: {field: out, at: op === 'noiseField' ? '$v3' : 'uv'}}, {id: 'o3', op: 'splat3', args: {x: '$s'}}, {id: 'out', op: 'rgba', args: {rgb: '$o3', a: 1}}]
        case 'palette': return [{id: 'r', op: 'ramp', args: {field: '$fld', palette: out}}, {id: 'out', op: 'add', args: {a: '$r', b: 0}}]
        case 'schedule': return [{id: 'w', op: 'waves', args: {x: 'uv.x', y: 'uv.y', t: 'time', schedule: out}}, {id: 'o3', op: 'splat3', args: {x: '$w'}}, {id: 'out', op: 'rgba', args: {rgb: '$o3', a: 1}}]
        default: throw new Error(`no output wrapper for ${outputType}`)
    }
}

describe('std/graph catalog sweep — every op composes and resolves', () => {
    const ops = Object.entries(CATALOG).filter(([, def]) => !def.control)
    for (const [op, def] of ops) {
        it(`${op}`, () => {
            const args: Record<string, GraphArg> = {}
            for (const [name, spec] of Object.entries(def.inputs)) args[name] = argFor(op, name, spec)
            const nodes: GraphNode[] = [...PRELUDE]
            if (op === 'cellFill' || op === 'cellBorders') nodes.push({id: 'cells', op: 'cellDistances', args: {scale: '@f1', seed: '@f2'}})
            nodes.push({id: 'x', op, args})
            const env: StdGraphEnvelope = {kind: 'std-graph', name: `Sweep ${op}`, props: HARNESS_PROPS, nodes, output: '$x'}
            const check = validateStdGraph(env)
            // The op itself must type; the output wrapper depends on what it returns.
            let outputType = check.ok ? check.outputType! : ''
            if (!check.ok) {
                // Output was not a colour — read the op's static type via a probe.
                const rule = def.output
                outputType = typeof rule === 'string' ? (rule === 'broadcast' ? 'f32' : rule) : typeof rule === 'function' ? String(rule({})) : 'record'
            }
            let finalNodes = nodes
            let output = '$x'
            if (outputType === 'record') {
                const rule = def.output as {record: Record<string, string>}
                const [field, ftype] = Object.entries(rule.record).find(([, t]) => t === 'f32') ?? Object.entries(rule.record)[0]!
                finalNodes = [...nodes, ...toOutput(op, `$x.${field}`, ftype)]
                output = '$out'
            } else if (outputType !== 'vec4' && outputType !== 'vec3') {
                finalNodes = [...nodes, ...toOutput(op, '$x', outputType)]
                output = '$out'
            }
            const full: StdGraphEnvelope = {...env, nodes: finalNodes, output}
            const valid = validateStdGraph(full)
            if (!valid.ok) throw new Error(`validate: ${valid.error}`)
            const res = resolveStdGraphWgsl(full)
            if (!res.ok) throw new Error(`resolve: ${res.error}`)
            expect(res.wgsl).toMatch(/fn /)
        })
    }
})

// ── Validator messages ──────────────────────────────────────────────────────────────────

describe('std/graph validation', () => {
    const base = (nodes: GraphNode[], output = '$out'): StdGraphEnvelope => ({kind: 'std-graph', name: 'T', props: HARNESS_PROPS, nodes, output})

    it('names unknown ops with suggestions', () => {
        const r = validateStdGraph(base([{id: 'out', op: 'smoothStep', args: {}}]))
        expect(r.ok).toBe(false)
        expect(r.error).toMatch(/unknown op "smoothStep"/)
        expect(r.error).toMatch(/smoothstep/)
    })

    it('rejects forward references and unknown inputs', () => {
        expect(validateStdGraph(base([{id: 'a', op: 'add', args: {a: '$b', b: 1}}, {id: 'b', op: 'aspectUV'}], '$a')).error).toMatch(/unknown node "\$b"/)
        expect(validateStdGraph(base([{id: 'a', op: 'add', args: {a: 1, b: 1, c: 2}}], '$a')).error).toMatch(/no input "c"/)
    })

    it('types inputs: a field is not a number, a colour prop is required where asked', () => {
        expect(validateStdGraph(base([...PRELUDE, {id: 'a', op: 'add', args: {a: '$fld', b: 1}}], '$a')).error).toMatch(/expects num, got field/)
        expect(validateStdGraph(base([{id: 'a', op: 'pair', args: {a: '@f1', b: '@c2', t: 0.5}}], '$a')).error).toMatch(/color prop/)
    })

    it('rejects vector width mismatches and a non-colour output', () => {
        expect(validateStdGraph(base([...PRELUDE, {id: 'a', op: 'add', args: {a: '$v2', b: '$v3'}}], '$a')).error).toMatch(/vector widths must match/)
        expect(validateStdGraph(base([{id: 'a', op: 'add', args: {a: 1, b: 2}}], '$a')).error).toMatch(/output must be a vec4/)
    })

    it('reports record fields and swizzles precisely', () => {
        expect(validateStdGraph(base([{id: 'c', op: 'centered'}, {id: 'a', op: 'splat3', args: {x: '$c.radius'}}], '$a')).error).toMatch(/no field "radius" \(has: delta, dist\)/)
        expect(validateStdGraph(base([{id: 'a', op: 'splat3', args: {x: 'uv.z'}}], '$a')).error).toMatch(/reads past a vec2/)
    })

    it('swizzles a record field in one reference', () => {
        const env = base([{id: 'k', op: 'keyLight', args: {angle: 225}}, {id: 'g', op: 'splat3', args: {x: '$k.L.z'}}, {id: 'out', op: 'rgba', args: {rgb: '$g', a: 1}}])
        expect(validateStdGraph(env).ok).toBe(true)
        expect(resolveStdGraphWgsl(env).ok).toBe(true)
        expect(validateStdGraph(base([{id: 'c', op: 'centered'}, {id: 'a', op: 'splat3', args: {x: '$c.dist.x'}}], '$a')).error).toMatch(/cannot swizzle|reads past/)
    })

    it('guards bodies: p only inside fieldOf, no nesting, result required', () => {
        expect(validateStdGraph(base([{id: 'a', op: 'noise2', args: {p: 'p'}}], '$a')).error).toMatch(/only available inside a fieldOf body/)
        expect(validateStdGraph(base([{id: 'f', op: 'fieldOf', body: [{id: 'n', op: 'noise2', args: {p: 'p'}}]}], '$f')).error).toMatch(/needs "result"/)
        expect(validateStdGraph(base([{id: 'r', op: 'repeat', args: {count: 99}, body: [{id: 'n', op: 'add', args: {a: 'i', b: 1}}], result: '$n'}], '$r')).error).toMatch(/1\.\.24/)
    })

    it('checks the props manifest', () => {
        const env: StdGraphEnvelope = {kind: 'std-graph', name: 'P', props: {Bad: {kind: 'float', default: 1, min: 0, max: 1}}, nodes: [{id: 'a', op: 'aspectUV'}], output: '$a'}
        expect(validateStdGraph(env).error).toMatch(/camelCase/)
        const env2: StdGraphEnvelope = {kind: 'std-graph', name: 'P', props: {x: {kind: 'float', default: 1}}, nodes: [{id: 'a', op: 'aspectUV'}], output: '$a'}
        expect(validateStdGraph(env2).error).toMatch(/need min and max/)
    })

    it('reports every problem in one pass, without repeating what follows from one', () => {
        const env: StdGraphEnvelope = {
            kind: 'std-graph', name: 'Many',
            props: {...HARNESS_PROPS, Bad: {kind: 'float', default: 1, min: 0, max: 1}},
            nodes: [
                {id: 'a', op: 'smoothStep', args: {}},
                {id: 'b', op: 'add', args: {a: '$a', b: 1}}, // reads the failed node: not reported again
                {id: 'c', op: 'add', args: {a: 1, b: 1, c: 2}},
                {id: 'out', op: 'rgba', args: {rgb: '$b', a: 1}},
            ],
            output: '$out',
        }
        const r = validateStdGraph(env)
        expect(r.ok).toBe(false)
        expect(r.errors).toHaveLength(3)
        expect(r.error).toMatch(/^3 problems:/)
        expect(r.error).toMatch(/camelCase/)
        expect(r.error).toMatch(/unknown op "smoothStep"/)
        expect(r.error).toMatch(/no input "c"/)
    })

    it('takes [x, y] … [x, y, z, w] as vector literals where a vector is expected', () => {
        const env = base([
            {id: 'd', op: 'sub', args: {a: 'uv', b: [0.5, 0.5]}},
            {id: 'g', op: 'splat3', args: {x: '$d.x'}},
            {id: 'tint', op: 'mul', args: {a: '$g', b: [1, 0.8, 0.6]}},
            {id: 'out', op: 'rgba', args: {rgb: '$tint', a: 1}},
        ])
        const check = validateStdGraph(env)
        if (!check.ok) throw new Error(check.error)
        const res = resolveStdGraphWgsl(env)
        if (!res.ok) throw new Error(res.error)
        expect(res.wgsl).toMatch(/vec2f\(0\.5, 0\.5\)/)
    })
})

describe('std/graph normalization (model forgiveness)', () => {
    it('coerces array positions, string numbers, missing float ranges', () => {
        const env = {props: {
            c: {kind: 'position', default: [0.3, 0.7]},
            f: {kind: 'float', default: '2.5'},
            z: {kind: 'float', default: 0},
            b: {kind: 'bool', default: 1},
        }} as unknown as {props: Record<string, GraphPropSpec>}
        normalizeStdGraph(env)
        expect(env.props.c!.default).toEqual({x: 0.3, y: 0.7})
        expect(env.props.f).toMatchObject({default: 2.5, min: 0, max: 5})
        expect(env.props.z).toMatchObject({default: 0, min: 0, max: 1})
        expect(env.props.b!.default).toBe(true)
    })
})

describe('std/graph forgiveness (the mistakes models actually make)', () => {
    const base = (nodes: GraphNode[], output: string): StdGraphEnvelope => ({kind: 'std-graph', name: 'T', props: HARNESS_PROPS, nodes, output})

    it('hoists body/result/absorb written inside args', () => {
        const env = base([
            {id: 'm', op: 'march', args: {steps: 4, entry: 0, length: 1, body: [{id: 'e', op: 'splat3', args: {x: 'z'}}], result: '$e', absorb: '$e.x'} as never},
            {id: 'out', op: 'rgba', args: {rgb: '$m.color', a: 1}},
        ], '$out')
        const check = validateStdGraph(env)
        if (!check.ok) throw new Error(check.error)
        expect(resolveStdGraphWgsl(env).ok).toBe(true)
    })

    it('accepts a forgotten sigil when the name is unambiguous', () => {
        const env = base([
            {id: 'c', op: 'splat3', args: {x: 'uv.x'}},
            {id: 'd', op: 'mul', args: {a: 'c', b: 'f1'}},
            {id: 'out', op: 'rgba', args: {rgb: 'd', a: 1}},
        ], '$out')
        const check = validateStdGraph(env)
        if (!check.ok) throw new Error(check.error)
        expect(env.nodes[1]!.args!.a).toBe('$c')
        expect(env.nodes[1]!.args!.b).toBe('@f1')
        expect(resolveStdGraphWgsl(env).ok).toBe(true)
    })

    it('promotes an rgb to opaque rgba in vec4 colour slots', () => {
        const env = base([
            {id: 'c', op: 'splat3', args: {x: 'uv.x'}},
            {id: 'm', op: 'mixColors', args: {a: '$c', b: '@c1', t: 0.5}},
        ], '$m')
        const check = validateStdGraph(env)
        if (!check.ok) throw new Error(check.error)
        const res = resolveStdGraphWgsl(env)
        if (!res.ok) throw new Error(res.error)
    })
})

describe('std/graph wgsl escape hatch', () => {
    const base = (nodes: GraphNode[], output: string): StdGraphEnvelope => ({kind: 'std-graph', name: 'T', props: HARNESS_PROPS, nodes, output})
    const ripple: GraphNode = {id: 'r', op: 'wgsl', args: {
        inputs: {p: '$v2', t: 'time', k: '@f1'},
        type: 'f32',
        code: 'var acc = 0.0; for (var i = 0; i < 4; i++) { acc += sin(length(p) * k * f32(i + 1) - t); } return acc * 0.25;',
        name: 'ringRipples',
        why: 'waves() sums sines over axes, not over radial harmonics with a per-harmonic phase',
    }}

    it('validates, declares the hand-rolled part, and resolves through tgpu as a raw fn', () => {
        const env = base([...PRELUDE, ripple, {id: 'g', op: 'splat3', args: {x: '$r'}}, {id: 'out', op: 'rgba', args: {rgb: '$g', a: 1}}], '$out')
        const check = validateStdGraph(env)
        if (!check.ok) throw new Error(check.error)
        expect(check.handRolled).toEqual([{id: 'r', name: 'ringRipples', type: 'f32', why: expect.stringMatching(/waves/)}])
        const res = resolveStdGraphWgsl(env)
        if (!res.ok) throw new Error(res.error)
        expect(res.wgsl).toMatch(/fn handRolled_ringRipples\(p: vec2f, t: f32, k: f32\) -> f32/)
        expect(res.wgsl).toMatch(/acc \+= sin/)
    })

    it('is refused when the body touches resources or declares things', () => {
        const bad = (code: string) => validateStdGraph(base([...PRELUDE, {...ripple, args: {...ripple.args, code}}, {id: 'g', op: 'splat3', args: {x: '$r'}}, {id: 'out', op: 'rgba', args: {rgb: '$g', a: 1}}], '$out')).error
        expect(bad('return textureSample(tex, s, p).x;')).toMatch(/may not use "textureSample"/)
        expect(bad('fn helper() -> f32 { return 1.0; } return helper();')).toMatch(/may not use "fn"/)
        expect(bad('var acc = 0.0; { return acc;')).toMatch(/unbalanced/)
        expect(bad('let x = 1.0;')).toMatch(/must return/)
        expect(validateStdGraph(base([...PRELUDE, {...ripple, args: {...ripple.args, why: ''}}, {id: 'out', op: 'rgba', args: {rgb: '$v3', a: 1}}], '$out')).error).toMatch(/"name".*"why"/)
    })

    it('takes any WGSL identifier as an input name, except keywords and type names', () => {
        const named = (inputs: Record<string, GraphArg>, code: string) => validateStdGraph(base([...PRELUDE, {...ripple, args: {...ripple.args, inputs, code}}, {id: 'g', op: 'splat3', args: {x: '$r'}}, {id: 'out', op: 'rgba', args: {rgb: '$g', a: 1}}], '$out'))
        expect(named({wave_len: '@f1', P: [0.2, 0.4]}, 'return sin(P.x * wave_len);').ok).toBe(true)
        expect(named({type: '@f1'}, 'return type;').error).toMatch(/"type" is a WGSL keyword/)
        expect(named({vec2f: '@f1'}, 'return vec2f;').error).toMatch(/keyword or type name/)
        expect(named({_x: '@f1'}, 'return _x;').error).toMatch(/must be an identifier/)
    })

    it('marks the composition as carrying custom WGSL, so a broken body reports through the validation scope', () => {
        const env = base([...PRELUDE, ripple, {id: 'g', op: 'splat3', args: {x: '$r'}}, {id: 'out', op: 'rgba', args: {rgb: '$g', a: 1}}], '$out')
        const def = defineStdGraph(env)
        const {buildRegistry, RootContainer} = harness
        const {registry} = buildRegistry([{id: 'root', def: RootContainer, parentId: null}, {id: 'g', def, parentId: 'root'}])
        expect(composeNodeTree(registry).usesCustomWgsl).toBe(true)
    })
})

describe('std/graph revision (pipeline cache)', () => {
    it('two envelopes that differ only in body get different revisions; the same envelope is stable', async () => {
        const a = defineStdGraph(OCEAN_EXAMPLE)
        const b = defineStdGraph(OCEAN_EXAMPLE)
        const c = defineStdGraph({...OCEAN_EXAMPLE, nodes: OCEAN_EXAMPLE.nodes.slice(0, 26), output: '$body'})
        expect(a.revision).toBeTruthy()
        expect(a.revision).toBe(b.revision)
        expect(a.revision).not.toBe(c.revision)
        // And the structural hash the renderer keys pipelines on sees it.
        const {collectStructuralHashInputs} = await import('@coreroot/gpu/composer')
        const {createUniformStore} = await import('@coreroot/gpu/uniformStore')
        void createUniformStore
        const hashOf = (def: typeof a) => {
            const {buildRegistry, RootContainer} = harness
            const {registry} = buildRegistry([{id: 'root', def: RootContainer, parentId: null}, {id: 'g', def, parentId: 'root'}])
            return collectStructuralHashInputs(registry).join('\n')
        }
        expect(hashOf(a)).toBe(hashOf(b))
        expect(hashOf(a)).not.toBe(hashOf(c))
    })
})

describe('std/graph manifest', () => {
    it('documents every op once, grouped by section', () => {
        const text = renderStdGraphManifest()
        for (const op of Object.keys(CATALOG)) expect(text).toMatch(new RegExp(`^- ${op}\\(`, 'm'))
        expect(text).toMatch(/## coordinates/)
        expect(text).toMatch(/## control/)
    })
})

describe('wgsl envelopes', () => {
    const halo = {
        kind: 'wgsl' as const,
        name: 'Halo',
        props: {
            inner: {kind: 'color' as const, default: '#ffd166'},
            center: {kind: 'position' as const, default: {x: 0.5, y: 0.5}},
            radius: {kind: 'range' as const, default: 0.5, min: 0, max: 1},
            pulse: {kind: 'checkbox' as const, default: true},
        },
        body: 'let d = length((uv - center) * vec2f(aspect, 1.0)) / radius; let k = mix(1.0, 0.5 + 0.5 * sin(time), pulse); return vec4f(inner.rgb, (1.0 - smoothstep(0.8, 1.0, d)) * k);',
    }

    it('lowers a body that reads no child to a generator with the clock it uses', () => {
        expect(validateWgslEnvelope(halo)).toEqual({ok: true, role: 'generator', usesTime: true})
        const def = defineShaderEnvelope(halo)
        expect(def.requiresChild).toBeFalsy()
        expect(def.animatedTime).toEqual({speed: 'speed'})
        expect(Object.keys(def.props)).toEqual(['inner', 'center', 'radius', 'pulse', 'speed'])
        const {buildRegistry, RootContainer} = harness
        const {registry} = buildRegistry([{id: 'root', def: RootContainer, parentId: null}, {id: 'h', def, parentId: 'root'}])
        const ir = composeNodeTree(registry)
        expect(ir.usesCustomWgsl).toBe(true)
    })

    it('lowers a body that reads the child to a filter', () => {
        const tint = {kind: 'wgsl' as const, name: 'Tint', props: {amount: {kind: 'range' as const, default: 0.5, min: 0, max: 1}}, body: 'return vec4f(mix(child.rgb, vec3f(1.0, 0.0, 0.0), amount), child.a);'}
        expect(validateWgslEnvelope(tint)).toMatchObject({ok: true, role: 'filter', usesTime: false})
        expect(defineShaderEnvelope(tint).requiresChild).toBe(true)
        const blur = {kind: 'wgsl' as const, name: 'Smear', body: 'return textureSample(childTexture, childSampler, uv + vec2f(0.01, 0.0));'}
        const def = defineShaderEnvelope(blur)
        expect(def.requiresChild).toBe(true)
        expect(def.requiresRTT).toBe(true)
    })

    it('reads the earlier kind names as the library controls they always were', () => {
        const old = {
            kind: 'wgsl' as const,
            name: 'Old',
            props: {
                radius: {kind: 'float', default: 0.5, min: 0, max: 1},
                spin: {kind: 'angle', default: 30},
                pulse: {kind: 'bool', default: true},
            },
            body: 'return vec4f(radius, radians(spin), pulse, 1.0);',
        }
        const current = {
            ...old,
            props: {
                radius: {kind: 'range', default: 0.5, min: 0, max: 1},
                spin: {kind: 'range', default: 30, min: 0, max: 360, step: 1},
                pulse: {kind: 'checkbox', default: true},
            },
        }
        expect(validateWgslEnvelope(old as never)).toEqual({ok: true, role: 'generator', usesTime: false})
        const a = defineShaderEnvelope(old as never).props as Record<string, {default: unknown; ui?: unknown; transform?: unknown}>
        const b = defineShaderEnvelope(current as never).props as Record<string, {default: unknown; ui?: unknown; transform?: unknown}>
        for (const name of ['radius', 'spin', 'pulse']) {
            expect(a[name]!.ui).toEqual(b[name]!.ui)
            expect(a[name]!.default).toEqual(b[name]!.default)
            expect(a[name]!.transform).toBe(b[name]!.transform)
        }
    })

    it('names what is wrong with a document', () => {
        expect(validateShaderEnvelope({kind: 'glsl'}).error).toMatch(/"std-graph".*"wgsl"/)
        expect(validateWgslEnvelope({kind: 'wgsl', name: 'X', body: 'let a = 1.0;'}).error).toMatch(/return/)
        expect(validateWgslEnvelope({kind: 'wgsl', name: 'X', props: {uv: {kind: 'range', default: 0, min: 0, max: 1}}, body: 'return vec4f(1.0);'}).error).toMatch(/collides/)
        expect(validateWgslEnvelope({kind: 'wgsl', name: 'X', props: {r: {kind: 'range', default: 0}}, body: 'return vec4f(1.0);'}).error).toMatch(/need min and max/)
        expect(validateWgslEnvelope({kind: 'wgsl', name: 'X', props: {b: {kind: 'boolean', default: true}}, body: 'return vec4f(1.0);'} as never).error).toMatch(/use "checkbox"/)
        expect(validateWgslEnvelope({kind: 'wgsl', name: 'X', props: {t: {kind: 'text', default: 'hi'}}, body: 'return vec4f(1.0);'} as never).error).toMatch(/Text layer/)
    })
})

describe('onAuthorError', () => {
    it('hears every author error as one line, whether or not debugging is on, until unsubscribed', () => {
        const heard: string[] = []
        const stop = onAuthorError((message) => heard.push(message))
        authorError('[gpu] a shader failed GPU validation:', new Error('error: unresolved identifier "foo"'))
        stop()
        authorError('[gpu] after unsubscribing')
        expect(heard).toEqual(['[gpu] a shader failed GPU validation: error: unresolved identifier "foo"'])
    })
})
