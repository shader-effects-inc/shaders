import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree} from '@coreroot/gpu/composer'
import {defineShader, wgsl, p} from '@coreroot/std'
import {child, blur, bloom, previousFrame, pass, downsample, directionalBlur, text} from '@coreroot/std/inputs'
import {planInputs} from '@coreroot/std/inputs/graph'
import {libFn} from '@coreroot/std/wgslLibrary'
import {transformColor} from '@coreroot/utilities/transformations'
import {buildRegistry, RootContainer} from './_patternHarness'
import {tgpu as kitTgpu, d} from '@coreroot/gpu/kit'
import {call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'

const genBody = kitTgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

/**
 * Inputs, library functions, helpers and frame built-ins in the agent format: the GPU-free half.
 * Compositions resolve to WGSL here; the compute work itself needs a device and is not run.
 */

function resolveTree(specs: Parameters<typeof buildRegistry>[0]) {
    const {registry} = buildRegistry(specs)
    const ir = composeNodeTree(registry)
    const wgslText = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
    return {ir, wgsl: wgslText}
}

const wobble = libFn({
    name: 'wobble',
    family: 'test',
    signature: 'wobble(p: vec2f) -> f32',
    doc: 'A test wave.',
    code: 'fn wobble(p: vec2f) -> f32 { return sin(p.x * 10.0) * 0.5 + 0.5; }',
})

describe('helpers, library functions and built-ins', () => {
    const Rings = defineShader({
        name: 'Rings',
        props: {
            tint: {default: '#ff8800', transform: transformColor},
            bands: {default: 6},
        },
        inputs: {wobble},
        helpers: wgsl`
            struct Band { value: f32, edge: f32 }
            fn band(d: f32) -> Band {
                let v = fract(d * bands + wobble(vec2f(d, time)));
                return Band(v, smoothstep(0.45, 0.5, v));
            }
            fn shade(d: f32) -> vec3f { return tint.rgb * band(d).edge; }
        `,
        paint: wgsl`
            let d = length(uv - 0.5) + dt * 0.0;
            let b = band(d);
            return vec4f(shade(d) * b.value, 1.0);
        `,
    })

    it('threads props, built-ins and library functions through helpers as parameters', () => {
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'r', def: Rings, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        // `band` reads bands + time + wobble; `shade` reads tint and calls band, so it inherits its needs.
        expect(out).toMatch(/fn band\(d: f32, bands: f32, time: f32\) -> Band/)
        expect(out).toMatch(/fn shade\(d: f32, bands: f32, time: f32, tint: vec4f\) -> vec3f/)
        expect(out).toMatch(/fn wobble\(p: vec2f\) -> f32/)
        expect(out).toMatch(/struct Band/)
        // `dt` reads its per-frame field.
        expect(out).toMatch(/_b_dt/)
    })

    it('declares the frame field a built-in reads', () => {
        expect(Rings.extraFields).toHaveProperty('_b_dt')
    })

    it('refuses a non-input in inputs', () => {
        expect(() => defineShader({
            name: 'Bad',
            props: {},
            inputs: {nope: 4 as never},
            paint: wgsl`return vec4f(1.0);`,
        })).toThrow(/'nope' must be input words/)
    })
})

describe('inputs in a body', () => {
    const Dreamy = defineShader({
        name: 'Dreamy',
        props: {softness: {default: 20}, glow: {default: 0.5}},
        inputs: {
            soft: blur(child, {radius: p('softness')}),
            halo: bloom(child, {threshold: 0.7}),
        },
        effect: wgsl`
            let band = smoothstep(0.2, 0.6, abs(uv.y - 0.5));
            return mix(child, soft(uv), band) + halo(uv) * glow;
        `,
    })

    it('reads each input through a sampler call, and asks for the child', () => {
        expect(Dreamy.requiresChild).toBe(true)
        expect(Dreamy.compute).toBeTypeOf('function')
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'd', def: Dreamy, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'd', metadata: {renderOrder: 0}},
        ])
        expect(out).toMatch(/_inRead_soft\(_inTex_soft, _inSamp, uv\)/)
        expect(out).toMatch(/_inRead_\w+\(_inTex_halo, _inSamp, uv\)/)
        expect(out).toMatch(/textureSampleLevel/)
    })

    it('plans nested and named inputs deps-first', () => {
        const small = downsample(child, {scale: 0.25})
        const plan = planInputs({
            streak: directionalBlur(small, {angle: 30, length: 40}),
            clouds: pass(wgsl`return streak(uv) * 0.5;`, {scale: 0.5}),
        })
        expect(plan.inputs.map((i) => i.name)).toEqual(['_in0', 'streak', 'clouds'])
        expect(plan.reads.map((r) => r.call)).toEqual(['streak', 'clouds'])
        expect(plan.readsChild).toBe(true)
    })

    it('renders a body with a history to a texture and samples it', () => {
        const Trail = defineShader({
            name: 'Trail',
            props: {fade: {default: 0.95}},
            inputs: {last: previousFrame()},
            paint: wgsl`
                let ink = 1.0 - smoothstep(0.0, 0.03, distance(uv, pointer));
                return max(last(uv) * fade, vec4f(ink));
            `,
        })
        expect(Trail.usesPointer).toBe(true)
        const {ir} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 't', def: Trail, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        // The body is its own render pass now (the history copy reads it next frame).
        expect(ir.rttPasses.length).toBeGreaterThanOrEqual(1)
    })

    it('stores inputs as plain data (the builder is not serialised)', () => {
        const node = blur(child, {radius: p('softness')})
        expect(JSON.parse(JSON.stringify(node))).toEqual({kind: 'blur', source: {kind: 'child'}, radius: {kind: 'prop', name: 'softness'}})
    })
})

describe('text shape outputs (distance, nearest, letters)', () => {
    it('plans each output with its own read kind', () => {
        const word = text(p('label'), {distance: true, letters: 12})
        expect(word.outputNames).toEqual(['distance', 'nearest', 'letters'])
        const plan = planInputs({word})
        const by = Object.fromEntries(plan.reads.map((r) => [r.call, [r.by, !!r.unfilterable]]))
        expect(by).toEqual({
            word: ['uv', false],
            'word.distance': ['uv', true],
            'word.nearest': ['texel', true],
            'word.letters': ['index', true],
        })
    })

    it('leaves a plain text input with only its coverage', () => {
        const word = text('HELLO')
        expect(word.outputNames).toBeUndefined()
        expect(planInputs({word}).reads.map((r) => r.call)).toEqual(['word'])
    })

    it('resolves a body reading the field smoothly, exactly and by letter', () => {
        const word = text(p('label'), {distance: true, letters: true})
        const Halo = defineShader({
            name: 'Halo',
            props: {label: {default: 'HI', ui: {type: 'text', label: 'Text'}}},
            inputs: {word},
            paint: wgsl`
                let f = word.distance(uv);
                let n = word.nearest(uv);
                let box = word.letters(i32(n.w));
                return vec4f(vec3f(exp(-max(f.x, 0.0) / 40.0)) * step(0.0, box.z), 1.0);
            `,
        })
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'h', def: Halo, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        // distance: four exact texels blended; nearest: the one texel under p; letters: slot i.
        expect(out).toMatch(/textureLoad\(t, clamp\(vec2i\(floor\(p \* vec2f\(size\)\)\)/)
        expect(out).toMatch(/let q = p \* vec2f\(size\) - 0\.5/)
        expect(out).toMatch(/clamp\(i, 0, i32\(textureDimensions\(tex\)\.x\) - 1\)/)
    })

    it('keeps the options as plain data', () => {
        const node = JSON.parse(JSON.stringify(text('A', {distance: true, letters: 300})))
        expect(node.distance).toBe(true)
        expect(node.letters).toBe(256)
    })
})
