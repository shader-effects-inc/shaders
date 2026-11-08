import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {defineShader, wgsl} from '@coreroot/std'
import {libFn} from '@coreroot/std/wgslLibrary'
import {transformPosition} from '@coreroot/utilities/transformations'
import {buildRegistry, RootContainer} from './_patternHarness'

/** The `shape:` and `map:` slots written in WGSL: the distance and coordinate bodies. */

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

function resolveTree(specs: Parameters<typeof buildRegistry>[0]) {
    const {registry} = buildRegistry(specs)
    const ir = composeNodeTree(registry)
    return {ir, wgsl: tgpu.resolve([ir.finalPass.entry], {names: 'strict'})}
}

const roundness = libFn({
    name: 'roundBox',
    family: 'test',
    signature: 'roundBox(p, b, r) -> f32',
    doc: 'A rounded box.',
    code: 'fn roundBox(p: vec2f, b: vec2f, r: f32) -> f32 { let q = abs(p) - b + r; return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r; }',
})

describe('shape: a distance body', () => {
    const Pill = defineShader({
        name: 'Pill',
        props: {
            width: {default: 0.4, ui: {type: 'range', min: 0, max: 1}},
            height: {default: 0.15, ui: {type: 'range', min: 0, max: 1}},
        },
        inputs: {roundBox: roundness},
        shape: wgsl`return roundBox(p, vec2f(width, height), height);`,
        bounds: {size: {width: 'width', height: 'height'}},
    })

    it('gets the shape scaffold: fill, stroke, position, and its own props', () => {
        expect(Object.keys(Pill.props)).toEqual(expect.arrayContaining(['width', 'height', 'color', 'center', 'rotation']))
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 's', def: Pill, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        expect(out).toMatch(/fn pillDistance\(p: vec2f, height: f32, width: f32\) -> f32/)
        expect(out).toMatch(/fn roundBox/)
    })

    it('requires a return', () => {
        expect(() => defineShader({name: 'Bad', props: {}, shape: wgsl`let x = 1.0;`})).toThrow(/must return/)
    })
})

describe('map: a coordinate body', () => {
    const Ripple = defineShader({
        name: 'Ripple',
        props: {
            center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
            amount: {default: 0.02},
        },
        map: wgsl`
            let d = uv - center;
            return uv + normalize(d + vec2f(1e-5)) * sin(length(d * vec2f(aspect, 1.0)) * 40.0 - time * 3.0) * amount;
        `,
    })

    it('gets its own clock (and a speed prop) when it reads time', () => {
        expect(Ripple.animatedTime).toEqual({speed: 'speed'})
        expect(Ripple.props).toHaveProperty('speed')
        expect(Ripple.uvRemap).toBeTypeOf('function')
    })

    it('composes as a warp over its child', () => {
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'w', def: Ripple, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'w', metadata: {renderOrder: 0}},
        ])
        expect(out).toMatch(/fn rippleMap\(uv: vec2f, amount: f32, aspect: f32, center: vec2f, time: f32\) -> vec2f/)
    })
})
