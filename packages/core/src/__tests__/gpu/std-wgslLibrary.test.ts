import {describe, expect, it} from 'vitest'
import tgpu from 'typegpu'
import {libFn, libSource, linkLibFns} from '@coreroot/std/wgslLibrary'
import {parseWgslDecls, schemaForWgslType, structSchemas} from '@coreroot/std/wgslSource'

const hash = libFn({
    name: 'testHash',
    family: 'test',
    signature: 'testHash(p: vec2f) -> f32',
    doc: 'A hash.',
    code: `
        const K: f32 = 43758.5453;
        fn testHash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(12.9898, 78.233))) * K); }
    `,
})

const cell = libFn({
    name: 'testCell',
    family: 'test',
    signature: 'testCell(p: vec2f) -> TestCell',
    doc: 'A cell.',
    deps: [hash],
    code: `
        struct TestCell { id: vec2f, value: f32 }
        fn _testCell_floor(p: vec2f) -> vec2f { return floor(p); }
        fn testCell(p: vec2f) -> TestCell {
            let id = _testCell_floor(p);
            return TestCell(id, testHash(id));
        }
    `,
})

describe('wgsl source reader', () => {
    it('reads structs, fns and consts with their types', () => {
        const decls = parseWgslDecls(`
            // a comment { with braces }
            struct A { x: f32, y: array<vec2f, 4>, }
            const N = 3;
            fn f(a: A, @builtin(position) p: vec4f) -> vec3<f32> { if (true) { return vec3f(0.0); } return vec3f(1.0); }
            fn g() { }
        `)
        expect(decls.structs.map((s) => s.name)).toEqual(['A'])
        expect(decls.structs[0].fields).toEqual([{name: 'x', type: 'f32'}, {name: 'y', type: 'array<vec2f, 4>'}])
        expect(decls.consts).toEqual([expect.objectContaining({name: 'N', value: '3'})])
        expect(decls.fns.map((f) => [f.name, f.ret])).toEqual([['f', 'vec3<f32>'], ['g', null]])
        expect(decls.fns[0].params).toEqual([{name: 'a', type: 'A'}, {name: 'p', type: 'vec4f'}])
    })

    it('refuses statements at module scope with a line number', () => {
        expect(() => parseWgslDecls('fn a() {}\nlet x = 1;')).toThrow(/line 2/)
    })

    it('builds schemas for nested structs declared in any order', () => {
        const decls = parseWgslDecls('struct Outer { inner: Inner } struct Inner { v: vec2f }')
        const schemas = structSchemas(decls.structs)
        expect(schemas.has('Outer')).toBe(true)
        expect(() => schemaForWgslType('texture_2d<f32>')).toThrow(/cannot be passed/)
    })
})

describe('wgsl library linking', () => {
    it('pastes dependency-closed source in order for compute modules', () => {
        const src = libSource([cell])
        expect(src.indexOf('fn testHash')).toBeLessThan(src.indexOf('fn testCell'))
    })

    it('links a library fn, its private helper, its dep and its struct into one resolvable fn', () => {
        const externals = linkLibFns([cell])
        expect(Object.keys(externals).sort()).toEqual(['TestCell', 'testCell'])
        const wgsl = tgpu.resolve([externals.testCell as never], {names: 'strict'})
        expect(wgsl).toContain('struct TestCell')
        expect(wgsl).toMatch(/fn testCell/)
        expect(wgsl).toContain('43758.5453') // the module const is inlined into the body
        expect(wgsl).toMatch(/fn _testCell_floor/)
    })

    it('rejects a private helper without the owner prefix', () => {
        expect(() => libFn({name: 'x', family: 't', signature: '', doc: '', code: 'fn x() {} fn helper() {}'})).toThrow(/_x_/)
    })
})
