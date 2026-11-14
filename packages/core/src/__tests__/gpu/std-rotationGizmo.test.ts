import {describe, expect, it} from 'vitest'
import ts from 'typescript'
import * as std from '@coreroot/std'
import {eulerRotate, eulerRotateInverse} from '@coreroot/std/library/index'
import {rotateVecCpu} from '@coreroot/gpu/kit/sdf3d'
import {parseComponentSource, evaluateComponentIR} from '@coreroot/std/document/index'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'

/**
 * `rotationGizmo`: a component names the three degree props the editor's 3D rotation gizmo drives.
 * The declaration must survive the agent format (parsed without running), be checked against the
 * props, and the WGSL helpers that apply the rotation must match the library's 3D shapes exactly,
 * since that is the convention the gizmo's rings are drawn in.
 */

const SOURCE = `
import {defineShader, wgsl} from 'shaders/std'
import {eulerRotate} from 'shaders/std/wgsl'

export default defineShader({
  name: 'Spinner',
  props: {
    tiltX: {default: 20, ui: {type: 'range', min: -180, max: 180, label: 'Tilt X'}},
    tiltY: {default: -10, ui: {type: 'range', min: -180, max: 180, label: 'Tilt Y'}},
    spin: {default: 0, ui: {type: 'range', min: -180, max: 180, label: 'Spin'}},
  },
  rotationGizmo: {x: 'tiltX', y: 'tiltY', z: 'spin', radius: 0.4},
  inputs: {eulerRotate},
  paint: wgsl\`
    let rd = eulerRotate(vec3f(uv - 0.5, 1.0), vec3f(tiltX, tiltY, spin));
    return vec4f(rd * 0.5 + 0.5, 1.0);
  \`,
})
`

const props = {
    a: {default: 0, ui: {type: 'range', min: -180, max: 180}},
    b: {default: 0, ui: {type: 'range', min: -180, max: 180}},
    c: {default: 0, ui: {type: 'range', min: -180, max: 180}},
    word: {default: 'hi', ui: {type: 'text'}},
}
const define = (rotationGizmo: unknown) => std.defineShader({
    name: 'G',
    props,
    rotationGizmo,
    paint: std.wgsl('return vec4f(a, b, c, 1.0);'),
} as never) as GpuShaderDefinition

describe('rotationGizmo', () => {
    it('survives the agent format: parsed without running, evaluated back onto the definition', () => {
        const parsed = parseComponentSource(SOURCE, ts)
        expect(parsed.ok).toBe(true)
        if (!parsed.ok) return
        const def = evaluateComponentIR(parsed.ir)
        expect(def.rotationGizmo).toEqual({x: 'tiltX', y: 'tiltY', z: 'spin', radius: 0.4})
    })

    it('is carried by every role, not just generators', () => {
        const surface = std.defineShader({
            name: 'S',
            props: {rx: {default: 0, ui: {type: 'range', min: -180, max: 180}}, ry: {default: 0, ui: {type: 'range', min: -180, max: 180}}, rz: {default: 0, ui: {type: 'range', min: -180, max: 180}}},
            rotationGizmo: {x: 'rx', y: 'ry', z: 'rz'},
            surface: std.wgsl('return vec4f(vec3f(rx + ry + rz), step(shapeDistance, 0.0));'),
        } as never) as GpuShaderDefinition
        expect(surface.rotationGizmo).toEqual({x: 'rx', y: 'ry', z: 'rz'})
    })

    it('needs all three axes, each a numeric prop', () => {
        expect(define({x: 'a', y: 'b', z: 'c'}).rotationGizmo).toEqual({x: 'a', y: 'b', z: 'c'})
        expect(() => define({x: 'a', y: 'b'})).toThrow(/rotationGizmo\.z must name a prop/)
        expect(() => define({x: 'a', y: 'b', z: 'nope'})).toThrow(/'nope', which is not a prop/)
        expect(() => define({x: 'a', y: 'b', z: 'word'})).toThrow(/must be a number/)
        expect(() => define({x: 'a', y: 'b', z: 'c', radius: 0})).toThrow(/radius must be a positive number/)
    })

    it('is absent unless declared', () => {
        expect(std.defineShader({name: 'N', props, paint: std.wgsl('return vec4f(1.0);')} as never).rotationGizmo).toBeUndefined()
    })
})

/** Run a library function's WGSL on the CPU: a straight transliteration of its plain-scalar code. */
type V3 = {x: number; y: number; z: number}
function runWgslFn(code: string, name: string): (v: V3, deg: V3) => V3 {
    const js = code
        .replace(/fn (\w+)\(([^)]*)\) -> \w+ \{/, (_m, fn, params) => `function ${fn}(${params.split(',').map((p: string) => p.split(':')[0].trim()).join(', ')}) {`)
        .replace(/\blet\b/g, 'const')
    const helpers = {
        vec3f: (x: number, y: number, z: number): V3 => ({x, y, z}),
        radians: (d: V3): V3 => ({x: d.x * Math.PI / 180, y: d.y * Math.PI / 180, z: d.z * Math.PI / 180}),
        cos: Math.cos,
        sin: Math.sin,
    }
    return new Function(...Object.keys(helpers), `${js}\nreturn ${name};`)(...Object.values(helpers))
}

describe('eulerRotate / eulerRotateInverse', () => {
    const rotate = runWgslFn(eulerRotate.code, 'eulerRotate')
    const inverse = runWgslFn(eulerRotateInverse.code, 'eulerRotateInverse')
    const cases: [V3, V3][] = [
        [{x: 0.3, y: -0.7, z: 0.65}, {x: 38, y: -30, z: 0}],
        [{x: -1, y: 0.2, z: 0.1}, {x: -75, y: 140, z: 33}],
        [{x: 0, y: 0, z: 1}, {x: 90, y: 0, z: -12}],
    ]

    it('matches the library 3D shapes (rotateVec3, ZXY)', () => {
        for (const [v, deg] of cases) {
            const D = Math.PI / 180
            const s = {cx: Math.cos(deg.x * D), sx: Math.sin(deg.x * D), cy: Math.cos(deg.y * D), sy: Math.sin(deg.y * D), cz: Math.cos(deg.z * D), sz: Math.sin(deg.z * D)}
            const want = rotateVecCpu(v.x, v.y, v.z, s)
            const got = rotate(v, deg)
            expect(got.x).toBeCloseTo(want.x, 6)
            expect(got.y).toBeCloseTo(want.y, 6)
            expect(got.z).toBeCloseTo(want.z, 6)
        }
    })

    it('inverse undoes it', () => {
        for (const [v, deg] of cases) {
            const back = inverse(rotate(v, deg), deg)
            expect(back.x).toBeCloseTo(v.x, 6)
            expect(back.y).toBeCloseTo(v.y, 6)
            expect(back.z).toBeCloseTo(v.z, 6)
        }
    })
})
