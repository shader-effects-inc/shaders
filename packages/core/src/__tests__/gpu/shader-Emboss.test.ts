import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import Emboss from '@coreroot/shaders/Emboss/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Emboss port gate (W6-C). An RTT relief filter (requiresRTT/requiresChild): it RTTs the child,
 * samples the analytic SDF, and shades a debossed/embossed relief over the parallax-warped child
 * sample. GPU-free → flat analytic path (default circleSDF). Relief math golden-tested in
 * kit-effects-lighting.test.ts; this validates the shader's routing + RTT composition.
 */
const E = Emboss as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Emboss (a) flat analytic circle over an RTT child', () => {
    it('RTTs the child, samples the analytic circle, and shades the relief (unpremultiplied)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'e', def: E, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'e', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        const rttWgsl = tgpu.resolve([ir.rttPasses[0].fragment.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/embossComposite/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        expect(rttWgsl).toMatch(/genBody/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})
