import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import Sparkle from '@coreroot/shaders/Sparkle/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Sparkle gate — an RTT gather filter. Two scattered grids (`scatterPoint` → `hash32`) each
 * sample the child at their cell's point, light a `starGlint` (`glowSpot` core + rays) gated
 * by `flashes`, and the glints add over the centre sample before the unpremultiply tail.
 */
const SP = Sparkle as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

const compose = (props?: Record<string, unknown>): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'sp', def: SP, parentId: 'root', metadata: {renderOrder: 0}, props},
        {id: 'gen', def: Generator, parentId: 'sp', metadata: {renderOrder: 0}},
    ])
    const ir = composeNodeTree(registry)
    expect(ir.rttPasses.length).toBe(1)
    return tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
}
const childSamples = (wgsl: string): number => wgsl.match(/textureSample\(rtt_/g)?.length ?? 0

describe('Sparkle RTT gather path', () => {
    it('RTTs the child, scatters 2 grids × 4 nearest cells of star glints, unpremultiplies', () => {
        const wgsl = compose()
        expect(wgsl).toMatch(/hash32/)
        expect(wgsl).toMatch(/glowSpot/)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
        // Centre sample + one point sample per cell (2 grids × 4 cells).
        expect(childSamples(wgsl)).toBe(9)
        expect(wgsl).toMatchSnapshot('final-pass')
    })

    it('expand > 0 compiles in a 4-tap ring around every point', () => {
        expect(childSamples(compose({expand: 20}))).toBe(9 + 8 * 4)
    })
})
