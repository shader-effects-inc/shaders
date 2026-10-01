import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import Crystal from '@coreroot/shaders/Crystal/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Crystal gate (W7-A → std sweep — the batch's hardest: a full RTT lens with no shared kit
 * effect). The facet trace + composite are std algebra in the definition file. GPU-free → the
 * volumetric compute pre-march is skipped, so the fragment takes the FLAT analytic path (default
 * polygonSDF). Validates the SHADER's routing + composition: RTT filter over a child, the
 * kaleidoscope-Worley facet taps, the unpremultiplied child samples, the two key-light frames,
 * and cutout as a structural-hash input.
 */
const C = Crystal as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

const withChild = (props: Record<string, unknown> = FLAT_SHAPE) =>
    buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'c', def: C, parentId: 'root', props, metadata: {renderOrder: 0}},
        {id: 'gen', def: Generator, parentId: 'c', metadata: {renderOrder: 0}},
    ]).registry

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the polygonSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'polygonSDF', radius: 0.35, sides: 10}), shapeType: 'polygonSDF'}

describe('Crystal (a) flat analytic polygon lens over an RTT child', () => {
    it('RTTs the child, traces the kaleidoscope Worley facets, and assembles the crystal composite', () => {
        const ir = composeNodeTree(withChild())
        expect(ir.rttPasses.length).toBe(1) // requiresRTT → the child renders to a texture

        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        const rttWgsl = tgpu.resolve([ir.rttPasses[0].fragment.entry], {names: 'strict'})
        // Flat analytic path: Crystal's default shape is polygonSDF (10 sides).
        expect(finalWgsl).toMatch(/analyticSdf_polygonSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/keyLight/)
        expect(finalWgsl).toMatch(/insideMask/)
        expect(finalWgsl).toMatch(/mxWorleyNoiseFloat2Pub/)
        // RTT sampling → unpremultiply on the way back into the straight-alpha pipeline (Twirl trap #2).
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        // The analytic sub-prop extraFields are declared + read (driveAnalyticSubProps).
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(rttWgsl).toMatch(/genBody/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('Crystal (b) cutout is a compile-time structural-hash input', () => {
    const hashWith = (props: Record<string, unknown>) => collectStructuralHashInputs(withChild(props)).join('\n')
    it('toggling cutout changes the structural hash (recompose)', () => {
        expect(hashWith({cutout: true})).not.toBe(hashWith({cutout: false}))
    })
})
