import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import Glass from '@coreroot/shaders/Glass/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Glass port gate (W6-C — the flagship shape-effect port, first real consumer of the D3 kit).
 * GPU-free: the composer builds the raw-WGSL entries; we resolve + snapshot. Without a device the
 * volumetric compute pre-march + the frosted-blur compute are BOTH skipped (createVolumetricFieldComputeNode
 * / createGaussianBlurCompute return null), so the fragment takes the FLAT analytic path with the
 * in-shader Vogel blur fallback — exactly the GPU-free branch. The heavy glass lighting/refraction
 * math + the `applyGlassEffect` builder are golden/resolve-tested in kit-effects-lighting.test.ts and
 * the samplers in kit-sdf-samplers.test.ts; this suite validates the SHADER's routing + composition:
 * RTT filter over a child, analytic circle sampler, cutout as a structural-hash input, and the
 * Vogel blur fallback path.
 */
const G = Glass as GpuShaderDefinition

// A trivial colored generator child (an RTT source for the filter).
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
        {id: 'g', def: G, parentId: 'root', props, metadata: {renderOrder: 0}},
        {id: 'gen', def: Generator, parentId: 'g', metadata: {renderOrder: 0}},
    ]).registry

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Glass (a) flat analytic circle over an RTT child', () => {
    it('RTTs the child, samples the analytic circle SDF, and assembles the glass composite', () => {
        const ir = composeNodeTree(withChild())
        expect(ir.rttPasses.length).toBe(1) // requiresRTT → the child renders to a texture

        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        const rttWgsl = tgpu.resolve([ir.rttPasses[0].fragment.entry], {names: 'strict'})
        // Flat analytic path: the circle sampler + the glass lens/composite math.
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/glassComposite/)
        expect(finalWgsl).toMatch(/glassLensUVs/)
        // RTT sampling → unpremultiply on the way back into the straight-alpha pipeline (Twirl trap #2).
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        // The analytic sub-prop extraFields are declared + read (driveAnalyticSubProps).
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(rttWgsl).toMatch(/genBody/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('Glass (b) cutout is a compile-time structural-hash input', () => {
    const hashWith = (props: Record<string, unknown>) => collectStructuralHashInputs(withChild(props)).join('\n')
    it('toggling cutout changes the structural hash (recompose)', () => {
        expect(hashWith({cutout: true})).not.toBe(hashWith({cutout: false}))
    })
})

describe('Glass (c) in-shader Vogel blur fallback (GPU-free, no compute)', () => {
    it('blur > 0 with no device → the fragment blurs via the Vogel disk (glassBlurOffset)', () => {
        const finalWgsl = tgpu.resolve([composeNodeTree(withChild({blur: 5})).finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/glassBlurOffset/)
        expect(finalWgsl).toMatch(/glassComposite/)
    })
    it('blur = 0 → no Vogel offset taps', () => {
        const finalWgsl = tgpu.resolve([composeNodeTree(withChild({blur: 0})).finalPass.entry], {names: 'strict'})
        expect(finalWgsl).not.toMatch(/glassBlurOffset/)
    })
})
