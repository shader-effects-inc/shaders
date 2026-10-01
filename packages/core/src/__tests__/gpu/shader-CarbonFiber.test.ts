import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import CarbonFiber from '@coreroot/shaders/CarbonFiber/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * CarbonFiber gate (W7-A → std sweep). A STATIC (non-animated) GENERATOR whose woven-carbon
 * material is std algebra in the definition file over shared kit parts (weave layout + tow relief +
 * fibre filaments + Ward anisotropic sheen + one clearcoat tap of the shared studio5Softbox bank),
 * inside the `guarded` shape region. The compileTime `weaveStyle` bakes its parity math via a
 * build-time JS branch. GPU-free → flat analytic path.
 */
const C = CarbonFiber as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('CarbonFiber (a) flat twill weave', () => {
    it('emits the analytic circle sampler + the shared material parts + noise, no RTT pass, no _animTime', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def: C, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/wardAnisotropicSpecular/)
        expect(finalWgsl).toMatch(/studio5Softbox/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/mxNoiseFloat2/)
        expect(finalWgsl).toMatch(/_saRadius/)
        // Twill parity: the diagonal 2-tow floored mod-4 (with its negative-index +64 bias).
        expect(finalWgsl).toMatch(/\+ 64\.0/)
        // CarbonFiber is NOT animated — no per-node animated-time field.
        expect(finalWgsl).not.toMatch(/_animTime/)
        expect(C.animatedTime).toBeUndefined()
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers in the structural hash surface', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def: C, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('CarbonFiber')
    })
})

describe('CarbonFiber (b) plain weave compile-time branch', () => {
    it('weaveStyle:plain bakes the checkerboard parity (no twill mod-4 bias)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def: C, parentId: 'root', props: {weaveStyle: 'plain'}, metadata: {renderOrder: 0}},
        ])
        const finalWgsl = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        expect(finalWgsl).not.toMatch(/\+ 64\.0/)
        expect(finalWgsl).toMatch(/\/ 2\.0\)/)
    })
})
