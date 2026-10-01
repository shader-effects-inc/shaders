import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Water from '@coreroot/shaders/Water/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Water gate (W7-A → std sweep). A GENERATOR whose translucent-water material is std algebra in
 * the definition file over shared kit parts (wave relief + reflected/transmitted procedural sky +
 * Beer–Lambert absorption + caustics + Fresnel + foam, inside the `guarded` shape region).
 * GPU-free → flat analytic path. Validates the composition + the animated `_animTime` read.
 */
const W = Water as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Water (a) flat analytic water generator', () => {
    it('emits the analytic circle sampler + the shared material parts + noise, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'w', def: W, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/beerLambert/)
        expect(finalWgsl).toMatch(/sharpGlint/)
        expect(finalWgsl).toMatch(/flowWarpOffset/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/mxNoiseFloat2/)
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(W.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'w', def: W, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Water')
    })
})
