import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import LiquidMetal from '@coreroot/shaders/LiquidMetal/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * LiquidMetal gate (W7-A → std sweep). A GENERATOR whose flowing-chrome material is std algebra in
 * the definition file over shared kit parts (geometric normal + domain-warped molten relief via the
 * shared flow-warp / clamped-Perlin-gradient parts + LiquidMetal's own inline 3-bank studio +
 * sharp glint + dispersion), inside the `guarded` shape region. GPU-free → flat analytic path
 * (default circleSDF); the compute volumetric field is skipped. Validates: the analytic sampler +
 * the shared material parts + the perlin relief + the animated `_animTime` read all reach the
 * emitted WGSL.
 */
const L = LiquidMetal as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('LiquidMetal (a) flat analytic chrome generator', () => {
    it('emits the analytic circle sampler + the shared material parts + perlin relief, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'm', def: L, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/clampedPerlinGrad/)
        expect(finalWgsl).toMatch(/flowWarpOffset/)
        expect(finalWgsl).toMatch(/sharpGlint/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/keyLight/)
        expect(finalWgsl).toMatch(/perlin12d/)
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(L.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'm', def: L, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('LiquidMetal')
    })
})
