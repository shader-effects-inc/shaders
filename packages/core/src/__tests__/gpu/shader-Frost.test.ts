import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Frost from '@coreroot/shaders/Frost/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Frost gate (W6-C → std sweep). A GENERATOR whose frozen-ice material is std algebra in the
 * definition file over shared kit parts (subsurface Beer–Lambert + rim frost crystals + sparkle,
 * inside the `guarded` shape region). GPU-free → flat analytic path (default circleSDF); the
 * compute volumetric field is skipped. Validates the composition: the analytic sampler + the
 * shared material parts + the animated `_animTime` read + the frost-crystal noise
 * (mxNoiseFloat2 / worley) all reach the emitted WGSL.
 */
const F = Frost as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Frost (a) flat analytic ice generator', () => {
    it('emits the analytic circle sampler + the shared material parts + the crystal noise, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'f', def: F, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/beerLambert/)
        expect(finalWgsl).toMatch(/opticalThickness/)
        expect(finalWgsl).toMatch(/nudgeNormal/)
        expect(finalWgsl).toMatch(/mxWorleyNoiseFloat2Pub/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        // Sparkle twinkle reads the per-node animated-time field; frost crystals use MaterialX noise.
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/mxNoiseFloat2/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        // Frost declares animatedTime — the _animTime field is registered + the sparkle reads it.
        expect(F.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'f', def: F, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Frost')
    })
})
