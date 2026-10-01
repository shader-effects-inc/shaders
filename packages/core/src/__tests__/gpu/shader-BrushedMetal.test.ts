import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import BrushedMetal from '@coreroot/shaders/BrushedMetal/index'
import {materialParts} from '@coreroot/gpu/kit'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * BrushedMetal gate (W7-A → std sweep). A GENERATOR whose brushed-metal material is std algebra in
 * the definition file over shared kit parts (two-scale brushed normal + perspective view ray +
 * 9-tap along-grain studio smear unrolled over the shared studio5Softbox bank + Ward anisotropic
 * specular), inside the `guarded` shape region. GPU-free → flat analytic path. Validates the
 * composition + the animated `_animTime` read + a CPU golden on the shared kit studio bank.
 */
const B = BrushedMetal as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('BrushedMetal (a) flat analytic satin-metal generator', () => {
    it('emits the analytic circle sampler + the shared material parts + noise, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'b', def: B, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/wardAnisotropicSpecular/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/studio5Softbox/)
        expect(finalWgsl).toMatch(/mxNoiseFloat2/)
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(B.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'b', def: B, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('BrushedMetal')
    })
})

describe('BrushedMetal (b) studio5Softbox CPU golden (BrushedMetal sky gain 0.38)', () => {
    it('overhead key bank lights the (0, 0.55) direction; envStr 0 leaves only the sky gradient', () => {
        // envStr=0 → no boxes → only the squared sky gradient. High ey → ~0.38, low ey → ~0.
        const skyTop = materialParts.studio5Softbox(0, 1.0, 1, 0, 0.42, 0, 0, 0.38) as unknown as number
        expect(skyTop).toBeCloseTo(0.38, 5)
        const floor = materialParts.studio5Softbox(0, -1.0, 1, 0, 0.42, 0, 0, 0.38) as unknown as number
        expect(floor).toBeCloseTo(0, 5)
        // The overhead key bank at (0, 0.55) lights up with envStr > 0.
        const keyLit = materialParts.studio5Softbox(0, 0.55, 1, 0, 0.42, 0, 1, 0.38) as unknown as number
        expect(keyLit).toBeGreaterThan(1.0)
    })
})
