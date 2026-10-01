import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Plastic from '@coreroot/shaders/Plastic/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Plastic gate (W7-A → std sweep). A GENERATOR whose glossy-plastic material is std algebra in the
 * definition file over shared kit parts (pillow normal + ridged crumple + procedural studio +
 * Blinn glint + rim, inside the `guarded` shape region), combined with a body-color GRADIENT
 * (multi-stop / two-color, colorSpace compile-time) resolved at builder level. GPU-free → flat
 * analytic path, two-color default (linear). Its env orbit maps v1's CPU envDrift onto the
 * animatedTime clock (×6).
 */
const P = Plastic as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Plastic (a) flat analytic two-color gloss generator', () => {
    it('emits the analytic circle sampler + the shared material parts + two-color mixColors, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'p', def: P, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/nudgeNormal/)
        expect(finalWgsl).toMatch(/keyLight/)
        expect(finalWgsl).toMatch(/grazingView/)
        expect(finalWgsl).toMatch(/mixColors/) // two-color body gradient (default: no stops)
        expect(finalWgsl).toMatch(/_animTime/) // env orbit drift
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(P.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'p', def: P, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Plastic')
    })
})

