import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Obsidian from '@coreroot/shaders/Obsidian/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Obsidian gate. A shape-effect material: dark body on facing surfaces, a multi-stop palette folded
 * along a flow frame on the grazing walls, a Fresnel-weighted studio reflection. GPU-free → flat
 * analytic path. Validates the composition (multi-stop palette by default, the mirror fold rather
 * than the fwidth wrap inside the guard, the shared studio bank) + the animated clock.
 */
const O = Obsidian as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Obsidian (a) flat analytic dark-glass material', () => {
    it('emits the analytic circle sampler + the material parts + the folded stop palette, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'o', def: O, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/bevelledFlatNormal/)
        expect(finalWgsl).toMatch(/perspectiveViewRay/)
        expect(finalWgsl).toMatch(/patternCoords/)
        expect(finalWgsl).toMatch(/gradientEdgeMirror/)
        expect(finalWgsl).not.toMatch(/fwidth/)
        expect(finalWgsl).toMatch(/gradientStopsInSpace/) // multi-stop palette is the default
        expect(finalWgsl).toMatch(/studio5Softbox/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('falls back to the colorA/colorB pair when stops are cleared', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'o', def: O, parentId: 'root', metadata: {renderOrder: 0}, props: {stops: null}},
        ])
        const ir = composeNodeTree(registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).not.toMatch(/gradientStopsInSpace/)
        expect(finalWgsl).toMatch(/mixColors/)
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(O.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'o', def: O, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Obsidian')
    })
})
