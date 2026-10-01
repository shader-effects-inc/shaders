import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Nebula from '@coreroot/shaders/Nebula/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Nebula gate. A shapeEffect material on the shapedSurface spine: an unrolled volumetric
 * emission/absorption march through the shape interior (3D fbm gas + worley star planes) under a
 * glass shell (studio softboxes + key glint + spectral rim). The default shape is prism3D, so the
 * GPU-free default resolve exercises the VOLUMETRIC material path (analytic 3D sampler, marched
 * normal); the circleSDF override exercises the FLAT path (bevelled flat normal) — the both-2D-
 * and-3D contract of the material.
 */
const N = Nebula as GpuShaderDefinition

// Nebula's gates describe the prism3D volumetric path; shape effects default to a sphere3D since
// 4.0, so the prism is passed explicitly.
const PRISM_SHAPE = {shape: JSON.stringify({type: 'prism3D', radius: 0.28, height: 0.24, rotX: 25, rotY: 30, rotZ: 0}), shapeType: 'prism3D'}

describe('Nebula (a) prism3D — the volumetric material path', () => {
    it('emits the analytic prism3D sampler, the marched normal, the gas march noise and the shell parts', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'n', def: N, parentId: 'root', props: PRISM_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // self-contained material, no child RTT
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        // Shape routing: the 3D prism resolves to the analytic 3D field sampler.
        expect(finalWgsl).toMatch(/analyticSdf_prism3D/)
        expect(finalWgsl).toMatch(/volumetricNormal/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        // The interior march: 3D fbm gas + optical chord + worley star planes + march jitter.
        expect(finalWgsl).toMatch(/mxNoiseFloat3/)
        expect(finalWgsl).toMatch(/opticalThickness/)
        expect(finalWgsl).toMatch(/mxWorleyNoiseFloat2Pub/)
        expect(finalWgsl).toMatch(/hash12/)
        // The glass shell: studio reflection, key light, spectral rim, filmic shoulder.
        expect(finalWgsl).toMatch(/studio5Softbox/)
        expect(finalWgsl).toMatch(/keyLight/)
        expect(finalWgsl).toMatch(/tonemapNeutral/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        // The churn/twinkle clock.
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatchSnapshot('final-pass-prism3D')
    })

    it('registers the animatedTime clock and the structural hash entry', () => {
        expect(N.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'n', def: N, parentId: 'root', props: PRISM_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Nebula')
    })
})

describe('Nebula (b) circleSDF override — the flat material path', () => {
    it('emits the flat analytic sampler + the bevelled flat normal', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {
                id: 'n', def: N, parentId: 'root', metadata: {renderOrder: 0},
                props: {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'},
            },
        ])
        const ir = composeNodeTree(registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/bevelledFlatNormal/)
        expect(finalWgsl).not.toMatch(/analyticSdf_prism3D/)
        // The gas march + shell are path-independent.
        expect(finalWgsl).toMatch(/mxNoiseFloat3/)
        expect(finalWgsl).toMatch(/opticalThickness/)
        expect(finalWgsl).toMatch(/studio5Softbox/)
        expect(finalWgsl).toMatchSnapshot('final-pass-circleSDF')
    })
})
