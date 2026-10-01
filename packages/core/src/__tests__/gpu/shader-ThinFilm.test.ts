import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import ThinFilm from '@coreroot/shaders/ThinFilm/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * ThinFilm port gate (W7-A). A transparent-interior GENERATOR that consumes the shared
 * `kit/effects/thinFilm` `applyThinFilmEffect` builder (like Neon consumes applyNeonEffect).
 * GPU-free → flat analytic path (default circleSDF); the compute volumetric field is skipped.
 * Validates: the analytic sampler + `thinFilmShade` + the rainbow spectrum + `thinFilmCompose` +
 * the animated `_animTime` read all reach the emitted WGSL, plus the custom-mode colorSpace branch.
 */
const T = ThinFilm as GpuShaderDefinition

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('ThinFilm (a) flat rainbow iridescent generator', () => {
    it('emits the analytic circle sampler + thinFilmShade + thinFilmRainbow + thinFilmCompose, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 't', def: T, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/thinFilmShade/)
        expect(finalWgsl).toMatch(/thinFilmRainbow/)
        expect(finalWgsl).toMatch(/thinFilmCompose/)
        // Spectrum rotation reads the per-node animated-time field.
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(T.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 't', def: T, parentId: 'root', props: FLAT_SHAPE, metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('ThinFilm')
    })
})

describe('ThinFilm (b) custom 3-color cycle', () => {
    it('mode:custom emits thinFilmCustomParams + the colorSpace mixColors chain (not the rainbow palette)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 't', def: T, parentId: 'root', props: {mode: 'custom', colorSpace: 'oklch'}, metadata: {renderOrder: 0}},
        ])
        const finalWgsl = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/thinFilmCustomParams/)
        expect(finalWgsl).toMatch(/mixColors/)
        expect(finalWgsl).not.toMatch(/thinFilmRainbow/)
    })
})
