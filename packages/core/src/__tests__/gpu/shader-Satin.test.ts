import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Satin from '@coreroot/shaders/Satin/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Satin gate — an analytic draped-cloth GENERATOR written as a std recipe: `foldSet` +
 * `drapedFolds` (the fold height field, unrolled per compile-time fold), `fdCurvature` (five
 * taps → slope + Laplacian), the thread grain (`grainNoise`), two Ward anisotropic lobes (the
 * streak + the broad bloom), the `sheenLobe` velvet term and the neutral tone-map. No RTT, no
 * compute. Validates the composition, the animated `_animTime` read and the compile-time fold
 * count.
 */
const S = Satin as GpuShaderDefinition

const resolveFinal = (props?: Record<string, unknown>): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 's', def: S, parentId: 'root', props, metadata: {renderOrder: 0}},
    ])
    const ir = composeNodeTree(registry)
    expect(ir.rttPasses.length).toBe(0) // generator: no RTT pass
    return tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
}
const hash = (props: Record<string, unknown>): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 's', def: S, parentId: 'root', props, metadata: {renderOrder: 0}},
    ])
    return collectStructuralHashInputs(registry).join('\n')
}

describe('Satin (a) analytic draped-cloth generator', () => {
    it('emits the fold field + curvature taps, the thread grain, the two Ward lobes and the tone-map', () => {
        const wgsl = resolveFinal()
        // One definition + the streak + the bloom.
        expect(wgsl.match(/wardAnisotropicSpecular\(/g)!.length).toBeGreaterThanOrEqual(3)
        expect(wgsl).not.toMatch(/studio5Softbox/)
        expect(wgsl).toMatch(/mxNoiseFloat2/)
        // The thread grain is sampled in cloth coordinates (clothCoords), not the screen point.
        expect(wgsl).toMatch(/let clothAcross/)
        expect(wgsl).toMatch(/let onCloth/)
        expect(wgsl).toMatch(/nudgeNormal/)
        expect(wgsl).toMatch(/perspectiveViewRay/)
        expect(wgsl).toMatch(/tonemapNeutral/)
        expect(wgsl).toMatch(/_animTime/)
        // The fold layout's seeded hashes (one `43758.5453` per hash, four per fold, seven folds).
        expect(wgsl.match(/43758\.5453/g)!.length).toBe(28)
        expect(wgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop)', () => {
        expect(S.animatedTime).toEqual({speed: 'speed'})
    })
})

describe('Satin (b) foldCount is compile-time', () => {
    it('unrolls one fold per count and recompiles when the count changes', () => {
        expect(resolveFinal({foldCount: 2}).match(/43758\.5453/g)!.length).toBe(8)
        expect(resolveFinal({foldCount: 12}).match(/43758\.5453/g)!.length).toBe(48)
        expect(hash({foldCount: 7})).not.toBe(hash({foldCount: 2}))
    })
})
