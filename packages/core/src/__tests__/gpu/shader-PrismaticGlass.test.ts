import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import PrismaticGlass from '@coreroot/shaders/PrismaticGlass/index'
import SolidColor from '@coreroot/shaders/SolidColor/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * PrismaticGlass gate. A shapeEffect material on the shapedSurface spine that reads the layers
 * beneath (child RTT): the glass body rebuilt from the field (−half chord + analytic slope on 3D,
 * the rounded slab on flat shapes), derivative-measured pixel footprints, a real spectral loop
 * (`sampleSum` → `for (… < 8u …)`) tracing each slice through both faces (`refract` twice, the
 * fire bounce), a crisp area-light studio, a thin-film coat — and, only when `caustics` > 0, the
 * cast shadow + colour-split caustic taps.
 */
const PG = PrismaticGlass as GpuShaderDefinition
const SC = SolidColor as GpuShaderDefinition

function compose(glassProps: Record<string, unknown> = {}): string {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'bg', def: SC, parentId: 'root', props: {color: '#203040'}, metadata: {renderOrder: 0}},
        {id: 'pg', def: PG, parentId: 'root', props: glassProps, metadata: {renderOrder: 1}},
    ])
    const ir = composeNodeTree(registry)
    return tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
}

describe('PrismaticGlass (a) sphere3D default — the volumetric body', () => {
    it('emits the 3D sampler, the derivative footprints, the spectral loop and both refractions', () => {
        const wgsl = compose()
        expect(wgsl).toMatch(/analyticSdf_sphere3D/)
        expect(wgsl).toMatch(/sdfSpaceUV/)
        // Footprints are PER-PIXEL screen derivatives (fine, never shared across a 2×2 block),
        // emitted before the guard; the lookup's Jacobian spreads each slice over the footprint.
        expect(wgsl).toMatch(/fwidthFine\(/)
        expect(wgsl).toMatch(/dpdxFine\(/)
        expect(wgsl).toMatch(/lookupStepX/)
        expect(wgsl).not.toMatch(/\bfwidth\(|\bdpdx\(/)
        // GPU-free resolve has no marched field, so no analytic depth slope (tilt falls to 0).
        expect(wgsl).not.toMatch(/fieldDepthSlopeArg/)
        // One real loop over the spectrum: the body is emitted once.
        expect(wgsl).toMatch(/for \(var \w+ = 0u; \w+ < 8u;/)
        expect(wgsl.match(/textureSampleLevel\(/g)?.length).toBe(1)
        // In through the front face and out through the back, plus the fire bounce.
        expect(wgsl.match(/refract\(/g)!.length).toBeGreaterThanOrEqual(3)
        expect(wgsl).toMatch(/reflect\(/)
        expect(wgsl).toMatch(/beerLambert/)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
        expect(wgsl).toMatch(/_animTime/)
        expect(wgsl).toMatchSnapshot('final-pass-sphere3D')
    })

    it('compiles the caustics out at 0 (the default)', () => {
        const wgsl = compose()
        expect(wgsl).not.toMatch(/castShadow/)
        expect(wgsl).not.toMatch(/castLight/)
    })

    it('registers the animatedTime clock and the structural hash entry', () => {
        expect(PG.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'bg', def: SC, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'pg', def: PG, parentId: 'root', metadata: {renderOrder: 1}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('PrismaticGlass')
    })
})

describe('PrismaticGlass (b) circleSDF — the flat rounded slab', () => {
    it('emits the flat analytic sampler and no 3D tilt', () => {
        const wgsl = compose({shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'})
        expect(wgsl).toMatch(/analyticSdf_circleSDF/)
        expect(wgsl).not.toMatch(/analyticSdf_sphere3D/)
        expect(wgsl).not.toMatch(/midDepth/)
        expect(wgsl).toMatch(/slabSlope/)
        expect(wgsl).toMatch(/for \(var \w+ = 0u; \w+ < 8u;/)
        expect(wgsl).toMatchSnapshot('final-pass-circleSDF')
    })
})

describe('PrismaticGlass (c) caustics on — the cast shadow and caustic', () => {
    it('adds the four cast taps and the shadow/light terms', () => {
        const wgsl = compose({caustics: 1})
        expect(wgsl).toMatch(/castShadow/)
        expect(wgsl).toMatch(/castLight/)
        expect(wgsl).toMatchSnapshot('final-pass-caustics')
    })
})
