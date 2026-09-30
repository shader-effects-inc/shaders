import {describe, it, expect} from 'vitest'
import * as std from '@coreroot/std'

/**
 * The public authoring surface (`shaders/std`). Every vocabulary module under `src/std` must be
 * reachable from the one entry an application can import — a word that exists in the repo but
 * not here is a word custom components cannot use. Add a line when a module is added.
 */
describe('shaders/std exposes the whole vocabulary', () => {
    it('definition + wgsl + registry', () => {
        for (const name of ['defineShader', 'defineStd', 'wgsl', 'p', 'ctx', 'schema', 'registerShader', 'getRegisteredShader', 'animatedTime', 'colorStopsPropConfig', 'listPropConfig', 'listOf', 'accumulate', 'uniformOf', 'resolveArg', 'paintFrame']) {
            expect(typeof (std as Record<string, unknown>)[name], name).not.toBe('undefined')
        }
    })

    it('namespaces', () => {
        const namespaces: Record<string, string> = {
            math: 'smoothstep', shape: 'circle', paint: 'rampOver', gradients: 'beam', patterns: 'stripeBands',
            light: 'glowAt', noise: 'fractalNoise', materials: 'keyLightAt', volume: 'volumeMarch',
            compose: 'dithered', figures: 'strokedSegment', media: 'imageMedia', radiance: 'irradianceField',
            voxels: 'voxelSurface', frames: 'surfaceOf', warps: 'twirl', motion: 'oscillate', mask: 'softDisc',
            signal: 'pointer',
        }
        for (const [ns, sample] of Object.entries(namespaces)) {
            const mod = (std as Record<string, Record<string, unknown>>)[ns]
            expect(mod, ns).toBeDefined()
            expect(typeof mod[sample], `${ns}.${sample}`).toBe('function')
        }
    })

    it('effect and simulation families', () => {
        const effects = std.effects as Record<string, Record<string, unknown>>
        for (const [family, sample] of Object.entries({blurs: 'gaussianBlur', color: 'saturate', edgeGlow: 'edgeGlowCompose', fracture: 'crackGeom', instances: 'repeatInstances', lens: 'spectralLens', overlay: 'detectionOverlay', pointerFields: 'pointerSplatField', reveal: 'reveal', stylize: 'kuwahara'})) {
            expect(typeof effects[family]?.[sample], `effects.${family}.${sample}`).toBe('function')
        }
        const sim = std.sim as Record<string, Record<string, unknown>>
        for (const [family, sample] of Object.entries({agents: 'agentSim', agentForces: 'composeForces3', agentFrame: 'expDecay', agentRender: 'splat', feedback: 'feedbackSim', fluids: 'fluidSim', grids: 'gridSim', shapeFields: 'shapeField'})) {
            // Some words are grouped objects (`splat`, `shapeField`), so presence is the contract.
            expect(sim[family]?.[sample], `sim.${family}.${sample}`).toBeDefined()
        }
        // The grid-simulation nouns stay top-level.
        expect(typeof std.simulate).toBe('object')
        expect(typeof std.op).toBe('object')
    })

    it('filters, slots, values and warps-map combinators', () => {
        for (const name of ['tintToward', 'pointwise', 'gather', 'displaceBy', 'paintThrough', 'radialMask', 'crosses', 'recompileWhen', 'isZero', 'isValue', 'allOf', 'identityWhenever', 'lerpToIdentity', 'selectMap', 'pointer', 'pointerSpeed', 'layered', 'layers', 'transformColor', 'transformPosition', 'transformColorSpace']) {
            expect(typeof (std as Record<string, unknown>)[name], name).toBe('function')
        }
    })
})
