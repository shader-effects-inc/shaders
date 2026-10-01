import {describe, it, expect, vi} from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import {tgpu, d, sdf3d} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {buildRegistry, RootContainer} from './_patternHarness'
import Flip from '@coreroot/shaders/Flip/index'
import LightEdge from '@coreroot/shaders/LightEdge/index'
import LightLeak from '@coreroot/shaders/LightLeak/index'
import Heatmap from '@coreroot/shaders/Heatmap/index'
import KeyFrames from '@coreroot/shaders/KeyFrames/index'
import {buildPursuitKernel as buildTrackKernel, buildFeatureGridKernel, featureScoreFn} from '@coreroot/gpu/kit/trackerSim'
import DataMosh from '@coreroot/shaders/DataMosh/index'

/**
 * Batch port gate for the 2026-07 effects drop: Flip, ColorKey, LightEdge, LightLeak, Heatmap,
 * KeyFrames, DataMosh + the five animated 3D SDF shapes (ribbon/blob/torusKnot/gyroid/gyroscope).
 * GPU-free: compose → tgpu.resolve, asserting each shader's body fns appear in the WGSL (no
 * snapshots — the per-shader gates own those). The 3D shapes get CPU-golden inside/outside sanity
 * plus a baked-setup march resolve (the createAnalytic3dSdfSetup branch chain).
 */

const v3 = (x: number, y: number, z: number) => d.vec3f(x, y, z)

// A trivial colored generator child (an RTT source for the filters).
const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

// WGSL reserved words that read like natural local names — `tgpu.resolve` does NOT validate
// these (only the browser's WGSL parser rejects them), so raw-WGSL builders must dodge them.
const WGSL_RESERVED_LOCALS = /\b(?:let|var)\s+(?:meta|half|filter|common|active|target|self|super|type|mod|pass|new|delete|do|export|import|package|std)\s*[=:]/
const expectNoReservedLocals = (wgsl: string): void => {
    expect(wgsl).not.toMatch(WGSL_RESERVED_LOCALS)
}

const resolveWithChild = (def: GpuShaderDefinition, props?: Record<string, unknown>): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'fx', def, parentId: 'root', props, metadata: {renderOrder: 0}},
        {id: 'gen', def: Generator, parentId: 'fx', metadata: {renderOrder: 0}},
    ])
    return tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
}
const resolveStandalone = (def: GpuShaderDefinition, props?: Record<string, unknown>): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'fx', def, parentId: 'root', props, metadata: {renderOrder: 0}},
    ])
    return tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
}
const hashOf = (def: GpuShaderDefinition, props: Record<string, unknown>): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'fx', def, parentId: 'root', props, metadata: {renderOrder: 0}},
    ])
    return collectStructuralHashInputs(registry).join('\n')
}

// The flat (2D) path these gates describe. Shape effects default to a sphere3D since 4.0, so
// the circleSDF shape is passed explicitly.
const FLAT_SHAPE = {shape: JSON.stringify({type: 'circleSDF', radius: 0.35}), shapeType: 'circleSDF'}

describe('Flip — RTT filter + uvRemap', () => {
    it('samples the flipped coordinate (composer takes the analytic uvRemap fast path)', () => {
        const wgsl = resolveWithChild(Flip as GpuShaderDefinition)
        expect(wgsl).toMatch(/flipUV/)
        expect(wgsl).toMatch(/textureSample\(rtt_/)
    })
})

describe('LightEdge — border light as an SDF shape effect', () => {
    it('emits the border band + per-color accumulation + compose over the analytic circle field', () => {
        const wgsl = resolveStandalone(LightEdge as GpuShaderDefinition, FLAT_SHAPE)
        expect(wgsl).toMatch(/edgeGlowBand/)
        expect(wgsl).toMatch(/orbitSpotsAccum/)
        expect(wgsl).toMatch(/heartbeatPulse/)
        expect(wgsl).toMatch(/edgeGlowCompose/)
        expect(wgsl).toMatch(/analyticSdf_circleSDF/)
        expect(wgsl).toMatch(/sdfSpaceUV/)
        expect(wgsl).toMatch(/_animTime/)
    })
    it('the stop count drives the per-color unroll (count fingerprinting lives in the renderer extraHashInputs)', () => {
        const stops2 = [{color: '#ff0000', position: 0}, {color: '#00ff00', position: 1}]
        const stops5 = [
            {color: '#ff0000', position: 0}, {color: '#00ff00', position: 0.25}, {color: '#0000ff', position: 0.5},
            {color: '#ffff00', position: 0.75}, {color: '#ff00ff', position: 1},
        ]
        const accumCalls = (w: string): number => (w.match(/orbitSpotsAccum\(/g) ?? []).length
        const w2 = resolveStandalone(LightEdge as GpuShaderDefinition, {stops: stops2})
        const w5 = resolveStandalone(LightEdge as GpuShaderDefinition, {stops: stops5})
        expect(w5).toMatch(/colorAtIndex/) // multi-stop path active
        expect(accumCalls(w5)).toBeGreaterThan(accumCalls(w2))
    })
    it('colorSpace is structural and selects the blend variant', () => {
        expect(hashOf(LightEdge as GpuShaderDefinition, {colorSpace: 'linear'}))
            .not.toBe(hashOf(LightEdge as GpuShaderDefinition, {colorSpace: 'oklch'}))
        expect(resolveStandalone(LightEdge as GpuShaderDefinition, {colorSpace: 'oklch'})).toMatch(/oklab|oklch/i)
    })
})

describe('LightLeak — film exposure composite', () => {
    it('emits the composed leak parts + composite over the child with animated time', () => {
        const wgsl = resolveWithChild(LightLeak as GpuShaderDefinition)
        expect(wgsl).toMatch(/screenGlowComposite/)
        expect(wgsl).toMatch(/heatRamp3/)
        expect(wgsl).toMatch(/leakBloom/)
        expect(wgsl).toMatch(/_animTime/)
    })
})

describe('Heatmap — SDF shape effect (flat analytic path, GPU-free)', () => {
    it('emits the heat-field algebra (cold-front value noise + dither hash + ramp) over the analytic circle sampler', () => {
        const wgsl = resolveStandalone(Heatmap as GpuShaderDefinition, FLAT_SHAPE)
        expect(wgsl).toMatch(/value12/)
        expect(wgsl).toMatch(/hash12/)
        expect(wgsl).toMatch(/gradientStopsInSpace/)
        expect(wgsl).toMatch(/analyticSdf_circleSDF/)
        expect(wgsl).toMatch(/sdfSpaceUV/)
    })
})

describe('raw-WGSL emitters — no WGSL reserved words as local names (source scan)', () => {
    // A raw builder's `s.push(\`let meta = …\`)` resolves fine (tgpu doesn't validate reserved
    // words) but the BROWSER's WGSL parser rejects it. GPU-gated emission branches never appear
    // in the GPU-free resolve, so scan the emitter SOURCE of every shader instead.
    it('no shader emits `let <reserved> =` inside a template string', () => {
        const shadersDir = path.resolve(process.cwd(), 'src/shaders')
        const offenders: string[] = []
        const emitPattern = /push\(`[^`]*\b(?:let|var)\s+(?:meta|half|target|filter|common|active|self|super|type|mod|pass|new|do|std)\b/
        for (const dir of fs.readdirSync(shadersDir)) {
            const file = path.join(shadersDir, dir, 'index.ts')
            if (!fs.existsSync(file)) continue
            const src = fs.readFileSync(file, 'utf8')
            if (emitPattern.test(src)) offenders.push(dir)
        }
        expect(offenders).toEqual([])
    })
})

describe('KeyFrames — motion-tracking overlay (GPU-free: sim skipped, passthrough)', () => {
    it('resolves the raw-WGSL overlay over an RTT child', () => {
        const wgsl = resolveWithChild(KeyFrames as GpuShaderDefinition)
        expect(wgsl).toMatch(/fwidth/)
        expect(wgsl).toMatch(/textureSampleLevel/)
        expectNoReservedLocals(wgsl)
    })
    it('detect mode is structural; tracker count is runtime (no recompile on scrub)', () => {
        expect(hashOf(KeyFrames as GpuShaderDefinition, {detect: 'bright'}))
            .not.toBe(hashOf(KeyFrames as GpuShaderDefinition, {detect: 'dark'}))
        expect(hashOf(KeyFrames as GpuShaderDefinition, {trackers: 1}))
            .toBe(hashOf(KeyFrames as GpuShaderDefinition, {trackers: 50}))
    })
    it('the feature-grid + sim kernels resolve to WGSL (GPU-gated — never seen by the composition resolve)', () => {
        const wgsl = tgpu.resolve([buildTrackKernel('bright') as never, buildFeatureGridKernel(featureScoreFn('bright')) as never], {names: 'strict'})
        expect(wgsl).toMatch(/textureStore/)
        expect(wgsl).toMatch(/featureGridScan/)
        expect(wgsl).toMatch(/trackerPursuit/)
    })
})

describe('DataMosh — GPU-free fallback (compute skipped without a device)', () => {
    it('falls back to the live child passthrough', () => {
        const wgsl = resolveWithChild(DataMosh as GpuShaderDefinition)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
    })
})

// ═══════════════════════════════════════════════════════════════════════════════════════
// New 3D shapes — CPU goldens + baked-setup march resolve
// ═══════════════════════════════════════════════════════════════════════════════════════

describe('kit/sdf3d new animated primitives — CPU golden sanity', () => {
    it('sdRibbon: centre inside (−thickness), beyond the length cap outside', () => {
        // Zero wave/twist → flat band: centre distance is −th (the thinner half-extent).
        const dc = sdf3d.sdRibbon(v3(0, 0, 0), 0.3, 0.035, 0.42, 0, 0, 0, 0, 0, 0, 1) as unknown as number
        expect(dc).toBeCloseTo(-0.035, 5)
        const dOut = sdf3d.sdRibbon(v3(0.62, 0, 0), 0.3, 0.035, 0.42, 0, 0, 0, 0, 0, 0, 1) as unknown as number
        expect(dOut).toBeCloseTo(0.2, 5)
    })
    it('sdWobbleBlob: zero-phase centre = plain sphere; wobble displaces the surface', () => {
        const dc = sdf3d.sdWobbleBlob(v3(0, 0, 0), 0.3, 0.07, 5, 0, 0, 1) as unknown as number
        expect(dc).toBeCloseTo(-0.3, 5)
        const dOut = sdf3d.sdWobbleBlob(v3(0.5, 0, 0), 0.3, 0, 5, 0, 0, 1) as unknown as number
        expect(dOut).toBeCloseTo(0.2, 5)
    })
    it('sdGyroscope: centre is inside the core; a point on the outer band centreline is −thick/2', () => {
        const dc = sdf3d.sdGyroscope(v3(0, 0, 0), 0.34, 0.1, 0.028, 0.006, 0.14, 1, 0, 1, 0, 1, 0) as unknown as number
        expect(dc).toBeCloseTo(-0.14, 5)
        const ring = sdf3d.sdGyroscope(v3(0.34, 0, 0), 0.34, 0.1, 0.028, 0.006, 0.14, 1, 0, 1, 0, 1, 0) as unknown as number
        expect(ring).toBeCloseTo(-0.014, 5)
    })
})

describe('kit/sdf3d new shapes — baked setup + march resolve to WGSL', () => {
    function mockRoot() {
        const uniform = {buffer: {}, write: vi.fn(), patch: vi.fn()}
        return {createUniform: vi.fn(() => uniform)} as never
    }
    const NEW_TYPES = ['ribbon3D', 'blob3D', 'gyroscope3D'] as const
    it.each(NEW_TYPES)('%s bakes, updates, and its march resolves', (type) => {
        const cfg = {type}
        const setup = sdf3d.createAnalytic3dSdfSetup(mockRoot(), type, cfg, () => cfg)
        setup.update({deltaTime: 0.016}) // advances the CPU animation state without throwing
        const march = sdf3d.buildRaymarchedFieldFn(setup.sdfFn)
        const wgsl = tgpu.resolve([march as never], {names: 'strict'})
        expect(typeof wgsl).toBe('string')
        expect(wgsl).toMatch(/raymarchedField/)
    })
    it('every new type reports a positive bounding radius', () => {
        for (const type of NEW_TYPES) {
            expect(sdf3d.shape3dBoundingRadius({type})).toBeGreaterThan(0.1)
        }
    })
})
