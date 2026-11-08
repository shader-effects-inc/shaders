import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import FlowingGradient from '@coreroot/shaders/FlowingGradient/index'
import {buildRegistry, RootContainer} from './noiseHarness'

/**
 * FlowingGradient port gate (D1-G). Resolve+snapshot the 4-color 2D-blend generator; assert the
 * animatedTime read, the three chained mixColors (row1/row2 by t1, base by t2), fold-lighting
 * compose, and compile-time colorSpace (default oklch). mx noise is hash-based → GPU-only.
 */
const F = FlowingGradient as GpuShaderDefinition

const resolveFinal = (props?: Record<string, unknown>, animTime?: number): string => {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'f', def: F, parentId: 'root', props, metadata: {renderOrder: 0}, animTime},
    ])
    return tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
}

describe('FlowingGradient (a) default generator', () => {
    it('emits flowingFactors + flowingCompose + the animated-time read + the preconverted mix chain', () => {
        const wgsl = resolveFinal()
        // The look lives in FlowingGradient/index.ts — no flowing* kernels remain.
        expect(wgsl).not.toMatch(/flowingFactors|flowingCompose|flowingWarpLevel/)
        expect(wgsl.match(/chainedWarpStep\(/g)!.length).toBeGreaterThanOrEqual(3) // 1 def + 2 levels
        expect(wgsl).toMatch(/_animTime/)
        // Default colorSpace is oklch → PRECONVERTED endpoints (CPU forward conversion into the
        // conv* extraFields): two in-space row mixes + ONE back-converting final mix. The per-pixel
        // forward conversion must not re-enter the body.
        expect(wgsl).toMatch(/mixPreconvertedInSpaceOklch/) // OKLCh default: the hue-aware in-space step
        expect(wgsl).toMatch(/mixPreconvertedColors/)
        expect(wgsl).toMatch(/oklchToOklab/) // back-conversion (once, in the final mix)
        expect(wgsl).not.toMatch(/rgbToOklab/) // forward conversion is CPU-side now
        expect(wgsl).toMatchSnapshot('final-pass-default')
    })
})

describe('FlowingGradient (b) compile-time colorSpace', () => {
    it('linear does not back-convert; colorSpace in the recompile hash', () => {
        expect(resolveFinal({colorSpace: 'linear'})).not.toMatch(/oklchToOklab/)
        const hash = (props: Record<string, unknown>) => {
            const {registry} = buildRegistry([
                {id: 'root', def: RootContainer, parentId: null},
                {id: 'f', def: F, parentId: 'root', props, metadata: {renderOrder: 0}},
            ])
            return collectStructuralHashInputs(registry).join('\n')
        }
        expect(hash({colorSpace: 'linear'})).not.toBe(hash({colorSpace: 'oklch'}))
    })
})
