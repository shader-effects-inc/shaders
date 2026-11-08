import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Strands from '@coreroot/shaders/Strands/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Strands port gate (Phase W6-B — the batch's hardest). Animated multi-stop generator: a runtime
 * loop over the active strand count accumulates each strand's WORKING-SPACE gradient color via the
 * kit's `gradientStopsInSpace`, and the builder does the single back-conversion at the compile-time
 * colorSpace mode. Two independent clocks (speed + colorSpeed via extraAnimatedTimes) + a pinEdges
 * boolean (transformBoolean). Nested loops → GPU-only (resolve+snapshot, no CPU golden).
 */
const SD = Strands as GpuShaderDefinition

describe('Strands (a) animated multi-stop generator', () => {
    it('final pass runs the strand loop over gradientStopsInSpace, reads BOTH clocks, no RTT', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'sd', def: SD, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/ribbonsField/) // the generic ribbons genre kernel
        expect(finalWgsl).not.toMatch(/strandsField|strandsTonePow/)
        expect(finalWgsl).toMatch(/gradientStopsInSpace/)
        expect(finalWgsl).toMatch(/toneLift/)
        // Default colorSpace is oklab (mode 2) → the back-convert emits oklabToRgb.
        expect(finalWgsl).toMatch(/oklabToRgb/)
        expect(finalWgsl).toMatch(/_animTime\b/)
        expect(finalWgsl).toMatch(/_animTime_color/)
        expect(ir.rttPasses.length).toBe(0)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('a cylindrical colorSpace switches to the circular-mean hue variant of the strand loop', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'sd', def: SD, parentId: 'root', props: {colorSpace: 'oklch'}, metadata: {renderOrder: 0}},
        ])
        const finalWgsl = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/ribbonsFieldOklch/)
        expect(finalWgsl).toMatch(/gradientStopsInSpaceOklch/)
        expect(finalWgsl).toMatch(/atan2/)
    })

    it('is registered with both animated clocks', () => {
        expect(SD.animatedTime).toEqual({speed: 'speed'})
        expect(SD.extraAnimatedTimes).toEqual({color: 'colorSpeed'})
    })

    it('colorSpace is part of the structural hash', () => {
        const hashWith = (props: Record<string, unknown>) => {
            const {registry} = buildRegistry([
                {id: 'root', def: RootContainer, parentId: null},
                {id: 'sd', def: SD, parentId: 'root', props, metadata: {renderOrder: 0}},
            ])
            return collectStructuralHashInputs(registry).join('\n')
        }
        expect(hashWith({colorSpace: 'oklab'})).not.toBe(hashWith({colorSpace: 'linear'}))
    })
})
