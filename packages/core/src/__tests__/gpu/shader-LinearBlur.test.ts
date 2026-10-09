import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import LinearBlur from '@coreroot/shaders/LinearBlur/index'
import {linearBlurTapCoord, MOTION_BLUR_WEIGHTS, MOTION_BLUR_TAP_COUNT} from '@coreroot/gpu/kit/motionBlur'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * LinearBlur port gate (W6-D) — an RTT multi-tap FILTER (requiresRTT/requiresChild, no uvRemap). It
 * RTTs the child and accumulates 32 weighted Gaussian samples along the (aspect-corrected) blur
 * direction, then unpremultiplies (the RTT stores premultiplied alpha — Twirl trap #2). v1's JS `for`
 * loop unrolled at graph-build time; this builder unrolls the same 32 `texture.sample()` calls into a
 * flat weighted sum. `linearBlurTapCoord` is pure trig → CPU-goldened.
 */
const LB = LinearBlur as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

describe('LinearBlur (a) RTT multi-tap filter path', () => {
    it('RTTs the child, unrolls 32 taps of linearBlurTapCoord, unpremultiplies', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'lb', def: LB, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'lb', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/linearBlurTapCoord/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        // One textureSample per tap.
        expect((finalWgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('LinearBlur (b) weights', () => {
    it('normalized Gaussian weights sum to 1 over 32 taps', () => {
        expect(MOTION_BLUR_WEIGHTS.length).toBe(32)
        expect(MOTION_BLUR_WEIGHTS.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    })
})

describe('LinearBlur (c) CPU golden — tap sample coordinate', () => {
    it('uv + (dir·intensity/viewport·2) · tapT (angle in degrees)', () => {
        const cases = [
            {uv: [0.5, 0.5] as const, angle: 0, intensity: 30, vp: [800, 600] as const, tapT: 0.5},
            {uv: [0.3, 0.7] as const, angle: 90, intensity: 60, vp: [1024, 768] as const, tapT: -0.5},
            {uv: [0.4, 0.4] as const, angle: 45, intensity: 100, vp: [1000, 1000] as const, tapT: 0},
        ]
        for (const {uv, angle, intensity, vp, tapT} of cases) {
            const angleRad = angle * (Math.PI / 180)
            const aspect = vp[0] / vp[1]
            const dirX = Math.cos(angleRad) / aspect
            const dirY = Math.sin(angleRad)
            const bvX = ((dirX * intensity) / vp[0]) * 2
            const bvY = ((dirY * intensity) / vp[1]) * 2
            const ex = uv[0] + bvX * tapT
            const ey = uv[1] + bvY * tapT
            const out = linearBlurTapCoord(
                d.vec2f(uv[0], uv[1]), angle, intensity, d.vec2f(vp[0], vp[1]), tapT,
            ) as unknown as {x: number; y: number}
            expect(out.x).toBeCloseTo(ex, 5)
            expect(out.y).toBeCloseTo(ey, 5)
        }
    })
})

// (d) Detail dials — a non-zero dial switches to the kit's detail gather (same linear coords,
// per-channel tap weights, falloff/focus measured from the canvas centre since a streak has no
// centre prop); all dials at 0 keep the plain gather snapshotted in (a).
describe('LinearBlur (d) detail gather', () => {
    it('highlights + focus switch to the detail gather over linearBlurTapCoord', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'lb', def: LB, parentId: 'root', props: {highlights: 0.7, focus: 0.2}, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'lb', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl).toMatch(/motionBlurDetailField\(uv, vec2f\(0\.5, 0\.5\)/)
        expect(wgsl).toMatch(/motionBlurDetailTapWeight/)
        expect(wgsl).toMatch(/linearBlurTapCoord/)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
        expect((wgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect(wgsl).toMatchSnapshot('final-pass-detail')
    })
})
