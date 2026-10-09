import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import AngularBlur from '@coreroot/shaders/AngularBlur/index'
import {angularBlurTapCoord, MOTION_BLUR_WEIGHTS, MOTION_BLUR_TAP_COUNT} from '@coreroot/gpu/kit/motionBlur'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * AngularBlur port gate (W6-D) — an RTT multi-tap FILTER (requiresRTT/requiresChild, no uvRemap). It
 * RTTs the child and accumulates 32 weighted Gaussian samples rotated around `center`, then
 * unpremultiplies (RTT premultiplied — Twirl trap #2). v1's JS `for` loop + rotation recurrence
 * unrolled at build; this builder unrolls 32 `texture.sample()` calls, each tap's rotated coord from
 * a shared body fn computing `cos/sin(tapIndex·angleStep)` DIRECTLY (== the recurrence). CPU-goldened.
 */
const AB = AngularBlur as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

describe('AngularBlur (a) RTT multi-tap filter path', () => {
    it('RTTs the child, unrolls 32 taps of angularBlurTapCoord, unpremultiplies', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'ab', def: AB, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'ab', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/angularBlurTapCoord/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        expect((finalWgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('AngularBlur (b) weights', () => {
    it('normalized Gaussian weights sum to 1 over 32 taps', () => {
        expect(MOTION_BLUR_WEIGHTS.length).toBe(32)
        expect(MOTION_BLUR_WEIGHTS.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    })
})

describe('AngularBlur (c) CPU golden — rotated tap coordinate', () => {
    it('rotate aspect-corrected (uv - center) by (tapIndex − 15.5)·(intensity·0.005/31), un-correct', () => {
        // center transformed = (cx, 1 - cy_authored); the body flips y back via 1 - center.y.
        const cases = [
            {center: [0.5, 0.5] as const, intensity: 20, uv: [0.7, 0.6] as const, aspect: 1.5, tap: 0},
            {center: [0.5, 0.5] as const, intensity: 100, uv: [0.2, 0.3] as const, aspect: 1.0, tap: 31},
            {center: [0.3, 0.4] as const, intensity: 50, uv: [0.6, 0.9] as const, aspect: 2.0, tap: 16},
        ]
        for (const {center, intensity, uv, aspect, tap} of cases) {
            const cpX = center[0]
            const cpY = 1 - center[1]
            const angleStep = (intensity * 0.005) / 31
            const angle = (tap - 15.5) * angleStep
            const cosA = Math.cos(angle)
            const sinA = Math.sin(angle)
            const acX = (uv[0] - cpX) * aspect
            const acY = uv[1] - cpY
            const rotX = acX * cosA - acY * sinA
            const rotY = acX * sinA + acY * cosA
            const ex = rotX / aspect + cpX
            const ey = rotY + cpY
            const out = angularBlurTapCoord(
                d.vec2f(center[0], center[1]), intensity, d.vec2f(uv[0], uv[1]), aspect, tap,
            ) as unknown as {x: number; y: number}
            expect(out.x).toBeCloseTo(ex, 5)
            expect(out.y).toBeCloseTo(ey, 5)
        }
        // The 32-tap sweep is CENTERED on the source pixel: taps 15 and 16 sit at ∓0.5·step, so
        // they land symmetrically about the un-rotated uv (their midpoint IS the source sample).
        const lo = angularBlurTapCoord(d.vec2f(0.5, 0.5), 20, d.vec2f(0.7, 0.6), 1.5, 15) as unknown as {x: number; y: number}
        const hi = angularBlurTapCoord(d.vec2f(0.5, 0.5), 20, d.vec2f(0.7, 0.6), 1.5, 16) as unknown as {x: number; y: number}
        expect((lo.x + hi.x) / 2).toBeCloseTo(0.7, 6)
        expect((lo.y + hi.y) / 2).toBeCloseTo(0.6, 6)
    })
})

// (d) Detail dials — a non-zero dial switches to the kit's detail gather (same orbit coords,
// per-channel tap weights); all dials at 0 keep the plain gather snapshotted in (a).
describe('AngularBlur (d) detail gather', () => {
    it('bias + jitter switch to the detail gather over angularBlurTapCoord', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'ab', def: AB, parentId: 'root', props: {bias: 0.5, jitter: 0.3}, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'ab', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const wgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(wgsl).toMatch(/motionBlurDetailField/)
        expect(wgsl).toMatch(/motionBlurDetailTapWeight/)
        expect(wgsl).toMatch(/angularBlurTapCoord/)
        expect(wgsl).toMatch(/unpremultiplyAlpha/)
        expect((wgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect(wgsl).toMatchSnapshot('final-pass-detail')
    })
})
