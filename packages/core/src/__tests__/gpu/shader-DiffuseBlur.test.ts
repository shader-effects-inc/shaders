import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import DiffuseBlur from '@coreroot/shaders/DiffuseBlur/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * DiffuseBlur port gate (W6-D) — an RTT FILTER (requiresRTT/requiresChild, no uvRemap → always the
 * fragment path, matching v1 which shipped fragment-only because the offset is a per-pixel hash). It
 * RTTs the child, samples it ONCE at a random displaced UV with compile-time edge handling, and
 * unpremultiplies (the RTT stores premultiplied alpha — Twirl trap #2). `diffuseBlurUV` is pure
 * sin/fract → CPU-goldenable.
 */
const DB = DiffuseBlur as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

describe('DiffuseBlur (a) RTT single-displaced-sample filter path', () => {
    it('RTTs the child, samples at the displaced UV, unpremultiplies (default stretch)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'db', def: DB, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'db', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/diffuseBlurUV/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        // stretch (mode 0): no edge helper emitted (the linearClamp sampler clamps).
        expect(finalWgsl).not.toMatch(/edgeMirrorUV|edgeWrapUV|edgeTransparentMask/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('compile-time edges=mirror emits edgeMirrorUV', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'db', def: DB, parentId: 'root', props: {edges: 'mirror'}, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'db', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/edgeMirrorUV/)
    })
})

// NO CPU golden: the displacement is now `noise.hash22` (integer bitcast via raw WGSL
// `bitcast<u32>` — the iOS-safe hash), which is not CPU-executable, so the whole fn is GPU-only
// (the "hash-based → GPU-only, don't CPU-golden" rule; the resolve snapshot + smoke cover it).

// (b) chromatic — off at 0 (the single tap snapshotted above); a non-zero value adds a red and a
// blue tap on their own salted hashes, with green and alpha from the shared tap.
describe('DiffuseBlur (b) chromatic scatter', () => {
    it('chromatic > 0 samples three displaced taps and recombines per channel', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'db', def: DB, parentId: 'root', props: {chromatic: 0.5}, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'db', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/diffuseBlurChannelUV/)
        expect(finalWgsl).toMatch(/diffuseBlurUV\(/)
        expect((finalWgsl.match(/textureSample\(rtt_/g) ?? []).length).toBe(3)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatchSnapshot('final-pass-chromatic')
    })

    it('chromatic 0 keeps the single-tap gather', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'db', def: DB, parentId: 'root', props: {chromatic: 0}, metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'db', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).not.toMatch(/diffuseBlurChannelUV/)
        expect((finalWgsl.match(/textureSample\(rtt_/g) ?? []).length).toBe(1)
    })
})
