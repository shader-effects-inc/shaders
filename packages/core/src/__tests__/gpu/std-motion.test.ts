import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {defineShader, p, paintThrough} from '@coreroot/std'
import {cycle, cycleSeed, oscillate, pulseTrain, oscillating} from '@coreroot/std/motion'
import {twirl} from '@coreroot/std/warps'
import {solidColor} from '@coreroot/std/paint/gradients'
import {add, mul, splat3, vec4} from '@coreroot/std/math'
import {buildRegistry, RootContainer} from './_patternHarness'
import {transformColor} from '@coreroot/utilities/transformations'
import {centerPropConfig, edgesPropConfig} from '@coreroot/utilities/propConfigs'

/**
 * Gates for the vocabulary built ahead of consumers: the motion words (cycle/pulse/
 * oscillate/seed), slot signals (`oscillating` driving a warp's ArgSpec slot), and
 * `paintThrough` (a paint masked by the child's coverage — the pure-Expr pointwise path).
 * These pin emission until real shaders ride them.
 */

// A generator whose paint is pure rhythm: a pulse-train blink, an oscillating brightness,
// and a per-cycle random seed — everything the "Blink"/"Meteor" class of shader needs.
const Rhythm: GpuShaderDefinition = defineShader({
    name: 'RhythmProbe',
    role: 'generator',
    category: 'Textures',
    description: 'motion words probe',
    acceptsUVContext: true,
    props: {},
    paint: (params: GpuFragmentParams): Expr => {
        const t = params.ctx.time
        const beat = pulseTrain(t, {period: 2, duty: 0.25, soften: 0.05})
        const glowing = oscillate(t, {rate: 0.5, min: 0.2, max: 1})
        const pass = cycle(t, 3)
        const rand = cycleSeed(pass.index, 7)
        return vec4(splat3(mul(beat, glowing)), add(mul(pass.progress, 0.001), mul(rand, 0.001)))
    },
})

const SolidChild: GpuShaderDefinition = defineShader({
    name: 'SolidProbe',
    role: 'generator',
    category: 'Textures',
    description: 'child probe',
    acceptsUVContext: true,
    props: {color: {default: '#ff0000', transform: transformColor, description: 'c'}},
    paint: solidColor(p('color')),
})

// A UV-dependent child — a warp over a UV-independent paint legitimately no-ops.
const RampChild: GpuShaderDefinition = {
    name: 'RampProbe',
    acceptsUVContext: true,
    props: {},
    fragment: (params: GpuFragmentParams): Expr => {
        const uv = params.uvContext ?? params.ctx.uv
        return vec4(splat3(uv.member('x')), 1)
    },
}

// paintThrough: the RhythmProbe paint shown through the child's alpha.
const Stencilled: GpuShaderDefinition = defineShader({
    name: 'StencilProbe',
    role: 'filter',
    species: 'pointwise',
    category: 'Stylize',
    description: 'paintThrough probe',
    props: {},
    effect: paintThrough((params) => vec4(splat3(oscillate(params.ctx.time, {rate: 1})), 1)),
})

// A warp slot driven by a time signal — no prop, no hand-written map.
const Wobbling: GpuShaderDefinition = defineShader({
    name: 'WobbleProbe',
    role: 'warp',
    category: 'Distortions',
    description: 'signal-slot probe',
    animatedTime: {speed: 'speed'},
    props: {
        center: centerPropConfig('probe center'),
        edges: edgesPropConfig('stretch', 'probe edges'),
        speed: {default: 1, description: 's', ui: {type: 'range', min: 0, max: 4, step: 0.01, label: 'Speed', group: 'Effect'}},
    },
    map: twirl({center: p('center'), intensity: oscillating({rate: 0.5, min: 0.4, max: 1.6})}),
})

describe('std motion + paintThrough + slot signals', () => {
    it('motion words emit rhythm over the global clock', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'r', def: Rhythm, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        const wgsl = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        expect(wgsl).toMatch(/fract/)
        expect(wgsl).toMatch(/sin\(/)
        expect(wgsl).toMatch(/floor/)
        expect(wgsl).toMatch(/43758.5453/)
        expect(wgsl).toMatchSnapshot('rhythm')
    })

    it('paintThrough masks a paint by the child alpha (pure-Expr pointwise path)', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'f', def: Stencilled, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'c', def: SolidChild, parentId: 'f', metadata: {renderOrder: 0}},
        ])
        const wgsl = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        expect(wgsl).toMatch(/let painted_/)
        expect(wgsl).toMatch(/painted_\d+\.a \* /)
        expect(wgsl).toMatchSnapshot('stencil')
    })

    it('oscillating() drives a warp ArgSpec slot from time', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'w', def: Wobbling, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'c', def: RampChild, parentId: 'w', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        const wgsl = tgpu.resolve([ir.finalPass.entry, ...ir.rttPasses.map((pass) => pass.entry)], {names: 'strict'})
        // The oscillating signal lands inside the warp map: sine over the node clock.
        expect(wgsl).toMatch(/_animTime/)
        expect(wgsl).toMatch(/sin\(/)
        expect(wgsl).toMatch(/1.2000000000000002|0.4/)
        expect(wgsl).toMatchSnapshot('wobble')
    })
})
