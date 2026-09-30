import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {defineShader, wgsl, ctx, p, scanIdentifiers, wgslTypeForProp} from '@coreroot/std'
import {transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from '@coreroot/utilities/transformations'
import {colorStopsPropConfig} from '@coreroot/utilities/colorStops'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * The `wgsl` word — a raw WGSL body lowered as a generator paint or a filter effect at runtime
 * (string-form `tgpu.fn`, no transpiler). These gates resolve real compositions to WGSL and
 * assert the binding contract: only referenced names become parameters, types follow the
 * prop configs, the child arrives as `color` (pointwise) or `childTexture` + `childSampler`
 * (gather), and the species is inferred from the body when not declared.
 */

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

function resolveTree(specs: Parameters<typeof buildRegistry>[0]) {
    const {registry} = buildRegistry(specs)
    const ir = composeNodeTree(registry)
    const wgslText = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
    return {ir, wgsl: wgslText}
}

// ── Generator ───────────────────────────────────────────────────────────────────────────

const Halo = defineShader({
    name: 'Halo',
    animatedTime: {speed: 'speed'},
    props: {
        inner: {default: '#ffd166', transform: transformColor},
        outer: {default: '#073b4c', transform: transformColor},
        center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
        radius: {default: 0.6},
        bands: {default: 3},
        speed: {default: 1},
        label: {default: 'unused-cpu-only-string'},
        colorSpace: {default: 'linear', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
    },
    paint: wgsl`
        // bands is referenced in a comment only: not a binding
        let q = (uv - center) * vec2f(aspect, 1.0);
        let dd = length(q) / radius + sin(time) * 0.05;
        let t = smoothstep(0.0, 1.0, dd);
        return vec4f(mix(inner.rgb, outer.rgb, t), 1.0);
    `,
})

describe('wgsl generator paint', () => {
    it('binds only the referenced props + context, typed from the prop configs', () => {
        const {ir, wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'h', def: Halo, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        expect(ir.rttPasses.length).toBe(0)
        // The generated function signature: parameters in binding order, WGSL types.
        expect(out).toMatch(/fn haloPaint\(uv: vec2f, time: f32, aspect: f32, inner: vec4f, outer: vec4f, center: vec2f, radius: f32\) -> vec4f/)
        // Unreferenced props (bands, speed) and CPU-only props (label) are not parameters.
        expect(out).not.toMatch(/fn haloPaint\([^)]*bands/)
        expect(out).not.toMatch(/fn haloPaint\([^)]*label/)
        // animatedTime declared → `time` is the node clock, not the global one.
        expect(out).toMatch(/haloPaint\(uv, [^,]*_animTime/)
        // A position prop is handed over in uv space (the stored `1 - y` undone).
        expect(out).toMatch(/vec2f\(uniforms\.n_h\.center\.x, 1\.0 - uniforms\.n_h\.center\.y\)/)
        expect(out).toMatchSnapshot('halo-final-pass')
    })

    it('reads the global clock when animatedTime is not declared', () => {
        const Pulse = defineShader({
            name: 'Pulse',
            role: 'generator',
            props: {},
            paint: wgsl`return vec4f(vec3f(0.5 + 0.5 * sin(time)), 1.0);`,
        })
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'p', def: Pulse, parentId: 'root'},
        ])
        expect(out).toMatch(/fn pulsePaint\(time: f32\) -> vec4f/)
        expect(out).toMatch(/pulsePaint\([^)]*_sys\.time\)/)
    })

    it('explicit inputs rename context values and splice literals', () => {
        const Named = defineShader({
            name: 'Named',
            role: 'generator',
            props: {scale: {default: 2}},
            paint: wgsl({
                inputs: {t: ctx.time, k: 4, s: p('scale')},
                body: 'return vec4f(fract(uv * s * k + t), 0.0, 1.0);',
            }),
        })
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'n', def: Named, parentId: 'root'},
        ])
        expect(out).toMatch(/fn namedPaint\(uv: vec2f, t: f32, k: f32, s: f32\) -> vec4f/)
        expect(out).toMatch(/namedPaint\(uv, [^,]*_sys\.time, 4\.0, /)
    })

    it('tagged-template values are inlined as WGSL text (integers stay abstract-int literals)', () => {
        const octaves = 3
        const body = wgsl`var v = 0.0; for (var i = 0; i < ${octaves}; i++) { v += ${0.25}; } return vec4f(v);`
        expect(body.spec.body).toContain('i < 3;')
        expect(body.spec.body).toContain('v += 0.25;')
        expect(body.identifiers.has('return')).toBe(true)
        expect(body.samplesChild).toBe(false)
    })
})

// ── Filters ─────────────────────────────────────────────────────────────────────────────

// No role anywhere below: `effect:` says filter, the body says which species.
const Warm = defineShader({
    name: 'Warm',
    props: {
        tint: {default: '#ff9900', transform: transformColor},
        amount: {default: 0.5},
    },
    effect: wgsl`return vec4f(mix(child.rgb, tint.rgb, amount), child.a);`,
})

const Mosaic = defineShader({
    name: 'Mosaic',
    props: {cells: {default: 24}},
    effect: wgsl`
        let cell = floor(uv * cells) / cells + 0.5 / cells;
        return textureSample(childTexture, childSampler, cell);
    `,
})

describe('wgsl filter effects', () => {
    it('pointwise: role + species inferred, child color arrives as `child`, no RTT pass', () => {
        const {ir, wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'w', def: Warm, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'w', metadata: {renderOrder: 0}},
        ])
        expect(Warm.requiresChild).toBe(true)
        expect(Warm.requiresRTT).toBeFalsy()
        expect(ir.rttPasses.length).toBe(0)
        expect(out).toMatch(/fn warmFilter\(child: vec4f, tint: vec4f, amount: f32\) -> vec4f/)
        expect(out).toMatch(/warmFilter\(genBody\(uv\)/)
        expect(out).not.toMatch(/unpremultiplyAlpha/)
        expect(out).toMatchSnapshot('warm-final-pass')
    })

    it('gather: inferred from `childTexture`, texture + sampler parameters, unpremultiplied tail', () => {
        const {ir, wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'm', def: Mosaic, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'm', metadata: {renderOrder: 0}},
        ])
        expect(Mosaic.requiresRTT).toBe(true)
        expect(ir.rttPasses.length).toBe(1)
        expect(out).toMatch(/fn mosaicFilter\(uv: vec2f, childTexture: texture_2d<f32>, childSampler: sampler, cells: f32\) -> vec4f/)
        // The call site passes the raw RTT accessor + shared sampler, which the entry binds.
        expect(out).toMatch(/mosaicFilter\(uv, rtt_0, linearClamp, /)
        expect(out).toMatch(/unpremultiplyAlpha\(mosaicFilter/)
        expect(out).toMatchSnapshot('mosaic-final-pass')
    })

    it("gather with alpha: 'straight' skips the unpremultiply tail", () => {
        const Straight = defineShader({
            name: 'Straight',
            role: 'filter',
            species: 'gather',
            props: {},
            effect: wgsl({
                alpha: 'straight',
                body: 'let s = textureSample(childTexture, childSampler, uv); return vec4f(s.rgb / max(s.a, 1e-4), s.a);',
            }),
        })
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 's', def: Straight, parentId: 'root'},
            {id: 'gen', def: Generator, parentId: 's'},
        ])
        expect(out).toMatch(/straightFilter\(uv, rtt_0, linearClamp\)/)
        expect(out).not.toMatch(/unpremultiplyAlpha\(straightFilter/)
    })

    it('a declared species wins over inference', () => {
        const Declared = defineShader({
            name: 'Declared',
            role: 'filter',
            species: 'pointwise',
            props: {},
            effect: wgsl`return vec4f(1.0 - child.rgb, child.a);`,
        })
        expect(Declared.requiresRTT).toBeFalsy()
        expect(Declared.requiresChild).toBe(true)
    })

    it('a missing child returns transparent (the scaffold guard)', () => {
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'w', def: Warm, parentId: 'root'},
        ])
        expect(out).not.toMatch(/warmFilter/)
    })
})

// ── Contract details ────────────────────────────────────────────────────────────────────

describe('wgsl word contract', () => {
    it('rejects an empty body and a body without a return', () => {
        expect(() => wgsl('')).toThrow(/non-empty/)
        expect(() => wgsl('let x = 1.0;')).toThrow(/return/)
        // A `return` inside a comment does not count.
        expect(() => wgsl('// return nothing\nlet x = 1.0;')).toThrow(/return/)
    })

    it('identifier scan skips comments and member accesses', () => {
        const ids = scanIdentifiers('/* uv */ let a = color.rgb; // time\nreturn vec4f(a, pointer.x);')
        expect(ids.has('uv')).toBe(false)
        expect(ids.has('time')).toBe(false)
        expect(ids.has('rgb')).toBe(false)
        expect(ids.has('color')).toBe(true)
        expect(ids.has('pointer')).toBe(true)
        expect(ids.has('x')).toBe(false)
    })

    it('prop types follow the bridge packing rules', () => {
        expect(wgslTypeForProp({default: '#fff', transform: transformColor})).toBe('vec4f')
        expect(wgslTypeForProp({default: {x: 0, y: 0}, transform: transformPosition})).toBe('vec2f')
        expect(wgslTypeForProp({default: 0.5})).toBe('f32')
        expect(wgslTypeForProp({default: true})).toBe('f32')
        expect(wgslTypeForProp({default: {value: 12, unit: 'px'}})).toBe('f32')
        expect(wgslTypeForProp({default: 'oklch', transform: transformColorSpace})).toBe('f32')
        expect(wgslTypeForProp({default: 'https://example.com/a.png'})).toBeNull()
        expect(wgslTypeForProp({default: {type: 'sphere'}})).toBeNull()
        expect(wgslTypeForProp(colorStopsPropConfig() as never)).toBeNull()
    })

    it('an explicit input naming a CPU-only prop is rejected at definition time', () => {
        expect(() =>
            defineShader({
                name: 'Bad',
                role: 'generator',
                props: {url: {default: 'x.png'}},
                paint: wgsl({inputs: {u: p('url')}, body: 'return vec4f(1.0);'}),
            }),
        ).toThrow(/CPU-only/)
    })

    it('the lowered fn is created once per signature and shared across compositions', () => {
        const Once = defineShader({
            name: 'Once',
            role: 'generator',
            props: {},
            paint: wgsl`return vec4f(uv, 0.0, 1.0);`,
        })
        const a = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'o', def: Once, parentId: 'root'},
        ]).wgsl
        const b = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'o', def: Once, parentId: 'root'},
        ]).wgsl
        expect(a).toBe(b)
    })
})

describe('defineShader infers role and species from the definition shape', () => {
    it('paint: → generator (accepts UV context by default), effect: → filter', () => {
        expect(Halo.requiresChild).toBeFalsy()
        expect(Halo.acceptsUVContext).toBe(true)
        expect(Warm.requiresChild).toBe(true)
        expect(Warm.requiresRTT).toBeFalsy()
        expect(Mosaic.requiresRTT).toBe(true)
    })

    it('a prop named `color` binds in a generator (only child*, viewport and ctx names are reserved)', () => {
        const Fill = defineShader({
            name: 'Fill',
            props: {color: {default: '#ff0000', transform: transformColor}},
            paint: wgsl`return color;`,
        })
        const {wgsl: out} = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'f', def: Fill, parentId: 'root'},
        ])
        expect(out).toMatch(/fn fillPaint\(color: vec4f\) -> vec4f/)
    })

    it('a declared role or species that contradicts the shape is rejected', () => {
        expect(() =>
            defineShader({name: 'Wrong', role: 'filter', props: {}, paint: wgsl`return vec4f(1.0);`} as never),
        ).toThrow(/role 'filter' contradicts its paint: field/)
        expect(() =>
            defineShader({
                name: 'Wrong2',
                species: 'pointwise',
                props: {},
                effect: {kind: 'gather', build: () => undefined as never},
            } as never),
        ).toThrow(/species 'pointwise' contradicts/)
        expect(() => defineShader({name: 'Empty', props: {}} as never)).toThrow(/give it a GPU half/)
    })

    it('a declared species still wins for a wgsl body', () => {
        const Forced = defineShader({
            name: 'Forced',
            species: 'gather',
            props: {},
            effect: wgsl`let s = textureSample(childTexture, childSampler, uv); return s;`,
        })
        expect(Forced.requiresRTT).toBe(true)
    })
})

describe('defineShader rejects prop names the component or renderer already owns', () => {
    it('layer props, renderer synthetics and invalid identifiers throw', () => {
        const make = (props: Record<string, unknown>) =>
            defineShader({name: 'Bad', props: props as never, paint: wgsl`return vec4f(1.0);`})
        expect(() => make({flow: {default: 1}})).toThrow(/'flow' is a layer prop/)
        expect(() => make({opacity: {default: 1}})).toThrow(/layer prop/)
        expect(() => make({children: {default: 1}})).toThrow(/layer prop/)
        expect(() => make({_animTime: {default: 0}})).toThrow(/renderer manages/)
        expect(() => make({_bbox_centerX: {default: 0}})).toThrow(/renderer manages/)
        expect(() => make({'my-prop': {default: 0}})).toThrow(/not a valid identifier/)
        expect(() => make({'2x': {default: 0}})).toThrow(/not a valid identifier/)
    })

    it("'src' is reserved too (it is how <CustomShader> receives the definition)", () => {
        expect(() => defineShader({name: 'Srcy', props: {src: {default: 'x'}} as never, paint: wgsl`return vec4f(1.0);`})).toThrow(/'src' is a layer prop/)
    })
})

describe('wgsl revision and declared input types', () => {
    const make = (inputs: Record<string, unknown>, body = 'return vec4f(k);') =>
        defineShader({name: 'Rev', props: {scale: {default: 2}}, paint: wgsl({inputs: inputs as never, body})})

    it('the revision changes when an input value or declared type changes, not only its name', () => {
        const base = make({k: 4}).revision
        expect(base).toBeDefined()
        expect(make({k: 5}).revision).not.toBe(base)
        expect(make({k: p('scale')}).revision).not.toBe(base)
        expect(make({k: ctx.time}).revision).not.toBe(make({k: ctx.aspect}).revision)
        expect(make({k: {value: 4, type: 'f32'}}).revision).not.toBe(base)
        // Same inputs in another order → same revision.
        expect(make({a: 1, b: 2}, 'return vec4f(a + b);').revision).toBe(make({b: 2, a: 1}, 'return vec4f(a + b);').revision)
    })

    it('a declared type that contradicts the source is rejected', () => {
        expect(() => make({k: {value: p('scale'), type: 'vec2f'}})).toThrow(/declared vec2f but its source is f32/)
        expect(() => make({k: {value: ctx.uv, type: 'f32'}})).toThrow(/declared f32 but its source is vec2f/)
        expect(() => make({k: {value: 4, type: 'vec4f'}})).toThrow(/declared vec4f but its source is f32/)
        expect(() => make({k: {value: ctx.uv, type: 'vec2f'}}, 'return vec4f(k, 0.0, 1.0);')).not.toThrow()
    })
})

describe('a composition records whether it carries user-authored WGSL', () => {
    it('true with a wgsl body in the tree, false for library-only trees', () => {
        const withCustom = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'h', def: Halo, parentId: 'root'},
        ]).ir
        expect(withCustom.usesCustomWgsl).toBe(true)
        const libraryOnly = resolveTree([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'gen', def: Generator, parentId: 'root'},
        ]).ir
        expect(libraryOnly.usesCustomWgsl).toBe(false)
    })
})
