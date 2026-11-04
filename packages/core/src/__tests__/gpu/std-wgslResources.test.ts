import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest'
import {tgpu, d, sdf3d} from '@coreroot/gpu/kit'
import {composeNodeTree, call, expr, type ComposeOptions} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {defineShader, wgsl, selectIndex, colorStopsTransform, listPropTransform, transformColor} from '@coreroot/std'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * The props a `wgsl` body reads beyond numbers, colours and positions: images, videos, other
 * layers, gradients, lists and selects, plus the `surface:` material on a picked shape. Each
 * gate composes a definition and resolves it to WGSL, checking what the body is handed and that
 * a called prop (`logo(p)`, `ramp(t)`) became a call of the library helper.
 */

beforeAll(() => vi.useFakeTimers())
afterAll(() => vi.useRealTimers())

const composeOptions: ComposeOptions = {
    createMediaTexture: (opts) => ({texture: {__mediaTex: Symbol('media')}, width: opts.width, height: opts.height, write: vi.fn(), unwrap: vi.fn(() => ({})), destroy: vi.fn()}) as never,
    // The SVG shape field (shapeAt over an uploaded SVG) is a data texture.
    createDataTexture: (opts: {width: number; height: number}) => ({texture: {__dataTex: Symbol('data')}, width: opts.width, height: opts.height, write: vi.fn(), unwrap: vi.fn(() => ({})), destroy: vi.fn()}) as never,
    gpu: {device: {queue: {}} as never, root: {} as never},
}

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

function resolve(def: GpuShaderDefinition, props: Record<string, unknown> = {}, extra: Parameters<typeof buildRegistry>[0] = []) {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        ...extra,
        {id: 'x', def, parentId: 'root', props, metadata: {renderOrder: 9}},
    ])
    ;(registry as {resolveCustomId: (id: string) => string | null}).resolveCustomId = (id) => (id === 'src' ? 'src' : null)
    const ir = composeNodeTree(registry, composeOptions)
    const entries = [...ir.rttPasses.map((pass) => pass.fragment.entry), ir.finalPass.entry]
    return {ir, wgsl: entries.map((entry) => tgpu.resolve([entry as never], {names: 'strict'})).join('\n')}
}

describe('image props', () => {
    const Poster = defineShader({
        name: 'Poster',
        props: {logo: {default: 'https://example.com/a.png', ui: {type: 'image-upload', fit: 'contain'}}},
        paint: wgsl`
            let wobble = logo(uv + vec2f(0.01, 0.0));
            return mix(logo, wobble, 0.5);`,
    })

    it('loads the image as a media texture and hands the body the fitted sample, a callable, and nothing else', () => {
        const {ir, wgsl: out} = resolve(Poster)
        expect(ir.textures.filter((t) => t.kind === 'media').length).toBe(1)
        expect(out).toMatch(/fn posterPaint\(uv: vec2f, logo: vec4f, logoTexture: texture_2d<f32>, logoSampler: sampler, logo_viewport: vec2f\) -> vec4f/)
        // The call became the media helper (contain fit, explicit mip level)
        expect(out).toMatch(/shaders_logo\(logoTexture, logoSampler, logo_viewport, uv \+ vec2f\(0\.01, 0\.0\)\)/)
        expect(out).toMatch(/textureSampleLevel\(tex, samp, q, log2/)
        // contain: the fit scale takes the smaller ratio
        expect(out).toMatch(/fn fitScale[^}]*min\(/)
    })

    it('binds only the callable when the body never reads the bare value', () => {
        const Only = defineShader({
            name: 'Only',
            props: {logo: {default: '', ui: {type: 'image-upload'}}},
            paint: wgsl`return logo(uv * 2.0);`,
        })
        const {wgsl: out} = resolve(Only)
        expect(out).toMatch(/fn onlyPaint\(uv: vec2f, logoTexture: texture_2d<f32>, logoSampler: sampler, logo_viewport: vec2f\) -> vec4f/)
        expect(out).toMatch(/fn fitScale[^}]*max\(/)
    })
})

describe('video props', () => {
    it('samples an external texture and gets a Loop toggle', () => {
        const Clip = defineShader({
            name: 'Clip',
            props: {clip: {default: 'https://example.com/a.mp4', ui: {type: 'video-upload'}}},
            paint: wgsl`return clip;`,
        })
        expect(Object.keys(Clip.props)).toEqual(['clip', 'clipLoop'])
        const {ir, wgsl: out} = resolve(Clip)
        expect(ir.externalTextures.length).toBe(1)
        expect(out).toMatch(/textureSampleBaseClampToEdge\(tex, samp, q\)/)
    })
})

describe('layer props', () => {
    const Lens = defineShader({
        name: 'Lens',
        props: {source: {default: '', compileTime: true, ui: {type: 'layer'}}},
        paint: wgsl`return source(uv.yx);`,
    })

    it('samples the chosen layer through its shared RTT', () => {
        const {ir, wgsl: out} = resolve(Lens, {source: 'src'}, [{id: 'src', def: Generator, parentId: 'root', metadata: {renderOrder: 0}}])
        expect(ir.rttPasses.length).toBe(1)
        expect(out).toMatch(/fn lensPaint\(uv: vec2f, sourceTexture: texture_2d<f32>, sourceSampler: sampler\) -> vec4f/)
        expect(out).toMatch(/c\.rgb \/ c\.a/)
    })

    it('reads transparent while no layer is chosen', () => {
        const {ir} = resolve(Lens)
        expect(ir.rttPasses.length).toBe(0)
        expect(ir.textures.filter((t) => t.kind === 'media').length).toBe(1)
    })
})

describe('gradient props', () => {
    const Ramp = defineShader({
        name: 'Ramp',
        props: {ramp: {default: [{color: '#ff0000', position: 0}, {color: '#0000ff', position: 1}], transform: colorStopsTransform, ui: {type: 'gradient-stops'}}},
        paint: wgsl`return ramp(uv.x);`,
    })

    it('adds Color Space and mixes the stops in it', () => {
        expect(Object.keys(Ramp.props)).toEqual(['ramp', 'colorSpace'])
        const linear = resolve(Ramp).wgsl
        expect(linear).toMatch(/fn rampPaint\(uv: vec2f, ramp_stopColors: array<vec4f,8>, ramp_stopPositions: array<vec4f,2>, ramp_stopConverted: array<vec3f,8>, ramp_stopCount: f32\) -> vec4f/)
        expect(linear).not.toMatch(/oklabToRgb/)
        const oklab = resolve(Ramp, {colorSpace: 'oklab'}).wgsl
        expect(oklab).toMatch(/oklabToRgb/)
    })
})

describe('list and select props', () => {
    it('hands a list over as lane arrays and a count, and a select as its option index', () => {
        const Dots = defineShader({
            name: 'Dots',
            props: {
                dots: {
                    default: [{at: {x: 0.3, y: 0.5}, tint: '#ffffff'}],
                    transform: listPropTransform,
                    ui: {type: 'list', item: {at: {kind: 'position', default: {x: 0.5, y: 0.5}}, tint: {kind: 'color', default: '#ffffff'}}, maxItems: 6},
                },
                mode: {default: 'soft', transform: selectIndex(['soft', 'hard']), ui: {type: 'select', options: [{label: 'Soft', value: 'soft'}, {label: 'Hard', value: 'hard'}]}},
                ink: {default: '#000000', transform: transformColor},
            },
            paint: wgsl`
                var c = ink;
                for (var i = 0; i < i32(dotsCount); i++) {
                    let p = vec2f(dots_at[i].x, 1.0 - dots_at[i].y);
                    let k = select(smoothstep(0.1, 0.0, distance(uv, p)), step(distance(uv, p), 0.05), mode > 0.5);
                    c = mix(c, dots_tint[i], k);
                }
                return c;`,
        })
        const {wgsl: out} = resolve(Dots)
        expect(out).toMatch(/fn dotsPaint\(uv: vec2f, dots_at: array<vec4f,6>, dots_tint: array<vec4f,6>, dotsCount: f32, mode: f32, ink: vec4f\) -> vec4f/)
    })
})

describe('surface: a material on a picked shape', () => {
    const Stamp = defineShader({
        name: 'Stamp',
        props: {ink: {default: '#ff0066', transform: transformColor}},
        surface: wgsl`
            let aa = fwidth(shape);
            let inside = 1.0 - smoothstep(-aa, aa, shape);
            return vec4f(ink.rgb * (0.8 + 0.2 * shapeUV.y), inside);`,
    })

    it('adds the shape and placement props, and reads the shape field at this pixel', () => {
        expect(Object.keys(Stamp.props)).toEqual(['ink', 'shape', 'origin', 'center', 'scale', 'rotation', 'shapeSdfUrl', 'shapeType'])
        expect(Stamp.boundingBoxDeclaration?.propBindings?.x).toEqual({prop: 'center', as: 'position-x'})
        const {wgsl: out} = resolve(Stamp)
        expect(out).toMatch(/fn stampSurface\(shape: f32, shapeUV: vec2f, ink: vec4f\) -> vec4f/)
    })

    it('shades the layer inside when the body reads child', () => {
        const Tint = defineShader({
            name: 'ShapeTint',
            props: {},
            surface: wgsl`
                let aa = fwidth(shape);
                return mix(child, vec4f(1.0 - child.rgb, child.a), 1.0 - smoothstep(-aa, aa, shape));`,
        })
        expect(Tint.requiresChild).toBe(true)
        const {wgsl: out} = resolve(Tint, {}, [{id: "g", def: Generator, parentId: "x", metadata: {renderOrder: 0}}])
        expect(out).toMatch(/fn shapeTintSurface\(shape: f32, child: vec4f\) -> vec4f/)
    })

    describe('shapeAt(p): the shape anywhere', () => {
        const Column = defineShader({
            name: 'Column',
            props: {},
            surface: wgsl`
                var fuel = 0.0;
                for (var i = 1; i <= 4; i++) {
                    fuel += step(shapeAt(shapeUV + vec2f(0.0, 0.05 * f32(i))), 0.0);
                }
                return vec4f(vec3f(fuel * 0.25), 1.0);`,
        })

        it('links a helper that samples the analytic field, its inputs passed from the fragment', () => {
            const {wgsl: out} = resolve(Column, {shape: JSON.stringify({type: 'starSDF', sides: 5})})
            // The body takes the field's inputs (8 analytic sub-props + scale) ahead of its own.
            expect(out).toMatch(/fn columnSurface\(shapeUV: vec2f, _shapeArg0: f32,[^)]*_shapeArg7: f32, _shapeScale: f32\) -> vec4f/)
            // The helper samples the analytic SDF at the point and divides by scale like `shape`.
            expect(out).toMatch(/fn _shapeAtFn\([^)]*p: vec2f\) -> f32 \{\s*return analyticSdf\w*\(p, [^;]*\)\.x \/ _shapeScale;/)
            expect(out).toMatch(/_shapeAtFn\(_shapeArg0, [^;]*_shapeScale, \(?shapeUV/)
        })

        it('passes the SVG field texture and sampler in, so the pass still binds them', () => {
            const {wgsl: out} = resolve(Column, {shapeSdfUrl: 'https://example.com/logo.bin'})
            expect(out).toMatch(/fn _shapeAtFn\(_shapeArg0: texture_2d<f32>, _shapeArg1: sampler, _shapeScale: f32, p: vec2f\) -> f32/)
            expect(out).toMatch(/textureSampleLevel\(_shapeArg0, _shapeArg1, p, 0\.0\)\.x/)
        })

        it('samples a marched 3D field through its fast tap, texture and footprint passed in', () => {
            const fieldTex = {accessor: () => expr('tex.$.vf'), sample: () => expr('0'), sampleLevel: () => expr('0')}
            const uniforms = new Proxy({}, {get: (_t, k) => expr(`u_${String(k)}`)})
            const {portable} = sdf3d.resolveShapeFieldSampler({
                computeOutputs: {volumetricFieldTexture: fieldTex}, uniforms, propValues: {}, getCpuValue: () => undefined,
            } as never)
            expect(portable.key).toBe('volumetric')
            expect(portable.args.map((a) => a.type)).toEqual(['texture_2d<f32>', 'f32', 'f32', 'f32', 'f32', 'f32', 'f32'])
            const names = ['t', 'a', 'b', 'c', 'e', 'f', 'g']
            const shell = (tgpu.fn as unknown as (args: unknown[], ret: unknown) => (impl: string) => {$uses: (u: Record<string, unknown>) => unknown})(
                [d.texture2d(d.f32), d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.vec2f], d.vec4f,
            )
            const probe = shell(`(t: texture_2d<f32>, a: f32, b: f32, c: f32, e: f32, f: f32, g: f32, p: vec2f) -> vec4f {\n  return ${portable.sample(names, 'p')};\n}`).$uses(portable.uses)
            expect(tgpu.resolve([probe as never], {names: 'strict'})).toMatch(/fieldSampleFastArg\(p, t, a, b, c, e, f, g\)/)
        })
    })
})
