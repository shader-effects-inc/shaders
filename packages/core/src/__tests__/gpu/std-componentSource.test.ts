import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest'
import {mkdirSync, rmSync, writeFileSync} from 'node:fs'
import {resolve} from 'node:path'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, call, type ComposeOptions} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {defineWgslEnvelope, wgslComponentSource, type WgslEnvelope} from '@coreroot/std'
import {d} from '@coreroot/gpu/kit'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Code export writes a custom component (a `wgsl` envelope) as a `defineShader` source file.
 * The gate: importing that file gives the same shader the editor builds from the envelope:
 * the same WGSL, the same props, the same clock and the same kind of layer.
 */

const STD_ENTRY = resolve(__dirname, '../../std/index.ts')
const TMP = resolve(__dirname, '.component-source-tmp')

beforeAll(() => vi.useFakeTimers())
afterAll(() => {
    vi.useRealTimers()
    rmSync(TMP, {recursive: true, force: true})
})

const composeOptions: ComposeOptions = {
    createMediaTexture: (opts) => ({texture: {}, width: opts.width, height: opts.height, write: vi.fn(), unwrap: vi.fn(() => ({})), destroy: vi.fn()}) as never,
    gpu: {device: {queue: {}} as never, root: {} as never},
}

/** Write the emitted source with `shaders/std` pointed at this package's std, and import it. */
async function importSource(source: string, file: string): Promise<GpuShaderDefinition> {
    mkdirSync(TMP, {recursive: true})
    const path = resolve(TMP, `${file}.ts`)
    writeFileSync(path, source.replace(`from 'shaders/std'`, `from '${STD_ENTRY}'`))
    const module = await import(/* @vite-ignore */ path)
    return Object.values(module)[0] as GpuShaderDefinition
}

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Child: GpuShaderDefinition = {
    name: 'Child',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

function resolveWgsl(def: GpuShaderDefinition): string {
    const filter = def.requiresChild || def.requiresRTT || !def.boundingBoxDeclaration
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'x', def, parentId: 'root', metadata: {renderOrder: 0}},
        ...(filter ? [{id: 'c', def: Child, parentId: 'x', metadata: {renderOrder: 0}}] : []),
    ])
    const ir = composeNodeTree(registry, composeOptions)
    const entries = [...ir.rttPasses.map((pass) => pass.fragment.entry), ir.finalPass.entry]
    // Whitespace aside: the file re-indents the body
    return entries.map((entry) => tgpu.resolve([entry as never], {names: 'strict'})).join('\n').replace(/\s+/g, ' ')
}

/** A transform compared by name and by what it makes of the default (selectIndex is a fresh closure each time). */
function transformOf(transform: unknown, value: unknown) {
    if (typeof transform !== 'function') return undefined
    let output: unknown
    try {
        output = JSON.stringify(transform(value))
    } catch {
        output = 'throws'
    }
    return {name: transform.name, output}
}

/** What the renderer and editor read from a definition. */
function shapeOf(def: GpuShaderDefinition) {
    const props = Object.fromEntries(Object.entries(def.props as Record<string, {default: unknown; transform?: unknown; ui?: unknown; compileTime?: boolean}>)
        .map(([name, config]) => [name, {default: config.default, transform: transformOf(config.transform, config.default), ui: config.ui, compileTime: config.compileTime}]))
    return {
        name: def.name,
        props,
        animatedTime: def.animatedTime,
        acceptsUVContext: def.acceptsUVContext,
        boundingBoxDeclaration: def.boundingBoxDeclaration?.propBindings ?? def.boundingBoxDeclaration,
        requiresChild: def.requiresChild,
        requiresRTT: def.requiresRTT,
    }
}

const ENVELOPES: Record<string, WgslEnvelope> = {
    paint: {
        kind: 'wgsl',
        name: 'Halo_1a2b3c4d',
        description: 'Rings that breathe outward',
        props: {
            inner: {kind: 'color', default: '#ffd166', label: 'Inner'},
            outer: {kind: 'color', default: '#073b4c'},
            center: {kind: 'position', default: {x: 0.5, y: 0.5}},
            radius: {kind: 'range', default: 0.6, min: 0.05, max: 2, description: 'How far the rings reach'},
            spin: {kind: 'range', default: 30, min: 0, max: 360},
            invert: {kind: 'checkbox', default: false},
        },
        body: `
            let q = (uv - center) * vec2f(aspect, 1.0);
            let a = radians(spin);
            let dd = length(q) / radius + sin(time + a) * 0.05;
            let t = smoothstep(0.0, 1.0, dd);
            let c = mix(inner.rgb, outer.rgb, select(t, 1.0 - t, invert > 0.5));
            return vec4f(c, 1.0);`,
    },
    pointwise: {
        kind: 'wgsl',
        name: 'Tint_00ff00aa',
        props: {color: {kind: 'color', default: '#ff0066'}, amount: {kind: 'range', default: 0.5, min: 0, max: 1}},
        body: 'return vec4f(mix(child.rgb, color.rgb * child.a, amount), child.a);',
    },
    gather: {
        kind: 'wgsl',
        name: 'Smear_12345678',
        props: {reach: {kind: 'range', default: 0.01, min: 0, max: 0.1}},
        body: `
\tvar sum = vec4f(0.0);
\tfor (var i = 0; i < 4; i++) {
\t\tsum += textureSample(childTexture, childSampler, uv + vec2f(f32(i) * reach, 0.0));
\t}
\treturn sum / 4.0;`,
    },
    controls: {
        kind: 'wgsl',
        name: 'Controls_0000aaaa',
        props: {
            mode: {kind: 'select', default: 'rings', options: ['waves', {value: 'rings', label: 'Concentric rings'}]},
            count: {kind: 'number', default: 6},
            ramp: {kind: 'gradient-stops', default: [{color: '#0b132b', position: 0}, {color: '#ffd166', position: 0.6}, {color: '#ffffff', position: 1}]},
            dots: {kind: 'list', default: [{at: {x: 0.3, y: 0.4}, big: true}], item: {at: {kind: 'position', default: {x: 0.5, y: 0.5}}, big: {kind: 'boolean', default: false}}, maxItems: 4},
        },
        body: `
            var t = select(uv.x, fract(length(uv - 0.5) * count), mode > 0.5);
            for (var i = 0; i < i32(dotsCount); i++) {
                t += 0.1 / max(distance(uv, vec2f(dots_at[i].x, 1.0 - dots_at[i].y)), 0.01) * select(0.5, 1.0, dots_big[i].x > 0.0);
            }
            return ramp(t);`,
    },
    media: {
        kind: 'wgsl',
        name: 'Media_bbbb0000',
        props: {
            photo: {kind: 'image-upload', default: 'https://example.com/photo.jpg', fit: 'contain'},
            clip: {kind: 'video-upload', default: ''},
            source: {kind: 'layer', default: ''},
        },
        body: 'return mix(photo, clip, 0.5) + source(uv.yx) * 0.25 + textureSampleLevel(photoTexture, photoSampler, uv, 0.0) * 0.0;',
    },
    surface: {
        kind: 'wgsl',
        name: 'Stamp_cccc1111',
        props: {
            shape: {kind: 'shape', default: {type: 'starSDF', sides: 5, radius: 0.35, innerRatio: 0.5}},
            ink: {kind: 'color', default: '#ff0066'},
        },
        body: `
            let aa = fwidth(shape);
            let inside = 1.0 - smoothstep(-aa, aa, shape);
            return vec4f(mix(child.rgb, ink.rgb * (0.8 + 0.2 * shapeUV.y), inside), max(child.a, inside));`,
    },
}

describe('wgslComponentSource', () => {
    for (const [kind, envelope] of Object.entries(ENVELOPES)) {
        it(`a ${kind} component's file is the shader the editor builds`, async () => {
            const source = wgslComponentSource(envelope, {identifier: 'Exported', header: ['Generated by Shaders']})
            const exported = await importSource(source, kind)
            const editor = defineWgslEnvelope(envelope)
            expect(shapeOf(exported)).toEqual(shapeOf(editor))
            expect(resolveWgsl(exported)).toBe(resolveWgsl(editor))
        })
    }

    it('reads like hand-written code', () => {
        const source = wgslComponentSource(ENVELOPES.paint!, {identifier: 'Halo', header: ['Generated by Shaders']})
        expect(source).toMatch(/^\/\/ Generated by Shaders\nimport \{ defineShader, wgsl, transformBool01, transformColor, transformPosition \} from 'shaders\/std'/)
        expect(source).toContain(`export const Halo = defineShader({\n  name: 'Halo_1a2b3c4d',`)
        expect(source).toContain(`animatedTime: { speed: 'speed' },`)
        expect(source).toContain(`  paint: wgsl\`\n    let q = (uv - center) * vec2f(aspect, 1.0);`)
        // An authored description is kept; one that only repeats the label is not
        expect(source).toContain(`description: 'How far the rings reach'`)
        expect(source).not.toMatch(/inner: \{[^\n]*description/)
    })

    it('rejects an identifier that is not one', () => {
        expect(() => wgslComponentSource(ENVELOPES.paint!, {identifier: 'my halo'})).toThrow(/not a valid identifier/)
    })
})
